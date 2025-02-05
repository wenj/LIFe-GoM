import os
import cv2
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data
import torch.optim as optim

from configs import make_cfg

from models.model import Model

from utils.train_util import cpu_data_to_gpu
from utils.image_util import to_8b_image
from utils.tb_util import TBLogger
from utils.lpips import LPIPS
from utils.network_util import mesh_laplacian_smoothing, ssim


EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--cfg",
		default=None,
		type=str
	)
	parser.add_argument(
		"--resume",
		action="store_true",
	)

	return parser.parse_args()


def unpack(rgbs, masks, bgcolors):
	rgbs = rgbs * masks.unsqueeze(-1) + bgcolors[:, None, None, :] * (1 - masks).unsqueeze(-1)
	return rgbs


def compute_loss(rgb_pred, mask_pred, outputs, rgb_gt, mask_gt, loss_cfg, **kwargs):
	losses = {}
	if loss_cfg.rgb.type == 'l1':
		loss_rgb = torch.mean(torch.abs(rgb_pred - rgb_gt))
	elif loss_cfg.rgb.type == 'l2':
		loss_rgb = torch.mean((rgb_pred - rgb_gt) ** 2)
	else:
		raise NotImplementedError(f'rgb loss type {loss_cfg.rgb.type} not implemented')
	losses['rgb'] = {
		'unscaled': loss_rgb,
		'scaled': loss_rgb * loss_cfg.rgb.coeff
	}

	if loss_cfg.mask.type == 'l1':
		loss_mask = torch.mean(torch.abs(mask_pred - mask_gt))
	elif loss_cfg.mask.type == 'l2':
		loss_mask = torch.mean((mask_pred - mask_gt) ** 2)
	elif loss_cfg.mask.type == 'ce':
		loss_mask = torch.nn.functional.binary_cross_entropy(mask_pred, mask_gt[None].repeat(mask_pred.shape[0], 1, 1, 1, 1), reduction='mean')
	else:
		raise NotImplementedError(f'mask loss type {loss_cfg.mask.type} not implemented')
	losses['mask'] = {
		'unscaled': loss_mask,
		'scaled': loss_mask * loss_cfg.mask.coeff
	}

	if loss_cfg.lpips.coeff > 0:
		scale_for_lpips = lambda x: 2 * x - 1
		# rgb_gt = rgb_gt[None].expand_as(rgb_pred)
		s = rgb_gt.shape[-3:]
		if loss_cfg.balance_weight:
			loss_lpips = kwargs['lpips_func'](
				scale_for_lpips(torch.nn.functional.interpolate(rgb_pred.reshape(-1, *s).permute(0, 3, 1, 2),
																scale_factor=loss_cfg.lpips.scale_factor)),
				scale_for_lpips(torch.nn.functional.interpolate(rgb_gt.reshape(-1, *s).permute(0, 3, 1, 2),
																scale_factor=loss_cfg.lpips.scale_factor)),
			)
			loss_lpips = torch.mean(loss_lpips[:, :, :-1]) + torch.mean(loss_lpips[:, :, -1]) * loss_cfg.balance_weight
		else:
			loss_lpips = torch.mean(kwargs['lpips_func'](
				scale_for_lpips(torch.nn.functional.interpolate(rgb_pred.reshape(-1, *s).permute(0, 3, 1, 2), scale_factor=loss_cfg.lpips.scale_factor)),
				scale_for_lpips(torch.nn.functional.interpolate(rgb_gt.reshape(-1, *s).permute(0, 3, 1, 2), scale_factor=loss_cfg.lpips.scale_factor)),
			))
		losses["lpips"] = {
			'unscaled': loss_lpips,
			'scaled': loss_lpips * loss_cfg.lpips.coeff
		}

	if loss_cfg.ssim.coeff > 0:
		h, w, c = rgb_pred.shape[-3:]
		X = rgb_pred.reshape(-1, h, w, c).permute(0, 3, 1, 2)
		Y = rgb_gt[None].repeat(rgb_pred.shape[0], 1, 1, 1, 1, 1).reshape(-1, h, w, c).permute(0, 3, 1, 2)
		loss_ssim = 1 - ssim(X, Y, size_average=True)
		losses['ssim'] = {
			'unscaled': loss_ssim,
			'scaled': loss_ssim * loss_cfg.ssim.coeff
		}

	if loss_cfg.laplacian.coeff > 0:
		loss_laplacian_canonical = mesh_laplacian_smoothing(outputs['mesh_canonical'])
		losses['laplacian_canoincal'] = {
			'unscaled': loss_laplacian_canonical,
			'scaled': loss_laplacian_canonical * loss_cfg.laplacian.coeff
		}

	total_loss = sum([item['scaled'] for item in losses.values()])
	return total_loss, losses


def main(args):
	cfg = make_cfg(args.cfg)
	# fix the cfg for training
	cfg.model.reconstructor.n_subdivide = cfg.model.reconstructor.n_subdivide_train

	# fix seed
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)

	os.makedirs(cfg.save_dir, exist_ok=True)
	# setup logger
	logging_path = os.path.join(cfg.save_dir, 'log.txt')
	logging.basicConfig(
		handlers=[
			logging.FileHandler(logging_path),
			logging.StreamHandler()
		],
		format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
		level=logging.INFO
	)

	# save config file
	os.makedirs(os.path.join(cfg.save_dir), exist_ok=True)
	with open(os.path.join(cfg.save_dir, 'config.yaml'), 'w') as f:
		f.write(cfg.dump())
	logging.info(f'configs: \n{cfg.dump()}')

	# setup tensorboard logger
	tb_logger = TBLogger(os.path.join(cfg.save_dir, 'tb'), freq=cfg.train.tb_freq)
	# create directory for checkpoints
	os.makedirs(os.path.join(cfg.save_dir, 'checkpoints'), exist_ok=True)

	# load training data
	if cfg.dataset.train.name == "default":
		from datasets.default import BaseDataset
		train_dataset = BaseDataset(
			cfg.dataset.train.dataset_path,
			scene_list=cfg.dataset.train.scene_list,
			bgcolor=cfg.bgcolor if not cfg.random_bgcolor else None,
			target_size=cfg.img_size,
			use_smplx=cfg.dataset.use_smplx,
			n_input_frames=cfg.dataset.train.n_input_frames,
			subdivide_iter=cfg.dataset.train.subdivide_iter,
		)
	elif cfg.dataset.train.name == "thuman2.0_view3":
		from datasets.thuman_view3 import THumanview3Dataset
		train_dataset = THumanview3Dataset(
			cfg.dataset.train.dataset_path,
			scene_list=cfg.dataset.train.scene_list,
			target_size=cfg.img_size,
			use_smplx=cfg.dataset.use_smplx,
			n_input_frames=cfg.n_input_frames,
			total_input_frames=cfg.dataset.train.total_input_frames,
			total_output_frames=cfg.dataset.train.total_output_frames,
			input_frame_interval=cfg.dataset.train.input_frame_interval,
		)
	elif cfg.dataset.train.name == "thuman2.0_view5":
		from datasets.thuman_view5 import THumanview5Dataset
		train_dataset = THumanview5Dataset(
			cfg.dataset.train.dataset_path,
			scene_list=cfg.dataset.train.scene_list,
			target_size=cfg.img_size,
			use_smplx=cfg.dataset.use_smplx,
			n_input_frames=cfg.dataset.train.n_input_frames,
			subdivide_iter=cfg.dataset.train.subdivide_iter,
		)
	else:
		raise NotImplementedError(f'dataset {cfg.dataset.train.name} not implemented')
	train_dataloader = torch.utils.data.DataLoader(
		batch_size=cfg.dataset.train.batch_size,
		dataset=train_dataset,
		shuffle=True,
		drop_last=True,
		num_workers=cfg.dataset.train.num_workers,
		pin_memory=True,
	)

	# create model and optimizer
	n_iters = 1

	model = Model(cfg.model, train_dataset.get_canonical_info())
	model.cuda()

	param_groups = model.get_param_groups(cfg.train)
	if cfg.train.optimizer == 'adam':
		optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))
	elif cfg.train.optimizer == 'adamw':
		optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
	else:
		raise NotImplementedError(f'optimizer name {cfg.train.optimizer} not known')

	# prepare the loss
	loss_funcs = {}
	if hasattr(cfg.train.losses, 'lpips'):
		loss_funcs['lpips_func'] = LPIPS(net='vgg')
		for param in loss_funcs['lpips_func'].parameters():
			param.requires_grad = False
		loss_funcs['lpips_func'].cuda()

	# resume training and init
	if args.resume:
		ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
		max_iter = max([int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir)])
		ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
		ckpt = torch.load(ckpt_path)

		model.load_state_dict(ckpt['network'])
		optimizer.load_state_dict(ckpt['optimizer'])

		n_iters = ckpt['iter'] + 1
		logging.info(f'continue training from iter {n_iters}')
	else:
		ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_0.pt')
		save_dict = {
			'iter': n_iters,
			'network': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		torch.save(save_dict, ckpt_path)
		logging.info(f'saved to {ckpt_path}')

	model.train()

	# training loop
	while n_iters <= cfg.train.total_iters:
		for batch_idx, batch in enumerate(train_dataloader):
			tb_logger.set_global_step(n_iters)

			optimizer.zero_grad()
			data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

			rgb, mask, outputs = model(
				data['target_rgbs'][:, :-1], data['target_masks'][:, :-1],
				data['K'], data['E'],
				data['cnl_gtfms'],
				data['dst_Rs'], data['dst_Ts'], dst_posevec=data['dst_posevec'],
				canonical_joints=data['dst_tpose_joints'], canonical_vertices=data['canonical_vertices'],
				bgcolor=data['bgcolor'],
				tb=tb_logger,
			)

			rgb = unpack(rgb, mask, data['bgcolor'])

			if 'source_imgs' in outputs:
				rgb = torch.cat([outputs['source_imgs'], rgb[:, :, None, :, :, :]], dim=2)
				mask = torch.cat([outputs['source_masks'], mask[:, :, None, :, :]], dim=2)
				target_rgbs = data['target_rgbs']
				target_masks = data['target_masks']
			else:
				target_rgbs = data['target_rgbs'][:, -1]
				target_masks = data['target_masks'][:, -1]

			loss, loss_items = compute_loss(
				rgb, mask, outputs,
				target_rgbs, target_masks,
				cfg.train.losses,
				**loss_funcs,
			)
			loss.backward()
			optimizer.step()

			# write to tensorboard
			if n_iters % cfg.train.tb_freq == 0:
				tb_logger.summ_scalar('loss_scaled/loss', loss)
				for loss_name, loss_value in loss_items.items():
					tb_logger.summ_scalar(f'loss_scaled/loss_{loss_name}', loss_value['scaled'])
					tb_logger.summ_scalar(f'loss_unscaled/loss_{loss_name}', loss_value['unscaled'])

				if len(rgb.shape) == 5:
					tb_logger.summ_image('output/rgb_gt', data['target_rgbs'][:, -1].permute(0, 3, 1, 2)[0])
					tb_logger.summ_image('output/mask_gt', data['target_masks'][:, -1].unsqueeze(1)[0])
					tb_logger.summ_image('output/rgb', rgb[-1].permute(0, 3, 1, 2)[0])
					tb_logger.summ_image('output/mask', mask[-1].unsqueeze(1)[0])
				else:
					tb_logger.summ_images('output/rgb_gt', data['target_rgbs'][0].permute(0, 3, 1, 2))
					tb_logger.summ_images('output/mask_gt', data['target_masks'][0].unsqueeze(1))
					tb_logger.summ_images('output/rgb', rgb[-1, 0].permute(0, 3, 1, 2))
					tb_logger.summ_images('output/mask', mask[-1, 0].unsqueeze(1))

			# write to log
			if n_iters % cfg.train.log_freq == 0:
				loss_str = f"iter {n_iters} - loss: {loss.item():.4f} ("
				for loss_name, loss_value in loss_items.items():
					loss_str += f"{loss_name}: {loss_value['scaled'].item():.4f}, "
				loss_str = loss_str[:-2] + ")"
				logging.info(loss_str)

			# save
			if n_iters % cfg.train.save_freq == 0:
				ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_{n_iters}.pt')
				save_dict = {
					'iter': n_iters,
					'network': model.state_dict(),
					'optimizer': optimizer.state_dict(),
				}
				torch.save(save_dict, ckpt_path)
				logging.info(f'saved to {ckpt_path}')

			n_iters += 1
			if n_iters > cfg.train.total_iters:
				break


if __name__ == "__main__":
	args = parse_args()
	main(args)
