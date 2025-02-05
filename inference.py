import argparse
import cv2
import numpy as np
import os
import seaborn as sns
from PIL import Image
import logging

import torch
import torch.utils.data

from pytorch3d.io import save_obj

from configs import make_cfg

from models.model import Model

from utils.train_util import cpu_data_to_gpu
from utils.image_util import to_8b_image
from utils.tb_util import TBLogger
from utils.geom_util import img_T_cam, cam_T_world
from utils.body_util import get_global_RTs, SMPL_PARENT

from utils.lpips import LPIPS
from skimage.metrics import structural_similarity

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def parse_args():
    parser = argparse.ArgumentParser()

    # arguments about the checkpoint and inference type
    parser.add_argument(
         "--type",
         default='view',
         choices=['view', 'pose', 'train', 'freeview', 'view_debug', 'view_single', 'tpose'],
         type=str
    )
    parser.add_argument(
        "--cfg",
        default=None,
        type=str
    )
    parser.add_argument(
        "--iter",
        default=None,
        type=int
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="evaluate on AIST++"
    )

    # argument to override the loaded one
    parser.add_argument(
        "--bgcolor",
        default=None,
        type=float
    )

    # arguments for freeview rendering and novel pose rendering
    parser.add_argument(
        "--scene_idx",
        default=0,
        type=int,
        help="for freeview and novel poses"
    )
    parser.add_argument(
        "--total_frames",
        default=100,
        type=int,
        help="for freeview and novel poses"
    )
    parser.add_argument(
        "--pose_path",
        default=None,
        type=str
    )
    parser.add_argument(
         "--pose_name",
         default='default',
         type=str
    )

    return parser.parse_args()


def unpack(rgbs, masks, bgcolors):
    rgbs = rgbs * masks.unsqueeze(-1) + bgcolors[:, None, None, :] * (1 - masks).unsqueeze(-1)
    return rgbs


def main(args):
    cfg = make_cfg(args.cfg)

    # fix seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    save_dir = os.path.join(cfg.save_dir, 'eval', args.type)
    os.makedirs(save_dir, exist_ok=True)

    # setup logger
    logging_path = os.path.join(cfg.save_dir, 'eval', f'log_{args.type}.txt')
    logging.basicConfig(
        handlers=[
           logging.FileHandler(logging_path),
           logging.StreamHandler()
        ],
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    if args.bgcolor is not None:
        cfg.bgcolor = [args.bgcolor, args.bgcolor, args.bgcolor]

    # override the settings
    if args.type == 'pose':
        cfg.dataset.test_pose.scene_idx = args.scene_idx
    if args.type == 'freeview':
        cfg.dataset.test_freeview.scene_idx = args.scene_idx
    if args.type == 'tpose':
         cfg.dataset.tpose.scene_idx = args.scene_idx
    if args.pose_path is not None:
        cfg.dataset.test_pose.pose_path = args.pose_path

    if args.type == 'view':
        if cfg.dataset.test_view.name == 'thuman2.0_view3':
            from datasets.thuman_view3 import THumanview3Dataset
            test_dataset = THumanview3Dataset(
                 cfg.dataset.test_view.dataset_path,
                 scene_list=cfg.dataset.test_view.scene_list,
                 target_size=cfg.img_size,
                 use_smplx=cfg.use_smplx,
                 n_input_frames=cfg.n_input_frames,
                 total_input_frames=cfg.dataset.test_view.total_input_frames,
                 total_output_frames=cfg.dataset.test_view.total_output_frames,
                 input_frame_interval=cfg.dataset.test_view.input_frame_interval,
                 bgcolor=cfg.bgcolor,
            )
        elif cfg.dataset.test_view.name == 'thuman2.0_view5':
            from datasets.thuman_view5 import THumanview5Dataset
            test_dataset = THumanview5Dataset(
                cfg.dataset.test_view.dataset_path,
                scene_list=cfg.dataset.test_view.scene_list,
                target_size=cfg.img_size,
                use_smplx=cfg.dataset.use_smplx,
                n_input_frames=cfg.n_input_frames,
                bgcolor=cfg.bgcolor,
            )
        elif cfg.dataset.test_view.name == 'aistpp':
            from datasets.aistpp import AISTppDataset
            test_dataset = AISTppDataset(
                 cfg.dataset.test_view.dataset_path,
                 scene_list=cfg.dataset.test_view.scene_list,
                 target_size=cfg.img_size,
                 use_smplx=cfg.dataset.use_smplx,
                 n_input_frames=cfg.n_input_frames,
                 bgcolor=cfg.bgcolor,  # must be white
            )
        else:
            raise NotImplementedError(f'dataset {cfg.dataset.train.name} not implemented')
    elif args.type == 'pose':
        from datasets.newpose import PoseDataset
        test_dataset = PoseDataset(
            cfg.dataset.test_pose.dataset_path,
            scene_list=cfg.dataset.test_pose.scene_list,
            scene_idx=cfg.dataset.test_pose.scene_idx,
            pose_path=cfg.dataset.test_pose.pose_path,
            pose_name=args.pose_name,
            bgcolor=cfg.bgcolor,
            use_smplx=cfg.dataset.use_smplx,
        )
    elif args.type == 'freeview':
        from datasets.freeview import FreeviewDataset
        test_dataset = FreeviewDataset(
            cfg.dataset.test_freeview.dataset_path,
            scene_list=cfg.dataset.test_freeview.scene_list,
            scene_idx=cfg.dataset.test_freeview.scene_idx,
            total_frames=args.total_frames,
            bgcolor=cfg.bgcolor,
            use_smplx=cfg.dataset.use_smplx,
            target_size=cfg.img_size,
        )
    test_dataloader = torch.utils.data.DataLoader(
        batch_size=cfg.dataset.test_view.batch_size,
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
    if args.iter is None:
        max_iter = max([int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir)])
        ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, f'iter_{args.iter}.pt')
    logging.info(f'loading model from {ckpt_path}')
    ckpt = torch.load(ckpt_path)

    model = Model(cfg.model, test_dataset.get_canonical_info())
    model.load_state_dict(ckpt['network'])

    model.cuda()
    model.eval()

    for batch_idx, batch in enumerate(test_dataloader):
        data = cpu_data_to_gpu(
           batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
           pred, mask, outputs = model(
                data['target_rgbs'][:, :-1], data['target_masks'][:, :-1],
                data['K'], data['E'],
                data['cnl_gtfms'],
                data['dst_Rs'], data['dst_Ts'], dst_posevec=data['dst_posevec'],
                canonical_joints=data['dst_tpose_joints'], canonical_vertices=data['canonical_vertices'],
                bgcolor=data['bgcolor'],
                tb=None,
           )

        bgcolor_tensor = torch.tensor(cfg.bgcolor).float()[None].to(pred.device) / 255.
        pred = unpack(pred, mask, bgcolor_tensor)

        pred_imgs = pred.detach().cpu().numpy()
        if args.type == 'view':
            truth_imgs = data['target_rgbs'][:, -1].detach().cpu().numpy()

        for i, (frame_name, pred_img) in enumerate(zip(batch['frame_name'], pred_imgs)):
            print(frame_name)
            pred_img = to_8b_image(pred_img)
            Image.fromarray(pred_img).save(os.path.join(save_dir, frame_name + '.png'))

            if args.type == 'view' and args.eval:
                truth_img = to_8b_image(truth_imgs[i])
                Image.fromarray(truth_img).save(os.path.join(save_dir, frame_name + '_gt.png'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
