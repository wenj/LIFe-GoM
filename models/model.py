import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.img_encoder import Resnet18Encoder
from .modules.reconstructor import Reconstructor
from .modules.renderer import Renderer

from utils.body_util import apply_lbs, get_global_RTs
from utils.geom_util import img_T_cam, cam_T_world
from utils.base_util import vector_gather, pack, unpack

from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map
from pytorch3d.structures import Meshes


def get_transformation_from_triangle_steiner(triangles, sigma=0.001):
	"""
	triangles: B x NF x 3 x 3
	"""
	centroid = triangles.mean(dim=-2)

	f1 = 0.5 * (triangles[..., 2, :] - centroid)
	f2 = 1 / (2 * np.sqrt(3)) * (triangles[..., 1, :] - triangles[..., 0, :])
	t0 = torch.atan2((2 * f1 * f2).sum(-1), ((f1 * f1).sum(-1) - (f2 * f2).sum(-1))) / 2
	t0 = t0[..., None]

	axis0 = f1 * torch.cos(t0) + f2 * torch.sin(t0)
	axis1 = f1 * torch.cos(t0 + np.pi / 2) + f2 * torch.sin(t0 + np.pi / 2)

	normal = torch.cross(axis0, axis1, dim=-1)
	normal = F.normalize(normal, dim=-1) * sigma
	transform = torch.stack([axis0 * 2, axis1 * 2, normal], dim=-1)
	return transform


class Model(nn.Module):
	def __init__(self, model_cfg, canonical_info):
		super().__init__()

		self.cfg = model_cfg
		# override attributes
		module_list = ['img_encoder', 'reconstructor', 'renderer']
		for module_name in module_list:
			module_cfg = getattr(model_cfg, module_name)
			module_cfg.img_size = model_cfg.img_size
			module_cfg.use_smplx = model_cfg.use_smplx

		self.img_encoder = Resnet18Encoder(model_cfg.img_encoder)
		self.reconstructor = Reconstructor(model_cfg.reconstructor, canonical_info)
		self.renderer = Renderer(model_cfg.renderer, canonical_info)

	def reconstruct(self,
					input_imgs, input_masks,
					K, E,
					global_Rs, global_Ts,
					canonical_vertices=None,
					bgcolor=None,
					tb=None):
		"""
		Parameters
		----------
		input_imgs : B x N x H x W x 3
		input_masks: B x N x H x W
		K: B x N x 3 x 3
		cnl_gtfms: B x J x 4 x 4

		Returns
		-------
		xyz: B x NP x 3
		so3: B x NF x 3
		scale: B x NF x 3
		appr: B x NF x 3
		lbs_weights: B x NP x 3
		"""
		input_feats = self.img_encoder(input_imgs, input_masks, tb=tb)  # B x N x C x H x W

		canonical_rep = self.reconstructor(
			input_imgs.permute(0, 1, 4, 2, 3), # B x N x 3 x H x W
			input_masks,
			input_feats,
			K, E,
			global_Rs, global_Ts,
			canonical_vertices,
			renderer=self.renderer, # from renderer to bgcolor are for feedback
			img_encoder=self.img_encoder,
			bgcolor=bgcolor,
			tb=tb,
		)
		return canonical_rep

	def render(self,
			   canonical_rep,
			   K, E, global_Rs, global_Ts,
			   tb=None):
		xyz, so3, scale, appearance, lbs_weights, offset = canonical_rep

		# K is B x N x 3 x 3
		B, N = global_Rs.shape[:2]

		NF_new = self.reconstructor.faces_highres.shape[0]
		interp_points = self.reconstructor.interp_points
		interp_weights = self.reconstructor.interp_weights
		faces_highres = self.reconstructor.faces_highres

		xyzs_skeleton_ = apply_lbs(
			pack(xyz.unsqueeze(1).repeat(1, N, 1, 1)),
			pack(global_Rs), pack(global_Ts),
			pack(lbs_weights.unsqueeze(1).repeat(1, N, 1, 1))) # (B*N) x NP x 3

		xyz_skeleton_highres_ = \
			(vector_gather(xyzs_skeleton_, interp_points.reshape(-1)[None].repeat(B * N, 1)).reshape(B * N, -1, 3, 3) \
			 * interp_weights[..., None]).sum(-2)
		xyz_per_face_ = vector_gather(
			xyz_skeleton_highres_,
			faces_highres.reshape(-1)[None].repeat(B * N, 1)
		).reshape(B * N, NF_new, 3, -1) # (B*N) x NF x 3 x 3
		centroid_ = xyz_per_face_.mean(-2)

		if tb is not None:
			tb.summ_pointcloud('gaussians', centroid_, torch.zeros_like(centroid_))

		S_ = torch.diag_embed(scale)
		R_ = unpack(so3_exp_map(pack(so3)), [S_.shape[0], S_.shape[1]])
		cov_local_ = R_ @ S_ @ S_.permute(0, 1, 3, 2) @ R_.permute(0, 1, 3, 2)  # B x NF x 3 x 3
		world_T_observation_ = get_transformation_from_triangle_steiner(xyz_per_face_)  # B x NF x 3 x 3
		cov_observation_ = world_T_observation_ @ cov_local_ @ world_T_observation_.permute(0, 1, 3, 2)

		if offset is not None:
			offset = offset[:, None].repeat(1, N, 1, 1)
			centroid_ += (world_T_observation_ @ pack(offset).unsqueeze(-1)).squeeze(-1)

		appearance_ = pack(appearance.unsqueeze(1).repeat(1, N, 1, 1))
		bg_feat = torch.zeros_like(appearance_[0, 0])
		bg_col = torch.cat([bg_feat, bg_feat.new_zeros([1])])

		K_ = pack(K)
		E_ = pack(E)

		rgbs_ = []
		masks_ = []
		render_infos_ = []
		for b in range(B * N):
			rgb, mask, render_info = self.renderer(
				centroid_[b:b+1],
				appearance_[b:b+1],
				K_[b:b+1], E_[b:b+1],
				bg_col=bg_col,
				skeleton_info={'cov': cov_observation_[b:b+1]})
			rgbs_.append(rgb)
			masks_.append(mask)
			render_infos_.append(render_info)
		rgbs = unpack(torch.cat(rgbs_, dim=0), [B, N])
		masks = unpack(torch.cat(masks_, dim=0), [B, N])
		render_infos = {}
		for key in render_infos_[0].keys():
			render_infos[key] = unpack(
				torch.cat([render_info_[key] for render_info_ in render_infos_], dim=0),
				[B, N])

		return rgbs, masks, {}

	def forward(self, input_imgs, input_masks,
				K, E,
				cnl_gtfms,
				dst_Rs, dst_Ts, dst_posevec=None,
				canonical_joints=None, canonical_vertices=None,
				bgcolor=None,
				tb=None,
				**kwargs):
		"""
			Parameters
			----------
			input_imgs : B x N x H x W x 3
			K: B x (N+1) x 3 x 3
			cnl_gtfms: B x J x 4 x 4

			Returns
			-------
		"""
		B, N = input_imgs.shape[:2]
		cnl_gtfms = cnl_gtfms.unsqueeze(1).repeat(1, input_imgs.shape[1] + 1, 1, 1, 1)  # B x (N+1) x J x 4 x 4
		global_Rs_, global_Ts_ = get_global_RTs(
			pack(cnl_gtfms), pack(dst_Rs), pack(dst_Ts),
			use_smplx=self.cfg.use_smplx,
			cnl_gtfms_inverse=True)
		global_Rs = unpack(global_Rs_, [B, N + 1])
		global_Ts = unpack(global_Ts_, [B, N + 1])

		outputs = {}

		# reconstruct
		canonical_rep = self.reconstruct(
			input_imgs, input_masks,
			K[:, :-1], E[:, :-1],
			global_Rs[:, :-1], global_Ts[:, :-1],
			canonical_vertices,
			bgcolor,
			tb=tb,
		)
		additional_outputs = canonical_rep[-1]
		canonical_rep = canonical_rep[:-1]
		outputs['canonical_rep'] = canonical_rep
		mesh_canonical = Meshes(pack(canonical_rep[0]), self.reconstructor.faces.repeat(B * canonical_rep[0].shape[0], 1, 1))
		outputs['mesh_canonical'] = mesh_canonical

		# render
		if self.training:
			# if training, render images from all iterations
			rgbs_all, masks_all = [], []
			for _ in range(canonical_rep[0].shape[0]):
				rgbs, masks, _ = self.render(
					[canonical_rep_item[_] for canonical_rep_item in canonical_rep],
					K[:, -1:], E[:, -1:],
					global_Rs[:, -1:], global_Ts[:, -1:],
					tb=tb
				)
				rgbs_all.append(rgbs)
				masks_all.append(masks)
			rgbs_all = torch.stack(rgbs_all).squeeze(2)  # Niters x B x H x W x C
			masks_all = torch.stack(masks_all).squeeze(2)

			outputs['source_imgs'] = additional_outputs['source_imgs']
			outputs['source_masks'] = additional_outputs['source_masks']
		else:
			# if inference, only render from the last iteration
			if len(canonical_rep[0].shape) == 4:
				canonical_rep = [canonical_rep_item[-1] for canonical_rep_item in canonical_rep]
			rgbs_all, masks_all, _ = self.render(
				canonical_rep,
				K[:, -1:], E[:, -1:],
				global_Rs[:, -1:], global_Ts[:, -1:],
				tb=tb
			)
			rgbs_all = rgbs_all.squeeze(1)
			masks_all = masks_all.squeeze(1)
		outputs.update(additional_outputs)

		return rgbs_all, masks_all, outputs

	def get_param_groups(self, cfg):
		param_groups = [
			{'params': self.img_encoder.parameters(), 'lr': cfg.lr.img_encoder},
			{'params': self.reconstructor.parameters(), 'lr': cfg.lr.reconstructor},
			{'params': self.renderer.parameters(), 'lr': cfg.lr.renderer}
		]
		return param_groups
