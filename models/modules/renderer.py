import logging
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.geom_util import cam_T_world
from utils.camera_util import focal2fov


class Renderer(nn.Module):
	def __init__(self, module_cfg, canonical_info, **kwargs):
		super().__init__()

		self.cfg = module_cfg

		self.renderer = GaussianRasterizer(None)
		self.save_K = module_cfg.save_K
		self.K_ndc = None

	def forward(self, xyzs_observation, appearance_feats, K, E, bg_col=None, skeleton_info=None, **kwargs):
		xyzs_observation = xyzs_observation.float()
		appearance_feats = appearance_feats.float()
		K = K.float()
		E = E.float()

		B, N, _ = xyzs_observation.shape
		assert B == 1

		assert 'cov' in skeleton_info
		cov = skeleton_info['cov'].float()  # B x N x 3 x 3

		w, h = self.cfg.img_size

		# if Ks are always save, we could save K to avoid synchronization time
		# remember to turn it off if K is changing across dataset, e.g., AIST++
		if self.K_ndc is None or not self.save_K:
			# set up the rasterization setting
			focalx, focaly = K[0, 0, 0].item(), K[0, 1, 1].item()
			px, py = K[0, 0, 2].item(), K[0, 1, 2].item()
			fovx = focal2fov(focalx, w)
			fovy = focal2fov(focaly, h)
			tanfovx = math.tan(fovx * 0.5)
			tanfovy = math.tan(fovy * 0.5)

			znear = 0.001
			zfar = 100

			K_ndc = torch.tensor([
				[2 * focalx / w, 0, (2 * px - w) / w, 0],
				[0, 2 * focaly / h, (2 * py - h) / h, 0],
				[0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
				[0, 0, 1, 0]
			]).float().to(K.device)

			self.K_ndc = K_ndc
			self.tanfovx, self.tanfovy = tanfovx, tanfovy

		cam_center = torch.linalg.inv_ex(E[0].T)[0][3, :3]

		feat = torch.cat([appearance_feats, torch.ones_like(appearance_feats[:, :, 0:1])], dim=-1)
		if bg_col is None:
			bg_col = feat.new_zeros([self.cfg.feat_dim])

		if 'img_size' in kwargs:
			w, h = kwargs['img_size']
		render_setting = GaussianRasterizationSettings(
			image_height=h,
			image_width=w,
			tanfovx=self.tanfovx,
			tanfovy=self.tanfovy,
			bg=bg_col,
			scale_modifier=1.,
			viewmatrix=E[0].T,
			projmatrix=E[0].T @ self.K_ndc.T,
			sh_degree=0,
			campos=cam_center,
			prefiltered=False,
			debug=False
		)
		self.renderer.raster_settings = render_setting

		means2D = torch.zeros_like(xyzs_observation[0], requires_grad=True)

		cov_packed = torch.stack([
			cov[0, :, 0, 0], cov[0, :, 0, 1], cov[0, :, 0, 2],
			cov[0, :, 1, 1], cov[0, :, 1, 2],
			cov[0, :, 2, 2]
		], dim=-1)

		C = feat.shape[-1]
		assert C == 4
		pred, _ = self.renderer(
			means3D=xyzs_observation[0],
			means2D=means2D,
			colors_precomp=feat[0],
			shs=None,
			opacities=torch.ones_like(appearance_feats[0, :, 0:1]),
			scales=None,
			rotations=None,
			cov3D_precomp=cov_packed)

		# transpose the image
		# gaussian splatting defines x along height
		pred = pred.permute(1, 2, 0)[None]

		return pred[..., :-1], pred[..., -1], {}
