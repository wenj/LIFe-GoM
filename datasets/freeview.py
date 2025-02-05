import os
import pickle
import logging
import json

import numpy as np
import cv2
import torch
from .default import BaseDataset

from utils.image_util import load_image
from utils.body_util import \
	body_pose_to_body_RTs, \
	get_canonical_global_tfms
from utils.file_util import list_files, split_path
from utils.camera_util import \
	apply_global_tfm_to_camera, \
	rotate_camera_by_frame_idx


class FreeviewDataset(BaseDataset):
	ROT_CAM_PARAMS = {'rotate_axis': 'y', 'inv_angle': False}

	def __init__(
			self,
			dataset_root,
			scene_list,
			scene_idx=0,
			total_frames=100,
			bgcolor=None,
			target_size=None,
			use_smplx=False,
			subdivide_iter=0,
			**_):

		self.dataset_root = dataset_root
		self.total_frames = total_frames

		# load the scene
		# meta files which contains all information about scene, selected input frames and output frames
		if scene_list[0].endswith('.json'):
			meta_infos = []
			for scene_list_file in scene_list:
				with open(scene_list_file) as f:
					meta_infos.extend(json.load(f))

		self.meta_info = meta_infos[scene_idx]
		self.scene = self.meta_info['scene']
		self.input_frames = self.meta_info['input_frames']
		self.output_frame = self.meta_info['output_frame']
		logging.info(f' -- Scene Idx: {scene_idx}')

		self.use_smplx = use_smplx

		self.image_dir = os.path.join(dataset_root, self.scene, 'images')
		self.mask_dir = os.path.join(dataset_root, self.scene, 'masks')

		self.canonical_info = self.load_canonical_joints(self.scene, subdivide_iter=subdivide_iter)

		self.input_cameras = self.load_train_cameras(self.scene)
		self.mesh_infos = self.load_train_mesh_infos(self.scene)

		self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]

		self.target_size = [1024, 1024]
		if target_size is not None:
			self.target_size = target_size

	def query_dst_skeleton(self, frame_idx):
		return {
			'poses': self.mesh_infos[frame_idx]['poses'].astype('float32'),
			'dst_tpose_joints': \
				self.mesh_infos[frame_idx]['tpose_joints'].astype('float32'),
			'Rh': self.mesh_infos[frame_idx]['Rh'].astype('float32'),
			'Th': self.mesh_infos[frame_idx]['Th'].astype('float32')
		}

	def get_freeview_camera(self, E, frame_idx, total_frames, trans=None):
		E = rotate_camera_by_frame_idx(
			extrinsics=E,
			frame_idx=frame_idx,
			period=total_frames,
			trans=trans,
			**self.ROT_CAM_PARAMS)
		K = self.input_cameras['frame_{:06d}'.format(self.output_frame)]['intrinsics'].copy()
		return K, E

	def load_image(self, frame_name, bg_color):
		imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
		orig_img = np.array(load_image(imagepath))
		orig_H, orig_W, _ = orig_img.shape

		maskpath = os.path.join(self.mask_dir, '{}.png'.format(frame_name))
		alpha_mask = np.array(load_image(maskpath))

		# undistort image
		if frame_name in self.input_cameras and 'distortions' in self.input_cameras[frame_name]:
			K = self.input_cameras[frame_name]['intrinsics']
			D = self.input_cameras[frame_name]['distortions']
			orig_img = cv2.undistort(orig_img, K, D)
			alpha_mask = cv2.undistort(alpha_mask, K, D)

		alpha_mask = alpha_mask / 255.
		img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
		if self.target_size is not None:
			w, h = self.target_size
			img = cv2.resize(img, [w, h],
							 interpolation=cv2.INTER_LANCZOS4)
			alpha_mask = cv2.resize(alpha_mask, [w, h],
									interpolation=cv2.INTER_LINEAR)

		return img, alpha_mask, orig_W, orig_H

	def load_data(self, frame_idx, bgcolor=None, tpose=False):
		frame_name = f'frame_{frame_idx:06d}'
		results = {}

		if bgcolor is None:
			bgcolor = (np.random.rand(3) * 255.).astype('float32')

		img, alpha, orig_W, orig_H = self.load_image(frame_name, bgcolor)
		img = (img / 255.).astype('float32')

		H, W = img.shape[0:2]

		dst_skel_info = self.query_dst_skeleton(frame_name)
		dst_poses = dst_skel_info['poses']
		dst_tpose_joints = dst_skel_info['dst_tpose_joints']

		assert frame_name in self.input_cameras
		K = self.input_cameras[frame_name]['intrinsics'][:3, :3].copy()
		if self.target_size is not None:
			scale_w, scale_h = self.target_size[0] / orig_W, self.target_size[1] / orig_H
		else:
			scale_w, scale_h = 1., 1.
		K[:1] *= scale_w
		K[1:2] *= scale_h

		E = self.input_cameras[frame_name]['extrinsics']
		E = apply_global_tfm_to_camera(
			E=E,
			Rh=dst_skel_info['Rh'],
			Th=dst_skel_info['Th'])

		results.update({
			'K': K.astype(np.float32),
			'E': E.astype(np.float32),
		})

		results['target_rgbs'] = img
		results['target_masks'] = alpha[:, :, 0].astype(np.float32)

		if tpose:
			dst_poses_new = np.zeros_like(dst_poses)
			dst_poses = dst_poses_new
		dst_Rs, dst_Ts = body_pose_to_body_RTs(
			dst_poses, dst_tpose_joints, use_smplx=self.use_smplx
		)
		results.update({
			'dst_Rs': dst_Rs,
			'dst_Ts': dst_Ts,
		})

		# 1. ignore global orientation
		# 2. add a small value to avoid all zeros
		dst_posevec_69 = dst_poses.reshape(-1)[3:] + 1e-2
		results.update({
			'dst_posevec': dst_posevec_69,
		})

		results.update({
			'dst_tpose_joints': dst_tpose_joints,
		})

		return results, scale_w, scale_h

	def __len__(self):
		return self.total_frames

	def __getitem__(self, idx):
		input_idxs = self.input_frames

		results = {
			'K': [],
			'E': [],
			'dst_Rs': [],
			'dst_Ts': [],
			'dst_posevec': [],
			'dst_tpose_joints': [],
			'target_rgbs': [],
			'target_masks': [],
		}
		if self.bgcolor is None:
			bgcolor = (np.random.rand(3) * 255.).astype('float32')
		else:
			bgcolor = np.array(self.bgcolor, dtype='float32')
		results['bgcolor'] = bgcolor / 255.
		for input_idx in input_idxs:
			result, _, _ = self.load_data(input_idx, bgcolor)
			for key, item in result.items():
				results[key].append(item)

		result, scale_w, scale_h = self.load_data(self.output_frame, bgcolor, tpose=False)
		for key, item in result.items():
			results[key].append(item)

		# replace the target view with new camera poses
		dst_skel_info = self.query_dst_skeleton('frame_{:06d}'.format(self.output_frame))
		dst_Rh = dst_skel_info['Rh']
		dst_Th = dst_skel_info['Th']

		E = self.input_cameras['frame_{:06d}'.format(self.output_frame)]['extrinsics']
		K, E = self.get_freeview_camera(
			E=E,
			frame_idx=idx,
			total_frames=self.total_frames,
			trans=dst_Th)
		K[:1] *= scale_w
		K[1:2] *= scale_h

		E = apply_global_tfm_to_camera(
			E=E,
			Rh=dst_Rh,
			Th=dst_Th)

		results['K'][-1] = K.astype(np.float32)
		results['E'][-1] = E.astype(np.float32)

		for key, item in results.items():
			results[key] = np.stack(item, axis=0)

		results['frame_name'] = f'scene_{self.scene}_frame_{self.output_frame:06d}_free_{idx:04d}'

		# shared information
		canonical_joints, canonical_vertices, canonical_lbs_weights, canonical_edges, canonical_faces \
			= self.canonical_info

		cnl_gtfms = get_canonical_global_tfms(canonical_joints, use_smplx=self.use_smplx)
		results['cnl_gtfms'] = np.linalg.inv(cnl_gtfms)
		results['canonical_vertices'] = canonical_vertices

		return results

	def get_canonical_info(self):
		# only return the canonical information shared across the dataset
		canonical_joints, canonical_vertex, canonical_lbs_weights, canonical_edges, canonical_faces \
			= self.canonical_info
		info = {
			'edges': canonical_edges,
			'faces': canonical_faces,
			'canonical_lbs_weights': canonical_lbs_weights,
			'canonical_vertex': canonical_vertex,
		}
		return info

	def get_all_Es(self):
		Es = []
		for idx in range(self.total_frames):
			dst_skel_info = self.query_dst_skeleton()
			dst_Rh = dst_skel_info['Rh']
			dst_Th = dst_skel_info['Th']

			E = self.train_camera['extrinsics']
			K, E = self.get_freeview_camera(
				E=E,
				frame_idx=idx,
				total_frames=self.total_frames,
				trans=dst_Th)
			E = apply_global_tfm_to_camera(
				E=E,
				Rh=dst_Rh,
				Th=dst_Th)

			Es.append(E)
		return np.stack(Es, axis=0)