import os
import pickle
import numpy as np
import cv2
import logging
import pickle
import json

import torch
from .default import BaseDataset

from utils.image_util import load_image
from utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms
from utils.file_util import list_files, split_path
from utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_camrot


class PoseDataset(BaseDataset):
    RENDER_SIZE = 1024
    CAM_PARAMS = {
        # 'radius': 3.5, 'focal': 1250.
        'radius': 2.0, 'focal': 1250.
    }
    #
    # RENDER_SIZE = 1080
    # CAM_PARAMS = {
    #     # 'radius': 3.5, 'focal': 1250.
    #     'radius': 4.0, 'focal': 1250.
    # }

    def __init__(
            self,
            dataset_root,
            scene_list,
            scene_idx,
            pose_path,
            pose_name,
            bgcolor=[0.0, 0.0, 0.0],
            subdivide_iter=0,
            use_smplx=True,
            **_):
        self.use_smplx = use_smplx

        logging.info(f'[Dataset root] {dataset_root}')
        logging.info(f'[Pose path] {pose_path}')

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

        self.pose_path = pose_path

        self.dataset_root = dataset_root
        self.image_dir = os.path.join(dataset_root, self.scene, 'images')
        self.mask_dir = os.path.join(dataset_root, self.scene, 'masks')

        self.canonical_info = self.load_canonical_joints(self.scene, subdivide_iter=subdivide_iter)

        self.mesh_infos = self.load_train_mesh_infos(self.scene)
        self.pose_infos = self.load_pose_infos(self.pose_path)
        self.total_frames = len(self.pose_infos['Rh'])
        self.pose_name = pose_name

        # load input cameras
        self.input_cameras = self.load_train_cameras(self.scene)
        # setup evaluation camera
        K, E = self.setup_camera(img_size=self.RENDER_SIZE,
                                 **self.CAM_PARAMS)
        self.camera = {
            'K': [K] * self.total_frames,
            'E': [E] * self.total_frames
        }
        self.bgcolor = bgcolor
        self.target_size = [self.RENDER_SIZE, self.RENDER_SIZE]

    @staticmethod
    def setup_camera(img_size, radius, focal):
        x = 0.
        # x = radius * 0.9
        y = 1.2
        y = 0. # need to set to zeros for predicted sequence
        z = radius
        campos = np.array([x, y, z], dtype='float32')
        camrot = get_camrot(campos,
                            lookat=np.array([0, y, 0.]),
                            inv_camera=True)

        E = np.eye(4, dtype='float32')
        E[:3, :3] = camrot
        E[:3, 3] = -camrot.dot(campos)

        K = np.eye(3, dtype='float32')
        K[0, 0] = focal
        K[1, 1] = focal
        K[:2, 2] = img_size / 2.

        return K, E

    def load_pose_infos(self, path):
        data = dict(np.load(path))
        poses = data['poses'].reshape(-1, 55, 3)
        Rh = poses[:, 0].copy()

        Rh_matrix = np.zeros([Rh.shape[0], 3, 3])
        for i in range(Rh.shape[0]):
            Rh_matrix[i, :, :] = cv2.Rodrigues(Rh[i:i+1])[0]
        Rh_matrix = np.linalg.inv(Rh_matrix[0])[None] @ Rh_matrix
        for i in range(Rh.shape[0]):
            Rh[i] = cv2.Rodrigues(Rh_matrix[i])[0][:, 0]

        Th = data['trans']
        Th -= Th[:1]
        poses[:, 0] = 0.
        poses = poses.reshape(poses.shape[0], -1)
        pose_infos = {
            'poses': poses,
            'Rh': Rh,
            'Th': Th,
        }
        return pose_infos

    def query_input_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints':
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32') # need to set to zeros for predicted poses
        }

    def query_output_dst_skeleton(self, idx):
        return {
            'poses': self.pose_infos['poses'][idx].astype('float32'),
            'dst_tpose_joints': \
                self.canonical_info[0],
            'Rh': self.pose_infos['Rh'][idx].astype('float32'),
            'Th': self.pose_infos['Th'][0].astype('float32')
        }

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

    def __len__(self):
        return self.total_frames

    def load_input_data(self, frame_idx, bgcolor=None):
        frame_name = f'frame_{frame_idx:06d}'
        results = {}

        if bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')

        img, alpha, orig_W, orig_H = self.load_image(frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        dst_skel_info = self.query_input_dst_skeleton(frame_name)
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

        return results

    def load_output_data(self, idx, bgcolor=None):
        if isinstance(self.RENDER_SIZE, int):
            H, W = self.RENDER_SIZE, self.RENDER_SIZE
        else:
            H, W = self.RENDER_SIZE[1], self.RENDER_SIZE[0]
        results = {}

        dst_skel_info = self.query_output_dst_skeleton(idx)
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        results.update({
            'dst_tpose_joints': dst_tpose_joints,
        })

        K = self.camera['K'][idx].copy()

        E = self.camera['E'][idx]

        # recover
        E = apply_global_tfm_to_camera(
            E=E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'] - self.canonical_info[0][0])

        results.update({
            'K': K.astype(np.float32),
            'E': E.astype(np.float32),
        })

        # fake rgbs and masks
        results['target_rgbs'] = np.zeros([H, W, 3], dtype=np.float32)
        results['target_masks'] = np.zeros([H, W], dtype=np.float32)

        dst_Rs, dst_Ts = body_pose_to_body_RTs(
            dst_poses, dst_tpose_joints, use_smplx=self.use_smplx
        )
        results.update({
            'dst_Rs': dst_Rs,
            'dst_Ts': dst_Ts,
        })
        # 1. ignore global orientation
        # 2. add a small value to avoid all zeros
        dst_posevec_69 = dst_poses[3:] + 1e-2
        results.update({
            'dst_posevec': dst_posevec_69,
        })

        return results

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
            result = self.load_input_data(input_idx, bgcolor)
            for key, item in result.items():
                results[key].append(item)

        result = self.load_output_data(idx, bgcolor)
        for key, item in result.items():
            results[key].append(item)

        for key, item in results.items():
            results[key] = np.stack(item, axis=0)

        results['frame_name'] = f'scene_{self.scene}_name_{self.pose_name}_pose_{idx:04d}'

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
