import os
import pickle
import logging
import numpy as np
import cv2
import json

import torch
import torch.utils.data

from utils.image_util import load_image
from utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes, \
    get_joints_from_pose
from utils.file_util import list_files, split_path
from utils.camera_util import apply_global_tfm_to_camera
from utils.geom_util import subdivide


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_root,
            scene_list,
            bgcolor=None,
            target_size=None,
            use_smplx=False,
            n_input_frames=5,
            subdivide_iter=0,
    ):
        # if scenes are listed in a file, load the list
        if scene_list[0].endswith('.json'):
            new_scene_list = []
            for scene_list_file in scene_list:
                with open(scene_list_file) as f:
                    new_scene_list.extend(json.load(f))
            scene_list = new_scene_list

        logging.info(f'[Dataset root]: {dataset_root}, {len(scene_list)} scenes')

        self.dataset_root = dataset_root
        self.scene_list = scene_list

        self.bgcolor = bgcolor
        self.target_size = target_size
        self.use_smplx = use_smplx
        self.n_input_frames = n_input_frames

        # per scene information
        self.image_dirs = [os.path.join(dataset_root, scene_name, 'images') for scene_name in self.scene_list]
        self.mask_dirs = [os.path.join(dataset_root, scene_name, 'masks') for scene_name in self.scene_list]

        self.canonical_infos = [
            self.load_canonical_joints(scene_name, subdivide_iter=subdivide_iter) for scene_name in self.scene_list
        ]

        self.cameras = [self.load_train_cameras(scene_name) for scene_name in self.scene_list]
        self.mesh_infos = [self.load_train_mesh_infos(scene_name) for scene_name in self.scene_list]

        self.framelists = []
        for scene_name in self.scene_list:
            framelist = self.load_train_frames(scene_name)
            self.framelists.append(framelist)

        # per frame information
        self.create_per_frame_info()

    def create_per_frame_info(self):
        self.scene_ids_per_frame = []
        self.frame_ids_per_frame = []  # the frame id in each scene
        for scene_id, framelist in enumerate(self.framelists):
            self.scene_ids_per_frame.extend([scene_id] * len(framelist))
            self.frame_ids_per_frame.extend(list(range(len(framelist))))

    def load_canonical_joints(self, scene_name, subdivide_iter=0):
        cl_joint_path = os.path.join(self.dataset_root, scene_name, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')

        canonical_vertices = cl_joint_data['vertex'].astype('float32')
        canonical_lbs_weights = cl_joint_data['weights'].astype('float32')

        if 'edges' in cl_joint_data:
            canonical_edges = cl_joint_data['edges'].astype(int)
        else:
            canonical_edges = None

        if 'faces' in cl_joint_data:
            canonical_faces = cl_joint_data['faces']
        else:
            canonical_faces = None

        # subdivide
        for _ in range(subdivide_iter):
            attributes = {
                'weights': canonical_lbs_weights,
            }
            canonical_vertices, canonical_faces, attributes, canonical_edges, face_index = subdivide(
                canonical_vertices,
                canonical_faces,
                attributes,
                return_edges=True)
            canonical_lbs_weights = attributes['weights']

            canonical_vertices = canonical_vertices.astype(np.float32)
            canonical_lbs_weights = canonical_lbs_weights.astype(np.float32)

        return canonical_joints, canonical_vertices, canonical_lbs_weights, canonical_edges, canonical_faces

    def load_train_cameras(self, scene_name):
        with open(os.path.join(self.dataset_root, scene_name, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    def load_train_mesh_infos(self, scene_name):
        with open(os.path.join(self.dataset_root, scene_name, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)
        return mesh_infos

    def load_train_frames(self, scene_name):
        img_paths = list_files(os.path.join(self.dataset_root, scene_name, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, scene_id, frame_name):
        return {
            'poses': self.mesh_infos[scene_id][frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': 
                self.mesh_infos[scene_id][frame_name]['tpose_joints'].astype('float32'),
            'Rh': self.mesh_infos[scene_id][frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[scene_id][frame_name]['Th'].astype('float32')
        }
    
    def load_image(self, scene_id, frame_name, bg_color):
        imagepath = os.path.join(self.image_dirs[scene_id], '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))
        orig_H, orig_W, _ = orig_img.shape

        maskpath = os.path.join(self.mask_dirs[scene_id], '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))

        # undistort image
        if frame_name in self.cameras[scene_id] and 'distortions' in self.cameras[scene_id][frame_name]:
            K = self.cameras[scene_id][frame_name]['intrinsics']
            D = self.cameras[scene_id][frame_name]['distortions']
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

    def load_data(self, scene_id, idx, bgcolor=None):
        frame_name = self.framelists[scene_id][idx]
        results = {}

        if bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')

        img, alpha, orig_W, orig_H = self.load_image(scene_id, frame_name, bgcolor)
        img = (img / 255.).astype('float32')
        results.update({
            'target_rgbs': img,
            'target_masks': alpha[:, :, 0].astype(np.float32)
        })

        dst_skel_info = self.query_dst_skeleton(scene_id, frame_name)
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras[scene_id]
        K = self.cameras[scene_id][frame_name]['intrinsics'][:3, :3].copy()
        if self.target_size is not None:
            scale_w, scale_h = self.target_size[0] / orig_W, self.target_size[1] / orig_H
        else:
            scale_w, scale_h = 1., 1.
        K[:1] *= scale_w
        K[1:2] *= scale_h

        E = self.cameras[scene_id][frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
            E=E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'])

        results.update({
            'K': K.astype(np.float32),
            'E': E.astype(np.float32),
        })

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

    def get_total_frames(self):
        return sum([len(framelist) for framelist in self.framelists])

    def __len__(self):
        return self.get_total_frames()

    def gen_input_idxs(self, idx):
        scene_id = self.scene_ids_per_frame[idx]
        while True:
            input_idxs = np.random.choice(np.arange(len(self.framelists[scene_id])), size=self.n_input_frames, replace=False)
            if self.frame_ids_per_frame[idx] not in input_idxs:
                break
        return input_idxs

    def __getitem__(self, idx):
        scene_id = self.scene_ids_per_frame[idx]
        scene_name = self.scene_list[scene_id]
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

        input_idxs = self.gen_input_idxs(idx)
        results['bgcolor'] = bgcolor / 255.
        for input_idx in input_idxs:
            result = self.load_data(scene_id, input_idx, bgcolor)
            for key, item in result.items():
                results[key].append(item)

        result = self.load_data(scene_id, self.frame_ids_per_frame[idx], bgcolor)
        for key, item in result.items():
            results[key].append(item)

        for key, item in results.items():
            results[key] = np.stack(item, axis=0)

        frame_name = self.framelists[scene_id][self.frame_ids_per_frame[idx]]
        results['frame_name'] = f'scene_{scene_name}_{frame_name}'

        # shared information
        canonical_joints, canonical_vertices, canonical_lbs_weights, canonical_edges, canonical_faces \
            = self.canonical_infos[scene_id]

        cnl_gtfms = get_canonical_global_tfms(canonical_joints, use_smplx=self.use_smplx)
        results['cnl_gtfms'] = np.linalg.inv(cnl_gtfms)
        results['canonical_vertices'] = canonical_vertices

        return results

    def get_canonical_info(self):
        # only return the canonical information shared across the dataset
        canonical_joints, canonical_vertex, canonical_lbs_weights, canonical_edges, canonical_faces \
            = self.canonical_infos[0]
        info = {
            'edges': canonical_edges,
            'faces': canonical_faces,
            'canonical_lbs_weights': canonical_lbs_weights,
            'canonical_vertex': canonical_vertex,
        }
        return info
