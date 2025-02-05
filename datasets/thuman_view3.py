import os
import pickle
import logging
import numpy as np
import cv2
import json

import torch
from .default import BaseDataset


class THumanview3Dataset(BaseDataset):
    """
    in each subject,
    during training, sample n_input_frames from the first total_input_frames frames as inputs,
    frames[n_input_frames:] are output frames
    during validation, the first n_input_frames are input frames, the rest are output frames
    TODO: simplify the dataloader and data preprocessing
    """
    def __init__(
            self, 
            dataset_root,
            scene_list,
            bgcolor=None,
            target_size=None,
            use_smplx=False,
            n_input_frames=5,
            subdivide_iter=0,
            total_input_frames=16,
            total_output_frames=48,
            input_frame_interval=[5, 6],
    ):
        self.input_frame_interval = input_frame_interval
        self.total_input_frames = total_input_frames
        self.total_output_frames = total_output_frames

        super(THumanview3Dataset, self).__init__(
            dataset_root,
            scene_list,
            bgcolor=bgcolor,
            target_size=target_size,
            use_smplx=use_smplx,
            n_input_frames=n_input_frames,
            subdivide_iter=subdivide_iter,
        )

    def create_per_frame_info(self):
        self.scene_ids_per_frame = []
        self.frame_ids_per_frame = []  # the frame id in each scene
        for scene_id, framelist in enumerate(self.framelists):
            # assert len(framelist) == self.total_input_frames + self.total_output_frames
            self.scene_ids_per_frame.extend([scene_id] * self.total_output_frames)
            self.frame_ids_per_frame.extend(list(range(self.total_input_frames, len(framelist))))

    def get_total_frames(self):
        return self.total_output_frames * len(self.scene_list)

    def gen_input_idxs(self, idx):
        if self.total_input_frames > self.n_input_frames:
            # select n_input_frames depending on pre-defined intervals
            input_idxs = [np.random.randint(0, self.total_input_frames)]
            for _ in range(self.n_input_frames - 1):
                next_idx = (input_idxs[-1] + np.random.choice(self.input_frame_interval)) % self.total_input_frames
                input_idxs.append(next_idx)
        else:
            input_idxs = list(range(self.total_input_frames))
        return input_idxs
