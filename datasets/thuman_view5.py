import os
import pickle
import logging
import numpy as np
import cv2
import json

import torch
from .default import BaseDataset


class THumanview5Dataset(BaseDataset):
    """
    in each subject, the first n_input_frames are input frames
    the reset are output frames
    same for trainset and valset
    TODO: simplify the dataloader and data preprocessing
    """
    def create_per_frame_info(self):
        self.scene_ids_per_frame = []
        self.frame_ids_per_frame = []  # the frame id in each scene
        for scene_id, framelist in enumerate(self.framelists):
            self.scene_ids_per_frame.extend([scene_id] * (len(framelist) - self.n_input_frames))
            self.frame_ids_per_frame.extend(list(range(self.n_input_frames, len(framelist))))

    def get_total_frames(self):
        return sum([len(framelist) - self.n_input_frames for framelist in self.framelists])

    def gen_input_idxs(self, idx):
        input_idxs = list(range(self.n_input_frames))
        return input_idxs

