import os
import argparse

import torch

from .yacs import CfgNode as CN


# pylint: disable=redefined-outer-name


def make_cfg(cfg_filename):
	cfg = CN()
	cfg.merge_from_file('configs/default.yaml')
	if cfg_filename is not None:
		cfg.merge_from_file(cfg_filename)

	if hasattr(cfg, 'save_root'):
		cfg.save_dir = os.path.join(cfg.save_root, cfg.exp_name)
	else:
		cfg.save_dir = os.path.join('log', cfg.exp_name)

	# prepare the cfg
	cfg.dataset.use_smplx = cfg.use_smplx
	cfg.model.use_smplx = cfg.use_smplx
	cfg.model.img_size = cfg.img_size
	cfg.model.reconstructor.n_input_frames = cfg.n_input_frames

	return cfg
