import numpy as np
import os
import argparse
import pickle
import cv2

import torch

import sys
sys.path.append('.')
from utils.lpips import LPIPS
from skimage.metrics import structural_similarity

from utils.image_util import load_image

class Evaluator:
    """
    copied from https://github.com/zju3dv/neuralbody/blob/6bf1905822f71d1e568ef831110728fd1d06c94d/lib/evaluators/neural_volume.py
    adapted from https://github.com/escapefreeg/humannerf-eval/blob/master/eval.py
    """

    def __init__(self):
        self.lpips_model = LPIPS(net='vgg').cuda()
        for param in self.lpips_model.parameters():
           param.requires_grad = False
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt):
        try:
            ssim = structural_similarity(img_pred, img_gt, multichannel=True)
        except:
            ssim = structural_similarity(img_pred, img_gt, data_range=1, channel_axis=-1)
        return ssim

    def lpips_metric(self, img_pred, img_gt):
         # convert range from 0-1 to -1-1
         processed_pred = torch.from_numpy(img_pred).float().unsqueeze(0).cuda() * 2. - 1.
         processed_gt = torch.from_numpy(img_gt).float().unsqueeze(0).cuda() * 2. - 1.

         lpips_loss = self.lpips_model(processed_pred.permute(0, 3, 1, 2), processed_gt.permute(0, 3, 1, 2))
         return torch.mean(lpips_loss).cpu().detach().item() * 1000

    def evaluate(self, rgb_pred, rgb_gt):
        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt)
        self.ssim.append(ssim)

        lpips = self.lpips_metric(rgb_pred, rgb_gt)
        self.lpips.append(lpips)
        # print(mse, psnr, ssim, lpips)

    def summarize(self):
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpips: {}'.format(np.mean(self.lpips)))
        return np.mean(self.mse), np.mean(self.psnr), np.mean(self.ssim), np.mean(self.lpips)


def evaluate_scene(scene_name, gt_path, pred_path):
    evaluator = Evaluator()

    with open(os.path.join(gt_path, 'cameras.pkl'), 'rb') as f:
        cameras = pickle.load(f)

    for frame_name in sorted(cameras.keys()):
        if int(frame_name.split('_')[-1]) < 5:
            # input frames
            continue

        pred_img = np.array(load_image(os.path.join(pred_path, f'scene_{scene_name}_{frame_name}.png')))
        img = np.array(load_image(os.path.join(pred_path, f'scene_{scene_name}_{frame_name}_gt.png')))
        gt_img = (img / 255.).astype(np.float32)
        pred_img = (pred_img / 255.).astype(np.float32)
        evaluator.evaluate(pred_img, gt_img)

    print('scene_name: {}'.format(scene_name))
    mse, psnr, ssim, lpips = evaluator.summarize()
    return mse, psnr, ssim, lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='./data/aistpp/view5_trainval')
    parser.add_argument('--ours_path', type=str, default='./log/exp_aistpp_view5/eval/view')
    args = parser.parse_args()

    mse_all, psnr_all, ssim_all, lpips_all = [], [], [], []
    for scene in ['d16_val', 'd17_val', 'd18_val', 'd19_val', 'd20_val']:
        mse, psnr, ssim, lpips = evaluate_scene(scene, os.path.join(args.gt_path, scene), args.ours_path)
        mse_all.append(mse)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips)
    print('average:')
    print('mse: {}'.format(np.mean(mse_all)))
    print('psnr: {}'.format(np.mean(psnr_all)))
    print('ssim: {}'.format(np.mean(ssim_all)))
    print('lpips: {}'.format(np.mean(lpips_all)))

