###########################################
# imports
###########################################

import sys, glob
import imageio.v2 as imageio
import skimage.metrics
import numpy as np
import torch
import cv2
import os
from lpips import LPIPS
import shutil
import pdb
import argparse
import shutil


# Modifying from https://github.com/humansensinglab/Generalizable-Human-Gaussians/blob/main/metrics/compute_metrics.py
###########################################

tmp_ours = './tmp/eval_view{}/pred'
tmp_gt = './tmp/eval_view{}/gt'
tmperr = './tmp/eval_view{}/error'

def mae(imageA, imageB):
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err


###########################################

def mse(imageA, imageB):
    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, 2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err, errImage


###########################################

def func(gt_path, ours_path, n_input_frames):
    H = 1000
    W = 500

    psnrs, ssims, mses, maes = [], [], [], []

    humans = os.listdir(gt_path)
    humans.sort()

    for human_idx in range(len(humans)):

        human = humans[human_idx]

        for i in range(n_input_frames, n_input_frames + n_input_frames):

            sample_name = '{}_{:03d}'.format(human, i)
            print(sample_name)

            ours = os.path.join(ours_path, 'scene_{}_frame_{:06d}.png'.format(human, i))
            ours = imageio.imread(ours).astype('float32') / 255.

            gt_mask = os.path.join(gt_path, human, 'masks', 'frame_{:06d}.png'.format(i))
            gt_mask = imageio.imread(gt_mask).astype('float32') / 255.

            gt = os.path.join(gt_path, human, 'images', 'frame_{:06d}.png'.format(i))
            gt = imageio.imread(gt).astype('float32') / 255.


            if (ours.shape[0] != 1024) or (ours.shape[1] != 1024):
                ours = cv2.resize(ours, (gt.shape[1], gt.shape[0]))

            h0, w0 = h, w = ours.shape[0], ours.shape[1]

            # ---------------

            ii, jj = np.where(
                ~(gt == 0).all(-1))  # all background pixel coordinates

            try:
                # bounds for V direction
                hmin, hmax = np.min(ii), np.max(ii)
                uu = (H - (hmax + 1 - hmin)) // 2
                vv = H - (hmax - hmin) - uu
                if hmin - uu < 0:
                    hmin, hmax = 0, H
                elif hmax + vv > h:
                    hmin, hmax = h - H, h
                else:
                    hmin, hmax = hmin - uu, hmax + vv

                # bounds for U direction
                wmin, wmax = np.min(jj), np.max(jj)
                uu = (W - (wmax + 1 - wmin)) // 2
                vv = W - (wmax - wmin) - uu
                if wmin - uu < 0:
                    wmin, wmax = 0, W
                elif wmax + vv > w:
                    wmin, wmax = w - W, w
                else:
                    wmin, wmax = wmin - uu, wmax + vv

            except ValueError:
                print(f"target is empty")
                continue

            # crop images
            ours = ours[hmin: hmax, wmin: wmax]
            gt = gt[hmin: hmax, wmin: wmax]


            h, w = ours.shape[0], ours.shape[1]

            assert (h == H) and (w == W), f"error {hmin} {hmax} {wmin} {wmax} {h0} {w0} {uu} {vv}"

            mseValue, errImg = mse(ours, gt)

            errImg = (errImg * 255.0).astype(np.uint8)
            errImg = cv2.applyColorMap(errImg, cv2.COLORMAP_JET)

            subject_angle_name = '{}_{:03d}.png'.format(human, i)
            cv2.imwrite(os.path.join(tmperr, subject_angle_name), errImg)


            mseValue_ours_gt, errImg_ours_gt = mse(ours, gt)
            maeValue = mae(ours, gt)
            psnr = 10 * np.log10((1 ** 2) / mseValue_ours_gt)

            imageio.imsave("{}/{}_source.png".format(tmp_ours, sample_name),
                           (ours * 255).astype('uint8'))  # ours

            imageio.imsave("{}/{}_target.png".format(tmp_gt, sample_name),
                           (gt * 255).astype('uint8'))  # gt


            psnrs += [psnr]
            ssims += [skimage.metrics.structural_similarity(ours, gt, channel_axis=2,data_range=1)]
            # maes += [maeValue]
            mses += [mseValue]



    return np.asarray(psnrs), np.asarray(ssims), np.asarray(mses), np.asarray(maes)


###########################################


def evaluateErr(gt_path, ours_path, n_input_frames):

    psnrs, ssims, mses, maes = func(gt_path, ours_path, n_input_frames)

    ###########################################
    # PSNR & SSIM
    psnr = psnrs.mean()
    print(f"PSNR mean {psnr}", flush=True)
    ssim = ssims.mean()
    print(f"SSIM mean {ssim}", flush=True)

    ###########################################
    # LPIPS

    lpips = LPIPS(net='alex', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):

        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS Alex Mean {lpips}", flush=True)


    ###########

    lpips = LPIPS(net='vgg', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS VGG mean {lpips}", flush=True)


    ###########################################
    # FID

    os.system('python -m pytorch_fid --device cuda {} {}'.format(tmp_ours, tmp_gt))


######################################################################################
# parameters
######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='./data/thuman2.0/view3_val')
parser.add_argument('--ours_path', type=str, default='./log/exp_thuman2.0_view3/eval/view')
parser.add_argument('--n_input_frames', type=int, default=3)
args = parser.parse_args()

# path to GT rgb
gt = args.gt_path
# path to prediction
ours = args.ours_path
n_input_frames = args.n_input_frames

tmp_ours = tmp_ours.format(n_input_frames)
tmp_gt = tmp_gt.format(n_input_frames)
tmperr = tmperr.format(n_input_frames)

######################################################################################


print('###############################################', flush=True)
shutil.rmtree(tmperr, ignore_errors=True)
shutil.rmtree(tmp_ours, ignore_errors=True)
shutil.rmtree(tmp_gt, ignore_errors=True)

if not os.path.exists(tmperr):
    os.makedirs(tmperr, exist_ok=True)

if not os.path.exists(tmp_ours):
    os.makedirs(tmp_ours, exist_ok=True)

if not os.path.exists(tmp_gt):
    os.makedirs(tmp_gt, exist_ok=True)

evaluateErr(gt, ours, n_input_frames)

######################################################################################
