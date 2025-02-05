import cv2
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import body_util


def grayscale_visualization(im, vmin=None, vmax=None):
    # im should have shape H x W
    im_np = im.detach().cpu().numpy()
    fig = plt.figure()
    plt.imshow(im_np, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('mean value: %.3f' % np.mean(im_np))
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


class TBLogger:
    def __init__(self, log_dir, freq=1, only_scalar=False):
        self.sw = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.freq = freq
        self.only_scalar = only_scalar

        self.pca = None

    def set_global_step(self, global_step):
        self.global_step = global_step

    def summ_images(self, tag, images, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_images(tag, images.clamp(0, 1), global_step=global_step)

    def summ_image(self, tag, image, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_image(tag, image.clamp(0, 1), global_step=global_step)

    def summ_video(self, tag, video, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_video(tag, video.clamp(0, 1), global_step=global_step)

    def summ_scalar(self, tag, scalar, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_scalar(tag, scalar, global_step=global_step)

    def summ_text(self, tag, text_string, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_text(tag, text_string, global_step=global_step)

    def summ_pts_on_image(self, tag, image, pts, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            N = pts.shape[0]
            colors = np.array(sns.color_palette("coolwarm", N))
            image = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
            pts = pts.detach().cpu().numpy()
            for pt, color in zip(pts, colors):
                x, y = pt.astype(int)
                color = (color * 255.).astype(np.uint8).tolist()
                cv2.circle(image, (x, y), 2, color, -1)
            self.sw.add_image(tag, (torch.tensor(image) / 255.).permute(2, 0, 1), global_step=global_step)

    def summ_feat(self, tag, feat, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            feat_np = feat.detach().cpu().numpy()
            C, H, W = feat_np.shape
            feat_np = feat_np.transpose(1, 2, 0).reshape(-1, C)

            if self.pca is None:
                self.pca = PCA(n_components=3)
                self.pca.fit(feat_np)
            feat_rgb = self.pca.transform(feat_np).reshape(H, W, 3).transpose(2, 0, 1)
            feat_min = np.min(feat_rgb)
            feat_max = np.max(feat_rgb)
            feat_rgb = (feat_rgb - feat_min) / (feat_max - feat_min)
            self.sw.add_image(tag, feat_rgb, global_step=global_step)

    def summ_feats(self, tag, feats, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            feat_np = feats.detach().cpu().numpy()
            N, C, H, W = feat_np.shape
            feat_np = feat_np.transpose(0, 2, 3, 1).reshape(-1, C)

            if self.pca is None:
                self.pca = PCA(n_components=3)
                self.pca.fit(feat_np)
            feat_rgb = self.pca.transform(feat_np).reshape(N, H, W, 3).transpose(0, 3, 1, 2)
            feat_min = np.min(feat_rgb)
            feat_max = np.max(feat_rgb)
            feat_rgb = (feat_rgb - feat_min) / (feat_max - feat_min)
            self.sw.add_images(tag, feat_rgb, global_step=global_step)

    def summ_hist(self, tag, values, bins='tensorflow', global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_histogram(tag, values.reshape(-1), bins=bins, global_step=global_step)

    def summ_error(self, tag, err, vmin=None, vmax=None, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            err_img = grayscale_visualization(err, vmin, vmax)
            self.sw.add_image(tag, err_img.transpose(2, 0, 1), global_step=global_step)

    def summ_graph(self, model, inputs=None):
        self.sw.add_graph(model, inputs)

    def summ_pointcloud(self, tag, pts, colors, radius=None, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            colors = (colors * 255.).int()
            self.sw.add_mesh(tag, pts, colors=colors, global_step=global_step)

    def summ_mesh(self, tag, pts, faces, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_mesh(tag, pts, faces=faces, global_step=global_step)

    def summ_joints_2d(self, tag, imgs, pts, use_smplx=False, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            imgs = np.clip(imgs[0].permute(0, 2, 3, 1).detach().cpu().numpy(), a_min=0., a_max=1.)
            imgs = (imgs * 255.).astype(np.uint8)

            PARENT = body_util.SMPL_PARENT if not use_smplx else body_util.SMPLX_PARENT
            pts = pts.detach().cpu().numpy()[0].astype(int)
            for n in range(imgs.shape[0]):
                for key, value in PARENT.items():
                    if value == -1:
                        continue
                    cv2.line(imgs[n], pts[n, :, key], pts[n, :, value], (0, 0, 255), 2)
            self.sw.add_images(tag, imgs, global_step=global_step, dataformats='NHWC')


    def close(self):
        self.sw.close()