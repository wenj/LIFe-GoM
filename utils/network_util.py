import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.base_util import vector_gather


###############################################################################
## Network Components - Convolutional Decoders
###############################################################################

class ConvDecoder3D(nn.Module):
    r""" Convolutional 3D volume decoder."""

    def __init__(self, embedding_size=256, volume_size=128, voxel_channels=4):
        r""" 
            Args:
                embedding_size: integer
                volume_size: integer
                voxel_channels: integer
        """    
        super(ConvDecoder3D, self).__init__()

        self.block_mlp = nn.Sequential(nn.Linear(embedding_size, 1024), 
                                       nn.LeakyReLU(0.2))
        block_conv = []
        inchannels, outchannels = 1024, 512
        for _ in range(int(np.log2(volume_size)) - 1):
            block_conv.append(nn.ConvTranspose3d(inchannels, 
                                                 outchannels, 
                                                 4, 2, 1))
            block_conv.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        block_conv.append(nn.ConvTranspose3d(inchannels, 
                                             voxel_channels, 
                                             4, 2, 1))
        self.block_conv = nn.Sequential(*block_conv)

        for m in [self.block_mlp, self.block_conv]:
            initseq(m)

    def forward(self, embedding):
        """ 
            Args:
                embedding: Tensor (B, N)
        """    
        return self.block_conv(self.block_mlp(embedding).view(-1, 1024, 1, 1, 1))


###############################################################################
## Network Components - 3D rotations
###############################################################################

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)


###############################################################################
## Network Components - compute motion base
###############################################################################


SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}


class MotionBasisComputer(nn.Module):
    r"""Compute motion bases between the target pose and canonical pose."""

    def __init__(self, total_bones=24):
        super(MotionBasisComputer, self).__init__()
        self.total_bones = total_bones

    def _construct_G(self, R_mtx, T):
        r''' Tile ration matrix and translation vector to build a 4x4 matrix.

        Args:
            R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
            T:     Tensor (B, TOTAL_BONES, 3)

        Returns:
            G:     Tensor (B, TOTAL_BONES, 4, 4)
        '''
        batch_size, total_bones = R_mtx.shape[:2]
        assert total_bones == self.total_bones

        G = torch.zeros(size=(batch_size, total_bones, 4, 4),
                        dtype=R_mtx.dtype, device=R_mtx.device)
        G[:, :, :3, :3] = R_mtx
        G[:, :, :3, 3] = T
        G[:, :, 3, 3] = 1.0
    
        return G

    def forward(self, dst_Rs, dst_Ts, cnl_gtfms):
        r"""
        Args:
            dst_Rs:    Tensor (B, TOTAL_BONES, 3, 3)
            dst_Ts:    Tensor (B, TOTAL_BONES, 3)
            cnl_gtfms: Tensor (B, TOTAL_BONES, 4, 4)
                
        Returns:
            scale_Rs: Tensor (B, TOTAL_BONES, 3, 3)
            Ts:       Tensor (B, TOTAL_BONES, 3)
        """
        dst_gtfms = torch.zeros_like(cnl_gtfms)

        local_Gs = self._construct_G(dst_Rs, dst_Ts)    
        dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]

        for i in range(1, self.total_bones):
            dst_gtfms[:, i, :, :] = torch.matmul(
                                        dst_gtfms[:, SMPL_PARENT[i], 
                                                  :, :].clone(),
                                        local_Gs[:, i, :, :])

        dst_gtfms = dst_gtfms.view(-1, 4, 4)
        inv_dst_gtfms = torch.inverse(dst_gtfms)
        
        cnl_gtfms = cnl_gtfms.view(-1, 4, 4)
        f_mtx = torch.matmul(cnl_gtfms, inv_dst_gtfms)
        f_mtx = f_mtx.view(-1, self.total_bones, 4, 4)

        scale_Rs = f_mtx[:, :, :3, :3]
        Ts = f_mtx[:, :, :3, 3]

        return scale_Rs, Ts


class TVLoss(nn.Module):
    def __init__(self, res):
        super(TVLoss, self).__init__()
        self.res = res

    def forward(self, x):
        x = x.reshape(x.shape[0], self.res, self.res, self.res)
        c, h, w, d = x.shape

        count_h = x[:, 1:, :, :].numel()
        count_w = x[:, :, 1:, :].numel()
        count_d = x[:, :, :, 1:].numel()
        h_tv = torch.pow((x[:, 1:, :, :] - x[:, :h-1, :, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :w-1, :]), 2).sum()
        d_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :d-1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w + d_tv / count_d) / c


###############################################################################
## Init Functions
###############################################################################

def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


###############################################################################
## misc functions
###############################################################################


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


###############################################################################
## loss functions
###############################################################################

def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1) ** 2

    # loss = loss * weights
    return loss.mean()


def mesh_color_consistency(color, face_connectivity):
    # color is B x N x 3
    # face_connectivity is B x M x 2
    B, M, _ = face_connectivity.shape
    color_gather = vector_gather(color, face_connectivity.reshape(B, M * 2)) # B x (M*2) x 3
    color_gather = color_gather.reshape(B, M, 2, 3)
    color0 = color_gather[:, :, 0]
    color1 = color_gather[:, :, 1]
    loss = torch.abs(color0 - color1).mean()
    return loss

# --------------------------------------------
# following about GAN loss is copied from https://github.com/hyunbo9/SwinIR/
# --------------------------------------------

def basicblock_sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def basicblock_conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return basicblock_sequential(*L)


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='IL'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = basicblock_conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = basicblock_conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 64
        conv2 = basicblock_conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = basicblock_conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 128
        conv4 = basicblock_conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = basicblock_conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 256
        conv6 = basicblock_conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = basicblock_conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv8 = basicblock_conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = basicblock_conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = basicblock_sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.discriminator = Discriminator_VGG_96()

        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def compute_loss(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

    def forward_G(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2)
        gt = gt.permute(0, 3, 1, 2)

        pred_g_fake = self.discriminator(pred)
        pred_d_real = self.discriminator(gt)
        G_loss = (
            self.compute_loss(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
            self.compute_loss(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
        return G_loss

    def forward_D(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2)
        gt = gt.permute(0, 3, 1, 2)

        pred_d_fake = self.discriminator(pred).detach()  # 1) fake data, detach to avoid BP to G
        pred_d_real = self.discriminator(gt)  # 2) real data
        D_loss = 0.
        D_loss += 0.5 * self.compute_loss(pred_d_real - torch.mean(pred_d_fake, 0, True), True)

        pred_d_fake = self.discriminator(pred.detach())
        D_loss += 0.5 * self.compute_loss(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
        return D_loss

# --------------------------------------------
# SSIM loss: https://github.com/humansensinglab/Generalizable-Human-Gaussians/blob/6057f6719b686ab721ae274cfee70645c9ce698f/lib/loss.py#L53
# --------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
