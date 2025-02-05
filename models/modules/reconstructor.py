import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from trimesh.remesh import faces_to_edges, grouping

from pytorch3d.transforms.so3 import so3_exp_map
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
import nvdiffrast
import nvdiffrast.torch

from utils.body_util import apply_lbs, get_global_RTs, get_joints
from utils.geom_util import clip_T_world, ndc_T_world, img_T_cam, cam_T_world
from utils.base_util import pack, unpack, vector_gather

from models.nets.point_transformer import pointtransformer_enc_repro, PointTransformerLayer


def visibility_check_nvdiffrast(rasterize_context, xyz, K, E, faces, resolution):
    # now assume resolution is [H, W]
    NP = xyz.shape[1]
    NF = faces.shape[0]
    visibility_map = xyz.new_zeros(xyz.shape[0], NP + 1)

    resolution_new_0 = (resolution[0] // 8 + ((resolution[0] % 8) > 0)) * 8
    resolution_new_1 = (resolution[1] // 8 + ((resolution[1] % 8) > 0)) * 8

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        xyzs_clip = clip_T_world(xyz.permute(0, 2, 1).float(), K.float(), E.float(), resolution_new_0,
                                 resolution_new_1).contiguous()

        # the resolution for nvdiffrast is [H, W]
        outputs, _ = nvdiffrast.torch.rasterize(rasterize_context, xyzs_clip, faces.type(torch.int32),
                                                [resolution_new_0, resolution_new_1])
        triangle_ids = outputs[..., -1].long() - 1

    for i, triangle_id in enumerate(triangle_ids):
        visible_faces = triangle_id.reshape(-1)
        visibility_face = torch.zeros([NF + 1], dtype=torch.bool, device=xyz.device)
        visibility_face.index_fill_(0, visible_faces + 1, 1)
        visibility_face = visibility_face[..., None].repeat(1, 3)

        faces_w_invalid = faces.clone()
        faces_w_invalid.masked_fill_(~visibility_face[1:], -1)
        visibility_map[i].index_fill_(0, faces_w_invalid.reshape(-1) + 1, 1)

    return visibility_map[:, 1:]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask[:, None].repeat(1, self.num_heads, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, d_out):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_out, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, d_out)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        attn_output = self.self_attn(Q, K, V, mask)
        x = self.norm1(Q + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


def get_transformation_from_triangle_steiner(triangles, sigma=0.001):
    """
    triangles: B x NF x 3 x 3
    """
    centroid = triangles.mean(dim=-2)

    f1 = 0.5 * (triangles[..., 2, :] - centroid)
    f2 = 1 / (2 * np.sqrt(3)) * (triangles[..., 1, :] - triangles[..., 0, :])
    t0 = torch.atan2((2 * f1 * f2).sum(-1), ((f1 * f1).sum(-1) - (f2 * f2).sum(-1))) / 2
    t0 = t0[..., None]

    axis0 = f1 * torch.cos(t0) + f2 * torch.sin(t0)
    axis1 = f1 * torch.cos(t0 + np.pi / 2) + f2 * torch.sin(t0 + np.pi / 2)

    normal = torch.cross(axis0, axis1, dim=-1)
    normal = F.normalize(normal, dim=-1) * sigma
    transform = torch.stack([axis0 * 2, axis1 * 2, normal], dim=-1)
    return transform


def subdivide(faces, n_subdivide, NP):
    edges = np.sort(faces_to_edges(faces), axis=1) # (NF*3) x 2
    unique, inverse = grouping.unique_rows(edges)
    edges = edges[unique]

    # each point can be interpolated by at most three points
    # initial points are only related to themselves
    interp_points = np.zeros([NP, 3], dtype=np.int64)
    interp_points[:] = -1
    interp_points[:, 0] = np.arange(0, NP)

    weights = np.zeros([NP, 3], dtype=np.float32)
    weights[:, 0] = 1.

    for i in range(n_subdivide):
        NP_add = edges.shape[0]
        interp_points = np.concatenate([interp_points, np.full([edges.shape[0], 3], -1, dtype=np.int64)], axis=0)
        weights = np.concatenate([weights, np.zeros([edges.shape[0], 3], dtype=np.float32)], axis=0)

        # compute the interp points and weights for new created points
        for edge_idx, edge in enumerate(edges):
            pts = np.concatenate([interp_points[edge[0]], interp_points[edge[1]]], axis=0)
            ws = np.concatenate([weights[edge[0]], weights[edge[1]]], axis=0)

            unique_pts = []
            unique_ws = []
            for pt, w in zip(pts, ws):
                if pt != -1:
                    if pt in unique_pts:
                        idx = unique_pts.index(pt)
                        unique_ws[idx] += w / 2
                    else:
                        unique_pts.append(pt)
                        unique_ws.append(w / 2)

            interp_points[NP + edge_idx, :len(unique_pts)] = np.array(unique_pts)
            weights[NP + edge_idx, :len(unique_pts)] = np.array(unique_ws)

        # get new faces
        mid_idx = inverse.reshape((-1, 3)) + NP
        faces = np.column_stack([faces[:, 0],
                                 mid_idx[:, 0],
                                 mid_idx[:, 2],
                                 mid_idx[:, 0],
                                 faces[:, 1],
                                 mid_idx[:, 1],
                                 mid_idx[:, 2],
                                 mid_idx[:, 1],
                                 faces[:, 2],
                                 mid_idx[:, 0],
                                 mid_idx[:, 1],
                                 mid_idx[:, 2]]).reshape((-1, 3))

        # get new edges
        edges = np.sort(faces_to_edges(faces), axis=1)
        unique, inverse = grouping.unique_rows(edges)
        edges = edges[unique]

        NP += NP_add

    return faces, interp_points, weights, NP


class Reconstructor(nn.Module):
    def __init__(self, module_cfg, canonical_info, **kwargs):
        super().__init__()

        self.cfg = module_cfg

        self.appearance_feat_dim = module_cfg.appearance_feat_dim
        self.n_input_frames = module_cfg.n_input_frames

        NF = canonical_info['faces'].shape[0]
        NP = canonical_info['canonical_lbs_weights'].shape[0]
        self.NP = NP
        self.NF = NF

        # subdivide the mesh
        self.n_subdivide = module_cfg.n_subdivide
        faces = canonical_info['faces'].astype(np.int64)
        faces_highres, interp_points, interp_weights, NP_new = subdivide(faces, self.n_subdivide, NP)
        self.faces = torch.tensor(faces).long().cuda()
        self.faces_highres = torch.tensor(faces_highres).long().cuda()
        self.interp_points = torch.tensor(interp_points).long().cuda() # NP x 3; -1 indicates None
        # remove -1, otherwise it will trigger error
        # won't affect since interp_weights are set to 0s.
        self.interp_points[self.interp_points == -1] = 0
        self.interp_weights = torch.tensor(interp_weights).float().cuda() # NP x 3
        self.NP_new = NP_new
        self.NF_new = faces_highres.shape[0]

        # gaussian splat attributes; init
        self.so3 = torch.zeros(self.faces_highres.shape[0], 3).float().cuda()
        self.scale = torch.ones(self.faces_highres.shape[0], 3).float().cuda()
        self.offset = torch.zeros(self.faces_highres.shape[0], 3).float().cuda()

        # appearance feature
        self.appearance = torch.zeros(self.faces_highres.shape[0], module_cfg.appearance_feat_dim).float().cuda()

        # lbs weights
        self.lbs_weights = torch.tensor(canonical_info['canonical_lbs_weights']).float().cuda()

        # visilibity check using nvdiffrast
        self.rasterize_context = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
        self.visibility_embed = nn.Parameter(torch.zeros([module_cfg.img_feat_dim]))
        self.resolution = (module_cfg.img_size[1], module_cfg.img_size[0])  # (H, W)

        # multiview fusion
        self.vertex_embed = nn.Parameter(torch.randn(self.lbs_weights.shape[0], module_cfg.img_feat_dim))
        self.multiview_embed = nn.Sequential(
            EncoderLayer(module_cfg.img_feat_dim, 6, module_cfg.img_feat_dim, module_cfg.img_feat_dim),
            EncoderLayer(module_cfg.img_feat_dim, 6, module_cfg.img_feat_dim, module_cfg.img_feat_dim)
        )
        gom_encoder_in_dim = module_cfg.img_feat_dim * 2 + 3

        # point transformer to embed pointcloud with features
        self.gom_encoder = pointtransformer_enc_repro(c=gom_encoder_in_dim)
        self.gom_encoder.load_pretrained_weight(module_cfg.pretrain)
        self.gom_encoder_out_dim = 32

        # update heads
        self.xyz_update_layer = nn.Sequential(
            nn.Linear(self.gom_encoder_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.embed_update_layer = nn.Linear(self.gom_encoder_out_dim, module_cfg.img_feat_dim)

        self.gaussian_update_in_dim = module_cfg.img_feat_dim * self.n_input_frames + self.gom_encoder_out_dim * 3
        self.so3_update_layer = nn.Sequential(
            nn.Linear(self.gaussian_update_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.scale_update_layer = nn.Sequential(
            nn.Linear(self.gaussian_update_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.appearance_update_layer = nn.Sequential(
            nn.Linear(self.gaussian_update_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.offset_update_layer = nn.Sequential(
            nn.Linear(self.gaussian_update_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # stabilize the training
        init_val = 1e-5
        self.xyz_update_layer[-1].weight.data.uniform_(-init_val, init_val)
        self.xyz_update_layer[-1].bias.data.zero_()
        self.so3_update_layer[-1].weight.data.uniform_(-init_val, init_val)
        self.so3_update_layer[-1].bias.data.zero_()
        self.scale_update_layer[-1].weight.data.uniform_(-init_val, init_val)
        self.scale_update_layer[-1].bias.data.zero_()
        self.offset_update_layer[-1].weight.data.uniform_(-init_val, init_val)
        self.offset_update_layer[-1].bias.data.zero_()

    def update(self,
               update_list,
               imgs, img_masks, img_feats,
               rendered_imgs, rendered_img_masks, rendered_img_feats,
               K, E,
               global_Rs, global_Ts,
               tb=None):
        B, N, *_ = K.shape
        xyz_all = update_list['xyz']  # B x NP x 3
        NP = xyz_all.shape[1]
        NP_new = self.NP_new
        NF = self.faces.shape[0]
        NF_new = self.faces_highres.shape[0]
        _, _, _, H, W = imgs.shape
        device = imgs.device

        # warp by lbs
        Rs_, Ts_ = pack(global_Rs), pack(global_Ts)
        lbs_weights = update_list['lbs_weights']
        lbs_weights = pack(lbs_weights[:, None].repeat(1, N, 1, 1))
        xyz_skeleton_all_ = apply_lbs(
            pack(xyz_all[:, None].repeat(1, N, 1, 1)),
            Rs_, Ts_,
            lbs_weights)  # (B*N) x NP x 3

        xyz_skeleton_highres_ = \
            (vector_gather(xyz_skeleton_all_, self.interp_points.reshape(-1)[None].repeat(B * N, 1)).reshape(B * N, -1, 3, 3) \
             * self.interp_weights[..., None]).sum(-2)

        # visibility check
        visibility_map_highres = visibility_check_nvdiffrast(
            self.rasterize_context, xyz_skeleton_highres_,
            pack(K), pack(E),
            self.faces_highres, self.resolution)  # (B*N) x NP_new
        visibility_map = visibility_map_highres[:, :NP]

        # project to 2d images and sample pixel-aligned feats
        xyz_2d_ = img_T_cam(cam_T_world(xyz_skeleton_all_.permute(0, 2, 1), pack(E)), pack(K)) # (B*N) x 2 x NP
        xyz_2d_[:, 0] = xyz_2d_[:, 0] / W * 2 - 1.
        xyz_2d_[:, 1] = xyz_2d_[:, 1] / H * 2 - 1.
        xyz_feats_2d_ = []
        for img_feat in img_feats:
            xyz_feats_2d_.append(F.grid_sample(pack(img_feat), xyz_2d_.permute(0, 2, 1).unsqueeze(1), align_corners=False).squeeze(2)) # (B*N) x C x (P*K)
        xyz_feats_2d_ = torch.cat(xyz_feats_2d_, dim=1)
        xyz_feats_2d_ = unpack(xyz_feats_2d_, [B, N]).permute(0, 3, 1, 2) # B x NP x N x C

        # multiview fusion
        xyz_feats_2d_ = xyz_feats_2d_ * visibility_map.reshape(B, N, NP).permute(0, 2, 1)[:, :, :, None] \
            + self.visibility_embed[None, None, None, :] * (1 - visibility_map.reshape(B, N, NP).permute(0, 2, 1)[:, :, :, None])

        xyz_feats_2d_ = self.multiview_embed[0](pack(xyz_feats_2d_), pack(xyz_feats_2d_), pack(xyz_feats_2d_))
        xyz_feats_2d_ = self.multiview_embed[1](
            pack(update_list['embed'])[:, None], xyz_feats_2d_,
            xyz_feats_2d_)
        xyz_feats_2d = unpack(xyz_feats_2d_, [B, NP])

        # same for rendered images
        rendered_xyz_feats_2d_ = []
        for img_feat in rendered_img_feats:
            rendered_xyz_feats_2d_.append(
                F.grid_sample(pack(img_feat), xyz_2d_.permute(0, 2, 1).unsqueeze(1), align_corners=False).squeeze(2))  # (B*N) x C x (P*K)
        rendered_xyz_feats_2d_ = torch.cat(rendered_xyz_feats_2d_, dim=1)
        rendered_xyz_feats_2d_ = unpack(rendered_xyz_feats_2d_, [B, N]).permute(0, 3, 1, 2)  # B x NP x N x C
        rendered_visibility_map = F.grid_sample(pack(rendered_img_masks)[:, None],
                                                xyz_2d_.permute(0, 2, 1).unsqueeze(1),
                                                align_corners=False).squeeze(2).squeeze(1)  # (B*N) x NP
        rendered_xyz_feats_2d_ = rendered_xyz_feats_2d_ * rendered_visibility_map.reshape(B, N, NP).permute(0, 2, 1)[:, :, :, None] \
                                 + self.visibility_embed[None, None, None, :] * (1 - rendered_visibility_map.reshape(B, N, NP).permute(0, 2, 1)[:, :, :, None])

        rendered_xyz_feats_2d_ = self.multiview_embed[0](pack(rendered_xyz_feats_2d_),
                                                         pack(rendered_xyz_feats_2d_),
                                                         pack(rendered_xyz_feats_2d_))
        rendered_xyz_feats_2d_ = self.multiview_embed[1](
            pack(update_list['embed'])[:, None], rendered_xyz_feats_2d_,
            rendered_xyz_feats_2d_)
        rendered_xyz_feats_2d = unpack(rendered_xyz_feats_2d_, [B, NP])
        rendered_xyz_feats_2d = rendered_xyz_feats_2d.detach()

        # PointTransformer
        xyz_feats_2d = torch.cat([xyz_feats_2d.reshape(B, NP, -1), rendered_xyz_feats_2d.reshape(B, NP, -1)], dim=-1)
        delta_feat = self.gom_encoder((
            update_list['xyz'].reshape(-1, 3),
            xyz_feats_2d.reshape(B * NP, -1),
            torch.full([B], NP, dtype=torch.int, device=xyz_feats_2d.device)))
        delta_feat = delta_feat.reshape(B, NP, -1)

        # update point locations and point embedding in low-res mesh
        delta_xyz = self.xyz_update_layer(delta_feat)
        delta_embed = self.embed_update_layer(delta_feat)

        ## now work on high-res mesh
        # collect face features from vertex features
        delta_feat_highres = \
            (vector_gather(delta_feat, self.interp_points.reshape(-1)[None].repeat(B, 1)).reshape(B, NP_new, 3, -1) \
             * self.interp_weights[..., None]).sum(-2)
        delta_face_feat_high = vector_gather(
            delta_feat_highres,
            self.faces_highres.reshape(-1)[None].repeat(B, 1)
        ).reshape(B, NF_new, -1)

        # xyz_2d_ is (B*N) x 2 x (NP*NA), it is normalized by H and W
        xyz_2d_highres_ = img_T_cam(cam_T_world(xyz_skeleton_highres_.permute(0, 2, 1), pack(E)), pack(K))  # (B*N) x 2 x NP_new
        xyz_2d_highres_[:, 0] = xyz_2d_highres_[:, 0] / W * 2 - 1.
        xyz_2d_highres_[:, 1] = xyz_2d_highres_[:, 1] / H * 2 - 1.
        gaussian_means = vector_gather(
            xyz_2d_highres_.permute(0, 2, 1),
            self.faces_highres.reshape(-1)[None].repeat(B * N, 1)).reshape(B * N, NF_new, 3, -1).mean(-2)  # (B*N) x NF_new x 2

        # get face visibility in high-resolution mesh
        visibility_face = vector_gather(
            visibility_map_highres[..., None],
            self.faces_highres.reshape(-1)[None].repeat(B * N, 1)
        ).reshape(B * N, NF_new, 3, -1).min(-2).values  # (B*N) x NF_new x 1

        # sample pixel-aligned features corresponding to each face centroid of the high-res mesh
        # the features will be concatenated with features from PointTransformer to preserve texture details
        face_feats_2d_ = []
        for img_feat in img_feats:
            face_feats_2d_.append(
                F.grid_sample(pack(img_feat), gaussian_means.unsqueeze(1), align_corners=False).squeeze(2))  # (B*N) x C x NF_new
        delta_face_feat_low = torch.cat(face_feats_2d_, dim=1).permute(0, 2, 1) * visibility_face
        delta_face_feat_low = \
            delta_face_feat_low.reshape(B, N, NF_new, -1).permute(0, 2, 1, 3).reshape(B, NF_new, -1)

        delta_face_feat = torch.cat((delta_face_feat_high, delta_face_feat_low), dim=-1)

        # update Gaussian params associated with high-res mesh
        delta_appearance = self.appearance_update_layer(delta_face_feat)
        delta_so3 = self.so3_update_layer(delta_face_feat)
        delta_scale = self.scale_update_layer(delta_face_feat)
        delta_offset = self.offset_update_layer(delta_face_feat)

        last_appearance = torch.zeros_like(update_list['appearance'])
        last_so3 = self.so3.unsqueeze(0).repeat(B, 1, 1)
        last_scale = self.scale.unsqueeze(0).repeat(B, 1, 1)
        last_offset = self.offset.unsqueeze(0).repeat(B, 1, 1)

        new_update_list = {
            'xyz': update_list['xyz'] + delta_xyz,
            'so3': last_so3 + delta_so3,
            'scale': last_scale + delta_scale,
            'offset': last_offset + delta_offset,
            'appearance': last_appearance + delta_appearance,
            'lbs_weights': update_list['lbs_weights'],
            'embed': delta_embed,
        }
        return new_update_list

    def render(self, renderer, update_list, K, E, global_Rs, global_Ts, bgcolor):
        B, N = global_Rs.shape[:2]

        xyzs_observation_ = apply_lbs(
            pack(update_list['xyz'][:, None].repeat(1, N, 1, 1)),
            pack(global_Rs), pack(global_Ts),
            pack(update_list['lbs_weights'][:, None].repeat(1, N, 1, 1)))  # (B*N) x NP x 3

        xyzs_observation_highres_ = (vector_gather(
            xyzs_observation_,
            self.interp_points.reshape(-1)[None].repeat(B * N, 1)
        ).reshape(B * N, self.NP_new, 3, 3) * self.interp_weights[..., None]).sum(-2)

        xyz_per_face_ = vector_gather(
            xyzs_observation_highres_,
            self.faces_highres.reshape(-1)[None].repeat(B * N, 1)
        ).reshape(B * N, self.NF_new, 3, -1)  # (B*N) x NF_new x 3 x 3
        centroid_ = xyz_per_face_.mean(-2) # (B*N) x NF_new x 3

        S = torch.diag_embed(update_list['scale'])
        R = unpack(so3_exp_map(pack(update_list['so3'])), [S.shape[0], S.shape[1]])
        cov_local = R @ S @ S.permute(0, 1, 3, 2) @ R.permute(0, 1, 3, 2)  # B x NF_new x 3 x 3
        cov_local_ = pack(cov_local[:, None].repeat(1, N, 1, 1, 1)) # (B*N) x NF_new x 3 x 3
        world_T_observation_ = get_transformation_from_triangle_steiner(xyz_per_face_)  # (B*N) x NF_new x 3 x 3
        cov_observation_ = world_T_observation_ @ cov_local_ @ world_T_observation_.permute(0, 1, 3, 2)

        offset = update_list['offset'][:, None].repeat(1, N, 1, 1) # B x N x NF_new x 3
        centroid_ += (world_T_observation_ @ pack(offset).unsqueeze(-1)).squeeze(-1)

        bg_feat = torch.zeros([3], dtype=S.dtype, device=S.device)
        bg_col = torch.cat([bg_feat, bg_feat.new_zeros([1])])

        appearance_ = pack(update_list['appearance'][:, None].repeat(1, N, 1, 1))
        K_ = pack(K)
        E_ = pack(E)

        rgbs = []
        masks = []
        for i in range(B * N):
            rgb, mask, render_info = renderer(
                centroid_[i:i+1],
                appearance_[i:i+1],
                K_[i:i+1], E_[i:i+1],
                bg_col=bg_col,
                skeleton_info={'cov': cov_observation_[i:i+1]})

            rgbs.append(rgb)
            masks.append(mask)

        rgbs = unpack(torch.cat(rgbs, dim=0), [B, N]) # B x N x H x W x 3
        masks = unpack(torch.cat(masks, dim=0), [B, N]) # B x N x H x W

        # apply background colors
        rgbs = rgbs * masks.unsqueeze(-1) + bgcolor[:, None, None, None, :] * (1 - masks).unsqueeze(-1)

        return rgbs, masks

    def forward(self,
                imgs, img_masks, img_feats,
                K, E,
                global_Rs, global_Ts,
                xyzs_init,
                renderer=None, img_encoder=None, bgcolor=[0, 0, 0], # additional information for feedback
                tb=None,
                **kwargs):
        B, N = imgs.shape[:2]
        update_list = {
            'xyz': xyzs_init, # B x NP x 3
            'so3': self.so3.unsqueeze(0).repeat(B, 1, 1), # B x NF_new x 3
            'scale': self.scale.unsqueeze(0).repeat(B, 1, 1), # B x NF_new x 3
            'offset': self.offset.unsqueeze(0).repeat(B, 1, 1), # B x NF_new x 3
            'appearance': self.appearance.unsqueeze(0).repeat(B, 1, 1), # B x NF_new x 3
            'lbs_weights': self.lbs_weights.unsqueeze(0).repeat(B, 1, 1), # B x NP x J
            'embed': self.vertex_embed.unsqueeze(0).repeat(B, 1, 1), # B x NP x C
        }
        update_list_all = []
        rendered_imgs_all = []
        rendered_masks_all = []
        for _ in range(self.cfg.n_iters):
            # get rendered images from last iteration and generate image features
            rendered_imgs, rendered_masks = \
                self.render(renderer, update_list, K, E, global_Rs, global_Ts, bgcolor=bgcolor)
            rendered_imgs_all.append(rendered_imgs)
            rendered_masks_all.append(rendered_masks)

            # do not backprop the gradients to save memory
            with torch.no_grad():
                rendered_img_feats = img_encoder(rendered_imgs.detach(), rendered_masks.detach())

            update_list = self.update(
                update_list,
                imgs, img_masks, img_feats,
                rendered_imgs, rendered_masks, rendered_img_feats,
                K, E,
                global_Rs, global_Ts,
                tb=tb)
            update_list_all.append(update_list)

        xyz = torch.stack([update_list['xyz'] for update_list in update_list_all])
        so3 = torch.stack([update_list['so3'] for update_list in update_list_all])
        scale = torch.stack([update_list['scale'] for update_list in update_list_all])
        offset = torch.stack([update_list['offset'] for update_list in update_list_all])
        appearance = torch.stack([update_list['appearance'] for update_list in update_list_all])
        lbs_weights = torch.stack([update_list['lbs_weights'] for update_list in update_list_all])

        outputs = {}
        rendered_imgs, rendered_masks = \
            self.render(renderer, update_list, K, E, global_Rs, global_Ts, bgcolor=bgcolor)
        rendered_imgs_all.append(rendered_imgs)
        rendered_masks_all.append(rendered_masks)
        rendered_imgs_all = torch.stack(rendered_imgs_all[1:], axis=0)  # remove the initial one
        rendered_masks_all = torch.stack(rendered_masks_all[1:], axis=0)
        # save it for supervision
        outputs['source_imgs'] = rendered_imgs_all
        outputs['source_masks'] = rendered_masks_all
        return xyz, so3, scale, appearance, lbs_weights, offset, outputs
