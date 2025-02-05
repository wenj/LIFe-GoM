import torch
import einops
import trimesh
from trimesh.remesh import faces_to_edges, grouping
import numpy as np


gl_transform = torch.tensor([[1., 0, 0, 0],
                             [0, -1., 0, 0],
                             [0, 0, -1., 0],
                             [0, 0, 0, 1.]], device='cuda')


def cam_T_world(xyzs_world, E):
	xyzs_world_ = torch.cat([xyzs_world, torch.ones_like(xyzs_world[:, :1])], dim=1)
	xyzs_cam_ = torch.bmm(E, xyzs_world_)
	xyzs_cam = xyzs_cam_[:, :3] / xyzs_cam_[:, 3:]
	return xyzs_cam


def img_T_cam(xyzs_cam, K):
	xys_ = torch.bmm(K, xyzs_cam)
	xys = xys_[:, :2] / xys_[:, 2:]
	return xys


def ndc_T_world(xyzs_world, K, E, H, W):
	xyzs_cam = cam_T_world(xyzs_world, E)
	xys_2d = img_T_cam(xyzs_cam, K)

	# normalize to NDC space. flip xy because the ndc coord definition
	# IMPORTANT: CHECK THE DEFINITION OF NDC
	# if H < W:
	# 	xs = -((xys_2d[:, 0, :] / H) * 2. - (W / H))
	# 	ys = -((xys_2d[:, 1, :] / H) * 2. - 1.)
	# else:
	# 	xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
	# 	ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))
	xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
	ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))
	zs = xyzs_cam[:, 2]
	xyzs_ndc = torch.stack([xs, ys, zs], dim=-1)
	return xyzs_ndc


def clip_T_world(xyzs_world, K, E, H, W):
    xyzs = torch.cat([xyzs_world, torch.ones_like(xyzs_world[..., 0:1, :])], dim=-2)
    K_expand = torch.zeros_like(E)
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_expand[:, 0, 0] = 2.0 * fx / W
    K_expand[:, 1, 1] = 2.0 * fy / H
    K_expand[:, 0, 2] = 1.0 - 2.0 * cx / W
    K_expand[:, 1, 2] = 1.0 - 2.0 * cy / H
    znear, zfar = 1e-3, 1e3
    K_expand[:, 2, 2] = -(zfar + znear) / (zfar - znear)
    K_expand[:, 3, 2] = -1.
    K_expand[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)

    # gl_transform = torch.tensor([[1., 0, 0, 0],
    #                              [0, -1., 0, 0],
    #                              [0, 0, -1., 0],
    #                              [0, 0, 0, 1.]], device=K.device)
    # gl_transform = torch.eye(4, dtype=K.dtype, device=K.device)
    # gl_transform[1, 1] = gl_transform[2, 2] = -1.
    return (K_expand @ gl_transform[None] @ E @ xyzs).permute(0, 2, 1)


def _subdivide(vertices,
               faces,
               face_index=None,
               vertex_attributes=None,
               return_index=False):
    """
    this function is adapted from trimesh
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    # turn the mask back into integer indexes
    nonzero = np.nonzero(face_mask)[0]
    # new faces start past the original faces
    # but we've removed all the faces in face_mask
    start = len(faces) - len(nonzero)
    # indexes are just offset from start
    stack = np.arange(
        start, start + len(f) * 4).reshape((-1, 4))
    # reformat into a slightly silly dict for some reason
    index_dict = {k: v for k, v in zip(nonzero, stack)}

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            if key == 'so3':
                attr_mid = np.zeros([unique.shape[0], 3], values.dtype)
            elif key == 'scale':
                edge_len = np.linalg.norm(values[edges[unique][:, 1]] - values[edges[unique][:, 0]], axis=-1)
                attr_mid = np.ones([unique.shape[0], 3], values.dtype) * edge_len[..., None]
            else:
                attr_mid = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes, index_dict

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide(vertices, faces, attributes, return_edges=False):
    mesh = trimesh.Trimesh(vertices, faces, vertex_attributes=attributes)
    new_vertices, new_faces, new_attributes, index_dict = _subdivide(mesh.vertices, mesh.faces, vertex_attributes=mesh.vertex_attributes)
    if return_edges:
        edges = trimesh.Trimesh(new_vertices, new_faces).edges
        return new_vertices, new_faces, new_attributes, edges, index_dict
    else:
        return new_vertices, new_faces, new_attributes, index_dict

