# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import igl
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from datautils.datautils import save_bin_strands
from utils.pcutils import load_pc
from utils.utils import translate2mat, homo_rot_mat

def read_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh', process=True)
    return mesh

def write_mesh(mesh_path, mesh):
    mesh.export(mesh_path)

def quad2trimesh(quad_faces, vertices):
    assert quad_faces.shape[1] == 4, "Mesh is not a quad mesh."
    num_quad_faces = quad_faces.shape[0]
    num_tri_faces = num_quad_faces * 2
    tri_faces = np.zeros((num_tri_faces, 3), dtype=np.uint32)
    tri_faces[::2] = quad_faces[:, [0, 1, 2]]
    tri_faces[1::2] = quad_faces[:, [0, 2, 3]]

    return trimesh.Trimesh(vertices=vertices, faces=tri_faces)

def vertices_pairwise_dis(vertices):
    inner_vertices = -2 * (vertices @ vertices.T)
    vertices_2 = np.sum(vertices**2, axis=1, keepdims=True)
    pairwise_dis = vertices_2 + inner_vertices + vertices_2.T
    return  pairwise_dis

def mesh_deformation(head_mesh, target_pc, scalp_faces_mask, smooth=True, smooth_iterations=10, thres_min_movement=10.0):
    target_kdtree = target_pc.kdtree
    v = head_mesh.vertices
    f = head_mesh.faces
    u = v.copy()
    num_vertices = v.shape[0]
    dis, idx = target_kdtree.query(v, 1)

    s = np.zeros(num_vertices)
    for i_face in range(head_mesh.faces.shape[0]):
        if scalp_faces_mask[i_face]:
            for i_v in range(3):
                v_idx = f[i_face, i_v]
                if dis[v_idx] <= thres_min_movement:
                    s[v_idx] = 1

    b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] > 0]]).T

    # Boundary conditions directly on deformed positions
    u_bc = np.zeros((b.shape[0], v.shape[1]))
    v_bc = np.zeros((b.shape[0], v.shape[1]))

    for bi in range(b.shape[0]):
        v_bc[bi] = v[b[bi]]
        
        offset = target_pc.vertices[idx[b[bi]]] - v[b[bi]]
        u_bc[bi] = v[b[bi]] + offset

    u_bc_anim = v_bc + (u_bc - v_bc)
    d_bc = u_bc_anim - v_bc
    d = igl.harmonic_weights(v, f, b.astype(f.dtype), d_bc, 1)
    u = v + d

    head_mesh.vertices = u
    if smooth:
        smoothe_head_mesh = copy.deepcopy(head_mesh)
        trimesh.smoothing.filter_mut_dif_laplacian(smoothe_head_mesh, iterations=smooth_iterations)
        # trimesh.smoothing.filter_laplacian(head_mesh, iterations=smooth_iterations)
        head_mesh.vertices[head_mesh.faces[scalp_faces_mask]] = smoothe_head_mesh.vertices[head_mesh.faces[scalp_faces_mask]]
    
    return head_mesh

def get_alignment_matrix(head_mesh, head_texture, target_roots_pc_path, target_face_base):
    uv_coords = head_mesh.visual.uv # num_vertices X 2
    head_tex_width, head_tex_height, _ = head_texture.shape

    head_mesh_pairwise_dis = vertices_pairwise_dis(head_mesh.vertices)
    head_mesh_eye = np.eye(head_mesh_pairwise_dis.shape[0])
    head_mesh_pairwise_dis = head_mesh_pairwise_dis + head_mesh_eye
    UV_bound_vertices = np.where(head_mesh_pairwise_dis < 1e-4)    # boundary vertices in UV

    # for each face determiner whether it is scalp
    num_faces = head_mesh.faces.shape[0]
    face_uv_coords = uv_coords[head_mesh.faces] * [head_tex_height, head_tex_width]
    face_uv_coords = np.around(face_uv_coords).astype(np.uint16)
    face_uv_coords = np.clip(face_uv_coords, [0, 1], [head_tex_width - 1, head_tex_height])
    face_uv_colors = head_texture[head_tex_height - face_uv_coords[:, :, 1], face_uv_coords[:, :, 0], :]
    face_avg_colors = np.sum(face_uv_colors, axis=1, keepdims=False)

    scalp_faces_mask = face_avg_colors[:, 0] > 255 * 0.3
    scalp_faces_idx = np.where(face_avg_colors[:, 0] > 255 * 0.3)[0]

    scalp_mesh = copy.deepcopy(head_mesh)
    scalp_mesh.update_faces(scalp_faces_mask)
    scalp_mesh.remove_unreferenced_vertices()
    scalp_sampled_points = scalp_mesh.sample(50000)

    target_points, target_normals = load_pc(target_roots_pc_path, load_color=False, load_normal=True)

    source_pc = trimesh.points.PointCloud(scalp_sampled_points)
    target_pc = trimesh.points.PointCloud(target_points)
    trans_mat = np.eye(4)

    # align bound sphere size
    scale_ratio = math.pow(target_pc.bounding_sphere.volume / source_pc.bounding_sphere.volume, 1./3.)
    scalp_sampled_points = scalp_sampled_points * scale_ratio
    trans_offset = [0., 0., 0.] - (source_pc.centroid * scale_ratio)
    scalp_sampled_points += trans_offset
    trans_mat[0:3] = trans_mat[0:3] * scale_ratio
    trans_mat = translate2mat(trans_offset) @ trans_mat

    # base rotate to original coord
    base_rot = R.from_euler('xyz', [[0., 0., 0.]])  # MannequinHeadB
    base_rot_mat = base_rot.as_matrix()[0]
    scalp_sampled_points = np.dot(base_rot_mat, scalp_sampled_points.T).T
    trans_mat = homo_rot_mat(base_rot_mat) @ trans_mat

    # change of basis
    # target_face_base = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1]])
    target_face_base_inv = target_face_base.T
    scalp_sampled_points = np.dot(target_face_base_inv, scalp_sampled_points.T).T
    trans_mat = homo_rot_mat(target_face_base_inv) @ trans_mat

    # move to same center with target
    scalp_sampled_points += target_pc.centroid
    trans_mat = translate2mat(target_pc.centroid) @ trans_mat

    # registration
    reg_mat, reg_points, cost = trimesh.registration.icp(scalp_sampled_points, target_points)
    trans_mat = reg_mat @ trans_mat

    return trans_mat

def process_head_model(head_mesh, head_texture, target_roots_pc_path, target_face_base, is_deformation=True):
    print('Utils::MeshUtils Start processing head model (registration & deformation)...')
    uv_coords = head_mesh.visual.uv # num_vertices X 2
    head_tex_width, head_tex_height, _ = head_texture.shape

    head_mesh_pairwise_dis = vertices_pairwise_dis(head_mesh.vertices)
    head_mesh_eye = np.eye(head_mesh_pairwise_dis.shape[0])
    head_mesh_pairwise_dis = head_mesh_pairwise_dis + head_mesh_eye
    UV_bound_vertices = np.where(head_mesh_pairwise_dis < 1e-4)    # boundary vertices in UV

    # for each face determiner whether it is scalp
    num_faces = head_mesh.faces.shape[0]
    face_uv_coords = uv_coords[head_mesh.faces] * [head_tex_height, head_tex_width]
    face_uv_coords = np.around(face_uv_coords).astype(np.uint16)
    face_uv_coords = np.clip(face_uv_coords, [0, 1], [head_tex_width - 1, head_tex_height])
    face_uv_colors = head_texture[head_tex_height - face_uv_coords[:, :, 1], face_uv_coords[:, :, 0], :]
    face_avg_colors = np.sum(face_uv_colors, axis=1, keepdims=False)

    scalp_faces_mask = face_avg_colors[:, 0] > 255 * 0.3
    scalp_faces_idx = np.where(face_avg_colors[:, 0] > 255 * 0.3)[0]

    scalp_mesh = copy.deepcopy(head_mesh)
    scalp_mesh.update_faces(scalp_faces_mask)
    scalp_mesh.remove_unreferenced_vertices()
    scalp_sampled_points = scalp_mesh.sample(50000)

    target_points = load_pc(target_roots_pc_path, load_color=False, load_normal=False)

    source_pc = trimesh.points.PointCloud(scalp_sampled_points)
    target_pc = trimesh.points.PointCloud(target_points)
    trans_mat = np.eye(4)

    # align bound sphere size
    scale_ratio = math.pow(target_pc.bounding_sphere.volume / source_pc.bounding_sphere.volume, 1./3.)
    scalp_sampled_points = scalp_sampled_points * scale_ratio
    trans_offset = [0., 0., 0.] - (source_pc.centroid * scale_ratio)
    scalp_sampled_points += trans_offset
    trans_mat = translate2mat(trans_offset) @ trans_mat

    # base rotate to original coord
    # base_rot = R.from_euler('yzx', [[211. / 180. * np.pi, -8. / 180. * np.pi, 0.]])  # Mugsy Head
    # base_rot = R.from_euler('xzy', [[180. / 180. * np.pi, 2. / 180. * np.pi, 3. / 180. * np.pi]])  # old MannequinHeadA
    base_rot = R.from_euler('xyz', [[0., 0., 0.]])  # MannequinHeadA and B
    base_rot_mat = base_rot.as_matrix()[0]
    scalp_sampled_points = np.dot(base_rot_mat, scalp_sampled_points.T).T
    trans_mat = homo_rot_mat(base_rot_mat) @ trans_mat

    # change of basis
    # target_face_base = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1]])
    target_face_base_inv = target_face_base.T
    scalp_sampled_points = np.dot(target_face_base_inv, scalp_sampled_points.T).T
    trans_mat = homo_rot_mat(target_face_base_inv) @ trans_mat

    # move to same center with target
    scalp_sampled_points += target_pc.centroid
    trans_mat = translate2mat(target_pc.centroid) @ trans_mat

    # registration
    reg_mat, reg_points, cost = trimesh.registration.icp(scalp_sampled_points, target_points)  # type: ignore (for avoid pyplace error report)
    trans_mat = reg_mat @ trans_mat

    # apply transformatioin to the head model
    head_mesh.apply_scale(scale_ratio)
    head_mesh.apply_transform(trans_mat)
    # head_mesh.export('temp/reg_head.ply')

    if is_deformation:
        # head_mesh = mesh_deformation(head_mesh, target_pc, scalp_faces_mask, smooth=False)
        # head_mesh = mesh_deformation(head_mesh, target_pc, scalp_faces_mask)
        head_mesh = mesh_deformation(head_mesh, target_pc, scalp_faces_mask, smooth_iterations=12, thres_min_movement=24)
        # head_mesh = mesh_deformation(head_mesh, target_pc, scalp_faces_mask, smooth_iterations=6, thres_min_movement=24)
        # head_mesh.export('temp/smoothed_deform_reg_head.ply')

    # sew vertices
    sewed_v = head_mesh.vertices.copy()
    for i_v in range(UV_bound_vertices[0].shape[0]):
        sewed_v[UV_bound_vertices[0][i_v]] = (head_mesh.vertices[UV_bound_vertices[0][i_v]] + head_mesh.vertices[UV_bound_vertices[1][i_v]]) / 2.
        sewed_v[UV_bound_vertices[1][i_v]] = (head_mesh.vertices[UV_bound_vertices[0][i_v]] + head_mesh.vertices[UV_bound_vertices[1][i_v]]) / 2.
    head_mesh.vertices = sewed_v

    # compute transed & registered & deformed scalp mesh again
    scalp_mesh = copy.deepcopy(head_mesh)
    scalp_mesh.update_faces(scalp_faces_mask)
    scalp_mesh.remove_unreferenced_vertices()

    print('Utils::MeshUtils End processing.')
    return head_mesh, scalp_mesh, scalp_faces_idx

def seg_head_model(head_mesh, head_texture):
    uv_coords = head_mesh.visual.uv # num_vertices X 2
    head_tex_width, head_tex_height, _ = head_texture.shape

    head_mesh_pairwise_dis = vertices_pairwise_dis(head_mesh.vertices)
    head_mesh_eye = np.eye(head_mesh_pairwise_dis.shape[0])
    head_mesh_pairwise_dis = head_mesh_pairwise_dis + head_mesh_eye
    UV_bound_vertices = np.where(head_mesh_pairwise_dis < 1e-4)    # boundary vertices in UV

    # for each face determiner whether it is scalp
    face_uv_coords = uv_coords[head_mesh.faces] * [head_tex_height, head_tex_width]
    face_uv_coords = np.around(face_uv_coords).astype(np.uint16)
    face_uv_coords = np.clip(face_uv_coords, [0, 1], [head_tex_width - 1, head_tex_height])
    face_uv_colors = head_texture[head_tex_height - face_uv_coords[:, :, 1], face_uv_coords[:, :, 0], :]
    face_avg_colors = np.sum(face_uv_colors, axis=1, keepdims=False)

    scalp_faces_mask = face_avg_colors[:, 0] > 255 * 0.3
    scalp_faces_idx = np.where(face_avg_colors[:, 0] > 255 * 0.3)[0]

    scalp_mesh = copy.deepcopy(head_mesh)
    scalp_mesh.update_faces(scalp_faces_mask)
    scalp_mesh.remove_unreferenced_vertices()

    return head_mesh, scalp_mesh, scalp_faces_idx

import torch
from typing import Union, Tuple
from trimesh import Trimesh
# from trimesh.proximity import closest_point # Too slow
from trimesh.triangles import points_to_barycentric

def closest_point_barycentrics(v, vi, points, filtering=False, filter_dis_thres=2.):
    """Given a 3D mesh and a set of query points, return closest point barycentrics
    Args:
        v: np.array (float)
        [N, 3] mesh vertices
        vi: np.array (int)
        [N, 3] mesh triangle indices
        points: np.array (float)
        [M, 3] query points
    Returns:
        Tuple[approx, barys, interp_idxs, face_idxs]
            approx:       [M, 3] approximated (closest) points on the mesh
            barys:        [M, 3] barycentric weights that produce "approx"
            interp_idxs:  [M, 3] vertex indices for barycentric interpolation
            face_idxs:    [M] face indices for barycentric interpolation. interp_idxs = vi[face_idxs]
    """
    mesh = Trimesh(vertices=v, faces=vi)
    # p, distances, face_idxs = closest_point(mesh, points)   # Slow, Change to IGL
    sqr_distances, face_idxs, p = igl.point_mesh_squared_distance(points, mesh.vertices, mesh.faces)    # type: ignore for avoiding pylance error

    if filtering:
        valid_q_idx = np.where(np.sqrt(sqr_distances) < filter_dis_thres)[0]
        p = p[valid_q_idx]
        face_idxs = face_idxs[valid_q_idx]
    else:
        valid_q_idx = np.arange(p.shape[0])

    barys = points_to_barycentric(mesh.triangles[face_idxs], p)
    b0, b1, b2 = np.split(barys, 3, axis=1)

    interp_idxs = vi[face_idxs]
    v0 = v[interp_idxs[:, 0]]
    v1 = v[interp_idxs[:, 1]]
    v2 = v[interp_idxs[:, 2]]
    approx = b0 * v0 + b1 * v1 + b2 * v2
    return approx, barys, interp_idxs, face_idxs, valid_q_idx

def make_closest_uv_barys(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for the closest point on a UV triangle.
    Args:
        vt: torch.Tensor
        Texture coordinates. Shape = [n_texcoords, 2]
        vti: torch.Tensor
        Face texture coordinate indices. Shape = [n_faces, 3]
        uv_shape: Tuple[int, int] or int
        Shape of the texture map. (HxW)
        flip_uv: bool
        Whether or not to flip UV coordinates along the V axis (OpenGL -> numpy/pytorch convention).
    Returns:
        torch.Tensor: index_img: Face index image, shape [uv_shape[0], uv_shape[1]]
        torch.Tensor: Barycentric coordinate map, shape [uv_shape[0], uv_shape[1], 3]Â
    """

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    # Texel to UV mapping (as per OpenGL linear filtering)
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf
    # Sect. 8.14, page 261
    # uv=(0.5,0.5)/w is at the center of texel [0,0]
    # uv=(w-0.5, w-0.5)/w is the center of texel [w-1,w-1]
    # texel = floor(u*w - 0.5)
    # u = (texel+0.5)/w
    uv_grid = torch.meshgrid(
        torch.linspace(0.5, uv_shape[0] - 1 + 0.5, uv_shape[0]) / uv_shape[0],
        torch.linspace(0.5, uv_shape[1] - 1 + 0.5, uv_shape[1]) / uv_shape[1], indexing='ij')  # HxW, v,u
    uv_grid = torch.stack(uv_grid[::-1], dim=2)  # HxW, u, v

    uv = uv_grid.reshape(-1, 2).data.to("cpu").numpy()
    vth = np.hstack((vt, vt[:, 0:1] * 0 + 1))
    uvh = np.hstack((uv, uv[:, 0:1] * 0 + 1))
    approx, barys, interp_idxs, face_idxs, _ = closest_point_barycentrics(vth, vti, uvh)
    index_img = torch.from_numpy(face_idxs.reshape(uv_shape[0], uv_shape[1])).long()
    bary_img = torch.from_numpy(barys.reshape(uv_shape[0], uv_shape[1], 3)).float()
    return index_img, bary_img

def compute_tbn_uv(tri_xyz, tri_uv, eps=1e-5):
    """Compute tangents, bitangents, normals.
    Args:
        tri_xyz: [B,N,3,3] vertex coordinates
        tri_uv: [B,N,3,2] texture coordinates
    Returns:
        tangents, bitangents, normals
    """
    v01 = tri_xyz[:, :, 1] - tri_xyz[:, :, 0]
    v02 = tri_xyz[:, :, 2] - tri_xyz[:, :, 0]

    normals = torch.cross(v01, v02, dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp(min=eps)

    vt01 = tri_uv[:, :, 1] - tri_uv[:, :, 0]
    vt02 = tri_uv[:, :, 2] - tri_uv[:, :, 0]

    f = 1.0 / (vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0])

    tangents = f[..., np.newaxis] * (
        v01 * vt02[..., 1][..., np.newaxis] - v02 * vt01[..., 1][..., np.newaxis])
    tangents = tangents / torch.norm(tangents, dim=-1, keepdim=True).clamp(min=eps)

    bitangents = torch.cross(normals, tangents, dim=-1)
    bitangents = bitangents / torch.norm(bitangents, dim=-1, keepdim=True).clamp(min=eps)
    return tangents, bitangents, normals

def strands_world2tbn(strands, head_mesh, scalp_mesh, scalp_faces_idx):
    # print('Utils::MeshUtils Convert strands to TBN space...')
    num_strands = strands.shape[0]
    # get all roots points
    roots_pc = []
    for i_strand in range(num_strands):
        roots_pc.append(strands[i_strand][0, 0:3])
    roots_pc = np.array(roots_pc)

    approx, barys, interp_idxs, faces_idxs, valid_q_idxs = closest_point_barycentrics(scalp_mesh.vertices, scalp_mesh.faces, roots_pc, filtering=True, filter_dis_thres=6.4)    # 3.6 -> 6.4, 7.2
    valid_strands = strands[valid_q_idxs]
    # invalid_q_idxs = list(set(np.arange(roots_pc.shape[0])) - set(valid_q_idxs))
    # invalid_strands = strands[invalid_q_idxs]
    # save_bin_strands('temp/valid_strands.bin', valid_strands)
    # save_bin_strands('temp/invalid_strands.bin', invalid_strands)

    num_valid_strands = valid_strands.shape[0]

    triangled_vertices = torch.tensor(head_mesh.vertices[head_mesh.faces, :])[None, :]
    triangled_vertices_uv = torch.tensor(head_mesh.visual.uv[head_mesh.faces, :])[None, :]
    tangents, bitangents, normals = compute_tbn_uv(triangled_vertices, triangled_vertices_uv)   # get tbn for each face
    scalp_tangents = tangents[0][scalp_faces_idx].detach().cpu().numpy()
    scalp_bitangents = bitangents[0][scalp_faces_idx].detach().cpu().numpy()
    scalp_normals = normals[0][scalp_faces_idx].detach().cpu().numpy()

    tbn_strands = []
    for i_strand in range(num_valid_strands):
        tangent = scalp_tangents[faces_idxs[i_strand]]
        bitangent = scalp_bitangents[faces_idxs[i_strand]]
        normal = scalp_normals[faces_idxs[i_strand]]

        tbn_basis_T = np.array([tangent, bitangent, normal])
        tbn_strand = (tbn_basis_T @ valid_strands[i_strand][:, 0:3].T).T
        tbn_strand = tbn_strand - tbn_strand[0]
        tbn_strands.append(tbn_strand)
    
    # print('Utils::MeshUtils End converting, number of original strands: %d, number of valid strands: %d'%(num_strands, num_valid_strands))
    return tbn_strands, barys, interp_idxs, faces_idxs, valid_q_idxs, tangents, bitangents, normals

def strands_align_normal(strands, head_mesh):
    num_strands = len(strands)
    # get all roots points
    roots_pc = []
    for i_strand in range(num_strands):
        roots_pc.append(strands[i_strand][0])
    roots_pc = np.array(roots_pc)[:, 0:3]

    sqr_distances, face_idxs, p = igl.point_mesh_squared_distance(roots_pc, head_mesh.vertices, head_mesh.faces)
    closest_faces = head_mesh.faces[face_idxs]
    closest_triangles = torch.tensor(head_mesh.vertices[closest_faces, :])[None, :]
    v01 = closest_triangles[:, :, 1] - closest_triangles[:, :, 0]
    v02 = closest_triangles[:, :, 2] - closest_triangles[:, :, 0]
    normals = torch.cross(v01, v02, dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-5)
    z_aixs = torch.zeros_like(normals)
    z_aixs[:, :, 2] = 1
    t_axises = torch.cross(normals, z_aixs)
    t_axises = t_axises / torch.norm(t_axises, dim=-1, keepdim=True).clamp(min=1e-5)
    b_axises = torch.cross(normals, t_axises)
    b_axises = b_axises / torch.norm(b_axises, dim=-1, keepdim=True).clamp(min=1e-5)

    tangents = t_axises[0].detach().cpu().numpy()
    bitangents = b_axises[0].detach().cpu().numpy()
    normals = normals[0].detach().cpu().numpy()

    aligned_strands = []
    valid_rot_mats = []
    valid_roots_pts = []
    for i_strand in range(num_strands):
        tangent = tangents[i_strand]
        bitangent = bitangents[i_strand]
        normal = normals[i_strand]
        strand = np.array(strands[i_strand])
        root_pts = strand[0]

        strand = strand - root_pts
        tbn_basis_T = np.array([tangent, bitangent, normal])
        aligned_strand = (tbn_basis_T @ strand.T).T

        if np.sum(aligned_strand ** 2) < 1e-7 or np.isnan(np.sum(aligned_strand)):  # delete some noise data for avoiding nan
            continue

        aligned_strands.append(aligned_strand)
        valid_rot_mats.append(tbn_basis_T)
        valid_roots_pts.append(root_pts)

    return aligned_strands, valid_rot_mats, valid_roots_pts