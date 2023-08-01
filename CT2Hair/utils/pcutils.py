# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import open3d as o3d
from copy import deepcopy
from matplotlib import cm

def volume2pc(voxels, threshold=1e-1, scale_ratio=np.array([0.125, 0.125, 0.125]), get_colors=True):
    start_time = time.time()
    x, y, z = np.where(voxels > threshold)
    points = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=1).astype(np.float32)
    points = points * scale_ratio
    values = voxels[x, y, z]

    if get_colors:
        BuGn_color_map = cm.get_cmap('BuGn', 256)
        colors = np.array(BuGn_color_map(values))[:, 0:3]
        print('Finish volume to pc stage, used %fs'%(time.time() - start_time))
        return points, colors
    else:
        print('Finish volume to pc stage, used %fs'%(time.time() - start_time))
        return points, values

def pc2volume(points, colors=None, normals=None, num_angles=12):
    min_bound = np.min(points, axis=0).astype(np.int16)
    max_bound = np.max(points, axis=0).astype(np.int16)

    voxel_size = max_bound - min_bound + 1
    voxel_size = np.append(voxel_size, [4])
    voxels = np.zeros(voxel_size)

    points = points.astype(np.int16)
    points = points - min_bound

    if colors is not None:
        voxels[points[:, 0], points[:, 1], points[:, 2], 0] = colors[:, 0] # confidence
        voxels[points[:, 0], points[:, 1], points[:, 2], 1] = colors[:, 1] * num_angles # thete
        voxels[points[:, 0], points[:, 1], points[:, 2], 2] = colors[:, 2] * num_angles # phi
        voxels[points[:, 0], points[:, 1], points[:, 2], 3] = np.arange(points.shape[0]) # point_index
    elif normals is not None:
        voxels[points[:, 0], points[:, 1], points[:, 2], 0:3] = normals # confidence
        voxels[points[:, 0], points[:, 1], points[:, 2], 3] = np.arange(points.shape[0]) # point_index

    return voxels, min_bound

def strands2pc(strands, step_size=None, rand_color=True):
    num_strands = strands.shape[0]
    if step_size == None:
        strands_points = []
        strands_normals = []
        strands_colors = []
        strands_tangents = []

        strands_sep = [] # number of points for each strand
        for i_strand in range(num_strands):
            num_points = strands[i_strand].shape[0]
            points = strands[i_strand][:, :3]
            normals = strands[i_strand][:, 3:]

            tangents = points[1:] - points[:-1]
            tangents = tangents / np.linalg.norm(tangents, axis=-1, keepdims=True)
            tangents = np.concatenate((tangents, tangents[-1:]), axis=0)

            points = points.tolist()
            normals = normals.tolist()
            tangents = tangents.tolist()
            strands_points.extend(points)
            strands_normals.extend(normals)
            strands_tangents.extend(tangents)

            if rand_color:
                strand_color = np.random.rand(1, 3)
                strand_colors = np.repeat(strand_color, num_points, axis=0)
                strand_colors = strand_colors.tolist()
                strands_colors.extend(strand_colors)

            strands_sep.append(num_points)
        
        strands_points = np.array(strands_points)
        strands_tangents = np.array(strands_tangents)
        if rand_color:
            strands_colors = np.array(strands_colors)
            return strands_points, strands_colors, strands_sep
        else:
            return strands_points, strands_tangents, strands_sep
    else:
        max_step_lenght = 0
        strands_steps_pos_norm = []
        strands_steps_colors = []
        for i_strand in range(num_strands):
            num_steps = strands[i_strand].shape[0] // step_size
            num_points = num_steps * step_size
            strand = np.reshape(strands[i_strand][:num_points], (num_steps, step_size, strands[i_strand].shape[-1]))
            strands_steps_pos_norm.append(strand)

            if rand_color:
                strand_color = np.random.rand(1, 3)
                strand_colors = np.repeat(strand_color, num_points, axis=0)
                strand_colors = np.reshape(strand_colors, (num_steps, step_size, 3))
                strands_steps_colors.append(strand_colors)
            
            if num_steps > max_step_lenght:
                max_step_lenght = num_steps

        steps_points = []
        steps_normals = []
        steps_colors = []
        for i_step in range(max_step_lenght):
            step_points = []
            step_normals = []
            step_colors = []
            for j_strand in range(num_strands):
                step_lenght = strands_steps_pos_norm[j_strand].shape[0]
                if (step_lenght <= i_step):
                    continue
                step_points.append(strands_steps_pos_norm[j_strand][i_step, :, :3])
                step_normals.append(strands_steps_pos_norm[j_strand][i_step, :, 3:])
                if rand_color:
                    step_colors.append(strands_steps_colors[j_strand][i_step])
            
            steps_points.append(np.array(step_points).reshape(-1, 3))
            steps_normals.append(np.array(step_normals).reshape(-1, 3))
            if rand_color:
                steps_colors.append(np.array(step_colors).reshape(-1, 3))

        if rand_color:
            return max_step_lenght, steps_points, steps_colors
        else:
            return max_step_lenght, steps_points, None

def read_pc(pc_path):
    point_cloud = o3d.io.read_point_cloud(pc_path)
    return point_cloud

def load_pc(pc_path, load_color=True, load_normal=False):
    point_cloud = o3d.io.read_point_cloud(pc_path)
    
    points = np.asarray(point_cloud.points)
    if load_color:
        assert point_cloud.has_colors(), "Loaded point cloud has no colors"
        colors = np.asarray(point_cloud.colors)
        return points, colors
    elif load_normal:
        assert point_cloud.has_normals(), "Loaded point cloud has no normals"
        normals = np.asarray(point_cloud.normals)
        return points, normals
    else:
        return points

def save_pc_float64(pc_path, points, colors=None, normals=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        assert points.shape[0] == colors.shape[0], "points and colors should have same numbers"
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    
    return o3d.io.write_point_cloud(pc_path, point_cloud)

def save_pc(pc_path, points, colors=None, normals=None):
    pc_device = o3d.core.Device("CPU:0")
    pc_type = o3d.core.float32
    
    point_cloud = o3d.t.geometry.PointCloud(pc_device)
    point_cloud.point["positions"] = o3d.core.Tensor(points.astype(np.float32), pc_type, pc_device)
    if normals is not None:
        point_cloud.point["normals"] = o3d.core.Tensor(normals.astype(np.float32), pc_type, pc_device)
    if colors is not None:
        assert points.shape[0] == colors.shape[0], "points and colors should have same numbers"
        colors = (colors * 255).astype(np.int8) # need to do this for open3d version 0.15.1, after this I can vis it via meshlab
        point_cloud.point["colors"] = o3d.core.Tensor(colors, o3d.core.uint8, pc_device)
        
    return o3d.t.io.write_point_cloud(pc_path, point_cloud, compressed=True, print_progress=True)

def get_bbox(points):
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    bbox = np.array([[x_min, x_max],
                     [y_min, y_max],
                     [z_min, z_max]])

    center = ((bbox[:, 1] + bbox[:, 0]) / 2.).T
    return bbox, center

def pc_voxelization(points, shape):
    segments = []
    steps = []
    for i in range(3):
        s, step = np.linspace(0, shape[i] - 1, num=shape[i], retstep=True)
        segments.append(s)
        steps.append(step)

    vidx_x = np.clip(np.searchsorted(segments[0], points[:, 0]), 0, shape[0] - 1)
    vidx_y = np.clip(np.searchsorted(segments[1], points[:, 1]), 0, shape[1] - 1)
    vidx_z = np.clip(np.searchsorted(segments[2], points[:, 2]), 0, shape[2] - 1)

    vidx = np.concatenate((vidx_x[:, None], vidx_y[:, None], vidx_z[:, None]), axis=-1)
    vidx = np.unique(vidx, axis=0)
    return vidx[:, 0], vidx[:, 1], vidx[:, 2]

def patch_filter_major(points, voxels, weights, kernel_size=5):
    assert voxels.ndim == 3, "Only works for 1-dim voxel"
    assert voxels.dtype == np.int16, "Only works for int voxel"
    num_points = points.shape[0]
    offset = kernel_size // 2

    padded_voxels = np.pad(voxels, ((offset, offset), (offset, offset), (offset, offset)), mode='reflect')
    padded_weights = np.pad(weights, ((offset, offset), (offset, offset), (offset, offset)), mode='reflect')
    filtered_voxels = deepcopy(voxels)

    for i_point in range(num_points):
        grid_idx = points[i_point]
        # selected_region_start_pos = grid_idx - offset
        selected_region = padded_voxels[grid_idx[0] : grid_idx[0] + kernel_size,
                                        grid_idx[1] : grid_idx[1] + kernel_size,
                                        grid_idx[2] : grid_idx[2] + kernel_size,]
        selected_weights = padded_weights[grid_idx[0] : grid_idx[0] + kernel_size,
                                          grid_idx[1] : grid_idx[1] + kernel_size,
                                          grid_idx[2] : grid_idx[2] + kernel_size,]
        major_value = np.bincount(selected_region.reshape(-1), selected_weights.reshape(-1)).argmax()
        filtered_voxels[grid_idx[0], grid_idx[1], grid_idx[2]] = major_value
    return filtered_voxels