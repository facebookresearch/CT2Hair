# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import pathlib
import struct
import numpy as np

from utils.pcutils import pc_voxelization, save_pc

def load_raw(path, raw_shape=[2048, 2048, 2048], offset=0, drop_masks=[], crop=[[0, -1], [0, -1], [0, -1]], is_downsample=True, downsample_ratio=2):
    start_time = time.time()

    if pathlib.Path(path).suffix == '.npy':
        raw_data = np.load(path)
    else:
        raw_data = np.fromfile(path, dtype=np.ushort)

    raw_data = raw_data[offset:]
    raw_data = raw_data.reshape(raw_shape)

    for i_del in range(len(drop_masks)):
        drop_mask = drop_masks[i_del]
        raw_data[drop_mask[0][0]:drop_mask[0][1], drop_mask[1][0]:drop_mask[1][1], drop_mask[2][0]:drop_mask[2][1]] = 0

    raw_data = raw_data[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]

    current_shape = np.array(raw_data.shape)
    if is_downsample:
        downsample_shape = current_shape // downsample_ratio
        x_even = (np.arange(downsample_shape[0]) * downsample_ratio).astype(np.int16)
        y_even = (np.arange(downsample_shape[1]) * downsample_ratio).astype(np.int16)
        z_even = (np.arange(downsample_shape[2]) * downsample_ratio).astype(np.int16)
        raw_data = raw_data[x_even]
        raw_data = raw_data[:, y_even]
        raw_data = raw_data[:, :, z_even]
        path_ds = path.replace(pathlib.Path(path).suffix, '_ds.npy')
        np.save(path_ds, raw_data)

    print('Finish load the volume, used %fs. Original and final shapes are: '%(time.time() - start_time), current_shape, np.array(raw_data.shape))
    return raw_data

def crop_hair(raw_data, hair_density_range):
    start_time = time.time()
    hair_mask = (raw_data > hair_density_range[0]) & (raw_data < hair_density_range[1])
    cropped_hair = raw_data * hair_mask
    print('Finish crop hair, used  %fs.'%(time.time() - start_time))
    return cropped_hair, hair_mask

def crop_scalp(raw_data, scalp_range):
    start_time = time.time()
    scalp_mask = (raw_data > scalp_range[0]) & (raw_data < scalp_range[1])
    cropped_hair = raw_data * scalp_mask
    print('Finish crop scalp, used  %fs.'%(time.time() - start_time))
    return cropped_hair, scalp_mask

def get_hair_mask(raw_data, hair_density_range):
    hair_mask = (raw_data > hair_density_range[0]) & (raw_data < hair_density_range[1])
    return hair_mask

def expand_vidx(vidx_x, vidx_y, vidx_z, scale_rate=3):
    size = int(2 ** scale_rate)
    o_x, o_y, o_z = np.meshgrid(np.linspace(0, size - 1, size),
                                np.linspace(0, size - 1, size),
                                np.linspace(0, size - 1, size))

    vidx_x = vidx_x[:, None].repeat(size ** 3, axis=-1).reshape(-1, size, size, size)
    vidx_y = vidx_y[:, None].repeat(size ** 3, axis=-1).reshape(-1, size, size, size)
    vidx_z = vidx_z[:, None].repeat(size ** 3, axis=-1).reshape(-1, size, size, size)

    vidx_x = vidx_x * size + o_x[None, ...]
    vidx_y = vidx_y * size + o_y[None, ...]
    vidx_z = vidx_z * size + o_z[None, ...]

    vidx_x = vidx_x.reshape(-1).astype(np.uint16)
    vidx_y = vidx_y.reshape(-1).astype(np.uint16)
    vidx_z = vidx_z.reshape(-1).astype(np.uint16)

    return vidx_x, vidx_y, vidx_z

def del_wig_net(hair_data, scalp_mesh, voxel_size, scale_rate=3):
    start_time = time.time()
    print('Start delete wig net...')

    hair_voxel_shape = hair_data.shape
    scalp_voxel_shape = np.array(hair_voxel_shape) / (2 ** scale_rate)

    scalp_points = scalp_mesh.sample(100000) * (1 / voxel_size[None, :]) / (2 ** scale_rate)
    vidx_x, vidx_y, vidx_z = pc_voxelization(scalp_points, scalp_voxel_shape.astype(np.uint16))
    vidx_x, vidx_y, vidx_z = expand_vidx(vidx_x, vidx_y, vidx_z, scale_rate)

    hair_data[vidx_x, vidx_y, vidx_z] = 0

    print('Delete wig net finished, used %fs.'%(time.time() - start_time))
    return hair_data

def save_raw(data, path):
    data.astype('int16').tofile(path)

def get_slide(data, id=0, axis='x', range=1):
    # x switch z
    if axis == 'z':
        slide = data[(id - range + 1):(id + range), :, :]
        slide = np.sum(slide, axis=0, keepdims=False)
        return slide
    elif axis == 'y':
        slide = data[:, (id - range + 1):(id + range), :]
        slide = np.sum(slide, axis=1, keepdims=False)
        return slide
    elif axis == 'x':
        slide = data[:, :, (id - range + 1):(id + range)]
        slide = np.sum(slide, axis=2, keepdims=False)
        return slide

def load_bin_strands(bin_path):
    file = open(bin_path, 'rb')
    num_strands = struct.unpack('i', file.read(4))[0]

    strands = []
    max_strds_pts = 0

    for i in range(num_strands):
        num_verts = struct.unpack('i', file.read(4))[0]
        strand = np.zeros((num_verts, 6), dtype=np.float32)
        for j in range(num_verts):
            x = struct.unpack('f', file.read(4))[0]
            y = struct.unpack('f', file.read(4))[0]
            z = struct.unpack('f', file.read(4))[0]
            nx = struct.unpack('f', file.read(4))[0]
            ny = struct.unpack('f', file.read(4))[0]
            nz = struct.unpack('f', file.read(4))[0]
            label = struct.unpack('f', file.read(4))[0]
            strand[j][0] = x
            strand[j][1] = y
            strand[j][2] = z
            strand[j][3] = nx
            strand[j][4] = ny
            strand[j][5] = nz

        if np.isnan(np.sum(strand)):    # FIXME why did I save some nan data
            continue
        
        if num_verts < 5:
            continue

        if max_strds_pts < num_verts:
            max_strds_pts = num_verts

        strands.append(strand)

    strands = np.array(strands, dtype=object)
    return strands

def load_usc_data_strands(data_path):
    file = open(data_path, 'rb')
    num_strands = struct.unpack('i', file.read(4))[0]

    strands = []
    for i in range(num_strands):
        num_verts = struct.unpack('i', file.read(4))[0]

        strand = np.zeros((num_verts, 3), dtype=np.float32)
        for j in range(num_verts):
            x = struct.unpack('f', file.read(4))[0]
            y = struct.unpack('f', file.read(4))[0]
            z = struct.unpack('f', file.read(4))[0]
            strand[j][0] = x
            strand[j][1] = y
            strand[j][2] = z

        if num_verts <= 1:
            continue
        if np.isnan(np.sum(strand)):
            continue
        strands.append(strand)

    strands = np.array(strands, dtype=object)
    return strands

def save_bin_strands(filepath, strands, tangents=None):
    num_strands = strands.shape[0]
    file = open(filepath, 'wb')

    file.write(struct.pack('i', num_strands))
    for i_strand in range(num_strands):
        num_points = int(strands[i_strand].shape[0])
        file.write(struct.pack('i', num_points))
        for j_point in range(num_points):
            file.write(struct.pack('f', strands[i_strand][j_point, 0]))
            file.write(struct.pack('f', strands[i_strand][j_point, 1]))
            file.write(struct.pack('f', strands[i_strand][j_point, 2]))
            if tangents is None:
                file.write(struct.pack('f', 0.0))
                file.write(struct.pack('f', 0.0))
                file.write(struct.pack('f', 0.0))
            else:
                file.write(struct.pack('f', tangents[i_strand][j_point, 0]))
                file.write(struct.pack('f', tangents[i_strand][j_point, 1]))
                file.write(struct.pack('f', tangents[i_strand][j_point, 2]))
            file.write(struct.pack('f', 0.0))

# save strands with colors (PCA features mapping)
def save_color_strands(filepath, strands, colors=None):
    num_strands = strands.shape[0]
    file = open(filepath, 'wb')

    file.write(struct.pack('i', num_strands))
    for i_strand in range(num_strands):
        num_points = int(strands[i_strand].shape[0])
        file.write(struct.pack('i', num_points))
        for j_point in range(num_points):
            file.write(struct.pack('f', strands[i_strand][j_point, 0]))
            file.write(struct.pack('f', strands[i_strand][j_point, 1]))
            file.write(struct.pack('f', strands[i_strand][j_point, 2]))
            if colors is not None:
                file.write(struct.pack('f', colors[i_strand, 0]))
                file.write(struct.pack('f', colors[i_strand, 1]))
                file.write(struct.pack('f', colors[i_strand, 2]))
            else:
                assert strands[i_strand].shape[1] == 6, 'DataUtils::DataUtils No color of strands.'
                file.write(struct.pack('f', strands[i_strand][j_point, 3]))
                file.write(struct.pack('f', strands[i_strand][j_point, 4]))
                file.write(struct.pack('f', strands[i_strand][j_point, 5]))
            file.write(struct.pack('f', 0.0))

def save_bin_strandspc(filepath, pc, sep, tangents=None):
    num_strands = len(sep)
    file = open(filepath, 'wb')

    file.write(struct.pack('i', num_strands))
    point_count = 0
    for i_strand in range(num_strands):
        num_points = int(sep[i_strand])
        file.write(struct.pack('i', num_points))
        for j_point in range(num_points):
            file.write(struct.pack('f', pc[point_count, 0]))
            file.write(struct.pack('f', pc[point_count, 1]))
            file.write(struct.pack('f', pc[point_count, 2]))
            if tangents is None:
                file.write(struct.pack('f', 0.0))
                file.write(struct.pack('f', 0.0))
                file.write(struct.pack('f', 0.0))
            else:
                file.write(struct.pack('f', tangents[point_count, 0]))
                file.write(struct.pack('f', tangents[point_count, 1]))
                file.write(struct.pack('f', tangents[point_count, 2]))
            file.write(struct.pack('f', 0.0))
            point_count += 1

def merge_save_bin_strands(filepath, strands_list, tangents_list=None):
    num_all_strands = 0
    num_groups = len(strands_list)
    for i_group in range(num_groups):
        num_all_strands += strands_list[i_group].shape[0]
    
    file = open(filepath, 'wb')
    file.write(struct.pack('i', num_all_strands))
    for i_group in range(num_groups):
        strands = strands_list[i_group]
        num_strands = strands.shape[0]

        if tangents_list is None:
            tangents = strands
        else:
            tangents = tangents_list[i_group]

        for i_strand in range(num_strands):
            num_points = int(strands[i_strand].shape[0])
            file.write(struct.pack('i', num_points))

            for j_point in range(num_points):
                file.write(struct.pack('f', strands[i_strand][j_point, 0]))
                file.write(struct.pack('f', strands[i_strand][j_point, 1]))
                file.write(struct.pack('f', strands[i_strand][j_point, 2]))
                file.write(struct.pack('f', tangents[i_strand][j_point, 0]))
                file.write(struct.pack('f', tangents[i_strand][j_point, 1]))
                file.write(struct.pack('f', tangents[i_strand][j_point, 2]))
                file.write(struct.pack('f', 0.0))

def save_usc_strands(filepath, strands):
    num_strands = strands.shape[0]
    file = open(filepath, 'wb')

    file.write(struct.pack('i', num_strands))
    for i_strand in range(num_strands):
        num_points = int(strands[i_strand].shape[0])
        file.write(struct.pack('i', num_points))
        for j_point in range(num_points):
            file.write(struct.pack('f', strands[i_strand][j_point, 0]))
            file.write(struct.pack('f', strands[i_strand][j_point, 1]))
            file.write(struct.pack('f', strands[i_strand][j_point, 2]))