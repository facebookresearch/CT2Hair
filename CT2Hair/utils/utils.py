# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import math
import torch
import torch.nn as nn
import numpy as np
from matplotlib import cm

def polar2vector(theta, phi, step_length=1, start_vector=np.array([1, 0, 0])):
    sin_a, cos_a = math.sin(0), math.cos(0)
    sin_b, cos_b = math.sin(phi), math.cos(phi)
    sin_g, cos_g = math.sin(theta), math.cos(theta)

    R_x = np.array([[1,     0,      0],
                    [0, cos_a, -sin_a],
                    [0, sin_a,  cos_a]])
    R_y = np.array([[ cos_b, 0, sin_b],
                    [     0, 1,     0],
                    [-sin_b, 0, cos_b]])
    R_z = np.array([[cos_g, -sin_g, 0],
                    [sin_g,  cos_g, 0],
                    [    0,      0, 1]],)

    R =  R_z @ R_y @ R_x

    vector = start_vector * step_length
    vector = vector.T

    vector = R @ vector
    return vector

def polar2vector_torch(theta, phi, step_length=1, start_vector=torch.tensor([1, 0, 0]), device='cuda'):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=device)
    if not torch.is_tensor(phi):
        phi = torch.tensor(phi, device=device)
    start_vector = start_vector.float().to(device)

    num = theta.shape[0]
    sin_a, cos_a = torch.sin(torch.zeros(num, device=device)), torch.cos(torch.zeros(num, device=device))
    sin_b, cos_b = torch.sin(phi), torch.cos(phi)
    sin_g, cos_g = torch.sin(theta), torch.cos(theta)

    R_x = torch.zeros(size=(num, 3, 3)).to(device)
    R_x[:, 1, 1] = cos_a
    R_x[:, 1, 2] = -sin_a
    R_x[:, 2, 1] = sin_a
    R_x[:, 2, 2] = cos_a
    R_x[:, 0, 0] = 1

    R_y = torch.zeros(size=(num, 3, 3)).to(device)
    R_y[:, 0, 0] = cos_b
    R_y[:, 0, 2] = sin_b
    R_y[:, 2, 0] = -sin_b
    R_y[:, 2, 2] = cos_b
    R_y[:, 1, 1] = 1

    R_z = torch.zeros(size=(num, 3, 3)).to(device)
    R_z[:, 0, 0] = cos_g
    R_z[:, 0, 1] = -sin_g
    R_z[:, 1, 0] = sin_g
    R_z[:, 1, 1] = cos_g
    R_z[:, 2, 2] = 1

    with torch.no_grad():
        R =  R_z @ R_y @ R_x
        vector = start_vector * step_length
        vector = R @ vector
    return vector.detach().cpu().numpy()

def downsample3dpool(data, ratio=2, mode='avg', dtype=torch.float32):
    data_shape = data.shape
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=dtype, device='cuda')
    data = data.view((1, 1, data_shape[0], data_shape[1], data_shape[2])).contiguous()

    if mode == 'max':
        pool = nn.MaxPool3d(kernel_size=ratio)
    elif mode == 'avg':
        pool = nn.AvgPool3d(kernel_size=ratio)

    data = pool(data)  # type: ignore (for avoid pyplace error report)
    return data[0, 0].detach().cpu().numpy()

def get_color_mapping(samples=1024):
    # hsv_color_map = cm.get_cmap('hsv', 256)
    twi_color_map = cm.get_cmap('twilight', 256)
    twi_shift_color_map = cm.get_cmap('twilight_shifted', 256)

    x, y = np.meshgrid(np.linspace(0, 1, samples), np.linspace(0, 1, samples))

    # hsv_rgb = np.float32(hsv_color_map(x))
    # hsv_bgr = cv2.cvtColor(hsv_rgb, cv2.COLOR_RGBA2BGRA)
    # cv2.imwrite('temp/mapping.png', hsv_bgr * 255)

    twi_rgb = np.float32(twi_color_map(x))  # type: ignore (for avoid pyplace error report)
    twi_bgr = cv2.cvtColor(twi_rgb, cv2.COLOR_RGBA2BGRA)

    twi_sh_rgb = np.float32(twi_shift_color_map(y))  # type: ignore (for avoid pyplace error report)
    twi_sh_bgr = cv2.cvtColor(twi_sh_rgb, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite('temp/mapping_theta.png', twi_bgr * 255)
    cv2.imwrite('temp/mapping_phi.png', twi_sh_bgr * 255)

def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out

def scale2mat(scale_ratio):
    mat44 = np.eye(4)
    for i in range(3):
        mat44[i, i] = scale_ratio
    return mat44

def translate2mat(offset):
    mat44 = np.eye(4)
    mat44[0:3, 3] = offset.T
    return mat44

def homo_rot_mat(mat33):
    mat44 = np.eye(4)
    mat44[0:3, 0:3] = mat33
    return mat44

def idx_map_2_rgb(idx_map):
    [map_height, map_width] = idx_map.shape[:2]

    idx_map_rgb = np.zeros((map_height, map_width, 3))
    # R G B for cv2.imwrite
    # TODO convert to binary operator later
    idx_map_rgb[:, :, 2] = idx_map // (256 * 256)
    idx_map_rgb[:, :, 1] = (idx_map - (idx_map_rgb[:, :, 2] * 256 * 256)) // 256
    idx_map_rgb[:, :, 0] = (idx_map - (idx_map_rgb[:, :, 2] * 256 * 256 +
                                       idx_map_rgb[:, :, 1] * 256))

    return idx_map_rgb

def idx_rgb_recover(idx_bgr):
    [map_height, map_width] = idx_bgr.shape[:2]
    idx_map = np.zeros((map_height, map_width))

    idx_rgb = cv2.cvtColor(idx_bgr, cv2.COLOR_BGR2RGB).astype(np.int64)
    idx_map = idx_rgb[:, :, 0] * 256 * 256 + idx_rgb[:, :, 1] * 256 + idx_rgb[:, :, 2] - 1

    return idx_map

def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)


