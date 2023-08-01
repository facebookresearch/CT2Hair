# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import splines
import torch
import numpy as np

from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from utils.pcutils import get_bbox

def scale_roots_positions(roots_points, scale_ratio):
    _, bbox_center = get_bbox(roots_points)
    temp_points = (roots_points - bbox_center) * scale_ratio + bbox_center
    roots_points = temp_points
    return roots_points

def get_roots_normals(roots_points):
    _, bbox_center = get_bbox(roots_points)
    normals = roots_points - bbox_center
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return normals

def get_strand_length(strand):
    delta = strand[:-1] - strand[1:]
    delta_length = np.sqrt(np.sum(delta**2, axis=1, keepdims=False))
    length = np.sum(delta_length, axis=0, keepdims=False)
    return length, delta_length

def get_strands_length(strands):
    deltas = strands[:, :-1] - strands[:, 1:]
    
    if torch.torch.is_tensor(strands):
        delta_lengths = torch.sqrt(torch.sum(deltas**2, dim=2, keepdim=False))
        lengths = torch.sum(delta_lengths, dim=1, keepdim=False)
    else:
        delta_lengths = np.sqrt(np.sum(deltas**2, axis=2, keepdims=False))
        lengths = np.sum(delta_lengths, axis=1, keepdims=False)
    
    return lengths, delta_lengths

def get_strands_roots(strands, scale_ratio=1.0):
    roots = []
    num_strands = strands.shape[0]

    for i_strand in range(num_strands):
        roots.append(strands[i_strand][0][:3])

    points = np.array(roots)

    if not scale_ratio == 1.0:
        points = scale_roots_positions(points, scale_ratio)
    
    normals = get_roots_normals(points)
    return points, normals

def line_interpolate(start_point, end_point, interp_count):
    interped_points = []
    if interp_count == 0:
        return interped_points

    delta = end_point - start_point
    delta_length = math.sqrt(np.sum(delta**2, axis=0, keepdims=True))
    step_dir = delta / delta_length
    step_size = delta_length / (interp_count + 1)
    for i in range(interp_count):
        interped_points.append(start_point + step_dir * (i + 1) * step_size)
    return interped_points 

def resample_strand(strand, tangents=None, num_strand_points=200):
    num_ori_points = strand.shape[0]
    assert num_ori_points < num_strand_points, "number of resampled points must larger than the original one"

    strand_length, delta_length = get_strand_length(strand)
    step_length = strand_length / (num_strand_points - 1)

    resampled_strand = []
    if tangents is None:
        interp_idxs = np.where(delta_length > step_length)[0]

        interp_segs = delta_length[interp_idxs]
        interp_segs_rank_idxs = np.argsort(-1 * interp_segs)

        new_step_length = np.sum(interp_segs) / (num_strand_points - (num_ori_points - interp_idxs.shape[0]))
        interp_counts = np.clip((interp_segs / new_step_length).astype(np.int16) - 1, 0, num_strand_points - 1)    # supposed to always be postive or zero
        interp_counts_sum = np.sum(interp_counts, axis=0, keepdims=False)   # supposed to always less than num_strand_points
        assert interp_counts_sum + num_ori_points <= num_strand_points, "utils:strandsutils.py, FIXME, strand resample error, Interp counts: %d, Original Counts: %d"%(interp_counts_sum, num_ori_points)

        num_ext_interp = num_strand_points - num_ori_points - interp_counts_sum
        ext_interp_segs = interp_segs_rank_idxs[:num_ext_interp]
        interp_counts[ext_interp_segs] += 1 # Interpolate one more point in this segs

        interp_delta_count = 0
        for i_delta in range(num_ori_points - 1):
            resampled_strand.append(strand[i_delta])
            if delta_length[i_delta] > step_length:
                interped_points = line_interpolate(strand[i_delta], strand[i_delta + 1], interp_counts[interp_delta_count])
                resampled_strand.extend(interped_points)
                interp_delta_count += 1
        resampled_strand.append(strand[num_ori_points - 1])

    resampled_strand = np.array(resampled_strand)
    assert resampled_strand.shape[0] == 200, "interpolation failed, number of resampled: %d."%(resampled_strand.shape[0])
    return resampled_strand

def augment_strand(strand, aug_config):
    if aug_config["rotation_z_max_angle"] > 0:
        theta_z = aug_config["rotation_z_max_angle"]
        rtheta = (np.random.rand() * 2. - 1.) * theta_z * np.pi / 180.
        rot_mat = np.asarray([[np.cos(rtheta), -np.sin(rtheta), 0.],
                              [np.sin(rtheta), np.cos(rtheta), 0.],
                              [            0.,              0., 1.]], dtype=np.float32)

        strand = (rot_mat[:, :] @ strand.T).T

    if np.sum(aug_config["random_stretch_xyz_magnitude"]) > 0:
        sc = np.random.rand(3) * 2 - 1
        sc = 1 + np.asarray(aug_config["random_stretch_xyz_magnitude"]) * sc
        strand = strand * sc
    return strand

def spline_strand(strand, num_strand_points=100):
    num_ori_points = strand.shape[0]
    interp_spline = splines.CatmullRom(strand)
    interp_idx = np.arange(num_strand_points) / (num_strand_points / (num_ori_points - 1))
    interp_strand = interp_spline.evaluate(interp_idx)
    assert interp_strand.shape[0] == num_strand_points, "Spline error."

    return interp_strand

def pad_strand(strand, num_strand_points=100):
    num_ori_points = strand.shape[0]
    if num_ori_points > num_strand_points:
        return strand[:num_strand_points]
    
    num_pad = num_strand_points - num_ori_points
    last_delta = strand[-1] - strand[-2]
    offsets = np.arange(num_pad) + 1
    offsets = offsets[:, None]
    last_delta = last_delta[None, :]
    offsets = offsets * last_delta
    # padded_strand = np.zeros_like(offsets) + strand[-1]
    padded_strand = offsets + strand[-1]
    padded_strand = np.concatenate((strand, padded_strand), axis=0)

    ori_time = np.linspace(0, 1, num_ori_points)
    strd_len, delta_len = get_strand_length(strand) # modify time by length
    ori_time[1:] = delta_len / strd_len
    ori_time = np.add.accumulate(ori_time)

    padded_time = 1. + (np.arange(num_pad) + 1) * (1. / num_ori_points)
    padded_time = np.concatenate((ori_time, padded_time), axis=0)
    return padded_strand, padded_time

def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    A_upper, _ = torch.broadcast_tensors(A_upper[:, None, :], b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower[:, None, :], b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal[:, None, :], b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)

def cubic_spline_coeffs(t, x):
    # x should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = x.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[..., 1:] - t[..., :1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    else:
        # Set up some intermediate values
        time_diffs = t[..., 1:] - t[..., :-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared[:, None, :]

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty((x.shape[0], length), dtype=x.dtype, device=x.device)
        system_diagonal[..., :-1] = time_diffs_reciprocal
        system_diagonal[..., -1] = 0
        system_diagonal[..., 1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal,
                                             system_diagonal, time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        time_diffs_reciprocal = time_diffs_reciprocal[:, None, :]
        time_diffs_reciprocal_squared = time_diffs_reciprocal_squared[:, None, :]
        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d

def natural_cubic_spline_coeffs(t, x):
    a, b, two_c, three_d = cubic_spline_coeffs(t, x.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    c = two_c.transpose(-1, -2) / 2
    d = three_d.transpose(-1, -2) / 3
    return t, a, b, c, d

class NaturalCubicSpline:
    def __init__(self, coeffs, **kwargs):
        super(NaturalCubicSpline, self).__init__(**kwargs)

        t, a, b, c, d = coeffs

        self._t = t
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def evaluate(self, t):
        maxlen = self._b.size(-2) - 1
        inners = torch.zeros((t.shape[0], t.shape[1], 3)).to(t.device)
        for i_b in range(self._t.shape[0]):
            index = torch.bucketize(t.detach()[i_b], self._t[i_b]) - 1
            index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._t; this is correct behaviour
            fractional_part = t[i_b] - self._t[i_b][index]
            fractional_part = fractional_part.unsqueeze(-1)
            inner = self._c[i_b, index, :] + self._d[i_b, index, :] * fractional_part
            inner = self._b[i_b, index, :] + inner * fractional_part
            inner = self._a[i_b, index, :] + inner * fractional_part
            inners[i_b] = inner
        return inners

    def derivative(self, t, order=1):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        if order == 1:
            inner = 2 * self._c[..., index, :] + 3 * self._d[..., index, :] * fractional_part
            deriv = self._b[..., index, :] + inner * fractional_part
        elif order == 2:
            deriv = 2 * self._c[..., index, :] + 6 * self._d[..., index, :] * fractional_part
        else:
            raise ValueError('Derivative is not implemented for orders greater than 2.')
        return deriv

# post-processing

def merge_strands(strands_list):
    strands_all = []
    for strands in strands_list:
        for i_strand in range(strands.shape[0]):
            strands_all.append(strands[i_strand])

    strands_all = np.array(strands_all, dtype=object)
    return strands_all

def strandspc2strands(strandspc, sep):
    num_strands = len(sep)
    strands = []
    num_pts = 0
    for i_strand in range(num_strands):
        strands.append(strandspc[num_pts : num_pts + int(sep[i_strand])])
        num_pts += sep[i_strand]

    strands = np.array(strands, dtype=object)
    return strands

def smnooth_strand(strand, lap_constraint=2.0, pos_constraint=1.0, fix_tips=False):
    num_pts = strand.shape[0]
    num_value = num_pts * 3 - 2 + num_pts
    smoothed_strand = np.copy(strand)

    # construct laplacian sparse matrix
    i, j, v = np.zeros(num_value, dtype=np.int16), np.zeros(num_value, dtype=np.int16), np.zeros(num_value)

    i[0], i[1], i[2 + (num_pts - 2) * 3], i[2 + (num_pts - 2) * 3 + 1] = 0, 0, num_pts - 1, num_pts - 1
    i[2 : num_pts * 3 - 4] = np.repeat(np.arange(1, num_pts - 1), 3)
    i[num_pts * 3 - 2:] = np.arange(num_pts) + num_pts

    j[0], j[1], j[2 + (num_pts - 2) * 3], j[2 + (num_pts - 2) * 3 + 1] = 0, 1, num_pts - 2, num_pts - 1
    j[2 : num_pts * 3 - 4] = np.repeat(np.arange(1, num_pts - 1), 3) \
                           + np.repeat(np.array([-1, 0, 1], dtype=np.int16), num_pts - 2).reshape(num_pts - 2, 3, order='F').ravel()
    j[num_pts * 3 - 2:] = np.arange(num_pts)

    v[0], v[1], v[2 + (num_pts - 2) * 3], v[2 + (num_pts - 2) * 3 + 1] = 1, -1, -1, 1
    v[2 : num_pts * 3 - 4] = np.repeat(np.array([-1, 2, -1], dtype=np.int16), num_pts - 2).reshape(num_pts - 2, 3, order='F').ravel()
    v = v * lap_constraint
    v[num_pts * 3 - 2:] = pos_constraint

    A = coo_matrix((v, (i, j)), shape=(num_pts * 2, num_pts))
    At = A.transpose()
    AtA = At.dot(A)

    # solving
    for j_axis in range(3):
        b = np.zeros(num_pts * 2)
        b[num_pts:] = smoothed_strand[:, j_axis] * pos_constraint
        Atb = At.dot(b)

        x = spsolve(AtA, Atb)
        smoothed_strand[:, j_axis] = x[:num_pts]

    if fix_tips:
        strand[1:-1] = smoothed_strand[1:-1]
    else:
        strand = smoothed_strand

    return strand

def smooth_strands(strands, lap_constraint=2.0, pos_constraint=1.0, fix_tips=False):
    loop = tqdm(range(strands.shape[0]))
    loop.set_description("Smoothing strands")
    for i_strand in loop:
        strands[i_strand] = smnooth_strand(strands[i_strand], lap_constraint, pos_constraint, fix_tips)
    
    return strands

def downsample_strands(strands, min_num_pts=5, tgt_num_pts=64):
    loop = tqdm(range(strands.shape[0]))
    loop.set_description("Downsampling strands points")
    for i_strand in loop:
        num_pts = strands[i_strand].shape[0]
        downsampled_strand = np.copy(strands[i_strand][:, 0:3])

        if num_pts <= 2:
            pass
        elif num_pts > 2 and num_pts < min_num_pts:
            interp_pts = (downsampled_strand[:-1] + downsampled_strand[1:]) / 2.
            interp_strand = np.zeros((num_pts * 2 - 1, 3))
            interp_strand[::2] = downsampled_strand
            interp_strand[1::2] = interp_pts
            downsampled_strand = interp_strand
        elif num_pts > min_num_pts and num_pts < tgt_num_pts:
            pass
        else:
            interp_spline = splines.CatmullRom(downsampled_strand)
            interp_idx = np.arange(tgt_num_pts) / (tgt_num_pts / (num_pts - 1))
            downsampled_strand = interp_spline.evaluate(interp_idx)
    
        strands[i_strand] = downsampled_strand

    return strands

def duplicate_strands(strands, ratio=5, perturation=1.0):
    loop = tqdm(range(strands.shape[0]))
    loop.set_description('Duplicating strands')

    duplicated_strands_list = []
    for i_strand in loop:
        strand = strands[i_strand][:, 0:3]
        num_pts = strand.shape[0]
        duplicated_strands = np.repeat(strand.reshape(1, num_pts, 3), ratio, axis=0)

        start_tangent = strand[1] - strand[0]
        offsets = np.random.rand(ratio, 3)
        offsets[:, 2] = -(offsets[:, 0] * start_tangent[0] + offsets[:, 1] * start_tangent[1]) / (start_tangent[2] + 1e-6)
        offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)

        offsets[0] *= 0
        scale_ratio = np.random.rand(ratio, 1) * perturation + perturation
        offsets = offsets * scale_ratio
        offsets = np.repeat(offsets.reshape(ratio, 1, 3), num_pts, axis=1)

        duplicated_strands = duplicated_strands + offsets
        for j in range(ratio):
            duplicated_strands_list.append(duplicated_strands[j])

    strands = np.array(duplicated_strands_list, dtype=object)
    return strands