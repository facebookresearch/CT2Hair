# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

from libs.chamfer_distance import ChamferDistance
from utils.strandsutils import spline_strand, pad_strand, natural_cubic_spline_coeffs, NaturalCubicSpline

def uncompress_strand(strands_pc, strands_sep):
    sidx = 0
    strands = []
    for v in strands_sep:
        strands.append(strands_pc[sidx:sidx+v])
        sidx += v

    return strands

def strands_kdtree_query(input_pc, target_kdtree, target_pc, k=10, radius=None):
    if radius:
        idx = target_kdtree.query_ball_point(input_pc, radius)
    else:
        k = np.arange(k) + 1
        dis, idx = target_kdtree.query(input_pc, k)
    idx = idx.reshape(-1)
    idx = np.unique(idx)
    knn_target_pc = target_pc[0, idx, :]
    knn_target_pc = knn_target_pc[None, :]

    return knn_target_pc, dis, idx

def densify_pc(input_pc, density, dup=4):
    dup_sep = 256 // dup

    dense_pc = []

    if torch.is_tensor(input_pc):
        for i in range(len(input_pc)):
            dense_pc.append(input_pc[i].detach().cpu().numpy().tolist())
            num_dup = density[i] // dup_sep
            for j in range(int(num_dup)):
                dense_pc.append(input_pc[i].detach().cpu().numpy().tolist())

        dense_pc = torch.tensor(dense_pc)[None, :].cuda()
    else:
        print("Densifying point cloud...")
        for i in tqdm(range(len(input_pc))):
            dense_pc.append(input_pc[i])
            num_dup = density[i] // dup_sep
            for j in range(int(num_dup)):
                dense_pc.append(input_pc[i])

        dense_pc = np.array(dense_pc)
        print("Number of origina points: %d, number of densified points: %d"%(input_pc.shape[0], dense_pc.shape[0]))

    return dense_pc

def compute_len_loss(strands_pc, strands_pc_next, strands_sep, losstype="l2", **kwargs):
    strands = []
    loss = 0
    strands = uncompress_strand(strands_pc, strands_sep)
    strands_next = uncompress_strand(strands_pc_next, strands_sep)

    for s, s_next in zip(strands, strands_next):
        delta1 = s[:-1] - s[1:]
        delta2 = s_next[:-1] - s_next[1:]

        delta1 = torch.sqrt(torch.sum(delta1**2, dim=-1))
        delta2 = torch.sqrt(torch.sum(delta2**2, dim=-1))

        delta = delta1 - delta2
        if losstype == "l2":
            loss += torch.mean(delta**2)
        elif losstype == "l1":
            loss += torch.mean(torch.abs(delta))
        else:
            raise NotImplementedError(f"losstype {losstype} is not implemented for compute_len_loss")

    loss = loss / len(strands)
    return loss

def compute_len2_loss(strands_pc, strands_pc_next, strands_sep, losstype="max", max_ratio=0.1, **kwargs):
    strands = uncompress_strand(strands_pc, strands_sep)
    strands_next = uncompress_strand(strands_pc_next, strands_sep)

    loss = 0
    for s_ori, s_next in zip(strands, strands_next):
        delta_ori = s_ori[:-2] - s_ori[2:]
        delta_next = s_next[:-2] - s_next[2:]

        delta_ori = torch.sqrt(torch.sum(delta_ori**2, dim=-1))
        delta_next = torch.sqrt(torch.sum(delta_next**2, dim=-1))

        if losstype == "l1":
            loss += torch.mean(torch.abs(delta_next - delta_ori))
        elif losstype == "l2":
            loss += torch.mean((delta_next - delta_ori)**2)
        elif losstype == "max":
            dismat = torch.abs(delta_next - delta_ori)
            thres = max_ratio * delta_ori
            dismat = F.relu(dismat - thres)
            loss += torch.mean(dismat)
        else:
            raise NotImplementedError(f"losstype {losstype} is not defined for compute_len2_loss")

    loss = loss / len(strands)
    return loss

def compute_tangential_loss(strands_pc, strands_pc_next, strands_sep, losstype="l2", cycle=False, **kwargs):
    loss = 0
    strands = uncompress_strand(strands_pc, strands_sep)
    strands_next = uncompress_strand(strands_pc_next, strands_sep)
    
    for s, s_next in zip(strands, strands_next):
        delta = s_next - s
        hair_dirs = s[1:] - s[:-1]
        hair_dirs_normalized = F.normalize(hair_dirs, p=2, dim=-1)
        dot_root = torch.sum(delta[:-1] * hair_dirs_normalized, dim=-1)
        dot_child = torch.sum(delta[1:] * hair_dirs_normalized, dim=-1)
        if cycle:
            hair_dirs_next = s_next[1:] - s_next[:-1]
            hair_dirs_next_normalized = F.normalize(hair_dirs_next, p=2, dim=-1)
            dot_root_next = torch.sum(delta[:-1] * hair_dirs_next_normalized, dim=-1)
            dot_child_next = torch.sum(delta[1:] * hair_dirs_next_normalized, dim=-1)

        if losstype == "l2":
            loss += torch.mean((dot_root - dot_child)**2)
            if cycle:
                loss += torch.mean((dot_root_next - dot_child_next)**2)
        elif losstype == "l1":
            loss += torch.mean(torch.abs(dot_root - dot_child))
            if cycle:
                loss += torch.mean(torch.abs(dot_root_next - dot_child_next))
        else:
            raise NotImplementedError(f"losstype {losstype} is not implemented for compute_tangential_loss")

    loss = loss / len(strands)
    return loss

class StrandsOptimizerNeuralCubic():
    def __init__(self, input_strands, target_pc, target_density, num_strd_pts=128, num_strands_per_opt=1600):
        self.target_pc = target_pc
        self.target_density = target_density * 255
        self.target_pc = densify_pc(self.target_pc, self.target_density)
        print('Building KDTree for target point cloud...')
        self.target_kdtree = KDTree(self.target_pc)

        self.num_strands_per_opt = num_strands_per_opt
        num_origi_strands = input_strands.shape[0]

        filtered_strands = self.filtering_strands(input_strands)
        self.num_strands = len(filtered_strands)
        print('Number original strands: %d, filtered strands: %d'%(num_origi_strands, self.num_strands))

        print('Pre-padding strands for neural cubic interpolation...')
        self.num_strd_pts = num_strd_pts
        self.input_strands = []
        self.times = []
        self.input_num_strds_pts = []
        for i_strd in tqdm(range(self.num_strands)):
            strand = filtered_strands[i_strd][:, :3].astype(np.float32)
            if strand.shape[0] > self.num_strd_pts:
                strand = spline_strand(strand, num_strand_points=self.num_strd_pts)
            self.input_num_strds_pts.append(strand.shape[0])
            strand, time = pad_strand(strand, num_strand_points=self.num_strd_pts)
            self.input_strands.append(strand)
            self.times.append(time)
        self.input_strands = np.array(self.input_strands)
        self.times = np.array(self.times)

        if not torch.is_tensor(self.target_pc):
            self.target_pc = torch.tensor(self.target_pc).float().cuda()[None, :]

        self.epoch = 80
        self.eps = 1e-1

        self.chamfer_dis = ChamferDistance().cuda()
        self.learning_rate = 1e-1

        self.forward_weight = 1.0
        self.backward_weight = 1.0
        self.length_weight = 100.0
        self.tangent_weight = 100.0

    def filtering_strands(self, input_strands, eps=3.0):
        print("Filtering strands outliers...")
        num_strands = input_strands.shape[0]
        filtered_strands = []
        for i_strd in tqdm(range(num_strands)):
            strand = np.array(input_strands[i_strd]).astype(np.float32)[:, :3]
            _, dis, _ = strands_kdtree_query(strand, self.target_kdtree, self.target_pc[None, :])
            if (np.mean(dis) < eps):
                filtered_strands.append(strand)

        return filtered_strands

    def diff_spline(self, strands, times):
        coeffs = natural_cubic_spline_coeffs(times, strands)
        spline = NaturalCubicSpline(coeffs)
        time_pts = torch.arange(self.num_strd_pts).to(strands.device) / (self.num_strd_pts - 1)
        time_pts = time_pts.repeat(strands.shape[0], 1)
        splined_points = spline.evaluate(time_pts)

        return splined_points

    def optimization(self, regularization=True):
        num_opts = self.num_strands // self.num_strands_per_opt + 1
        ori_splined_points = []
        opted_splined_points = []
        opted_strands_pc = []

        strands_seps = np.ones(self.num_strands).astype(np.int16) * self.num_strd_pts

        print('Start optimization...')
        for i_opt in tqdm(range(num_opts)):
            i_start = i_opt * self.num_strands_per_opt
            i_end = min((i_opt + 1) * self.num_strands_per_opt, self.num_strands)
            num_strds_this_opt = i_end - i_start
            strands = torch.tensor(self.input_strands[i_start:i_end]).cuda()
            times = torch.tensor(self.times[i_start:i_end]).cuda()

            strands_noroots = strands[:, 1:, :].clone().detach()
            strands_roots = strands[:, 0:1, :].clone().detach()
            strands_noroots = strands_noroots.requires_grad_(True)
            strands_roots = strands_roots.requires_grad_(True)

            self.optimizer = torch.optim.Adam([strands_noroots], lr=self.learning_rate)

            # before optimization
            strands = torch.concat((strands_roots, strands_noroots), dim=1)
            splined_points = self.diff_spline(strands, times)
            ori_splined_points.extend(splined_points.view(-1, 3).detach().cpu().numpy().tolist())

            constraint_pc = splined_points.view(-1, 3).clone().detach()
            strands_sep = np.ones(num_strds_this_opt).astype(np.int16) * self.num_strd_pts

            for i_epoch in range(self.epoch):
                strands = torch.concat((strands_roots, strands_noroots), dim=1)
                splined_points = self.diff_spline(strands, times)
                input_pc = splined_points.view(1, -1, 3)
                input_pc_numpy = input_pc.clone().detach().cpu().numpy()[0]
                knn_target_pc, _, knn_idx = strands_kdtree_query(input_pc_numpy, self.target_kdtree, self.target_pc)

                dist1, dist2 = self.chamfer_dis(input_pc, knn_target_pc)
                chamfer_loss = self.forward_weight * torch.mean(dist1) + self.backward_weight * torch.mean(dist2)
                if regularization:
                    len_loss = compute_len_loss(constraint_pc, input_pc[0], strands_sep)
                    len2_loss = compute_len2_loss(constraint_pc, input_pc[0], strands_sep)
                    tangent_loss = compute_tangential_loss(constraint_pc, input_pc[0], strands_sep)
                    loss = chamfer_loss + \
                           self.length_weight * len_loss + self.length_weight * len2_loss + \
                           self.tangent_weight * tangent_loss
                else:
                    loss = chamfer_loss

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                print('\topts: %d/%d, epochs: %d/%d, number of points: %d, current loss: %f'%(i_opt, num_opts, i_epoch, self.epoch, input_pc.shape[1], loss.data), end='\r')

            # after optimization
            strands = torch.concat((strands_roots, strands_noroots), dim=1)
            splined_points = self.diff_spline(strands, times)
            opted_splined_points.extend(splined_points.view(-1, 3).detach().cpu().numpy().tolist())

            # original control points
            num_strds_pts = self.input_num_strds_pts[i_start:i_end]
            strands_pc = np.zeros((np.sum(num_strds_pts, keepdims=False), 3))
            sidx = 0
            for i_strd in range(num_strds_this_opt):
                strands_pc[sidx:sidx + num_strds_pts[i_strd]] = strands.detach().cpu().numpy()[i_strd, :num_strds_pts[i_strd]]
                sidx += num_strds_pts[i_strd]
            opted_strands_pc.extend(strands_pc.tolist())

        return np.array(ori_splined_points), np.array(opted_splined_points), strands_seps, np.array(opted_strands_pc), self.input_num_strds_pts