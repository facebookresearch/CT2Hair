# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.utils.data as th_data

from utils.strandsutils import spline_strand, pad_strand

class TbnStrandsBinDataset(th_data.Dataset):
    def __init__(self, tbn_strands, is_resampled=True, num_strds_points=100):
        self.num_strands = len(tbn_strands)
        self.tbn_strands = tbn_strands
        self.batch_size = 300
        self.num_workers = 12
        self.num_strds_points = num_strds_points
        self.is_resampled = is_resampled

    def __len__(self):
        return self.num_strands

    def __getitem__(self, idx):
        strand = self.tbn_strands[idx].astype(np.float32)
        out_dict = {}

        if not self.is_resampled:
            if strand.shape[0] > self.num_strds_points:
                strand = spline_strand(strand, num_strand_points=self.num_strds_points)
            strand, time = pad_strand(strand, num_strand_points=self.num_strds_points)
            out_dict['times'] = torch.tensor(time).float()

        assert strand.shape[0] == self.num_strds_points, 'Need resample strands to a fixed number.'

        # Scale unit from mm to m
        strand = strand / 1000.

        out_dict['points'] = torch.tensor(strand).float()
        return out_dict

    def get_dataloader(self):
        return th_data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)