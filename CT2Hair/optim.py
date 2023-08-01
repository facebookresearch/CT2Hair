# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from utils.pcutils import load_pc
from utils.strandsutils import strandspc2strands, smooth_strands
from datautils.datautils import load_bin_strands, save_bin_strandspc, save_bin_strands
from modules.strands_opt import StrandsOptimizerNeuralCubic

def strands_opt(conf):
    input_strands = load_bin_strands(conf['strands']['interp_strds'])
    print("Load strands finished!")
    # target_pc = load_pc(conf['pc']['pc_path'], load_color=False, load_normal=False)
    target_pc, target_pc_colors = load_pc(conf['pc']['pc_path'], load_color=True, load_normal=False)
    print("Load point cloud finished!")

    strands_opt = StrandsOptimizerNeuralCubic(input_strands, target_pc, target_pc_colors[:, 0], num_strd_pts=64)

    ori_splined_pts, opted_splined_pts, strands_seps, opted_strands_pc, input_num_strds_pts = strands_opt.optimization()

    output_folder = os.path.join(conf['output']['dir'], conf['output']['name'])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    opted_strands = smooth_strands(strandspc2strands(opted_splined_pts, sep=strands_seps))
    save_bin_strands('%s/%s_opted.bin'%(output_folder, conf['output']['name']), opted_strands)