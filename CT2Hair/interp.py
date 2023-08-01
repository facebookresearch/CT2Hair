# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import numpy as np

from utils.meshutils import read_mesh, process_head_model
from utils.strandsutils import smooth_strands, downsample_strands, duplicate_strands, merge_strands
from datautils.datautils import load_bin_strands, save_bin_strands
from modules.neural_strands import NeuralStrands

def neural_interp(conf):
    output_folder = os.path.join(conf['output']['dir'], conf['output']['name']) # for synthesized data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # prepressing
    strands = load_bin_strands(conf['strands']['guide_strds'])
    strands = smooth_strands(strands, lap_constraint=4.0, pos_constraint=2.0)
    # strands = downsample_strands(strands)

    # fit head model
    head_mesh = read_mesh(conf['head']['head_path'])
    head_texture = cv2.imread(conf['head']['head_scalp_tex'])
    head_mesh, scalp_mesh, scalp_faces_idx = process_head_model(head_mesh, head_texture, 
                                                                conf['head']['roots_path'],
                                                                np.array(conf['head']['target_face_base'], dtype=np.float32),
                                                                is_deformation=True)

    head_write = head_mesh.export('%s/%sface_reg.ply'%(output_folder, conf['output']['name']))

    neural_strands = NeuralStrands(is_resampled=False)
    neural_strands.prep_strands_data(strands, head_mesh, scalp_mesh, scalp_faces_idx)

    # interpolation
    neural_strands.get_neural_representations(iter_opt=0)
    # neural_strands.get_neural_representations(iter_opt=300, lr=1e-2)  # new trained model fits very well, no need to fit again
    denoised_strds_idxs = neural_strands.denoise_neural_texture(num_del_cls=0, do_denoise=False)
    texel_roots_mask = cv2.imread(conf['head']['head_scalp_tex'], 2) / 255.
    neural_strands.interpolation_knn(texel_roots_mask, interp_kernel_size=5, interp_neig_pts=3)
    interp_strds = neural_strands.world_strands_from_texels(neural_strands.interp_neural_texture, neural_strands.interp_strds_idx_map)

    # save results
    save_bin_strands('%s/%s_interp.bin'%(output_folder, conf['output']['name']), interp_strds.detach().cpu().numpy().astype(np.float32))
    merged_strands = merge_strands([neural_strands.original_strands, interp_strds.detach().cpu().numpy().astype(np.float32)])
    merged_strands = downsample_strands(merged_strands) # TODO use neural spline with GPU to speed up.
    merged_strands = duplicate_strands(merged_strands, ratio=4)
    save_bin_strands('%s/%s_merged.bin'%(output_folder, conf['output']['name']), merged_strands)
    print('Saving done!')