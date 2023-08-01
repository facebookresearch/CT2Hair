# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import argparse
from shutil import copyfile
from pyhocon import ConfigFactory
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default='conf/data/Curly.conf')
parser.add_argument('--datapath', type=str, default='data')
parser.add_argument('--gpu', type=str, default='0')
k_args = parser.parse_args()

if __name__ == '__main__':
    conf_text = open(k_args.conf).read()
    conf = ConfigFactory.parse_string(conf_text)
    conf_text = conf_text.replace('DATAPATH', k_args.datapath)
    conf_text = conf_text.replace('CASENAME', conf['output']['name'])
    conf = ConfigFactory.parse_string(conf_text)

    strands_out_dir = os.path.join(conf['output']['dir'], conf['output']['name'])
    oriens_name = conf['output']['name'] \
                + '_oriens' \
                + '_wd_' + str(conf['guide']['wignet_dis']) \
                + '.ply'
    
    if not os.path.exists(os.path.join(strands_out_dir, oriens_name)):
        print(colored("Orientations not found, please run scripts/est_orientations.py first.", "red"))
        exit(1)

    strands_out_name = conf['output']['name'] \
                     + '_oriens' \
                     + '_wd_' + str(conf['guide']['wignet_dis']) \
                     + '_nr_' + str(conf['guide']['nei_radius']) \
                     + '_se_' + str(conf['guide']['sigma_e']) \
                     + '_so_' + str(conf['guide']['sigma_o']) \
                     + '_ts_' + str(conf['guide']['thres_shift']) \
                     + '_nrs_' + str(conf['guide']['nei_radius_seg']) \
                     + '_to_' + str(conf['guide']['thres_orient']) \
                     + '_tl_' + str(conf['guide']['thres_length']) \
                     + '_tnrd_' + str(conf['guide']['thres_nn_roots_dis']) \
                     + '_tlg_' + str(conf['guide']['thres_length_grow']) \
                     + '.ply'

    strands_out_name_simp = conf['output']['name'] + '_guide.bin'

    if platform.system() == 'Linux':
        exe_path = 'CT2Hair/GuideHairStrands/GuideHairStrands'
    elif platform.system() == 'Windows':
        exe_path = 'CT2Hair\\GuideHairStrands\\Release\\GuideHairStrands.exe'

    cmd = '{} 1 '.format(exe_path) \
        + os.path.join(strands_out_dir, oriens_name) + ' ' \
        + os.path.join(strands_out_dir, strands_out_name) + ' ' \
        + str(conf['guide']['nei_radius']) + ' ' \
        + str(conf['guide']['sigma_e']) + ' ' \
        + str(conf['guide']['sigma_o']) + ' ' \
        + str(conf['guide']['thres_shift']) + ' ' \
        + str(conf['guide']['use_cuda']) + ' ' \
        + k_args.gpu + ' ' \
        + str(conf['guide']['nei_radius_seg']) + ' ' \
        + str(conf['guide']['thres_orient']) + ' ' \
        + str(conf['guide']['thres_length']) + ' ' \
        + conf['head']['roots_path'] + ' ' \
        + str(conf['guide']['thres_nn_roots_dis']) + ' ' \
        + str(conf['guide']['thres_length_grow'])
    
    print(colored("Running command:", "yellow"), colored(cmd, "green"))
    os.system(cmd)

    copyfile(os.path.join(strands_out_dir, strands_out_name).replace('ply', 'bin'),
             os.path.join(strands_out_dir, strands_out_name_simp))
