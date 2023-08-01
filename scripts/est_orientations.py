# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import argparse
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

    if not os.path.exists(str(conf['vdb']['path'])):
        print("Input VDB file does not exists.")
        exit(1)

    oriens_out_dir = os.path.join(conf['output']['dir'], conf['output']['name'])
    os.makedirs(oriens_out_dir, exist_ok=True)
    oriens_out_name = conf['output']['name'] \
                    + '_oriens' \
                    + '_wd_' + str(conf['guide']['wignet_dis']) \
                    + '.ply'

    if platform.system() == 'Linux':
        exe_path = 'CT2Hair/GuideHairStrands/GuideHairStrands'
    elif platform.system() == 'Windows':
        exe_path = 'CT2Hair\\GuideHairStrands\\Release\\GuideHairStrands.exe'

    cmd = '{} 0 '.format(exe_path) \
        + str(conf['vdb']['path']) + ' ' \
        + str(conf['vdb']['voxel_size']) + ' ' \
        + conf['head']['roots_path'] + ' ' \
        + str(conf['guide']['wignet_dis']) + ' ' \
        + os.path.join(oriens_out_dir, oriens_out_name)
    
    print(colored("Running command:", "yellow"), colored(cmd, "green"))
    os.system(cmd)
