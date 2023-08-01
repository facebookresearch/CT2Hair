# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
from pyhocon import ConfigFactory
from termcolor import colored

sys.path.append('CT2Hair/')
from optim import strands_opt

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
    pc_name = conf['output']['name'] \
              + '_oriens' \
              + '_wd_' + str(conf['guide']['wignet_dis']) \
              + '.ply'
    strands_name = conf['output']['name'] \
                 + '_merged.bin'

    conf['pc']['pc_path'] = os.path.join(strands_out_dir, pc_name)
    conf['strands']['interp_strds'] = os.path.join(strands_out_dir, strands_name)

    if not os.path.exists(os.path.join(strands_out_dir, strands_name)):
        print(colored("Interpolated hair strands not found, please run scripts/interpolation.py first.", "red"))
        exit(1)
    
    print(colored("Running optimization:", "yellow"))
    strands_opt(conf)
