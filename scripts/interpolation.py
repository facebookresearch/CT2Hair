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
from interp import neural_interp

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default='conf/data/Curly.conf')
parser.add_argument('--datapath', type=str, default='data')
parser.add_argument('--gpu', type=str, default='0')
k_args = parser.parse_args()

if __name__ == '__main__':
    conf_text = open(k_args.conf).read()
    conf_text = conf_text.replace('DATAPATH', k_args.datapath)
    conf = ConfigFactory.parse_string(conf_text)
    conf_text = conf_text.replace('DATAPATH', k_args.datapath)
    conf_text = conf_text.replace('CASENAME', conf['output']['name'])
    conf = ConfigFactory.parse_string(conf_text)

    strands_out_dir = os.path.join(conf['output']['dir'], conf['output']['name'])
    strands_name = conf['output']['name'] \
                 + '_guide.bin'

    conf['strands']['guide_strds'] = os.path.join(strands_out_dir, strands_name)

    if not os.path.exists(os.path.join(strands_out_dir, strands_name)):
        print(colored("Guide hair strands not found, please run scripts/gen_guide_strands.py first.", "red"))
        exit(1)

    print(colored("Running interpolation:", "yellow"))
    neural_interp(conf)
