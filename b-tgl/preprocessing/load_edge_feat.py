import argparse
import os
import json
import sys
import os
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import *
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
args=parser.parse_args()


edge_feats = loadBin(f'/raid/guorui/DG/dataset/{args.data}/edge_features.bin')
