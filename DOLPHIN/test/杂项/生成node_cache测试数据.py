


import torch
import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *

data = 'TALK'
path = f'/raid/guorui/DG/dataset/{data}'
src = loadBin(f'{path}/df-src.bin')
dst = loadBin(f'{path}/df-dst.bin')
node_num = torch.unique(torch.cat((src,dst))).shape[0]

node_reorder_map = torch.arange(node_num, dtype = torch.int32)
node_cache_map = torch.arange(10000, dtype = torch.int32)

saveBin(node_reorder_map, f'{path}/node_reorder_map.bin')
saveBin(node_cache_map, f'{path}/node_cache_map.bin')
