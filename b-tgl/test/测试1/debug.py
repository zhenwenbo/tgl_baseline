

import numpy as np
import torch

import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler.sampler import *


# path = '/raid/gr/DG/dataset/TALK/part-60000-[10, 10]/part5_node_map.bin'
# path1 = '/raid/gr/DG/dataset/TALK/正确的二跳/part5_node_map.bin'

# res1 = loadBin(path)
# res2 = loadBin(path1)

# print(torch.sum(res1 != res2))

# res1 = np.fromfile(path, dtype = np.float32)
# res2 = np.fromfile(path1, dtype = np.float32)

node_map = torch.load('/home/gr/workspace/bucket_cache_node_map.pt')
node_feat = torch.load('/home/gr/workspace/bcnf.pt')

node_feats = loadBin('/raid/guorui/DG/dataset/TALK/node_features.bin')
real_node_feat = node_feats[node_map]

pos_node_map = torch.tensor([3,7,12,13,31,37,38,200000], dtype = torch.int32)
cmask = torch.isin(pos_node_map, node_map)
cfeat = node_feat[torch.isin(node_map, pos_node_map)]
feat1 = node_feats[pos_node_map.long()]

pos_node_map = torch.load('/home/gr/workspace/pos_node_map.pt')
pos_node_feat = node_feats[pos_node_map.long()]
torch.save(pos_node_feat, '/home/gr/workspace/pos_node_feat.pt')

bucket_cache_node_map = torch.load('/home/gr/workspace/bcnp.pt')
pos_node_map = torch.load('/home/gr/workspace/pos_node_map.pt')
bucket_cache_node_feat = torch.load('/home/gr/workspace/bcnf.pt')

hybrid_node_cache_mask = torch.isin(pos_node_map, bucket_cache_node_map)
bucket_cache_node_map_sort, bucket_cache_node_map_indices = torch.sort(bucket_cache_node_map)

# 需要找出pos_node在bucket中的实际位置
# 先找出pos中有，bucket中也有的
pos_in_cache_mask = torch.isin(pos_node_map, bucket_cache_node_map, assume_unique=True)
pos_in_cache = pos_node_map[pos_in_cache_mask]
bucket_cache_node_map_sort, bucket_cache_node_map_indices = torch.sort(bucket_cache_node_map)
result = bucket_cache_node_map_indices[torch.searchsorted(bucket_cache_node_map_sort, pos_in_cache)]
hybrid_node_cache_feat = bucket_cache_node_feat[result]

tensor2 = torch.tensor([2,5,3,7,12,9,4])
tensor1 = torch.tensor([2,7,12,2])
tensor2_sorted, argsort = torch.sort(tensor2)
result = argsort[torch.searchsorted(tensor2_sorted, tensor1)]

hybrid_node_cache_feat = bucket_cache_node_feat[torch.isin(bucket_cache_node_map_sort, pos_node_map)][bucket_cache_node_map_indices]

need_nodes = pos_node_map[hybrid_node_cache_mask]
node_feats[need_nodes]


import torch
path = '/home/gr/workspace/false/'
pos_node_map = torch.load( path + 'pos_node_map.pt')
node_feats = torch.load( path + 'node_feats.pt')
node_d_map = torch.load( path + 'node_d_map.pt')

path = '/home/gr/workspace/true/'
pos_node_map1 = torch.load( path + 'pos_node_map.pt')
node_feats1 = torch.load( path + 'node_feats.pt')
node_d_map1 = torch.load( path + 'node_d_map.pt')

print(f"{torch.sum(pos_node_map != pos_node_map1)}")
print(f"{torch.sum(node_feats != node_feats1)}")
print(f"{torch.sum(node_d_map != node_d_map1)}")

import torch
node_map1 = torch.load('/home/gr/workspace/node_d_map_pre1.pt')
node_map = torch.load('/home/gr/workspace/node_d_map_pre.pt')