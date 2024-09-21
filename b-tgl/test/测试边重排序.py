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

dataset = 'MAG'

g, df = load_graph(dataset)

eid = torch.from_numpy(g['eid']).cuda()
print(f'eid shape: {eid.shape} unique shape: {torch.unique(eid).shape}')

ef = loadBin(f'/raid/guorui/DG/dataset/{dataset}/edge_features.bin')

ef = ef[eid]

import torch

max_val = eid.max().item()
res = torch.zeros(max_val + 1, dtype=torch.long).cuda()
res.scatter_(0, eid, torch.arange(len(eid)).cuda())

print(res)

# 保存后，例如需要寻找eid = 100的边特征，那么需要edge_feat_reorder[map[eid]]即可
saveBin(ef.cpu(), f'/raid/guorui/DG/dataset/{dataset}/edge_features_reorder.bin')
saveBin(res.cpu().to(torch.int32), f'/raid/guorui/DG/dataset/{dataset}/edge_reorder_map.bin')