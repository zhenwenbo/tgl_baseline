import torch
import numpy as np
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

data = 'BITCOIN'


if (data in ['BITCOIN']):
    train_edge_end = 86063713
    val_edge_end = 110653345



e_src = loadBin(f'/raid/guorui/DG/dataset/{data}/df-src.bin')
e_dst = loadBin(f'/raid/guorui/DG/dataset/{data}/df-dst.bin')


t_src = e_src[:train_edge_end].cuda()
t_dst = e_dst[:train_edge_end].cuda()
node_num = max(torch.max(t_src), torch.max(t_dst))
inDegree = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
outDegree = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
dgl.sumDegree(inDegree, outDegree,t_src, t_dst)
d = inDegree + outDegree


inDegree_sort, inDegree_sort_ind = torch.sort(inDegree, descending=True)
d_sort, d_sort_ind = torch.sort(d, descending=True)

saveBin(inDegree_sort_ind[:1000000], f'/raid/guorui/DG/dataset/{data}/ind_1000000.bin')
saveBin(d_sort_ind[:1000000], f'/raid/guorui/DG/dataset/{data}/d_1000000.bin')