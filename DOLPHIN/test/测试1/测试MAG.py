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
d = 'MAG'
batch_size = 600000

g, datas, df_conf = load_graph_bin(d)
if (d=='MAG'):
    train_edge_end = 1111701860
    val_edge_end = 1198206616

    # group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler.sampler_gpu import *
fan_nums = [10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

# dgl.sumDegree()

total_src = datas['src'].cuda().to(torch.int32)
total_dst = datas['dst'].cuda().to(torch.int32)

inDegree = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
outDegree = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
dgl.sumDegree(inDegree, outDegree, total_src, total_dst)

total_src = total_src.cpu();
total_dst = total_dst.cpu();
inDegree_sort, inDegree_idx = torch.sort(inDegree, descending=True)
saveBin(inDegree_idx, "/raid/guorui/DG/dataset/MAG/inDegreeSort_idx.bin")