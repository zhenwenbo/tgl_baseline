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
# 1.平均度数 2.最大度数 3.时间戳范围(max) 4.重复action率


def cal(data):
    path = f'/raid/guorui/DG/dataset/{data}'
    src = loadBin(path + f'/df-src.bin').cuda()
    dst = loadBin(path + f'/df-dst.bin').cuda()
    time = loadBin(path + f'/df-time.bin').cuda()

    max_node = max(torch.max(src), torch.max(dst))
    inDegree = torch.zeros(max_node, dtype = torch.int32, device = 'cuda:0')
    outDegree = torch.zeros_like(inDegree)
    dgl.sumDegree(inDegree, outDegree, src, dst)
    d_max = torch.max(inDegree)
    d_aver = torch.mean(inDegree.float())

    max_t = torch.max(time) - torch.min(time)

    edges = torch.stack((src, dst), dim = 1)
    uni_edges = torch.unique(edges, dim = 0)
    
    print(f"数据集: {data}, d_max:{d_max}, d_aver:{d_aver}, max_t:{max_t}")

ds = ['LASTFM', 'TALK', 'STACK', 'BITCOIN', 'GDELT']
for d in ds:
    cal(d)