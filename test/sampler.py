


# 测试sample结果
# GPU采样与TGL采样




import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler import *


d = 'WIKI'
batch_size = 600
df = pd.read_csv('/raid/gr/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/gr/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
neg_link_sampler = NegLinkSampler(num_nodes)

for _, rows in df[:train_edge_end].groupby(group_indexes):  
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)).cuda()
    root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()


    start = time.time()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
    

    sampler.sample(root_nodes.cpu().numpy(), root_ts.cpu().numpy())
    ret_tgl = sampler.get_ret()

    emptyCache()
    break


