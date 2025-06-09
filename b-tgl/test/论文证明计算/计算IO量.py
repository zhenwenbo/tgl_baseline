


# 测试sample结果
# GPU采样与TGL采样



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
import random
from sampler.sampler_core import ParallelSampler, TemporalGraphBlock

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"设置随机种子{seed}")

set_seed(42)

from sampler.sampler_gpu import *

def cal_tgl_etc(d, batch_size):
    df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
    g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))

    if (d=='MAG'):
        train_edge_end = 1111701860
        val_edge_end = 1198206616
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]



    fan_nums = [10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)

    class NegLinkSampler:

        def __init__(self, num_nodes):
            self.num_nodes = num_nodes

        def sample(self, n):
            return np.random.randint(self.num_nodes, size=n)

    num_nodes = (g['indptr'].shape[0] - 1)
    neg_link_sampler = NegLinkSampler(num_nodes)

    max_edge_num = 0
    max_node_num = 0
    left = 0
    right = 0
    cur_batch = 0

    reverse = False

    def cal_io(tensors, feat_len = [472+172,172]):
        res = 0
        for i,t in enumerate(tensors):
            res += t.shape[0] * feat_len[i] * 4 / 1024 ** 2
        return res

    tgl_IO = 0
    etc_IO = 0


    while True:
        
        
        if (right == train_edge_end):
            break
        right += batch_size
        right = min(train_edge_end, right)
        rows = df[left:right]
        # break

        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        # print(root_ts)
        # print(root_nodes)
        root_nodes = torch.from_numpy(root_nodes).cuda()
        root_ts = torch.from_numpy(root_ts).cuda()
        start = time.time()

        sample_start = time.time()
        ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
        sample_time = time.time() - sample_start

        gen_start = time.time()
        # for ret in ret_list:
        #     src,dst,outts,outeid,root_nodes,root_ts = ret
        #     mask = src > -1
        #     src = src[mask]
        #     dst = dst[mask]
        #     outeid = outeid[mask]
        #     outts = outts[mask]
        mfgs = sampler_gpu.gen_mfgs(ret_list)
        nids = mfgs[0][0].ndata['ID']['_N'].cuda()
        eids = mfgs[0][0].edata['ID'].cuda()
        nids_uni = torch.unique(nids)
        eids_uni = torch.unique(eids)

        tgl_IO += cal_io([nids, eids])
        etc_IO += cal_io([nids_uni, eids_uni])
        gen_time = time.time() - gen_start

        left = right
        cur_batch += 1

    print(f"tgl_IO: {tgl_IO:.4f}MB etc:{etc_IO:.4f}MB")



d = 'TALK'
batch_sizes = [500,1000,1500,2000,60000,600000]
for batch_size in batch_sizes:
    cal_tgl_etc(d, batch_size)