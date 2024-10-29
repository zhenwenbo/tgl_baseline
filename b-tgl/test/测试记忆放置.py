




# 测试将node memory和mailbox 的读取下移到磁盘中的性能，每个block是否能限制在0.1s 内？





# 测试sample结果
# GPU采样与TGL采样



import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler.sampler import *
import random
from sampler.sampler_gpu import *
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"设置随机种子{seed}")

set_seed(0)


def pre():
    d = 'BITCOIN'
    batch_size = 60000
    g, df = load_graph(d)

    if (d in ['BITCOIN']):
        train_edge_end = 86063713
        val_edge_end = 110653345
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]

        # group_indexes = np.array(df[:train_edge_end].index // batch_size)



    
    fan_nums = [10,10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)

    sample_param, memory_param, gnn_param, train_param = parse_config('/raid/guorui/workspace/dgnn/b-tgl/config/TGN-2.yml')
    sample_param['layer'] = len(fan_nums)
    sample_param['neighbor'] = fan_nums

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

    block_his = []
    block_his_len = 0
    while True:
        
        if (right == train_edge_end):
            break
        right += batch_size
        right = min(train_edge_end, right)
        rows = df[left:right]
        # break

        root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        root_ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)

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
        gen_time = time.time() - gen_start

        src_nodes = torch.unique(mfgs[0][0].srcdata['ID']).cpu()
        block_his.append(src_nodes)
        block_his_len += src_nodes.shape[0]
        print(f"block his总长度{block_his_len}, 占用{block_his_len * 4/ 1024**2}MB")
        
        left = right
        cur_batch += 1
        print(f"当前batch: {cur_batch}:{train_edge_end//60000}采样花销: {sample_time:.7f}s gen block花销{gen_time:.7f}s")

        emptyCache()

# pre()
path = '/raid/guorui/DG/dataset/BITCOIN'
def init_memory(dim_edge_feat, num_nodes):
    memory_param = {'dim_out': 100, 'mailbox_size': 1}
    
    memory = torch.randn((num_nodes, memory_param['dim_out']), dtype=torch.float32).reshape(num_nodes, -1)
    memory_ts = torch.randn(num_nodes, dtype=torch.float32).reshape(num_nodes, -1)
    mailbox = torch.randn((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32).reshape(num_nodes, -1)
    mailbox_ts = torch.randn((num_nodes, memory_param['mailbox_size']), dtype=torch.float32).reshape(num_nodes, -1)

    total_memory = torch.cat([memory, memory_ts, mailbox, mailbox_ts], dim = 1)
    del(memory, memory_ts, mailbox, mailbox_ts)
    saveBin(total_memory, path + '/total_memory.bin')

def load_mem(nodes, t):
    # print(f"先只拿mailbox(最多的)")
    if (t):
        print(f"测试纯顺序读")
        nodes = torch.arange(nodes[0], nodes.shape[0] + nodes[0], dtype = nodes.dtype, device = nodes.device)
    load_start = time.time()
    res = loadBinDisk(path + '/total_memory.bin', nodes)

    print(f"load mailbox, nodes shape: {nodes.shape[0]}, use time: {time.time() - load_start:.6f}s {(res.shape[0] * res.shape[1] * 4 / 1024 ** 2 / (time.time() - load_start)):.2f}MB/s")

def test():
    d = 'BITCOIN'
    batch_size = 60000
    g, df = load_graph(d)

    if (d in ['BITCOIN']):
        train_edge_end = 86063713
        val_edge_end = 110653345
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]

        # group_indexes = np.array(df[:train_edge_end].index // batch_size)



    
    fan_nums = [10,10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)

    sample_param, memory_param, gnn_param, train_param = parse_config('/raid/guorui/workspace/dgnn/b-tgl/config/TGN-2.yml')
    sample_param['layer'] = len(fan_nums)
    sample_param['neighbor'] = fan_nums

    class NegLinkSampler:

        def __init__(self, num_nodes):
            self.num_nodes = num_nodes

        def sample(self, n):
            return np.random.randint(self.num_nodes, size=n)

    num_nodes = (g['indptr'].shape[0] - 1)
    neg_link_sampler = ReNegLinkSampler(num_nodes, 0.9)

    max_edge_num = 0
    max_node_num = 0
    left = 0
    right = 0
    cur_batch = 0

    reverse = False

    block_his = []
    block_his_len = 0
    pre_src_nodes = None
    while True:
        
        if (right == train_edge_end):
            break
        right += batch_size
        right = min(train_edge_end, right)
        rows = df[left:right]
        # break

        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)

        # root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        # root_ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)


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
        gen_time = time.time() - gen_start

        src_nodes = torch.unique(mfgs[0][0].srcdata['ID']).cpu()
        # print(f"测试纯顺序读")
        # src_nodes = torch.arange(src_nodes[0], src_nodes.shape[0] + src_nodes[0], dtype = torch.int64, device = src_nodes.device)
        
        if (pre_src_nodes is not None):
            #只需要找出不一样的
            dis_nodes = src_nodes[~torch.isin(src_nodes, pre_src_nodes)]
        else:
            dis_nodes = src_nodes
        src_nodes_mem = load_mem(dis_nodes, True)
        pre_src_nodes = src_nodes
        # block_his.append(src_nodes)
        # block_his_len += src_nodes.shape[0]
        # print(f"block his总长度{block_his_len}, 占用{block_his_len * 4/ 1024**2}MB")
        
        left = right
        cur_batch += 1
        # print(f"当前batch: {cur_batch}:{train_edge_end//60000}采样花销: {sample_time:.7f}s gen block花销{gen_time:.7f}s")

        emptyCache()


init_memory(172, 24575383)
asdasdas = 1
# test()