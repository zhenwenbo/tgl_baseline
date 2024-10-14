


#首先需要一个预采样的东西获取每个block出现的节点


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

data = 'STACK'

g, df = load_graph(data)
if (data in ['BITCOIN']):
    train_edge_end = 86063713
    val_edge_end = 110653345
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

from sampler.sampler_gpu import *
sampler_gpu = Sampler_GPU(g, [10,10], 2, None)


def gen_part():
    #当分区feat不存在的时候做输出
    res = []
    node_count = torch.zeros(g['indptr'].shape[0], dtype = torch.int32)
    d = data
    # if os.path.exists(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}'):
    #     print(f"already  partfeat")
    #     return

    batch_size = 60000
    # node_feats, edge_feats = load_feat(d)

    df_start = 0
    df_end = train_edge_end

        
    group_indexes = np.array(df[df_start:df_end].index // batch_size)
    group_indexes -= group_indexes[0]
    left, right = df_start, df_start
    batch_num = 0
    while True:
    # for batch_num, rows in df[df_start:df_end].groupby(group_indexes):
        # emptyCache()
        right += batch_size
        right = min(df_end, right)
        if (left >= right):
            break
        rows = df[left:right]
        root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()

        # eids = torch.from_numpy(rows['Unnamed: 0']).to(torch.int32).cuda()
        start = time.time()
        ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
        nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()
        # nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
            
        #前面层出现的节点会在最后一层的dst中出现,因此所有节点就是最后一层的Src,dst
        ret = ret_list[-1]
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret
        del ret_list
        del outts, outeid, root_ts, dst
        # emptyCache()

        mask = src > -1
        src = src[mask]
        nid_uni = torch.cat((src, root_nodes))
        nid_uni = torch.unique(nid_uni)


        nid_uni,_ = torch.sort(nid_uni)
        node_count[nid_uni.long()] += 1
        res.append(nid_uni)

        left = right
        batch_num += 1
        print(f"batch_num: {batch_num} over, node_num: {nid_uni.shape[0]}")
    
    return res,node_count


block_info, node_count = gen_part()

node_count_sort,_ = torch.sort(node_count, descending=True)
print(node_count_sort[:100])

import time
start = time.time()
for i in range(len(block_info)):
    cur_start = time.time()
    cur = loadBinDisk(f'/raid/guorui/DG/dataset/{data}/node_features.bin', block_info[i])
    # print(f"单次: {time.time() - cur_start:.4f}s")

print(f"用时: {time.time() - start:.4f}s")

block_info_test = []
for i in range(len(block_info)):
    block_info_test.append(torch.arange(block_info[i].shape[0], dtype = torch.int32))

start = time.time()
for i in range(len(block_info)):
    cur_start = time.time()
    cur = loadBinDisk(f'/raid/guorui/DG/dataset/{data}/node_features.bin', block_info_test[i])
    # print(f"单次: {time.time() - cur_start:.4f}s")

print(f"用时: {time.time() - start:.4f}s")