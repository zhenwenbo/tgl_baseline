


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

data = 'BITCOIN'

g, df = load_graph(data)
if (data in ['BITCOIN']):
    train_edge_end = 86063713
    val_edge_end = 110653345
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

from sampler.sampler_gpu import *
sampler_gpu = Sampler_GPU(g, [10,10], 2, None)
# sampler_gpu = Sampler_GPU(g, [10], 1, None)

e_src = loadBin(f'/raid/guorui/DG/dataset/{data}/df-src.bin')
e_dst = loadBin(f'/raid/guorui/DG/dataset/{data}/df-dst.bin')

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

    pre_root_nodes = None

    while True:
    # for batch_num, rows in df[df_start:df_end].groupby(group_indexes):
        # emptyCache()
        right += batch_size
        right = min(df_end, right)
        if (left >= right):
            break
        rows = df[left:right]
        root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()

        start = time.time()
        nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()

        nid_uni = torch.unique(nid_uni)
        nid_uni,_ = torch.sort(nid_uni)

        node_count[nid_uni.long()] += 1
        

        left = right
        batch_num += 1
        print(f"batch_num: {batch_num} over, node_num: {nid_uni.shape[0]}")
        
        if (pre_root_nodes is not None):
            pre_root_nodes = pre_root_nodes[torch.isin(pre_root_nodes, nid_uni, invert = True)]
            print(f"pre_node_num: {pre_root_nodes.shape[0]}")
            res.append(pre_root_nodes.cpu())

        pre_root_nodes = nid_uni.clone()

    res.append(nid_uni.cpu())
    return res,node_count


block_info, node_count = gen_part()
node_num = node_count.shape[0]


import numba
from numba.typed import List


# block_info_nb = List(block_info)
# node_info_nb = List(node_info)
# block2node(block_info, node_info)

node_count_sort,node_count_sort_ind = torch.sort(node_count, descending=True)
saveBin(node_count_sort_ind[:1000000], f'/raid/guorui/DG/dataset/{data}/pre_1000000.bin')
print(node_count_sort[:100])


# 通过e_src和e_dst 计算度数
t_src = e_src[:train_edge_end].cuda()
t_dst = e_dst[:train_edge_end].cuda()
node_num = node_count.shape[0]
inDegree = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
outDegree = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
dgl.sumDegree(inDegree, outDegree,t_src, t_dst)
d = inDegree + outDegree

print(f"d < 2: {torch.sum(d < 2)} d < 1:{torch.sum(d<1)} count<2:{torch.sum(node_count<2)} count<1:{torch.sum(node_count<1)}")
inf1 = torch.sum(node_count < 1)
inf2 = torch.sum((torch.logical_and(node_count < 2, node_count > 0)))
inf3 = torch.sum((torch.logical_and(node_count < 4, node_count > 0)))
inf4 = torch.sum((torch.logical_and(node_count < 6, node_count > 0)))
inf5 = torch.sum((torch.logical_and(node_count < 8, node_count > 0)))
print(f"count < 1的放一起有{inf1}  count大于0小于2/4/6/8的放一起有{inf2} {inf3} {inf4} {inf5}")
print(f"节点访问总数{torch.sum(node_count)} 前10w,50w,100w个节点做缓存可以消除 {torch.sum(node_count_sort[:100000])} {torch.sum(node_count_sort[:500000])} {torch.sum(node_count_sort[:1000000])}")

# count高的和入度高的重合率
l = [100000,500000,1000000]
inDegree_sort, inDegree_sort_ind = torch.sort(inDegree, descending=True)
d_sort, d_sort_ind = torch.sort(d, descending=True)
for li in l:
    count_ind = node_count_sort_ind[:li]
    inde_ind = d_sort_ind[:li]
    
    ii = torch.isin(count_ind, inde_ind.cpu())
    print(f"l: {li}, count在ind: {torch.sum(ii)}")



# 计算block_info总长度
total_len = 0
blocks_ptr = []
for i in range(len(block_info)):
    total_len += block_info[i].shape[0]
    blocks_ptr.append(block_info[i].shape[0])
print(f"总节点个数:{total_len}, {total_len * 4 / 1024 ** 3:.2f}GB")
blocks = torch.cat(block_info)
blocks_ptr = torch.tensor(blocks_ptr, dtype = torch.int32)
blocks_ptr = torch.cumsum(blocks_ptr, dim = 0)

blocks_ptr = torch.cat((torch.zeros(1, dtype = torch.int32, device = blocks_ptr.device), blocks_ptr))
blocks_ptr_diff = torch.diff(blocks_ptr)
blocks_ind = torch.repeat_interleave(torch.arange(0, blocks_ptr_diff.shape[0]), blocks_ptr_diff)

dis_node = node_count_sort_ind[:1000000]
torch.save(dis_node, '/raid/guorui/DG/dataset/BITCOIN/node_cache_map.bin')
saveBin(dis_node, '/raid/guorui/DG/dataset/BITCOIN/node_cache_map.bin')
dis_ind = torch.isin(blocks, node_count_sort_ind[:1000000])
blocks = blocks[~dis_ind]
blocks_ind = blocks_ind[~dis_ind]

blocks_sort, blocks_sort_ind = torch.sort(blocks)
node_info = blocks_ind[blocks_sort_ind]

node = blocks_sort.cpu()
block = node_info.cpu()
torch.save(node, '/home/guorui/workspace/tmp/root_node.pt')
torch.save(block, '/home/guorui/workspace/tmp/root_block.pt')