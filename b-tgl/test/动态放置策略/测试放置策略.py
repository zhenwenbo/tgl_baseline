


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
        res.append(nid_uni.cpu())

        left = right
        batch_num += 1
        print(f"batch_num: {batch_num} over, node_num: {nid_uni.shape[0]}")
    
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


dis_ind = torch.isin(blocks, node_count_sort_ind[:1000000])
blocks[dis_ind] = -1
#将blocks中去除node_count_sort_ind前n项


def block2node(blocks, blocks_ptr, node_info):
    pre_ptr = 0
    block_count = 0
    for ptr in blocks_ptr:
        cur_block = blocks[pre_ptr:ptr]
        for cur in cur_block:
            if (cur == -1):
                continue
            if (node_info[cur] is None):
                node_info[cur] = [block_count]
            else:
                node_info[cur].append(block_count)
        block_count += 1
        pre_ptr = ptr
node_info = [None] * node_num

block2node(blocks, blocks_ptr, node_info)

from datasketch import MinHash, MinHashLSH
def create_minhash(data):
    minhash = MinHash(num_perm=128)  # num_perm 是哈希函数的数量，可以根据需要调整
    for d in data:
        minhash.update(d)
    return minhash


class EdgeHashTable:
    def __init__(self, table_size = 100000007):
        # 选择一个接近 1.5 到 2 倍预期元素数量的质数
        self.table_size = table_size
        self.table = [None] * self.table_size
    
    def _hash(self, x, y):
        # 生成 64 位组合键
        key = ((x << 32) | y) % self.table_size
        # 计算哈希值
        return key
    
    def insert(self, x, y):
        index = self._hash(x, y)
        if self.table[index] is None:
            self.table[index] = []
        if (len(self.table[index]) > 0):
            for cur in self.table[index]:
                # print(cur)
                if ((cur[0] == x and cur[1] == y) or (cur[0] == y and cur[1] == x)):
                    cur[2] += 1
                    break
        else:
            self.table[index].append((x, y, 1))
    
    def get(self, x, y):
        index = self._hash(x, y)
        if self.table[index] is not None:
            for stored_x, stored_y, data in self.table[index]:
                if stored_x == x and stored_y == y:
                    return data
        return None

import numba
@numba.jit(nopython=True) 
def cal_block(blocks_ptr, blocks):
    pre_ptr = 0
    table_size = 100000007
    table = [[[-1,-1,-1]]] * table_size
    for ptr in blocks_ptr:
        # print(f"处理{pre_ptr}:{ptr}")
        cur_block = blocks[pre_ptr:ptr]
        for i in range(len(cur_block)):
            for j in range(len(cur_block)):
                if (cur_block[i] == -1 or cur_block[j] == -1):
                    continue
                x = cur_block[i]
                y = cur_block[j]
                index = ((x << 32) | y) % table_size

                if (len(table[index]) > 0):
                    for cur in table[index]:
                        # print(cur)
                        if ((cur[0] == x and cur[1] == y) or (cur[0] == y and cur[1] == x)):
                            cur[2] += 1
                            break
                else:
                    table[index].append([x, y, 1])

            # print(f"i:{i} 用时{time.time() - start_i:.6f}s")
            # print(i)
        pre_ptr = ptr
        print(ptr)
    return table

res = cal_block(blocks_ptr.numpy(), blocks.numpy())


for r in res:
    if (len(r) > 1):
        print(len(r))
import numba
@numba.jit(nopython=True)
def test():
    res = [[[-1,-1,-1]]] * 10
    for i in range(10):
        for j in range(10):
            res[i].append([j,j+1,j+2])
    return res
res = test()

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

