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
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    # group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler.sampler_gpu import *
fan_nums = [10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

# dgl.sumDegree()

total_src = datas['src'].pin_memory().to(torch.int32)
total_dst = datas['dst'].pin_memory().to(torch.int32)
total_time = datas['time'].pin_memory().to(torch.float32)
total_eid = datas['eid'].pin_memory().to(torch.int32)

inDegree_idx = loadBin("/raid/guorui/DG/dataset/MAG/inDegreeSort_idx.bin")
idx1000 = inDegree_idx[:10000000]

count = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
count_sort, count_idx = torch.sort(count, descending=True)


preIdx = None
left = 996600000
right = 996600000
batch_num = 0
total_res = 0
pre_nid = None
# 统计前1一亿条边采样出正桶后，出现次数最多的1000w个节点？然后重新判断正桶实际需要塞入的特征
count = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
batch_count = []
while True:
    right += batch_size
    right = min(train_edge_end, right)
    if (left >= right):
        break
    first_flag = False

    src = total_src[left: right]
    dst = total_dst[left: right]
    times = total_time[left: right]
    eid = total_eid[left: right]
    root_nodes = torch.cat((src, dst)).cuda()
    root_ts = torch.cat((times, times)).cuda()
    # root_nodes = torch.from_numpy(root_nodes).cuda()
    # root_ts = torch.from_numpy(root_ts).cuda()

    # eids = torch.from_numpy(rows['Unnamed: 0']).to(torch.int32).cuda()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
    # print(f"采样用时: {time.time() - start:.8f}s")
    eid_uni = eid.to(torch.int32).cuda()
    nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()
    
    
    for ret in ret_list:
        #找出每层的所有eid即可
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret
        eid = outeid[outeid > -1]

        cur_eid = torch.unique(eid)
        eid_uni = torch.cat((cur_eid, eid_uni))
        eid_uni = torch.unique(eid_uni)
    # print(f"t1: {time.time() - start:.8f}s")
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
    batch_count.append(nid_uni)
    count[nid_uni.long()] += 1
    if (pre_nid is not None):
        count[nid_uni[torch.isin(nid_uni, pre_nid)].long()] += 1
    else:
        count[nid_uni.long()] += 1
    left = right
    batch_num += 1

    isinNum = torch.sum(torch.isin(nid_uni, idx1000))
    print(f"{batch_num}-{left}:{right} node num:{nid_uni.shape[0]} 在前1000w中的有{isinNum}个, 不在的{nid_uni.shape[0] - isinNum}")
    pl = 50

    if (batch_num % pl == 0):
        print(f"开始重新判断，取当前的一亿条边构成的正桶，找出这{pl}个正桶的最频繁出现的1000w个节点，并将其视为缓存不需要以正桶形式存储")
        count_sort, count_idx = torch.sort(count, descending=True)
        idx1000 = count_idx[:10000000]
        if (preIdx is not None):
            idxIsIn = torch.isin(idx1000, preIdx)
            idxIsInSum = torch.sum(idxIsIn)
            print(f"增量加载{10000000 - idxIsInSum}个节点，共{(10000000 - idxIsInSum) * 768 * 4 / 1024 ** 2}MB")
        print(f"{idx1000}, shape: {idx1000.shape[0]}")

        res = []
        pre_batch_info = None
        for batch_info in batch_count:
            if (pre_batch_info is not None):
                batch_info = batch_info[torch.isin(batch_info, pre_batch_info, invert = True)]
            isInNum = torch.sum(torch.isin(batch_info, idx1000))
            # print(f"node num:{batch_info.shape[0]} 不存在缓存中的: {batch_info.shape[0] - isInNum}")
            res.append(batch_info.shape[0] - isInNum)

            pre_batch_info = batch_info

        res = torch.tensor(res, dtype = torch.int32)
        print(f"平均单个正桶节点存储{torch.mean(res.float()) * 4 * 768 / 1024 ** 2 :.2f}MB 最大{torch.max(res) * 4 * 768 / 1024 ** 2 :.2f}MB 当前{pl * batch_size}条边带来的总正边存储{torch.sum(res) * 4 * 768 / 1024 ** 3 :.2f}GB")
        total_res += torch.sum(res) * 4 * 768 / 1024 ** 3
        count = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
        batch_count = []
        preIdx = idx1000.clone()

        print(f"当前执行{right}条边，构建正桶所需存储为{total_res}GB")

    pre_nid = nid_uni