
#测试将数据集分割，得出增量加载的feat文件并保存到IO中
#正边用IO，负边用cpu？
import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
import os


d = 'BITCOIN'
gen_feat(d, 172, 172)
# path = f'/raid/guorui/DG/dataset/{d}'
# df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
# g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
# train_edge_end = df[df['ext_roll'].gt(0)].index[0]
# val_edge_end = df[df['ext_roll'].gt(1)].index[0]

# rand_de = 100
# rand_dn = 100
# if d == 'LASTFM':
#     edge_feats = torch.randn(1293103, rand_de)
# elif d == 'MOOC':
#     edge_feats = torch.randn(411749, rand_de)
# if rand_dn > 0:
#     if d == 'LASTFM':
#         node_feats = torch.randn(1980, rand_dn)


# batch_size = 600000
# group_indexes = np.array(df[:train_edge_end].index // batch_size)

# from sampler_gpu import *
# fan_nums = [10,10]
# layers = len(fan_nums)
# sampler = Sampler_GPU(g, fan_nums, layers)



class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
neg_link_sampler = NegLinkSampler(num_nodes)

from feat_buffer import *
feat_buffer = Feat_buffer(d,g,df,{'batch_size': 600},train_edge_end, 600000//600, sampler, neg_link_sampler)

feat_buffer.gen_part()

pre_eid = []

for batch_num, rows in df[:train_edge_end].groupby(group_indexes):
    emptyCache()
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
    root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()


    start = time.time()
    ret_list = sampler.sample_layer(root_nodes, root_ts)
    eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    # emptyCache()

    for ret in ret_list:
        #找出每层的所有eid即可
        src,dst,outts,outeid,root_nodes,root_ts = ret
        eid = outeid[outeid > -1]

        cur_eid = torch.unique(eid)
        eid_uni = torch.cat((cur_eid, eid_uni))
        eid_uni = torch.unique(eid_uni)

    #处理这个eid_uni，抽特征然后存就行。这里eid是个全局的
    #存起来后需要保存一个map，map[i]表示e_feat[i]保存的是哪条边的特征即eid
    cur_edge_feat = edge_feats[eid_uni.to(torch.int64)]
    saveBin(cur_edge_feat.cpu(), path + f'/part/part{batch_num}_edge_feat.bin')
    saveBin(eid_uni.cpu(), path + f'/part/part{batch_num}_edge_map.bin')

    sampleTime = time.time() - start
    # mfgs = sampler.gen_mfgs(ret_list)
    
    print(f"{root_nodes.shape}单层单block采样 + 转换block batch: {batch_num} batchsize: {batch_size} 纯采样用时{sampleTime:.7f}s 用时:{time.time() - start:.7f}s")
 
    # if (batch_num == 30):
    #     break

from feat_buffer import *
feat_buffer = Feat_buffer(d,g,df,{'batch_size': 600},train_edge_end, 600000//600, sampler, neg_link_sampler)

emptyCache()


for i, eid in enumerate(eids):

    eid_sort, _ = torch.sort(eid)
    print(torch.nonzero(eid != eid_sort).shape)


    pre_eid = torch.empty(0)
    #下面做增量加载优化
    if (i > 0):
        pre_eid = eids[i - 1]

        eid_sort,_ = torch.sort(eid)
        pre_sort,_ = torch.sort(pre_eid)
        table1 = torch.zeros_like(eid_sort)
        table2 = torch.zeros_like(pre_sort)
        dgl.findSameNode(eid_sort, pre_sort, table1, table2)
        edge_feat_mem = eid.shape[0] * 128 * 4 / (1024 ** 3)

        print(f"batch:{i} edge: {eid.shape[0]}, pre_batch:{pre_eid.shape[0]}  same:{table1[table1 > 0].shape[0]}\
               {table1[table1 > 0].shape[0] / eid.shape[0] * 100:.2f}%, cur edge feat: {edge_feat_mem:.2f}GB")

eid_uni = torch.unique(eid)
print(f"batch_size: {batch_size}, eid: {eid.shape}, uni_eid: {eid_uni.shape}")
print(f"edge feat占显存{eid_uni.shape[0] * 128 * 4 / (1024 ** 3):.2f}GB")



