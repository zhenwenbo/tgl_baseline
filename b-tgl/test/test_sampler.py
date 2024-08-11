


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

def gen_table(seed_num):
    fan_num = 2
    start = time.time()
    table = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')
    ar = torch.arange(seed_num, dtype = torch.int32, device = 'cuda:0')
    for i in range(fan_num):
        table[i * seed_num: (i+1) * seed_num] = ar
    table = table.reshape(fan_num, -1).T
    # print(f"gen_table use time {time.time() - start:.5f}s")
    
    return table

d = 'TALK'
batch_size = 1000
df = pd.read_csv('/raid/gr/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/gr/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)



from sampler.sampler_gpu import *
fan_nums = [10,10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

sample_param, memory_param, gnn_param, train_param = parse_config('/home/gr/workspace/dgnn/b-tgl/config/TGN-1.yml')
sample_param['layer'] = len(fan_nums)
sample_param['neighbor'] = fan_nums
sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              1, 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
neg_link_sampler = NegLinkSampler(num_nodes)

for _, rows in df[:train_edge_end].groupby(group_indexes):  
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
    root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()


    start = time.time()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
    

    sampler.sample(root_nodes.cpu().numpy(), root_ts.cpu().numpy())
    ret_tgl = sampler.get_ret()

    emptyCache()
    
    
    for i in range(len(ret_list)):
            
        src,dst,outts,outeid,root_nodes,root_ts = ret_list[i]
        outeid = outeid[outeid > -1]
        t_col, t_row, t_ts, t_eid, t_nodes, t_dts = ret_tgl[i].col(), ret_tgl[i].row(), ret_tgl[i].ts(), ret_tgl[i].eid(), ret_tgl[i].nodes(), ret_tgl[i].dts()

        # print(root_nodes)
        # print(t_nodes)
        # print(root_ts)
        # print(t_ts)
        # print(src[:10])
        # print(dst[:10])
        # print(root_nodes[:10])
        print(f"第{i}层 tgl边数{t_eid.shape[0]} gpu边数{outeid.shape[0]}")
        asd = 1

