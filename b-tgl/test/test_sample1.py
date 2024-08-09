


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




from sampler.sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)

sample_param, memory_param, gnn_param, train_param = parse_config('/home/gr/workspace/dgnn/b-tgl/config/TGAT-2.yml')
sample_param['layer'] = 2
sample_param['neighbor'] = [10, 10]

indptr = torch.tensor([0,0,0,6,6,6,6,7]).numpy()
indices = torch.tensor([1,2,3,4,5,6,7,0]).numpy()
eid = torch.tensor([1,2,3,4,5,6,7,8]).numpy()
ts = torch.tensor([1,2,3,4,5,6,7,8]).numpy()
g = {'indptr': indptr,'indices': indices,'eid':eid,'ts':ts}
sampler_gpu = Sampler_GPU(g, fan_nums, layers)


sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              8, 1, 1, [3],
                              True, sample_param['prop_time'],
                              1, float(sample_param['duration']))


# sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
#                               sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
#                               sample_param['strategy']=='recent', sample_param['prop_time'],
#                               sample_param['history'], float(sample_param['duration']))

root_nodes = torch.tensor([0,2,2])
root_ts = torch.tensor([2,5,6])
sampler.sample(root_nodes.cpu().numpy(), root_ts.cpu().numpy())
ret_tgl = sampler.get_ret()
t_col, t_row, t_ts, t_eid, t_nodes, t_dts = ret_tgl[0].col(), ret_tgl[0].row(), ret_tgl[0].ts(), ret_tgl[0].eid(), ret_tgl[0].nodes(), ret_tgl[0].dts()



d = 'WIKI'
batch_size = 2000
df = pd.read_csv('/raid/gr/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/gr/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

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
    
    if (_ == 2):
        break

src,dst,outts,outeid,root_nodes,root_ts = ret_list[0]
seed_num = root_nodes.shape[0]

mask = src>-1
# table = ((torch.arange(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(-1, fan_num)) / fan_num).to(torch.int64)

table = gen_table(seed_num).to(torch.int64)

dst_node = table[mask].to(torch.int32)
# table[mask]可以直接作为0-200的dst节点 ,souce_nodes作为节点id

# src[mask]中，每个值都是独立的节点编号，因此直接从200开始arange即可， 而节点id就直接拿src[mask]
src_table = src[mask]
src_node = torch.arange(src_table.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

#nodes为所有节点的id，src的节点前面拼dst的节点，id的话，dst节点id就是source_nodes
nodes = torch.cat((root_nodes, src_table))
tss = torch.cat((root_ts, outts[mask]))
t_col, t_row, t_ts, t_eid, t_nodes, t_dts = ret_tgl[0].col(), ret_tgl[0].row(), ret_tgl[0].ts(), ret_tgl[0].eid(), ret_tgl[0].nodes(), ret_tgl[0].dts()

print(src)

