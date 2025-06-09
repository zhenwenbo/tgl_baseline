


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


d = 'TALK'
batch_size = 60000
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))

if (d=='MAG'):
    train_edge_end = 1111701860
    val_edge_end = 1198206616
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    # group_indexes = np.array(df[:train_edge_end].index // batch_size)



from sampler.sampler_gpu import *
fan_nums = [10,10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

sample_param, memory_param, gnn_param, train_param = parse_config('/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
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

num_nodes = (g['indptr'].shape[0] - 1)
neg_link_sampler = NegLinkSampler(num_nodes)

max_edge_num = 0
max_node_num = 0
left = 0
right = 0
cur_batch = 0

while True:
    
    
    if (right == train_edge_end):
        break
    right += batch_size
    right = min(train_edge_end, right)
    rows = df[left:right]
    # break

    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    root_nodes = torch.from_numpy(root_nodes).cuda()
    root_ts = torch.from_numpy(root_ts).cuda()
    start = time.time()

    sample_start = time.time()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=False)
    sample_time = time.time() - sample_start

    src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]
    cur_edge_num = outeid[outeid > -1].shape[0]
    cur_node_num = torch.unique(torch.cat((root_nodes, src[src>-1]))).shape[0]
    print(f"batch: {cur_batch} 不裁剪 {cur_node_num}-{cur_edge_num}")


    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    root_nodes = torch.from_numpy(root_nodes).cuda()
    root_ts = torch.from_numpy(root_ts).cuda()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=True)
    src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]
    cur_edge_num = outeid[outeid > -1].shape[0]
    cur_node_num = torch.unique(torch.cat((root_nodes, src[src>-1]))).shape[0]

    print(f"batch: {cur_batch} 做裁剪 {cur_node_num}-{cur_edge_num}")


    
    left = right
    cur_batch += 1

    asdasdasd = 1