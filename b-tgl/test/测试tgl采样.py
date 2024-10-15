


# 测试sample结果
# GPU采样与TGL采样



import sys

# root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
# if root_dir not in sys.path:
#     sys.path.append(root_dir)
# from sampler.sampler import *


root_dir = '/raid/guorui/workspace/dgnn/simple/SIMPLE'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from sampler import *

import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"设置随机种子{seed}")

set_seed(0)

d = 'TALK'
batch_size = 2000
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))

if (d=='MAG'):
    train_edge_end = 1111701860
    val_edge_end = 1198206616
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    # group_indexes = np.array(df[:train_edge_end].index // batch_size)


fan_nums = [10]
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

reverse = False

while True:
    
    
    if (right == train_edge_end):
        break
    right += batch_size
    right = min(train_edge_end, right)
    rows = df[left:right]
    # break

    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    print(root_ts)
    print(root_nodes)
    root_nodes = torch.from_numpy(root_nodes).cuda()
    root_ts = torch.from_numpy(root_ts).cuda()
    start = time.time()



    sampler.sample(root_nodes.cpu().numpy(), root_ts.cpu().numpy())
    ret_tgl = sampler.get_ret()
    
    # mfgs = sampler_gpu.gen_mfgs(ret_list)

    mfgs_tgl = to_dgl_blocks(ret_tgl, sample_param['history'])

    b2 = mfgs_tgl[0][0]
    print(f"node num: {b2.num_nodes()} edge num: {b2.num_edges()}")
    asdasd = 1