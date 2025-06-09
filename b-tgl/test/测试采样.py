



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


from sampler.sampler_gpu import *
fan_nums = [3]
layers = len(fan_nums)
sample_param, memory_param, gnn_param, train_param = parse_config('/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
sample_param['layer'] = len(fan_nums)
sample_param['neighbor'] = fan_nums



import dgl
import torch
indices = torch.tensor([2,3,3,5,9,2,3,3,3,5,5,3,6,7,8,9,10,5,3,3,3,4]).to(torch.int32).cuda()
totalts = torch.tensor([100,101,102,102,102,103,103,104,200,200,201,201,201,202,203,204,204,204,204,204,204,206]).to(torch.float32).cuda()
indptr = torch.tensor([0,8,21,22]).to(torch.int32).cuda()



sampleIDs = torch.tensor([0,0,1,1,1,1,1,2]).to(torch.int32).cuda()
sampleMask = torch.tensor([0,0,0,0,0,0,0,0]).to(torch.int32).cuda() #mask掉源节点，因为要做正边采样

curts = torch.tensor([101,103,201,100,203,204,205,206]).to(torch.float32).cuda()
totaleid = torch.arange(totalts.shape[0]).to(torch.int32).cuda()

sampler = ParallelSampler(indptr.cpu().numpy(), indices.cpu().numpy(), totaleid.cpu().numpy(), totalts.cpu().numpy(),
                              1, 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))



seed_num = sampleIDs.shape[0]
fan_num = 3
out_src = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')-1
out_dst = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')-1
outts = torch.zeros(seed_num * fan_num, dtype = torch.float32, device = 'cuda:0')-1
outeid = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')-1
dts = torch.zeros(seed_num * fan_num, dtype = torch.float32, device = 'cuda:0')-1

totaleid = totaleid.cpu().pin_memory().to(torch.int32)
totalts = totalts.cpu().pin_memory()
indices = indices.cpu().pin_memory()
NUM = dgl.sampling.sample_with_ts_recent(indptr,indices,curts,dts,totalts,totaleid,sampleIDs,sampleMask,seed_num,fan_num,out_src,out_dst,outts,outeid)
out_src = out_src.reshape(-1, fan_num)
out_dst = out_dst.reshape(-1, fan_num)
print(sampleIDs)
print(curts)
print(out_src)
print(out_dst)
print(dts)

mask = out_src>-1
eid = outeid[outeid>-1]
dts = dts[dts>-1]
outts = outts[outts>-1]

sampler.sample(sampleIDs.cpu().numpy(), curts.cpu().numpy())

ret_tgl = sampler.get_ret()

for i in range((1)):
        
    t_col, t_row, t_ts, t_eid, t_nodes, t_dts = ret_tgl[i].col(), ret_tgl[i].row(), ret_tgl[i].ts(), ret_tgl[i].eid(), ret_tgl[i].nodes(), ret_tgl[i].dts()

