
import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler import *


d = 'WIKI'
batch_size = 600
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

sample_param, memory_param, gnn_param, train_param = parse_config('./config/TGN.yml')
sample_param['layer'] = 2
sample_param['neighbor'] = [10, 10]
sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

#测试全批采样
# total_batch_node = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
# total_batch_ts = torch.empty(0, dtype = torch.float32, device = 'cuda:0')

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
    break

srcr,dstr,outts,outeid,root_nodes,root_ts = ret_list[1]
mask = srcr > -1
src = srcr[mask]
dst = dstr[mask]
eid = outeid[mask]

ret = ret_tgl[1]
src_t, dst_t, eid_t, dts = ret.col(), ret.row(), ret.eid(), ret.dts()[300:]

print(f"src_t: {src_t[:10]} \ndst_t: {dst_t[:10]}, \neid_t: {eid_t[:10]}\ndts:{dts[:10]}")
print(f"\nsrc: {src[:10]}\ndst:{dst[:10]}\neid:{eid[:10]}")

eid_t = torch.from_numpy(eid_t).cuda()
eid_g = eid
dis = torch.nonzero(eid_g != eid_t[:eid_g.shape[0]]).reshape(-1)

print(f"\ntgl sample边个数: {len(src_t)}, gpu sample边个数: {src.shape}, 不同边个数{dis}")



b = mfgs[2][0]
b1 = mfgs[1][0]

seed_num = root_nodes.shape[0]
fan_num = 10

src,dst,outts,outeid,root_nodes,root_ts = ret_list[0]
src1,dst1,outts1,outeid1,root_nodes1,root_ts1 = ret_list[1]
src2,dst2,outts2,outeid2,root_nodes2,root_ts2 = ret_list[2]

mask = src>-1
table = ((torch.arange(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(-1, fan_num)) / fan_num).to(torch.int64)

dst_node = table[mask].to(torch.int32)
# table[mask]可以直接作为0-200的dst节点 ,souce_nodes作为节点id

# src[mask]中，每个值都是独立的节点编号，因此直接从200开始arange即可， 而节点id就直接拿src[mask]
src_table = src[mask]
src_node = torch.arange(src_table.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

#nodes为所有节点的id，src的节点前面拼dst的节点，id的话，dst节点id就是source_nodes
nodes = torch.cat((root_nodes, src_table))
tss = torch.cat((root_ts, outts[mask]))

b = dgl.create_block((src_node, dst_node), num_src_nodes = nodes.shape[0], num_dst_nodes = root_nodes.shape[0])
b.srcdata['ID'] = nodes
b.srcdata['ts'] = tss

outdts = root_ts[table][mask] - outts[mask]
b.edata['dt'] = outdts
b.edata['ID'] = outeid[mask]

eid = outeid[mask][:10]
ts = outts[mask][:10]
src = src[mask][:10]
dst = dst[mask][:10]
dts = outdts[:10]

b_src = b.edges()[0][:10].to(torch.int64)
b_dst = b.edges()[1][:10].to(torch.int64)
b_n = b.ndata['ID']['_N']
b_eid = b.edata['ID'][:10]
print(b_n[b_src])
print(b_dst)
print(b_eid)

for i in range(10):
    print(b.edges[i])

b = sampler.gen_block(ret, root_nodes, root_ts)


# nodeTable = torch.unique(root_nodes)
# uniTable = torch.zeros_like(nodeTable).to(torch.int32).cuda()
# tmp_del,root_nodes_uni,uniTable = dgl.mapByNodeSet(nodeTable,uniTable,root_nodes,root_nodes)

src_table = src
src = torch.arange(src_table.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

b = dgl.create_block((src, dst), num_src_nodes = src.shape[0], num_dst_nodes = seed_num)

b.srcdata['ID'] = uniTable
b.edata['dt'] = outdts #这个是dt不是ts

#src需要修改...src中有几个节点就是几个节点，没有一个是重复的，即使采样到了同一个节点，由于时间戳不同，也应当是不同的节点，但是src nodeID是一致的
b.srcdata['ts'] = ts


for i in range(100):
    start = time.time()
    sampler.sample(total_batch_node, total_batch_ts)
    print(f"第{i}次全批采样源节点个数: {total_batch_node.shape} 时间: {time.time() - start:.4f}s")


# 测试tgl采样结果
import numpy as np

ret = np.load("/raid/guorui/workspace/dgnn/tgl/ret0.npz")
row = ret['row']
col = ret['col']
eid = ret['eid']
ts = ret['ts']
nodes = ret['nodes']
dts = ret['dst']
dim_in = ret['dim_in']
dim_out = ret['dim_out']



#生成table
import torch
import time
seed_num = 1800000
fan_num = 10

start = time.time()
table = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')
ar = torch.arange(seed_num, dtype = torch.int32, device = 'cuda:0')
for i in range(fan_num):
    table[i * seed_num: (i+1) * seed_num] = ar
table = table.reshape(fan_num, -1).T
print(f"用时{time.time() - start}")




import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler import *


d = 'WIKI'
batch_size = 600
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)


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
    break
