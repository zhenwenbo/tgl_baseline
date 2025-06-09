
import torch
import dgl
import numpy as np
import pandas as pd
import time
def repeat(tensor, times):
    for i in range(times):
        tensor = torch.cat((tensor,tensor))
    return tensor


d = 'WIKI'

batch_size = 200
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // 200)



class Sampler_GPU:
    def __init__(self, g, fan_num):
        self.indptr = torch.from_numpy(g['indptr']).cuda().to(torch.int32)
        self.indices = torch.from_numpy(g['indices']).cuda().to(torch.int32)
        self.totalts = torch.from_numpy(g['ts']).cuda().to(torch.float32)
        self.totaleid = torch.from_numpy(g['eid']).cuda().to(torch.int32)
        self.fan_num = fan_num

    def sample(self, sampleIDs, curts):
        seed_num = sampleIDs.shape[0]
        out_src = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        out_dst = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        outts = torch.zeros(seed_num * self.fan_num, dtype = torch.float32, device = 'cuda:0')-1
        outeid = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1

        NUM = dgl.sampling.sample_with_ts_recent(self.indptr,self.indices,curts,self.totalts,self.totaleid,sampleIDs,seed_num,self.fan_num,out_src,out_dst,outts,outeid)

        return (out_src.reshape(seed_num, -1), out_dst.reshape(seed_num, -1), outts.reshape(seed_num, -1), outeid.reshape(seed_num, -1))

sampler = Sampler_GPU(g, 10)

#测试全批采样
total_batch_node = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
total_batch_ts = torch.empty(0, dtype = torch.float32, device = 'cuda:0')

for _, rows in df[:train_edge_end].groupby(group_indexes):  
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
    ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()

    total_batch_node = torch.cat((total_batch_node, root_nodes))
    total_batch_ts = torch.cat((total_batch_ts, ts))
    # root_nodes = root_nodes[:2]
    print(f"root_nodes: {root_nodes}")

    (src, dst, ts, eid) = sampler.sample(root_nodes, ts)
    print(src[:10])
    print(dst[:10])
    print(ts)

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