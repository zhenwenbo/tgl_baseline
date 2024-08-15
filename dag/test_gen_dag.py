

# 生成DAG




# 测试sample结果
# GPU采样与TGL采样




import torch
import dgl
import numpy as np
import pandas as pd
import time


d = 'TALK'
batch_size = 600
df = pd.read_csv('/raid/gr/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/gr/DG/dataset/{}/ext_full.npz'.format(d))
# train_edge_end = df[df['ext_roll'].gt(0)].index[0]
# val_edge_end = df[df['ext_roll'].gt(1)].index[0]
# group_indexes = np.array(df[:train_edge_end].index // batch_size)


src = torch.from_numpy(np.concatenate([df.src.values]).astype(np.int32)).cuda()
dst = torch.from_numpy(np.concatenate([df.dst.values]).astype(np.int32)).cuda()
node_num = g['indptr'].shape[0] - 1
in_table = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
out_table = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')

dgl.sumDegree(in_table, out_table, src, dst)

res_src = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
res_dst = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
for i in range(node_num):
    cur_node = i

    node_src = torch.nonzero(src == cur_node).reshape(-1)
    node_dst = torch.nonzero(dst == cur_node).reshape(-1)

    edges = torch.cat((node_src, node_dst))
    edges = torch.unique(edges)

    edges_src = edges[:-1].to(torch.int32)
    edges_dst = edges[1:].to(torch.int32)
    
    res_src = torch.cat((res_src, edges_src))
    res_dst = torch.cat((res_dst, edges_dst))

res_num = res_src.shape[0]
in_table = torch.zeros(res_num, dtype = torch.int32, device = 'cuda:0')
out_table = torch.zeros(res_num, dtype = torch.int32, device = 'cuda:0')
dgl.sumDegree(in_table, out_table, res_src, res_dst)
