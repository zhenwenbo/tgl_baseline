import numpy as np

ret0 = np.load('./ret0.npz')
ret1 = np.load('./ret1.npz')
g = np.load('DATA/{}/ext_full.npz'.format('WIKI'))

ret = ret0
row = ret['row']
col = ret['col']
eid = ret['eid']
rts = ret['ts']

indptr = g['indptr']
indices = g['indices']

import dgl
import dgl.function as fn
import torch
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
# g.ndata['x'] = torch.ones(5, 2)
g.edata['e'] = torch.ones(4,2)
g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'x'))
g.ndata['x']

import torch

def count_judge(src_node, dst_node):
    count = 0
    cur_node = torch.empty(0).to(torch.int32)
    for i,node in enumerate(src_node):
        if (torch.nonzero(cur_node == node).shape[0] > 0):
            continue
        cur_node = torch.cat((cur_node, torch.tensor([node], dtype = torch.int32)), dim = 0)

        #判断node在后面出现的次数    
        # print(i)
        #判断src_node中值等于node的个数，要求索引大于i
        indices = torch.nonzero(src_node[i + 1:] == node).reshape(-1)
        len1 = src_node[i+1:][indices].shape[0]

        indices = torch.nonzero(dst_node[i + 1:] == node).reshape(-1)
        len2 = dst_node[i+1:][indices].shape[0]

        
        if (len1 == 0 and len2 == 0):
            print(f"node: {node}在后面从未出现过...")
        else:
            print(f"node: {node}在src_node后面总共出现了{len1}次,在dst_node后面共出现了{len2}次")


src_node = torch.randint(0,100,(1,100)).reshape(-1).to(torch.int32)
dst_node = torch.randint(0,100,(1,100)).reshape(-1).to(torch.int32)
count_judge(src_node, dst_node)