import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os



block_size = 60000
g, df = load_graph('STACK')


from sampler.sampler_gpu import *
sampler_gpu = Sampler_GPU(g, [10,10], 2, None)
neg_sampler = ReNegLinkSampler(g['indptr'].shape[0] - 1, 0.9)

cur_neg_nodes = None
def pre_neg_sample(block_num):
    global g,df,neg_sampler,sampler_gpu,cur_neg_nodes
    #需要根据self.cur_batch判断从何处开始计算
    time_pre_neg_s = time.time()
    #边数 * 0.7
    train_edge_end = int(g['indptr'][-1] * 0.7)
    start = block_num * 60000
    end = min(train_edge_end, (block_num + 1) * 60000)
    if(start >= end):
        return None

    
    rows = df[start:end]
    
    neg_nodes = neg_sampler.sample(len(rows))
    cur_neg_nodes = torch.from_numpy(neg_nodes).cuda()
    # neg_nodes = np.arange(60000)
    root_nodes = torch.from_numpy(np.concatenate([neg_nodes]).astype(np.int32)).cuda()
    root_ts = torch.from_numpy(np.concatenate([rows.time.values]).astype(np.float32)).cuda()


    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
    src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]

    mask = src > -1
    src = src[mask]
    dst = dst[mask]
    nodes = torch.cat((src, root_nodes))
    nodes = torch.unique(nodes)

    eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    for ret in ret_list:
        #找出每层的所有eid即可
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret
        eid = outeid[outeid > -1]

        cur_eid = torch.unique(eid)
        eid_uni = torch.cat((eid_uni, cur_eid))
        eid_uni = torch.unique(eid_uni)
    
    return nodes, eid_uni

pre_list = []
window = 3

pre_range_node = None
pre_range_eid = None
for i in range(2000000):
    nodes,eid_uni = pre_neg_sample(i)
    nodes,_ = torch.sort(nodes)
    eid_uni,_ = torch.sort(eid_uni)
    pre_list.append([nodes,eid_uni])

    cur_nodet = torch.zeros_like(nodes, dtype = torch.int32, device = 'cuda:0') - 1
    cur_eidt = torch.zeros_like(eid_uni, dtype = torch.int32, device = 'cuda:0') - 1

    train_edge_end = int(g['indptr'][-1] * 0.7)
    start = i * 60000
    end = min(train_edge_end, (i + 1) * 60000)
    if(start >= end):
        break
    rows = df[start:end]

    range_node = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
    range_eid = torch.arange(start, end, dtype = torch.int32, device = 'cuda:0')

    if (pre_range_node is not None):
        cur_nodet[torch.isin(nodes, pre_range_node)] = 1
        cur_eidt[torch.isin(eid_uni, pre_range_eid)] = 1
    
    pre_range_node = range_node
    pre_range_eid = range_eid

    cur_nodet[torch.isin(nodes, range_node)] = 1
    cur_eidt[torch.isin(eid_uni, range_eid)] = 1

    node_n = torch.sum(cur_nodet == 1)
    edge_n = torch.sum(cur_eidt == 1)
    print(f"节点总数{nodes.shape[0]} 直接命中{node_n} 未命中{nodes.shape[0] - node_n}; 边总数{eid_uni.shape[0]} 直接命中{edge_n} 未命中{eid_uni.shape[0] - edge_n}")
    # 窗口为3
    if (len(pre_list) > window + 1):
        for i in range(1):
            pre_node, pre_eid = pre_list[0 - 2 - i]
            
            pre_nodet = torch.zeros_like(pre_node, dtype = torch.int32, device = 'cuda:0') - 1
            pre_eidt =  torch.zeros_like(pre_eid, dtype = torch.int32, device = 'cuda:0') - 1

            cur_nodet[torch.isin(nodes, pre_node)] = 1
            cur_eidt[torch.isin(eid_uni, pre_eid)] = 1
            
            # node_num = torch.
            node_n = torch.sum(cur_nodet == 1)
            edge_n = torch.sum(cur_eidt == 1)
            print(f"第{i}个 节点总数{nodes.shape[0]} 直接命中{node_n} 未命中{nodes.shape[0] - node_n}; 边总数{eid_uni.shape[0]} 直接命中{edge_n} 未命中{eid_uni.shape[0] - edge_n}")

            miss_ind = torch.nonzero(cur_eidt != 1).reshape(-1)
            assad = 1

            
        
        # 测试兑掉原始负节点后节点命中率
        cur_nodet[torch.isin(nodes, cur_neg_nodes)] = 1
        node_n = torch.sum(cur_nodet == 1)
        edge_n = torch.sum(cur_eidt == 1)
        print(f"兑掉原始负节点 节点总数{nodes.shape[0]} 直接命中{node_n} 未命中{nodes.shape[0] - node_n}; 边总数{eid_uni.shape[0]} 直接命中{edge_n} 未命中{eid_uni.shape[0] - edge_n}")

