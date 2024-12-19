


#首先需要一个预采样的东西获取每个block出现的节点


import torch
import numpy as np
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

total_start = time.time()
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='STACK')
parser.add_argument('--pre_sample_size', type=int, default=60000, help='pre sample size')
args=parser.parse_args()


data = 'TALK'

g, datas, df_conf = load_graph_bin(data)

train_edge_end = df_conf['train_edge_end']
val_edge_end = df_conf['val_edge_end']
e_src = datas['src']
e_dst = datas['dst']

batch_size = args.pre_sample_size
def gen_part():
    #当分区feat不存在的时候做输出
    res = []
    node_count = torch.zeros(g['indptr'].shape[0], dtype = torch.int32)

    df_start = 0
    df_end = train_edge_end
    left, right = df_start, df_start
    batch_num = 0

    pre_root_nodes = None

    while True:
    # for batch_num, rows in df[df_start:df_end].groupby(group_indexes):
        # emptyCache()
        right += batch_size
        right = min(df_end, right)
        if (left >= right):
            break
        # rows = df[left:right]
        root_nodes = torch.from_numpy(np.concatenate([e_src[left:right], e_dst[left:right]]).astype(np.int32)).cuda()

        start = time.time()
        nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()

        nid_uni = torch.unique(nid_uni)
        nid_uni,_ = torch.sort(nid_uni)

        node_count[nid_uni.long()] += 1
        

        left = right
        batch_num += 1
        # print(f"batch_num: {batch_num} over, node_num: {nid_uni.shape[0]}")
        
        if (pre_root_nodes is not None):
            pre_root_nodes = pre_root_nodes[torch.isin(pre_root_nodes, nid_uni, invert = True)]
            # print(f"pre_node_num: {pre_root_nodes.shape[0]}")
            res.append(pre_root_nodes.cpu().numpy())

        pre_root_nodes = nid_uni.clone()

    res.append(nid_uni.cpu().numpy())
    return res,node_count


block_info, node_count = gen_part()
node_num = node_count.shape[0]
import numba as nb

def gen_intervals(uni_list, num_data):
    #initialize
    n = len(uni_list)
    last_flag = np.zeros(num_data,dtype=np.bool_)
    last_used = np.ones(num_data,dtype=int) * n
    start, end, IDs, interval_weight = [], [], [], []
    interval_weight_ = []
    #start gen intervals
    for i in range(n):
        #analyze the current batch 这个节点在当前批内出现了几次
        uni = uni_list[i]

        #generate intervals
        interval_flag = last_flag[uni] == True
        target_ID = uni[interval_flag]
        target_last_used = last_used[target_ID] 
        len_interval = i - target_last_used 
        net_weight = 1 / len_interval 
        #append
        start.extend(target_last_used) #start
        end.extend([i]*len(target_last_used)) #end
        IDs.extend(target_ID)
        interval_weight.extend(net_weight) 
        #update
        last_used[uni] = i
        last_flag[uni] = 1
    return start, end, IDs, interval_weight

def transform_intervals(start, end, IDs, interval_weight):
    start = np.array(start,dtype=np.int32)
    end = np.array(end,dtype=np.int32)
    IDs = np.array(IDs, dtype=np.int32)
    interval_weight = np.array(interval_weight)
    weight_order = np.argsort(interval_weight)[::-1]
    start = start[weight_order]
    end = end[weight_order]
    IDs = IDs[weight_order]
    return start, end, IDs

@nb.njit()
def select_interval_ID(start, end, num_batch, threshold):
    sel_interval_ID = []
    budget = np.ones(num_batch-1,dtype=np.int32)*threshold
    for i in range(len(start)):
        start_, end_ = start[i], end[i]
        flag = (budget[start_:end_]<1).any()
        if not flag:
            budget[start_:end_] -= 1
            sel_interval_ID.append(i)
        else:
            continue
    return sel_interval_ID

def test_saved_ratio(start, end, interval_weight):
    saved = np.sum(interval_weight)
    print('total saved:', saved)

def select_interval(sel_ID, start, end, IDs):
    sel_ID = np.array(sel_ID)
    start = start[sel_ID]
    end = end[sel_ID]
    IDs = IDs[sel_ID]
    
    return start, end, IDs


def sort_interval_by_time(start, end, IDs):
    order = np.argsort(start)
    start = start[order]
    end = end[order]
    IDs = IDs[order]
    return start, end, IDs


def data_placement_single(uni_list, num_data, num_batch, limit):
    #邻域节点表，邻域per-batch num表,节点总个数，batch总个数，显存预算
    t0 = time.time()
    start, end, IDs, interval_weight = gen_intervals(uni_list, num_data)
    t1 = time.time()
    print('Interval generation takes:{:.2f}s'.format(t1-t0))

    #根据weight进行排序。start为节点起始batch，end为结束batch,ID为节点ID
    start, end, IDs = transform_intervals(start, end, IDs, interval_weight)
    del interval_weight
    t2 = time.time()
    print('Interval transformation takes:{:.2f}s'.format(t2-t1))
    
    #逐个进行判断选择，如果能放就放
    sel_ID = select_interval_ID(start, end, num_batch, limit)
    t3 = time.time()
    print('Interval-ID selection takes:{:.2f}s'.format(t3-t2))
    #sel_ID表示选择的interval的ID
    if len(sel_ID) != 0:
        start, end, IDs = select_interval(sel_ID, start, end, IDs) #提炼出被选择的区间
        del sel_ID
    t4 = time.time()
    print('Interval selection takes:{:.2f}s'.format(t4-t3))
    
    if len(start) != 0:
        start, end, IDs = sort_interval_by_time(start, end, IDs)
    t5 = time.time()
    print('Interval sorting takes:{:.2f}s'.format(t5-t4))
    
    return start, end, IDs
    

start,end,IDs = data_placement_single(block_info, node_num, len(block_info), 1000000)

def gen_batch_plan_tensor(start, end, IDs, batch_id):
    flag = (start <= batch_id) & (end > batch_id)
    return IDs[flag].long()

def pre_load_all(start, end, IDs, num_batch, to_gpu):
    if to_gpu:
        batch_plan = []
        for i in range(num_batch):
            batch_plan.append(gen_batch_plan_tensor(start, end, IDs, i).cpu())

    return batch_plan

start = torch.from_numpy(start).cuda()
end = torch.from_numpy(end).cuda()
IDs = torch.from_numpy(IDs).cuda()
batch_plan = pre_load_all(start,end,IDs,len(block_info), True)

saveBin(start, f'/raid/guorui/DG/dataset/{data}/simple_start-{batch_size}.bin')
saveBin(end, f'/raid/guorui/DG/dataset/{data}/simple_end-{batch_size}.bin')
saveBin(IDs, f'/raid/guorui/DG/dataset/{data}/simple_IDs-{batch_size}.bin')


def gen_simple(start, end, IDs):
    # 返回start_cache_ind: 表示新的root_nodes中需要放入缓存的<索引>
    # start_flush_ind: 表示新的root_nodes中需要直接刷入硬盘的<索引>
    # end_flush_idx：表示需要刷回的cache中的<节点id>
    # 还需要返回配套的start_cache_ptr, start_flush_ptr, end_flush_ptr

    start_cache_ind = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    start_flush_ind = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    end_flush_idx = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    start_cache_ptr = torch.zeros(1, dtype = torch.int32, device = 'cuda:0')
    start_flush_ptr = torch.zeros(1, dtype = torch.int32, device = 'cuda:0')
    end_flush_ptr = torch.zeros(1, dtype = torch.int32, device = 'cuda:0')

    simple_node = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
    simple_block = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

    for i in range(len(block_info)):
        cur_start = IDs[start == i]
        cur_end_flush_idx = IDs[end == i]

        cur_root_nodes = torch.from_numpy(block_info[i]).cuda()
        # 找出 block_info中不在start部分的就是要直接刷回不缓存的
        cur_start_mask = torch.isin(cur_root_nodes, cur_start)
        cur_start_cache_ind = torch.nonzero(cur_start_mask).to(torch.int32).reshape(-1)
        cur_start_flush_ind = torch.nonzero(~cur_start_mask).to(torch.int32).reshape(-1)

        simple_node = torch.cat((simple_node, cur_root_nodes[cur_start_flush_ind.long()]))
        simple_block = torch.cat((simple_block, torch.zeros(cur_start_flush_ind.shape[0], device = 'cuda:0', dtype = torch.int32) + i))


        start_cache_ind = torch.cat((start_cache_ind, cur_start_cache_ind))
        start_flush_ind = torch.cat((start_flush_ind, cur_start_flush_ind))
        end_flush_idx = torch.cat((end_flush_idx, cur_end_flush_idx))

        start_cache_ptr = torch.cat((start_cache_ptr, torch.tensor([cur_start_cache_ind.shape[0]], dtype = torch.int32, device = 'cuda:0')))
        start_flush_ptr = torch.cat((start_flush_ptr, torch.tensor([cur_start_flush_ind.shape[0]], dtype = torch.int32, device = 'cuda:0')))
        end_flush_ptr = torch.cat((end_flush_ptr, torch.tensor([cur_end_flush_idx.shape[0]], dtype = torch.int32, device = 'cuda:0')))

    
    start_cache_ptr = torch.cumsum(start_cache_ptr, dim = 0)
    start_flush_ptr = torch.cumsum(start_flush_ptr, dim = 0)
    end_flush_ptr = torch.cumsum(end_flush_ptr, dim = 0)
 
    saveBin(start_cache_ind, f'/raid/guorui/DG/dataset/{data}/start_cache_ind-{batch_size}.bin')
    saveBin(start_flush_ind, f'/raid/guorui/DG/dataset/{data}/start_flush_ind-{batch_size}.bin')
    saveBin(end_flush_idx, f'/raid/guorui/DG/dataset/{data}/end_flush_idx-{batch_size}.bin')
    saveBin(start_cache_ptr, f'/raid/guorui/DG/dataset/{data}/start_cache_ptr-{batch_size}.bin')
    saveBin(start_flush_ptr, f'/raid/guorui/DG/dataset/{data}/start_flush_ptr-{batch_size}.bin')
    saveBin(end_flush_ptr, f'/raid/guorui/DG/dataset/{data}/end_flush_ptr-{batch_size}.bin')

    return simple_node, simple_block
simple_node, simple_block = gen_simple(start, end, IDs)

node, node_sort_ind = torch.sort(simple_node)
block = simple_block[node_sort_ind]

max_num = 0
for i in range(len(batch_plan)):
    max_num = max(max_num, batch_plan[i].shape[0])
saveBin(torch.tensor([max_num], dtype = torch.int32), f'/raid/guorui/DG/dataset/{data}/simple_max_num-{batch_size}.bin')



import torch
import numpy as np
import dgl

import math
import torch
import dgl
reorder_res = torch.empty(0, dtype = torch.int32)

def uni_simple(node, block, freq, reorder_res):
    #简单将出现p次的做完全去重
    count_labels = torch.zeros(torch.max(node), dtype = torch.int32, device = 'cuda:0')
    count_labels = count_labels[1:]
    dgl.bincount(node, count_labels)
    label_cum = torch.cumsum(count_labels, dim = 0)


    dis_node_idx = torch.nonzero(count_labels == freq).reshape(-1) #这个是count_labels的indices，实际就是node的id

    res = None

    for i in range(freq):
        if (res is None):
            res = block[label_cum[dis_node_idx] - freq].reshape(1,-1)
        else:
            res = torch.cat((res, block[label_cum[dis_node_idx] - freq + i].reshape(1,-1)), dim = 0)
    
    res = res.T
    res_uni, res_uni_inv,res_uni_count = torch.unique(res, return_inverse=True, return_counts = True, dim = 0)
    threshold = freq
    #没有重复超过2次的都去掉

    res_uni_inv_sort, res_uii = torch.sort(res_uni_inv) # 此处res_uii代表dis_node_idx中的node dis_node_idx[res_uii]就是节点idx

    res_uni_inv_label = torch.zeros(torch.max(res_uni_inv_sort), dtype = torch.int32, device = 'cuda:0')
    dgl.bincount(res_uni_inv_sort.to(torch.int32), res_uni_inv_label)

    res_uni_inv_threshold_ind = torch.nonzero(res_uni_inv_label < threshold).reshape(-1)
    res_uni_inv_threshold_mask = ~torch.isin(res_uni_inv_sort, res_uni_inv_threshold_ind) #表示res_uni_inv_sort中小于threshold的mask
    
    res_ret = dis_node_idx[res_uii[res_uni_inv_threshold_mask]] # 最终的node序列

    reorder_res = torch.cat((reorder_res, res_ret.cpu()))

    # 最后把node和block中已经构建好序列的node去掉
    node_reordered_mask = torch.isin(node, res_ret)
    node = node[~node_reordered_mask]
    block = block[~node_reordered_mask]

    return node, block, reorder_res


node, block, reorder_res = uni_simple(node, block, 1, reorder_res)


if (data == 'BITCOIN'):
    node_num = 24575383
elif (data == 'TALK'):
    node_num = 1140149
elif (data == 'STACK'):
    node_num = 2601977
elif (data == 'GDELT'):
    node_num = 16682
elif (data == 'LASTFM'):
    node_num = 1980
total_node = torch.arange(0, node_num + 1, dtype = torch.int32).cuda()
reorder_res = reorder_res.cuda()
dis_node = total_node[torch.isin(total_node, reorder_res, invert=True)]

nid = torch.cat((reorder_res, dis_node))

res = torch.zeros(node_num + 1, dtype=torch.long).cuda()
res.scatter_(0, nid, torch.arange(len(nid)).cuda())
import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
saveBin(res.cpu(), f'/raid/guorui/DG/dataset/{data}/node_simple_reorder_map-{batch_size}.bin')

path = f'/raid/guorui/DG/dataset/{data}'
def init_memory(dim_edge_feat, num_nodes):
    memory_param = {'dim_out': 100, 'mailbox_size': 1}
    mem_dim = 100
    mail_size = 1
    
    # memory = torch.randn((num_nodes, memory_param['dim_out']), dtype=torch.float32).reshape(num_nodes, -1)
    # memory_ts = torch.randn(num_nodes, dtype=torch.float32).reshape(num_nodes, -1)
    # mailbox = torch.randn((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32).reshape(num_nodes, -1)
    # mailbox_ts = torch.randn((num_nodes, memory_param['mailbox_size']), dtype=torch.float32).reshape(num_nodes, -1)

    mem_total_shape = (1 * mem_dim) + (1) + (1 * mail_size * (2 * mem_dim + dim_edge_feat)) + (1 * mail_size)
    # mem_diff_shape = np.cumsum(np.array([0, (1 * mem_dim) , (1) , (1 * mail_size * (2 * mem_dim + dim_edge_feat)) , (1 * mail_size)]))

    # total_memory = torch.cat([memory, memory_ts, mailbox, mailbox_ts], dim = 1)

    stream_rand_save(path + '/total_memory.bin', num_nodes, mem_total_shape, 10 * 1024 ** 3, torch.float32)
    # del(memory, memory_ts, mailbox, mailbox_ts)
    # saveBin(total_memory, path + '/total_memory.bin')

feat_len = 128
if (data in ['TALK', 'STACK', 'BITCOIN']):
    feat_len = 172
init_memory(feat_len, node_num)

print(f"共用时: {time.time() - total_start:.2f}s")