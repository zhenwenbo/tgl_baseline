


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

data = 'LASTFM'

g, datas, df_conf = load_graph_bin(data)

train_edge_end = df_conf['train_edge_end']
val_edge_end = df_conf['val_edge_end']
e_src = datas['src']
e_dst = datas['dst']

def gen_part():
    #当分区feat不存在的时候做输出
    res = []
    node_count = torch.zeros(g['indptr'].shape[0], dtype = torch.int32)
    batch_size = 60000

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

saveBin(start, f'/raid/guorui/DG/dataset/{data}/simple_start.bin')
saveBin(end, f'/raid/guorui/DG/dataset/{data}/simple_end.bin')
saveBin(IDs, f'/raid/guorui/DG/dataset/{data}/simple_IDs.bin')


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
 
    saveBin(start_cache_ind, f'/raid/guorui/DG/dataset/{data}/start_cache_ind.bin')
    saveBin(start_flush_ind, f'/raid/guorui/DG/dataset/{data}/start_flush_ind.bin')
    saveBin(end_flush_idx, f'/raid/guorui/DG/dataset/{data}/end_flush_idx.bin')
    saveBin(start_cache_ptr, f'/raid/guorui/DG/dataset/{data}/start_cache_ptr.bin')
    saveBin(start_flush_ptr, f'/raid/guorui/DG/dataset/{data}/start_flush_ptr.bin')
    saveBin(end_flush_ptr, f'/raid/guorui/DG/dataset/{data}/end_flush_ptr.bin')

    return simple_node, simple_block
simple_node, simple_block = gen_simple(start, end, IDs)

node, node_sort_ind = torch.sort(simple_node)
block = simple_block[node_sort_ind]

max_num = 0
for i in range(len(batch_plan)):
    max_num = max(max_num, batch_plan[i].shape[0])
saveBin(torch.tensor([max_num], dtype = torch.int32), f'/raid/guorui/DG/dataset/{data}/simple_max_num.bin')





# saveBin(IDs, f'/raid/guorui/DG/dataset/{data}/simple_IDs.bin')
# total_num = 0
# for i in range(torch.max(end)):
#     print(f"{i}:{torch.nonzero(end == i).shape[0]}", end = ' ')
#     total_num += torch.nonzero(end == i).shape[0]

# end_sort, end_sort_ind = torch.sort(end)
# IDs_sort = IDs[end_sort_ind]

# node = IDs_sort
# block = end_sort

# node_count_sort,node_count_sort_ind = torch.sort(node_count, descending=True)
# cache_nodes = node_count_sort_ind[:1000000].cuda()


# simple_node = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
# simple_block = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
# for i in range(len(batch_plan)):
#     cur_block = torch.from_numpy(block_info[i]).cuda()
#     cur_cache = batch_plan[i].cuda()
#     simple_cache = cur_block[torch.isin(cur_block, cur_cache, invert=True)]
    
#     simple_node = torch.cat((simple_node, simple_cache))
#     simple_block = torch.cat((simple_block, torch.zeros(simple_cache.shape[0], dtype = torch.int32, device = 'cuda:0') + i))

# torch.save(simple_node, '/home/guorui/workspace/tmp/root_node_simple.pt')
# torch.save(simple_block, '/home/guorui/workspace/tmp/root_block_simple.pt')

# # 判断缓存后的每个block写回多少
# for i in range(len(block_info)):
#     cur_block = torch.from_numpy(block_info[i]).cuda()
#     cur_cache = batch_plan[i].cuda()
#     simple_cache = torch.sum(torch.isin(cur_block, cur_cache, invert=True))
#     static_cache = torch.sum(torch.isin(cur_block, cache_nodes, invert=True))

#     simple_cache_diff = (cur_block.shape[0] - simple_cache) / cur_block.shape[0] * 100
#     static_cache_diff = (cur_block.shape[0] - static_cache) / cur_block.shape[0] * 100
#     print(f"block{i} 原始需写回{cur_block.shape[0]}  simple缓存后需写回{simple_cache} 减少{simple_cache_diff:.2f}% 静态缓存后写回{static_cache} 减少{static_cache_diff:.2f}%")

# for i in range(100):
#     cur_block = torch.from_numpy(block_info[i]).cuda()
#     cur_cache = batch_plan[i].cuda()
#     simple_cache = torch.sum(torch.isin(cur_block, cur_cache, invert=True))
#     static_cache = torch.sum(torch.isin(cur_block, cache_nodes, invert=True))

#     simple_cache_diff = (cur_block.shape[0] - simple_cache) / cur_block.shape[0] * 100
#     static_cache_diff = (cur_block.shape[0] - static_cache) / cur_block.shape[0] * 100
#     print(f"block{i} 原始需写回{cur_block.shape[0]}  simple缓存后需写回{simple_cache} 减少{simple_cache_diff:.2f}% 静态缓存后写回{static_cache} 减少{static_cache_diff:.2f}%")


