
import torch
import dgl
import numpy as np
import pandas as pd
import time
import sys
import os
total_start = time.time()
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.train_conf import *
GlobalConfig.conf = 'basic_conf.json'

from utils import *
from sampler import *

from sampler.sampler_gpu import *


def get_max_edge_num(indptr):
    dif = torch.diff(indptr)
    ind = torch.nonzero(dif > 10).reshape(-1)
    dif[ind] = 10

    return torch.sum(dif)

def gen_expire(args):

    d = args.data
    zombie_block = args.zombie_block
    batch_size = args.bs

    # d = 'LASTFM'
    # batch_size = 600000
    # df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
    # g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
    g, datas, df_conf = load_graph_bin(args.data)


    fan_nums = [10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)


    expired = torch.zeros_like(sampler_gpu.indices).cuda()

    cal_max_edge_num = 0
    cal_edge_num = 0
    max_edge_num = get_max_edge_num(sampler_gpu.indptr) + batch_size * 3

    totaleid = sampler_gpu.totaleid.cuda()

    # 处理无向图，给出eid_flag
    eid_flag = torch.zeros_like(totaleid)
    eid_uni, counts = torch.unique(totaleid, return_counts = True)
    eid_uni, indices = torch.sort(eid_uni)
    counts = counts[indices]
    eid_flag = counts

    start_eid = 0
    end_eid = 0
    end_ptr = 0

    map = torch.zeros(max_edge_num, dtype = torch.int32, device = 'cuda:0') + (2**31 - 1) 

    exp_eids = None

    # node_feats, edge_feats = load_feat(d)
    path = f'/raid/guorui/DG/dataset/{d}'

    part_path = path + f'/part-{batch_size}'

    if not os.path.exists(part_path):
        os.mkdir(part_path)

    unexpire_ind = torch.empty(0, dtype = torch.int32, device = 'cuda:0')


    left, right = 0, 0
    batch_num = 0
    total_src = datas['src'].cuda()
    total_dsts = datas['dst'].cuda()
    total_time = datas['time'].to(torch.float32).cuda()
    edge_end = datas['src'].shape[0]
    while True:
        total_s = time.time()
        right += batch_size
        right = min(edge_end, right)
        if (left >= right):
            break

        src = total_src[left: right]
        dst = total_dsts[left: right]
        times = total_time[left: right]

        # root_nodes = torch.from_numpy(np.concatenate([src, dst]).astype(np.int32)).cuda()
        # root_ts = torch.from_numpy(np.concatenate([times, times]).astype(np.float32)).cuda()
        root_nodes = torch.cat([src, dst])
        root_ts = torch.cat([times, times]).to(torch.float32)

        start_eid = end_eid
        end_eid += root_nodes.shape[0] // 2

        cur_eids = torch.arange(start_eid, end_eid, dtype = torch.int32, device='cuda:0')
        # cur_edge_feat = edge_feats[cur_eids.cpu().long()]
        # cur_edge_feat = loadBinDisk(f'/raid/guorui/DG/dataset/{d}/edge_features.bin', cur_eids.cpu().long())
        # saveBin(cur_edge_feat.cpu(), part_path + f'/part{cur_batch}_edge_incre.pt')
        ind_se = torch.tensor([start_eid, end_eid], dtype = torch.int32)
        saveBin_concurrent(ind_se, part_path + f'/part{batch_num}_edge_incre_bound.pt' )
        #TODO incre feat可以和gen_part里的正边采样出的边特征合作一下
        print(f"抽取时间{time.time() - total_s:.6f}s")

        judge_s = time.time()
        replace_idx = None
        if (exp_eids == None):
            #第一次，直接放入
            map[:cur_eids.shape[0]] = cur_eids
            end_ptr = end_ptr + cur_eids.shape[0]
            replace_idx = torch.arange(cur_eids.shape[0], dtype = torch.int32)
        else:
            #计算出exp在map中的位置
            exp_eids_sort,_ = torch.sort(exp_eids)
            map_sort,map_sort_indices = torch.sort(map)
            table1 = torch.zeros_like(exp_eids_sort) - 1
            table2 = torch.zeros_like(map_sort) - 1
            dgl.findSameIndex(exp_eids_sort, map_sort, table1, table2)
            table1 = map_sort_indices[table1.long()]

            unalloc_ptr = None
            if (torch.nonzero(table1 == -1).reshape(-1).shape[0] > 0):
                print(f"出现异常")
            # if (root_nodes.shape[0] // 2 < batch_size):
            #     replace_idx = table1[:cur_eids.shape[0]].to(torch.int64)
            #     map[replace_idx] = cur_eids
            else:
                replace_idx = torch.zeros_like(cur_eids)
                #此处作判断：若淘汰的边数不足batch_size，则考虑先前未淘汰的ind
                #若淘汰的边数 + 先前未淘汰的足以满足batch_size，做特殊处理
                if (table1.shape[0] >= cur_eids.shape[0]):
                    cur_unexpire = table1[cur_eids.shape[0]:].to(torch.int32)
                    unexpire_ind = torch.cat((unexpire_ind, cur_unexpire))
                    replace_idx[:cur_eids.shape[0]] = table1[:cur_eids.shape[0]]
                elif (table1.shape[0] + unexpire_ind.shape[0] >= cur_eids.shape[0]):
                    use_expire_num = cur_eids.shape[0] - table1.shape[0]
                    use_expire = unexpire_ind[:use_expire_num]
                    unexpire_ind = unexpire_ind[use_expire_num:]
                    replace_idx[:table1.shape[0]] = table1
                    replace_idx[table1.shape[0]:] = use_expire
                else:
                    #不够用,全用了
                    unalloc_ptr = table1.shape[0] + unexpire_ind.shape[0] #不需要开辟的长度
                    replace_idx[:table1.shape[0]] = table1
                    if (unexpire_ind.shape[0] > 0):
                        replace_idx[table1.shape[0]:table1.shape[0] + unexpire_ind.shape[0]] = unexpire_ind
                        unexpire_ind = torch.empty(0, dtype = torch.int32, device='cuda:0')

                    replace_idx[unalloc_ptr:] = torch.arange(end_ptr, end_ptr + (replace_idx.shape[0] - unalloc_ptr), dtype = torch.int32, device = 'cuda:0')
                
                
                replace_idx = replace_idx.to(torch.int64)

                if (unalloc_ptr is None):
                    unalloc_ptr = cur_eids.shape[0]
                end_ptr = end_ptr + (replace_idx.shape[0] - unalloc_ptr)
                if (torch.max(replace_idx) >= map.shape[0]):
                    map = torch.cat((map, torch.zeros(torch.max(replace_idx) - map.shape[0] + 1, dtype = torch.int32, device = 'cuda:0') + (2**31 - 1) ))
                map[replace_idx] = cur_eids
                # break
        print(f"判断时间 {time.time() - judge_s:.6f}s")

        save_s = time.time()
        if (batch_num == 0):
            saveBin_concurrent(map.cpu(), part_path + f'/part{batch_num}_edge_incre_map.pt')
        print(f"save 时间 {time.time() - save_s:.6f}s")
        if (replace_idx is not None):
            saveBin_concurrent(replace_idx.cpu(), part_path + f'/part{batch_num}_edge_incre_replace.pt')
        print(f"save 时间 {time.time() - save_s:.6f}s")
        start = time.time()
        expired_clone = expired.clone()
        ret_list = sampler_gpu.sample_layer(root_nodes, root_ts, expired=expired, sample_mode = 'expire', sample_param={'cur_block': batch_num, 'zombie_block': zombie_block})
        # zombie_edge = torch.nonzero(expired == 2).reshape(-1)
        # erro_edge = torch.nonzero(torch.bitwise_and(expired == 2, expired_clone == 1)).reshape(-1)
        # print(f"异常个数: {erro_edge.shape[0]}")
        # print(f"僵尸边个数: {zombie_edge.shape[0]}")
        # expired[zombie_edge] = 0
        expired_cur = expired ^ expired_clone

        # emptyCache()

        
        ind = torch.nonzero(expired_cur).reshape(-1)
        exp_eids = totaleid[ind]

        # break
        exp_eids, counts = torch.unique(exp_eids, return_counts = True)
        eid_flag_clone = eid_flag.clone()
        eid_flag[exp_eids.long()] -= counts
        #满足两个条件 1.eid_flag改变。2.eid_flag变为0
        exp_eids = torch.nonzero( (eid_flag ^ eid_flag_clone).to(torch.bool) & (eid_flag == 0) ).to(torch.int32).reshape(-1)
        
        print(f"计算时间 {time.time() - start:.6f}s")

        cal_edge_num += root_nodes.shape[0] // 2
        cal_max_edge_num = max(cal_edge_num, cal_max_edge_num)
        # print(torch.sum(map < (2**30)))
        print(f"{start_eid}:{end_eid},end_ptr:{end_ptr},全体最多边数{max_edge_num} 新加了{root_nodes.shape[0] // 2}条边, 待丢弃{exp_eids.shape[0]}条边, 目前总边数{cal_edge_num}, 历史上最大边数{cal_max_edge_num} loop 用时{time.time() - total_s:.6f}s")
        
        cal_edge_num -= exp_eids.shape[0]
        # if (_ == 100):
        #     break

        left = right
        batch_num += 1
    return end_ptr

    # srcr,dstr,outts,outeid,root_nodes,root_ts = ret_list[-1]
    # mask = srcr > -1
    # src = srcr[mask]
    # dst = dstr[mask]
    # eid = outeid[mask]
    # print(expired[:10])
    # print(sampler_gpu.totaleid[:10])
    # print(sampler_gpu.indices[:10])
    # print(sampler_gpu.indptr[:10])

    # totaleid = sampler_gpu.totaleid
    # ind = torch.nonzero(expired_cur).reshape(-1)
    # exp_eids = totaleid[ind]

    # uni, counts = torch.unique(exp_eids, return_counts = True)
    # real_ind = counts > 1
    # if (torch.sum(real_ind) > 0):
    #     print(f"是无向图")

    #     uni = uni[counts > 1]

    # print(f"丢弃边数{uni.shape}")



import argparse
import os
import json

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='BITCOIN')
parser.add_argument('--bs', type=int, help='batch size', default='600000')
parser.add_argument('--zombie_block', type=int, help='zombie block', default='2')
args=parser.parse_args()

max_edge_num = gen_expire(args)
data = {args.data: max_edge_num}

file_path = f'/raid/guorui/workspace/dgnn/b-tgl/preprocessing/expire-{args.bs}.json'

if os.path.exists(file_path):
    # 文件存在，读取现有数据
    with open(file_path, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
    
    # 更新键值
    existing_data.update(data)
else:
    # 文件不存在，使用定义的数据
    existing_data = data

# 将数据写入JSON文件
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(existing_data, file, ensure_ascii=False, indent=4)

print(f"共用时{time.time() - total_start:.4f}s")