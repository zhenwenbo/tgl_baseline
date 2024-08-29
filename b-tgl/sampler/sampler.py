import argparse
import yaml
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sampler.sampler_core import ParallelSampler, TemporalGraphBlock
from utils import *

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)
    

class ReNegLinkSampler:

    pre_res = None
    ratio = 1

    def __init__(self, num_nodes, ratio):
        self.ratio = ratio
        print(f"重放负采样，重放比例为 {self.ratio}")
        self.num_nodes = num_nodes

    def sample(self, n):
        
        cur_res = np.random.randint(self.num_nodes, size=n)
        if (self.pre_res is not None):
            #将上一次pre_res中的百分之k 随机替换到cur_res中
            pre_ind = np.random.randint(self.pre_res.shape[0], size = int(cur_res.shape[0] * self.ratio))
            cur_ind = torch.randperm((cur_res.shape[0]), dtype = torch.int32)[:int(cur_res.shape[0]  * self.ratio)].numpy()

            cur_res[cur_ind] = self.pre_res[pre_ind]
            # cur_res[]   
            sameNum = np.sum(np.isin(cur_res, self.pre_res))

            # print(f'相同的负采样节点个数: {sameNum} 占比: {sameNum / n * 100:.2f}%')
        self.pre_res = cur_res
        return cur_res

import math
from config.train_conf import *
class TrainNegLinkSampler:

    # 做block分区，每个分区中有block num个负节点，每次sample都是一次block级的采样
    def __init__(self, num_nodes, num_edges, k = 8):

        config = GlobalConfig()
        block_size = config.pre_sample_size
        block_num = math.ceil(num_edges/ block_size)

        block_neg = torch.randint(0, num_nodes, (num_edges,))

        self.part = []
        nodes = torch.randperm(num_nodes, dtype = torch.int32)
        
        per_node_num = block_size
        left, right = 0, 0
        self.k = k
        
        for i in range(block_num):
            self.part.append(block_neg[i * block_size: min((i+1) * block_size, block_neg.shape[0])])

        self.num_nodes = num_nodes

        self.ptr = 0

        

    def sample(self, n, i = 0, cur_batch = 0):
        part_len = len(self.part)
        # res = np.random.randint(self.num_nodes, size=n)
        i = self.ptr % part_len
        res = np.random.choice(self.part[i], size=n, replace = True)
        # print(f"train neg sampler... {res}")
        self.ptr += 1
        return res

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name', default='GDELT')
    parser.add_argument('--config', type=str, help='path to config file', default='./config/TGN.yml')
    parser.add_argument('--batch_size', type=int, default=600000, help='path to config file')
    parser.add_argument('--num_thread', type=int, default=64, help='number of thread')
    args=parser.parse_args()

    g, df = load_graph(args.data)

    sample_config = yaml.safe_load(open(args.config, 'r'))['sampling'][0]

    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              args.num_thread, 1, sample_config['layer'], sample_config['neighbor'],
                              sample_config['strategy']=='recent', sample_config['prop_time'],
                              sample_config['history'], float(sample_config['duration']))

    num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
    neg_link_sampler = NegLinkSampler(num_nodes)

    tot_time = 0
    ptr_time = 0
    coo_time = 0
    sea_time = 0
    sam_time = 0
    uni_time = 0
    total_nodes = 0
    unique_nodes = 0
    for _, rows in (df.groupby(df.index // args.batch_size)):
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)

        #不采样负边节点，测试采样
        # root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        # ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)

        start = time.time()
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        print(f"第{_}次采样 batch_size: {args.batch_size} 时间{time.time() - start:.6f}")

        cur_ret = ret[0]
        # print(root_nodes)
        # np.savez(f'./ret{_}.npz', row = cur_ret.row(), col = cur_ret.col(), eid = cur_ret.eid(), ts = cur_ret.ts(),
        #           nodes = cur_ret.nodes(), dst = cur_ret.dts(), dim_in = cur_ret.dim_in(), dim_out = cur_ret.dim_out())
        
        
        tot_time += ret[0].tot_time()
        ptr_time += ret[0].ptr_time()
        coo_time += ret[0].coo_time()
        sea_time += ret[0].search_time()
        sam_time += ret[0].sample_time()
        # for i in range(sample_config['history']):
        #     total_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     unique_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     if ret[i].dim_in() > ret[i].dim_out():
        #         ts = torch.from_numpy(ret[i].ts()[ret[i].dim_out():])
        #         nid = torch.from_numpy(ret[i].nodes()[ret[i].dim_out():]).float()
        #         nts = torch.stack([ts,nid],dim=1).cuda()
        #         uni_t_s = time.time()
        #         unts, idx = torch.unique(nts, dim=0, return_inverse=True)
        #         uni_time += time.time() - uni_t_s
        #         total_nodes += idx.shape[0]
        #         unique_nodes += unts.shape[0]

    print('total time  : {:.4f}'.format(tot_time))
    print('pointer time: {:.4f}'.format(ptr_time))
    print('coo time    : {:.4f}'.format(coo_time))
    print('search time : {:.4f}'.format(sea_time))
    print('sample time : {:.4f}'.format(sam_time))
    # print('unique time : {:.4f}'.format(uni_time))
    # print('unique per  : {:.4f}'.format(unique_nodes / total_nodes))


    