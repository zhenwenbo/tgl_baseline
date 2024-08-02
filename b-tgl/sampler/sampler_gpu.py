
import dgl
import torch
import numpy as np
import time
from utils import emptyCache

class Sampler_GPU:
    def __init__(self, g, fan_nums, layer, emb_buffer = None):
        self.indptr = torch.from_numpy(g['indptr']).cuda().to(torch.int32)
        self.indices = torch.from_numpy(g['indices']).to(torch.int32).pin_memory()
        self.totalts = torch.from_numpy(g['ts']).to(torch.float32).pin_memory()
        self.totaleid = torch.from_numpy(g['eid']).to(torch.int32).pin_memory()
        # self.indices = torch.from_numpy(g['indices']).to(torch.int32).cuda()
        # self.totalts = torch.from_numpy(g['ts']).to(torch.float32).cuda()
        # self.totaleid = torch.from_numpy(g['eid']).to(torch.int32).cuda()
        self.fan_nums = fan_nums
        self.layer = layer
        self.fan_num = fan_nums[0]
        self.emb_buffer = emb_buffer
        

    def sample(self, sampleIDs, curts, sample_mask = None, expired = None, mode = 'normal', sample_param = {}):
        seed_num = sampleIDs.shape[0]
        out_src = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        out_dst = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        outts = torch.zeros(seed_num * self.fan_num, dtype = torch.float32, device = 'cuda:0')-1
        outeid = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        if (sample_mask == None):
            sample_mask = torch.zeros_like(sampleIDs)

        if (mode == 'normal'):
            NUM = dgl.sampling.sample_with_ts_recent(self.indptr,self.indices,curts,self.totalts,self.totaleid,sampleIDs,
                                                    sample_mask,
                                                    seed_num,self.fan_num,out_src,out_dst,outts,outeid)
        elif (mode == 'expire'):
            # print(f"expire模式，{expired.shape}, {self.indices.shape}")
            NUM = dgl.sampling.sample_with_expired(self.indptr,self.indices,curts,self.totalts,self.totaleid,sampleIDs,
                                                    expired,
                                                    seed_num,self.fan_num,sample_param['cur_block'], sample_param['zombie_block'], out_src,out_dst,outts,outeid)
            
        return (out_src.reshape(seed_num, -1), out_dst.reshape(seed_num, -1), outts.reshape(seed_num, -1), outeid.reshape(seed_num, -1),
                sampleIDs, curts)

    def sample_layer(self, sampleIDs, curts, expired = None, sample_mode = 'normal', sample_param = {}):

        ts = curts
        root_nodes = sampleIDs
        sample_list = []
        sample_mask = None

        for i in range(self.layer):
            self.fan_num = self.fan_nums[i]

            
            sample_ret = self.sample(root_nodes, ts, sample_mask, expired=expired, mode = sample_mode, sample_param = sample_param)
            (out_src, out_dst, outts, outeid, root_nodes, curts) = sample_ret
            # print(f"layer: {i} 不采样的节点:{torch.sum(sample_mask) if sample_mask != None else 0}, 采样边数{torch.sum(outeid > -1)}")
                        
            if (i < self.layer - 1):
                mask = out_src>-1
                root_nodes = torch.cat((root_nodes,out_src[mask])) #src_table是实际节点ID
                ts = torch.cat((ts, outts[mask]))
                if (self.emb_buffer and self.emb_buffer.use_buffer and self.emb_buffer.cur_mode == 'train' and i == 0):
                    res = self.emb_buffer.gen_his_indices(root_nodes)
                    sample_mask = (res > -1).to(torch.int32)
                    
               
            sample_list.append(sample_ret)

        return sample_list
            




    def gen_table(self, seed_num):
        fan_num = self.fan_num
        start = time.time()
        table = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0')
        ar = torch.arange(seed_num, dtype = torch.int32, device = 'cuda:0')
        for i in range(fan_num):
            table[i * seed_num: (i+1) * seed_num] = ar
        table = table.reshape(fan_num, -1).T
        # print(f"gen_table use time {time.time() - start:.5f}s")
        
        return table

    def gen_block(self, sample_ret):
        #src dst ts eid做成dgl block
        start = time.time()
        src,dst,outts,outeid,root_nodes,root_ts = sample_ret
        
        seed_num = root_nodes.shape[0]
        fan_num = self.fan_num

        mask = src>-1
        # table = ((torch.arange(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(-1, fan_num)) / fan_num).to(torch.int64)
        
        table = self.gen_table(seed_num).to(torch.int64)
        
        dst_node = table[mask].to(torch.int32)
        # table[mask]可以直接作为0-200的dst节点 ,souce_nodes作为节点id

        # src[mask]中，每个值都是独立的节点编号，因此直接从200开始arange即可， 而节点id就直接拿src[mask]
        src_table = src[mask]
        src_node = torch.arange(src_table.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

        #nodes为所有节点的id，src的节点前面拼dst的节点，id的话，dst节点id就是source_nodes
        nodes = torch.cat((root_nodes, src_table))
        tss = torch.cat((root_ts, outts[mask]))

        # print(f"dst_node shape: {dst_node.shape}, max in dst node: {torch.max(dst_node)}, num_dst: {root_nodes.shape[0]}")
        b = dgl.create_block((src_node.to(torch.int64), dst_node.to(torch.int64)), num_src_nodes = nodes.shape[0], num_dst_nodes = root_nodes.shape[0])
        b.srcdata['ID'] = nodes
        b.srcdata['ts'] = tss

        outdts = root_ts[table][mask] - outts[mask]
        b.edata['dt'] = outdts
        b.edata['ID'] = outeid[mask]

        # print(f"gen block用时{time.time() - start:.5f}s")
        return b

    def gen_mfgs(self, sample_ret_list):
        mfgs = list()

        start = time.time()
        for i, sample_ret in enumerate(sample_ret_list):
            self.fan_num = self.fan_nums[i]
            b = self.gen_block(sample_ret)
            # TODO 多层采样 及 块批采样
            
            mfgs.append(b.to('cuda:0'))
        # print(f"gen all block: {time.time() - start:.5f}s")

        mfgs = list(map(list, zip(*[iter(mfgs)] * 1)))
        mfgs.reverse()
        return mfgs