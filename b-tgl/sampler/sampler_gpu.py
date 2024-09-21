
import dgl
import torch
import numpy as np
import time
from utils import emptyCache

class Sampler_GPU:
    def __init__(self, g, fan_nums, layer, emb_buffer = None):

        if ((not g['indices'].shape[0] > 1000000000) and (layer < 2 or g['indices'].shape[0] < 100000000)):
            self.indptr = torch.from_numpy(g['indptr']).cuda().to(torch.int32)
            self.indices = torch.from_numpy(g['indices']).to(torch.int32).cuda()
            self.totalts = torch.from_numpy(g['ts']).to(torch.float32).cuda()
            self.totaleid = torch.from_numpy(g['eid']).to(torch.int32).cuda()
        else:
            self.indptr = torch.from_numpy(g['indptr']).cuda().to(torch.int32)
            self.indices = torch.from_numpy(g['indices']).to(torch.int32).pin_memory()
            self.totalts = torch.from_numpy(g['ts']).to(torch.float32).pin_memory()
            self.totaleid = torch.from_numpy(g['eid']).to(torch.int32).pin_memory()
        
        # del g
        # self.indices = torch.from_numpy(g['indices']).to(torch.int32).cuda()
        # self.totalts = torch.from_numpy(g['ts']).to(torch.float32).cuda()
        # self.totaleid = torch.from_numpy(g['eid']).to(torch.int32).cuda()
        self.fan_nums = fan_nums
        self.layer = layer
        self.fan_num = fan_nums[0]
        self.emb_buffer = emb_buffer
        
        self.mask_time = 0

    def sample(self, sampleIDs, curts, sample_mask = None, expired = None, mode = 'normal', sample_param = {}):
        seed_num = sampleIDs.shape[0]
        out_src = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        out_dst = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        outts = torch.zeros(seed_num * self.fan_num, dtype = torch.float32, device = 'cuda:0')-1
        outeid = torch.zeros(seed_num * self.fan_num, dtype = torch.int32, device = 'cuda:0')-1
        outdts = torch.zeros(seed_num * self.fan_num, dtype = torch.float32, device = 'cuda:0')-1
        if (sample_mask == None):
            sample_mask = torch.zeros_like(sampleIDs)

        if (mode == 'normal'):
            NUM = dgl.sampling.sample_with_ts_recent(self.indptr,self.indices,curts,outdts,self.totalts,self.totaleid,sampleIDs,
                                                    sample_mask,
                                                    seed_num,self.fan_num,out_src,out_dst,outts,outeid)
        elif (mode == 'expire'):
            # print(f"expire模式，{expired.shape}, {self.indices.shape}")
            NUM = dgl.sampling.sample_with_expired(self.indptr,self.indices,curts,self.totalts,self.totaleid,sampleIDs,
                                                    expired,
                                                    seed_num,self.fan_num,sample_param['cur_block'], sample_param['zombie_block'], out_src,out_dst,outts,outeid)
        
        mask_time_s = time.time()
        mask = out_src > -1
        res = [out_src[mask], out_dst[mask], outts[mask], outeid[mask],
                sampleIDs, curts, outdts[mask]]
        self.mask_time += time.time() - mask_time_s
        return res
    
        # return (out_src.reshape(seed_num, -1), out_dst.reshape(seed_num, -1), outts.reshape(seed_num, -1), outeid.reshape(seed_num, -1),
        #         sampleIDs, curts,outdts)

    def sample_layer(self, sampleIDs, curts, expired = None, sample_mode = 'normal', sample_param = {}, cut_zombie = False):

        ts = curts
        root_nodes = sampleIDs
        sample_list = []
        sample_mask = None

        for i in range(self.layer):
            self.fan_num = self.fan_nums[i]

            
            [out_src, out_dst, outts, outeid, root_nodes, curts, dts] = self.sample(root_nodes, ts, sample_mask, expired=expired, mode = sample_mode, sample_param = sample_param)
            # [out_src, out_dst, outts, outeid, root_nodes, curts, dts] = sample_ret
            
            if (cut_zombie):
                # noncut_indices = torch.nonzero(dts <= torch.mean(dts)).reshape(-1)
                # print(f"比平均值多的边数: {noncut_indices.shape[0]} 占比{noncut_indices.shape[0] / dts.shape[0] * 100 :.4f}%")
                # out_src = out_src[noncut_indices]
                # out_dst = out_dst[noncut_indices]
                # outts = outts[noncut_indices]
                # outeid = outeid[noncut_indices]
                # dts = dts[noncut_indices]


                cut_mask = torch.zeros(dts.shape[0], dtype=torch.bool, device = 'cuda:0')
                cut_indices = torch.nonzero(dts > torch.median(dts)).reshape(-1)

                # 随机裁剪一半
                rand_indices = torch.randperm(cut_indices.shape[0], dtype = torch.int64, device = 'cuda:0')
                cut_indices = cut_indices[rand_indices[:rand_indices.shape[0]//2]]
                cut_mask[cut_indices] = True
                noncut_mask = ~cut_mask

                out_src = out_src[noncut_mask]
                out_dst = out_dst[noncut_mask]
                outts = outts[noncut_mask]
                outeid = outeid[noncut_mask]
                dts = dts[noncut_mask]


            # print(f"layer: {i} 不采样的节点:{torch.sum(sample_mask) if sample_mask != None else 0}, 采样边数{torch.sum(outeid > -1)}")
            sample_list.append([out_src, out_dst, outts, outeid, root_nodes, curts, dts])

            if (i < self.layer - 1):
                root_nodes = torch.cat((root_nodes,out_src)) #src_table是实际节点ID
                ts = torch.cat((ts, outts))
                if (self.emb_buffer and self.emb_buffer.use_buffer and self.emb_buffer.cur_mode == 'train' and i == 0):
                    res = self.emb_buffer.gen_his_indices(root_nodes)
                    sample_mask = (res > -1).to(torch.int32)
                    
               
            

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

    def gen_block(self, sample_ret, reverse = False):
        start = time.time()
        src,dst,outts,outeid,root_nodes,root_ts,dts = sample_ret
        
        seed_num = root_nodes.shape[0]

        # mask = src>-1
        
        # table = self.gen_table(seed_num).to(torch.int64)
        
        dst_node = dst
        # table[mask]可以直接作为0-200的dst节点 ,souce_nodes作为节点id

        # src[mask]中，每个值都是独立的节点编号，因此直接从200开始arange即可， 而节点id就直接拿src[mask]
        src_table = src


        src_node = torch.arange(src_table.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

        #nodes为所有节点的id，src的节点前面拼dst的节点，id的话，dst节点id就是source_nodes
        nodes = torch.cat((root_nodes, src_table))
        tss = torch.cat((root_ts, outts))

        # print(f"dst_node shape: {dst_node.shape}, max in dst node: {torch.max(dst_node)}, num_dst: {root_nodes.shape[0]}")
        if (not reverse):
            b = dgl.create_block((src_node.to(torch.int64), dst_node.to(torch.int64)), num_src_nodes = nodes.shape[0], num_dst_nodes = root_nodes.shape[0])
            b.srcdata['ID'] = nodes
            b.srcdata['ts'] = tss

            outdts = dts - outts
            b.edata['dt'] = outdts
            b.edata['ID'] = outeid
        else:
            b = dgl.create_block((dst_node.to(torch.int64), src_node.to(torch.int64)), num_src_nodes = root_nodes.shape[0], num_dst_nodes = nodes.shape[0])
            b.dstdata['ID'] = nodes
            b.dstdata['ts'] = tss

            outdts = dts - outts
            b.edata['dt'] = outdts
            b.edata['ID'] = outeid

        # print(f"gen block用时{time.time() - start:.5f}s")
        return b

    def gen_block_1(self, sample_ret):
        #src dst ts eid做成dgl block
        start = time.time()
        src,dst,outts,outeid,root_nodes,root_ts,dts = sample_ret
        
        seed_num = root_nodes.shape[0]
        # fan_num = self.fan_num

        # mask = src>-1
        # table = ((torch.arange(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(-1, fan_num)) / fan_num).to(torch.int64)
        
        # table = self.gen_table(seed_num).to(torch.int64)

        nodeTable = torch.cat((root_nodes))
        uniTable = torch.zeros_like(nodeTable).to(torch.int32).cuda()

        #TODO 采样时应该让src中输入的是root_nodes的索引而不是全局的节点id，这样就可以避免再次处理。
        #       这样做就可以直接使用这个src，然后用root_node做他的node_id

        # dst_node = table[mask].to(torch.int32)
        # table[mask]可以直接作为0-200的dst节点 ,souce_nodes作为节点id

        # src[mask]中，每个值都是独立的节点编号，因此直接从200开始arange即可， 而节点id就直接拿src[mask]
        # src_table = src[mask]
        src_node = torch.arange(src.shape[0], dtype = torch.int32, device = 'cuda:0') + seed_num

        #nodes为所有节点的id，src的节点前面拼dst的节点，id的话，dst节点id就是source_nodes
        # nodes = torch.cat((root_nodes, src_table))
        nodes = uniTable
        tss = torch.cat((root_ts, outts))

        # print(f"dst_node shape: {dst_node.shape}, max in dst node: {torch.max(dst_node)}, num_dst: {root_nodes.shape[0]}")
        b = dgl.create_block((src_node.to(torch.int64), dst_node.to(torch.int64)), num_src_nodes = nodes.shape[0], num_dst_nodes = root_nodes.shape[0])
        b.srcdata['ID'] = nodes
        b.srcdata['ts'] = tss

        outdts = dts - outts
        b.edata['dt'] = outdts
        b.edata['ID'] = outeid

        # print(f"gen block用时{time.time() - start:.5f}s")
        return b

    def gen_mfgs(self, sample_ret_list, reverse = False):
        mfgs = list()

        start = time.time()
        for i, sample_ret in enumerate(sample_ret_list):
            self.fan_num = self.fan_nums[i]
            b = self.gen_block(sample_ret, reverse)
            # TODO 多层采样 及 块批采样
            
            mfgs.append(b.to('cuda:0'))
        # print(f"gen all block: {time.time() - start:.5f}s")

        mfgs = list(map(list, zip(*[iter(mfgs)] * 1)))
        mfgs.reverse()
        return mfgs