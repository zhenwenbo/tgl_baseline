
#历史嵌入重用缓存 Orca
import dgl
import torch
import numpy as np
import time
from sampler.sampler_gpu import *
from config.train_conf import *
import os
from utils import *
import concurrent.futures
from queue import Queue
import multiprocessing
import json

class Feat_buffer:
    def __init__(self, d, g, df, train_param, memory_param, train_edge_end, presample_batch, sampler, neg_sampler, prefetch_conn = None, feat_dim = None):
        self.d = d
        self.g = g
        self.df = df
        self.sampler = sampler
        self.neg_sampler = neg_sampler
        self.presample_batch = presample_batch
        batch_size = train_param['batch_size'] * presample_batch
        train_batch_size = train_param['batch_size']
        self.path = f'/raid/guorui/DG/dataset/{d}'

        self.node_feat_dim, self.edge_feat_dim = feat_dim
        self.memory_param = memory_param
        self.use_memory = memory_param['type'] != 'none'
        

        self.batch_size = batch_size #预采样做分析的batch_size
        self.train_batch_size = train_batch_size
        self.train_edge_end = train_edge_end

        assert batch_size % train_batch_size == 0, "预采样batch size必须是训练的batch_size的整数倍!"
        self.batch_num = batch_size / train_batch_size # batch_num是预采样的batch个数

        self.batch_num = batch_size // train_batch_size
        #例如，batch_size分别是1000, 30000 即每次预采样采样到30个batch， 上界为5，那么每运行25个batch就需要做一次预采样

        self.cur_batch = 0 #当当前执行的batch为0,25,50时做一次预采样获得下面25次batch的plan

        self.cur_mode = 'train'
        self.use_buffer = True

        self.part_edge_feats = None
        self.part_edge_map = None
        self.part_node_feats = None
        self.part_node_map = None

        self.part_memory = None
        self.part_memory_ts = None
        self.part_mailbox = None
        self.part_mailbox_ts = None

        self.part_memory_map = None
        self.mode = ''
        
        self.time_load = 0
        self.time_analyze = 0
        self.time_presample = 0
        self.time_pre_neg_sample = 0
        self.time_exec_mem = 0
        self.time_neg_analyze = 0
        self.time_refresh = 0

        self.err_num = 0
        self.err_detection = False

        
        #下面是运行时做异步处理的部分
        self.preFetchExecutor = concurrent.futures.ThreadPoolExecutor(2)  # 线程池
        # ctx = multiprocessing.get_context("spawn")
        # self.pool = ctx.Pool(processes=1)
        # self.pipe = multiprocessing.Pipe(duplex=False)
        self.share_node_num = 1000000

        self.share_edge_num = 3000000 #TODO 动态扩容share tensor
        if (d == 'STACK'):
            self.share_edge_num = 40000000 #TODO 动态扩容share tensor
            self.share_node_num = 10000000

        if (prefetch_conn[0] is None):
            self.prefetch_conn = None
        else:
            self.prefetch_conn, self.prefetch_only_conn = prefetch_conn
            self.init_share_tensor()

        self.preFetchDataCache = Queue()
        self.cur_block = 0
        self.config = GlobalConfig()

        self.use_ayscn = self.config.use_ayscn_prefetch


        #下面是过期时间特征淘汰策略
        self.use_valid_edge = self.config.use_valid_edge

        if (self.use_valid_edge):
            self.unexpired_edge_map = None
            self.unexpired_edge_feats = None


            expire_info = None
            file_path = f'/raid/guorui/workspace/dgnn/b-tgl/preprocessing/expire-{self.batch_size}.json'
            with open(file_path, 'r', encoding='utf-8') as file:
                expire_info = json.load(file)
            if (self.prefetch_conn is not None):
                self.prefetch_conn.send(('init_valid_edge', (expire_info[d], self.edge_feat_dim, (self.path, self.batch_size, self.sampler.fan_nums))))
                self.prefetch_conn.recv()
            



    def init_share_tensor(self):
        node_num = self.share_node_num
        edge_num = self.share_edge_num
        node_feat_dim = self.node_feat_dim
        edge_feat_dim = self.edge_feat_dim
        

        part_node_map = torch.zeros(node_num, dtype = torch.int32).share_memory_()
        node_feats = torch.zeros((node_num, node_feat_dim), dtype = torch.float32).share_memory_()

        part_edge_map = torch.zeros(edge_num, dtype = torch.int32).share_memory_()
        edge_feats = torch.zeros((edge_num, edge_feat_dim), dtype = torch.float32).share_memory_()

        if (self.use_memory):
            mem_dim = self.memory_param['dim_out']
            mailbox_size = self.memory_param['mailbox_size']
            part_memory = torch.zeros((node_num, mem_dim), dtype = torch.float32).share_memory_()
            part_memory_ts = torch.zeros(node_num, dtype = torch.float32).share_memory_()
            part_mailbox = torch.zeros((node_num, mailbox_size, 2 * mem_dim + edge_feat_dim), dtype = torch.float32).share_memory_()
            part_mailbox_ts = torch.zeros((node_num, mailbox_size), dtype = torch.float32).share_memory_()
        else:
            part_memory = None
            part_memory_ts = None
            part_mailbox = None
            part_mailbox_ts = None

        pre_same_nodes = torch.zeros(node_num, dtype = torch.bool).share_memory_()
        cur_same_nodes = torch.zeros(node_num, dtype = torch.bool).share_memory_()

        shared_tensor = (part_node_map, node_feats, part_edge_map, edge_feats, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes)
        shared_ret_len = torch.zeros(len(shared_tensor), dtype = torch.int32).share_memory_()

        shared_tensor = (*shared_tensor, shared_ret_len)
        
        self.share_part_node_map = part_node_map
        self.share_part_node_feats = node_feats
        self.share_part_edge_map = part_edge_map
        self.share_part_edge_feats = edge_feats
        self.share_part_memory = part_memory
        self.share_part_memory_ts = part_memory_ts
        self.share_part_mailbox = part_mailbox
        self.share_part_mailbox_ts = part_mailbox_ts
        self.share_pre_same_nodes = pre_same_nodes
        self.share_cur_same_nodes = cur_same_nodes
        self.shared_ret_len = shared_ret_len

        #TODO 这里预先开辟1GB内存的 共享内存用于select_index
        #TODO 为APAN的mailbox，这里提高10倍给mailbox用
        self.share_tmp_tensor = torch.zeros(3000000000).share_memory_()
        shared_tensor = (*shared_tensor, self.share_tmp_tensor)

        self.prefetch_conn.send(('init_share_tensor', (shared_tensor,)))
        self.prefetch_conn.recv()

        


    def init_feat(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats

    def init_memory(self, memory, memory_ts, mailbox, mailbox_ts):
        #这里只会在没有子程序并行的时候使用
        self.memory = memory
        self.memory_ts = memory_ts
        self.mailbox = mailbox
        self.mailbox_ts = mailbox_ts

    def select_index(self, name, indices):
        self_v = getattr(self,name, None)
        if (self_v is not None):
            return self_v[indices]
        else:
            self.prefetch_conn.send(('select_index', (name, indices)))
            dim, shape = self.prefetch_conn.recv()
            result = self.share_tmp_tensor[:shape].reshape(dim)
            return result
        
    def update_index(self, name, indices, value):
        self_v = getattr(self,name, None)
        if (self_v is not None):
            self_v[indices] = value
        else:
            dim = value.shape
            value = value.reshape(-1)
            shape = value.shape[0]

            self.share_tmp_tensor = self.share_tmp_tensor.to(value.dtype)
            self.share_tmp_tensor[:shape] = value

            self.prefetch_conn.send(('update_index', (name, indices, (dim, shape))))
            self.prefetch_conn.recv()

    def reset_time(self):
        assert self.err_num == 0, "feat_buffer出现异常情况!"

    def print_time(self):
        print(f"feat_buffer time: IO load: {self.time_load:.3f}s 异步阻塞: {self.time_async:.3f}s presample and analyze: {self.time_analyze:.3f}s 预采样: {self.time_presample:.3f}s, 预采样负采样:{self.time_pre_neg_sample:.3f}s\
              记忆刷入和更新:{self.time_exec_mem:.3f}s 纯刷入时间:{self.time_refresh:.3f}s 负采样分析: {self.time_neg_analyze:.3f}s")


    def reset(self):
        self.part_edge_feats = None
        self.part_edge_map = None
        self.part_node_feats = None
        self.part_node_map = None

        self.part_memory = None
        self.part_memory_ts = None
        self.part_mailbox = None
        self.part_mailbox_ts = None

        self.cur_block = 0

        print(f"IO load: {self.time_load:.3f}s presample and analyze: {self.time_analyze:.3f}s 预采样: {self.time_presample:.3f}s, 预采样负采样:{self.time_pre_neg_sample:.3f}s")
        self.time_load = 0
        self.time_analyze = 0
        self.time_presample = 0
        self.time_pre_neg_sample = 0
        self.time_exec_mem = 0
        self.time_neg_analyze = 0
        self.time_refresh = 0
        self.time_async = 0

    def input_mails(self, b):

        nid = b.srcdata['ID']
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)

        #非实时刷入，因此这个检测失效。
        # if (self.err_detection):
        #     dis_ind = torch.nonzero(table2 == -1).reshape(-1)
        #     print(f"mail中从未出现过的节点: {dis_ind.shape[0]}")
        #     real_mem = self.memory[nid.to(torch.int64)].cuda()
        #     real_box = self.mailbox[nid.to(torch.int64)].cuda()
        #     print(f"input_mail 中不符mailbox的节点数: {torch.sum(real_box != self.part_mailbox[table2])}")
        #     print(f"input_mail 中不符mem的节点数: {torch.sum(real_mem != self.part_memory[table2])}")
        #     if (dis_ind.shape[0] + torch.sum(real_box != self.part_mailbox[table2]) + torch.sum(real_mem != self.part_memory[table2]) > 0):
        #         raise BufferError("buffer内部出现与非缓存情况不符合的事故!")

        b.srcdata['mem'] = self.part_memory[table2].cuda()
        b.srcdata['mem_ts'] = self.part_memory_ts[table2].cuda()
        b.srcdata['mem_input'] = self.part_mailbox[table2].reshape(b.srcdata['ID'].shape[0], -1).cuda()
        b.srcdata['mail_ts'] = self.part_mailbox_ts[table2].cuda()


    #TODO 下一个预采样时需要将mem数据刷入回去
    def update_mailbox(self, nid, mail, mail_ts):
        #将part_mail中的nid改成mail和mail_ts
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)

        self.part_mailbox[table2, 0] = mail
        self.part_mailbox_ts[table2, 0] = mail_ts

    def update_memory(self, nid, memory, ts):
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)

        self.part_memory[table2] = memory
        self.part_memory_ts[table2] = ts

    def get_e_feat(self, eid):
        table1 = torch.zeros_like(self.part_edge_map) - 1
        table2 = torch.zeros_like(eid) - 1
        dgl.findSameIndex(self.part_edge_map, eid, table1, table2)

        # print(f"缓存的边特征中,未命中的{torch.nonzero(table2 == -1).shape[0]}")
        res = self.part_edge_feats[table2.to(torch.int64)]
        if (self.err_detection):
            err_num = torch.sum(res.cpu() != self.select_index('edge_feats',eid.long().cpu()))
            # print(f"edge缓存...与非缓存对比不一致的个数: {err_num}")
            if (err_num + torch.nonzero(table2 == -1).shape[0]):
                raise BufferError("buffer内部出现与非缓存情况不符合的事故!")

        #table2[i]表示
        return res.cuda()

    def get_n_feat(self, nid):
        start = time.time()
        table1 = torch.zeros_like(self.part_node_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_node_map, nid, table1, table2)

        # print(f"缓存的节点特征中,未命中的{torch.nonzero(table2 == -1).shape[0]}")
        res = self.part_node_feats[table2.to(torch.int64)]
        if (self.err_detection):
            err_num = torch.sum(res.cpu() != self.select_index('node_feats',nid.long().cpu()))
            if (err_num + torch.nonzero(table2 == -1).shape[0]):
                print(f"节点缓存...与非缓存对比不一致的个数: {err_num}")
                raise BufferError("buffer内部出现与非缓存情况不符合的事故!")
        
        # print(f"get_n_feat时间{time.time() - start:.8f}")
        return res.cuda()

    def pre_fetch(self, block_num, memory_info):
        #1.预采样下一个块的负节点的neg_nodes和neg_eids
        #2.预取下一个块的正边采样出现的nodes和eids
        #3.将1和2获得的组成下一个块会出现的所有的nodes和eids
        #4.根据下一个块出现的nodes和eids，得到需要增量加载的incre_nodes_mask和incre_eids_mask（incre即当前处理块中未出现的但下一个块出现的）
        #5.将增量的节点特征、边特征、节点记忆 使用cpu预取出来（后面可以替换为IO）
        #6.对于特征的处理已经结束，之后将增量特征返回即可。对于记忆的处理需要额外做处理(记忆刷入)，块1运行时需要异步刷入块1中不出现而块0中出现的节点记忆
        # print(f"子进程...")
        #TODO 这里暂时不考虑任何增量策略
        neg_nodes, neg_eids = self.pre_neg_sample(block_num)
        neg_nodes, _ = torch.sort(neg_nodes)
        neg_eids, _ = torch.sort(neg_eids)

        path = self.path
        pos_edge_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_edge_feat.pt')
        pos_edge_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_edge_map.pt').cuda()
        pos_node_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_node_feat.pt')
        pos_node_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_node_map.pt').cuda()

        table1 = torch.zeros_like(neg_nodes) - 1
        table2 = torch.zeros_like(pos_node_map) - 1
        dgl.findSameNode(neg_nodes, pos_node_map, table1, table2)
        neg_nodes, pos_node_map, table1, table2 = neg_nodes.cpu(), pos_node_map.cpu(), table1.cpu(), table2.cpu()
        dis_ind = table1 == -1
        dis_neg_nodes = neg_nodes[dis_ind]
        neg_node_feats = self.node_feats[dis_neg_nodes.to(torch.int64)]

        pos_node_map = torch.cat((pos_node_map, dis_neg_nodes))
        pos_node_map,indices = torch.sort(pos_node_map)
        node_feats = torch.cat((pos_node_feats, neg_node_feats))
        node_feats = node_feats[indices]


        table1 = torch.zeros_like(neg_eids) - 1
        table2 = torch.zeros_like(pos_edge_map) - 1
        dgl.findSameNode(neg_eids, pos_edge_map, table1, table2)
        neg_eids, pos_edge_map, table1, table2 = neg_eids.cpu(), pos_edge_map.cpu(), table1.cpu(), table2.cpu()
        dis_ind = table1 == -1
        dis_neg_eids = neg_eids[dis_ind]
        dis_neg_eids_feat = self.edge_feats[dis_neg_eids.to(torch.int64)]

        pos_edge_map = torch.cat((pos_edge_map, dis_neg_eids))
        pos_edge_map,indices = torch.sort(pos_edge_map)
        edge_feats = torch.cat((pos_edge_feats, dis_neg_eids_feat))
        edge_feats = edge_feats[indices]

        #此时获得了下一个块的所有nodes eids node_feats edge_feats
        #分别为 pos_node_map pos_edge_map node_feats edge_feats
        #还需要获得下一个块的所有node_memory信息
        if (memory_info):
            #当前正在异步运行块i，需要将块(i-1)的memory信息传入并做刷入，
            part_memory_map, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = memory_info
            part_map = part_memory_map.cpu().clone().long()
            self.memory[part_map] = part_memory.cpu().clone()
            self.memory_ts[part_map] = part_memory_ts.cpu().clone()
            self.mailbox[part_map] = part_mailbox.cpu().clone()
            self.mailbox_ts[part_map] = part_mailbox_ts.cpu().clone()

        nodes = pos_node_map.long()
        part_memory = self.memory[nodes]
        part_memory_ts = self.memory_ts[nodes]
        part_mailbox = self.mailbox[nodes]
        part_mailbox_ts = self.mailbox_ts[nodes]
        #此处要返回的memory信息不包括当前执行块的，因此需要在后面处理加上当前处理块后的最新memory信息
        #即当前异步处理的块的memory结果会实时更新到self.memory中，这里返回的下一个块需要用到的memory需要在下一个块开始之前和self.memory结合
        #因此这里预先判断：当前异步处理的块中出现的节点哪些在下一个块也出现了，即self.part_node_map(当前块)和pos_node_map(下一个块)的关系
        
        pre_same_nodes = torch.isin(self.part_node_map.cpu(), pos_node_map) #mask形式的
        cur_same_nodes = torch.isin(pos_node_map, self.part_node_map.cpu())

        print(f"pre fetch over...")
        self.preFetchDataCache.put({'node_info': [pos_node_map, node_feats], 'edge_info': [pos_edge_map, edge_feats],\
                                     'memory_info': [part_memory, part_memory_ts, part_mailbox, part_mailbox_ts],\
                                        'memory_update_info': [pre_same_nodes, cur_same_nodes]})

        #nodes做sort + unique找出最终的indices

    def test(self, pipe):
        #做CPU开销
        for i in range(10):
            time.sleep(1)

            # tensor = torch.arange(1000000000)

            # path = '/raid/guorui/DG/dataset/STACK/part-600000-[10]/part1_edge_feat.pt'
            test_feats = torch.load(f'/raid/guorui/DG/dataset/STACK/part-600000-[10]/part{i+30}_edge_feat.pt')
        
        print(f'test over..')

    def move_to_gpu(self, datas, flag=False, use_pin = False):
        res = []
        for data in datas:
            try:
                if (use_pin):
                    data = data.pin_memory()
                else:
                    data = data.cuda()
                res.append(data)
            except RuntimeError as e:
                if (flag):
                    print(f"清除Cache仍然无法分配显存...")
                    exit(-1)
                print(f"显存OOM, 尝试清除cache")
                emptyCache()
                res.append(self.move_to_gpu([data], flag=True, use_pin=use_pin)[0])

        return res


    def prefetch_after(self):

        node_num = self.shared_ret_len[0]
        edge_num = self.shared_ret_len[2]
        pre_num = self.shared_ret_len[-2]
        cur_num = self.shared_ret_len[-1]
        use_pin = hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory

        allo1 = cuda_GB()

        self.part_edge_map, self.part_edge_feats = self.share_part_edge_map[:edge_num], self.share_part_edge_feats[:edge_num]
        self.part_edge_map = self.move_to_gpu([self.part_edge_map])[0]
        self.part_edge_feats = self.move_to_gpu([self.part_edge_feats], use_pin=use_pin)[0]

        self.part_node_map, self.part_node_feats = self.share_part_node_map[:node_num], self.share_part_node_feats[:node_num]
        self.part_node_map = self.move_to_gpu([self.part_node_map])[0]
        self.part_node_feats = self.move_to_gpu([self.part_node_feats], use_pin = use_pin)[0]


        self.part_memory_map = self.part_node_map

        if (self.use_memory):
            part_memory, part_memory_ts, part_mailbox, part_mailbox_ts \
                = self.share_part_memory[:node_num], self.share_part_memory_ts[:node_num], self.share_part_mailbox[:node_num], self.share_part_mailbox_ts[:node_num]
            
            part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = self.move_to_gpu([part_memory, part_memory_ts, part_mailbox, part_mailbox_ts])
        
            pre_same_nodes, cur_same_node = self.share_pre_same_nodes[:pre_num], self.share_cur_same_nodes[:cur_num]
            pre_same_nodes, cur_same_node = pre_same_nodes.cuda(), cur_same_node.cuda()

            part_memory[cur_same_node] = self.part_memory[pre_same_nodes]
            part_memory_ts[cur_same_node] = self.part_memory_ts[pre_same_nodes]
            part_mailbox[cur_same_node] = self.part_mailbox[pre_same_nodes]
            part_mailbox_ts[cur_same_node] = self.part_mailbox_ts[pre_same_nodes]

            self.part_memory, self.part_memory_ts,self.part_mailbox,self.part_mailbox_ts = part_memory,part_memory_ts,part_mailbox,part_mailbox_ts

        allo2 = cuda_GB()
        # print(f"prefetch_after: {allo1} -> {allo2}")

    def run_batch(self, cur_batch):
        #主程序正准备执行batch i的时候，判断是否需要预取IO
        
        self.cur_batch = cur_batch
        if (cur_batch % self.batch_num == 0):
            # print(f"cur batch: {cur_batch}, start pre sample and analyze...")
            

            if (not self.use_ayscn):
                time_load_s = time.time()
                self.load_part(cur_batch // self.batch_num)
                self.time_load += time.time() - time_load_s

                time_ana_s = time.time()
                self.analyze()
                self.time_analyze += time.time() - time_ana_s
                self.neg_sample_nodes = self.neg_sample_nodes_async

                #TODO 做测试
                # self.pool.apply_async(self.test, args=[self.pipe])
            else:
                #使用异步策略
                if (self.cur_block == 0):
                    #第一个块使用同步加载
                    time_first = time.time()
                    time_load_s = time.time()
                    if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
                        self.load_part_pin(cur_batch // self.batch_num)
                    else:
                        self.load_part(cur_batch // self.batch_num)
                    self.time_load += time.time() - time_load_s

                    time_ana_s = time.time()
                    self.analyze()
                    self.time_analyze += time.time() - time_ana_s
                    print(f"第一个块的加载时间: {time.time() - time_first:.3f}s")

                    
                else:
                    #后面的块直接加载异步的Queue队列信息并做进一步处理
                    time_load_s = time.time()
                    flag = self.prefetch_only_conn.recv()
                    self.time_async += time.time() - time_load_s

                    if (flag is not None and 'extension' in flag):
                        print(f"子程序需要扩容,主程序进行share tensor扩容")

                    self.prefetch_after()

                    self.time_load += time.time() - time_load_s

                time_ana_s = time.time()
                self.neg_sample_nodes = self.neg_sample_nodes_async
                memory_info = (self.part_memory_map, self.part_memory, self.part_memory_ts, self.part_mailbox, self.part_mailbox_ts)
                #此时开始运行当前块，我们异步的加载下一个块的信息

                # print(f"运行子进程，传入memory_info")
                # self.pool.apply_async(self.pre_fetch, args=(self.cur_block + 1, memory_info, self.pipe[1]))

                neg_info = self.pre_neg_sample(self.cur_block + 1)
                if (neg_info is not None):
                    self.prefetch_conn.send(('pre_fetch', (self.cur_block + 1,memory_info,neg_info, self.part_node_map,\
                                            (self.path, self.batch_size, self.sampler.fan_nums))))
                # self.preFetchExecutor.submit(self.pre_fetch,self.cur_block + 1,\
                #      memory_info)
                self.time_analyze += time.time() - time_ana_s



            self.cur_block += 1


    def load_part(self, part_num):
        path = self.path
        self.part_edge_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_feat.pt').cuda()
        self.part_edge_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_map.pt').cuda()
        self.part_node_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_feat.pt').cuda()
        self.part_node_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_map.pt').cuda()

    def load_part_pin(self, part_num):
        path = self.path
        self.part_edge_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_feat.pt').pin_memory()
        self.part_edge_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_map.pt').cuda()
        self.part_node_feats = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_feat.pt').pin_memory()
        self.part_node_map = torch.load(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_map.pt').cuda()

    def pre_sample(self):
        time_presample_s = time.time()
        #需要根据self.cur_batch判断从何处开始计算
        emptyCache()
        start = self.cur_batch * self.train_batch_size
        end = min(self.train_edge_end, (self.cur_batch + self.batch_num) * self.train_batch_size)

        df = self.df

        rows = df[start:end]
        self.neg_sample_nodes = self.neg_sampler.sample(len(rows))
        root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, self.neg_sample_nodes]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()


        ret_list = self.sampler.sample_layer(root_nodes, root_ts)
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]

        mask = src > -1
        src = src[mask]
        dst = dst[mask]
        del dst, mask, outts, outeid, root_ts, ret_list
        emptyCache()
        nodes = torch.cat((src, root_nodes))
        nodes = torch.unique(nodes)
        
        self.time_presample += time.time() - time_presample_s
        return nodes

    def pre_neg_sample(self, block_num):
        #需要根据self.cur_batch判断从何处开始计算
        time_pre_neg_s = time.time()
        start = (block_num * self.batch_num) * self.train_batch_size
        end = min(self.train_edge_end, ((block_num * self.batch_num) + self.batch_num) * self.train_batch_size)
        if(start >= end):
            return None

        df = self.df

        rows = df[start:end]
        

        self.neg_sample_nodes_async = self.neg_sampler.sample(len(rows))
        root_nodes = torch.from_numpy(np.concatenate([self.neg_sample_nodes_async]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([rows.time.values]).astype(np.float32)).cuda()


        ret_list = self.sampler.sample_layer(root_nodes, root_ts)
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
        
        self.time_pre_neg_sample += time.time() - time_pre_neg_s
        return nodes, eid_uni

    def refresh_memory(self):
        time_refresh_s = time.time()
        if (not self.use_memory):
            return
        
        if (self.part_memory) != None:
            part_map = self.part_memory_map.long()
            self.update_index('memory', part_map, self.part_memory.cpu())
            self.update_index('memory_ts', part_map, self.part_memory_ts.cpu())
            self.update_index('mailbox', part_map, self.part_mailbox.cpu())
            self.update_index('mailbox_ts', part_map, self.part_mailbox_ts.cpu())
            # self.memory[part_map] = self.part_memory.cpu()
            # self.memory_ts[part_map] = self.part_memory_ts.cpu()
            # self.mailbox[part_map] = self.part_mailbox.cpu()
            # self.mailbox_ts[part_map] = self.part_mailbox_ts.cpu()
        self.time_refresh += time.time() - time_refresh_s

    
    def analyze_mem(self, nodes):
        time_exec_mem_s = time.time()
        self.refresh_memory()

        if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory and False):
            self.part_memory = self.select_index('memory',nodes).pin_memory()
            self.part_memory_ts = self.select_index('memory_ts',nodes).pin_memory()
            self.part_mailbox = self.select_index('mailbox',nodes).pin_memory()
            self.part_mailbox_ts = self.select_index('mailbox_ts',nodes).pin_memory()
        else:
            self.part_memory = self.select_index('memory',nodes).cuda()
            self.part_memory_ts = self.select_index('memory_ts',nodes).cuda()
            self.part_mailbox = self.select_index('mailbox',nodes).cuda()
            self.part_mailbox_ts = self.select_index('mailbox_ts',nodes).cuda()

        self.part_memory_map = nodes.to(torch.int32).cuda()
        self.time_exec_mem += time.time() - time_exec_mem_s

    def analyze_async(self):
        #异步做处理..

        
        aasdas = 1

    #预采样分析 所有节点的memory通过cpu提取作为gpu缓存
    def analyze(self):
        #TODO 预采样去IO里取， memory lazy 刷入
        #这里的预采样目的只是获取出现的所有节点来更新memory，因此只需要做负采样在结合io加载的node即可实现
        # nodes = self.pre_sample()
        # nodes, _ = torch.sort(nodes)
        # nodes = nodes.long()


        #专门做一次负采样，缓存负采样带来的节点特征和边特征
        
        neg_nodes,neg_eids = self.pre_neg_sample(self.cur_batch // self.batch_num)

        time_neg_analyze_s = time.time()
        neg_nodes, _ = torch.sort(neg_nodes)
        neg_eids,_ = torch.sort(neg_eids)

        #找出map中没有的
        table1 = torch.zeros_like(neg_nodes) - 1
        table2 = torch.zeros_like(self.part_node_map) - 1
        dgl.findSameNode(neg_nodes, self.part_node_map, table1, table2)
        dis_ind = table1 == -1
        dis_neg_nodes = neg_nodes[dis_ind]
        dis_neg_nodes_feat = self.select_index('node_feats', dis_neg_nodes.to(torch.int64)).cuda()
        if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
            dis_neg_nodes_feat = dis_neg_nodes_feat.cpu().pin_memory()

        self.part_node_map = torch.cat((self.part_node_map, dis_neg_nodes))
        self.part_node_map,indices = torch.sort(self.part_node_map)
        self.part_node_feats = torch.cat((self.part_node_feats, dis_neg_nodes_feat))
        self.part_node_feats = self.part_node_feats[indices]



        table1 = torch.zeros_like(neg_eids) - 1
        table2 = torch.zeros_like(self.part_edge_map) - 1
        dgl.findSameNode(neg_eids, self.part_edge_map, table1, table2)
        dis_ind = table1 == -1
        dis_neg_eids = neg_eids[dis_ind]
        dis_neg_eids_feat = self.select_index('edge_feats',dis_neg_eids.to(torch.int64)).cuda()
        if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
            dis_neg_eids_feat = dis_neg_eids_feat.cpu().pin_memory()

        
        self.part_edge_map = torch.cat((self.part_edge_map, dis_neg_eids))
        self.part_edge_map,indices = torch.sort(self.part_edge_map)
        #TODO 当part_edge过大的时候，这里会产生一个较大的显存管理开销
        self.part_edge_feats = torch.cat((self.part_edge_feats, dis_neg_eids_feat))
        self.part_edge_feats = self.part_edge_feats[indices]

        self.time_neg_analyze += time.time() - time_neg_analyze_s


        #此时nodes可以直接用self.part_node_map代替
        # nodes是预采样中出现过的所有节点的id (unique的)
        #先将当前part_memory中缓存的memory刷回内存上的memory
        # print(f"nodes和map的差别： {torch.sum(nodes != self.part_node_map)}")
        nodes = self.part_node_map.long()
        
        if (self.use_memory):
            self.analyze_mem(nodes)


    
    def gen_part(self):
        #当分区feat不存在的时候做输出
        d = self.d
        path = self.path
        # if os.path.exists(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}'):
        #     print(f"already  partfeat")
        #     return

        df = self.df
        batch_size = self.batch_size
        # node_feats, edge_feats = load_feat(d)
        train_edge_end = self.train_edge_end
        group_indexes = np.array(df[:train_edge_end].index // batch_size)

        for batch_num, rows in df[:train_edge_end].groupby(group_indexes):
            emptyCache()
            root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
            root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()


            start = time.time()
            ret_list = self.sampler.sample_layer(root_nodes, root_ts)
            eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
            nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

            for ret in ret_list:
                #找出每层的所有eid即可
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                eid = outeid[outeid > -1]

                cur_eid = torch.unique(eid)
                eid_uni = torch.cat((cur_eid, eid_uni))
                eid_uni = torch.unique(eid_uni)
            
            #前面层出现的节点会在最后一层的dst中出现,因此所有节点就是最后一层的Src,dst
            ret = ret_list[-1]
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            del ret_list
            del outts, outeid, root_ts, dst
            emptyCache()

            mask = src > -1
            src = src[mask]
            nid_uni = torch.cat((src, root_nodes))
            nid_uni = torch.unique(nid_uni)

            #处理这个eid_uni，抽特征然后存就行。这里eid是个全局的
            #存起来后需要保存一个map，map[i]表示e_feat[i]保存的是哪条边的特征即eid
            #这里对eid进行排序，目的是保证map是顺序的，在后面就可以不对map排序了
            eid_uni,_ = torch.sort(eid_uni)
            cur_edge_feat = self.select_index('edge_feats',eid_uni.to(torch.int64))
            saveBin(cur_edge_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_feat.pt')
            saveBin(eid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_map.pt')

            nid_uni,_ = torch.sort(nid_uni)
            cur_node_feat = self.select_index('node_feats',nid_uni.to(torch.int64))
            saveBin(cur_node_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_feat.pt')
            saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_map.pt')

            sampleTime = time.time() - start
            # mfgs = sampler.gen_mfgs(ret_list)
            
            print(f"{root_nodes.shape}单层单block采样 + 转换block + 存储block数据 batch: {batch_num} batchsize: {batch_size} 用时:{time.time() - start:.7f}s")
            del root_nodes,eid_uni,nid_uni,src,mask,eid,_


    
    def incre_strategy(self, pre_map, cur_id):
        # 要做的就是找出上一次没有出现过的节点
        pre_map_sort, pre_map_sort_indices = torch.sort(pre_map)
        cur_id_sort, cur_id_sort_indices = torch.sort(cur_id)
        table1 = torch.zeros_like(pre_map_sort, dtype = torch.int32, device = 'cuda:0') - 1
        table2 = torch.zeros_like(cur_id_sort, dtype = torch.int32, device = 'cuda:0') - 1
        dgl.findSameIndex(pre_map_sort, cur_id_sort, table1, table2)

        # table1[pre_map_sort_indices]
        # table2[cur_id_sort_indices] #这个是cur_id中对应的pre_map_sort中出现的位置
        # ind1 = torch.argsort(pre_map_sort_indices)
        incre_map_index = pre_map_sort_indices[table1==-1] #map中需要改变的部分
        incre_nid = cur_id[cur_id_sort_indices[table2==-1].long()] #需要增量加载的部分
        print(f"实际个数: {cur_id.shape[0]} 需要增量加载 {incre_nid.shape[0]}个")
        incre_nfeat = torch.randn((incre_nid.shape[0],10), dtype = torch.float32)
        pre_map[incre_map_index] = -1

        #分配incre_indices(表示incre_nfeats要替换哪些已有的)
        indices_able = torch.nonzero(pre_map == -1).reshape(-1)
        len_able = indices_able.shape[0]
        if (len_able < incre_nid.shape[0]):
            # print(f"需要扩容")
            pre_map = torch.cat((pre_map, torch.zeros(incre_nid.shape[0] - len_able, dtype = torch.int32, device = 'cuda:0') - 1))
            indices_able = torch.nonzero(pre_map == -1).reshape(-1)

        incre_indices = indices_able[:incre_nid.shape[0]]
        pre_map[incre_indices] = incre_nid

        # print(pre_map)
        # print(incre_indices)
        # print(incre_nid)
        return pre_map, incre_indices, incre_nid


    def gen_part_incre(self):
        #当分区feat不存在的时候做输出
        d = self.d
        path = self.path
        # if os.path.exists(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}'):
        #     print(f"already  partfeat")
        #     return

        df = self.df
        batch_size = self.batch_size
        # node_feats, edge_feats = load_feat(d)
        train_edge_end = self.train_edge_end
        group_indexes = np.array(df[:train_edge_end].index // batch_size)

        edge_map, node_map = None,None

        for batch_num, rows in df[:train_edge_end].groupby(group_indexes):
            emptyCache()
            root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
            root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()


            start = time.time()
            ret_list = self.sampler.sample_layer(root_nodes, root_ts)
            eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
            nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

            for ret in ret_list:
                #找出每层的所有eid即可
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                eid = outeid[outeid > -1]

                cur_eid = torch.unique(eid)
                eid_uni = torch.cat((cur_eid, eid_uni))
                eid_uni = torch.unique(eid_uni)
            
            #前面层出现的节点会在最后一层的dst中出现,因此所有节点就是最后一层的Src,dst
            ret = ret_list[-1]
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            del ret_list
            del outts, outeid, root_ts, dst
            emptyCache()

            mask = src > -1
            src = src[mask]
            nid_uni = torch.cat((src, root_nodes))
            nid_uni = torch.unique(nid_uni)

            #处理这个eid_uni，抽特征然后存就行。这里eid是个全局的
            #存起来后需要保存一个map，map[i]表示e_feat[i]保存的是哪条边的特征即eid
            #这里对eid进行排序，目的是保证map是顺序的，在后面就可以不对map排序了
            eid_uni,_ = torch.sort(eid_uni)
            # cur_edge_feat = self.select_index('edge_feats',eid_uni.to(torch.int64))
            if (edge_map is None):
                edge_map = eid_uni
            edge_map, incre_edge_indices, incre_eid = self.incre_strategy(edge_map, eid_uni)
            cur_edge_feat = self.select_index('edge_feats',incre_eid.to(torch.int64))
            saveBin(cur_edge_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_edge_feat.pt')
            saveBin(edge_map.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_edge_map.pt')
            saveBin(incre_edge_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_edge_indices.pt')

            nid_uni,_ = torch.sort(nid_uni)
            # cur_node_feat = self.select_index('node_feats',nid_uni.to(torch.int64))
            if (node_map is None):
                node_map = nid_uni
            node_map, incre_node_indices, incre_nid = self.incre_strategy(node_map, nid_uni)
            cur_node_feat = self.select_index('node_feats',incre_nid.to(torch.int64))
            saveBin(cur_node_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_node_feat.pt')
            saveBin(node_map.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_node_map.pt')
            saveBin(incre_node_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_incre_node_indices.pt')

            sampleTime = time.time() - start
            # mfgs = sampler.gen_mfgs(ret_list)
            
            print(f"{root_nodes.shape}单层单block采样 + 转换block + 存储block数据 batch: {batch_num} batchsize: {batch_size} 用时:{time.time() - start:.7f}s")
            del root_nodes,eid_uni,nid_uni,src,mask,eid,_
