import multiprocessing
import time
import random
import torch
import dgl
import os
from utils import load_feat, emptyCache
from config.train_conf import *

import threading

class Pre_fetch:
    def __init__(self, conn, prefetch_child_conn):
        self.conn = conn
        self.prefetch_conn = prefetch_child_conn

        self.config = GlobalConfig()
        self.use_valid_edge = self.config.use_valid_edge

        self.async_load_dic = {}
        self.async_load_flag = {}

    def init_share_tensor(self, shared_tensor):
        part_node_map, node_feats, part_edge_map, edge_feats, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes, shared_ret_len, share_tmp_tensor = shared_tensor
        self.part_node_map = part_node_map
        self.part_node_feats = node_feats
        self.part_edge_map = part_edge_map
        self.part_edge_feats = edge_feats
        self.part_memory = part_memory
        self.part_memory_ts = part_memory_ts
        self.part_mailbox = part_mailbox
        self.part_mailbox_ts = part_mailbox_ts
        self.pre_same_nodes = pre_same_nodes
        self.cur_same_nodes = cur_same_nodes

        self.shared_ret_len = shared_ret_len

        self.share_tmp = share_tmp_tensor

    def prefetch_after(self, prefetch_res):
        node_info = prefetch_res
        
        total_allo = 0
        for i, tensor in enumerate(prefetch_res):
            self.shared_ret_len[i] = tensor.shape[0] if tensor is not None else 0
            total_allo += tensor.reshape(-1).shape[0] if tensor is not None else 0

        self.part_node_map[:self.shared_ret_len[0]] = prefetch_res[0]
        self.part_node_feats[:self.shared_ret_len[1]] = prefetch_res[1]
        self.part_edge_map[:self.shared_ret_len[2]] = prefetch_res[2]
        self.part_edge_feats[:self.shared_ret_len[3]] = prefetch_res[3]

        if (hasattr(self, 'memory')):
            self.part_memory[:self.shared_ret_len[4]] = prefetch_res[4]
            self.part_memory_ts[:self.shared_ret_len[5]] = prefetch_res[5]
            self.part_mailbox[:self.shared_ret_len[6]] = prefetch_res[6]
            self.part_mailbox_ts[:self.shared_ret_len[7]] = prefetch_res[7]

            self.pre_same_nodes[:self.shared_ret_len[8]] = prefetch_res[8]
            self.cur_same_nodes[:self.shared_ret_len[9]] = prefetch_res[9]

        # print(f"pre_fetch处理后cuda需要新增 {total_allo * 4 / 1024**3:.4f}GB显存 边数为: {self.shared_ret_len[2]} 节点数为{self.shared_ret_len[0]}")

    def select_index(self, name, indices):
        # print(f"子程序 select {name}")
        self_v = getattr(self,name,None)
        if (self_v is not None):
            if (name == 'edge_feats' and self.use_valid_edge):
                print(f"use valid ef...")
                res = self.get_ef_valid(indices)
            else:
                res = self_v[indices]
            dim = res.shape
            res = res.reshape(-1)
            shape = res.shape[0]
            self.share_tmp = self.share_tmp.to(res.dtype)
            self.share_tmp[:res.shape[0]] = res
            return (dim, shape)
        else:
            raise RuntimeError(f'pre_fetch {name} error')

    def update_index(self, name, indices, conf):
        # print(f"子程序 update {name}")
        self_v = getattr(self,name, None)
        if (self_v is not None):
            dim, shape = conf
            self_v[indices] = self.share_tmp[:shape].reshape(dim)
        else:
            raise RuntimeError(f'pre_fetch {name} error')


    def function(self, args, tensor_res):
        # 模拟执行任务
        # res = torch.randperm(100000000, dtype = torch.int32)
        print(f"share tensor 长度: {self.part_node_map.shape}")
        tensor_res[:] = 1

        self.part_node_map[0] = 1321213
        print(f"测试管道传递，传递{tensor_res.shape[0] * 4 / 1024 ** 2:.3f}MB的torch向量")
        return None

    def init_memory(self, memory_param, num_nodes, dim_edge_feat):
        self.memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32)
        self.memory_ts = torch.zeros(num_nodes, dtype=torch.float32)
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32)
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32)
        
    def reset_memory(self):
        self.memory.fill_(0)
        self.memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)

    def init_feats(self, dataset):
        # node_feats, edge_feats = load_feat(dataset)
        # self.node_feats = node_feats

        if (not self.use_valid_edge):
            node_feats, edge_feats = load_feat(dataset)
            self.node_feats = node_feats
            self.edge_feats = edge_feats
        else:
            node_feats, edge_feats = load_feat(dataset, load_edge=False)
            self.node_feats = node_feats
            self.edge_feats = []

    def init_valid_edge(self, max_valid_num, edge_feat_len, conf):
        # 淘汰过期边策略 这里先对TALK数据集
        self.path, self.batch_size, self.fan_nums = conf

        self.valid_ef = torch.zeros((max_valid_num, edge_feat_len), dtype = torch.float32)

        self.update_valid_edge(0)

    def update_valid_edge(self, block_num):
        cur_ef = torch.load(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre.pt')
        cur_map = torch.load(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_map.pt')
        replace_idx = torch.load(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_replace.pt')
        self.valid_map = cur_map
        self.valid_ef[replace_idx.long()] = cur_ef

    def get_ef_valid(self, eids):

        eids = eids.cpu()
        map = self.valid_map
        sorted_map, indices = torch.sort(map)
        positions = torch.searchsorted(sorted_map, eids)

        original_positions = indices[positions]
        return self.valid_ef[original_positions.long()]


    def load_file(self, paths, tags, i):

        if (os.path.exists(paths[i])):
            self.async_load_dic[tags[i]] = torch.load(paths[i])
        else:
            self.async_load_dic[tags[i]] = None


        #这个完成之后加载下一个
        if ((i + 1) < len(paths)):
            thread = threading.Thread(target=self.load_file, args=(paths, tags, i + 1))
            self.async_load_flag[tags[i + 1]] = thread
            thread.start()
    
    def async_load(self, paths, tags):
        thread = threading.Thread(target=self.load_file, args=(paths, tags, 0))
        self.async_load_flag[tags[0]] = thread
        thread.start()

    def pre_fetch(self, block_num, memory_info, neg_info, part_node_map, conf):
        #1.预采样下一个块的负节点的neg_nodes和neg_eids
        #2.预取下一个块的正边采样出现的nodes和eids
        #3.将1和2获得的组成下一个块会出现的所有的nodes和eids
        #4.根据下一个块出现的nodes和eids，得到需要增量加载的incre_nodes_mask和incre_eids_mask（incre即当前处理块中未出现的但下一个块出现的）
        #5.将增量的节点特征、边特征、节点记忆 使用cpu预取出来（后面可以替换为IO）
        #6.对于特征的处理已经结束，之后将增量特征返回即可。对于记忆的处理需要额外做处理(记忆刷入)，块1运行时需要异步刷入块1中不出现而块0中出现的节点记忆
        # print(f"子进程...")
        #TODO 这里暂时不考虑任何增量策略
        # print(f"子线程prefetch")
        has_ef = self.edge_feats.shape[0] > 0
        has_nf = self.node_feats.shape[0] > 0
        t0 = time.time()
        neg_nodes, neg_eids = neg_info
        neg_nodes,neg_eids = neg_nodes.cpu(),neg_eids.cpu()
        path, batch_size, fan_nums = conf

        #incre_ef, part_ef, part_nf
        if (self.use_valid_edge):
            load_paths = [path + f'/part-{self.batch_size}/part{block_num}_edge_incre.pt', path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat.pt']
            tags = ['valid_edge_feat', 'part_edge_feat', 'part_node_feat']
        else:
            load_paths = [path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat.pt']
            tags = ['part_edge_feat', 'part_node_feat']
        
        self.async_load(load_paths, tags)

        # neg_nodes, _ = torch.sort(neg_nodes)
        # neg_eids, _ = torch.sort(neg_eids)

        part_node_map = part_node_map.cpu()

        if (memory_info and memory_info[1] is not None):
            #当前正在异步运行块i，需要将块(i-1)的memory信息传入并做刷入，
            part_memory_map, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = memory_info
            part_map = part_memory_map.cpu().long()
            self.memory[part_map] = part_memory.cpu()
            self.memory_ts[part_map] = part_memory_ts.cpu()
            self.mailbox[part_map] = part_mailbox.cpu()
            self.mailbox_ts[part_map] = part_mailbox_ts.cpu()
        
            del part_memory_map, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, neg_info
            # emptyCache()
        
        
        t1 = time.time() - t0
        t0 = time.time()

        # pos_edge_feats = torch.load(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt')
        pos_edge_map = torch.load(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_map.pt')
        # pos_node_feats = torch.load(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat.pt')
        pos_node_map = torch.load(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_map.pt')
        t2 = time.time() - t0
        t0 = time.time()

        t3 = time.time() - t0
        t0 = time.time()
        # table1 = torch.zeros_like(neg_nodes) - 1
        # table2 = torch.zeros_like(pos_node_map) - 1
        # dgl.findSameNode(neg_nodes, pos_node_map, table1, table2)
        # neg_nodes, pos_node_map, table1, table2 = neg_nodes.cpu(), pos_node_map.cpu(), table1.cpu(), table2.cpu()
        # dis_ind = table1 == -1
        dis_ind = torch.isin(neg_nodes, pos_node_map, assume_unique=True,invert=True)
        dis_neg_nodes = neg_nodes[dis_ind]
        if (has_nf):
            neg_node_feats = self.node_feats[dis_neg_nodes.to(torch.int64)]

        pos_node_map = torch.cat((pos_node_map, dis_neg_nodes))
        pos_node_map,node_indices = torch.sort(pos_node_map)
        


        # table1 = torch.zeros_like(neg_eids) - 1
        # table2 = torch.zeros_like(pos_edge_map) - 1
        # dgl.findSameNode(neg_eids, pos_edge_map, table1, table2)
        # neg_eids, pos_edge_map, table1, table2 = neg_eids.cpu(), pos_edge_map.cpu(), table1.cpu(), table2.cpu()
        # dis_ind = table1 == -1
        dis_ind = torch.isin(neg_eids, pos_edge_map, assume_unique=True,invert=True)
        dis_neg_eids = neg_eids[dis_ind]
        # print(f"neg_nodes: {neg_nodes.shape[0]}, neg_eids: {neg_eids.shape[0]}, dis_neg_nodes: {dis_neg_nodes.shape[0]},dis_neg_eids: {dis_neg_eids.shape[0]}")
        t4 = time.time() - t0
        t0 = time.time()


        if (has_ef and not self.use_valid_edge):
            dis_neg_eids_feat = self.edge_feats[dis_neg_eids.to(torch.int64)]


        t5 = time.time() - t0
        t0 = time.time()

        pos_edge_map = torch.cat((pos_edge_map, dis_neg_eids))
        pos_edge_map,edge_indices = torch.sort(pos_edge_map)
        
        t6 = time.time() - t0
        t0 = time.time()

        #此时获得了下一个块的所有nodes eids node_feats edge_feats
        #分别为 pos_node_map pos_edge_map node_feats edge_feats
        #还需要获得下一个块的所有node_memory信息
        
        t7 = time.time() - t0
        t0 = time.time()

        nodes = pos_node_map.long()
        if (hasattr(self, 'memory')):
            part_memory = self.memory[nodes]
            part_memory_ts = self.memory_ts[nodes]
            part_mailbox = self.mailbox[nodes]
            part_mailbox_ts = self.mailbox_ts[nodes]

            #此处要返回的memory信息不包括当前执行块的，因此需要在后面处理加上当前处理块后的最新memory信息
            #即当前异步处理的块的memory结果会实时更新到self.memory中，这里返回的下一个块需要用到的memory需要在下一个块开始之前和self.memory结合
            #因此这里预先判断：当前异步处理的块中出现的节点哪些在下一个块也出现了，即self.part_node_map(当前块)和pos_node_map(下一个块)的关系
            
            pre_same_nodes = torch.isin(part_node_map.cpu(), pos_node_map) #mask形式的
            cur_same_nodes = torch.isin(pos_node_map, part_node_map.cpu())
        else:
            part_memory = torch.empty(0)
            part_memory_ts = torch.empty(0)
            part_mailbox = torch.empty(0)
            part_mailbox_ts = torch.empty(0)
            pre_same_nodes = torch.empty(0)
            cur_same_nodes = torch.empty(0)
        
        t8 = time.time() - t0
        t0 = time.time()

        # print(f"pre fetch over...")
        if (pos_node_map.shape[0] > self.part_node_map.shape[0]):
            self.prefetch_conn.send('node extension')
            self.prefetch_conn.recv()
        if (pos_edge_map.shape[0] > self.part_edge_map.shape[0]):
            self.prefetch_conn.send('edge extension')
            self.prefetch_conn.recv()
        t9 = time.time() - t0
        t0 = time.time()

        node_feats, edge_feats = torch.empty(0, dtype = torch.float32), torch.empty(0, dtype = torch.float32)
        for tag in tags:
            self.async_load_flag[tag].join()
            data = self.async_load_dic[tag]
            if (self.use_valid_edge and tag == 'valid_edge_feat' and has_ef):
                # print(tag)
                self.update_valid_edge(block_num)

                dis_neg_eids_feat = self.get_ef_valid(dis_neg_eids)
            elif (tag == 'part_node_feat' and has_nf):
                # print(tag)
                pos_node_feats = data
                node_feats = torch.cat((pos_node_feats, neg_node_feats))
                node_feats = node_feats[node_indices]
            elif (tag == 'part_edge_feat' and has_ef):
                # print(tag)
                pos_edge_feats = data
                edge_feats = torch.cat((pos_edge_feats, dis_neg_eids_feat))
                edge_feats = edge_feats[edge_indices]



        self.prefetch_after([pos_node_map, node_feats, pos_edge_map, edge_feats, part_memory,\
                              part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes])
        # self.preFetchDataCache.put({'node_info': [pos_node_map, node_feats], 'edge_info': [pos_edge_map, edge_feats],\
        #                              'memory_info': [part_memory, part_memory_ts, part_mailbox, part_mailbox_ts],\
        #                                 'memory_update_info': [pre_same_nodes, cur_same_nodes]})
        t10 = time.time() - t0
        t0 = time.time()
        #nodes做sort + unique找出最终的indices

        # print(f" {t1:.2f}s\n {t2:.2f}s\n {t3:.2f}s\n {t4:.2f}s\n {t5:.2f}s\n {t6:.2f}s\n {t7:.2f}s\n {t8:.2f}s\n {t9:.2f}s\n {t10:.2f}s\n")
        # print(f"pre fetch over...")

    def run(self):
        while True:
            if self.conn.poll():  # 检查是否有数据
                message = self.conn.recv()
                if message == "EXIT":
                    break
                function_name, args = message
                if hasattr(self, function_name):
                    # print(f"子程序调用程序: {function_name}")
                    func = getattr(self, function_name)
                    result = func(*args)
                    # print(f"子进程result传回")
                    if (function_name == 'pre_fetch'):
                        # print(f"使用prefetch传回")
                        self.prefetch_conn.send(result)
                    else:
                        self.conn.send(result)
                else:
                    self.conn.send(f"Function {function_name} not found")

def prefetch_worker(conn, prefetch_child_conn):
    prefetch = Pre_fetch(conn, prefetch_child_conn)
    prefetch.run()


