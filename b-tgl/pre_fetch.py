import multiprocessing
import time
import random
import torch
import dgl
import os
from utils import *
from config.train_conf import *

import threading

class Pre_fetch:
    def __init__(self, conn, prefetch_child_conn):
        self.conn = conn
        self.prefetch_conn = prefetch_child_conn
        self.use_memory = False

        self.config = GlobalConfig()
        self.use_valid_edge = self.config.use_valid_edge
        self.use_disk = self.config.use_disk
        self.data_incre = self.config.data_incre
        self.memory_disk = self.config.memory_disk
        self.use_async_IO = self.config.use_async_IO
        self.node_cache = self.config.node_cache
        self.node_simple_cache = self.config.node_simple_cache
        self.node_reorder = self.config.node_reorder
        self.use_bucket = self.config.use_bucket
        print(f"pre_fetch: 是否使用disk: {self.use_disk}")

        if (self.use_valid_edge and self.use_disk):
            print(f"无法同时使用过期边淘汰和diskGNN！将过期边淘汰取消.")
            self.use_valid_edge = False



        self.async_load_dic = {}
        self.async_load_flag = {}


    def init_share_tensor(self, shared_tensor):



        part_node_map, node_feats, part_edge_map, edge_feats, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes, shared_node_d_map, shared_edge_d_map, shared_ret_len, share_tmp_tensor = shared_tensor

        self.node_d_ind = shared_node_d_map
        self.edge_d_ind = shared_edge_d_map


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

        if (self.use_memory):
            self.part_memory[:self.shared_ret_len[4]] = prefetch_res[4]
            self.part_memory_ts[:self.shared_ret_len[5]] = prefetch_res[5]
            self.part_mailbox[:self.shared_ret_len[6]] = prefetch_res[6]
            self.part_mailbox_ts[:self.shared_ret_len[7]] = prefetch_res[7]

            self.pre_same_nodes[:self.shared_ret_len[8]] = prefetch_res[8]
            self.cur_same_nodes[:self.shared_ret_len[9]] = prefetch_res[9]

        
        self.node_d_ind[:self.shared_ret_len[10]] = prefetch_res[10]
        self.edge_d_ind[:self.shared_ret_len[11]] = prefetch_res[11]

        # print(f"pre_fetch处理后cuda需要新增 {total_allo * 4 / 1024**3:.4f}GB显存 边数为: {self.shared_ret_len[2]} 节点数为{self.shared_ret_len[0]}")

    def select_index(self, name, indices):
        # print(f"子程序 select {name}")
        self_v = getattr(self,name,None)
        if (self_v is not None):
            if (name == 'edge_feats' and self.use_valid_edge):
                print(f"use valid ef...")
                res = self.get_ef_valid(indices)
            elif (name in ['node_feats', 'edge_feats']):
                res = self.self_select(name, indices)
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

    def init_IO_load(self, conn):
        self.IO_conn = conn
        print(f"pre_fetch文件获取conn")


    def self_select(self, name, indices,  use_slice = False):
        if name == 'edge_feats' and self.use_valid_edge:
            return self.get_ef_valid(indices)
        self_v = getattr(self,name,None)
        if (self_v is not None or self.use_disk):
            if (self.use_disk):
                res = loadBinDisk(getattr(self, f'{name}_path'), indices, use_slice)
            else:
                res = self_v[indices]
            return res
        else:
            raise RuntimeError(f'pre_fetch {name} error')
    

    def print_time(self):
        print_IO()

    def mem_select(self, indices, base_ind = None,  base_mems = None):
        names = ['memory', 'memory_ts', 'mailbox', 'mailbox_ts']
        if (self.memory_disk):
            # indices = torch.arange(indices[0], indices[0] + indices.shape[0], dtype=indices.dtype)
            pre_mem_s = time.time()
            mem_total = torch.zeros((indices.shape[0], self.mem_total_shape), dtype = torch.float32)

            # 找到indices在flag中为True的
            load_ind = torch.nonzero(self.mem_flag[indices]).reshape(-1)
            pre_mem_t = time.time() - pre_mem_s
            # print(f"mem 预处理 {time.time() - pre_mem_s:.6f}s")

            if (self.node_cache and load_ind.shape[0] > 0):
                load_ind_cache_flag = self.node_cache_flag[indices[load_ind]].long()
                load_ind_cached_ind = torch.nonzero(load_ind_cache_flag).reshape(-1).long()
                mem_total[load_ind[load_ind_cached_ind]] = self.cache_memory[load_ind_cache_flag[load_ind_cached_ind]]
                load_ind = load_ind[torch.nonzero(load_ind_cache_flag == 0).reshape(-1)]

            if (load_ind.shape[0] > 0):
                if (self.node_reorder):
                    print(f"读取node memory时重排序")
                    mem_ind = self.node_reorder_map[indices[load_ind].long()]
                else:
                    mem_ind = indices[load_ind]
                mem_load = loadBinDisk(self.mem_path, mem_ind)
                mem_total[load_ind] = mem_load


            left = 0
            mem_diff_s = time.time()
            memory = mem_total[:,self.mem_diff_shape[0]:self.mem_diff_shape[1]]
            memory_ts = mem_total[:,self.mem_diff_shape[1]:self.mem_diff_shape[2]].reshape(-1)
            mailbox = mem_total[:,self.mem_diff_shape[2]:self.mem_diff_shape[3]].reshape(-1, self.mail_size, self.mail_dim)
            mailbox_ts = mem_total[:,self.mem_diff_shape[3]:self.mem_diff_shape[4]].reshape(-1, self.mail_size)
            mem_diff_t = time.time() - mem_diff_s

            # print(f"mem 预处理{pre_mem_t:.6f}s 拆分{mem_diff_t:.6f}s")
            if (base_mems is None):
                res = [memory, memory_ts,mailbox,mailbox_ts]
            else:
                mem_set_s = time.time()
                res = base_mems
                cur_res = [memory, memory_ts,mailbox,mailbox_ts]
                for i, name in enumerate(names):
                    res[i][base_ind] = cur_res[i]
                mem_set_t = time.time() - mem_set_s
            # print(f"mem 预处理{pre_mem_t:.6f}s 拆分{mem_diff_t:.6f}s 增量赋值{mem_set_t:.6f}s")
            return res

        else:
            if (base_mems is None):
                res = []
                for name in names:
                    self_v = getattr(self,name,None)
                    cur_res = self_v[indices]
                    res.append(cur_res)
            else:
                res = base_mems
                for i, name in enumerate(names):
                    self_v = getattr(self,name,None)
                    cur_res = self_v[indices]
                    res[i][base_ind] = cur_res

        return res



    def mem_update(self, indices, mems):
        names = ['memory', 'memory_ts', 'mailbox', 'mailbox_ts']
        
        if (not self.memory_disk):
            for i, name in enumerate(names):
                self_v = getattr(self,name, None)
                self_v[indices] = mems[i]
        else:
            base_shape = mems[0].shape[0]
            for i in range(len(mems)):
                mems[i] = mems[i].reshape(base_shape, -1)
            total_mem = torch.cat(mems, dim = 1)

            self.mem_flag[indices] = True
            if (self.node_cache and not self.node_simple_cache):
                load_ind_cache_flag = self.node_cache_flag[indices].long()
                load_ind_cached_ind = torch.nonzero(load_ind_cache_flag).reshape(-1).long()
                self.cache_memory[load_ind_cache_flag[load_ind_cached_ind]] = total_mem[load_ind_cached_ind]

                no_cache_indices = torch.nonzero(load_ind_cache_flag == 0).reshape(-1)
                indices = indices[no_cache_indices]
                total_mem = total_mem[no_cache_indices]
            elif (self.node_simple_cache):
                # 使用simple的动态缓存
                cur_block = self.pre_fetch_block - 2
                cur_end_flush_idx = self.end_flush_idx[self.end_flush_ptr[cur_block]:self.end_flush_ptr[cur_block + 1]].long()
                self.node_cache_used[self.node_cache_flag[cur_end_flush_idx]] = False
                self.node_cache_flag[cur_end_flush_idx] = 0

                cur_start_cache_idx = self.start_cache_ind[self.start_cache_ptr[cur_block]: self.start_cache_ptr[cur_block + 1]]
                cache_unused_idx = torch.nonzero(~self.node_cache_used)[:cur_start_cache_idx.shape[0]].reshape(-1)
                cache_idx = indices[cur_start_cache_idx]
                self.node_cache_flag[cache_idx] = cache_unused_idx
                self.cache_memory[cache_unused_idx] = total_mem[cur_start_cache_idx]
                self.node_cache_used[cache_unused_idx] = True
                

                cur_start_flush_idx = self.start_flush_ind[self.start_flush_ptr[cur_block]: self.start_flush_ptr[cur_block + 1]]
                indices = indices[cur_start_flush_idx]
                total_mem = total_mem[cur_start_flush_idx]

                



            if (self.node_reorder):
                print(f"写入node memory时重排序")
                mem_ind = self.node_reorder_map[indices.long()]
            else:
                mem_ind = indices

            updateBinDisk(self.mem_path, total_mem, mem_ind)
            
        

    def update_index(self, name, indices, conf):
        # print(f"子程序 update {name}")
        
        self_v = getattr(self,name, None)
        if (self_v is not None):
            dim, shape = conf
            self_v[indices] = self.share_tmp[:shape].reshape(dim)
        else:
            if (self.memory_disk):
                print(f"{name} 在disk，训练时不做更新...")
                return
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

        self.mem_dim = memory_param['dim_out']
        self.mail_dim = 2 * memory_param['dim_out'] + dim_edge_feat
        self.mail_size = memory_param['mailbox_size']
        self.mem_dtype = torch.float32
        self.use_memory = True

        if (self.memory_disk):
            self.mem_path = f'/raid/guorui/DG/dataset/{self.dataset}/total_memory.bin'
            self.mem_flag = torch.zeros((num_nodes), dtype = torch.bool)
            self.mem_total_shape = (1 * self.mem_dim) + (1) + (1 * self.mail_size * (2 * self.mem_dim + dim_edge_feat)) + (1 * self.mail_size)
            self.mem_diff_shape = np.cumsum(np.array([0, (1 * self.mem_dim) , (1) , (1 * self.mail_size * (2 * self.mem_dim + dim_edge_feat)) , (1 * self.mail_size)]))

                    
            if (self.node_reorder):
                # if (self.node_simple_cache):
                self.node_reorder_map = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/node_simple_reorder_map.bin')
                # else:
                # self.node_reorder_map = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/node_reorder_map.bin')
            
            if (self.node_cache and not self.node_simple_cache):
                cache_nodes = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/node_cache_map.bin')
                self.cache_memory = torch.zeros((cache_nodes.shape[0] + 1, self.mem_total_shape), dtype = torch.float32)
                self.node_cache_flag = torch.zeros(num_nodes + 1, dtype = torch.int32)
                self.node_cache_flag[cache_nodes.long()] = torch.arange(1, cache_nodes.shape[0] + 1, dtype = torch.int32)
            if (self.node_simple_cache):

                self.node_cache_flag = torch.zeros(num_nodes + 1, dtype = torch.int64)

                self.start_cache_ind = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/start_cache_ind.bin', device='cpu').long()
                self.start_flush_ind = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/start_flush_ind.bin', device='cpu').long()
                self.end_flush_idx = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/end_flush_idx.bin', device='cpu').long()
                self.start_cache_ptr = loadBin( f'/raid/guorui/DG/dataset/{self.dataset}/start_cache_ptr.bin', device='cpu')
                self.start_flush_ptr = loadBin( f'/raid/guorui/DG/dataset/{self.dataset}/start_flush_ptr.bin', device='cpu')
                self.end_flush_ptr = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/end_flush_ptr.bin', device='cpu')

                self.simple_max_num = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/simple_max_num.bin')[0]
                self.cache_memory = torch.zeros((self.simple_max_num+ 1, self.mem_total_shape), dtype = torch.float32)
                
                self.node_cache_used = torch.zeros(self.simple_max_num + 2, dtype = torch.bool)
                self.node_cache_used[0] = 1

            if (not os.path.exists(self.mem_path)):
                print(f"首次做mem_disk训练，初始化mem_disk文件")
                memory = torch.randn((num_nodes, memory_param['dim_out']), dtype=torch.float32).reshape(num_nodes, -1)
                memory_ts = torch.randn(num_nodes, dtype=torch.float32).reshape(num_nodes, -1)
                mailbox = torch.randn((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32).reshape(num_nodes, -1)
                mailbox_ts = torch.randn((num_nodes, memory_param['mailbox_size']), dtype=torch.float32).reshape(num_nodes, -1)

                # total_shape = 0
                # total_shape += (num_nodes * self.mem_dim) + (num_nodes) + (num_nodes * self.mail_size * (2 * self.mem_dim + dim_edge_feat)) + (num_nodes * self.mail_size)
                # TODO 后续可以分块写入，但这里认为此处的预处理操作开销很小，不计入开销代价
                total_memory = torch.cat([memory, memory_ts, mailbox, mailbox_ts], dim = 1)
                del(memory, memory_ts, mailbox, mailbox_ts)
                saveBin(total_memory, self.mem_path)

        else:
            self.memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32)
            self.memory_ts = torch.zeros(num_nodes, dtype=torch.float32)
            self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32)
            self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32)
        
        over_memory = 1

    
    def reset_memory(self):
        reset_IO()

        if (self.node_cache):
            self.cache_memory.fill_(0)
            if (self.node_simple_cache):
                self.node_cache_flag.fill_(0)
                self.node_cache_used.fill_(0)
                self.node_cache_used[0] = 1
        if (self.memory_disk):
            self.mem_flag.fill_(0)
        else:
            self.memory.fill_(0)
            self.memory_ts.fill_(0)
            self.mailbox.fill_(0)
            self.mailbox_ts.fill_(0)
        
        asd = 1
        
    def set_mode(self, mode):
        self.mode = mode

    def init_feats(self, dataset):
        # node_feats, edge_feats = load_feat(dataset)
        # self.node_feats = node_feats
        self.dataset = dataset
        
        feat_conf = loadConf(f'/raid/guorui/DG/dataset/{self.dataset}/1')
        self.feat_conf = {}
        for key in feat_conf:
            value = feat_conf[key]
            if ('node_feat' in key):
                self.feat_conf['node_feats'] = value
            elif ('edge_feat' in key):
                self.feat_conf['edge_feats'] = value

        if (self.use_disk):
            
                
            self.edge_feats_path = f'/raid/guorui/DG/dataset/{self.dataset}/edge_features.bin'
            self.node_feats_path = f'/raid/guorui/DG/dataset/{self.dataset}/node_features.bin'
            if (os.path.exists(self.node_feats_path)):
                self.node_feats = torch.empty(1)
            if (os.path.exists(self.edge_feats_path)):
                self.edge_feats = torch.empty(1)
                self.has_ef = True
            else:
                self.edge_feats = torch.empty(0)
                self.has_ef = False

            if (self.has_ef):
                self.edge_reorder_map = loadBin(f'/raid/guorui/DG/dataset/{self.dataset}/edge_reorder_map.bin')
                self.edge_features_reorder_path = f'/raid/guorui/DG/dataset/{self.dataset}/edge_features_reorder.bin'

            return
            
        
        
        if (not self.use_valid_edge):
            node_feats, edge_feats = load_feat(dataset)
            self.node_feats = node_feats
            self.edge_feats = edge_feats
        else:
            node_feats, edge_feats = load_feat(dataset, load_edge=False)
            self.node_feats = node_feats
            self.edge_feats = torch.empty(0)

    def init_valid_edge(self, max_valid_num, edge_feat_len, conf):
        # 淘汰过期边策略 这里先对TALK数据集
        self.path, self.batch_size, self.fan_nums = conf

        self.valid_ef = torch.zeros((max_valid_num, edge_feat_len), dtype = torch.float32)

        self.update_valid_edge(0)

    def reset_valid_edge(self):
        self.valid_ef.fill_(0)
        self.update_valid_edge(0)

    def update_valid_edge(self, block_num, cur_ef = None):
        if (cur_ef is None):
            self.cur_ef_bound = loadBin(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_bound.pt')
            self.valid_ind = torch.arange(self.cur_ef_bound[0], self.cur_ef_bound[1], dtype = torch.int32)
            cur_ef = loadBinDisk(self.path + '/edge_features.bin', self.valid_ind)
            # cur_ef = loadBin(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre.pt')
        # cur_map = loadBin(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_map.pt')
        replace_idx = loadBin(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_replace.pt')
        if (block_num > 0):
            self.valid_map[replace_idx.long()] = self.valid_ind
        else:
            self.valid_map = loadBin(self.path + f'/part-{self.batch_size}/part{block_num}_edge_incre_map.pt')
        self.valid_ef[replace_idx.long()] = cur_ef

    def get_ef_valid(self, eids):

        eids = eids.cpu()
        map = self.valid_map
        sorted_map, indices = torch.sort(map)
        positions = torch.searchsorted(sorted_map, eids)

        original_positions = indices[positions]
        return self.valid_ef[original_positions.long()]


    def load_file(self, paths, tags, i):

        if (os.path.exists(paths[i].replace('.pt', '.bin'))):
            if ((not self.use_bucket and 'part' in tags[i])):
                self.async_load_dic[tags[i]] = None
            else:
                if ('valid_edge_feat' == tags[i]):
                    self.cur_ef_bound = loadBin(self.path + f'/part-{self.batch_size}/part{self.block_num}_edge_incre_bound.pt')
                    self.valid_ind = torch.arange(self.cur_ef_bound[0], self.cur_ef_bound[1], dtype = torch.int32)
                    self.async_load_dic[tags[i]] = loadBinDisk(self.path + '/edge_features.bin', self.valid_ind)
                else:
                    self.async_load_dic[tags[i]] = loadBin(paths[i])
        else:
            self.async_load_dic[tags[i]] = None


        #这个完成之后加载下一个
        if ((i + 1) < len(paths)):
            thread = threading.Thread(target=self.load_file, args=(paths, tags, i + 1))
            self.async_load_flag[tags[i + 1]] = thread
            thread.start()
    
    def async_load(self, paths, tags):
        # self.IO_conn.send(('init_shared_tensor', (None,)))
        # self.IO_conn.recv()
        thread = threading.Thread(target=self.load_file, args=(paths, tags, 0))
        self.async_load_flag[tags[0]] = thread
        thread.start()


    def pre_fetch(self, block_num, memory_info, neg_info, part_node_map, part_edge_map, conf):
        self.pre_fetch_block = block_num
        #1.out-of-core
        total_s = time.time()
        has_ef = (self.use_valid_edge) or self.edge_feats.shape[0] > 0
        has_nf = self.node_feats.shape[0] > 0
        t0 = time.time()
        neg_nodes, neg_eids = neg_info
        neg_nodes,neg_eids = neg_nodes.cpu(),neg_eids.cpu()
        path, batch_size, fan_nums = conf
        incre_bucket = self.use_bucket

        #incre_ef, part_ef, part_nf
        self.block_num = block_num
        if (self.use_bucket and incre_bucket):
            if (self.use_valid_edge):
                load_paths = [path + f'/part-{self.batch_size}/part{block_num}_edge_incre.pt', path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat_incre.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat_incre.pt']
                tags = ['valid_edge_feat', 'part_edge_feat', 'part_node_feat']
            else:
                load_paths = [path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat_incre.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat_incre.pt']
                tags = ['part_edge_feat', 'part_node_feat']
        else: # 否则就不使用bucket
            if (self.use_valid_edge):
                load_paths = [path + f'/part-{self.batch_size}/part{block_num}_edge_incre.pt', path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat.pt']
                tags = ['valid_edge_feat', 'part_edge_feat', 'part_node_feat']
            else:
                load_paths = [path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt',path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_feat.pt']
                tags = ['part_edge_feat', 'part_node_feat']
        pos_edge_map = loadBin(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_map.pt')
        pos_node_map = loadBin(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_map.pt')
        pos_edge_shape = pos_edge_map.shape[0]
        pos_node_shape = pos_node_map.shape[0]
        self.async_load(load_paths, tags)

        # neg_nodes, _ = torch.sort(neg_nodes)
        # neg_eids, _ = torch.sort(neg_eids)
        # allo_s = time.time()
        # test_f = torch.zeros(100000000 * 3, dtype = torch.float32).share_memory_()
        # print(f"创建{test_f.reshape(-1).shape[0] * 4 / 1024 ** 3:.3f}GB的共享向量用时{time.time() - allo_s:.5f}s")
        part_node_map = part_node_map.cpu()
        part_edge_map = part_edge_map.cpu()

        
        if (self.use_bucket and incre_bucket):
            node_incre_mask = loadBin(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_node_map_incre_mask.pt')
            edge_incre_mask = loadBin(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_map_incre_mask.pt')

        if (not self.use_bucket):
            start_t = time.time()

            pos_node_feats = self.self_select('node_feats', pos_node_map.long())
            if (not self.use_valid_edge):
                pos_edge_feats = self.self_select('edge_feats', pos_edge_map.long())
            # print(f"no bucket load time:{time.time() - start_t}")


        t1 = 0
        if (memory_info and memory_info[1] is not None):
            #当前正在异步运行块i，需要将块(i-1)的memory信息传入并做刷入，
            part_map, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = memory_info
            # part_map是上一个block的数据，这里memory刷入只需要刷入part_map中不在part_node_map的部分
            # part_map = torch.unique(part_map)
            part_map = part_map.cpu().long()
            part_memory = part_memory.cpu()
            part_memory_ts = part_memory_ts.cpu()
            part_mailbox = part_mailbox.cpu()
            part_mailbox_ts = part_mailbox_ts.cpu()
            # part_dis_ind = torch.isin(part_map, part_node_map, assume_unique=True, invert = True)
            
            # part_map = part_map[part_dis_ind].long()
            # part_memory = part_memory[part_dis_ind].cpu()
            # part_memory_ts = part_memory_ts[part_dis_ind].cpu()
            # part_mailbox = part_mailbox[part_dis_ind].cpu()
            # part_mailbox_ts = part_mailbox_ts[part_dis_ind].cpu()
            t1 = time.time() - t0
            t0 = time.time()

            self.mem_update(part_map, [part_memory, part_memory_ts, part_mailbox, part_mailbox_ts])

        
            del part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, neg_info
            # emptyCache()
        
        
        t1_1 = time.time() - t0
        t0 = time.time()

        # pos_edge_feats = loadBin(path + f'/part-{batch_size}-{fan_nums}/part{block_num}_edge_feat.pt')
        
        t2 = time.time() - t0
        t0 = time.time()

        
        # table1 = torch.zeros_like(neg_nodes) - 1
        # table2 = torch.zeros_like(pos_node_map) - 1
        # dgl.findSameNode(neg_nodes, pos_node_map, table1, table2)
        # neg_nodes, pos_node_map, table1, table2 = neg_nodes.cpu(), pos_node_map.cpu(), table1.cpu(), table2.cpu()
        # dis_ind = table1 == -1
        dis_ind = torch.isin(neg_nodes, pos_node_map, assume_unique=True,invert=True)
        dis_neg_nodes = neg_nodes[dis_ind]

        # time.sleep(0.1)
        # 此处为负节点邻域涉及的节点，新加一个判断：这个节点集合是否在上一个block训练的所有节点集合中，若在则后续在下一个block接收时再填充他们的特征。
        # 验证：当使用负节点采样重用时，这个操作是否能减少大部分的IO开销？
        node_dd_ind = torch.isin(dis_neg_nodes, part_node_map, assume_unique=True,invert=True)
        if (not self.data_incre):
            # 实验：若不使用增量加载，此处作完整加载
            node_dd_ind.fill_(True)
        dd_neg_nodes = dis_neg_nodes[node_dd_ind].long()
        # print(f"preFetch的block中含有需要做IO加载的负节点邻域个数: {dis_neg_nodes.shape[0]} 若除去上一个block中出现的节点，那么剩余{dd_neg_nodes.shape[0]}")
        t3 = time.time() - t0
        t0 = time.time()
        if (has_nf):
            node_conf = self.feat_conf['node_feats']
            neg_node_feats = torch.zeros((dis_neg_nodes.shape[0], node_conf['shape'][1]), dtype = torch.float32)
            if dd_neg_nodes.shape[0] > 0:
                neg_node_feats[node_dd_ind] = self.self_select('node_feats', dd_neg_nodes.to(torch.int64))
                # print(neg_node_feats[node_dd_ind])
            
            # neg_node_feats = self.self_select('node_feats', dis_neg_nodes.to(torch.int64))

        if (self.use_bucket and incre_bucket):
            node_d_map = torch.cat((node_incre_mask, node_dd_ind))
        else:
            node_d_map = torch.cat((torch.ones(pos_node_map.shape[0], dtype = torch.bool), node_dd_ind))

        pos_node_map = torch.cat((pos_node_map, dis_neg_nodes))
        pos_node_map,node_indices = torch.sort(pos_node_map)
        
        node_d_map = torch.nonzero(~node_d_map[node_indices]).reshape(-1)


        t4 = time.time() - t0
        t0 = time.time()
        dis_ind = torch.isin(neg_eids, pos_edge_map, assume_unique=True,invert=True)
        dis_neg_eids = neg_eids[dis_ind]

        edge_dd_ind = torch.isin(dis_neg_eids, part_edge_map, assume_unique=True,invert=True)
        if (not self.data_incre):
            # 实验：若不使用增量加载，此处作完整加载
            edge_dd_ind.fill_(True)
        dd_neg_eids = dis_neg_eids[edge_dd_ind]
        # print(f"preFetch的block中含有需要做IO加载的负边个数: {dis_neg_eids.shape[0]} 若除去上一个block中出现的边特征，那么剩余{dd_neg_eids.shape[0]}")


        #测试重排序带来的效益

        # print(f"neg_nodes: {neg_nodes.shape[0]}, neg_eids: {neg_eids.shape[0]}, dis_neg_nodes: {dis_neg_nodes.shape[0]},dis_neg_eids: {dis_neg_eids.shape[0]}")
        


        if (has_ef and not self.use_valid_edge):
            
            edge_conf = self.feat_conf['edge_feats']
            dis_neg_eids_feat = torch.zeros((dis_neg_eids.shape[0], edge_conf['shape'][1]), dtype = getattr(torch,edge_conf['dtype']))
            if (dd_neg_eids.shape[0] > 0):
                if (not self.use_disk):
                    dis_neg_eids_feat[edge_dd_ind] = self.self_select('edge_feats',dd_neg_eids.to(torch.int64))
                else:
                    reorder_ind = self.edge_reorder_map[dd_neg_eids.long()]
                    reorder_ind,indices = torch.sort(reorder_ind)
                    dis_neg_eids_feat[torch.nonzero(edge_dd_ind).reshape(-1)[indices]]= self.self_select('edge_features_reorder',reorder_ind, use_slice=False)
                

            
            # dis_neg_eids_feat = self.self_select('edge_feats', dis_neg_eids.to(torch.int64))


        t5 = time.time() - t0
        t0 = time.time()
        if (self.use_bucket and incre_bucket):
            edge_d_map = torch.cat((edge_incre_mask, edge_dd_ind))
        else:
            edge_d_map = torch.cat((torch.ones(pos_edge_map.shape[0], dtype = torch.bool), edge_dd_ind))

        if (self.use_valid_edge and not self.use_bucket):
            pos_edge_map_clone = pos_edge_map.clone()
        pos_edge_map = torch.cat((pos_edge_map, dis_neg_eids))
        pos_edge_map,edge_indices = torch.sort(pos_edge_map)

        edge_d_map = torch.nonzero(~edge_d_map[edge_indices]).reshape(-1)

        t6 = time.time() - t0
        t0 = time.time()

        #此时获得了下一个块的所有nodes eids node_feats edge_feats
        #分别为 pos_node_map pos_edge_map node_feats edge_feats
        #还需要获得下一个块的所有node_memory信息
        
        
        nodes = pos_node_map.long()
        if (self.use_memory):
            pre_same_nodes = torch.isin(part_node_map.cpu(), pos_node_map, assume_unique = True) #mask形式的
            cur_same_nodes = torch.isin(pos_node_map, part_node_map.cpu(), assume_unique = True)
            # print(f"寻找pre和cur: {time.time() - t0:.6f}s")
            if (not self.data_incre):
                # 不做增量加载
                part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = self.mem_select(nodes)
                # part_memory = self.mem_select('memory',nodes)
                # part_memory_ts = self.mem_select('memory_ts',nodes)
                # part_mailbox = self.mem_select('mailbox',nodes)
                # part_mailbox_ts = self.mem_select('mailbox_ts',nodes)
            else:
                part_memory = torch.empty((nodes.shape[0], self.mem_dim), dtype = self.mem_dtype)
                part_memory_ts = torch.empty((nodes.shape[0]), dtype = self.mem_dtype)
                part_mailbox = torch.empty((nodes.shape[0], self.mail_size, self.mail_dim), dtype = self.mem_dtype)
                part_mailbox_ts = torch.empty((nodes.shape[0], self.mail_size), dtype = self.mem_dtype)
                # print(f"开辟memory: {time.time() - t0:.6f}s")
                dis_nodes = ~cur_same_nodes #表示下一个bucket中有但是当前bucket没有的，需要去做额外的读取操作
                dis_nodes_ind = pos_node_map[dis_nodes].long()
                
                # print(f"反向和ind: {time.time() - t0:.6f}s")
                part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = self.mem_select(dis_nodes_ind, base_ind = dis_nodes, base_mems=[part_memory, part_memory_ts,part_mailbox, part_mailbox_ts])

                # part_memory[dis_nodes] = self.mem_select('memory',dis_nodes_ind)
                # part_memory_ts[dis_nodes] = self.mem_select('memory_ts',dis_nodes_ind)
                # part_mailbox[dis_nodes] = self.mem_select('mailbox',dis_nodes_ind)
                # part_mailbox_ts[dis_nodes] = self.mem_select('mailbox_ts',dis_nodes_ind)

                asdasd = 1
            #此处要返回的memory信息不包括当前执行块的，因此需要在后面处理加上当前处理块后的最新memory信息
            #即当前异步处理的块的memory结果会实时更新到self.memory中，这里返回的下一个块需要用到的memory需要在下一个块开始之前和self.memory结合
            #因此这里预先判断：当前异步处理的块中出现的节点哪些在下一个块也出现了，即self.part_node_map(当前块)和pos_node_map(下一个块)的关系
            
            
        else:
            part_memory = torch.empty(0)
            part_memory_ts = torch.empty(0)
            part_mailbox = torch.empty(0)
            part_mailbox_ts = torch.empty(0)
            pre_same_nodes = torch.empty(0)
            cur_same_nodes = torch.empty(0)
        t7 = time.time() - t0
        t0 = time.time()



        # print(f"pre fetch over...")
        if (pos_node_map.shape[0] > self.part_node_map.shape[0]):
            self.prefetch_conn.send(f'node extension, {pos_node_map.shape[0]} -> {self.part_node_map.shape[0]}')
            self.prefetch_conn.recv()
        if (pos_edge_map.shape[0] > self.part_edge_map.shape[0]):
            self.prefetch_conn.send(f'edge extension, {pos_edge_map.shape[0]} -> {self.part_edge_map.shape[0]}')
            self.prefetch_conn.recv()
        t8 = time.time() - t0
        t0 = time.time()

        reorder_time = 0
        asy_time = 0
        node_feats, edge_feats = torch.empty(0, dtype = torch.float32), torch.empty(0, dtype = torch.float32)
        for tag in tags:
            asy_time_s = time.time()
            self.async_load_flag[tag].join()
            asy_time += time.time() - asy_time_s

            data = self.async_load_dic[tag]
            if (self.use_valid_edge and tag == 'valid_edge_feat' and has_ef):
                # print(tag)
                self.update_valid_edge(block_num, data)

                dis_neg_eids_feat = self.get_ef_valid(dis_neg_eids)
            elif (tag == 'part_node_feat' and has_nf):
                # print(tag)
                if (self.use_bucket):
                    if (incre_bucket):
                        pos_node_feats = torch.zeros((pos_node_shape, data.shape[1]), dtype = data.dtype)
                        pos_node_feats[node_incre_mask] = data
                    else:
                        pos_node_feats = data
                node_feats = torch.cat((pos_node_feats, neg_node_feats))
                reorder_s = time.time()
                node_feats = node_feats[node_indices]
                reorder_time += time.time() - reorder_s
            elif (tag == 'part_edge_feat' and has_ef):
                # print(tag)
                if (self.use_bucket):
                    if (incre_bucket):
                        pos_edge_feats = torch.zeros((pos_edge_shape, data.shape[1]), dtype = data.dtype)
                        pos_edge_feats[edge_incre_mask] = data
                    else:
                        pos_edge_feats = data
                if (not self.use_bucket and self.use_valid_edge):
                    pos_edge_feats = self.get_ef_valid(pos_edge_map_clone.long())
                edge_feats = torch.cat((pos_edge_feats, dis_neg_eids_feat))
                reorder_s = time.time()
                edge_feats = edge_feats[edge_indices]
                reorder_time += time.time() - reorder_s

            # print(f"cur tag: {tag} use time:{time.time() - asy_time_s:.4f}s")
        t9 = time.time() - t0
        t0 = time.time()

        # time.sleep(0.05)

        self.prefetch_after([pos_node_map, node_feats, pos_edge_map, edge_feats, part_memory,\
                              part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes, node_d_map, edge_d_map])
        # self.preFetchDataCache.put({'node_info': [pos_node_map, node_feats], 'edge_info': [pos_edge_map, edge_feats],\
        #                              'memory_info': [part_memory, part_memory_ts, part_mailbox, part_mailbox_ts],\
        #                                 'memory_update_info': [pre_same_nodes, cur_same_nodes]})
        t10 = time.time() - t0
        t0 = time.time()

        tt = time.time() - total_s
        #nodes做sort + unique找出最终的indices

        # print(f"tt:{tt:.2f}s t1:{t1:.2f}s t1_1:{t1_1:.2f}s t2:{t2:.2f}s t3:{t3:.2f}s t4:{t4:.2f}s", end=" ")
        # print(f"t5:{t5:.2f}s t6:{t6:.2f}s t7:{t7:.2f}s t8:{t8:.2f}s t9:{t9:.2f}s rt: {reorder_time:.2f}s ast: {asy_time:.2f}s t10:{t10:.2f}s")
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


