#历史嵌入重用缓存 Orca
import dgl
import torch
import numpy as np
import time
from sampler.sampler_gpu import *

class Embedding_buffer:
    def __init__(self, g, df, train_param, train_edge_end, presample_batch, batch_threshold, fan_nums, gnn_param, neg_sampler):
        self.g = g
        self.df = df
        batch_size = train_param['batch_size'] * presample_batch
        train_batch_size = train_param['batch_size']
        self.dim_out = gnn_param['dim_out']
        self.batch_size = batch_size #预采样做分析的batch_size
        self.train_batch_size = train_batch_size
        self.train_edge_end = train_edge_end

        assert batch_size % train_batch_size == 0, "预采样batch size必须是训练的batch_size的整数倍!"
        self.batch_num = batch_size / train_batch_size # batch_num是预采样的batch个数
        assert self.batch_num > batch_threshold, "预采样的batch个数需要大于限定batch上界"

        self.batch_num = batch_size // train_batch_size
        #例如，batch_size分别是1000, 30000 即每次预采样采样到30个batch， 上界为5，那么每运行25个batch就需要做一次预采样
        self.per_sample_batch = self.batch_num - batch_threshold
        self.cur_batch = 0 #当当前执行的batch为0,25,50时做一次预采样获得下面25次batch的plan

        self.batch_threshold = batch_threshold #在threshold作为个数，在个数以内出现的可重用嵌入
        self.fan_num = fan_nums[0]

        self.cache_plan = []
        self.cur_sample_num = 0

        #buffer做动态扩容?
        self.buffer = torch.zeros((train_batch_size * self.batch_threshold * 10, self.dim_out), dtype = torch.float32, device = 'cuda:0')
        #map[index] 表示buffer[index]存储的是哪个节点的历史嵌入(map[index] = node index i)
        self.map = torch.zeros(self.buffer.shape[0], dtype = torch.int32, device = 'cuda:0') - 1
        self.flag = torch.zeros(self.buffer.shape[0], dtype = torch.int32, device = 'cuda:0') - 1

        self.sampler = Sampler_GPU(g, [fan_nums[0]], 1, self)
        self.neg_sampler = neg_sampler
        self.nodes = None

        self.cur_mode = 'train'
        self.use_buffer = True


        self.time_inp = 0
        self.time_upd = 0
        self.hit_counts = []

    def reset_time(self):
        print(f"buffer input emb用时{self.time_inp:.5f}s, update emb用时{self.time_upd:.5f}s, buffer平均命中率:{torch.mean(torch.tensor(self.hit_counts).to(torch.float32)):.2f}%")
        self.time_inp = 0
        self.time_upd = 0

    #输入一批上一层采样出的src节点(会作为下一层的dst)，这里需要找出这一批节点中哪些节点的历史嵌入在缓存中存储
    #输出这批节点中有缓存的节点的index，并保存为cur_batch_his_indices
    def gen_his_indices(self, nodes):

        self.nodes = nodes
        map = self.map
        res = torch.zeros_like(nodes)
        #TODO 这里需要屏蔽源节点，不能重用源节点的
        root_len = self.train_batch_size * 3
        root_nodes = nodes[:root_len]

        node_sort, node_sort_indices = torch.sort(nodes)
        map_sort, map_sort_indices = torch.sort(map)


        #node_uni和map两个现在都是sorted uniqued
        table1 = torch.zeros_like(node_sort) - 1
        table2 = torch.zeros_like(map_sort) - 1
        table1, table2 = dgl.findSameIndex(node_sort, map_sort, table1, table2)

        map_sort_indices = torch.cat((map_sort_indices, torch.tensor([-1], dtype = torch.int32, device = 'cuda:0')))
        ind = map_sort_indices[table1.to(torch.int64)].to(torch.int32)

        res[node_sort_indices.to(torch.int64)] = ind
        #此处res长度与nodes一致，当nodes[i]有历史嵌入时，res[i]的值表示他的历史嵌入在map中的index
        #这个index实际上就是buffer的index，因为buffer[i]存储的就是map[i]的数据
        #例如，nodes[0]是节点8，节点8在map中存在，那么说明节点8有历史嵌入，res[0]表示map的索引，map[res[0]] = 8

        nodes = nodes[root_len:]
        node_sort, node_sort_indices = torch.sort(nodes)
        root_nodes_sort,_ = torch.sort(root_nodes)
        table1 = torch.zeros_like(nodes)
        table2 = torch.zeros_like(root_nodes)
        dgl.findSameIndex(node_sort, root_nodes_sort, table1, table2)
        root_nodes_mask = node_sort_indices[table1.to(torch.bool)]
        res[root_nodes_mask] = -1 #TODO 原先的

        # root_nodes = nodes[:root_len]
        # no_root = nodes[root_len:]

        # no_root_sort, no_root_sort_indices = torch.sort(no_root)
        # root_nodes_sort, root_nodes_sort_indices = torch.sort(root_nodes)
        # table1 = torch.zeros_like(no_root_sort) - 1
        # table2 = torch.zeros_like(root_nodes_sort) - 1
        # dgl.findSameIndex(no_root_sort, root_nodes_sort, table1, table2)

        # no_root_sort_indices = torch.cat((no_root_sort_indices, torch.zeros(1, dtype = torch.int32, device = 'cuda:0') - 1))
        # no_root_sort_indices = no_root_sort_indices[table2.to(torch.int64)]
        # no_root_ind = no_root_sort_indices[no_root_sort_indices > -1] + root_len

        # res[:self.train_batch_size * 3] = -1
        # res[no_root_ind] = -1
        

        res = res.to(torch.int64)
        
        hit_count = torch.nonzero(res > -1).shape[0]
        self.hit_counts.append(hit_count / nodes.shape[0] * 100)
        # print(f"节点总数{nodes.shape[0]},命中节点数{hit_count} 命中率{hit_count / nodes.shape[0] * 100:.4f}%")

        self.cur_buf_indices = res
        return res

    @torch.no_grad()
    def input_his_emb(self, nodes_emb):
        # torch.cuda.synchronize()
        start = time.time()
        #此处nodes_emb的节点顺序和gen_his_indices中的nodes顺序完全一致
        indices = self.cur_buf_indices
        mask = indices > -1
        nodes_emb[mask] = self.buffer[indices[mask]]
        # torch.cuda.synchronize()
        self.time_inp += time.time() - start
        return nodes_emb

    @torch.no_grad()
    def update_his_emb(self, nodes_emb):
        #在节点嵌入计算完成后，根据cache_plan更新buffer
        #同时淘汰那些距离上界已经过了的节点嵌入(例如，上界是5，当嵌入已经存在5轮且没被更新时，他不再被重用)
        #更新buffer时，将flag设为上界，每次更新时减1，当减为0时这个嵌入已经过了上界轮，将其淘汰。
        # torch.cuda.synchronize()
        start = time.time()
        nodes = self.nodes

        self.flag -= 1
        dis_mask = self.flag < 0
        self.map[dis_mask] = -1 #不再缓存过期节点
        self.flag[dis_mask] = -1

        cur_buf_indices = self.cur_buf_indices #和nodes一样长度，若值为-1表示重计算，值大于-1表示重用了indices[i]的历史嵌入
        plan = self.cache_plan[self.cur_batch % (self.per_sample_batch)].to(torch.int32)

        #找出plan中被重计算的
        #plan是unique的
        recompu_nodes = nodes[cur_buf_indices == -1]
        recompu_nodes_sort, _ = torch.sort(recompu_nodes)
        plan_sort, plan_sort_indices = torch.sort(plan)
        table1 = torch.zeros_like(recompu_nodes_sort)
        table2 = torch.zeros_like(plan_sort)
        dgl.findSameNode(recompu_nodes_sort, plan_sort, table1, table2)

        table2 = plan_sort_indices[table2.to(torch.bool)]
        plan = plan[table2] #此时plan中的节点都是当前batch被重计算的节点。。。


        #判断map当中是否有plan中的值，若有将其舍弃置为-1
        map = self.map
        plan_sort, plan_sort_indices = torch.sort(plan)
        map_sort, map_sort_indices = torch.sort(map)
        table1 = torch.zeros_like(map_sort)
        table2 = torch.zeros_like(plan_sort)
        dgl.findSameNode(map_sort, plan_sort, table1, table2)
        map_ind = map_sort_indices[table1.to(torch.bool)]
        self.map[map_ind] = -1
        self.flag[map_ind] = -1


        #最后，将plan中的所有节点放到map中为-1的地方
        num = plan.shape[0]
        map_avail = torch.nonzero(map == -1).reshape(-1)
        map_ind = map_avail[:num]
        self.map[map_ind] = plan
        self.flag[map_ind] = self.batch_threshold
        # self.buffer[map_ind] = 

        #最后获取plan中节点在nodes中的位置
        # plan_sort, plan_sort_indices = torch.sort(plan)

        nodes_sort, nodes_sort_indices = torch.sort(nodes)
        table1 = torch.zeros_like(plan_sort) - 1
        table2 = torch.zeros_like(nodes_sort) - 1
        dgl.findSameIndex(plan_sort, nodes_sort, table1, table2)

        nodes_ind = nodes_sort_indices[table1.to(torch.int64)]
        # torch.cuda.synchronize()
        self.time_upd += time.time() - start
        self.buffer[map_ind] = nodes_emb[nodes_ind]


    def gen_batch_table(self, nodes_l):
        batch_size = self.batch_size
        real_batch_size = self.train_batch_size
        fan_num = self.fan_num + 1 #补上root_nodes
        batch_num = batch_size / real_batch_size

        batch_num = (nodes_l // 3) // 600

        basic = torch.arange(batch_num, dtype = torch.int32, device = 'cuda:0')
        batch_table = torch.tile(basic, (real_batch_size * fan_num,1)).T.reshape(-1, fan_num)
        part_batch_size = (nodes_l // 3) % 600
        batch_table = torch.cat((batch_table, torch.zeros((part_batch_size, fan_num) , dtype=torch.int32, device = 'cuda:0') + batch_num))

        batch_table = torch.tile(batch_table, (3,1))

        return batch_table


    def gen_batch_flag(self, seed_num):
        fan_num = self.fan_num + 1

        table = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(fan_num, seed_num)
        table[0][:] = 1
        return table.T


    def pre_sample(self):
        #需要根据self.cur_batch判断从何处开始计算
        start = self.cur_batch * self.train_batch_size
        end = min(self.train_edge_end, (self.cur_batch + self.batch_num) * self.train_batch_size)

        df = self.df

        rows = df[start:end]
        root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, self.neg_sampler.sample(len(rows))]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()


        ret_list = self.sampler.sample_layer(root_nodes, root_ts)
        src,dst,outts,outeid,root_nodes,root_ts = ret_list[0]

        nodes = torch.cat((root_nodes.reshape(1,-1), src.T), dim = 0).T
        del src, dst, outts, outeid, root_nodes, root_ts


        
        return nodes

    def run_batch(self, batch_num):
        #主程序正准备执行batch i的时候，判断是否需要预采样
        self.cur_batch = batch_num
        if (batch_num % self.per_sample_batch == 0):
            # print(f"cur batch: {batch_num}, start pre sample and analyze...")
        
            self.analyze()


    #预采样分析
    def analyze(self):
        nodes = self.pre_sample()
        mask = nodes > -1


        batch_table = self.gen_batch_table(nodes.shape[0])
        flag = self.gen_batch_flag(nodes.shape[0])
        total_node = nodes[mask]
        total_flag = flag[mask]
        total_batch = batch_table[mask]


        total_node, indices = torch.sort(total_node, stable = True)#TODO 此处需要一个稳定的cuda排序，因为需要保证以节点排序后，节点内的batch出现也是按顺序的
        total_node = total_node
        total_flag = total_flag[indices]
        total_batch = total_batch[indices]

        edges = torch.stack((total_node, total_batch), dim=1)
        unique_edges, indices = torch.unique(edges, dim=0, return_inverse=True)



        output_flag = torch.zeros(unique_edges.shape[0], dtype = torch.int32, device = 'cuda:0')


        # 使用 scatter_add_ 函数将 tensor 的值累加到 output 中
        output_flag.scatter_add_(0, indices, total_flag)
        total_flag = output_flag.to(torch.bool)
        total_node = unique_edges.reshape(-1)[::2]
        total_batch = unique_edges.reshape(-1)[1::2]

        total_flag[:] = False

        table = torch.zeros(torch.max(total_node) + 1, device = 'cuda:0', dtype = torch.int32)
        bin_count = dgl.bincount(total_node, table)
        table = table[table > 0]
        table = torch.cumsum(table, dim = 0) - 1

        #flag作用：1.忽略节点分界处，即节点最后一个出现的batch不用重用其嵌入。2.节点下一次出现时为源节点时不需要重用其嵌入。
        #将源节点标记前移一位，flag为1的表示下一个batch出现时为源节点，此时不重用该节点嵌入
        flag = total_flag[1:]
        flag[table[:-1]] = True #节点的最后一个出现的batch不需要重用嵌入

        distance = torch.diff(total_batch)
        distance[flag] = 0
        aver_dis = torch.mean(distance.to(torch.float32))
        res_indices = torch.zeros(total_node.shape[0] - 1, dtype = torch.bool, device = 'cuda:0')
        res_indices[torch.bitwise_and(distance > 0,distance < self.batch_threshold)] = True

        res_node = total_node[:-1][res_indices]
        res_batch = total_batch[:-1][res_indices]


        #这个sort就不需要考虑稳定性了，只需要知道batch中需要重用哪些节点即可
        res_batch, indices = torch.sort(res_batch)
        res_node = res_node[indices]
        table = torch.zeros(torch.max(res_batch) + 1, device = 'cuda:0', dtype = torch.int32)
        bin_count = dgl.bincount(res_batch, table)
        res_plan = torch.split(res_node.to(torch.int64), table.tolist() + [0]) #TODO 这里看看能不能用cuda优化
        res_plan = list(res_plan)

        self.cache_plan = res_plan
        count = 0
        for plan in res_plan:
            count += plan.shape[0]
        
        # print(f"plan中节点总个数{count}, 平均每个plan{count / 100}个节点")
        return res_plan


