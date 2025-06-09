
import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import emptyCache


d = 'GDELT'
batch_size = 60000
real_batch_size = 600
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10]
layers = len(fan_nums)
sampler = Sampler_GPU(g, fan_nums, layers)

from emb_buffer import *

#测试全批采样
# total_batch_node = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
# total_batch_ts = torch.empty(0, dtype = torch.float32, device = 'cuda:0')

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
neg_link_sampler = NegLinkSampler(num_nodes)

buffer = Embedding_buffer(g, df, {'batch_size': 600}, train_edge_end, 100, 10, [10,10], {'dim_out': 100}, neg_link_sampler)
# buffer = Embedding_buffer(g, df, 10000, 100, train_edge_end, 10, fan_nums, neg_link_sampler)
for i in range(100000):
    cur = i * 90
    buffer.run_batch(cur)
    count = 0
    for plan in buffer.cache_plan:
        count += plan.shape[0]
    count /= len(buffer.cache_plan)
    print(f"{cur},平均长度 {count}")





for _, rows in df[:train_edge_end].groupby(group_indexes):  
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)).cuda()
    # mask_nodes = torch.from_numpy(np.concatenate([rows.dst.values, rows.src.values]).astype(np.int32)).cuda()
    # mask_nodes = torch.cat((mask_nodes, (torch.zeros(rows.src.values.shape[0], dtype = torch.int32, device = 'cuda:0') - 1)))
    root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()

    # root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, np.ones(rows.src.values.shape[0])]).astype(np.int32)).cuda()
    # mask_nodes = torch.from_numpy(np.concatenate([rows.dst.values, rows.src.values]).astype(np.int32)).cuda()
    # root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()
    # mask_nodes = torch.cat((mask_nodes, (torch.zeros(rows.src.values.shape[0], dtype = torch.int32, device = 'cuda:0') - 1)))


    # total_batch_node = torch.cat((total_batch_node, root_nodes))
    # total_batch_ts = torch.cat((total_batch_ts, root_ts))
    # root_nodes = root_nodes[:2]
    # print(f"root_nodes: {root_nodes}")

    start = time.time()
    ret_list = sampler.sample_layer(root_nodes, root_ts)
    # emptyCache()
    sampleTime = time.time() - start
    # mfgs = sampler.gen_mfgs(ret_list)
    
    print(f"{root_nodes.shape}单层单block采样 + 转换block batch: {_} batchsize: {batch_size} 纯采样用时{sampleTime:.7f}s 用时:{time.time() - start:.7f}s")
    emptyCache()
    break



src,dst,outts,outeid,root_nodes,root_ts = ret_list[0]

#吧root_nodes拼在src前面，就成为了所有batch的出现模式
nodes = torch.cat((root_nodes.reshape(1,-1), src.T), dim = 0).T
mask = nodes > -1

def gen_batch_table(batch_size, real_batch_size, fan_num):
    start = time.time()

    batch_num = batch_size / real_batch_size
    if (batch_size % real_batch_size != 0):
        print(f"应当取整数倍batch!")
    print(f"训练时batch size: {real_batch_size}, 预采样的size: {batch_size}, 相当于预采样了{batch_num}个训练batch")

    basic = torch.arange(batch_num, dtype = torch.int32, device = 'cuda:0')
    batch_table = torch.tile(basic, (real_batch_size * fan_num,1)).T.reshape(-1, fan_num)

    batch_table = torch.tile(batch_table, (3,1))

    return batch_table

def gen_batch_flag(seed_num, fan_num):
    table = torch.zeros(seed_num * fan_num, dtype = torch.int32, device = 'cuda:0').reshape(fan_num, seed_num)
    table[0][:] = 1
    return table.T



batch_table = gen_batch_table(batch_size, real_batch_size, 11)
flag = gen_batch_flag(root_nodes.shape[0], 11)
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
res_indices[torch.bitwise_and(distance > 0,distance < 5)] = True

res_node = total_node[:-1][res_indices]
res_batch = total_batch[:-1][res_indices]


#这个sort就不需要考虑稳定性了，只需要知道batch中需要重用哪些节点即可
res_batch, indices = torch.sort(res_batch)
res_node = res_node[indices]
table = torch.zeros(torch.max(res_batch) + 1, device = 'cuda:0', dtype = torch.int32)
bin_count = dgl.bincount(res_batch, table)
res_plan = torch.split(res_node.to(torch.int64), table.tolist() + [0]) #TODO 这里看看能不能用cuda优化
res_plan = list(res_plan)

res_plan_1 = res_plan


# from emb_buffer import *
# buf = Embedding_buffer(g, df, batch_size, real_batch_size, 10, [10], (train_edge_end, val_edge_end, group_indexes), neg_link_sampler)
# res_plan_2 = buf.analyze_total()


#输入一批上一层采样出的src节点(会作为下一层的dst)，这里需要找出这一批节点中哪些节点的历史嵌入在缓存中存储
#输出这批节点中有缓存的节点的index，并保存为cur_batch_his_indices
#def gen_his_indices(self, nodes):

import dgl
import torch
#需要保证map内除了-1值外，是唯一的
map = torch.tensor([2,5,3,1,4,-1,-1,8,-1,9,7,6,10], dtype = torch.int32, device = 'cuda:0')
nodes = torch.tensor([8,9,8,0,8,10,5,5,4,11,7,3,3,3,3,3], dtype = torch.int32, device = 'cuda:0')
root_nodes = nodes[:3]

res = torch.zeros_like(nodes)

node_sort, node_sort_indices = torch.sort(nodes)
map_sort, map_sort_indices = torch.sort(map)


#node_uni和map两个现在都是sorted uniqued
table1 = torch.zeros_like(node_sort) - 1
table2 = torch.zeros_like(map_sort) - 1
table1, table2 = dgl.findSameIndex(node_sort, map_sort, table1, table2)

map_sort_indices = torch.cat((map_sort_indices, torch.tensor([-1], dtype = torch.int32, device = 'cuda:0')))
ind = map_sort_indices[table1.to(torch.int64)].to(torch.int32)

res[node_sort_indices.to(torch.int64)] = ind

root_nodes_sort,_ = torch.sort(root_nodes)
table1 = torch.zeros_like(nodes)
table2 = torch.zeros_like(root_nodes)
dgl.findSameNode(node_sort, root_nodes_sort, table1, table2)
root_nodes_mask = node_sort_indices[table1.to(torch.bool)]
res[root_nodes_mask] = -1

mask = res > -1

plan = torch.tensor([8,6,6,7,9,5,7,0,6,11], device = 'cuda:0', dtype = torch.int32)
cur_buf_indices = res
recompu_nodes = nodes[cur_buf_indices == -1]
recompu_nodes_sort, _ = torch.sort(recompu_nodes)
plan_sort, plan_sort_indices = torch.sort(plan)
table1 = torch.zeros_like(recompu_nodes_sort)
table2 = torch.zeros_like(plan_sort)
dgl.findSameNode(recompu_nodes_sort, plan_sort, table1, table2)

table2 = plan_sort_indices[table2.to(torch.bool)]
plan = plan[table2] #此时plan中的节点都是当前batch被重计算的节点。。。


#判断map当中是否有plan中的值，若有将其舍弃置为-1
plan_sort, plan_sort_indices = torch.sort(plan)
map_sort, map_sort_indices = torch.sort(map)
table1 = torch.zeros_like(map_sort)
table2 = torch.zeros_like(plan_sort)
dgl.findSameNode(map_sort, plan_sort, table1, table2)
map_ind = map_sort_indices[table1.to(torch.bool)]
map[map_ind] = -1
# flag[map_ind] = -1


#最后，将plan中的所有节点放到map中为-1的地方
num = plan.shape[0]
map_avail = torch.nonzero(map == -1).reshape(-1)
map_ind = map_avail[:num]
map[map_ind] = plan
# self.flag[map_ind] = 5

#最后获取plan在nodes中对应的位置
plan_sort, plan_sort_indices = torch.sort(plan)
nodes_sort, nodes_sort_indices = torch.sort(nodes)
table1 = torch.zeros_like(plan_sort) - 1
table2 = torch.zeros_like(nodes_sort) - 1
dgl.findSameIndex(plan_sort, nodes_sort, table1, table2)

nodes[nodes_sort_indices[table1.to(torch.int64)]]