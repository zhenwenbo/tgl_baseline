



import torch

batch_size = self.batch_size
real_batch_size = self.train_batch_size
fan_num = self.fan_num + 1 #补上root_nodes
batch_num = batch_size / real_batch_size

basic = torch.arange(batch_num, dtype = torch.int32, device = 'cuda:0')
batch_table = torch.tile(basic, (real_batch_size * fan_num,1)).T.reshape(-1, fan_num)

batch_table = torch.tile(batch_table, (3,1))




#测试sort的性能
import torch
import time
import dgl

time_t = 0
count = 1000
for i in range(count):
    test_tensor = torch.randperm(1000000, dtype = torch.int32, device = 'cuda:0')
    tensor1 = test_tensor[:10000]
    tensor = test_tensor[:10000]
    tensor2 = test_tensor[10000:20000]
    table1 = torch.zeros_like(tensor1) - 1
    table2 = torch.zeros_like(tensor2) - 1
    table = torch.zeros(torch.max(tensor1) + 1, device = 'cuda:0', dtype = torch.int32)


    start = time.time()
    dgl.findSameIndex(tensor1, tensor2, table1, table2)
    # dgl.bincount(tensor1, table)
    # tensor_sort, indices = torch.sort(tensor)
    time_t += time.time() - start
    # print(f"用时{time.time() - start:.10f}s")
print(f"{count}次，用时{time_t}, 平均{time_t / count}")

from numba import jit
import numpy as np

@jit(nopython=True)  # 确保使用 Numba 编译
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)

# 使用 numpy 数组来利用 Numba 的优化
arr = np.array([3, 6, 8, 10, 1, 2, 1])

# 调用优化后的快速排序函数
sorted_arr = quicksort(arr)
print(sorted_arr)


import numba
from numba import jit
import numba as nb

@nb.njit(parallel=True)
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

@nb.njit(parallel=True)
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

def sort_array(arr):
    quicksort(arr, 0, len(arr) - 1)
    return arr

import cpp_sort
import torch
import time
for i in range(100):
    test_tensor = torch.randperm(1000000, dtype = torch.int32, device = 'cuda:0')
    tensor = test_tensor[:10000].cpu().numpy()

    start = time.time()
    sorted_arr = cpp_sort.sort(tensor)
    print(f"cpp_sort time: {time.time() - start:.7f}s")

    start = time.time()
    sorted_arr = cpp_sort.mp_sort(tensor)
    print(f"mp_sort time: {time.time() - start:.7f}s")


    tensor = torch.from_numpy(tensor)
    start = time.time()
    sorted_arr = torch.sort(tensor)
    print(f"torch cpu time: {time.time() - start:.7f}s")

import torch
import dgl
nodes = torch.tensor([6,3,7,10,2,3,5,7,1,2,8,11,4,3,12,10,7], dtype = torch.int32, device = 'cuda:0')
root_len = 5
root_nodes = nodes[:root_len]
no_root = nodes[root_len:]

no_root_sort, no_root_sort_indices = torch.sort(no_root)
root_nodes_sort, root_nodes_sort_indices = torch.sort(root_nodes)
table1 = torch.zeros_like(no_root_sort) - 1
table2 = torch.zeros_like(root_nodes_sort) - 1
dgl.findSameIndex(no_root_sort, root_nodes_sort, table1, table2)

no_root_sort_indices = torch.cat((no_root_sort_indices, torch.zeros(1, dtype = torch.int32, device = 'cuda:0') - 1))
no_root_sort_indices = no_root_sort_indices[table2.to(torch.int64)]
no_root_ind = no_root_sort_indices[no_root_sort_indices > -1] + root_len
print(no_root_ind)



nid = torch.tensor([1,5,3,5,6,2,2,3,2,7,1,5,2])
uni, inv = torch.unique(nid, return_inverse=True)
perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
nid = nid[perm]
mail = mail[perm]
mail_ts = mail_ts[perm]



path = '/raid/guorui/DG/dataset/MAG/ext_full.npz'
import numpy as np
import torch
ext = np.load(path)
indptr = torch.from_numpy(ext['indptr']).cuda()

dif = torch.diff(indptr)

big_node = torch.nonzero(dif > 10).reshape(-1)
dif[big_node] = 10
torch.sum(dif) #变成了4亿

dif_sort,_ = torch.sort(dif, descending=True)



#寻找torch的findSameNode 的cpu替代
import torch
tensor1 = torch.arange(10000000, dtype = torch.int32)
tensor2 = torch.arange(5000000,10000000, dtype = torch.int32)
test = torch.isin(tensor1, tensor2, invert=True)
same_node = tensor1[torch.isin(tensor1, tensor2)]

same_node_2 = tensor2[torch.isin(tensor2, tensor1)]

import torch
nodes = torch.tensor([2,5,3,1,2,8,3,10,2,3,5,11,9,4,8])
nodes_sort, nodes_sort_indices = torch.sort(nodes)
nodes_sort_uni, uni_indices = torch.unique(nodes, sorted= True, return_index = True)

nodes = torch.tensor([2,5,3,1,2,8,3,10,2,3,5,11,9,4,8])
uni, uni_indices = torch.unique(nodes, return_inverse = True)


nodes_sort_indices[uni_indices[nodes_sort_uni]]




import torch
tensor_total = torch.zeros(8000000000, dtype = torch.int32)

tensor_cuda = torch.zeros(3000000000, dtype = torch.int32, device = 'cuda:0')

test_cuda = torch.zeros(100000000, dtype = torch.int32, device = 'cuda:0')
import time

for i in range(10):
    test_cuda = torch.zeros(1000000000, dtype = torch.int32)

    start = time.time()
    test_cuda = test_cuda.share_memory_()
    print(f"test {test_cuda.shape[0] * 4 / 1024 ** 2}MB 上share 时间{time.time() - start:.3f}s")



import torch
tensor1 = torch.tensor([3,7,1,2,6,8,12,10,9,5])
tensor2 = torch.tensor([1,8,12,22,6,4,0,19])

table1 = torch.isin(tensor1, tensor2, assume_unique=True,invert=True)


path = '/raid/bear/data/raw/papers100M/dgl_double_PA.bin'
import numpy as np
import time
import torch
#生成一段1000w的
indices = torch.randperm(100000000, dtype = torch.int32)
import multiprocessing

#开10个进程试试
def read_file(indices, dim):
    start = time.time()
    res = np.zeros((indices.shape[0], dim), dtype = np.float32)
    for i in (indices):
        cur = np.fromfile(path, dtype = np.float32, count = dim, offset=i)
        res[i*dim: (i+1) * dim] = cur
    
    print(f"read_file完毕, 共{res.shape}的数据,用时{time.time() - start:.4f}s ")

start = time.time()
# read_file(indices[:100000], 172)

pool = multiprocessing.Pool(processes=10)

for i in range(5):
    result = pool.apply_async(read_file, (indices[i*100000:(i+1)*100000], 172))

pool.close()
pool.join()
print(f"执行完成.")






def cal_memory(s, dim):
    return (s * 4 * dim / 1024**3)



# 测试最终每个节点只取10条边可以优化掉多少条边
import numpy as np
import torch

ds = ['LASTFM','TALK','STACK','GDELT']

#[edge, node]
dim = {
    'LASTFM': [100,100],
    'TALK': [172,172],
    'STACK': [172,172],
    'GDELT': [182,413]
}

for d in ds:
    ext = np.load(f'/raid/guorui/DG/dataset/{d}/ext_full.npz')
    indptr = torch.from_numpy(ext['indptr']).cuda()
    indices = torch.from_numpy(ext['indices']).cuda()
    eid = torch.from_numpy(ext['eid']).cuda()

    dif = torch.diff(indptr)
    #吧大于10的去掉
    ind = torch.nonzero(dif > 10).reshape(-1)
    dif[ind] = 10
    pre_mem = cal_memory(indptr[-1].item(), dim[d][0])
    cur_mem = cal_memory(torch.sum(dif).item(), dim[d][0])
    print(f"{d}: {indptr[-1]} -> {torch.sum(dif)}, 优化了{100-(torch.sum(dif)/indptr[-1]*100):.2f}%,  {pre_mem:.2f}GB -> {cur_mem:.2f}GB")
    # STACK 63497049 -> 10549981
    # TALK 7833139 -> 897411

d = 'MAG'
ext = np.load(f'/raid/guorui/DG/dataset/{d}/ext_full.npz')
indptr = torch.from_numpy(ext['indptr']).cuda()
indices = torch.from_numpy(ext['indices'])
eid = torch.from_numpy(ext['eid'])

dif = torch.diff(indptr)
#吧大于10的去掉
ind = torch.nonzero(dif > 10).reshape(-1)
dif[ind] = 10
pre_mem = cal_memory(indptr[-1].item(), dim[d][0])
cur_mem = cal_memory(torch.sum(dif).item(), dim[d][0])
print(f"{d}: {indptr[-1]} -> {torch.sum(dif)}, 优化了{100-(torch.sum(dif)/indptr[-1]*100):.2f}%,  {pre_mem:.2f}GB -> {cur_mem:.2f}GB")

# LASTFM: 2586206 -> 19489, 优化了99.25%,  0.96GB -> 0.01GB 24372
# TALK: 7833139 -> 897411, 优化了88.54%,  5.02GB -> 0.58GB  691554 1227192
# STACK: 63497049 -> 10549981, 优化了83.39%,  40.69GB -> 6.76GB 7215928  7703249 
# GDELT: 191290882 -> 129529, 优化了99.93%,  129.70GB -> 0.09GB 122141 738143


#实际上
# 


import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler import *


d = 'WIKI'
batch_size = 600
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
neg_link_sampler = NegLinkSampler(num_nodes)

for _, rows in df[:train_edge_end].groupby(group_indexes):  
    root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)).cuda()
    root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)).cuda()


    start = time.time()
    ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
    

    sampler.sample(root_nodes.cpu().numpy(), root_ts.cpu().numpy())
    ret_tgl = sampler.get_ret()

    emptyCache()
    break



import dgl
import torch
map = torch.tensor([10,8,9,1,4,7,6,3], dtype = torch.int32, device = 'cuda:0')
exp_eids = torch.tensor([4,3,7], dtype = torch.int32, device = 'cuda:0')

# nodes_sort, nodes_sort_indices = torch.sort(nodes)
# find_sort, find_sort_indices = torch.sort(find)
# table1 = torch.zeros_like(nodes_sort)
# table2 = torch.zeros_like(find)
# dgl.findSameIndex(nodes_sort, find_sort, table1, table2)


exp_eids_sort,_ = torch.sort(exp_eids)
map_sort,map_sort_indices = torch.sort(map)
table1 = torch.zeros_like(exp_eids_sort) - 1
table2 = torch.zeros_like(map_sort) - 1
dgl.findSameIndex(exp_eids_sort, map_sort, table1, table2)
table1 = map_sort_indices[table1.long()]


import torch

# 示例输入
eids = torch.tensor([3, 1,1, 4, 2])
map = torch.tensor([5, 2, 3, 1, 4, 6])

# 先对map进行排序
sorted_map, indices = torch.sort(map)

# 找到tensor中每个元素在sorted_map中的位置
positions = torch.searchsorted(sorted_map, eids)

# 根据排序的indices来找到在原始map中的位置
original_positions = indices[positions]

print("Tensor:", eids)
print("Map:", map)
print("Positions in map:", original_positions)


import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler import *

d = 'TALK'
batch_size = 600
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
group_indexes = np.array(df[:train_edge_end].index // batch_size)

from sampler_gpu import *
fan_nums = [10, 10]
layers = len(fan_nums)
sampler_gpu = Sampler_GPU(g, fan_nums, layers)

indptr = sampler_gpu.indptr
totaleid = sampler_gpu.totaleid
totalts = sampler_gpu.totalts
indices = sampler_gpu.indices



import torch
import os
import json
def saveBin(tensor,savePath,addSave=False):

    savePath = savePath.replace('.pt','.bin')
    dir = os.path.dirname(savePath)
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    json_path = dir + '/saveBinConf.json'
    
    tensor_info = {
        'dtype': str(tensor.dtype).replace('torch.',''),
        'device': str(tensor.device),
        'shape': (tensor.shape)
    }

    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    
    config[savePath] = tensor_info
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)

    if isinstance(tensor, torch.Tensor):
        tensor.cpu().numpy().tofile(savePath)
    elif isinstance(tensor, np.ndarray):
        tensor.tofile(savePath)
data = 'GDELT'
path = f'/raid/guorui/DG/dataset/{data}/edge_features.pt'
tensor = torch.load(path)
saveBin(tensor, path)

path = f'/raid/guorui/DG/dataset/{data}/node_features.pt'
tensor = torch.load(path)
saveBin(tensor, path)


import numpy as np

def read_binary_file_indices(file_path, indices, shape=(172,), dtype=np.float32):
    """
    从二进制文件中读取特定索引的数据。

    Args:
        file_path (str): 文件路径。
        indices (list or np.array): 要读取的行索引。
        shape (tuple): 每行数据的形状，默认为(172,)。
        dtype (numpy.dtype): 数据类型，默认为np.int32。

    Returns:
        np.ndarray: 请求索引对应的数据。
    """
    # 计算每一行的字节大小
    row_bytes = np.prod(shape) * dtype().itemsize
    
    data = []
    with open(file_path, 'rb') as f:
        for idx in indices:
            # 计算偏移量
            offset = idx * row_bytes
            f.seek(offset)
            
            # 读取一行数据
            row_data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
            row_data = row_data.reshape(shape)
            
            data.append(row_data)
    
    return np.array(data)

# 使用示例
# import torch
# file_path = '/raid/guorui/DG/dataset/STACK/edge_features.bin'
# indices = np.random.randint(0,60000000,size=20000)
# indices = np.concatenate((indices, indices+1, indices+2, indices+3, indices+4))
# indices = np.sort(indices)
# import time
# start = time.time()
# data = torch.from_numpy(read_binary_file_indices(file_path, indices))
# print(f"读取用时{time.time() - start:.3f}s")
# print(data.shape)



import numpy as np
import torch

def read_data_from_file(file_path, indices, shape=(60000000, 172), dtype=np.float32, batch_size=10):
    # 利用 numpy 的内存映射函数来映射整个文件
    data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    
    result = []
    
    # 处理索引，分批读取
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        # 批量读取
        batch_data = data[batch_indices, :]
        result.append(torch.tensor(batch_data))  # 转换为torch tensor
    
    # 将结果合并成一个tensor
    return torch.cat(result, dim=0)

import torch
file_path = '/raid/guorui/DG/dataset/STACK/edge_features.bin'
indices = np.random.randint(0,60000000,size=100000)
# indices = np.concatenate((indices, indices+1, indices+2, indices+3, indices+4))
indices = np.sort(indices)
import time
start = time.time()
data = read_data_from_file(file_path, indices)
print(f"读取用时{time.time() - start:.3f}s")
print(data.shape)
# result现在是包含指定索引数据的torch tensor

start = time.time()
data1 = torch.from_numpy(read_binary_file_indices(file_path, indices))
print(f"读取用时{time.time() - start:.3f}s")
print(data1.shape)

# memmap有额外内存开销速度快，尤其是针对数据比较连续的情况下更快 （但是也有可能memmap走了系统缓存，纯离散读没有走）
# 后面两者都测试一遍... （也许也可以写论文里?）

