

import numpy as np
import torch

import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler.sampler import *


# path = '/raid/gr/DG/dataset/TALK/part-60000-[10, 10]/part5_node_map.bin'
# path1 = '/raid/gr/DG/dataset/TALK/正确的二跳/part5_node_map.bin'

# res1 = loadBin(path)
# res2 = loadBin(path1)

# print(torch.sum(res1 != res2))

# res1 = np.fromfile(path, dtype = np.float32)
# res2 = np.fromfile(path1, dtype = np.float32)

node_map = torch.load('/home/gr/workspace/bucket_cache_node_map.pt')
node_feat = torch.load('/home/gr/workspace/bcnf.pt')

node_feats = loadBin('/raid/guorui/DG/dataset/TALK/node_features.bin')
real_node_feat = node_feats[node_map]

pos_node_map = torch.tensor([3,7,12,13,31,37,38,200000], dtype = torch.int32)
cmask = torch.isin(pos_node_map, node_map)
cfeat = node_feat[torch.isin(node_map, pos_node_map)]
feat1 = node_feats[pos_node_map.long()]

pos_node_map = torch.load('/home/gr/workspace/pos_node_map.pt')
pos_node_feat = node_feats[pos_node_map.long()]
torch.save(pos_node_feat, '/home/gr/workspace/pos_node_feat.pt')

bucket_cache_node_map = torch.load('/home/gr/workspace/bcnp.pt')
pos_node_map = torch.load('/home/gr/workspace/pos_node_map.pt')
bucket_cache_node_feat = torch.load('/home/gr/workspace/bcnf.pt')

hybrid_node_cache_mask = torch.isin(pos_node_map, bucket_cache_node_map)
bucket_cache_node_map_sort, bucket_cache_node_map_indices = torch.sort(bucket_cache_node_map)

# 需要找出pos_node在bucket中的实际位置
# 先找出pos中有，bucket中也有的
pos_in_cache_mask = torch.isin(pos_node_map, bucket_cache_node_map, assume_unique=True)
pos_in_cache = pos_node_map[pos_in_cache_mask]
bucket_cache_node_map_sort, bucket_cache_node_map_indices = torch.sort(bucket_cache_node_map)
result = bucket_cache_node_map_indices[torch.searchsorted(bucket_cache_node_map_sort, pos_in_cache)]
hybrid_node_cache_feat = bucket_cache_node_feat[result]

tensor2 = torch.tensor([2,5,3,7,12,9,4])
tensor1 = torch.tensor([2,7,12,2])
tensor2_sorted, argsort = torch.sort(tensor2)
result = argsort[torch.searchsorted(tensor2_sorted, tensor1)]

hybrid_node_cache_feat = bucket_cache_node_feat[torch.isin(bucket_cache_node_map_sort, pos_node_map)][bucket_cache_node_map_indices]

need_nodes = pos_node_map[hybrid_node_cache_mask]
node_feats[need_nodes]


import torch
path = '/home/gr/workspace/false/'
pos_node_map = torch.load( path + 'pos_node_map.pt')
node_feats = torch.load( path + 'node_feats.pt')
node_d_map = torch.load( path + 'node_d_map.pt')

path = '/home/gr/workspace/true/'
pos_node_map1 = torch.load( path + 'pos_node_map.pt')
node_feats1 = torch.load( path + 'node_feats.pt')
node_d_map1 = torch.load( path + 'node_d_map.pt')

print(f"{torch.sum(pos_node_map != pos_node_map1)}")
print(f"{torch.sum(node_feats != node_feats1)}")
print(f"{torch.sum(node_d_map != node_d_map1)}")

import torch
node_map1 = torch.load('/home/gr/workspace/node_d_map_pre1.pt')
node_map = torch.load('/home/gr/workspace/node_d_map_pre.pt')


import torch
import dgl
def findIndex(tensor1, tensor2):
    tensor2, tensor2_sort_indices = torch.sort(tensor2)

    table1 = torch.zeros_like(tensor1, dtype = torch.int32) - 1
    table2 = torch.zeros_like(tensor2, dtype = torch.int32) - 1

    dgl.findSameIndex(tensor1, tensor2, table1, table2)
    res = tensor2_sort_indices[table1.long()]
    return res

tensor1 = torch.tensor([11,1,9,7,17], dtype = torch.int32, device = 'cuda:0')
tensor2 = torch.tensor([1,5,6,11,20,17,7,9], dtype = torch.int32, device=  'cuda:0')

res = findIndex(tensor1, tensor2)
# tensor1_sort[0] = tensor1[indices[0]]
# table1[0]表示tensor1_sort[0] 在 tensor2_sort[table1[0]]中的索引位置
# tensor1_sort[0] 即 tensor1[indices[0]] 在 tensor2_sort中的索引为table1[0]
# tensor1[indices[0]] 在 tensor2_sort[table1[0]]的索引位置为table1[0]
# 

import torch

def find_indices(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # 确保A和B在同一设备上
    assert A.device == B.device, "A and B must be on the same device"
    
    # 使用searchsorted找到插入位置
    indices = torch.searchsorted(B, A)
    
    # 创建有效索引的掩码（防止越界）
    valid_mask = indices < B.size(0)
    
    # 生成安全索引（将越界位置设为0以避免访问错误）
    safe_indices = torch.where(valid_mask, indices, torch.tensor(0, device=A.device))
    
    # 检查对应位置的元素是否匹配
    element_equal = B[safe_indices] == A
    
    # 合并有效性及元素匹配条件
    final_mask = valid_mask & element_equal
    
    # 生成最终结果，符合条件的保留索引，否则为-1
    result = torch.where(final_mask, indices, torch.tensor(-1, device=A.device))
    
    return result

import time
for i in range(100):
    tensor1 = torch.randint(0,1000000,(10000000,), dtype = torch.int32, device = 'cuda:0')
    tensor2 = torch.arange(2000000, dtype = torch.int32, device=  'cuda:0')
    start = time.time()
    res = find_indices(tensor2, tensor1)
    print(f"用时 {time.time() - start}")


import torch

def find_indices_in_vectorB(A, B):
    # 对向量 B 进行排序
    B_sorted, B_indices = torch.sort(B)
    
    # 使用 searchsorted 找到 A 中元素在排序后的 B 中的插入位置
    left = torch.searchsorted(B_sorted, A, side='left')
    right = torch.searchsorted(B_sorted, A, side='right')
    
    # 判断元素是否存在
    mask = (left == right)
    
    # 初始化结果索引为 -1
    result = -torch.ones_like(left)
    
    # 对于存在的元素，找到其在原始向量 B 中的索引
    valid_indices = left[mask]
    result[mask] = B_indices[valid_indices]
    
    return result

# 示例
A = torch.tensor([5, 3, 10, 7])
B = torch.tensor([1, 3, 5, 7, 9, 11])

result = find_indices_in_vectorB(A, B)
print(result)  # 输出: tensor([1, 2, 3, -1])