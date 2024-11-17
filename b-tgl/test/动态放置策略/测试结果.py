


import torch
import numpy as np
import dgl



# for i in range(1):
#     edges = torch.load(f'/home/guorui/workspace/dgnn/b-tgl/test/动态放置结果/edges_{i}.bin')
#     src = edges[::2]
#     dst = edges[1::2]

#     src = torch.cat((src, edges[1::2]))
#     dst = torch.cat((dst, edges[::2]))

#     src_sort, sort_ind = torch.sort(src)
#     dst_sort = dst[sort_ind]

#     edges_reorder = torch.unique(torch.stack((src_sort,dst_sort),dim=-1).reshape(-1))




import math
import torch
import dgl
reorder_res = torch.empty(0, dtype = torch.int32)

def uni_simple(node, block, freq, reorder_res):
    #简单将出现p次的做完全去重
    count_labels = torch.zeros(torch.max(node), dtype = torch.int32, device = 'cuda:0')
    count_labels = count_labels[1:]
    dgl.bincount(node, count_labels)
    label_cum = torch.cumsum(count_labels, dim = 0)


    dis_node_idx = torch.nonzero(count_labels == freq).reshape(-1) #这个是count_labels的indices，实际就是node的id

    res = None

    for i in range(freq):
        if (res is None):
            res = block[label_cum[dis_node_idx] - freq].reshape(1,-1)
        else:
            res = torch.cat((res, block[label_cum[dis_node_idx] - freq + i].reshape(1,-1)), dim = 0)
    
    res = res.T
    res_uni, res_uni_inv,res_uni_count = torch.unique(res, return_inverse=True, return_counts = True, dim = 0)
    threshold = freq
    #没有重复超过2次的都去掉

    res_uni_inv_sort, res_uii = torch.sort(res_uni_inv) # 此处res_uii代表dis_node_idx中的node dis_node_idx[res_uii]就是节点idx

    res_uni_inv_label = torch.zeros(torch.max(res_uni_inv_sort), dtype = torch.int32, device = 'cuda:0')
    dgl.bincount(res_uni_inv_sort.to(torch.int32), res_uni_inv_label)

    res_uni_inv_threshold_ind = torch.nonzero(res_uni_inv_label < threshold).reshape(-1)
    res_uni_inv_threshold_mask = ~torch.isin(res_uni_inv_sort, res_uni_inv_threshold_ind) #表示res_uni_inv_sort中小于threshold的mask
    
    res_ret = dis_node_idx[res_uii[res_uni_inv_threshold_mask]] # 最终的node序列

    reorder_res = torch.cat((reorder_res, res_ret.cpu()))

    # 最后把node和block中已经构建好序列的node去掉
    node_reordered_mask = torch.isin(node, res_ret)
    node = node[~node_reordered_mask]
    block = block[~node_reordered_mask]

    return node, block, reorder_res



import math
import torch
import dgl
node = torch.load('/home/guorui/workspace/tmp/root_node.pt').cuda().to(torch.int32)
block = torch.load('/home/guorui/workspace/tmp/root_block.pt').cuda().to(torch.int32)

node, block, reorder_res = uni_simple(node, block, 1, reorder_res)
node, block, reorder_res = uni_simple(node, block, 2, reorder_res)
print(f"node shape: {node.shape}")



bit_num = math.ceil((torch.max(block) + 1) / 31)
bitmap = torch.zeros((torch.max(node) + 1, bit_num), dtype = torch.int32, device = 'cuda:0').reshape(-1)
import time
start = time.time()
dgl.init_bitmap(node, block, bitmap, bit_num)
bitmap = bitmap.reshape(-1, bit_num)


# 分维度一个一个桶的做相似度计算
bucket_len = bitmap.shape[1]
for i in range(bucket_len):
    cur_bucket = torch.nonzero(bitmap.T[i] > 0).reshape(-1)
    edges = torch.load(f'/home/guorui/workspace/dgnn/b-tgl/test/动态放置结果/root_edges_{i}.bin')
    src = edges[::2]
    dst = edges[1::2]

    src = torch.cat((src, edges[1::2]))
    dst = torch.cat((dst, edges[::2]))

    src_sort, sort_ind = torch.sort(src)
    dst_sort = dst[sort_ind]

    edges_reorder = cur_bucket[torch.unique(torch.stack((src_sort,dst_sort),dim=-1).reshape(-1)).long()].cpu()
    edges_reorder = edges_reorder[~torch.isin(edges_reorder, reorder_res)]

    reorder_res = torch.cat((reorder_res, edges_reorder.cpu()))
    print(f"处理{i}后, res:{reorder_res.shape[0]}")

# 最终结果中，对node_num中未出现的统一置于最后面
node_num = 24575383
total_node = torch.arange(0, node_num + 1, dtype = torch.int32).cuda()
reorder_res = reorder_res.cuda()
dis_node = total_node[torch.isin(total_node, reorder_res, invert=True)]

nid = torch.cat((reorder_res, dis_node))

res = torch.zeros(node_num + 1, dtype=torch.long).cuda()
res.scatter_(0, nid, torch.arange(len(nid)).cuda())
import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
saveBin(res.cpu(), '/raid/guorui/DG/dataset/BITCOIN/node_reorder_map.bin')