


import torch
import numpy as np

part_node_map = torch.tensor([1,6,3,7,8,10,21,4,11,13,16,50,88,47], dtype = torch.int32)

pos_node_map = torch.tensor([1,20,21,22,23,24,25,26], dtype = torch.int32)
dis_neg_nodes = torch.tensor([10,11,12,13,14,15,17,6,88,50], dtype= torch.int32)

dd_ind = torch.isin(dis_neg_nodes, part_node_map, assume_unique=True,invert=True)
dd_neg_nodes = dis_neg_nodes[dd_ind]


# 找出dis_neg_nodes中在part_node_map出现的那些在最终的indices中的位置

res_d_map = torch.cat((torch.ones(pos_node_map.shape[0], dtype = torch.bool), dd_ind))
res_node_map = torch.cat((pos_node_map, dis_neg_nodes))
res_node_map,node_sort_indices = torch.sort(res_node_map)

cur_d_map = torch.nonzero(~res_d_map[node_sort_indices]).reshape(-1)
cur_d_node = res_node_map[cur_d_map]