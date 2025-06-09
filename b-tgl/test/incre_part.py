

# 只有三种情况：
# 上一次出现的节点比这一次的多，一样多，比这一次的少


import torch
import dgl


def incre_strategy(pre_map, cur_id):
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
    print(f"需要增量加载 {incre_nid.shape[0]}个")
    incre_nfeat = torch.randn((incre_nid.shape[0],10), dtype = torch.float32)
    pre_map[incre_map_index] = -1

    #分配incre_indices(表示incre_nfeats要替换哪些已有的)
    indices_able = torch.nonzero(pre_map == -1).reshape(-1)
    len_able = indices_able.shape[0]
    if (len_able < incre_nid.shape[0]):
        print(f"需要扩容")
        pre_map = torch.cat((pre_map, torch.zeros(incre_nid.shape[0] - len_able, dtype = torch.int32, device = 'cuda:0') - 1))
        indices_able = torch.nonzero(pre_map == -1).reshape(-1)

    incre_indices = indices_able[:incre_nid.shape[0]]
    pre_map[incre_indices] = incre_nid

    print(pre_map)
    print(incre_indices)
    print(incre_nid)
    return pre_map, incre_indices, incre_nid

pre_map = torch.tensor([1,3,5,6,-1,2,10,20,16,30,25,17], dtype=torch.int32, device = 'cuda:0')
cur_id = torch.tensor([25,9,6,38,100,120,5], dtype = torch.int32, device = 'cuda:0')
cur_map, incre_indices, incre_nid = incre_strategy(pre_map, cur_id)

cur_id = torch.tensor([25,9,6,38,100,120,5,101,102,103,104], dtype = torch.int32, device = 'cuda:0')
cur_map, incre_indices, incre_nid  = incre_strategy(cur_map, cur_id)

cur_id = torch.tensor([25,9,6,38,100,120,5,101,102,103,104,105,106], dtype = torch.int32, device = 'cuda:0')
cur_map, incre_indices, incre_nid  = incre_strategy(cur_map, cur_id)

cur_id = torch.tensor([25,9,6,38,100,120,5,101,102,103,104], dtype = torch.int32, device = 'cuda:0')
cur_map, incre_indices, incre_nid = incre_strategy(cur_map, cur_id)
