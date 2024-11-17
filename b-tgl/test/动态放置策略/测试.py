
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
node = torch.load('/home/guorui/workspace/tmp/node.pt').cuda().to(torch.int32)
block = torch.load('/home/guorui/workspace/tmp/block.pt').cuda().to(torch.int32)

node, block, reorder_res = uni_simple(node, block, 1, reorder_res)
node, block, reorder_res = uni_simple(node, block, 2, reorder_res)
print(f"node shape: {node.shape}")



bit_num = math.ceil((torch.max(block) + 1) / 32)
bitmap = torch.zeros((torch.max(node) + 1, bit_num), dtype = torch.int32, device = 'cuda:0').reshape(-1)
import time
start = time.time()
dgl.init_bitmap(node, block, bitmap, bit_num)
# torch.cuda.synchronize()
# print(f"init bitmap use time:{time.time() - start:.6f}s")
bitmap = bitmap.reshape(-1, bit_num)
matrix = bitmap
cur_count = 50
indices = torch.arange(0 * cur_count,(0+1) * cur_count, dtype = torch.int32, device = 'cuda:0')
labels = torch.zeros((indices.shape[0], matrix.shape[0]), dtype = torch.float32, device= 'cuda:0')

node_ind = torch.arange(matrix.shape[0], dtype = torch.int32, device = 'cuda:0')


count_labels = torch.zeros(torch.max(node), dtype = torch.int32, device = 'cuda:0')
dgl.bincount(node, count_labels)
count_labels_sort, count_labels_ind = torch.sort(count_labels, descending=True)
count_labels_ind = count_labels_ind.to(torch.int32)
count_labels_ind = count_labels_ind[count_labels_sort > 1]

for i in range(matrix.shape[0] // cur_count):
    start_t = time.time()
    indices = count_labels_ind[i * cur_count: (i+1) * cur_count]
    labels.fill_(0)
    dgl.matrix_sim(matrix, indices, labels)
    # print(labels)
    print(f"\n当前处理度数为{count_labels_sort[i * cur_count]}的节点 {cur_count}次用时{time.time() - start_t}")

    for labels_i in range(cur_count):
        print(f"{torch.sum(labels[labels_i] > 0.5)}", end=",")





import torch
test = torch.tensor([1,9,2,3,6,1,4,2,10,8,9,4,6,5,7,2])

test_sort, test_sort_ind = torch.sort(test)
test_uni, test_uni_inv = torch.unique(test_sort, return_inverse = True)

uni, inv = torch.unique(test_uni_inv, return_inverse=True)
uni, inv = uni.cpu(), inv.cpu()
perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
perm = perm.cuda()

ind = test_sort_ind[perm]
ind_sort,_ = torch.sort(ind)



uni, inv = torch.unique(test, return_inverse=True)
uni, inv = uni.cpu(), inv.cpu()
perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
perm = perm.cuda()