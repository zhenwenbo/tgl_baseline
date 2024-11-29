


import math
import torch
import dgl

def uni_simple(node, block, freq, reorder_res):
    #简单将出现p次的做完全去重
    count_labels = torch.zeros(torch.max(node) + 1, dtype = torch.int32, device = 'cuda:0')
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
# node = torch.load('/home/guorui/workspace/tmp/root_node_simple.pt').cuda().to(torch.int32)
# block = torch.load('/home/guorui/workspace/tmp/root_block_simple.pt').cuda().to(torch.int32)

def cal_sim(node, block):
    reorder_res = torch.empty(0, dtype = torch.int32)

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
    edges = torch.zeros(300000000, dtype = torch.int32, device = 'cuda:0')
    for i in range(bucket_len):
        cur_bucket = torch.nonzero(bitmap.T[i] > 0).reshape(-1)
        print(cur_bucket.shape[0])

        cur_bitmap = bitmap[cur_bucket]
        edges.fill_(0)
        threshold = 0.8
        print(f"开始CUDA计算相似度")
        start_time = time.time()
        th_num = dgl.matrix_sim_th(cur_bitmap, edges, threshold)
        print(f"cuda计算相似度用时{time.time() - start_time:.4f}s 阈值{threshold} 节点对{th_num}")

        sim = edges[200000000:200000000 + th_num].to(torch.float32) / 100000
        cur_edges = edges[:th_num * 2]

        torch.save(cur_edges, f'/home/guorui/workspace/dgnn/b-tgl/test/动态放置结果/root_edges_{i}.bin')
        torch.save(sim, f'/home/guorui/workspace/dgnn/b-tgl/test/动态放置结果/root_sim_{i}.bin')
