


import numpy as np
import torch

pre = None
pre_ttl = None
reuse_ratio = 0.8
num_nodes = 20

for i in range(10):
    cur_res = np.zeros(10, dtype = np.int32)
    cur_ttl = np.zeros(10, dtype = np.int32)
    if (pre is None):
        cur_res[:] = np.random.randint(num_nodes, size=10)
    else:
        #从pre中取最小的那几个
        reuse_num = int(pre.shape[0] * reuse_ratio)
        pre_ttl_ind = np.argsort(pre_ttl)
        pre_reuse_ind = pre_ttl_ind[:reuse_num]
        cur_res[:reuse_num] = pre[pre_reuse_ind]
        cur_ttl[:reuse_num] = pre_ttl[pre_reuse_ind] + 1
        cur_res[reuse_num:] = np.random.randint(num_nodes, size=cur_res.shape[0] - reuse_num)
        
        ind = torch.randperm(cur_res.shape[0])
        cur_res = cur_res[ind]
        cur_ttl = cur_ttl[ind]

    pre = cur_res
    pre_ttl = cur_ttl
    print(f"结果为: {cur_res}")