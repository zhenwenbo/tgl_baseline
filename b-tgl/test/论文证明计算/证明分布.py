
import torch
import numpy as np
class ReNegLinkSampler:

    pre_res = None
    pre_ttl = None
    ratio = 1

    def __init__(self, num_nodes, ratio):
        self.ratio = ratio
        self.pre_res = None
        print(f"重放负采样，重放比例为 {self.ratio}")
        self.num_nodes = num_nodes

    def sample(self, n):

        if (self.ratio <= 0 or (self.pre_res is not None and n < self.pre_res.shape[0])):
            return np.random.randint(self.num_nodes, size=n)
        
        cur_res = np.zeros(n, dtype = np.int32)
        cur_ttl = np.zeros(n, dtype = np.int32)
        if (self.pre_res is None):
            cur_res[:] = np.random.randint(self.num_nodes, size=n)
        else:
            #从pre中取最小的那几个
            reuse_num = int(self.pre_res.shape[0] * self.ratio)
            pre_ttl_ind = np.argsort(self.pre_ttl)
            pre_reuse_ind = pre_ttl_ind[:reuse_num]
            cur_res[:reuse_num] = self.pre_res[pre_reuse_ind]
            cur_ttl[:reuse_num] = self.pre_ttl[pre_reuse_ind] + 1
            cur_res[reuse_num:] = np.random.randint(self.num_nodes, size=cur_res.shape[0] - reuse_num)
            
            ind = torch.randperm(cur_res.shape[0])
            cur_res = cur_res[ind]
            cur_ttl = cur_ttl[ind]
            
        sameNum = 0
        if (self.pre_res is not None):
            sameNum = np.sum(np.isin(cur_res, self.pre_res))
        self.pre_res = cur_res
        self.pre_ttl = cur_ttl

        # print(f'相同的负采样节点个数: {sameNum} 占比: {sameNum / n * 100:.2f}%')
        return cur_res
    
# num_nodes = 2601977
# sampler = ReNegLinkSampler(num_nodes=num_nodes, ratio=0.8)
# res_num = torch.zeros(num_nodes, dtype = torch.int32, device = 'cuda:0')
# E = 60000000
# for i in range(E // 60000):
#     res = torch.from_numpy(sampler.sample(60000)).cuda()
#     res_num[res.long()] += 1

# print(res_num)
# print(f"var: {torch.var(res_num.float())}, max: {torch.max(res_num)}, min: {torch.min(res_num)}")


import math
def js(vb,v,a):
    res = 0.0
    res += (vb) * ((a + (1-a)/v) * math.log2((2 * a * v + 2 - 2*a)/(a*v+2-a)))
    res += (v-vb)*((1-a) / v) * math.log2((2-2*a)/(2-a))

    res += ((vb/v) * math.log2(2 / (a * v + 2 - a)) + math.log2(2 / (2 - a)) - (vb / v) * math.log2(2 / (2 - a)))
    res = res * 0.5
    return res

v = 2600000
vb = 54000
ass = [0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for a in ass:
    print(f"a: {a}, js: {js(vb,v,a)}")