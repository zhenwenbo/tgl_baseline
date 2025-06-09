import numpy as np
import dgl

import math
import numpy as np
import torch
import pandas as pd

def sknewness(data):
    n = len(data)
    data = data.cpu().numpy()
    average = np.mean(data)
    diff = data - average
    s2 = np.std(data)
    s3 = diff ** 3
    k = math.sqrt(n*(n-1))/(n-2)
    m = 0
    for i in range(n):
        m += s3[i]
    skewness= k * m / (n * (s2 ** 3))
    return skewness

def cuda_sknewness(data):
    data = np.array(data,dtype=np.float64)
    n = int(data.shape[0])
    data = torch.tensor(data).to('cuda:0')
    average = torch.mean(data).to('cuda:0')
    diff = data - average
    s2 = torch.std(data)
    s3 = diff.pow(3).to('cuda:0')
    k = math.sqrt(n*(n-1))/(n-2)
    m = torch.sum(s3)
    skewness = k * m / (n * (s2 ** 3))
    return skewness


def cal_skew(data):
    path = f'/raid/guorui/DG/dataset/{data}/ext_full.npz'
    ext = np.load(path)
    indptr = ext['indptr']
    indices = ext['indices']
    g = dgl.graph(('csr', (indptr, indices, torch.arange(indices.shape[0], dtype = torch.int32))))
    src = g.edges()[0].to(torch.int32).cuda()
    dst = g.edges()[1].to(torch.int32).cuda()
    node_num = indptr.shape[0]
    inNodeTable = torch.zeros(node_num, dtype = torch.int32, device='cuda:0')
    outNodeTable = torch.zeros(node_num, dtype = torch.int32, device='cuda:0')
    dgl.sumDegree(inNodeTable, outNodeTable, src, dst)

    inDegree = g.in_degrees()
    outDegree = g.out_degrees()
    res = sknewness(inNodeTable + outNodeTable)
    res1 = sknewness(inDegree)


    #计算度数前10%的节点占了多少边
    degree = inNodeTable + outNodeTable
    degree_sort, ind = torch.sort(degree, descending=True)
    total = torch.sum(degree)
    cur = torch.sum(degree_sort[:int(degree_sort.shape[0] * 0.1)])
    print(f"前10%有{cur} 总共有{total} 占比{cur / total * 100:.2f}%")
    cur = torch.sum(degree_sort[:int(degree_sort.shape[0] * 0.2)])
    print(f"前20%有{cur} 总共有{total} 占比{cur / total * 100:.2f}%")

    print(f"{data}空域偏度为{res:.2f}")


datas = ['LASTFM', 'TALK','STACK','GDELT']
datas = ['GDELT']
for data in datas:
    cal_skew(data)