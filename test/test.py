import pandas as pd
def count_edges_per_partition(file_path):
    # 读取txt文件到DataFrame中
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['src', 'dst', 'partition'], dtype={'src': int, 'dst': int, 'partition': int})
    # 使用groupby和size方法来计算每个分区的边数
    edges_count = df.groupby('partition').size().reset_index(name='edges')

    return edges_count

# 指定文件路径
file_path = '/raid/wsy/tmp/2pscpp10/32/uk.txt'

# 调用函数并打印结果
result = count_edges_per_partition(file_path)
print(result)


import torch
tensor1 = torch.ones(100000000 * 5, dtype = torch.int32, device = 'cuda:0')

import time

for i in range(100):
    start = time.time()
    test = tensor1[:100000000]
    print(f"用时: {time.time() - start:.8f}s")

tensor2 = torch.ones(100000000 * 5, dtype = torch.int32).pin_memory()

import time

for i in range(100):
    test2 = tensor1[:100000000]
    torch.sort(tensor1[:100000000])
    start = time.time()
    
    test = tensor1[:100000000].cuda()
    # test1 = test.cpu().pin_memory()
    print(f"用时: {time.time() - start:.8f}s")




#获取所有数据集的节点个数和边个数
datasets = ['LASTFM', 'TALK', 'STACK', 'GDELT']
import numpy as np
path = '/raid/guorui/DG/dataset'
res_data = {}

for dataset in datasets:
    cur_path = path + f"/{dataset}/ext_full.npz"

    g = np.load(cur_path)
    res_data[dataset] = {"node_num": g['indptr'].shape[0] - 1,"edge_num": g['indptr'][-1]}
    # break
print(res_data)


import json
path = '/home/gr/workspace/dgnn/dataset.json'

with open(path) as f:
    data = json.load(f)


for key in data:
        
    if (key == 'LASTFM'):
        data[key]['dim_edge_feat'] = 100
        data[key]['dim_node_feat'] = 100
    elif (key == 'TALK'):
        data[key]['dim_edge_feat'] = 172
        data[key]['dim_node_feat'] = 172
    elif (key == 'STACK'):
        data[key]['dim_edge_feat'] = 172
        data[key]['dim_node_feat'] = 172
    elif (key == 'GDELT'):
        data[key]['dim_edge_feat'] = 182 #TODO 为什么下载下来的数据集的edge feat是182呢？
        data[key]['dim_node_feat'] = 413
    else:
        continue

    data[key]['storage_edge_feat_GB'] = f"{data[key]['edge_num'] * data[key]['dim_edge_feat'] * 4 / 1024**3:.5f}"
    data[key]['storage_node_feat_GB'] = f"{data[key]['node_num'] * data[key]['dim_node_feat'] * 4 / 1024**3:.5f}"

print(data)

with open(path, 'w') as f:
    json.dump(data, f)