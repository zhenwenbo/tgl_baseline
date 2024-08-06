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