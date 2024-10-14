# 流式预处理：先获取多个子图的访问轨迹，然后流式加载特征文件对每个子图访问轨迹进行写入


import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
import torch
from utils import *
his = []
his_max = []
his_ind = []

# for i in range(100):
#     test = torch.randint(1000000000, (1000000,), dtype = torch.int32)
#     test,_ = torch.sort(test)
#     his.append(test)


his_path = '/raid/guorui/DG/dataset/TALK/part-60000-[10]/'
for i in range(92):
    cur_his_path = f'{his_path}part{i}_edge_map.bin'
    cur_his = loadBin(cur_his_path)
    his.append(cur_his)
    his_max.append(torch.max(cur_his))
    his_ind.append(i)


import time
import numpy as np
feat_path = '/raid/guorui/DG/dataset/TALK/edge_features.bin'
save_path = '/raid/guorui/DG/dataset/TALK/part-60000-[10]'

mask_time = 0
ind_time = 0
# 读取前100w的特征 大概0.64GB
window_size = 1000000
max_feat_len = max(his_max)


feat_len = 172
feat_type_size_Byte = 4
row_bytes = feat_len * feat_type_size_Byte
data_type = np.float32


def stream_extract(feat_path, window_size, his, his_ind, his_max, feat_len, np_type):

    
    mask_time = 0
    ind_time = 0
    max_feat_len = max(his_max).item()
    # print(his_max)
    start = 0
    end = 0

    time_start = time.time()
        

    while True:
        end += window_size
        # 读取start:end的特征
        end = min(end, max_feat_len)
        print(f"end: {end} max_feat_len: {max_feat_len}")
        if (start >= end):
            break

        row_data = np.fromfile(feat_path, dtype=np_type, offset=start * feat_len, count=(end - start) * feat_len)
        row_data = row_data.reshape(-1, feat_len)

        for i, tensor in enumerate(his):
            mask_s = time.time()  
            mask = torch.bitwise_and(tensor >= start, tensor < end)
            mask_time += time.time() - mask_s

            ind_s = time.time()
            cur_ind = tensor[mask]
            ind_time += time.time() - ind_s

            cur_data = torch.from_numpy(row_data[cur_ind - start])
            saveBin(cur_data, f'{save_path}/part{his_ind[i]}_edge_feat_test.bin', addSave=True)

        print(f"{start}:{end}抽取 mask: {mask_time:.2f}s  ind: {ind_time:.2f}s")
        start = end

    print(f"总抽取, mask: {mask_time:.2f}s  ind: {ind_time:.2f}s")
    print(f"总用时: {time.time() - time_start:.4f}s")

for i in range(10):
    stream_extract(1000000, his, his_ind, his_max, feat_len, feat_len * 4, np.float32)
    os.system('find /raid/guorui/DG/dataset/TALK/part-60000-[10] -type f -name \'*_test.bin\' -exec rm -f {} +')

import numpy as np
# 测试是否一样
test_path = '/raid/guorui/DG/dataset/TALK/part-60000-[10]/part10_edge_feat_test.bin'
test_path1 = '/raid/guorui/DG/dataset/TALK/part-60000-[10]/part10_edge_feat.bin'
test = np.fromfile(test_path, dtype = np.float32)
test1 = np.fromfile(test_path1, dtype = np.float32)

print(np.sum(test != test1))