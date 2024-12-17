


import sys
import os

def print_proc_io():
    # 获取当前进程的 PID
    pid = os.getpid()
    
    # 构造 /proc/<pid>/io 的路径
    io_file_path = f"/proc/{pid}/io"
    
    try:
        # 读取并打印 /proc/<pid>/io 文件内容
        with open(io_file_path, 'r') as io_file:
            io_data = io_file.read()
            print(f"Contents of /proc/{pid}/io:\n")
            print(io_data)
    except FileNotFoundError:
        print(f"Error: The file {io_file_path} does not exist.")
    except PermissionError:
        print(f"Error: Insufficient permissions to read {io_file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def drop_caches_with_os():
    with open('/proc/sys/vm/drop_caches', 'w') as stream:
        stream.write('1\n')



root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
path = '/raid/guorui/DG/dataset/TALK/edge_features.bin'
drop_caches_with_os()

# 读取10w个随机的feat
import time
# ind_random = torch.randint(0, 7833138, (100000, ), dtype = torch.int32)


ind_random = torch.arange(100000, dtype = torch.int32)
print(f"开始load 随机10w")
start = time.time()
loadBinDisk(path, ind_random)
print_proc_io()


# sudo PYTHONPATH=/usr/local/miniconda3/bin python /home/guorui/workspace/tgl-baseline/b-tgl/test/测试IO读取的真实传输.py


import torch

def create_interval_map(ind, tensor_len, interval=4096):

    tensor_start = ind * tensor_len * 4
    tensor_end = (ind + 1) * tensor_len * 4 - 1
    tensor_ind = torch.cat((tensor_start, tensor_end))
    tensor, _ = torch.sort(tensor_ind)

    # 将tensor中的每个值转换为对应的区间索引
    indices = tensor.div(interval, rounding_mode='floor').long()
    # 找到所有不同的区间索引
    unique_indices = torch.unique(indices)
    # 计算最大索引
    max_idx = int((tensor.max().item() // interval) + 1)
    # 创建一个全为False的布尔张量
    map_tensor = torch.zeros(max_idx, dtype=torch.bool)
    # 使用scatter_将存在的区间的索引位置设置为True
    map_tensor[unique_indices] = True
    
    return map_tensor

ind_random = torch.randint(0, 7833138, (1000000, ), dtype = torch.int64)
# ind_random = torch.arange(100000, dtype = torch.int32)

# tensor = torch.tensor([0, 2048, 4097, 8192, 12289], dtype=torch.float)
# 创建map
interval_map = create_interval_map(ind_random, 172)

page_num = torch.sum(interval_map)
io = page_num * 4 * 1024
print(f"实际读取了{page_num}个page, IO量: {io / 1024 ** 2}MB")