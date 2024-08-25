# 测试pre_fetch


path = '/raid/guorui/DG/dataset/MAG/part-600000-[10]'
import torch
import dgl

pre = None
for i in range(1700,1800):
    cp = path + f'/part{i}_node_map.pt'
    cur = torch.load(cp).cuda()
    if (pre is not None):
        cur,_ = torch.sort(cur)
        pre,_ = torch.sort(pre)

        table1 = torch.zeros_like(cur, dtype = torch.int32, device = 'cuda:0') - 1
        table2 = torch.zeros_like(pre, dtype = torch.int32, device = 'cuda:0') - 1
        dgl.findSameNode(cur, pre, table1, table2)
        print(f"cur: {i} pre: {i-1} 总量: {table1.shape[0]} 相同数：{torch.sum(table1 > 0)} 增量加载: {torch.sum(table1 < 0)}")

        # break
    pre = cur

import os
fd = os.open( "dat_edges.pt", os.O_RDWR|os.O_CREAT|os.O_DIRECT)

import os, subprocess, ctypes, binascii, directio
def open_device(path):
    fd = os.open(path, os.O_RDWR | os.O_DIRECT)
    return fd

def close_device(fd):
    fd = os.close(fd)

def write_disk(fd, wbuf, offset):
    os.lseek(fd, offset, os.SEEK_SET)
    rv = directio.write(fd, wbuf)
    if rv == None:
        return -1
    return rv

def read_disk(fd, rlen, offset):

    os.lseek(fd, offset, os.SEEK_SET)
    rbuf = directio.read(fd, rlen)
    return rbuf

import struct
def unpack( res):
    
    # 计算文件中 int32 数组的元素个数
    num_elements = len(res) // 4  # 每个 int32 占4字节
    
    # 读取整个文件内容
    
    # 使用 struct.unpack 来解析二进制数据
    # '<' 表示小端字节序，'i' 表示 int32
    int32_array = struct.unpack(f'<{num_elements}i', res)
    return int32_array


fd = open_device('test.bin')
res = read_disk(fd, 102400000 * 4, 0)
res = unpack(res)
res = torch.tensor(res, dtype = torch.int32)

res = np.fromfile('test.bin', dtype = np.int32)

import numpy as np
import torch
def saveBin(tensor,savePath,addSave=False):
    global saveTime
    if addSave :
        with open(savePath, 'ab') as f:
            if isinstance(tensor, torch.Tensor):
                tensor.numpy().tofile(f)
            elif isinstance(tensor, np.ndarray):
                tensor.tofile(f)
    else:
        if isinstance(tensor, torch.Tensor):
            tensor.numpy().tofile(savePath)
        elif isinstance(tensor, np.ndarray):
            tensor.tofile(savePath)

import numpy as np
import torch


test = torch.arange(1000000000, dtype = torch.int32)
saveBin(test, 'test.bin')

torch.save(test, 'test.pt')

import time
import torch
import numpy as np
base_path = '/raid/guorui/DG/dataset/STACK/part-600000-[10]/'
for i in range(0,100):
    start = time.time()
    res = torch.load(f'{base_path}part{i}_edge_feat.pt')
    print(f"torch加载 {time.time() - start:.8f}s")
    print(res)
    start = time.time()
    res = torch.from_numpy(np.fromfile(f'{base_path}part{i}_edge_feat.pt', dtype = np.float32))
    print(res)
    print(f"np二进制加载 {time.time() - start:.8f}s")