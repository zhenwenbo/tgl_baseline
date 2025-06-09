import torch
import numpy as np
import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
import dgl

src = loadBin('/raid/guorui/DG/dataset/BITCOIN/df-src.bin')
dst = loadBin('/raid/guorui/DG/dataset/BITCOIN/df-dst.bin')

# 训练集出现的节点数
train_num = int(src.shape[0] * 0.7)
src_train = src[:train_num]
dst_train = dst[:train_num]
train_nodes = torch.cat((src_train, dst_train))



pre_root = None
block = 60000
for i in range(src.shape[0] // block):
    cur = torch.unique(torch.cat((src[i * block: (i + 1)*block],dst[i * block: (i + 1)*block])))

    if (pre_root is not None):
        same_root_nodes = torch.isin(pre_root, cur)
        print(f"pre_root中共有节点{pre_root.shape[0]} 在下一次中又出现了{torch.sum(same_root_nodes)}")

    pre_root = cur.clone()