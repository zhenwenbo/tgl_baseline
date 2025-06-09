import numpy as np
import torch
from time import time

shape = (1000, 100, 32)
xt = torch.randn(shape, dtype=torch.float32)
xn = np.random.randn(*shape).astype(dtype=np.float32)
# 给定的整数列表
int_list = [i for i in range(8)]

# 目标形状# 例如，2行3列

# 生成具有目标形状的张量
xt1 = torch.randint(low=0, high=len(int_list), size=shape)
#print(xt1)
def np1(x):
    t1 = time()
    u = np.unique(x)
    t2 = time()
    print(f"Numpy just sort: {t2-t1} s")


def np2(x):
    t1 = time()
    u, idx, inv, c = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
    t2 = time()
    print(f"Numpy sort + indexes: {t2-t1} s")


def torch1(x: torch.Tensor):
    t1 = time()
    u = x.unique()
    t2 = time()
    print(f"1:Torch just sort: {t2-t1} s")


def torch2(x: torch.Tensor):
    t1 = time()
    u, idx, c = x.unique(return_inverse=True, return_counts=True)
    t2 = time()
    print(f"2:Torch sort + indexes: {t2-t1} s")


def torch3(x: torch.Tensor):
    t1 = time()
    u, idx, c = x.unique(return_inverse=True, return_counts=True,sorted=False)
    t2 = time()
    print(f"3:Torch sort + indexes: {t2-t1} s")
def torch4(x: torch.Tensor):
    t1 = time()
    x_sorted, _ = torch.sort(x)
    u= torch.unique_consecutive(x_sorted)
    t2 = time()
    print(f"4:Torch only sort: {t2-t1} s")
def torch5(x: torch.Tensor):
    t1 = time()
    x_sorted, _ = torch.sort(x)
    u, idx, c = torch.unique_consecutive(x_sorted, return_inverse=True, return_counts=True)
    t2 = time()
    print(f"5:Torch sort + indexes: {t2-t1} s")
np1(xn)
np2(xn)
torch1(xt)
torch2(xt)
torch3(xt)
torch1(xt1)
torch2(xt1)
torch3(xt1) 
torch4(xt1) 
xt = xt.cuda()
xt1 = xt1.cuda()
torch1(xt)
torch2(xt)
torch3(xt)
torch4(xt) 
torch5(xt)
torch1(xt1)
torch2(xt1)
torch3(xt1)
torch4(xt1)
torch5(xt1)