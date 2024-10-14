import numpy as np
import torch
from time import time

shape = (1000, 100, 32)
xt = torch.randn(shape, dtype=torch.float32)
xn = np.random.randn(*shape).astype(dtype=np.float32)


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
    print(f"Torch just sort: {t2-t1} s")


def torch2(x: torch.Tensor):
    t1 = time()
    u, idx, c = x.unique(return_inverse=True, return_counts=True)
    t2 = time()
    print(f"Torch sort + indexes: {t2-t1} s")


np1(xn)
np2(xn)
torch1(xt)
torch2(xt)

xt = xt.cuda()

torch1(xt)
torch2(xt)