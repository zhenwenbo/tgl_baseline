import numpy as np
from utils import *

tgl_nodes = loadBin('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/tgl-node.bin')
tgl_ts = loadBin('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/tgl-ts.bin')

print(tgl_nodes)
print(tgl_ts)

b_nodes = loadBin('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/b-node.bin')
b_ts = loadBin('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/b-ts.bin')

print(b_nodes)
print(b_ts)


b_neg = loadBin('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/b-neg.bin')
print(tgl_neg)

print(f"{torch.sum(tgl_neg != b_neg)}不一样的个数")

import torch

inv1 = torch.load('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/inv.bin')
uni1 = torch.load('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/uni.bin')
inv2 = torch.load('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/inv1.bin')
uni2 = torch.load('/home/guorui/workspace/dgnn/b-tgl/test/test-acc/uni1.bin')

def per(uni, inv):
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = torch.empty(uni.size(0), device=inv.device, dtype=inv.dtype).scatter_(0, inv, perm)
    return perm


def js(t1, t2):
    print(f"不同个数: {torch.sum(t1 != t2)}")

js(inv1, inv2)
js(uni1, uni2)
perm1 = per(uni1, inv1)
perm2 = per(uni2, inv2)
dis = torch.nonzero(perm1 != perm2).reshape
js(per(uni1,inv1), per(uni2,inv2))