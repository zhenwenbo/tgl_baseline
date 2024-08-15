import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import dgl

def cooTocsr(srcList,dstList,sliceNUM=1,device=torch.device('cpu')):
    
    srcList = srcList.cuda()
    max_value = max(torch.max(srcList).item(), torch.max(dstList).item()) + 1   # 保证对齐
    binAns = torch.zeros(max_value,dtype=torch.int32,device="cuda")
    dgl.bincount(srcList,binAns)
    srcList = srcList.cpu()
    ptrcum = torch.cumsum(binAns, dim=0)
    zeroblock=torch.zeros(1,device=ptrcum.device,dtype=torch.int32)
    inptr = torch.cat([zeroblock,ptrcum]).to(torch.int32).cuda()
    binAns,ptrcum,zeroblock = None,None,None
    indice = torch.zeros_like(srcList,dtype=torch.int32).cuda()
    eid = torch.zeros_like(srcList,dtype=torch.int32).cuda()

    addr = inptr.clone()[:-1]
    if sliceNUM == 1:
        dstList = dstList.cuda()
        srcList = srcList.cuda()
        dgl.cooTocsrArg(inptr,indice,addr,srcList,dstList,eid) # compact dst , exchange place
        inptr,indice = inptr.cpu(),indice.cpu()
        srcList,dstList,addr=None,None,None
        return inptr,indice, eid

parser=argparse.ArgumentParser()
parser.add_argument('--data', default='REDDIT', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')
args=parser.parse_args()

file_path = '/raid/guorui/DG/dataset/{}/edges.csv'.format(args.data)

if (args.data == 'BITCOIN'):
    df = pd.read_csv(file_path, sep=' ', names=['src', 'dst', 'time'])
else:
    df = pd.read_csv(file_path)
df['eid'] = range(0, len(df))


num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

src = df.src.values
dst = df.dst.values

src = torch.from_numpy(src).to(torch.int32)
dst = torch.from_numpy(dst).to(torch.int32)
ts = torch.from_numpy(df.time.values).to(torch.float64).cuda()

src = torch.cat((src, dst))
dst = torch.cat((dst, src))
ts = torch.cat((ts,ts))
indptr, indices, eid_res = cooTocsr(src, dst)


# eid = torch.from_numpy(df.eid.values).to(torch.int32).cuda()

indices = indices.cuda()
indptr = indptr.cuda()
eid = eid_res.cuda()
ts = ts[eid.long()]


dgl.indptr_sort(indptr, indices, ts, eid)

out_src = torch.zeros_like(indices) - 1
out_dst = torch.zeros_like(indices) - 1
dgl.csr2dag(indptr, indices, eid, out_src, out_dst)

# 去除不存在的边
mask = out_src > -1
out_src = out_src[mask]
out_dst = out_dst[mask]

max_num = max(torch.max(out_src), torch.max(out_dst))
inNodeTable = torch.zeros(max_num, dtype = torch.int32, device = 'cuda:0')
outNodeTable = torch.zeros(max_num, dtype = torch.int32, device = 'cuda:0')
dgl.sumDegree(inNodeTable,outNodeTable,out_src,out_dst)

