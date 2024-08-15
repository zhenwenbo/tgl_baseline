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
parser.add_argument('--data', default='TALK', type=str, help='dataset name')
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

indptr, indices, eid_res = cooTocsr(src, dst)


# eid = torch.from_numpy(df.eid.values).to(torch.int32).cuda()
ts = torch.from_numpy(df.time.values).to(torch.float64).cuda()

indices = indices.cuda()
indptr = indptr.cuda()
eid = eid_res.cuda()
ts = ts[eid.long()]

g = np.load('./ext_full.npz')
g_indptr = torch.from_numpy(g['indptr']).to(torch.int32)
g_eid = torch.from_numpy(g['eid']).to(torch.int32)
g_indices = torch.from_numpy(g['indices']).to(torch.int32)
g_ts = torch.from_numpy(g['ts']).to(torch.float64).cuda()
g_eid = g_eid.cuda()

print(f"start sort...")
dgl.indptr_sort(indptr, indices, ts, eid)
print(ts)

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


# int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
# int_train_indices = [[] for _ in range(num_nodes)]
# int_train_ts = [[] for _ in range(num_nodes)]
# int_train_eid = [[] for _ in range(num_nodes)]

# int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
# int_full_indices = [[] for _ in range(num_nodes)]
# int_full_ts = [[] for _ in range(num_nodes)]
# int_full_eid = [[] for _ in range(num_nodes)]

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    # if row['int_roll'] == 0:
    #     int_train_indices[src].append(dst)
    #     int_train_ts[src].append(row['time'])
    #     int_train_eid[src].append(idx)
    #     if args.add_reverse:
    #         int_train_indices[dst].append(src)
    #         int_train_ts[dst].append(row['time'])
    #         int_train_eid[dst].append(idx)
    #     # int_train_indptr[src + 1:] += 1
    # if row['int_roll'] != 3:
    #     int_full_indices[src].append(dst)
    #     int_full_ts[src].append(row['time'])
    #     int_full_eid[src].append(idx)
    #     if args.add_reverse:
    #         int_full_indices[dst].append(src)
    #         int_full_ts[dst].append(row['time'])
    #         int_full_eid[dst].append(idx)
    #     # int_full_indptr[src + 1:] += 1
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row['time'])
        ext_full_eid[dst].append(idx)
    # ext_full_indptr[src + 1:] += 1

for i in tqdm(range(num_nodes)):
    # int_train_indptr[i + 1] = int_train_indptr[i] + len(int_train_indices[i])
    # int_full_indptr[i + 1] = int_full_indptr[i] + len(int_full_indices[i])
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

# int_train_indices = np.array(list(itertools.chain(*int_train_indices)))
# int_train_ts = np.array(list(itertools.chain(*int_train_ts)))
# int_train_eid = np.array(list(itertools.chain(*int_train_eid)))

# int_full_indices = np.array(list(itertools.chain(*int_full_indices)))
# int_full_ts = np.array(list(itertools.chain(*int_full_ts)))
# int_full_eid = np.array(list(itertools.chain(*int_full_eid)))

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')
def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]

for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    # tsort(i, int_train_indptr, int_train_indices, int_train_ts, int_train_eid)
    # tsort(i, int_full_indptr, int_full_indices, int_full_ts, int_full_eid)
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

# import pdb; pdb.set_trace()
print('saving...')
# np.savez('DATA/{}/int_train.npz'.format(args.data), indptr=int_train_indptr, indices=int_train_indices, ts=int_train_ts, eid=int_train_eid)
# np.savez('DATA/{}/int_full.npz'.format(args.data), indptr=int_full_indptr, indices=int_full_indices, ts=int_full_ts, eid=int_full_eid)
np.savez('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr, indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)