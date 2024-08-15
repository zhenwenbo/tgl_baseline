
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


def cooToDAG(src, dst, ts):
    edge_num = src.shape[0]
    src = torch.cat((src, dst))
    dst = torch.cat((dst, src))
    # ts = torch.cat((ts,ts))
    indptr, indices, eid = cooTocsr(src, dst)

    indices = indices.cuda()
    indptr = indptr.cuda()
    eid = eid.cuda()
    #eid初始是从0开始的，eid个数和原始的src挂钩  0 -> (edge_num - 1) 
    #大于等于edge_num减去edge_num
    eid[eid >= edge_num] -= edge_num
    ts = ts[eid.long()]

    print(f"start indptr sort")
    dgl.indptr_sort(indptr, indices, ts, eid)
    print(f"end indptr sort")

    out_src = torch.zeros_like(indices) - 1
    out_dst = torch.zeros_like(indices) - 1
    dgl.csr2dag(indptr, indices, eid, out_src, out_dst)

    # 去除不存在的边
    mask = out_src > -1
    out_src = out_src[mask]
    out_dst = out_dst[mask]

    # 去除自环边
    mask = torch.nonzero(out_src != out_dst)
    out_src = out_src[mask]
    out_dst = out_dst[mask]

    # 去除重复的边
    out_edges = torch.stack((out_src, out_dst), dim = 1)
    out_edges = torch.unique(out_edges, dim = 0).reshape(-1)

    out_src = out_edges[0::2]
    out_dst = out_edges[1::2]

    return out_edges



src = torch.tensor([3,8,2,2,2,2,2,1,2,1,1,1,1,1,3,1,7], dtype = torch.int32, device = 'cuda:0')
dst = torch.tensor([1,8,1,2,2,2,2,2,1,3,4,1,1,1,1,5,1], dtype = torch.int32, device = 'cuda:0')
ts = torch.arange(0, src.shape[0], dtype = torch.float64, device = 'cuda:0')
ts = torch.zeros(src.shape[0], dtype = torch.float64, device = 'cuda:0')
# eid = torch.arange(src.shape[0], dtype = torch.float64, device = 'cuda:0')

res = cooToDAG(src, dst, ts)
src = res[::2]
dst = res[1::2]
import torch
import dgl

def inDegree(src, dst):
    node_num = max(torch.max(src), torch.max(dst))

    inNodeTable = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
    outNodeTable = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
    
    dgl.sumDegree(inNodeTable, outNodeTable, src, dst)

    return inNodeTable


d = inDegree(src, dst)

import argparse
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--data', default='LASTFM', type=str, help='dataset name')
args=parser.parse_args()

file_path = '/raid/guorui/DG/dataset/{}/edges.csv'.format(args.data)

if (args.data == 'BITCOIN'):
    df = pd.read_csv(file_path, sep=' ', names=['src', 'dst', 'time'])
else:
    df = pd.read_csv(file_path)

src = df.src.values
dst = df.dst.values

src = torch.from_numpy(src).to(torch.int32).cuda()
dst = torch.from_numpy(dst).to(torch.int32).cuda()
ts = torch.from_numpy(df.time.values).to(torch.float64).cuda()

out_edges = cooToDAG(src, dst, ts)

save_path = f'/raid/guorui/DG/dataset/{args.data}/dag_edges.pt'
torch.save(out_edges, save_path)


