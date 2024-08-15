
import torch
import dgl

def inDegree(src, dst, node_num):

    inNodeTable = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
    outNodeTable = torch.zeros(node_num, dtype = torch.int32, device = 'cuda:0')
    
    dgl.sumDegree(inNodeTable, outNodeTable, src, dst)

    return inNodeTable, outNodeTable

dataset = 'LASTFM'
g = torch.load(f'/raid/guorui/DG/dataset/{dataset}/dag_edges.pt')

src = g[0::2]
dst = g[1::2]

# 假设max为3，那么从0开始就有4个节点为0,1,2,3，所以计算度的时候要有max + 1个节点，这里
node_num = max(torch.max(src), torch.max(dst)) + 2
ind, outd = inDegree(src, dst, node_num)

# topo_res = torch.zeros(node_num, )
import time
#找到当前入度为0的节点，用作训练的正边
# while True:
#     # time.sleep(0.1)


#     cut_node = torch.nonzero(ind == 0).reshape(-1)
#     # ind[cut_node] = -1
#     cut_edge = torch.nonzero(torch.isin(src, cut_node)).reshape(-1)

#     src[cut_edge] = node_num - 1
#     dst[cut_edge] = node_num - 1
#     ind, outd = inDegree(src, dst, node_num)
#     # cut_node_dst = dst[cut_edge]
#     print(f"释放了{cut_edge.shape[0]}条边")
#     # ind[cut_node_dst.long()] -= 1

import torch
import networkx as nx

# 假设src和dst是PyTorch的张量，表示边的源点和终点
dataset = 'LASTFM'
g = torch.load(f'/raid/guorui/DG/dataset/{dataset}/dag_edges.pt')

src = g[0::2]
dst = g[1::2]
# 将PyTorch张量转换为Python列表
edges = list(zip(src.tolist(), dst.tolist()))

# 创建一个有向图
G = nx.DiGraph()

# 添加边到图中
G.add_edges_from(edges)

# 进行拓扑排序
try:
    topological_order = list(nx.topological_sort(G))
    # print("拓扑排序结果:", topological_order)
except nx.NetworkXUnfeasible:
    print("图中存在环，无法进行拓扑排序。")

res = torch.tensor(topological_order, dtype = torch.int32)

topo_path = f'/raid/guorui/DG/dataset/LASTFM/dag_topo.pt'
torch.save(res, topo_path)