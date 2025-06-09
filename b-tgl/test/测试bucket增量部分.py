

data = 'STACK'
fanout = '[10, 10]'
# fanout = '[10]'

path = f'/raid/guorui/DG/dataset/{data}/part-60000-{fanout}'
import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
import os
i = 0

pre_nodes = None
pre_edges = None
while True:
    cur_node_path = f'{path}/part{i}_node_map.bin'
    cur_edge_path = f'{path}/part{i}_edge_map.bin'

    if (not os.path.exists(cur_node_path)):
        break
    cur_nodes = loadBin(cur_node_path)
    cur_edges = loadBin(cur_edge_path)
    
    if (pre_nodes is not None):
        same_nodes = torch.isin(cur_nodes, pre_nodes)
        same_edges = torch.isin(cur_edges, pre_edges)
        same_nodes_num = torch.sum(same_nodes)
        same_edges_num = torch.sum(same_edges)

        print(f"edges cur:{cur_edges.shape[0]} pre:{pre_edges.shape[0]} same: {same_edges_num} 占比{same_edges_num / cur_edges.shape[0] * 100:.2f}%")
        print(f"nodes cur:{cur_nodes.shape[0]} pre:{pre_nodes.shape[0]} same: {same_nodes_num} 占比{same_nodes_num / cur_nodes.shape[0] * 100:.2f}%")

    pre_nodes = cur_nodes.clone()
    pre_edges = cur_edges.clone()


    i += 1