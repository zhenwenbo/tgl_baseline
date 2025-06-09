



import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
import numpy as np
import torch

def cal(dataset):
    if (dataset == 'LASTFM'):
        gnn_dim_edge = 128
        gnn_dim_node = 128
    elif (dataset == 'TALK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (dataset == 'STACK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (dataset == 'GDELT'):
        gnn_dim_edge = 182 #TODO 为什么下载下来的数据集的edge feat是182呢？
        gnn_dim_node = 413
    elif (dataset == 'BITCOIN'):
        gnn_dim_edge = 172
        gnn_dim_node = 172


    # def cal(dataset):
    src_path = f'/raid/guorui/DG/dataset/{dataset}/df-src.bin'
    dst_path = f'/raid/guorui/DG/dataset/{dataset}/df-dst.bin'
    time_path = f'/raid/guorui/DG/dataset/{dataset}/df-time.bin'

    src = loadBin(src_path).cuda()
    dst = loadBin(dst_path).cuda()
    times = loadBin(time_path).cuda()

    step = src.shape[0] // 1000
    res_x = []
    res_y = []
    for i in range(1, src.shape[0] // step):
        right = i * step
        cur_src = src[:right]
        cur_dst = dst[:right]
        cur_times = times[:right]

        node_num = torch.unique(torch.cat((cur_src, cur_dst))).shape[0]
        edge_num = cur_src.shape[0]

        cur_y = (node_num * gnn_dim_node + edge_num * gnn_dim_edge) * 4 / 1024 ** 3
        cur_x = times[right] - times[0]

        res_x.append(cur_x.item())
        res_y.append(cur_y)

    # res_x = np.array(res_x)
    # res_y = np.array(res_y)
    res = {}
    res['dataset'] = dataset
    res['res_x'] = res_x
    res['res_y'] = res_y

    # 指定要保存的JSON文件路径
    file_path = f'/home/guorui/workspace/dgnn/{dataset}.json'

    # 将字典保存为JSON文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(res, json_file, ensure_ascii=False, indent=4)

cal('GDELT')
asd= 1

