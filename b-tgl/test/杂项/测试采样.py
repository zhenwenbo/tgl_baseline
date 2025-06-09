


# 测试sample结果
# GPU采样与TGL采样



import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler.sampler import *
from sampler.sampler_gpu import *




def cal(batch_size, d):
    # df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))
    # g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
    g, df = load_graph(d)

    if (d in ['BITCOIN']):
        train_edge_end = 86063713
        val_edge_end = 110653345
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]

        # group_indexes = np.array(df[:train_edge_end].index // batch_size)



    fan_nums = [10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)

    sample_param, memory_param, gnn_param, train_param = parse_config('/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
    sample_param['layer'] = len(fan_nums)
    sample_param['neighbor'] = fan_nums

    num_nodes = (g['indptr'].shape[0] - 1)
    neg_link_sampler = ReNegLinkSampler(num_nodes, 0)

    max_edge_num = 0
    max_node_num = 0
    left = 0
    right = 0
    cur_batch = 0

    total_edge_num, total_node_num = 0, 0
    while True:
        
        
        if (right == train_edge_end):
            break
        right += batch_size
        right = min(train_edge_end, right)
        rows = df[left:right]
        # break

        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        root_ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        root_nodes = torch.from_numpy(root_nodes).cuda()
        root_ts = torch.from_numpy(root_ts).cuda()
        start = time.time()

        sample_start = time.time()
        ret_list = sampler_gpu.sample_layer(root_nodes, root_ts)
        # break


        # sample_time = time.time() - sample_start

        # gen_start = time.time()
        # for ret in ret_list:
        #     src,dst,outts,outeid,root_nodes,root_ts = ret
        #     mask = src > -1
        #     src = src[mask]
        #     dst = dst[mask]
        #     outeid = outeid[mask]
        #     outts = outts[mask]
        # # mfgs = sampler_gpu.gen_mfgs(ret_list)
        # gen_time = time.time() - gen_start




        src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[0]
        eid,_ = torch.sort(torch.unique(outeid))
        cur_edge_num = eid.shape[0]
        cur_node_num = torch.unique(torch.cat((root_nodes, src[src>-1]))).shape[0]

        total_edge_num += cur_edge_num
        total_node_num += cur_node_num
        # if (cur_edge_num >max_edge_num):
        #     print(f"batch{cur_batch}时 边数新高为{cur_edge_num}")
        #     max_edge_num = cur_edge_num
        # if (cur_node_num > max_node_num):
        #     print(f"batch{cur_batch}时 节点数新高为{cur_node_num}")
        #     max_node_num = cur_node_num
        # print(f"共访问edge个数: {total_edge_num}, node个数: {total_node_num}")
        
        left = right
        # # if (cur_batch == 3):
        # #     break
        cur_batch += 1
        # print(f"采样花销: {sample_time:.7f}s gen block花销{gen_time:.7f}s")
    
    return total_edge_num, total_node_num


datas = ['LASTFM', 'TALK', 'STACK', 'BITCOIN', 'GDELT']
sizes = [2000, 4000, 6000, 10000,20000,40000,60000,100000,200000,400000,600000]
res = {}
for d in datas:
    res[d] = {}
    for batch_size in sizes:
        cur_en, cur_nn = cal(batch_size, d)
        res[d][batch_size] = [cur_en, cur_nn]
        print(res)


data = {'LASTFM': {2000: [5438357, 771466], 4000: [3432273, 427311], 6000: [2625548, 293863], 10000: [1949033, 179707], 20000: [1430104, 91048], 40000: [1162988, 45540], 60000: [1082747, 31647], 100000: [1013142, 19767], 200000: [953631, 9900], 400000: [930736, 5940], 600000: [918276, 3960]}, 'TALK': {2000: [27858870, 16380956], 4000: [23916553, 13963237], 6000: [21659974, 12707827], 10000: [18929645, 11303676], 20000: [15673350, 9747216], 40000: [13053956, 8554671], 60000: [11813101, 7988124], 100000: [10449361, 7345235], 200000: [9021475, 6550881], 400000: [7875639, 5670276], 600000: [7356250, 5074755]}, 'STACK': {2000: [393518181, 258656185], 4000: [352774714, 221913986], 6000: [330969180, 202084055], 10000: [305154121, 178778897], 20000: [271993087, 149972690], 40000: [239423723, 124285130], 60000: [221009060, 110875366], 100000: [199581613, 95937367], 200000: [173036700, 78674227], 400000: [148424307, 64145532], 600000: [134609372, 56558530]}, 'BITCOIN': {2000: [243440867, 287579561], 4000: [227825901, 271978172], 6000: [219646687, 263126732], 10000: [209457518, 251840710], 20000: [195222378, 236395015], 40000: [180365008, 221242833], 60000: [171730400, 212758882], 100000: [161348216, 202590273], 200000: [147417236, 189021676], 400000: [134666614, 176006578], 600000: [128199857, 168715086]}, 'GDELT': {2000: [1111280992, 220361145], 4000: [964908349, 170340371], 6000: [875677508, 145678925], 10000: [754654209, 117324640], 20000: [574368081, 81571130], 40000: [395382480, 49872080], 60000: [308011275, 35005820], 100000: [227469684, 21413190], 200000: [164095246, 10726481], 400000: [132346913, 5371603], 600000: [121759652, 3586629]}}

res_data = {}
for d in data:
    res_data[d] = []
    base = data[d][600000][0] + data[d][600000][1]
    for s in data[d]:
        res_data[d].append((data[d][s][0] + data[d][s][1]) / base)

print(res_data)