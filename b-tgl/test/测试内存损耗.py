


# 测试实际运行过程中到底哪些东西在占用内存


import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
import argparse
import os
import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='TALK')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)

parser.add_argument('--train_conf', type=str, default='basic_conf', help='name of stored model')
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

from config.train_conf import *
GlobalConfig.conf = args.train_conf + '.json'
config = GlobalConfig()
args.use_ayscn_prefetch = config.use_ayscn_prefetch

if (sample_param['layer'] == 1):
    args.pre_sample_size = 600000
else:
    if (args.data == 'STACK'):
        args.pre_sample_size = 60000
    else:
        args.pre_sample_size = 600000
    

print(f"实际的block大小为: {args.pre_sample_size}")
# args.pre_sample_size = config.pre_sample_size
args.cut_zombie = config.cut_zombie


if (hasattr(config, 'model')):
    args.config = f'/raid/guorui/workspace/dgnn/b-tgl/config/{config.model}-{config.layer}.yml'

if (config.model_eval):
    args.model_eval = True
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print(f"训练配置: {config.config_data}")


global node_feats, edge_feats
node_feats, edge_feats = None,None
if (not args.use_ayscn_prefetch):
    node_feats, edge_feats = load_feat(args.data)

g, df = load_graph(args.data)

#TODO GDELT改fanout为[7,7]
# sample_param['neighbor'] = [7,7]
if (args.data in ['BITCOIN']):
    train_edge_end = 86063713
    val_edge_end = 110653345
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

if (args.use_ayscn_prefetch):
    if (args.data == 'LASTFM'):
        gnn_dim_edge = 100
        gnn_dim_node = 100
    elif (args.data == 'TALK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'STACK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'GDELT'):
        gnn_dim_edge = 182 #TODO 为什么下载下来的数据集的edge feat是182呢？
        gnn_dim_node = 413
    elif (args.data == 'BITCOIN'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'MAG'):
        gnn_dim_edge = 0
        gnn_dim_node = 768
    elif (args.data == 'WIKI'):
        gnn_dim_edge = 0
        gnn_dim_node = 0
    else:
        raise RuntimeError("have not this dataset config!")


combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True


from sampler.sampler_gpu import *
use_gpu_sample = False
use_gpu_sample = True
no_neg = 'no_neg' in sample_param and sample_param['no_neg']
from emb_buffer import *


if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values) #TODO 这里写错了吧
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

#TODO presample_batch = 100
emb_buffer = None
if (not args.no_emb_buffer):
    emb_buffer = Embedding_buffer(g, df, train_param, train_edge_end, 100, args.dis_threshold, sample_param['neighbor'], gnn_param, neg_link_sampler)
# no_neg = True
sampler_gpu = Sampler_GPU(g, sample_param['neighbor'], sample_param['layer'], emb_buffer)
edge_num = g['indptr'].shape[0] - 1