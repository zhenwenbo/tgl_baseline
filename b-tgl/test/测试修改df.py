
import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
import argparse
import numpy as np
from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='MAG')
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

dataset_conf = {}
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
dataset_conf['train_edge_end'] = train_edge_end.item()
dataset_conf['val_edge_end'] = val_edge_end.item()

src = torch.from_numpy(df.src.values.astype(np.int32))
dst = torch.from_numpy(df.dst.values.astype(np.int32))
eid = torch.from_numpy(df['Unnamed: 0'].values.astype(np.int32))
time = torch.from_numpy(df.time.values)

base_path = f'/raid/guorui/DG/dataset/{args.data}'
saveBin(src, f'{base_path}/df-src.bin')
saveBin(dst, f'{base_path}/df-dst.bin')
saveBin(eid, f'{base_path}/df-eid.bin')
saveBin(time, f'{base_path}/df-time.bin')
json_path = f'{base_path}/df-conf.json'

with open(json_path, 'w') as f:
    json.dump(dataset_conf, f, indent=4)