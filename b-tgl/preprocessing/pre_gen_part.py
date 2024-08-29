import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='TALK')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/exp/scripts/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)

parser.add_argument('--only_gen_part', action='store_true', default=False)
parser.add_argument('--use_ayscn_prefetch', action='store_true', default=False)
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--pre_sample_size', type=int, default=60000, help='pre sample size')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
import sys
import os
total_start = time.time()
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.train_conf import *
GlobalConfig.conf = 'basic_conf.json'


from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':

    node_feats, edge_feats = None,None
    if (not args.use_ayscn_prefetch):
        node_feats, edge_feats = load_feat(args.data)
    
    g, df = load_graph(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

    if (args.data in ['BITCOIN']):
        train_edge_end = 86063713
        val_edge_end = 110653345
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]

        
    gnn_dim_node = 0 if (node_feats.shape[0] == 0) else node_feats.shape[1]
    gnn_dim_edge = 0 if (edge_feats is None or edge_feats.shape[0] == 0) else edge_feats.shape[1]

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


    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method("spawn")
    from pre_fetch import *
    use_ayscn_prefetch = args.use_ayscn_prefetch

    parent_conn = None
    prefetch_conn = None
    if (use_ayscn_prefetch):
        parent_conn, child_conn = multiprocessing.Pipe()
        prefetch_conn, prefetch_child_conn = multiprocessing.Pipe()

        p = multiprocessing.Process(target=prefetch_worker, args=(child_conn, prefetch_child_conn))
        p.start()

        parent_conn.send(('init_feats', (args.data, )))
        print(f"Sent: {'init_feats'}")
        result = parent_conn.recv()
        print(f"Received: {result}")
        node_feats,edge_feats = 1,1
    # 主进程发送要调用的函数名和args变量

    #prefetch_others_conn处理其他的index这类串行操作
    #prefetch_conn 单独处理prefetch这个并行操作
    #这么做是为了防止prefetch的结果被其他的串行操作截胡了
    prefetch_only_conn = prefetch_conn
    prefetch_conn = parent_conn
    


    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, emb_buffer, combined=combine_first).cuda()

    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge, prefetch_conn=prefetch_conn) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])



    # parent_conn.send("EXIT")
    # p.join()

    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()

    sampler = None
    # if (True):
    if not (('no_sample' in sample_param and sample_param['no_sample']) or (use_gpu_sample)):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))


    if not os.path.isdir('models'):
        os.mkdir('models')
    if args.model_name == '':
        path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
    else:
        path_saver = 'models/{}.pkl'.format(args.model_name)
    best_ap = 0
    best_e = 0
    val_losses = list()
    group_indexes = list()
    group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))


    if 'reorder' in train_param:
        # random chunk shceduling
        reorder = train_param['reorder']
        group_idx = list()
        for i in range(reorder):
            group_idx += list(range(0 - i, reorder - i))
        group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
        group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
        group_indexes.append(group_indexes[0] + group_idx)
        base_idx = group_indexes[0]
        for i in range(1, train_param['reorder']):
            additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
            group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])



    from feat_buffer import *
    # gpu_sampler = Sampler_GPU(g, 10)


    feat_buffer = Feat_buffer(args.data, df, None, train_param, memory_param, train_edge_end, args.pre_sample_size//train_param['batch_size'],\
                              sampler_gpu,neg_link_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge))
    if (not use_ayscn_prefetch):
        feat_buffer.init_feat(node_feats, edge_feats)
    # feat_buffer.gen_part_incre()
    feat_buffer.gen_part()


    print(f"共用时{time.time() - total_start:.4f}s")