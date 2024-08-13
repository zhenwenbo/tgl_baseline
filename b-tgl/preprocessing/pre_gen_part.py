import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='GDELT')
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
parser.add_argument('--pre_sample_size', type=int, default=600000, help='pre sample size')
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


def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds


def eval(mode='val'):
    
    if (emb_buffer):
        emb_buffer.cur_mode = 'val'
    if (feat_buffer):
        feat_buffer.mode = 'val'

    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            
            if (use_gpu_sample):
                root_nodes = torch.from_numpy(root_nodes).cuda()
                mask_nodes = torch.from_numpy(np.concatenate([rows.dst.values, rows.src.values]).astype(np.int32)).cuda()
                mask_nodes = torch.cat((mask_nodes, (torch.zeros(rows.src.values.shape[0], dtype = torch.int32, device = 'cuda:0') - 1)))
                root_ts = torch.from_numpy(ts).cuda()
                ret = sampler_gpu.sample_layer(root_nodes, root_ts)
            else:
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = len(rows) * 2
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()

            if (use_gpu_sample):
                mfgs = sampler_gpu.gen_mfgs(ret)
                root_nodes = root_nodes.cpu().numpy()
            else:
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'])
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts)  
                    
            mfgs = prepare_input(mfgs, feat_buffer = feat_buffer, combine_first=combine_first)
            
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = torch.from_numpy(rows['Unnamed: 0'].values).cuda().to(torch.int32)
                mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

def count_judge(src_node, dst_node):
    cur_node = torch.empty(0).to(torch.int32)
    maxLen = 0
    countLen = 0
    for i,node in enumerate(src_node):
        if (torch.nonzero(cur_node == node).shape[0] > 0):
            continue
        cur_node = torch.cat((cur_node, torch.tensor([node], dtype = torch.int32)), dim = 0)

        #判断node在后面出现的次数    
        # print(i)
        #判断src_node中值等于node的个数，要求索引大于i
        indices = torch.nonzero(src_node[i + 1:] == node).reshape(-1)
        len1 = src_node[i+1:][indices].shape[0]

        indices = torch.nonzero(dst_node[i + 1:] == node).reshape(-1)
        len2 = dst_node[i+1:][indices].shape[0]

        len = len1 + len2
        countLen += len
        maxLen = max(maxLen, len)

        # if (len1 == 0 and len2 == 0):
        #     asdasd = 0
        #     print(f"node: {node}在后面从未出现过...")
        # else:
        #     print(f"node: {node}在src_node后面总共出现了{len1}次,在dst_node后面共出现了{len2}次")

    print(f"出现的最长的依赖长度为{maxLen},依赖链总长度为{countLen}")

# set_seed(0)

if __name__ == '__main__':

    node_feats, edge_feats = None,None
    if (not args.use_ayscn_prefetch):
        node_feats, edge_feats = load_feat(args.data)
    
    g, df = load_graph(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    if args.use_inductive:
        inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
        df = df.iloc[inductive_inds]
        
    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
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


    feat_buffer = Feat_buffer(args.data, g, df, train_param, memory_param, train_edge_end, args.pre_sample_size//train_param['batch_size'],\
                              sampler_gpu,neg_link_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge))
    if (not use_ayscn_prefetch):
        feat_buffer.init_feat(node_feats, edge_feats)
    # feat_buffer.gen_part_incre()
    feat_buffer.gen_part()


    print(f"共用时{time.time() - total_start:.4f}s")