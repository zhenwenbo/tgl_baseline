import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/b-tgl/config/TGAT-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"设置随机种子{seed}")

set_seed(42)

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)


if (args.data == 'BITCOIN'):
    train_param['epoch'] = 1

if (args.data in ['BITCOIN']):
    train_edge_end = 86063713
    val_edge_end = 110653345
else:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

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

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
    
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
# model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
# print(model.memory_updater.updater.weight_ih)
# mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None

if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
        left = val_edge_end
        right = len(df)
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for batch_num, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            if (mode == 'test'):
                cur_num =  len(rows) * neg_samples
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_test(left, left+cur_num)]).astype(np.int32)
                left += cur_num
            else:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
                
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            pos_loss = creterion(pred_pos, torch.ones_like(pred_pos)).item()
            neg_loss = creterion(pred_neg, torch.zeros_like(pred_neg)).item()
            if (batch_num % 100 == 0):
                print(f"正边loss:{pos_loss:.10f} 负边loss:{neg_loss:.10f}")
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
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

total_val_res = []
total_test_res = []
test_epo = 3

for i in range(test_epo):
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()


    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

    val_ap, test_ap = [], []
    for e in range(train_param['epoch']):
        print('Epoch {:d}:'.format(e))
        time_sample = 0
        time_prep = 0
        time_tot = 0
        total_loss = 0
        time_per_batch = 0


        
        time_total_prep = 0
        time_total_strategy = 0
        time_total_compute = 0
        time_total_update = 0
        time_total_epoch = 0
        time_total_epoch_s = time.time()

        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
        
        # ap, auc = eval('val')
        for batch_num, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
            # if (batch_num == 100):
            #     exit(-1)
            t_tot_s = time.time()
            time_total_prep_s = time.time()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
            
            pos_root_end = root_nodes.shape[0] * 2 // 3
            # saveBin(torch.from_numpy(root_nodes), '/home/guorui/workspace/dgnn/b-tgl/test/test-acc/tgl-node.bin')

            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            # saveBin(torch.from_numpy(ts), '/home/guorui/workspace/dgnn/b-tgl/test/test-acc/tgl-ts.bin')
        
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    # print(root_nodes)
                    # print(ts)
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                time_sample += ret[0].sample_time()

            if (batch_num % 1000 == 0):
                print(f"平均每个batch用时{time_per_batch / 1000:.5f}s, 预计epoch时间: {(time_per_batch / 1000 * (train_edge_end/train_param['batch_size'])):.3f}s")
                print(f"prep:{time_total_prep:.4f}s strategy: {time_total_strategy:.4f}s compute: {time_total_compute:.4f}s update: {time_total_update:.4f}s epoch: {time_total_epoch:.4f}s")

                time_per_batch = 0

            t_prep_s = time.time()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)

            # print(f"root_nodes: {root_nodes}, ts: {ts}")
            # print(f"node num: {mfgs[0][0].num_nodes()} edge num: {mfgs[0][0].num_edges()}")
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            
            # print(mfgs[0][0].edata)
            # print(mfgs[0][0].ndata)

            time_prep += time.time() - t_prep_s
            time_total_prep += time.time() - time_total_prep_s

            time_total_compute_s = time.time()
            optimizer.zero_grad()
            # if (batch_num == 30):
            #     print(mfgs[0][0].srcdata['ID'])
            #     print(mfgs[0][0].ndata['mem_input']['_N'])
            #     print(mfgs[0][0].ndata['mem']['_N'])
            #     print(mfgs[0][0].edata['f'])
            #     print(mfgs[0][0].ndata['ID']['_N'])
            #     mfgs1 = torch.load('/home/guorui/workspace/dgnn/b-tgl/mfgs')
            #     mailbox_b = torch.load('/home/guorui/workspace/dgnn/b-tgl/mailbox_b')
            #     ds = torch.nonzero(mailbox_b.cpu().reshape(1980, -1) != mailbox.mailbox.reshape(1980, -1))
            #     print(ds.reshape(-1)[:1000])
            #     print(f"mailbox: {torch.sum(mailbox_b.cpu().reshape(-1,300) != mailbox.mailbox.reshape(-1,300))}")
            #     b = mfgs[0][0]
            #     c = mfgs1[0][0]
            #     print(f"feat: {torch.sum(b.edata['f'] != c.edata['f'])} mem: {torch.sum(b.ndata['mem']['_N'] != c.ndata['mem']['_N'])}")
            #     for key in b.srcdata:
            #         print(f"{key}: {torch.sum(b.srcdata[key] != c.srcdata[key])}")
            # print(model.memory_updater.updater.weight_ih)
            pred_pos, pred_neg = model(mfgs)
            # model_structure(model)
            # print(pred_pos)
            # print(pred_neg)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            if (batch_num % 100 == 0):
                # print(root_nodes)
                print(f"loss: {loss.item()}")
            total_loss += float(loss) * train_param['batch_size']
            loss.backward()
            optimizer.step()
            time_total_compute += time.time() - time_total_compute_s

            t_prep_s = time.time()
            time_total_update_s = time.time()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                # if (batch_num == 29):
                #     print(eid)
                #     print(mem_edge_feats)
                block = None
                # print(model.memory_updater.last_updated_nid)
                # print(model.memory_updater.last_updated_memory)
                # print(root_nodes)
                # print(ts)
                if memory_param['deliver_to'] == 'neighbors':
                    blocks = to_dgl_blocks(ret, sample_param['history'], reverse=True)
                    block = blocks[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
            time_prep += time.time() - t_prep_s
            time_total_update += time.time() - time_total_update_s

            time_tot += time.time() - t_tot_s
            time_per_batch += time.time() - t_tot_s

        time_total_epoch += time.time() - time_total_epoch_s
        time_total_other = time_total_epoch - time_total_prep - time_total_strategy - time_total_compute - time_total_update
        print(f"prep:{time_total_prep:.4f}s strategy: {time_total_strategy:.4f}s compute: {time_total_compute:.4f}s update: {time_total_update:.4f}s epoch: {time_total_epoch:.4f}s other: {time_total_other:.4f}s")
        print(f"prep:{time_total_prep/time_total_epoch*100:.2f}% strategy: {time_total_strategy/time_total_epoch*100:.2f}% compute: {time_total_compute/time_total_epoch*100:.2f}% update: {time_total_update/time_total_epoch*100:.2f}% epoch: {time_total_epoch/time_total_epoch*100:.2f}% other: {time_total_other/time_total_epoch*100:.2f}%")

        args.model_eval = True
        # ap, auc = eval('val')
        # val_ap.append(f'{ap:.6f}')
        
        test_per_epoch = True
        if (test_per_epoch):
            if (args.model_eval):
                model.eval()

                if sampler is not None:
                    sampler.reset()
                if mailbox is not None:
                    mailbox.reset()
                    model.memory_updater.last_updated_nid = None
                    eval('train')
                    eval('val')
                ap, auc = eval('test')
                # if args.eval_neg_samples > 1:
                #     print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
                # else:
                #     print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
                test_ap.append(f'{ap:.6f}')
        print(f'val: {val_ap}; test: {test_ap}')

    total_val_res.append(val_ap)
    total_test_res.append(test_ap)

    print(total_val_res)
    print(total_test_res)

print(f"total_val_res: {total_val_res}")
print(f"total_test_res: {total_test_res}")