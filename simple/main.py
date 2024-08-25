import argparse
import os
import sys
MODULE_PATH = os.path.abspath("./SIMPLE")
if MODULE_PATH not in sys.path:
	sys.path.append(MODULE_PATH)

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
parser.add_argument('--config', type=str, help='path to config file', default = '/raid/guorui/workspace/dgnn/simple/config/TGAT-2.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--rand_edge_features', type=int, default=100, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=100, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--threshold',type=float, default=0.01, help='placement budget')
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from SIMPLE.trainer import *
from SIMPLE.data_processing import *
from SIMPLE.sampler import *
from sklearn.metrics import average_precision_score, roc_auc_score
from SIMPLE.memory_module import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)    
g, df = load_graph(args.data)
num_node = g['indptr'].shape[0] - 1
num_edge = len(df)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
interval_to_gpu = train_param['interval_to_gpu']
pre_load = train_param['pre_load']
emb_reuse = True if 'emb_reuse' in gnn_param and gnn_param['emb_reuse'] else False

#load intervals budget[0] --> node; budget[1] --> edge.
device = 'cuda:0' #already set using os.environment.

multi_layer = True if gnn_param['layer'] > 1 else False
mailbox_flag = True if memory_param['type'] != 'none' else False
    
budget = load_budget(args.data, memory_param['mailbox_size'], args.threshold, multi_layer, mailbox_flag, emb_reuse)
node_start, node_end, node_IDs, edge_start, edge_end, edge_IDs = \
load_total_intervals(args.data, budget, num_node, num_edge, memory_param['mailbox_size'], args.threshold, multi_layer, mailbox_flag, emb_reuse)

if interval_to_gpu:
    if node_start is not None:
        node_start, node_end, node_IDs = \
        intervals_to_gpu(node_start, node_end, node_IDs, device)
    if edge_start is not None:
        edge_start, edge_end, edge_IDs = \
        intervals_to_gpu(edge_start, edge_end, edge_IDs, device)
#此处将interval、budget都读取出来了，nodefeat、edgefeat存到了cpu
nfeat_flag = True if (gnn_param['arch'] != 'identity' and node_start is not None) else False
        
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, num_node, combined=combine_first, reuse=emb_reuse).cuda()
mailbox = MailBox(memory_param, num_node, gnn_dim_edge) if memory_param['type'] != 'none' else None
#此处model和tgl的差别只在于是否重用emb，即Orca的差别，实际上节点特征的缓存不在model中进行
    
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

#节点不缓存就是直接全部存储
if node_start is None:
    node_feats = node_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

# TODO 节点需要缓存的时候处理方式
#pre-allocate space for buffs.    
if mailbox is not None and node_start is not None:
    mailbox.ts_to_gpu() #pre_load ts.
    mailbox.allocate_mailbox_buffs(budget, memory_param)
    
nfeat_buffs, efeat_buffs = allocate_buffs(budget, node_feats, edge_feats, num_node, num_edge, device, nfeat_flag)
#初始化buff，此处就是简单的[budget, dim]的buff空间开辟


n_flag = (nfeat_buffs is not None and node_feats is not None) or (mailbox is not None and mailbox.mailbox_buffs is not None)
#flag表示是否通过缓存获取
gpu_flag_e, gpu_flag_n, gpu_map_e, gpu_map_n, map_curr_e, map_curr_n = \
init_flags_and_maps(num_node, num_edge, budget, device)
#初始化需要缓存的flag和map flag应该是标记是否被缓存，map用来映射

node_last_batch, node_left_index, node_stop = None, None, None
edge_last_batch, edge_left_index, edge_stop = None, None, None


sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

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
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            pos_root_end = len(rows) * 2
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            
            mfgs = prepare_input_eval(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails_eval(mfgs[0])
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
                mem_edge_feats = edge_feats[rows['Unnamed: 0'].values].cuda()
                root_nodes_gpu = torch.from_numpy(root_nodes[:pos_root_end]).cuda()
                ts_gpu = torch.from_numpy(ts).cuda()
                block = None
                mailbox.update_mailbox_eval(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory_eval(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
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
group_indexes = np.array(df[:train_edge_end].index // train_param['batch_size'])
final_batch = group_indexes[-1]

if pre_load:
    if node_start is not None:
        batch_plan_node = \
        pre_load_all(node_start, node_end, node_IDs, final_batch+1, interval_to_gpu)
        del node_start, node_end, node_IDs
    if edge_start is not None:
        batch_plan_edge = \
        pre_load_all(edge_start, edge_end, edge_IDs, final_batch+1, interval_to_gpu)
        del edge_start, edge_end, edge_IDs
#获得batch_plan_node即每个batch后缓存中应当持有的节点的ID，如果interval可以存到gpu直接用gpu加速即可，如果存不进去就需要一个优化算法。
        
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_mfgs = 0
    time_gen_flags = 0
    time_load_data = 0
    time_gen_plan = 0
    time_up_indicators = 0
    time_up_buffs = 0
    time_up_mail = 0
    time_prep = 0
    time_strategy = 0
    time_tot = 0

    #1. prep (准备阶段, 包括生成采样子图、特征抽取注入、memory抽取注入)
    #2. strategy (策略维护, 只有simple有这个)
    #2. compute (训练阶段，包括前向传播和反向传播)
    #3. update (更新阶段，包括更新memory与mailbox)
    time_total_prep = 0
    time_total_strategy = 0
    time_total_compute = 0
    time_total_update = 0
    time_total_epoch = 0
    time_total_epoch_s = time.time()

    time_model = 0
    time_loss = 0
    total_loss = 0
    # training
    model.train()
    #initialization
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    if n_flag:
        reset_indicators(gpu_flag_n, gpu_map_n)
        if nfeat_buffs is not None:
            reset_buffs(nfeat_buffs)
        if not pre_load:
            node_last_batch, node_left_index, node_stop, plan_node = init_batch_plan_fetch(node_start, device)
        else:
            plan_node = torch.tensor([], dtype=torch.long, device=device)
    if efeat_buffs is not None:
        reset_indicators(gpu_flag_e, gpu_map_e)
        reset_buffs(efeat_buffs)
        if not pre_load:
            edge_last_batch, edge_left_index, edge_stop, plan_edge = init_batch_plan_fetch(edge_start, device)
        else:
            plan_edge = torch.tensor([], dtype=torch.long, device=device)
    batch_id = 0
    node_gpu_mask, node_gpu_local_ids, node_cpu_ids = None, None, None
    edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids = None, None, None
    #iterate over batches

    time_per_batch = 0
    aver_node_num, aver_edge_num, aver_node_hit, aver_edge_hit = 0,0,0,0
    for batch_num, rows in df[:train_edge_end].groupby(group_indexes):  
        t_tot_s = time.time()
        #load target data.

        if (batch_num % 1000 == 0):
            print(f"平均每个batch用时{time_per_batch / 1000:.5f}s, 预计epoch时间: {(time_per_batch / 1000 * (train_edge_end/train_param['batch_size'])):.3f}s")
            time_per_batch = 0

        time_total_prep_s = time.time()
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        pos_root_end = root_nodes.shape[0] * 2 // 3
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        t0 = time.time()
        #create MFGs.
        if gnn_param['arch'] != 'identity':
            if not emb_reuse:
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = to_dgl_blocks_orca(ret)
        node_num = mfgs[0][0].num_nodes()
        edge_num = mfgs[0][0].num_edges()
        # aver_node_num = (aver_node_num * batch_num + node_num)
        # print(f"node num: {mfgs[0][0].num_nodes()} edge num: {mfgs[0][0].num_edges()}")

        t1 = time.time()    
        node_idx = mfgs[0][0].srcdata['ID'].long() #拿到第0层的idx，采样特性中，第0层的数据包含后面所有层的数据
        if gnn_param['arch'] != 'identity':
            if gnn_param['layer'] > 1:
                if not emb_reuse:
                    edge_idx = (mfgs[0][0].edata['ID'].long(), mfgs[1][0].edata['ID'].long())
                else:
                    edge_idx = mfgs[0][0].edata['ID'].long() #拿到第0层的idx，采样特性中，第0层的数据包含后面所有层的数据
            else:
                edge_idx = mfgs[0][0].edata['ID'].long() 
        else:
            edge_idx = []
           
        node_hit = 0
        edge_hit = 0
        strategy_s = time.time()
        if  nfeat_buffs is not None or (mailbox is not None and mailbox.mailbox_buffs):
            node_gpu_mask, node_gpu_local_ids, node_cpu_ids = \
            gen_flag_and_mask(node_idx, gpu_flag_n, gpu_map_n, plan_node)

            node_hit = torch.sum(node_gpu_mask).item() / node_gpu_mask.shape[0] * 100
        if efeat_buffs is not None:
            if gnn_param['layer'] > 1:
                if not emb_reuse:
                    edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids = \
                    gen_flag_and_mask_2layer(edge_idx, gpu_flag_e, gpu_map_e, plan_edge)
                    #TODO 这里对edge处理时将两层的分开取了，但是graphSage的采样方式而言，第0层应该包含所有边的。
                else:
                    edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids = \
                    gen_flag_and_mask(edge_idx, gpu_flag_e, gpu_map_e, plan_edge) 
                
                mask = edge_gpu_mask[0]
            else:
                edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids = \
                gen_flag_and_mask(edge_idx, gpu_flag_e, gpu_map_e, plan_edge) 

                mask = edge_gpu_mask
            
            
            edge_hit = (torch.sum(mask) / mask.shape[0] * 100).item()
        # print(f"node缓存命中率: {torch.sum(node_gpu_mask).item() / node_gpu_mask.shape[0] * 100:.2f}%, edge 缓存命中率: {torch.sum(edge_gpu_mask) / edge_gpu_mask.shape[0] * 100:.2f}%")
        time_strategy += time.time() - strategy_s
        

        aver_node_hit = (aver_node_hit * batch_num + node_hit) / (batch_num + 1)
        aver_edge_hit = (aver_edge_hit * batch_num + edge_hit) / (batch_num + 1)
        t2 = time.time()
        #load batch data.
        if gnn_param['arch'] != 'identity':
            if gnn_param['layer'] > 1:
                if not emb_reuse:
                    mfgs = load_batched_data_2layer(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs, efeat_buffs)
                else:
                    mfgs = load_batched_data_orca(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs, efeat_buffs)
            else:
                mfgs = load_batched_data(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs, efeat_buffs)
        
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0], node_idx, node_gpu_mask, node_gpu_local_ids, node_cpu_ids)
        t3 = time.time()
        time_total_prep += time.time() - time_total_prep_s

        #update buffs. 1. get batch plan. 2. update indicators. 3. update buffs.
        strategy_s = time.time()
        if n_flag and batch_id < final_batch:
            ta = time.time()
            if pre_load:
                plan_node_new = batch_plan_node[batch_id].cuda()
            else:
                if interval_to_gpu:
                    plan_node_new = gen_batch_plan_tensor(node_start, node_end, node_IDs, batch_id)
                else:
                    plan_node_new, node_last_batch, node_left_index = gen_batch_plan(node_start, node_end, node_IDs, node_last_batch, node_left_index, batch_id, node_stop)
                    plan_node_new = plan_to_gpu(plan_node_new, device)
            tb = time.time()
            gpu_flag_n, gpu_map_n, local_ID_I_n, local_ID_II_n = \
            update_indicators(node_idx, plan_node, plan_node_new, map_curr_n, gpu_flag_n, gpu_map_n, num_node, device)
            tc = time.time()
            if nfeat_buffs is not None:
                nfeat_buffs = update_buffs(nfeat_buffs, mfgs[0][0].srcdata['h'], local_ID_I_n, local_ID_II_n)
            plan_node = plan_node_new
            td = time.time()
            time_gen_plan += tb-ta
            time_up_indicators += tc-tb
            time_up_buffs += td-tc
        if edge_feats is not None and efeat_buffs is not None and batch_id < final_batch:
            ta = time.time()
            if (type(edge_idx) == tuple and len(edge_idx[0])!=0) or (type(edge_idx) != tuple and len(edge_idx)!=0):   
                if pre_load:
                    plan_edge_new = batch_plan_edge[batch_id].cuda()
                else:
                    if interval_to_gpu:
                        plan_edge_new = gen_batch_plan_tensor(edge_start, edge_end, edge_IDs, batch_id)
                    else:
                        plan_edge_new, edge_last_batch, edge_left_index = gen_batch_plan(edge_start, edge_end, edge_IDs, edge_last_batch, edge_left_index, batch_id, edge_stop)
                        plan_edge_new = plan_to_gpu(plan_edge_new, device)
                tb = time.time()
                if gnn_param['layer'] > 1:
                    if not emb_reuse:
                        gpu_flag_e, gpu_map_e, local_ID_I_e, local_ID_II_e, local_ID_III_e = update_indicators_2layer(edge_idx, plan_edge, plan_edge_new, map_curr_e, gpu_flag_e, gpu_map_e, num_edge, device)
                        tc = time.time()
                        efeat_buffs = update_buffs_2layer(efeat_buffs, mfgs[0][0].edata['f'], mfgs[1][0].edata['f'], local_ID_I_e, local_ID_II_e, local_ID_III_e)
                    else:
                        gpu_flag_e, gpu_map_e, local_ID_I_e, local_ID_II_e = \
            update_indicators(edge_idx, plan_edge, plan_edge_new, map_curr_e, gpu_flag_e, gpu_map_e, num_edge, device)
                        tc = time.time()
                        efeat_buffs = update_buffs(efeat_buffs, mfgs[0][0].edata['f'], local_ID_I_e, local_ID_II_e)
                else:
                    gpu_flag_e, gpu_map_e, local_ID_I_e, local_ID_II_e = \
                update_indicators(edge_idx, plan_edge, plan_edge_new, map_curr_e, gpu_flag_e, gpu_map_e, num_edge, device)
                    tc = time.time()
                    efeat_buffs = update_buffs(efeat_buffs, mfgs[0][0].edata['f'], local_ID_I_e, local_ID_II_e)
                plan_edge = plan_edge_new
                td = time.time()
                time_gen_plan += tb-ta
                time_up_indicators += tc-tb
                time_up_buffs += td-tc
        if mailbox is not None and mailbox.mailbox_buffs is not None and batch_id < final_batch:
            mailbox.update_mailbox_buffs(mfgs[0][0].srcdata['mem_input'], mfgs[0][0].srcdata['mem'], local_ID_I_n, local_ID_II_n, memory_param['mailbox_size'])
        time_strategy += time.time() - strategy_s
        time_total_strategy += time.time() - strategy_s

        t_prep_s = time.time()
        time_sample += t0 - t_tot_s
        time_prep += t_prep_s - t_tot_s
        time_mfgs += t1-t0
        time_gen_flags += t2 - t1
        time_load_data += t3 - t2
        
        batch_id += 1
        time_total_compute_s = time.time()
        optimizer.zero_grad()

        time_model_s = time.time()
        pred_pos, pred_neg = model(mfgs)
        time_model += time.time() - time_model_s

        # model_structure(model)

        time_loss_s = time.time()
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param['batch_size']
        loss.backward()
        optimizer.step()
        time_loss += time.time() - time_loss_s

        t_prep_s = time.time()

        time_total_compute += time.time() - time_total_compute_s
        time_total_update_s = time.time()
        if mailbox is not None:
            mem_edge_feats = edge_feats[rows['Unnamed: 0'].values].cuda()
            root_nodes_gpu = torch.from_numpy(root_nodes[:pos_root_end]).cuda()
            ts_gpu = torch.from_numpy(ts).cuda()
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                block = block.to(device)
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, block, gpu_flag_n, gpu_map_n)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, gpu_flag_n, gpu_map_n)
        t_end = time.time()
        time_prep += t_end  - t_prep_s
        time_tot += t_end - t_tot_s
        if mailbox is not None:
            time_up_mail += t_end - t_prep_s
        
        time_total_update += time.time() - time_total_update_s
        time_per_batch += t_end - t_tot_s
    
    print('\ttotal time:{:.2f}s prep time:{:.2f}s sample time:{:.2f}s mfgs time:{:.2f}s gen_flags time:{:.2f}s load_data time:{:.2f}s gen_plan time:{:.2f}s up_indicators time:{:.2f}s up_buffs time:{:.2f}s up_mail time:{:.2f}s'.format(time_tot, time_prep, time_sample, time_mfgs, time_gen_flags, time_load_data, time_gen_plan, time_up_indicators, time_up_buffs, time_up_mail))
    print(f"model time: {time_model}, loss time: {time_loss} aver_node_hit: {aver_node_hit:.2f}%, aver_edge_hit: {aver_edge_hit:.2f}% 策略开销: {time_strategy:.4f}s 注意prep time中包含策略开销 ")
    t0 = time.time()
    if gpu_flag_n is not None and mailbox is not None:
        mailbox.offload_for_eval(gpu_flag_n, gpu_map_n)
    
    time_total_epoch = time.time() - time_total_epoch_s
    time_total_other = time_total_epoch - time_total_prep - time_total_strategy - time_total_compute - time_total_update
    print(f"prep:{time_total_prep:.4f}s strategy: {time_total_strategy:.4f}s compute: {time_total_compute:.4f}s update: {time_total_update:.4f}s epoch: {time_total_epoch:.4f}s other: {time_total_other:.4f}s")
    print(f"prep:{time_total_prep/time_total_epoch*100:.2f}% strategy: {time_total_strategy/time_total_epoch*100:.2f}% compute: {time_total_compute/time_total_epoch*100:.2f}% update: {time_total_update/time_total_epoch*100:.2f}% epoch: {time_total_epoch/time_total_epoch*100:.2f}% other: {time_total_other/time_total_epoch*100:.2f}%")

    if (not args.model_eval):
        continue
    ap, auc = eval('val')
    t1 = time.time()
    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f} val time:{:.2f}s'.format(total_loss, ap, auc, t1-t0))

if (not args.model_eval):
    exit(-1)
    
print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
t0 = time.time()
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
    eval('train')
    eval('val')
ap, auc = eval('test')
t1 = time.time()
print('\ttest time:{:.2f}s'.format(t1-t0))
if args.eval_neg_samples > 1:
    print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
else:
    print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))