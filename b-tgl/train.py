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
from sampler.sampler_core import ParallelSampler, TemporalGraphBlock

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#TODO 在LASTFM下确实会影响时间, 但是在大数据集上的影响好像不大? 

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='TALK')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1-50.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)
parser.add_argument('--use_cpu_sample', action='store_true', default=False)

parser.add_argument('--reuse_ratio', type=float, default=0.9, help='reuse_ratio')
parser.add_argument('--train_conf', type=str, default='disk', help='name of stored model')
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--set_epoch', type=int, default=-1, help='distance threshold')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

from config.train_conf import *
GlobalConfig.conf = args.train_conf + '.json'
config = GlobalConfig()
args.use_async_prefetch = config.use_async_prefetch
args.use_async_IO = config.use_async_IO

args.pre_sample_size = 60000
if (sample_param['layer'] == 1):
    if (args.train_conf == 'disk'):
        # print(f"一跳disk全改为60w")
        args.pre_sample_size = 60000
else:
    if (args.data == 'STACK'):
        args.pre_sample_size = 60000
    else:
        args.pre_sample_size = 60000

if (config.pre_sample_size != -1):
    args.pre_sample_size = config.pre_sample_size

if (args.data == 'GDELT' and sample_param['layer'] == 2):
    sample_param['neighbor'] = [8, 8]
    train_param['epoch'] = 1
    print(f"GDELT二跳修改邻域为8,8")

if (args.data == 'BITCOIN'):
    train_param['epoch'] = 2

if (config.epoch != -1):
    train_param['epoch'] = config.epoch
    print(f"预设epoch为 {config.epoch}")

if (args.data in ['BITCOIN', 'STACK', 'GDELT'] and 'TGN' not in args.config):
    train_param['epoch'] = 1
    print(f"BITCOIN后面两个的disk只跑一个epoch")
train_param['epoch'] = 2
if (args.set_epoch != -1):
    train_param['epoch'] = args.set_epoch
print(sample_param)
print(train_param)

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


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"======================================================================")
    print(f"设置随机种子为{seed}")
    print(f"======================================================================")
set_seed(42)

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
        left = df_conf['train_edge_end']
        eval_df_end = val_edge_end
    elif mode == 'test':
        left = df_conf['val_edge_end']
        eval_df_end = datas['src'].shape[0]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        left = 0
        eval_df_end = df_conf['train_edge_end']
    
    right = left
    with torch.no_grad():
        total_loss = 0
        while True:
            right += batch_size
            right = min(right, eval_df_end)
            if (left >= right):
                break

            src = datas['src'][left: right]
            dst = datas['dst'][left: right]
            times = datas['time'][left: right]
            eid = datas['eid'][left: right]
            root_nodes = np.concatenate([src, dst, neg_link_sampler.sample(src.shape[0] * neg_samples)]).astype(np.int32)
            ts = np.tile(times, neg_samples + 2).astype(np.float32)
            
            if (use_gpu_sample):
                root_nodes = torch.from_numpy(root_nodes).cuda()
                root_ts = torch.from_numpy(ts).cuda()
                ret = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=args.cut_zombie)
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
                    
            mfgs = prepare_input(mfgs, node_feats, edge_feats, feat_buffer = feat_buffer, combine_first=combine_first)
            
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
                eid = eid.cuda().to(torch.int32)
                mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
            
            left = right

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

    try:
        global node_feats, edge_feats
        node_feats, edge_feats = None,None

        import torch.multiprocessing as multiprocessing
        multiprocessing.set_start_method("spawn")
        from pre_fetch import *
        from IO_load import *
        use_async_prefetch = args.use_async_prefetch

        parent_conn_IO = None
        if (args.use_async_IO):
            parent_conn_IO, child_conn_IO = multiprocessing.Pipe()
            prefetch_conn_IO, prefetch_child_conn_IO = multiprocessing.Pipe()

            p = multiprocessing.Process(target=prefetch_worker_IO, args=(child_conn_IO, prefetch_child_conn_IO))
            p.start()

            

        parent_conn = None
        prefetch_conn = None
        # if (use_async_prefetch):
        parent_conn, child_conn = multiprocessing.Pipe()
        prefetch_conn, prefetch_child_conn = multiprocessing.Pipe()

        p = multiprocessing.Process(target=prefetch_worker, args=(child_conn, prefetch_child_conn))
        p.start()

        parent_conn.send(('init_feats', (args.data, args.pre_sample_size )))
        print(f"Sent: {'init_feats'}")
        result = parent_conn.recv()
        print(f"Received: {result}")
        node_feats,edge_feats = 1,1

        if (args.use_async_IO):
            parent_conn.send(('init_IO_load', (parent_conn_IO,)))
            print(f"Sent: {'初始化pre_fetch中的IO_prefetch'}")
            result = parent_conn.recv()

        

        # multiprocessing.set_start_method("fork")


            
        # if (not args.use_async_prefetch):
        #     node_feats, edge_feats = load_feat(args.data)
        
        g, datas, df_conf = load_graph_bin(args.data)

        train_edge_end = df_conf['train_edge_end']
        val_edge_end = df_conf['val_edge_end']

        if args.use_inductive:
            inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
            df = df.iloc[inductive_inds]
            
        # gnn_dim_node = 0 if (node_feats is None or args.use_async_prefetch) else node_feats.shape[1]
        # gnn_dim_edge = 0 if (node_feats is None or args.use_async_prefetch) else edge_feats.shape[1]
        # gnn_dim_node = 0
        # gnn_dim_edge = 0

        if (args.data == 'LASTFM'):
            gnn_dim_edge = 128
            gnn_dim_node = 128
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
        use_gpu_sample = not args.use_cpu_sample

        print(f"====================================\n使用CPU采样? {not use_gpu_sample}")
        # use_gpu_sample = True
        no_neg = 'no_neg' in sample_param and sample_param['no_neg']
        from emb_buffer import *


        if args.use_inductive:
            test_df = df[val_edge_end:]
            inductive_nodes = set(test_df.src.values).union(test_df.src.values) #TODO 这里写错了吧
            print("inductive nodes", len(inductive_nodes))
            neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
        else:
            neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])

        #TODO presample_batch = 100
        emb_buffer = None
        # if (not args.no_emb_buffer):
        #     emb_buffer = Embedding_buffer(g, df, train_param, train_edge_end, 100, args.dis_threshold, sample_param['neighbor'], gnn_param, neg_link_sampler)
        # no_neg = True
        print(f"初始化GPU sampler")
        sampler_gpu = Sampler_GPU(g, sample_param['neighbor'], sample_param['layer'], emb_buffer)
        node_num = g['indptr'].shape[0] - 1
        edge_num = g['indices'].shape[0]

        if not (('no_sample' in sample_param and sample_param['no_sample']) or (use_gpu_sample)):
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
        else:
            sampler = None
        g = None
        del g
        emptyCache()



        # 主进程发送要调用的函数名和args变量
        if (gnn_dim_node == 0):
            node_feats = None
        if (gnn_dim_edge == 0):
            edge_feats = None
        #prefetch_others_conn处理其他的index这类串行操作
        #prefetch_conn 单独处理prefetch这个并行操作
        #这么做是为了防止prefetch的结果被其他的串行操作截胡了
        prefetch_only_conn = prefetch_conn
        prefetch_conn = parent_conn
        


        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, emb_buffer, combined=combine_first).cuda()

        mailbox = MailBox(memory_param, node_num, gnn_dim_edge, prefetch_conn=prefetch_conn) if memory_param['type'] != 'none' else None
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

        
        # if (True):



        if not os.path.isdir('models'):
            os.mkdir('models')
        if args.model_name == '':
            path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
        else:
            path_saver = 'models/{}.pkl'.format(args.model_name)
        best_ap = 0
        best_e = 0
        val_losses = list()




        from feat_buffer import *
        # gpu_sampler = Sampler_GPU(g, 10)
        train_neg_sampler = None
        if (config.part_neg_sample):
            train_neg_sampler = TrainNegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])
        elif (hasattr(config, 'reuse_neg_sample') and config.reuse_neg_sample):
            train_neg_sampler = ReNegLinkSampler(node_num, args.reuse_ratio)
        else:
            train_neg_sampler = neg_link_sampler
            
        feat_buffer = Feat_buffer(args.data, None, datas, train_param, memory_param, train_edge_end, args.pre_sample_size//train_param['batch_size'],\
                                sampler_gpu,train_neg_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge), node_num=node_num, edge_num = edge_num)
        # if (not use_async_prefetch):
        #     feat_buffer.init_feat(node_feats, edge_feats)
        # feat_buffer.gen_part()

        test_ap, val_ap = [], []
        for e in range(train_param['epoch']):
            print('Epoch {:d}:'.format(e))
            time_sample = 0
            time_prep = 0
            time_tot = 0
            time_feat = 0
            time_model = 0
            time_opt = 0
            time_presample = 0
            time_gen_dgl = 0
            total_loss = 0
            time_per_batch = 0
            time_update_mem = 0
            time_update_mail = 0

            time_total_prep = 0
            time_total_strategy = 0
            time_total_compute = 0
            time_total_update = 0
            time_total_epoch = 0
            # training
            time_total_epoch_s = time.time()
            model.train()
            feat_buffer.mode = 'train'
            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                mailbox.set_buffer(feat_buffer)
                model.memory_updater.last_updated_nid = None
            if (feat_buffer is not None):
                feat_buffer.reset()
            

            sampleTime = 0
            startTime = time.time()

            #TODO 此处reorder是干嘛的?
            sampler_gpu.mask_time = 0
            left, right = 0, 0
            batch_num = 0
            batch_size = train_param['batch_size']
            while True:
                right += batch_size
                right = min(train_edge_end, right)
                if (left >= right):
                    break

                src = datas['src'][left: right]
                dst = datas['dst'][left: right]
                times = datas['time'][left: right]
                eid = datas['eid'][left: right]

                loopTime = time.time()
                t_tot_s = time.time()
                time_presample_s = time.time()

                time_total_prep_s = time.time()
                feat_buffer.run_batch(batch_num)

                if (batch_num % 1000 == 0):
                    print(f"平均每个batch用时{time_per_batch / 1000:.5f}s, 预计epoch时间: {(time_per_batch / 1000 * (train_edge_end/train_param['batch_size'])):.3f}s")
                    # mailbox.print_time()
                    print(f"run batch{batch_num}total time: {time_tot:.2f}s,presample: {time_presample:.2f}s, sample: {time_sample:.2f}s, prep time: {time_prep:.2f}s, gen block: {time_gen_dgl:.2f}s, feat input: {time_feat:.2f}s, model run: {time_model:.2f}s,\
                        loss and opt: {time_opt:.2f}s, update mem: {time_update_mem:.2f}s update mailbox: {time_update_mail:.2f}s  mask_time: {sampler_gpu.mask_time:.4f}s")
                    if (feat_buffer):
                        feat_buffer.print_time()
                    time_per_batch = 0

                # if (emb_buffer and emb_buffer.use_buffer):
                #     emb_buffer.cur_mode = 'presample'
                #     emb_buffer.run_batch(batch_num)

                #     emb_buffer.cur_mode = 'train'
                time_presample += time.time() - time_presample_s

                # emptyCache()
                
                #此处和预采样用一样的负节点
                neg_start = (batch_num % feat_buffer.presample_batch) * train_param['batch_size']
                neg_end = min(feat_buffer.neg_sample_nodes.shape[0], ((batch_num % feat_buffer.presample_batch) + 1) * train_param['batch_size'])
                neg_sample_nodes = feat_buffer.neg_sample_nodes[neg_start: neg_end]

                root_nodes = np.concatenate([src, dst, neg_sample_nodes]).astype(np.int32)
                ts = np.concatenate([times, times, times]).astype(np.float32)
                
                t_sample_s = time.time()
                if (use_gpu_sample):
                    
                    root_nodes = torch.from_numpy(root_nodes).cuda()
                    root_ts = torch.from_numpy(ts).cuda()
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        ret = sampler_gpu.sample_layer(root_nodes[:pos_root_end], root_ts[:pos_root_end], cut_zombie=args.cut_zombie)
                    else:
                        ret = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=args.cut_zombie)
                else:
                    if sampler is not None:
                        if no_neg:
                            pos_root_end = root_nodes.shape[0] * 2 // 3
                            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                        else:
                            sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                        # time_sample += ret[0].sample_time()
                
                time_sample += time.time() - t_sample_s
                # if (_ == 10):
                #     np.save(f"./sample_res_epo{e}", {"col": ret[0].col(),"row": ret[0].row(),"eid": ret[0].eid(), "ts": ret[0].ts()})

                # print(f"one loop time1: {time.time() - loopTime:.4f}")
                time1 = time.time()
                t_gen_dgl_s = time.time()
                t_prep_s = time.time()
                if (use_gpu_sample):
                    if gnn_param['arch'] != 'identity':
                        mfgs = sampler_gpu.gen_mfgs(ret)
                        root_nodes = root_nodes.cpu().numpy()
                    else:
                        mfgs = th_node_to_dgl_blocks(root_nodes, root_ts)  
                else:
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts)  
                # emptyCache()
                # print(f"node num: {mfgs[0][0].num_nodes()} edge num: {mfgs[0][0].num_edges()}")
                time_gen_dgl += time.time() - t_gen_dgl_s
                #对依赖进行分析
                # node_num = rows.src.values.shape[0]
                # src_node = torch.tensor(root_nodes[:node_num]).to(torch.int32)
                # dst_node = torch.tensor(root_nodes[node_num:node_num * 2]).to(torch.int32)
                # count_judge(src_node, dst_node)
                # print(f"node num: {mfgs[0][0].num_nodes()} edge num: {mfgs[0][0].num_edges()}")

                time_feat_s = time.time()
                mfgs = prepare_input(mfgs, node_feats, edge_feats, feat_buffer = feat_buffer, combine_first=combine_first)
                # print(f"feat时间0: {time.time() - time_feat_s:.7f}s")
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                
                time_total_prep += time.time() - time_total_prep_s
                time_prep += time.time() - t_prep_s
                time_feat += time.time() - time_feat_s
                # print(f"feat时间: {time.time() - time_feat_s:.7f}s")

                time_total_compute_s = time.time()
                optimizer.zero_grad()
                # print(f"数据转dgl图流程: {time.time() - time1:.4f}")
                
                time1 = time.time()

                time_model_s = time.time()
                pred_pos, pred_neg = model(mfgs)
                # model_structure(model)
                # print(f"模型传播流程: {time.time() - time1:.4f}")
                time_model += time.time() - time_model_s

                time_opt_s = time.time()
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss.item()) * train_param['batch_size']
                # if (batch_num % 100 == 0):
                #     # print(root_nodes)
                #     print(f"loss: {loss.item()}")
                # print(f"one loop time2.1: {time.time() - loopTime:.4f}")
                loss.backward()
                optimizer.step()
                time_opt += time.time() - time_opt_s
                time_total_compute += time.time() - time_total_compute_s
                # print(f"one loop time3: {time.time() - loopTime:.4f}")
                t_prep_s = time.time()
                
                time_total_update_s = time.time()
                if mailbox is not None:
                    
                    
                    eid = eid.cuda()
                    # eid = torch.arange(batch_num * 2000, batch_num * 2000+root_nodes.shape[0] // 3, dtype = torch.int32, device = 'cuda:0')
                    
                    mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        # block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                        block = sampler_gpu.gen_mfgs(ret, reverse=True)[0][0]
                        # block = mfgs[0][0]

                    time_upd_s = time.time()
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    time_update_mail += time.time() - time_upd_s

                    time_upd_s = time.time()
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                    time_update_mem += time.time() - time_upd_s

                time_prep += time.time() - t_prep_s
                time_tot += time.time() - t_tot_s
                time_total_update += time.time() - time_total_update_s
                # print(f"one loop time: {time.time() - loopTime:.4f}")

                time_per_batch += time.time() - t_tot_s

                left = right
                batch_num += 1

            print(f"total loop use time: {time.time() - startTime:.4f}")
            print(f"run batch{batch_num}total time: {time_tot:.2f}s,presample: {time_presample:.2f}s, sample: {time_sample:.2f}s, prep time: {time_prep:.2f}s, gen block: {time_gen_dgl:.2f}s, feat input: {time_feat:.2f}s, model run: {time_model:.2f}s,\
                loss and opt: {time_opt:.2f}s, update mem: {time_update_mem:.2f}s update mailbox: {time_update_mail:.2f}s")
            if (feat_buffer):
                feat_buffer.print_time()
            feat_buffer.mode = 'val'
            feat_buffer.refresh_memory()

            time_total_epoch += time.time() - time_total_epoch_s
            time_total_other = time_total_epoch - time_total_prep - time_total_strategy - time_total_compute - time_total_update
            print(f"prep:{time_total_prep:.4f}s strategy: {time_total_strategy:.4f}s compute: {time_total_compute:.4f}s update: {time_total_update:.4f}s epoch: {time_total_epoch:.4f}s other: {time_total_other:.4f}s")
            print(f"prep:{time_total_prep/time_total_epoch*100:.2f}% strategy: {time_total_strategy/time_total_epoch*100:.2f}% compute: {time_total_compute/time_total_epoch*100:.2f}% update: {time_total_update/time_total_epoch*100:.2f}% epoch: {time_total_epoch/time_total_epoch*100:.2f}% other: {time_total_other/time_total_epoch*100:.2f}%")


            if (not args.model_eval):
                continue
            eval_time_s = time.time()
            ap, auc = eval('val')
            
            if e > 2 and ap > best_ap:
                best_e = e
                best_ap = ap
                torch.save(model.state_dict(), path_saver)
            print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}, eval time: {:.2f}'.format(total_loss, ap, auc, time.time() - eval_time_s))
            val_ap.append(f'{ap:.6f}')
            # print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))

            # if (emb_buffer and emb_buffer.use_buffer):
            #     emb_buffer.reset_time()

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
        
        if (args.model_eval):
            print('Loading model at epoch {}...'.format(best_e))
            model.load_state_dict(torch.load(path_saver))
            model.eval()

            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None
                eval('train')
                eval('val')
            ap, auc = eval('test')
            if args.eval_neg_samples > 1:
                print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
            else:
                print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
    except Exception as e:
        print(e)
    finally:
        print(f"训练完成，退出子进程")
        # if (use_async_prefetch):
        parent_conn.send(('EXIT', ()))
        p.terminate()



