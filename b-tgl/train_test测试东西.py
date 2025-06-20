import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='TALK')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)
parser.add_argument('--use_gpu_sampler', action='store_true', default=True)

parser.add_argument('--train_conf', type=str, default='basic_eval', help='name of stored model')
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--reuse_ratio', type=float, default=0, help='reuse_ratio')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

from config.train_conf import *
GlobalConfig.conf = args.train_conf + '.json'
config = GlobalConfig()
args.use_async_prefetch = config.use_async_prefetch
args.pre_sample_size = 60000
args.cut_zombie = config.cut_zombie

if (hasattr(config, 'model')):
    args.config = f'/raid/guorui/workspace/dgnn/b-tgl/config/{config.model}-{config.layer}.yml'

if (config.model_eval):
    args.model_eval = True
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print(f"训练配置: {config.config_data}")
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
import torch.multiprocessing as multiprocessing
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"设置随机种子为{seed}")


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

import numpy as np
import matplotlib.pyplot as plt

cur_version = f'{time.time():.0f}'
os.makedirs(f'./test_model/{cur_version}')

def paint(ac_log, mode):
    # 获取总的 batch 数量
    n_batches = len(ac_log)

    # 用于绘制的坐标
    x = []  # 存储所有预测错误的样本索引
    y = []  # 存储对应的 batch 索引

    # 遍历 ac_log 来构建 x 和 y
    for batch_idx, indices in enumerate(ac_log):
        x.extend(indices)
        y.extend([batch_idx] * len(indices))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y,s=10, alpha= 0.5, edgecolor='none')
    plt.xlabel('Sample Index within Batch')
    plt.ylabel('Batch Number')
    plt.title('Prediction Errors per Batch')
    plt.grid(True)
    plt.savefig(f'./test_model/{cur_version}/{e}_{mode}.png')

def paint_re(acc_log, mode):
    # 计算 batch 数
    num_batches = len(acc_log)

    # 收集所有的索引
    all_indices = [index for batch_indices in acc_log for index in batch_indices]
    min_index = min(all_indices)
    max_index = max(all_indices)

    # 设置分段数量
    num_bins = 20
    x_bins = np.linspace(min_index, max_index, num_batches + 1)
    y_bins = np.linspace(0, num_batches, num_bins + 1)
    # 扁平化 acc_log 以获得 y 轴的对应 batch 数
    y_values = []
    x_values = []

    for batch_index, indices in enumerate(acc_log):
        y_values.extend([batch_index] * len(indices))  # 每个错误的索引对应当前 batch
        x_values.extend(indices)  # 将错误的索引添加到 x_values

    # 生成热力图数据
    heatmap, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins])
    import matplotlib.pyplot as plt

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap.T, origin='lower', aspect='auto', cmap='hot', 
            extent=[min_index, max_index, 0, num_batches])
    plt.colorbar(label='Frequency')
    plt.xlabel('Index')
    plt.ylabel('Batch Number')
    plt.title('Heatmap of Prediction Errors by Batch and Index')
    plt.savefig(f'./test_model/{cur_version}/{e}_{mode}_热力图.png')

def eval(mode='val'):
    ac_log = []
    set_mode(mode)
    if (emb_buffer):
        emb_buffer.cur_mode = 'val'

    
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    print(f"eval: {mode}")
    first_test = False
    if mode == 'val':
        first_test = True
        eval_df = df[train_edge_end:val_edge_end]
        left = train_edge_end
        end = val_edge_end
    elif mode == 'test':
        first_test = True
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
        left = val_edge_end
        end = len(df)
    elif mode == 'test_train':
        eval_df = df[:train_edge_end]
        left = 0
        end = train_edge_end
    # feat_buffer.cur_block = 0
    right = left
    batch_num = 0
    batch_size = train_param['batch_size']

    if (mailbox is None and mode == 'test'):
        # 在TGAT下提前加载测试集所在block的数据
        feat_buffer.run_batch(left // train_param['batch_size'], test_block = left // args.pre_sample_size)

    with torch.no_grad():
        total_loss = 0
        for batch_num, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
        # while True:
            
            feat_buffer.run_batch(batch_num)
            # right += batch_size
            # right = min(end, right)
            # if (left >= right):
            #     break
            # rows = df[left:right]
            
            if (mode == 'test'):
                cur_num =  len(rows) * neg_samples
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_test(left, left+cur_num)]).astype(np.int32)
                left += cur_num
            else:
                # root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
                neg_start = (batch_num % feat_buffer.presample_batch) * train_param['batch_size']
                if (first_test):
                    first_test = False
                    neg_start += left % train_param['batch_size']
                neg_end = min(len(rows) + neg_start, ((batch_num % feat_buffer.presample_batch) + 1) * train_param['batch_size'])
                neg_sample_nodes = feat_buffer.neg_sample_nodes[neg_start: neg_end]

                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_sample_nodes]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            
            # neg_start = (batch_num % feat_buffer.presample_batch) * train_param['batch_size']
            # neg_end = min(feat_buffer.neg_sample_nodes.shape[0], ((batch_num % feat_buffer.presample_batch) + 1) * train_param['batch_size'])
            # neg_sample_nodes = feat_buffer.neg_sample_nodes[neg_start: neg_end]

            # root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_sample_nodes]).astype(np.int32)
            # ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            
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
            pos_loss = creterion(pred_pos, torch.ones_like(pred_pos)).item()
            neg_loss = creterion(pred_neg, torch.zeros_like(pred_neg)).item()
            if (batch_num % 100 == 0):
                # print(root_nodes)
                print(f"正边loss:{pos_loss:.10f} 负边loss:{neg_loss:.10f}")
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)


            y_pred_sort, ind = torch.sort(y_pred.reshape(-1), descending=True)
            res = ind[torch.nonzero((ind[ind.shape[0] // 2:] <= ind.shape[0] // 2)) .reshape(-1)+ ind.shape[0] // 2]

            ac_log.append(res.tolist())
            # print(f"mode: {mode}, cur_ap: {cur_ap}, 标签准确率: {torch.sum(pred_label == y_true) / y_true.shape[0]}")

            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = torch.from_numpy(rows['Unnamed: 0'].values).cuda().to(torch.int32)
                mem_edge_feats = feat_buffer.select_index('edge_feats', eid.long())
                # mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)

            # left = right
            # batch_num += 1
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())

    # paint(ac_log, mode)
    paint_re(ac_log, mode)
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

def set_mode(mode):
    if (use_async_prefetch):
        parent_conn.send(('set_mode', (mode, )))
        result = parent_conn.recv()
    if (feat_buffer):
        feat_buffer.mode = mode

set_seed(42)
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    
    total_val_res = []
    total_test_res = []
    test_epo = 1
    if (args.data == 'GDELT'):
        print(f"GDELT数据集改为只测一轮...")
        test_epo = 1

    if (args.data == 'BITCOIN'):
        print(f"BITCOIN数据集改为只测一轮...")
        test_epo = 1

    for i in range(test_epo):
        global node_feats, edge_feats
        node_feats, edge_feats = None,None

        if (not args.use_async_prefetch):
            node_feats, edge_feats = load_feat(args.data)
        sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
        if (args.data == 'GDELT' or args.data == 'BITCOIN'):
            print(f"GDELT和BITCOIN只跑5个epoch")
            train_param['epoch'] = 5
        g, df = load_graph(args.data)

        #TODO GDELT改fanout为[7,7]
        # sample_param['neighbor'] = [7,7]
        
        # train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        # val_edge_end = df[df['ext_roll'].gt(1)].index[0]

        json_path = f'/raid/guorui/DG/dataset/{args.data}/df-conf.json'
        
        with open(json_path, 'r') as f:
            df_conf = json.load(f)
        train_edge_end = df_conf['train_edge_end']
        val_edge_end = df_conf['val_edge_end']

        if args.use_inductive:
            inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
            df = df.iloc[inductive_inds]
            
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

        if (args.use_async_prefetch):
            if (args.data == 'LASTFM'):
                gnn_dim_edge = 128
                gnn_dim_node = 128
            elif (args.data == 'TALK'):
                gnn_dim_edge = 172
                gnn_dim_node = 172
            elif (args.data == 'STACK'):
                gnn_dim_edge = 172
                gnn_dim_node = 172
            elif (args.data == 'WIKI'):
                gnn_dim_edge = 172
                gnn_dim_node = 172
            elif (args.data == 'GDELT'):
                gnn_dim_edge = 182 #TODO 为什么下载下来的数据集的edge feat是182呢？
                gnn_dim_node = 413
            elif (args.data == 'BITCOIN'):
                gnn_dim_edge = 172
                gnn_dim_node = 172
            else:
                raise RuntimeError("have not this dataset config!")
        

        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True


        from sampler.sampler_gpu import *
        use_gpu_sample = args.use_gpu_sampler
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
        if (not args.no_emb_buffer):
            emb_buffer = Embedding_buffer(g, df, train_param, train_edge_end, 100, args.dis_threshold, sample_param['neighbor'], gnn_param, neg_link_sampler)
        # no_neg = True
        sampler_gpu = Sampler_GPU(g, sample_param['neighbor'], sample_param['layer'], emb_buffer)


        
        
        from pre_fetch import *
        use_async_prefetch = args.use_async_prefetch

        parent_conn = None
        prefetch_conn = None
        if (use_async_prefetch):
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
        # print(model.memory_updater.updater.weight_ih)

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

        use_reNegSample = True
        train_neg_sampler = None
        if (config.part_neg_sample):
            train_neg_sampler = TrainNegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])
        elif use_reNegSample:
            print(f"使用重放负采样")
            train_neg_sampler = ReNegLinkSampler(g['indptr'].shape[0] - 1, args.reuse_ratio)
        else:
            print(f"使用常规负采样")
            train_neg_sampler = neg_link_sampler
            
        feat_buffer = Feat_buffer(args.data, g, df, train_param, memory_param, train_edge_end, args.pre_sample_size//train_param['batch_size'],\
                                sampler_gpu,train_neg_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge))
        feat_buffer.val_edge_end = val_edge_end
        feat_buffer.test_edge_end = len(df)
        feat_buffer.use_b_test = True
        feat_buffer.test_neg_sampler = neg_link_sampler
        if (not use_async_prefetch):
            feat_buffer.init_feat(node_feats, edge_feats)
        # feat_buffer.gen_part()

        val_ap, test_ap = [], []

        for e in range(20):
            epoch_s = time.time()
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
            # training
            model.train()
            set_mode('train')
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
            test = df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)])
            # test1 = df[:train_edge_end].groupby(1)
            #TODO 此处reorder是干嘛的?
            # ap, auc = eval('val')
            feat_buffer.cur_block = 0
            for batch_num, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
                # if (batch_num == 30):
                    
                #     torch.save(feat_buffer.get_mailbox(torch.arange(1980, dtype = torch.int32, device = 'cuda:0')), '/raid/guorui/workspace/dgnn/b-tgl/mailbox_b')

                loopTime = time.time()
                t_tot_s = time.time()
                time_presample_s = time.time()
                feat_buffer.run_batch(batch_num)

                if (args.data in ['GDELT', 'STACK', 'TALK'] and batch_num % 1000 == 0):
                    print(f"平均每个batch用时{time_per_batch / 1000:.5f}s, 预计epoch时间: {(time_per_batch / 1000 * (train_edge_end/train_param['batch_size'])):.3f}s")
                    print(f"run batch{batch_num}total time: {time_tot:.2f}s,presample: {time_presample:.2f}s, sample: {time_sample:.2f}s, prep time: {time_prep:.2f}s, gen block: {time_gen_dgl:.2f}s, feat input: {time_feat:.2f}s, model run: {time_model:.2f}s,\
                        loss and opt: {time_opt:.2f}s, update mem: {time_update_mem:.2f}s update mailbox: {time_update_mail:.2f}s")
                    if (feat_buffer):
                        feat_buffer.print_time()
                    time_per_batch = 0

                if (emb_buffer and emb_buffer.use_buffer):
                    emb_buffer.cur_mode = 'presample'
                    emb_buffer.run_batch(batch_num)

                    emb_buffer.cur_mode = 'train'
                time_presample += time.time() - time_presample_s

                # emptyCache()
                
                #此处和预采样用一样的负节点
                neg_start = (batch_num % feat_buffer.presample_batch) * train_param['batch_size']
                neg_end = min(feat_buffer.neg_sample_nodes.shape[0], ((batch_num % feat_buffer.presample_batch) + 1) * train_param['batch_size'])
                neg_sample_nodes = feat_buffer.neg_sample_nodes[neg_start: neg_end]

                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_sample_nodes]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                # print(neg_sample_nodes)
                # saveBin(torch.from_numpy(root_nodes), '/raid/guorui/workspace/dgnn/b-tgl/test/test-acc/b-node.bin')
                # saveBin(torch.from_numpy(ts), '/raid/guorui/workspace/dgnn/b-tgl/test/test-acc/b-ts.bin')
                t_sample_s = time.time()
                if (use_gpu_sample):
                    
                    root_nodes = torch.from_numpy(root_nodes).cuda()
                    root_ts = torch.from_numpy(ts).cuda()
                    # print(root_nodes)
                    # print(root_ts)
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
                    mfgs = sampler_gpu.gen_mfgs(ret)
                    root_nodes = root_nodes.cpu().numpy()
                else:
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts)  
                # emptyCache()
                # print(f"root_nodes: {root_nodes}, ts: {ts}")
                # print(f"node num: {mfgs[0][0].num_nodes()} edge num: {mfgs[0][0].num_edges()}")
                time_gen_dgl += time.time() - t_gen_dgl_s
                #对依赖进行分析
                # node_num = rows.src.values.shape[0]
                # src_node = torch.tensor(root_nodes[:node_num]).to(torch.int32)
                # dst_node = torch.tensor(root_nodes[node_num:node_num * 2]).to(torch.int32)
                # count_judge(src_node, dst_node)

                time_feat_s = time.time()
                mfgs = prepare_input(mfgs, node_feats, edge_feats, feat_buffer = feat_buffer, combine_first=combine_first)
                # print(f"feat时间0: {time.time() - time_feat_s:.7f}s")
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])

                # print(mfgs[0][0].edata)
                # print(mfgs[0][0].ndata)

                time_prep += time.time() - t_prep_s
                time_feat += time.time() - time_feat_s
                # print(f"feat时间: {time.time() - time_feat_s:.7f}s")

                optimizer.zero_grad()
                # print(f"数据转dgl图流程: {time.time() - time1:.4f}")
                
                time1 = time.time()

                time_model_s = time.time()
                # if (batch_num == 30):
                    
                #     print(mfgs[0][0].srcdata['ID'])
                #     print(mfgs[0][0].ndata['mem_input']['_N'])
                #     print(mfgs[0][0].ndata['mem']['_N'])
                #     print(mfgs[0][0].edata['f'])
                #     print(mfgs[0][0].ndata['ID']['_N'])
                #     torch.save(feat_buffer.get_mailbox(torch.arange(1980, dtype = torch.int32, device = 'cuda:0')), '/raid/guorui/workspace/dgnn/b-tgl/mailbox_b')
                # print(model.memory_updater.updater.weight_ih)

                pred_pos, pred_neg = model(mfgs)
                # model_structure(model)
                # print(f"模型传播流程: {time.time() - time1:.4f}")
                time_model += time.time() - time_model_s

                time_opt_s = time.time()
                # print(pred_pos)
                # print(pred_neg)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                if (batch_num % 100 == 0):
                    # print(root_nodes)
                    print(f"loss: {loss.item()}")
                total_loss += float(loss.item()) * train_param['batch_size']
                # print(f"one loop time2.1: {time.time() - loopTime:.4f}")
                loss.backward()
                optimizer.step()
                time_opt += time.time() - time_opt_s

                # print(f"one loop time3: {time.time() - loopTime:.4f}")
                t_prep_s = time.time()
                
                if mailbox is not None:
                    
                    
                    # eid = torch.from_numpy(rows['Unnamed: 0'].values).to(torch.int32).cuda()
                    eid = torch.arange(batch_num * 2000, batch_num * 2000+root_nodes.shape[0] // 3, dtype = torch.int32, device = 'cuda:0')
                    t_prep_s = time.time()
                    mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                    # if (batch_num == 29):
                    #     print(eid)
                    #     print(mem_edge_feats)
                    t_prep_s = time.time()
                    # print(model.memory_updater.last_updated_nid)
                    # print(model.memory_updater.last_updated_memory)
                    # print(root_nodes)
                    # print(ts)
                    # print(block)
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        # block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                        block = mfgs[0][0]

                    time_upd_s = time.time()
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    time_update_mail += time.time() - time_upd_s

                    time_upd_s = time.time()
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                    time_update_mem += time.time() - time_upd_s

                time_prep += time.time() - t_prep_s
                time_tot += time.time() - t_tot_s
                # print(f"one loop time: {time.time() - loopTime:.4f}")

                time_per_batch += time.time() - t_tot_s

            # print(f"total loop use time: {time.time() - startTime:.4f}")
            print(f"run batch{batch_num}total time: {time_tot:.2f}s,presample: {time_presample:.2f}s, sample: {time_sample:.2f}s, prep time: {time_prep:.2f}s, gen block: {time_gen_dgl:.2f}s, feat input: {time_feat:.2f}s, model run: {time_model:.2f}s,\
                   loss and opt: {time_opt:.2f}s, update mem: {time_update_mem:.2f}s update mailbox: {time_update_mail:.2f}s")
            feat_buffer.refresh_memory()
            # total_memory = feat_buffer.select_index('memory', torch.arange(1980, dtype = torch.int64))
            # torch.save(total_memory, '/raid/guorui/workspace/dgnn/b-tgl/test/test-acc/mem_b.bin')
            # print(total_memory)

            args.model_eval = True
            if (not args.model_eval):
                continue
            eval_time_s = time.time()
            # ap, auc = eval('val')
            # val_ap.append(f'{ap:.6f}')
            
            test_per_epoch = True
            args.model_eval = True
            if (test_per_epoch):
                if (args.model_eval):
                    if mailbox is not None:
                        mailbox.reset()
                        mailbox.set_buffer(feat_buffer)
                        model.memory_updater.last_updated_nid = None
                    if (feat_buffer is not None):
                        feat_buffer.reset()

                    model.eval()

                    if sampler is not None:
                        sampler.reset()
                    if mailbox is not None:
                        mailbox.reset()
                        model.memory_updater.last_updated_nid = None
                        eval('test_train')
                        eval('val')
                    ap, auc = eval('test')
                    # if args.eval_neg_samples > 1:
                    #     print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
                    # else:
                    #     print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
                    test_ap.append(f'{ap:.6f}')
            print(f'val: {val_ap}; test: {test_ap}')

            print(f"epoch全长用时{time.time() - epoch_s}")




        total_val_res.append(val_ap)
        total_test_res.append(test_ap)

        print(total_val_res)
        print(total_test_res)

        if (use_async_prefetch):
            parent_conn.send(('EXIT', ()))
            p.terminate()
    
    print(f"total_val_res: {total_val_res}")
    print(f"total_test_res: {total_test_res}")
    
        

    print(f"训练完成，退出子进程")
    if (use_async_prefetch):
        parent_conn.send(('EXIT', ()))
        p.terminate()



