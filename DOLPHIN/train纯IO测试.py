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
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#TODO 在LASTFM下确实会影响时间, 但是在大数据集上的影响好像不大? 

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='TALK')
parser.add_argument('--config', type=str, help='path to config file', default='/raid/guorui/workspace/dgnn/b-tgl/config/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)

parser.add_argument('--reuse_ratio', type=float, default=0.9, help='reuse_ratio')
parser.add_argument('--train_conf', type=str, default='disk', help='name of stored model')
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
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

if (args.data == 'MAG'):
    args.pre_sample_size = 600000
        
if (args.data == 'GDELT' and sample_param['layer'] == 2):
    sample_param['neighbor'] = [8, 8]
    train_param['epoch'] = 1
    print(f"GDELT二跳修改邻域为8,8")

if (args.data == 'BITCOIN'):
    train_param['epoch'] = 2

# if (config.epoch != -1):
#     train_param['epoch'] = config.epoch
#     print(f"预设epoch为 {config.epoch}")

if (args.data == 'BITCOIN' and 'TGN' not in args.config):
    train_param['epoch'] = 1
    print(f"BITCOIN后面两个的disk只跑一个epoch")

train_param['epoch'] = 1
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
    print(f"设置随机种子为{seed}")
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


# set_seed(0)
if __name__ == '__main__':

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
    


    print(f"训练完成，退出子进程")
    # if (use_async_prefetch):
    parent_conn.send(('EXIT', ()))
    p.terminate()