import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
import gc
import json

def cuda_GB():
    return f"{torch.cuda.memory_allocated() / 1024**3:.4f}GB"

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def emptyCache():
    torch.cuda.empty_cache()
    gc.collect()

#在当前目录下保存这个tensor的数据类型以及所处容器(cpu or cuda)以便恢复
def saveBin(tensor,savePath,addSave=False):

    savePath = savePath.replace('.pt','.bin')
    dir = os.path.dirname(savePath)
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    json_path = dir + '/saveBinConf.json'
    
    tensor_info = {
        'dtype': str(tensor.dtype).replace('torch.',''),
        'device': str(tensor.device),
        'shape': list(tensor.shape)
    }

    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    
    if addSave:
        if savePath in config and config[savePath]['shape'] is not None:
            tensor_info['shape'][0] += config[savePath]['shape'][0]
    config[savePath] = tensor_info

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)

    if addSave :
        with open(savePath, 'ab') as f:
            if isinstance(tensor, torch.Tensor):
                tensor.numpy().tofile(f)
            elif isinstance(tensor, np.ndarray):
                tensor.tofile(f)
    else:
        if isinstance(tensor, torch.Tensor):
            tensor.cpu().numpy().tofile(savePath)
        elif isinstance(tensor, np.ndarray):
            tensor.tofile(savePath)



confs = {}

def loadConf(path):
    directory = os.path.dirname(path)
    json_path = directory + '/saveBinConf.json'
    with open(json_path, 'r') as f:
        res = json.load(f)
    confs[directory] = res

    return res

def loadBin(path):
    path = path.replace('.pt', '.bin')
    directory = os.path.dirname(path)
    if directory not in confs:
        loadConf(path)

    cur_conf = confs[directory][path]

    
        
    res = torch.from_numpy(np.fromfile(path, dtype = getattr(np, cur_conf['dtype']))).to(cur_conf['device']).reshape(cur_conf['shape'])
    if (cur_conf['dtype'] == 'float16'):
        res = res.to(torch.float32)
    return res

def read_data_from_file(file_path, indices, shape, dtype=np.float32, batch_size=128000, use_slice = False):
    # 利用 numpy 的内存映射函数来映射整个文件
    use_slice = False
    data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    result = []
    if (use_slice):
        print(f"预处理slice")
        batch_slice = torch.nonzero(torch.diff(indices) > 24).reshape(-1) + 1
        left = 0
        for right in batch_slice:
            batch_indices = indices[left:right]
            # 批量读取
            batch_data = data[batch_indices, :]
            result.append(torch.tensor(batch_data).reshape(-1))  # 转换为torch tensor
            left = right

        batch_indices = indices[left:indices.shape[0]]
        batch_data = data[batch_indices, :]
        result.append(torch.tensor(batch_data).reshape(-1))  # 转换为torch tensor
        left = right
    else:
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            # 批量读取
            batch_data = data[batch_indices, :]
            result.append(torch.tensor(batch_data).reshape(-1))  # 转换为torch tensor
    
    # 将结果合并成一个tensor
    if (len(result) == 0):
        return torch.from_numpy(np.empty(0, dtype=dtype))
    return torch.cat(result, dim=0).reshape(-1, shape[1])

def read_binary_file_indices(file_path, indices, feat_len = 172, dtype=np.float32):

    shape = (feat_len,)
    row_bytes = np.prod(shape) * dtype().itemsize
    
    data = []
    with open(file_path, 'rb') as f:
        for idx in indices:
            # 计算偏移量
            offset = idx * row_bytes
            f.seek(offset)
            
            # 读取一行数据
            row_data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
            row_data = row_data.reshape(shape)
            
            data.append(row_data)
    
    return torch.from_numpy(np.array(data))

def loadBinDisk(path, ind, use_slice = False):
    path = path.replace('.pt', '.bin')
    directory = os.path.dirname(path)
    if directory not in confs:
        loadConf(path)

    cur_conf = confs[directory][path]
    ind = ind.cpu()
    read_disk_s = time.time()
    # res = read_binary_file_indices(file_path=path, indices=ind, feat_len=cur_conf['shape'][1], dtype=getattr(np, cur_conf['dtype']))
    
    res = read_data_from_file(file_path=path, indices=ind, shape=tuple(cur_conf['shape']), dtype=getattr(np, cur_conf['dtype']), use_slice = use_slice)
    if (cur_conf['dtype'] == 'float16'):
        res = res.to(torch.float32)
    # print(f"读取disk用时 {time.time() - read_disk_s:.4f}s shape:{res.shape}")
    return res

def gen_feat(d, rand_de=0, rand_dn=0):
    path = f'/raid/guorui/DG/dataset/{d}'
    node_feats = None
    edge_feats = None

    if d == 'LASTFM':
        edge_feats = torch.randn(1293103, rand_de)
    elif d == 'MOOC':
        edge_feats = torch.randn(411749, rand_de)
    elif d == 'STACK':
        edge_feats = torch.randn(63497049, 172)
    elif d == 'TALK':
        edge_feats = torch.randn(7833139, 172)
    elif d == 'BITCOIN':
        edge_feats = torch.randn(122948162, 172)

    if d == 'LASTFM':
        node_feats = torch.randn(1980, rand_dn)
    elif d == 'MOOC':
        node_feats = torch.randn(7144, rand_dn)
    elif d == 'STACK':
        node_feats = torch.randn(2601977, 172)
    elif d == 'TALK':
        node_feats = torch.randn(1140149, 172)
    # elif d == 'BITCOIN':
    #     node_feats = torch.randn(24575383, 172)
    
    if (edge_feats != None):
        saveBin(edge_feats, path + '/edge_features.pt')
    if (node_feats != None):
        saveBin(node_feats, path + '/node_features.pt')



def load_feat(d, load_node = True, load_edge = True):
    node_feats = torch.empty(0)
    if load_node and os.path.exists('/raid/guorui/DG/dataset/{}/node_features.pt'.format(d)):
        node_feats = torch.load('/raid/guorui/DG/dataset/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = torch.empty(0)
    if load_edge and os.path.exists('/raid/guorui/DG/dataset/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('/raid/guorui/DG/dataset/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)

    # print(f"load feats. node_feat: {node_feats.shape},edge_feat: {edge_feats.shape}")
    feat_mem = 0
    feat_mem += node_feats.numel() * 4 / 1024**3
    feat_mem += edge_feats.numel() * 4 / 1024**3
    print(f"feat memory: {feat_mem:.3f}GB")
    return node_feats, edge_feats

def load_graph(d):
    file_path = '/raid/guorui/DG/dataset/{}/edges.csv'.format(d)
    if (d in ['BITCOIN']):
        df = pd.read_csv(file_path, sep=' ', names=['src', 'dst', 'time'])
        df['Unnamed: 0'] = range(1, len(df) + 1)
    else:
        df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))

    # datas = {}
    # base_path = f'/raid/guorui/DG/dataset/{d}'
    # datas['src'] = loadBin(f'{base_path}/df-src.bin')
    # datas['dst'] = loadBin(f'{base_path}/df-dst.bin')
    # datas['time'] = loadBin(f'{base_path}/df-time.bin')
    # datas['eid'] = loadBin(f'{base_path}/df-eid.bin')

    g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
    return g, df


def load_graph_bin(d):
    datas = {}
    base_path = f'/raid/guorui/DG/dataset/{d}'
    datas['src'] = loadBin(f'{base_path}/df-src.bin')
    datas['dst'] = loadBin(f'{base_path}/df-dst.bin')
    datas['time'] = loadBin(f'{base_path}/df-time.bin')
    datas['eid'] = loadBin(f'{base_path}/df-eid.bin')

    json_path = f'/raid/guorui/DG/dataset/{d}/df-conf.json'

    with open(json_path, 'r') as f:
        df_conf = json.load(f)

    g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
    return g, datas, df_conf

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            #注意，tgl的采样形式最后的block，对于正采样的源节点和正边，只出现源节点，正边被屏蔽。
            #因此，col和row的长度为非正边的边的长度。通过指定src节点数将所有出现的节点放在src
            #src中即使有重复的节点这里也将他们视为不同节点，因为他们的时间戳不同
            #dst中同src，dst的节点就是所有的正采样源节点(重复节点视为不同节点)
            #边数量由于去除了正边，因此dts赋值的时候也需要去除正边
            #src的ts值赋值所有采样节点的ts值。
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():] #屏蔽num_dst_nodes即采样正边的边。 
            #TODO 是否要屏蔽刚好出现的当前时间戳的负边? (不知道是不是这个原因导致的我们的采样器结果ap会高一点)
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def th_node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0]).to('cuda:0')
    b.srcdata['ID'] = root_nodes
    b.srcdata['ts'] = ts
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(mfgs, node_feats = None, edge_feats = None, feat_buffer = None, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if feat_buffer is not None and node_feats is not None: 
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                # srch = node_feats[b.srcdata['ID'].long()].float()
                if (feat_buffer != None and feat_buffer.mode == 'train'):
                    srch = feat_buffer.get_n_feat(b.srcdata['ID'])
                else:
                    if (feat_buffer != None and feat_buffer.prefetch_conn != None):
                        srch = feat_buffer.select_index('node_feats', b.srcdata['ID'].long()).float()
                    else:
                        srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if feat_buffer is not None and edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        if (feat_buffer != None and feat_buffer.mode == 'train'):
                            srch = feat_buffer.get_e_feat(b.edata['ID'])
                        else:
                            if (feat_buffer != None and feat_buffer.prefetch_conn != None):
                                srch = feat_buffer.select_index('edge_feats', b.edata['ID'].long()).float()
                            else:
                                srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

