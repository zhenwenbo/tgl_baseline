
import torch
import os
import json
import numpy as np
def saveBin(tensor,savePath,addSave=False):

    savePath = savePath.replace('.pt','.bin')
    dir = os.path.dirname(savePath)
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    json_path = dir + '/saveBinConf.json'
    
    tensor_info = {
        'dtype': str(tensor.dtype).replace('torch.',''),
        'device': str(tensor.device),
        'shape': (tensor.shape)
    }

    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    
    config[savePath] = tensor_info
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)

    if isinstance(tensor, torch.Tensor):
        tensor.cpu().numpy().tofile(savePath)
    elif isinstance(tensor, np.ndarray):
        tensor.tofile(savePath)
data = 'STACK'


import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *

# gen_feat(data, use_pt = True, budget = 10 * 1024 ** 3)




d = data
def df2bin():
    file_path = '/raid/guorui/DG/dataset/{}/edges.csv'.format(d)
    if (d in ['BITCOIN']):
        df = pd.read_csv(file_path, sep=' ', names=['src', 'dst', 'time'])
        df['Unnamed: 0'] = range(0, len(df))
    else:
        df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(d))

    print(f"加载df完毕")
    dataset_conf = {}

    if (data in ['BITCOIN']):
        train_edge_end = 86063713
        val_edge_end = 110653345
        dataset_conf['train_edge_end'] = train_edge_end
        dataset_conf['val_edge_end'] = val_edge_end
    else:
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]
        dataset_conf['train_edge_end'] = train_edge_end.item()
        dataset_conf['val_edge_end'] = val_edge_end.item()

    base_path = f'/raid/guorui/DG/dataset/{data}'

    src = torch.from_numpy(df.src.values.astype(np.int32))
    saveBin(src, f'{base_path}/df-src.bin')

    dst = torch.from_numpy(df.dst.values.astype(np.int32))
    saveBin(dst, f'{base_path}/df-dst.bin')

    eid = torch.from_numpy(df['Unnamed: 0'].values.astype(np.int32))
    saveBin(eid, f'{base_path}/df-eid.bin')

    time = torch.from_numpy(df.time.values)
    saveBin(time, f'{base_path}/df-time.bin')

    json_path = f'{base_path}/df-conf.json'

    with open(json_path, 'w') as f:
        json.dump(dataset_conf, f, indent=4)


# df2bin()

# eid = torch.from_numpy(g['eid']).cuda()
# print(f'eid shape: {eid.shape} unique shape: {torch.unique(eid).shape}')

# ef = loadBin(f'/raid/guorui/DG/dataset/{data}/edge_features.bin')

# ef = ef[eid]

def reorder_edge():
    print(f"reorder edge")
    import torch
    g = np.load('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(d))
    eid = torch.from_numpy(g['eid']).cuda()

    max_val = eid.max().item()
    res = torch.zeros(max_val + 1, dtype=torch.long).cuda()
    res.scatter_(0, eid, torch.arange(len(eid)).cuda())

    print(res)

    # 保存后，例如需要寻找eid = 100的边特征，那么需要edge_feat_reorder[map[eid]]即可
    # saveBin(ef.cpu(), f'/raid/guorui/DG/dataset/{data}/edge_features_reorder.bin')
    saveBin(res.cpu().to(torch.int32), f'/raid/guorui/DG/dataset/{data}/edge_reorder_map.bin')

reorder_edge()