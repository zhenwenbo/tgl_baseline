# import time
# import torch
# import numpy as np
# base_path = '/raid/guorui/DG/dataset/GDELT/part-600000-[10]/'
# for i in range(0,100):
#     start = time.time()
#     res = torch.load(f'{base_path}part{i}_edge_feat.pt')
#     print(f"torch加载 {time.time() - start:.8f}s")
#     print(res)
#     start = time.time()
#     res = torch.from_numpy(np.fromfile(f'{base_path}part{i}_edge_feat.pt', dtype = np.float32))
#     print(res)
    # print(f"np二进制加载 {time.time() - start:.8f}s")


#测试save的速度
import torch
import numpy as np
import os
import json
def saveBin(tensor,savePath,addSave=False):

    dir = os.path.dirname(savePath)
    print(f'dir: {dir}')
    json_path = dir + '/saveBinConf.json'
    
    tensor_info = {
        'dtype': str(tensor.dtype).replace('torch.',''),
        'device': str(tensor.device)
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

test_tensor = torch.arange(100000000, dtype = torch.int32).cuda()
saveBin(test_tensor, '/raid/guorui/DG/dataset/test.bin')

import torch
import numpy as np
import os
import json

confs = {}
path = '/raid/guorui/DG/dataset/test.bin'

def loadConf(path):
    print(f"load Conf")
    directory = os.path.dirname(path)
    json_path = directory + '/saveBinConf.json'
    with open(json_path, 'r') as f:
        res = json.load(f)
    confs[directory] = res

def loadBin(path):
    directory = os.path.dirname(path)
    if directory not in confs:
        loadConf(path)

    res = torch.from_numpy(np.fromfile(path, dtype = getattr(np, confs[directory][path]['dtype']))).to(confs[directory][path]['device'])
    return res

res = loadBin(path)