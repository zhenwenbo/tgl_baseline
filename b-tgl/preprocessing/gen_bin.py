
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
data = 'MAG'
# path = f'/raid/guorui/DG/dataset/{data}/edge_features.pt'
# tensor = torch.load(path)
# if (tensor.dtype == torch.bool):
#     tensor = tensor.to(torch.float32)
# saveBin(tensor, path)

path = f'/raid/guorui/DG/dataset/{data}/node_features.pt'
tensor = torch.load(path)
if (tensor.dtype == torch.bool):
    tensor = tensor.to(torch.float32)
saveBin(tensor, path)

