
import sys

root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import dgl
import numpy as np
import pandas as pd
import time
from utils import *
from sampler.sampler import *


from sampler.sampler_gpu import *

path = '/raid/guorui/DG/dataset/MAG/node_features.bin'
for i in range(100):
    idx = torch.randint(0, 121751665, (60000,))
    print(idx)
    start = time.time()
    # res = read_data_from_file_concurrent1(path, idx, (121751665, 768), dtype=np.float16)

    res = loadBinDisk(path, idx)
    # print(torch.sum(res != res1))
    print(f"{res.shape[0]} {time.time() - start:.4f}s")