

# import time
# import torch


# test = torch.randint(0, 100000000, size=(100000000,))

# start_time = time.time()
# for i in range(10):
#     torch.unique(test)
#     print(f"单次 {time.time() - start_time:.4f}s")
# print(f"10次sort用时 {time.time() - start_time:.5f}s")
import torch
import numpy as np
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

import numpy as np
import torch
import time
import threading
class File_Load():
    
    def __init__(self):
        self.async_load_dic = {}
        self.async_load_flag = {}

    def load_file(self, paths, tags, i):

        if (os.path.exists(paths[i].replace('.pt', '.bin'))):
            self.async_load_dic[tags[i]] = loadBin(paths[i])
        else:
            self.async_load_dic[tags[i]] = None


        #这个完成之后加载下一个
        if ((i + 1) < len(paths)):
            thread = threading.Thread(target=self.load_file, args=(paths, tags, i + 1))
            self.async_load_flag[tags[i + 1]] = thread
            thread.start()

    def async_load(self, paths, tags):
        thread = threading.Thread(target=self.load_file, args=(paths, tags, 0))
        self.async_load_flag[tags[0]] = thread
        thread.start()

file_load = File_Load()
paths = []
tags = []
for i in range(1000):
    paths.append(f"/raid/guorui/DG/dataset/BITCOIN/part-{60000}-[10]/part{i}_edge_feat.pt")
    tags.append(f"edge{i}")
# file_load.async_load(paths, tags)
print(file_load.async_load_flag)


test = torch.randint(0, 100000000, size=(100000000,)).numpy()
# start_time = time.time()
for i in range(10):
    start_time = time.time()
    np.sort(test)
    print(f"单次 {time.time() - start_time:.4f}s")
    print(len(file_load.async_load_flag))
print(f"10次sort用时 {time.time() - start_time:.5f}s")