

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

node_f = torch.load('/raid/guorui/DG/dataset/MAG/node_features.pt')
saveBin(node_f, '/raid/guorui/DG/dataset/MAG/node_features.bin')