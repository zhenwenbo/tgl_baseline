import sys
root_dir = '/raid/guorui/workspace/dgnn/b-tgl'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils import *
import torch
import dgl

start = loadBin('/raid/guorui/DG/dataset/BITCOIN/simple_start.bin')
end = loadBin('/raid/guorui/DG/dataset/BITCOIN/simple_end.bin')
IDs = loadBin('/raid/guorui/DG/dataset/BITCOIN/simple_IDs.bin')