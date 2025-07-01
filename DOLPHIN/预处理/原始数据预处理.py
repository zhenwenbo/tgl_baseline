import pandas as pd
import torch
import numpy as np
import csv
from tqdm import tqdm
import itertools
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default = "LASTFM1")
parser.add_argument('--txt', type=str, help='txt file path', default = "/home/guorui/tgl_baseline/DOLPHIN/预处理/预处理.txt")
args=parser.parse_args()

print('open txt file...')
with open(args.txt, "r") as f:
    data = f.readlines()
for i in range(len(data)):
    data[i] = data[i].split()

print('to DataFrame...')
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(args.data)) # TODO 这里是直接读csv，如果需要自制数据集这个scv应该是不存在的，可以参考 tgl_baseline/ETC/preprocess.py 里面有不是直接read_csv的方式
# TODO 这个预处理文件是参照ETC / TGL 的原始数据预处理，最后的结果是得到一批npz和csv文件作为训练的输入，然而这种输入格式在训练过程中数据处理速度慢，因此我后面改成了将csv转为.bin格式的文件，就是你发我的load_graph_bin那边报的错，你应该没有执行这个csv -> bin的流程。这个流程是在 数据特征生成.py的df2bin函数。 注意：数据特征生成.py里我注释了一些函数，里面的df2bin是 csv -> bin必须跑，reorder_edge也必须跑，我给你把那个文件改好了
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)
print('reindex...')
#reindex src & dst
src = df.src.astype(int)
dst = df.dst.astype(int)
cut = len(src)
total = np.concatenate((src,dst))
total,inv = np.unique(total,return_inverse=True)

new_data = np.arange(len(total))
df['src'] = new_data[inv][:cut]
df['dst'] = new_data[inv][cut:]

print('sort by time...')
# sort by time
df.time =  df.time.astype(int)
df.time -= df.time.min()
df = df.sort_values(by=['time'])
df = df.reset_index(drop=True)

print('split the data...')
#data split
values = np.zeros(len(df),dtype=int)
values[int(0.7*len(df)):int(0.85*len(df))] = 1
values[int(0.85*len(df)):] = 2
df['ext_roll'] = values

print('save csv...')
df.to_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(args.data))

print('generate graph...')
#gen_graph
df = pd.read_csv('/raid/guorui/DG/dataset/{}/edges.csv'.format(args.data))

num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_train_indices = [[] for _ in range(num_nodes)]
int_train_ts = [[] for _ in range(num_nodes)]
int_train_eid = [[] for _ in range(num_nodes)]

int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_full_indices = [[] for _ in range(num_nodes)]
int_full_ts = [[] for _ in range(num_nodes)]
int_full_eid = [[] for _ in range(num_nodes)]

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]


for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')
def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]

for i in tqdm(range(int_train_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

print('saving...')

np.savez('/raid/guorui/DG/dataset/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr, indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)