


# 测试并行的cpu操作是否会影响到纯cuda操作

import torch
import time
import concurrent.futures
preFetchExecutor = concurrent.futures.ThreadPoolExecutor(2)  # 线程池
import numpy as np
import dgl

def worker_function(tensor, pipe):
    # 这是子进程中运行的函数
    print(f"子进程测试是否会增加内存")
    # time.sleep(10000)
    tensor = tensor.cpu()
    print(tensor)
    print(tensor.shape)
    print(tensor.device)

    # tensor[:] = 0
    res_tensor = tensor[:100000]
    pipe.send(res_tensor) 
    # pipe.close() 

def gen_block():
    src_node = torch.arange(200000, dtype = torch.int32, device = 'cuda:0')
    dst_node = torch.arange(200000, 400000 ,dtype = torch.int32, device = 'cuda:0')
    b = dgl.create_block((src_node.to(torch.int64), dst_node.to(torch.int64)), num_src_nodes = src_node.shape[0], num_dst_nodes = 400000)
    # print(b)


def cuda_handle():

    test = torch.arange(100000000, dtype = torch.int32, device = 'cuda:0')
    test1 = np.arange(100000000)
    test_list = [i for i in range(100000000)]

    while True:
        time_sort = 0
        for i in range(200):
            time_sort_s = time.time()    

            # test2 = torch.from_numpy(test1).cuda()
            # gen_block()
            test2 = torch.tensor([test_list], device = 'cuda:0', dtype = torch.int32)
            
            test_sort, _ = torch.sort(test)

            time_sort += time.time() - time_sort_s
        print(f"cuda handle: {i + 1}次 {test.shape[0]}长度的sort操作用时{time_sort:.5f}s")

def cpu_handle():
    test = torch.arange(1000000000, dtype = torch.int32)
    indices = torch.randperm(100000000, dtype = torch.int64)
    
    while True:
        time_sort = 0
        for i in range(30):
            time_sort_s = time.time()    
            test_ind = test[indices]
            time_sort += time.time() - time_sort_s
        print(f"cpu handle: 30次 {test.shape[0]}长度的索引操作用时{time_sort:.5f}s")

def cpu_io_handle():
    while True:
        time_io = 0
        for i in range(50):
            # time_io_s = time.time()
            # test = torch.load(f'/raid/guorui/DG/dataset/STACK/part-600000_[10]/part{i}_edge_feat.pt')
            test = np.fromfile('/raid/bear/data/raw/papers100M/dgl_double_PA.bin')
            # print(f"加载完毕")
            # time_io += time.time() - time_io_s
        # print(f"io handle: 50次500MB的IO加载用时{time_io:.5f}s")

# cuda_handle()
# cpu_handle()
# preFetchExecutor.submit(cpu_handle)
# preFetchExecutor.submit(cpu_io_handle)
# cuda_handle()
# cpu_io_handle()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# 假设特征维度为10
num_features = 1000
# 生成1000个样本
num_samples = 1000000
X = torch.randn(num_samples, num_features).cuda()  # 随机生成特征
# 随机生成二分类标签
y = (torch.rand(num_samples) > 0.5).float().cuda()  # 0或1

class BinaryClassifier(nn.Module):
    def __init__(self, num_features):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(num_features, 1)  # 一个线性层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数用于二分类

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)
    
# preFetchExecutor.submit(cpu_handle)
# preFetchExecutor.submit(cpu_io_handle)

import multiprocessing
pool = multiprocessing.Pool(processes=1)
pipe = multiprocessing.Pipe(duplex=False)
# pool.apply_async(cpu_io_handle)

dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = BinaryClassifier(num_features).cuda()
criterion = nn.BCELoss().to('cuda:0')  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
num_epochs = 500  # 训练5个周期
for epoch in range(num_epochs):
    start = time.time()
    for inputs, labels in data_loader:
        # print(f"inputs device: {inputs.device} labels device: {labels.device}")
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels.view(-1, 1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
    print(f'Epoch {epoch+1}, Loss: {loss.item()}  time: {time.time() - start:.4f}s')


