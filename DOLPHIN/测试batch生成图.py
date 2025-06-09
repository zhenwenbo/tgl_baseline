import numpy as np
import torch
import matplotlib.pyplot as plt


ac_log = []



def paint(ac_log):
    # 获取总的 batch 数量
    n_batches = len(ac_log)

    # 用于绘制的坐标
    x = []  # 存储所有预测错误的样本索引
    y = []  # 存储对应的 batch 索引

    # 遍历 ac_log 来构建 x 和 y
    for batch_idx, indices in enumerate(ac_log):
        x.extend(indices)
        y.extend([batch_idx] * len(indices))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y,s=10, alpha= 0.5, edgecolor='none')
    plt.xlabel('Sample Index within Batch')
    plt.ylabel('Batch Number')
    plt.title('Prediction Errors per Batch')
    plt.grid(True)
    plt.savefig(f'./test_model/test.png')



for i in range(100):
    cur = torch.randperm(4000, dtype=torch.int32)[:150]
    ac_log.append(cur.tolist())

paint(ac_log)


import torch
pred = torch.tensor([0.3,0.8,0.62,0.4,0.6,0.6,0.6])
pred, ind = torch.sort(pred, descending=True)

res = ind[torch.nonzero((ind[ind.shape[0] // 2:] <= ind.shape[0] // 2)) .reshape(-1)+ ind.shape[0] // 2]
print(res)




import numpy as np
import torch
acc_log = []
for i in range(100):
    cur = torch.randperm(4000, dtype=torch.int32)[:150]
    acc_log.append(cur.tolist())


# 计算 batch 数
num_batches = len(acc_log)

# 收集所有的索引
all_indices = [index for batch_indices in acc_log for index in batch_indices]
min_index = min(all_indices)
max_index = max(all_indices)

# 设置分段数量
num_bins = 50
x_bins = np.linspace(min_index, max_index, num_bins + 1)
y_bins = np.linspace(0, num_batches, num_batches + 1)
# 扁平化 acc_log 以获得 y 轴的对应 batch 数
y_values = []
x_values = []

for batch_index, indices in enumerate(acc_log):
    y_values.extend([batch_index] * len(indices))  # 每个错误的索引对应当前 batch
    x_values.extend(indices)  # 将错误的索引添加到 x_values

# 生成热力图数据
heatmap, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins])
import matplotlib.pyplot as plt

# 绘制热力图
plt.figure(figsize=(12, 6))
plt.imshow(heatmap.T, origin='lower', aspect='auto', cmap='hot', 
           extent=[min_index, max_index, 0, num_batches])
plt.colorbar(label='Frequency')
plt.xlabel('Index')
plt.ylabel('Batch Number')
plt.title('Heatmap of Prediction Errors by Batch and Index')
plt.savefig(f'./test.png')



import torch

# 假设 Q 和 V 已经定义并具有相同的维度
Q = torch.randn(500, 2, 50)  # 示例数据
K = torch.randn(1000, 2, 50)  # 示例数据
Q = torch.cat((Q,Q), dim = 0)
Q = Q[torch.randperm(Q.shape[0], dtype = torch.int64)]

Q_uni, ind = torch.unique(Q, return_inverse = True, dim = 0)

# 第一步：去除重复的 Q
unique_Q, inverse_indices = torch.unique(Q, dim=0, return_inverse=True)

# 假设我们需要对每个 unique_Q 进行与 V 的乘法
result = torch.matmul(unique_Q, V.transpose(1, 2))  # 矩阵乘法

final_result = result[inverse_indices]


import torch

Q = torch.randn(2,2)  # 示例数据
K = torch.randn(2,2)  # 示例数据
