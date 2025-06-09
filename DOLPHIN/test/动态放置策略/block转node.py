import torch

# 原始tensor
tensor = torch.tensor([0, 0, 2, 3, 5, 6])

# 计算相邻元素的差值
differences = tensor[1:] - tensor[:-1]

# 根据差值创建索引
indices = torch.repeat_interleave(torch.arange(0, len(tensor) - 1), differences)

# 将原始tensor的第一个元素添加到结果中
result = torch.cat((tensor[:1], indices))

print(result)