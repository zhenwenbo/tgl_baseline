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
test_tensor = torch.arange(1000000000, dtype = torch.int32)

def saveBin(tensor,savePath,addSave=False):
    global saveTime
    if addSave :
        with open(savePath, 'ab') as f:
            if isinstance(tensor, torch.Tensor):
                tensor.numpy().tofile(f)
            elif isinstance(tensor, np.ndarray):
                tensor.tofile(f)
    else:
        if isinstance(tensor, torch.Tensor):
            tensor.numpy().tofile(savePath)
        elif isinstance(tensor, np.ndarray):
            tensor.tofile(savePath)

import time
for i in range(100):
    start = time.time()
    torch.save(test_tensor, './test.pt')
    print(f"torch存储 {time.time() - start:.8f}s")
    start = time.time()
    saveBin(test_tensor, './test.bin')
    print(f"np二进制加载 {time.time() - start:.8f}s")

