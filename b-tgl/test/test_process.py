import torch
import time
from test_prefetch import *
import torch.multiprocessing as mp

# def worker_function(tensor, pipe):
#     # 这是子进程中运行的函数
#     print(f"子进程测试是否会增加内存")
#     # time.sleep(10000)
#     tensor = tensor.cpu()
#     print(tensor)
#     print(tensor.shape)
#     print(tensor.device)

#     # tensor[:] = 0
#     res_tensor = tensor[:100000]
#     pipe.send(res_tensor) 
#     # pipe.close() 

if __name__ == '__main__':
    print(123123)
    pre = Pre_fetch()
    pre.real_init()
    pre.run_pre_fetch()
