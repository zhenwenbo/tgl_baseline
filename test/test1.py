
import gc
def emptyCache():
    gc.collect()
    torch.cuda.empty_cache()
import time
import torch
for i in range(10000):
    cur_len = 500000000
    if (i == 0):
        i_1 = 1
        tensor_0 = torch.arange(cur_len, dtype=torch.float32, device = 'cuda:0', requires_grad=False)
    elif (i == 1):
        i_2 = 1
        tensor_1 = torch.arange(cur_len, dtype=torch.float32, device = 'cuda:0', requires_grad=False)
    elif (i == 2):
        tensor_2 = torch.arange(cur_len, dtype=torch.float32, device = 'cuda:0', requires_grad=False)
    else:
        break
    emptyCache()
    time.sleep(1)
    print(f"{cur_len}, {torch.cuda.memory_allocated() / 1024**3:.4f}GB")

print(i_1)
print(tensor_1)