import torch
import time


test_tensor = torch.randint(0, 100000000, (1000000,), dtype = torch.int32, device = 'cuda:0')
sort_res,_ = torch.sort(test_tensor)

sort_time = 0
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    sort_res,_ = torch.sort(test_tensor)
    torch.cuda.synchronize()
    # print(f"sort time: {time.time() - start:.6f}s")
    sort_time += time.time() - start

print(f"100次sort {sort_time:.6f}s")

torch.cuda.synchronize()
uni_time = 0
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    uni_res,inv = torch.unique(test_tensor, return_inverse = True)
    torch.cuda.synchronize()
    # print(f"unique time: {time.time() - start:.6f}s")
    uni_time += time.time() - start

print(f"100次unique {uni_time:.6f}s")