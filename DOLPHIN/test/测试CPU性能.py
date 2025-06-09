
import torch
test = torch.randint(0, 1100000, (10729981,), dtype = torch.int32)

import time

for i in range(10):
    start = time.time()
    torch.sort(test)
    print(f"use {time.time() - start:.5f}s")