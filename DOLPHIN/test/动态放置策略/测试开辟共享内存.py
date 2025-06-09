import torch
import time

for i in range(10):
    start_t = time.time()
    test = torch.empty((100000000, 10), dtype = torch.int32)
    print(f"开辟 {test.numel() * 4 / 1024**3}GB共享内存 用时{time.time() - start_t:.4f}s")

from multiprocessing import shared_memory

shm_a = shared_memory.SharedMemory(create=True, size=100000000 * 8)


shm_b = shared_memory.SharedMemory(shm_a.name)
import numpy as np
a = np.array([1, 1, 2, 3, 5])  # Start with an existing NumPy array
b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_b.buf)

for i in range(10):
    start_t = time.time()
    test = torch.empty(100000000, dtype = torch.int32)
    # test = test.numpy()

    b = torch.arange(test.shape[0], dtype = test.dtype)
    b[:] = test[:]
    print(f"赋值 用时{time.time() - start_t:.4f}s")


for i in range(10):
    start_t = time.time()
    test = torch.empty((100000000, 1), dtype = torch.int32)
    print(f"开辟 {test.numel() * 4 / 1024**3}GB tensor 向量 用时{time.time() - start_t:.4f}s")
    shm_a = shared_memory.SharedMemory(create=True, size=test.numel() * 8)
    print(f"开辟共享内存 用时{time.time() - start_t:.4f}s")
    test = test.numpy()
    print(f"转为numpy数组 用时{time.time() - start_t:.4f}s")
    b = np.ndarray(test.shape, dtype=np.int32, buffer=shm_a.buf)
    print(f"创建对buf的映射 用时{time.time() - start_t:.4f}s")

    b[:] = test[:]
    print(f"赋值 用时{time.time() - start_t:.4f}s")

    print(f"将开辟的张量赋值给共享内存: {time.time() - start_t}")