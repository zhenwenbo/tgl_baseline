import numpy as np
import time

# 设置文件大小为1GB
file_size_gb = 5
file_size_bytes = file_size_gb * 1024 * 1024 * 1024
import gc

data = np.random.rand(file_size_bytes // 4).astype(np.int32)  # 使用int32类型，每个元素4字节

ts = time.time()

# 写入文件
for i in range(100):
    file_path = f'./test_write_{i}.bin'
    data = np.random.rand(file_size_bytes // 4).astype(np.int32)  # 使用int32类型，每个元素4字节
    write_start_time = time.time()
    data.tofile(file_path)
    write_end_time = time.time()
    write_time = write_end_time - write_start_time
    print(f"Write time: {write_time:.2f} seconds")
    write_speed = (file_size_gb / write_time) * 1024  # 转换为MB/s
    print(f"Write speed: {write_speed:.2f} MB/s")
    gc.collect()


# 清理生成的文件



for i in range(10):
    file_path = f'./test_write_{i}.bin'
    read_start_time = time.time()
    read_data = np.fromfile(file_path, dtype=np.int32)
    read_end_time = time.time()
    read_time = read_end_time - read_start_time
    print(f"Read time: {read_time:.2f} seconds")
    read_speed = (file_size_gb / read_time) * 1024  # 转换为MB/s
    print(f"secend Read speed: {read_speed:.2f} MB/s")
