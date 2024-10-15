import numpy as np
import time

# 设置文件大小为1GB
file_size_gb = 20
file_size_bytes = file_size_gb * 1024 * 1024 * 1024

# 生成一个1GB的随机数据数组
data = np.random.rand(file_size_bytes // 4).astype(np.int32)  # 使用int32类型，每个元素4字节

ts = time.time()
file_path = f'/raid/guorui/DG/test_write_{ts}.bin'

# 写入文件
write_start_time = time.time()
data.tofile(file_path)
write_end_time = time.time()
write_time = write_end_time - write_start_time
print(f"Write time: {write_time:.2f} seconds")

# 从文件读取数据
read_start_time = time.time()
read_data = np.fromfile(file_path, dtype=np.int32)
read_end_time = time.time()
read_time = read_end_time - read_start_time
print(f"Read time: {read_time:.2f} seconds")

# 清理生成的文件


# 计算写入和读取的速度（MB/s）
write_speed = (file_size_gb / write_time) * 1024  # 转换为MB/s
read_speed = (file_size_gb / read_time) * 1024  # 转换为MB/s
print(f"Write speed: {write_speed:.2f} MB/s")
print(f"Read speed: {read_speed:.2f} MB/s")


read_start_time = time.time()
read_data = np.fromfile(file_path, dtype=np.int32)
read_end_time = time.time()
read_time = read_end_time - read_start_time
print(f"Read time: {read_time:.2f} seconds")
read_speed = (file_size_gb / read_time) * 1024  # 转换为MB/s
print(f"secend Read speed: {read_speed:.2f} MB/s")

import os
os.remove(file_path)