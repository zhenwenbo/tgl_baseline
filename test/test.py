import pandas as pd
def count_edges_per_partition(file_path):
    # 读取txt文件到DataFrame中
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['src', 'dst', 'partition'], dtype={'src': int, 'dst': int, 'partition': int})
    # 使用groupby和size方法来计算每个分区的边数
    edges_count = df.groupby('partition').size().reset_index(name='edges')

    return edges_count

# 指定文件路径
file_path = '/raid/wsy/tmp/2pscpp10/32/uk.txt'

# 调用函数并打印结果
result = count_edges_per_partition(file_path)
print(result)