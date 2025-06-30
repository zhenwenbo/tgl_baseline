import pandas as pd
import os
from tqdm import tqdm
import argparse
import sys


import pandas as pd

def process_data_with_labels(input_file, output_file):
    """高效处理数据并添加标签"""
    # 读取原始数据
    df = pd.read_csv(input_file)
    total_rows = len(df)
    if total_rows == 0:
        print("输入文件为空，不执行任何操作")
        return
    
    # 计算start和end列，然后合并为time列（移除.0后缀）
    df_with_labels = df.copy().assign(
        start=(df['time'] - 1).clip(lower=1).astype(int),
        end=(df['time'] + 1).astype(int),
        time=lambda x: x.apply(lambda row: f"{int(row['start'])}_{int(row['end'])}", axis=1)#目前是字符串
    ).drop(columns=['start', 'end'])  # 移除临时列
    time_parts = df_with_labels['time'].str.split('_', expand=True)
    df_with_labels['time'] = time_parts[0] + time_parts[1]
    final_df = df_with_labels
    
    # 保存结果
    final_df.to_csv(output_file, index=False)
    
    print(f"成功处理数据并保存到 {output_file}")
    print(f"原文件行数: {total_rows}, 新文件行数: {len(final_df)}")
        
if __name__ == "__main__":
    # 定义输入和输出文件路径
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="LASTFM" ,help='dataset name')
    args=parser.parse_args()
    dataset=args.data
    input_file = f'/raid/guorui/DG/dataset/{dataset}/edges.csv'
    output_dir = f'/raid/guorui/DG/dataset/{dataset}_interval'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'/raid/guorui/DG/dataset/{dataset}_interval/edges.csv'
    
    # 执行数据处理
    process_data_with_labels(input_file, output_file)

