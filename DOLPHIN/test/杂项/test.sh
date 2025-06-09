#!/bin/bash

# 定义输入文件路径
INPUT_FILE="/home/guorui/workspace/dgnn/b-tgl/res.log" # 请将 input.txt 替换为实际文件名

# 提取并计算总的读取时间
total_read_time=$(grep "读取disk用时" "$INPUT_FILE" | awk '{sum += $2} END {print sum}')

# 提取并计算总的写入时间
total_write_time=$(grep "写入记忆disk用时" "$INPUT_FILE" | awk '{sum += $2} END {print sum}')

# 检查是否计算成功
if [[ -z "$total_read_time" || -z "$total_write_time" ]]; then
    echo "计算总时间失败，请检查输入文件的格式！"
    exit 1
fi

# 将结果追加到文件
{
    echo "总读取disk用时: ${total_read_time}s"
    echo "总写入disk用时: ${total_write_time}s"
} >> "$INPUT_FILE"

# 输出成功信息
echo "总时间已成功计算并追加到文件末尾！"
