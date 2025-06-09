#!/bin/bash

# 定义要运行的数据集列表
datasets=( "STACK")
configsets=("TGAT-1" "TGAT-2" "TimeSGN-1" "TimeSGN-2")
# 基础训练命令（根据实际路径调整）
BASE_TRAIN_CMD="python -u train.py"
timestamp1=$(date +%Y%m%d%H%M%S)
mkdir ${timestamp1}

# 循环执行每个数据集
for data in "${datasets[@]}"; do
    # 生成带时间戳的日志       
    for configs in "${configsets[@]}"; do
        timestamp=$(date +%Y%m%d%H%M%S)
        
        log_file="${timestamp1}/${data}_${configs}.log"
    
        echo "======================"
        echo "开始处理数据集: ${data}"
        echo "日志文件: ${log_file}"
        echo "======================"
    
        nohup ${BASE_TRAIN_CMD} --data="${data}" --config="/home/zwb/tgl-baseline/a-tgl-AP/config/${configs}.yml"  &> "${log_file}" &
        wait 
        echo "数据集 ${data} 处理完成！日志已保存至 ${log_file}"
        echo
    done
done
    echo "所有数据集执行完毕！"