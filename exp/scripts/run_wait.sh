#!/bin/bash

# 定义用户和进程名
USER="guorui"
PROCESS_NAME="python"


# 初始化检测结果变量
previous_check=false
current_check=false

# 无限循环，每隔1分钟执行一次检测
while true; do
    # 使用pgrep查找用户guorui执行的python进程
    current_check=$(pgrep -u $USER $PROCESS_NAME)
    

    if [[ -n "$current_check" ]]; then
        echo "有程序..."
    fi

    # 判断是否连续两次检测不到进程
    if [[ "$current_check" == "false" && "$previous_check" == "false" ]]; then
        echo "连续两次未检测到进程，跳出循环执行后续逻辑。"
        
        break
    fi

    # 更新上一次检测结果
    previous_check=$current_check

    # 等待1分钟
    sleep 60
done

echo "执行后续逻辑"
bash /home/guorui/workspace/dgnn/exp/scripts/pre.sh
bash /home/guorui/workspace/dgnn/exp/scripts/run_test.sh
# bash /home/guorui/workspace/dgnn/exp/scripts/run_b_tgl.sh