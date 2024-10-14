#!/bin/bash

while true; do
    # 查找名为 python 或 python3 的进程
    if ! pgrep -x "python" -o || ! pgrep -x "python3" -o; then
        echo "没有发现正在运行的Python程序，退出循环。"
        break
    else
        echo "检测到Python程序正在运行。"
    fi
    # 每次检查后等待60秒
    sleep 60
done


