#!/bin/bash

interval=1
logMsg=""
gpu_id=2

monitor_memory_usage() {
    local python_pid="$1"
    echo "start" > $memory_usage_file
    echo "=============================================" >> $memory_usage_file
    echo "=============================================" >> $memory_usage_file
    echo "=============================================" >> $memory_usage_file
    echo "$logMsg"  >> $memory_usage_file

    local total_memory_usage_kb=0
    local peak_memory_usage_kb=0

    local peak_gpu_usage=0

    local count=0
    while ps -p $python_pid > /dev/null; do
        PIDS=$(ps --ppid $python_pid -o pid=; echo $python_pid)

        TOTAL=0
        for PID in $PIDS
        do
            if ps -p $PID > /dev/null; then  # 检查进程是否存在
                MEMORY=$(ps -p $PID -o rss=)
                if [[ -z "$MEMORY" ]]; then
                    echo "MEMORY为空，退出循环"
                    break
                fi
                TOTAL=$(($TOTAL + $MEMORY))
            fi
        done
        # echo "ok."

        local memory_usage_kb=${TOTAL}
        local memory_usage_mb=$(($TOTAL / 1024))


        local gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)

        local current_time=$(date +'%Y-%m-%d %H:%M:%S')
        echo "$current_time, PID $python_pid , 内存占用: $memory_usage_mb MB 显存占用: $gpu_usage MB" >> $memory_usage_file

        total_memory_usage_kb=$((total_memory_usage_kb + memory_usage_kb))
        if [ $gpu_usage -gt $peak_gpu_usage ]; then
            peak_gpu_usage=$gpu_usage
        fi

        if [ $memory_usage_kb -gt $peak_memory_usage_kb ]; then
            peak_memory_usage_kb=$memory_usage_kb
        fi

        count=$((count + 1))
        sleep $interval
    done

    echo "${python_pid}" >> $memory_usage_file
    if [ $count -gt 0 ]; then
        local average_memory_usage_kb=$((total_memory_usage_kb / count))
        local average_memory_usage_mb=$(echo "scale=2; $average_memory_usage_kb / 1024" | bc)
    else
        local average_memory_usage_mb=0
    fi

    
    echo "平均内存占用: $average_memory_usage_mb MB" >> $memory_usage_file
    echo "峰值内存占用: $((peak_memory_usage_kb / 1024)) MB, $((peak_memory_usage_kb / 1024 / 1024)) GB" >> $memory_usage_file
    echo "峰值显存占用 $peak_gpu_usage MB" >> $memory_usage_file
}




ds=("LASTFM" "TALK" "STACK" "GDELT")

block_size=600000


for d in "${ds[@]}"; do

  echo "处理 $d-$block_size"


#   nohup python /raid/guorui/workspace/dgnn/b-tgl/train.py --data=${d} --pre_sample_size=${block_size} --config="/raid/guorui/workspace/dgnn/b-tgl/scripts/TGN-1.yml" >../res/${d}-1-${block_size}-res.log &
#   pid=$!
#   memory_usage_file="../res/${d}-1-${block_size}-res-mem.log"
#   monitor_memory_usage $pid
#   wait

  nohup python /raid/guorui/workspace/dgnn/a-tgl-s/train.py --data=${d} --config="/raid/guorui/workspace/dgnn/b-tgl/scripts/TGN-1.yml" >../res/TGL-${d}-1-res.log &
  pid=$!
  memory_usage_file="../res/TGL-${d}-1-res-mem.log"
  monitor_memory_usage $pid
  wait

  nohup python /raid/guorui/workspace/dgnn/simple-s/main.py --data=${d} --config="/raid/guorui/workspace/dgnn/simple/config/TGN-1.yml" >../res/SIMPLE-${d}-1-res.log &
  pid=$!
  memory_usage_file="../res/SIMPLE-${d}-1-res-mem.log"
  monitor_memory_usage $pid
  wait

#   nohup python /raid/guorui/workspace/dgnn/b-tgl/train.py --data=${d} --pre_sample_size=${block_size} --config="/raid/guorui/workspace/dgnn/b-tgl/scripts/TGN-2.yml" >../res/${d}-2-${block_size}_res.log &
#   pid=$!
#   memory_usage_file="../res/${d}-2-${block_size}_res_mem.log"
#   monitor_memory_usage $pid
#   wait

#   nohup python /raid/guorui/workspace/dgnn/a-tgl/train.py --data=${d} --config="/raid/guorui/workspace/dgnn/b-tgl/scripts/TGN-2.yml" >../res/TGL_${d}-2_res.log &
#   pid=$!
#   memory_usage_file="../res/TGL_${d}-2_res_mem.log"
#   monitor_memory_usage $pid
#   wait

done