#!/bin/bash

interval=1
logMsg=""
gpu_id=0

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




ds=("BITCOIN" "GDELT")
# ds=("STACK")
models=("TGN")
bss=(500 1000)


timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "../res-${timestamp}"

for model in "${models[@]}"; do
    for d in "${ds[@]}"; do
    for bs in "${bss[@]}"; do

    echo "处理 $d"
    mkdir -p "../res-${timestamp}/${d}"


        nohup python -u /raid/guorui/workspace/dgnn/a-tgl/train.py --data=${d} --bs=${bs} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-1.yml" &>../res-${timestamp}/${d}/TGL-${model}-batchsize${bs}-1_res.log &
        pid=$!
        memory_usage_file="../res-${timestamp}/${d}/TGL-${model}-batchsize${bs}-1_res_mem.log"
        monitor_memory_usage $pid
        wait


        nohup python -u /raid/guorui/workspace/dgnn/a-tgl/train.py --data=${d} --bs=${bs} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-2.yml" &>../res-${timestamp}/${d}/TGL-${model}-batchsize${bs}-2_res.log &
        pid=$!
        memory_usage_file="../res-${timestamp}/${d}/TGL-${model}-batchsize${bs}-2_res_mem.log"
        monitor_memory_usage $pid
        wait



        # nohup python -u /raid/guorui/workspace/dgnn/b-tgl/train.py --data=${d} --bs=${bs} --train_conf='disk' --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-b-1.yml" &>../res-${timestamp}/${d}/b-${model}-batchsize${bs}-1_res.log &
        # pid=$!
        # memory_usage_file="../res-${timestamp}/${d}/b-${model}-batchsize${bs}-1_res_mem.log"
        # monitor_memory_usage $pid
        # wait


        threshold=0.1
        if [ "$d" == "GDELT" ]; then
            threshold=0.01
        fi
        if [ "$d" == "BITCOIN" ]; then
        threshold=0.05
        fi

        # nohup python -u /raid/guorui/workspace/dgnn/simple/main.py --threshold=${threshold} --data=${d} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-simple-1.yml" &>../res-${timestamp}/${d}/SIMPLE-${model}-1-res.log &
        # pid=$!
        # memory_usage_file="../res-${timestamp}/${d}/SIMPLE-${model}-1-res-mem.log"
        # monitor_memory_usage $pid
        # wait

        nohup python -u /raid/guorui/workspace/dgnn/ETC/train.py --data=${d} --bs=${bs} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-1.yml" &>../res-${timestamp}/${d}/ETC-${model}-batchsize${bs}-1_res.log &
        pid=$!
        memory_usage_file="../res-${timestamp}/${d}/ETC-${model}-batchsize${bs}-1_res_mem.log"
        monitor_memory_usage $pid
        wait

        
        # nohup python -u /raid/guorui/workspace/dgnn/b-tgl/train.py --data=${d} --bs=${bs} --train_conf='disk' --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-b-2.yml" &>../res-${timestamp}/${d}/b-${model}-batchsize${bs}-2_res.log &
        # pid=$!
        # memory_usage_file="../res-${timestamp}/${d}/b-${model}-batchsize${bs}-2_res_mem.log"
        # monitor_memory_usage $pid
        # wait

        # if [ "$d" != "GDELT" ]; then
        #     nohup python -u /raid/guorui/workspace/dgnn/b-tgl/train.py --data=${d} --train_conf='basic_conf_disk' --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-b-2.yml" &>../res-${timestamp}/${d}/b-${model}-2_res.log &
        #     pid=$!
        #     memory_usage_file="../res-${timestamp}/${d}/b-${model}-2_res_mem.log"
        #     monitor_memory_usage $pid
        #     wait
        # fi

        nohup python -u /raid/guorui/workspace/dgnn/ETC/train.py --bs=${bs} --data=${d} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-2.yml" &>../res-${timestamp}/${d}/ETC-${model}-batchsize${bs}-2_res.log &
        pid=$!
        memory_usage_file="../res-${timestamp}/${d}/ETC-${model}-batchsize${bs}-2_res_mem.log"
        monitor_memory_usage $pid
        wait


        

        # if [ "$d" != "GDELT" ]; then
        #     nohup python -u /raid/guorui/workspace/dgnn/simple/main.py --threshold=${threshold} --data=${d} --config="/raid/guorui/workspace/dgnn/exp/scripts/${model}-simple-2.yml" &>../res-${timestamp}/${d}/SIMPLE-${model}-2-res.log &
        #     pid=$!
        #     memory_usage_file="../res-${timestamp}/${d}/SIMPLE-${model}-2-res-mem.log"
        #     monitor_memory_usage $pid
        #     wait
        # fi


        done
        
    done
done