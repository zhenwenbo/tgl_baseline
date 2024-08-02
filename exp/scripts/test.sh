#!/bin/bash
PARENT_PID=12312312

PIDS=$(ps --ppid $PARENT_PID -o pid=; echo $PARENT_PID)

TOTAL=0
for PID in $PIDS
do
    echo "处理${PID}"
    MEMORY=$(ps -p $PID -o rss=)
    echo "${MEMORY}"
    if [[ -z "$MEMORY" ]]; then
        echo "MEMORY为空，退出循环"
        break
    fi
    TOTAL=$(($TOTAL + $MEMORY))
done

memory_usage_kb=${TOTAL}
memory_usage_mb=$(($TOTAL / 1024))



echo "Total Memory Usage1: $memory_usage_mb MB"
