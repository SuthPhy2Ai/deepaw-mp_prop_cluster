#!/bin/bash
# 自动检查训练进度，每5分钟一次

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

    # 检查训练进程
    if ps aux | grep -q "[t]rain_multitask.py"; then
        echo "训练进程运行中"
        ps aux | grep "[t]rain_multitask.py" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $6/1024"MB", "TIME:", $10}'
    else
        echo "训练进程已停止"
    fi

    # 检查GPU
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "GPU: ${GPU_UTIL}% 使用率, ${GPU_MEM}MB 显存"

    # 检查运行目录
    LATEST_RUN=$(ls -t artifacts/runs/ 2>/dev/null | head -1)
    if [ ! -z "$LATEST_RUN" ] && [ "$LATEST_RUN" != "20260303_194001" ]; then
        echo "新运行目录: $LATEST_RUN"

        # 检查训练进度
        if [ -d "artifacts/runs/$LATEST_RUN/metrics" ]; then
            EPOCH_COUNT=$(ls artifacts/runs/$LATEST_RUN/metrics/epoch_*.json 2>/dev/null | wc -l)
            echo "  已完成 epoch: $EPOCH_COUNT"

            if [ -f "artifacts/runs/$LATEST_RUN/metrics/best_summary.json" ]; then
                echo "  训练已完成！"
                exit 0
            fi
        fi
    else
        echo "运行目录: 未创建或仍在初始化"
    fi

    echo ""
    sleep 300  # 5分钟
done
