#!/bin/bash
# 持续监控EXP-01训练进度

RUN_DIR="artifacts/runs/20260303_211013"
LOG_FILE="logs/exp01_progress.log"

echo "=== EXP-01训练监控启动 ===" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # 检查进程是否还在运行
    if ! ps aux | grep -q "[t]rain_multitask.py.*2487905"; then
        echo "[$TIMESTAMP] ❌ 训练进程已停止" | tee -a $LOG_FILE
        break
    fi

    # 检查epoch数
    if [ -d "$RUN_DIR/metrics" ]; then
        EPOCH_COUNT=$(ls $RUN_DIR/metrics/epoch_*.json 2>/dev/null | wc -l)

        if [ "$EPOCH_COUNT" -gt 0 ]; then
            # 读取最新epoch的指标
            LATEST_EPOCH=$(ls -t $RUN_DIR/metrics/epoch_*.json | head -1)
            TRAIN_LOSS=$(cat $LATEST_EPOCH | python3 -c "import sys, json; print(json.load(sys.stdin).get('train_loss', 'N/A'))" 2>/dev/null)
            VAL_LOSS=$(cat $LATEST_EPOCH | python3 -c "import sys, json; print(json.load(sys.stdin).get('val_loss', 'N/A'))" 2>/dev/null)

            echo "[$TIMESTAMP] Epoch $EPOCH_COUNT/50 完成 | Train Loss: $TRAIN_LOSS | Val Loss: $VAL_LOSS" | tee -a $LOG_FILE

            # 检查是否完成
            if [ -f "$RUN_DIR/metrics/best_summary.json" ]; then
                echo "[$TIMESTAMP] ✅ 训练完成！" | tee -a $LOG_FILE

                # 读取最佳指标
                BEST_EPOCH=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('best_epoch', 'N/A'))" 2>/dev/null)
                BEST_VAL_LOSS=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('best_val_loss', 'N/A'))" 2>/dev/null)

                echo "最佳Epoch: $BEST_EPOCH | 最佳Val Loss: $BEST_VAL_LOSS" | tee -a $LOG_FILE
                break
            fi
        else
            echo "[$TIMESTAMP] 训练中，等待第一个epoch完成..." | tee -a $LOG_FILE
        fi
    else
        echo "[$TIMESTAMP] 训练中，metrics目录未创建..." | tee -a $LOG_FILE
    fi

    # 检查GPU
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "[$TIMESTAMP] GPU: ${GPU_UTIL}% 利用率, ${GPU_MEM}MB 显存" | tee -a $LOG_FILE

    echo "" | tee -a $LOG_FILE

    # 每小时检查一次
    sleep 3600
done

echo "=== 监控结束 ===" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
