#!/bin/bash
# EXP-02监控脚本

LOG_FILE="logs/exp02_graph_baseline.log"
STATUS_FILE="reports/phase1_execution_log.md"
RUN_DIR_PATTERN="artifacts/runs/202603*"

echo "=== EXP-02监控启动 ===" >> logs/exp02_monitor.log
echo "开始时间: $(date)" >> logs/exp02_monitor.log

# 等待数据加载完成（新运行目录创建）
echo "等待EXP-02数据加载完成..." >> logs/exp02_monitor.log

LAST_RUN_COUNT=$(ls -d $RUN_DIR_PATTERN 2>/dev/null | wc -l)

while true; do
    CURRENT_RUN_COUNT=$(ls -d $RUN_DIR_PATTERN 2>/dev/null | wc -l)

    if [ "$CURRENT_RUN_COUNT" -gt "$LAST_RUN_COUNT" ]; then
        TIMESTAMP=$(date '+%H:%M')
        NEW_RUN=$(ls -td $RUN_DIR_PATTERN 2>/dev/null | head -1)

        echo "[$TIMESTAMP] ✅ EXP-02数据加载完成，训练开始" >> logs/exp02_monitor.log
        echo "[$TIMESTAMP] 新运行目录: $NEW_RUN" >> logs/exp02_monitor.log

        # 更新状态文件
        echo "" >> $STATUS_FILE
        echo "**$TIMESTAMP** - ✅ EXP-02数据加载完成，训练开始" >> $STATUS_FILE
        echo "- 运行目录: $(basename $NEW_RUN)" >> $STATUS_FILE

        # 获取GPU状态
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        echo "- GPU利用率: ${GPU_UTIL}%" >> $STATUS_FILE

        break
    fi

    sleep 60
done

# 监控训练进度
LAST_EPOCH=0
RUN_DIR=$(ls -td $RUN_DIR_PATTERN 2>/dev/null | head -1)

while true; do
    # 检查进程是否还在运行
    if ! ps aux | grep -q "[t]rain_multitask.py.*2545058"; then
        echo "[$(date '+%H:%M')] ⚠️  训练进程未检测到" >> logs/exp02_monitor.log
        sleep 60
        continue
    fi

    # 从日志中提取最新的epoch信息
    LATEST_EPOCH_LINE=$(grep -E "^epoch=[0-9]+" $LOG_FILE 2>/dev/null | tail -1)

    if [ ! -z "$LATEST_EPOCH_LINE" ]; then
        CURRENT_EPOCH=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'epoch=\K[0-9]+')
        TRAIN_LOSS=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'train_loss=\K[0-9.]+')
        VAL_LOSS=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'val_loss=\K[0-9.]+')

        # 如果有新的epoch完成，每5个epoch更新一次
        if [ "$CURRENT_EPOCH" -gt "$LAST_EPOCH" ] && [ $((CURRENT_EPOCH % 5)) -eq 0 ]; then
            TIMESTAMP=$(date '+%H:%M')
            PROGRESS=$((CURRENT_EPOCH * 100 / 50))

            echo "[$TIMESTAMP] Epoch $CURRENT_EPOCH/50 完成 ($PROGRESS%)" >> logs/exp02_monitor.log

            echo "" >> $STATUS_FILE
            echo "**$TIMESTAMP** - EXP-02 Epoch $CURRENT_EPOCH/50 ($PROGRESS%)" >> $STATUS_FILE
            echo "- Train Loss: $TRAIN_LOSS" >> $STATUS_FILE
            echo "- Val Loss: $VAL_LOSS" >> $STATUS_FILE

            # Gate-1检查（20 epochs）
            if [ "$CURRENT_EPOCH" -eq 20 ]; then
                echo "- 🚦 Gate-1检查点" >> $STATUS_FILE
            fi

            LAST_EPOCH=$CURRENT_EPOCH
        fi
    fi

    # 检查是否完成
    if [ -f "$RUN_DIR/metrics/best_summary.json" ]; then
        TIMESTAMP=$(date '+%H:%M')
        echo "[$TIMESTAMP] ✅ EXP-02训练完成！" >> logs/exp02_monitor.log

        echo "" >> $STATUS_FILE
        echo "**$TIMESTAMP** - ✅ EXP-02训练完成！" >> $STATUS_FILE

        break
    fi

    sleep 120
done

echo "EXP-02监控结束" >> logs/exp02_monitor.log
