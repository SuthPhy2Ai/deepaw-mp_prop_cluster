#!/bin/bash
# 改进的监控脚本 - 从日志读取进度

LOG_FILE="logs/exp01_composition_baseline.log"
STATUS_FILE="reports/phase1_execution_log.md"
RUN_DIR="artifacts/runs/20260303_211013"

LAST_EPOCH=0

echo "=== 改进监控脚本启动 ===" >> logs/improved_monitor.log
echo "开始时间: $(date)" >> logs/improved_monitor.log

while true; do
    # 检查进程是否还在运行
    if ! ps aux | grep -q "[t]rain_multitask.py.*2487905"; then
        echo "[$(date '+%H:%M')] ⚠️  训练进程未检测到" >> logs/improved_monitor.log
        sleep 60
        continue
    fi

    # 从日志中提取最新的epoch信息
    LATEST_EPOCH_LINE=$(grep -E "^epoch=[0-9]+" $LOG_FILE 2>/dev/null | tail -1)

    if [ ! -z "$LATEST_EPOCH_LINE" ]; then
        # 解析epoch号和loss
        CURRENT_EPOCH=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'epoch=\K[0-9]+')
        TRAIN_LOSS=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'train_loss=\K[0-9.]+')
        VAL_LOSS=$(echo "$LATEST_EPOCH_LINE" | grep -oP 'val_loss=\K[0-9.]+')

        # 如果有新的epoch完成
        if [ "$CURRENT_EPOCH" -gt "$LAST_EPOCH" ]; then
            TIMESTAMP=$(date '+%H:%M')
            PROGRESS=$((CURRENT_EPOCH * 100 / 50))

            echo "[$TIMESTAMP] Epoch $CURRENT_EPOCH/50 完成 ($PROGRESS%)" >> logs/improved_monitor.log

            # 更新状态文件（每5个epoch更新一次）
            if [ $((CURRENT_EPOCH % 5)) -eq 0 ]; then
                echo "" >> $STATUS_FILE
                echo "**$TIMESTAMP** - Epoch $CURRENT_EPOCH/50 完成 ($PROGRESS%)" >> $STATUS_FILE
                echo "- Train Loss: $TRAIN_LOSS" >> $STATUS_FILE
                echo "- Val Loss: $VAL_LOSS" >> $STATUS_FILE

                # 特殊里程碑
                if [ "$CURRENT_EPOCH" -eq 10 ]; then
                    echo "- 📊 已完成20%训练" >> $STATUS_FILE
                elif [ "$CURRENT_EPOCH" -eq 25 ]; then
                    echo "- 📊 已完成50%训练" >> $STATUS_FILE
                elif [ "$CURRENT_EPOCH" -eq 40 ]; then
                    echo "- 📊 已完成80%训练" >> $STATUS_FILE
                fi
            fi

            LAST_EPOCH=$CURRENT_EPOCH
        fi
    fi

    # 检查是否完成（metrics文件存在）
    if [ -f "$RUN_DIR/metrics/best_summary.json" ]; then
        TIMESTAMP=$(date '+%H:%M')
        echo "[$TIMESTAMP] ✅ 训练完成！" >> logs/improved_monitor.log

        echo "" >> $STATUS_FILE
        echo "**$TIMESTAMP** - ✅ EXP-01训练完成！" >> $STATUS_FILE

        # 读取最佳指标
        BEST_EPOCH=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('best_epoch', 'N/A'))" 2>/dev/null)
        BEST_VAL_LOSS=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d.get('best_val_loss', 0):.4f}\")" 2>/dev/null)

        echo "- 最佳Epoch: $BEST_EPOCH" >> $STATUS_FILE
        echo "- 最佳Val Loss: $BEST_VAL_LOSS" >> $STATUS_FILE

        break
    fi

    # 每2分钟检查一次
    sleep 120
done

echo "改进监控脚本结束" >> logs/improved_monitor.log
