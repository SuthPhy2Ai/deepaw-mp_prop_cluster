#!/bin/bash
# 智能监控脚本 - 在关键时刻更新状态

RUN_DIR="artifacts/runs/20260303_211013"
STATUS_FILE="reports/phase1_execution_log.md"

LAST_EPOCH=0

while true; do
    # 检查进程是否还在运行
    if ! ps aux | grep -q "[t]rain_multitask.py.*2487905"; then
        echo "[$(date '+%H:%M')] ❌ 训练进程已停止" >> $STATUS_FILE
        break
    fi

    # 检查epoch数
    if [ -d "$RUN_DIR/metrics" ]; then
        CURRENT_EPOCH=$(ls $RUN_DIR/metrics/epoch_*.json 2>/dev/null | wc -l)

        # 如果有新的epoch完成
        if [ "$CURRENT_EPOCH" -gt "$LAST_EPOCH" ]; then
            TIMESTAMP=$(date '+%H:%M')

            # 读取最新epoch的指标
            LATEST_FILE=$(ls -t $RUN_DIR/metrics/epoch_*.json | head -1)

            if [ -f "$LATEST_FILE" ]; then
                TRAIN_LOSS=$(cat $LATEST_FILE | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d.get('train_loss', 0):.4f}\")" 2>/dev/null)
                VAL_LOSS=$(cat $LATEST_FILE | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d.get('val_loss', 0):.4f}\")" 2>/dev/null)

                # 更新状态文件
                echo "" >> $STATUS_FILE
                echo "**$TIMESTAMP** - Epoch $CURRENT_EPOCH/50 完成" >> $STATUS_FILE
                echo "- Train Loss: $TRAIN_LOSS" >> $STATUS_FILE
                echo "- Val Loss: $VAL_LOSS" >> $STATUS_FILE

                # 特殊里程碑
                if [ "$CURRENT_EPOCH" -eq 1 ]; then
                    echo "- 🎉 第一个epoch完成！" >> $STATUS_FILE
                elif [ "$CURRENT_EPOCH" -eq 10 ]; then
                    echo "- 📊 已完成20%训练" >> $STATUS_FILE
                elif [ "$CURRENT_EPOCH" -eq 25 ]; then
                    echo "- 📊 已完成50%训练" >> $STATUS_FILE
                elif [ "$CURRENT_EPOCH" -eq 40 ]; then
                    echo "- 📊 已完成80%训练" >> $STATUS_FILE
                fi
            fi

            LAST_EPOCH=$CURRENT_EPOCH
        fi

        # 检查是否完成
        if [ -f "$RUN_DIR/metrics/best_summary.json" ]; then
            TIMESTAMP=$(date '+%H:%M')
            echo "" >> $STATUS_FILE
            echo "**$TIMESTAMP** - ✅ EXP-01训练完成！" >> $STATUS_FILE

            # 读取最佳指标
            BEST_EPOCH=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('best_epoch', 'N/A'))" 2>/dev/null)
            BEST_VAL_LOSS=$(cat $RUN_DIR/metrics/best_summary.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d.get('best_val_loss', 0):.4f}\")" 2>/dev/null)

            echo "- 最佳Epoch: $BEST_EPOCH" >> $STATUS_FILE
            echo "- 最佳Val Loss: $BEST_VAL_LOSS" >> $STATUS_FILE

            break
        fi
    fi

    # 每分钟检查一次
    sleep 60
done

echo "智能监控脚本结束"
