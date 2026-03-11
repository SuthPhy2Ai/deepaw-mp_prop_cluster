#!/bin/bash
# Monitor Stage B training progress

LOG_FILE="experiments/stage_b/phase3_baseline/exp101_baseline_graph/training_log.txt"
PID=3782650

echo "=== Stage B Training Monitor ==="
echo "Log file: $LOG_FILE"
echo "Process PID: $PID"
echo "Started at: $(date)"
echo ""

# Check if process is running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Training process is not running!"
    exit 1
fi

echo "✅ Training process is running"
echo ""

# Wait for log file to be created
while [ ! -f "$LOG_FILE" ]; do
    echo "Waiting for log file to be created..."
    sleep 5
done

echo "📝 Log file created"
echo ""

# Monitor initialization
echo "Monitoring sampler initialization..."
echo "(This will take ~47 minutes on first run)"
echo ""

LAST_LINE=""
while true; do
    # Check if process is still running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "❌ Training process stopped!"
        echo "Last 50 lines of log:"
        tail -50 "$LOG_FILE"
        exit 1
    fi

    # Get last line with progress
    CURRENT_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)

    if [ "$CURRENT_LINE" != "$LAST_LINE" ]; then
        # Check for key milestones
        if echo "$CURRENT_LINE" | grep -q "Epoch"; then
            echo ""
            echo "🎉 Training started!"
            echo "$CURRENT_LINE"
            echo ""
            echo "Monitor with: tail -f $LOG_FILE"
            exit 0
        elif echo "$CURRENT_LINE" | grep -q "Processed.*samples"; then
            echo "$CURRENT_LINE"
        elif echo "$CURRENT_LINE" | grep -q "Found.*elastic"; then
            echo "$CURRENT_LINE"
        elif echo "$CURRENT_LINE" | grep -q "Loaded.*cached"; then
            echo "✅ $CURRENT_LINE"
            echo "Training should start soon..."
        fi
        LAST_LINE="$CURRENT_LINE"
    fi

    sleep 10
done
