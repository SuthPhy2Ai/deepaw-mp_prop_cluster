#!/bin/bash
# Monitor cache generation progress

LOG_FILE="logs/precompute_graphs.log"
PID=3397989

echo "=== Cache Generation Monitor ==="
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process is running"

    # Get latest progress line
    echo ""
    echo "Latest progress:"
    tail -1 "$LOG_FILE"

    # Extract progress percentage if available
    PROGRESS=$(tail -1 "$LOG_FILE" | grep -oP '\d+%' | head -1)
    if [ -n "$PROGRESS" ]; then
        echo ""
        echo "Progress: $PROGRESS"
    fi

    # Check cache file size
    CACHE_FILE="data/cache/graphs_cc750d893c4f189a544347615f59bd0b.pkl"
    if [ -f "$CACHE_FILE" ]; then
        SIZE=$(du -h "$CACHE_FILE" | cut -f1)
        echo "Cache file size: $SIZE"
    else
        echo "Cache file not yet created"
    fi

else
    echo "❌ Process not running"

    # Check if completed successfully
    if tail -5 "$LOG_FILE" | grep -q "Cache saved"; then
        echo "✅ Cache generation completed successfully!"

        CACHE_FILE="data/cache/graphs_cc750d893c4f189a544347615f59bd0b.pkl"
        if [ -f "$CACHE_FILE" ]; then
            SIZE=$(du -h "$CACHE_FILE" | cut -f1)
            echo "Final cache file size: $SIZE"
        fi
    else
        echo "⚠️ Process may have failed. Check log file."
    fi
fi

echo ""
echo "Full log: $LOG_FILE"
