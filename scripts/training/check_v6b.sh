#!/bin/bash
# Quick status check for v6b training
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null

LOG="/home/exouser/logs/v6b_full.log"

if [ ! -f "$LOG" ]; then
    echo "No v6b log found."
    exit 1
fi

# Check if process is still running
PID=$(pgrep -f "train_v6b.sh" 2>/dev/null)
if [ -z "$PID" ]; then
    PID=$(pgrep -f "finetune_augmented_wrapper.*v6b" 2>/dev/null)
fi
if [ -z "$PID" ]; then
    PID=$(pgrep -f "imitate_episodes.*v6b" 2>/dev/null)
fi

if [ -n "$PID" ]; then
    echo "STATUS: RUNNING (PID: $PID)"
else
    echo "STATUS: NOT RUNNING (may have finished or crashed)"
fi

# GPU status
echo ""
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 | while read line; do
    echo "GPU: $line"
done

# Current epoch from log
echo ""
LAST_EPOCH=$(grep -oP 'Epoch \K[0-9]+' "$LOG" | tail -1)
TOTAL_EPOCHS=3500
if [ -n "$LAST_EPOCH" ]; then
    # Figure out which phase
    if grep -q "PHASE 2" "$LOG"; then
        PHASE="NT"
    else
        PHASE="KT"
    fi
    PCT=$((LAST_EPOCH * 100 / TOTAL_EPOCHS))
    echo "Phase: $PHASE | Epoch: $LAST_EPOCH / $TOTAL_EPOCHS ($PCT%)"
else
    echo "No epochs completed yet"
fi

# Latest val loss
LAST_VAL=$(grep "Val loss:" "$LOG" | tail -1)
if [ -n "$LAST_VAL" ]; then
    echo "Latest: $LAST_VAL"
fi

# Best checkpoint
BEST=$(grep "Best ckpt" "$LOG" | tail -1)
if [ -n "$BEST" ]; then
    echo "$BEST"
fi

# Last 3 loss lines
echo ""
echo "Recent losses:"
grep "^l1:" "$LOG" | tail -3

# Check for KT eval results
if [ -d "${HOME}/eval_results/v6b_ood/kt_tissue6_ensemble" ]; then
    echo ""
    echo "=== KT EVAL RESULTS AVAILABLE ==="
    cat "${HOME}/eval_results/v6b_ood/kt_tissue6_ensemble/summary.json" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k, v in d.items():
    pos = v.get('pos_l2_mean_mm', {})
    if isinstance(pos, dict):
        print(f'  KT tissue 6 ensemble: {pos[\"mean\"]:.3f}+/-{pos[\"std\"]:.3f}mm')
    else:
        print(f'  KT tissue 6 ensemble: {pos:.3f}mm')
" 2>/dev/null
fi

# Elapsed time
START_TS=$(cat /tmp/v6b_train_start 2>/dev/null)
if [ -n "$START_TS" ]; then
    NOW=$(date +%s)
    ELAPSED=$(( (NOW - START_TS) / 60 ))
    echo ""
    echo "Elapsed: ${ELAPSED} minutes"
fi
