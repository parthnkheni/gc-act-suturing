#!/bin/bash
set -o pipefail  # capture Python exit codes through tee pipes

# GC-ACT Training: Fine-tune v2 checkpoints with gesture conditioning
# Only NT and KT (no NP labels exist  -- NP already achieves 9/10)
# Lower LR for fine-tuning: 1e-4 (vs 5e-4 from scratch)
# Fewer epochs: 1000 (vs 3000)
# Resume from v2 checkpoints with strict=False (new gesture layers init random)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

COMMON_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --lr 1e-4 --batch_size 32 --grad_accum_steps 8 --image_encoder efficientnet_b3 --num_epochs 1000 --seed 0 --use_amp --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels"

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    echo ""
    echo "GC-ACT needs the full GPU. Kill other processes first or wait."
    exit 1
fi

echo "  GC-ACT Training  -- Gesture-Conditioned ACT (Tier 1)"
echo "  Config: bs=32x8accum(=256), lr=1e-4, kl=10, EfficientNet-B3, AMP"
echo "  Epochs: 1000 per subtask (fine-tuning from v2)"
echo "  Gesture: 10-class one-hot conditioning"
echo "  Mode:   Sequential (NT then KT)"
echo ""

# Needle Throw
echo "[$(date)] Starting NEEDLE THROW GC-ACT training..."
python imitate_episodes.py \
    --task_name needle_throw_all \
    --ckpt_dir ${CKPT_BASE}/act_nt_gcact \
    --resume_ckpt ${CKPT_BASE}/act_nt_v2/policy_best.ckpt \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_nt_gcact.log
NT_EXIT=$?
echo "[$(date)] NT GC-ACT training DONE (exit code: $NT_EXIT)"
echo ""

if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: NT GC-ACT training failed. Stopping."
    exit 1
fi

# Knot Tying
# Find the best KT v2 checkpoint dynamically
KT_V2_CKPT="${CKPT_BASE}/act_kt_v2/policy_best.ckpt"
if [ ! -f "$KT_V2_CKPT" ]; then
    # Fall back to latest policy_best_epoch_*.ckpt
    KT_V2_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$KT_V2_CKPT" ] || [ ! -f "$KT_V2_CKPT" ]; then
    echo "ERROR: No KT v2 checkpoint found. Run v2 training first."
    exit 1
fi
echo "Using KT v2 checkpoint: $KT_V2_CKPT"

echo "[$(date)] Starting KNOT TYING GC-ACT training..."
python imitate_episodes.py \
    --task_name knot_tying_all \
    --ckpt_dir ${CKPT_BASE}/act_kt_gcact \
    --resume_ckpt "$KT_V2_CKPT" \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_kt_gcact.log
KT_EXIT=$?
echo "[$(date)] KT GC-ACT training DONE (exit code: $KT_EXIT)"
echo ""

echo "  GC-ACT TRAINING COMPLETE"
echo "  Checkpoints:"
echo "    NT: ${CKPT_BASE}/act_nt_gcact/policy_best.ckpt"
echo "    KT: ${CKPT_BASE}/act_kt_gcact/policy_best.ckpt"
echo "  NP: Use v2 checkpoint (no gesture labels for NP)"
