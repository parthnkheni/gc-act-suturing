#!/bin/bash
set -o pipefail

# OOD KT Training  -- GC-ACT fine-tuned from EXISTING v2 checkpoint
# (v2 backbone trained on all 10 tissues  -- not fully OOD, but GC-ACT layers are fresh)
# Tissue 6 = held-out TEST, Tissue 10 = VAL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU. Aborting."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    exit 1
fi

# Use existing v2 KT checkpoint (trained on all 10 tissues)
V2_CKPT="${CKPT_BASE}/act_kt_v2/policy_best.ckpt"
if [ ! -f "$V2_CKPT" ]; then
    V2_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$V2_CKPT" ] || [ ! -f "$V2_CKPT" ]; then
    echo "ERROR: No v2 KT checkpoint found at ${CKPT_BASE}/act_kt_v2/"
    exit 1
fi

echo "  OOD GC-ACT KT  -- Fine-tune from existing v2"
echo "  Base: $V2_CKPT"
echo "  Train: tissues 2,3,4,5,7,8,9 (829 episodes incl recovery)"
echo "  Val: tissue 10 (4 episodes)"
echo "  Test: tissue 6 (51 episodes)  -- NEVER TOUCHED"
echo "  Epochs: 1000, LR: 1e-4, bs=32x8=256"
echo ""

# Write start timestamp for monitoring
echo "$(date +%s)" > /tmp/ood_kt_train_start

mkdir -p ${CKPT_BASE}/act_kt_gcact_ood

echo "[$(date)] Starting GC-ACT KT OOD fine-tuning..."
python imitate_episodes.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_gcact_ood \
    --resume_ckpt "$V2_CKPT" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 1e-4 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 1000 \
    --seed 0 \
    --use_amp \
    --use_gesture \
    --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_kt_gcact_ood.log

EXIT_CODE=$?
echo "[$(date)] GC-ACT KT OOD training DONE (exit code: $EXIT_CODE)"

echo ""
echo "  OOD KT TRAINING COMPLETE"
echo "  Checkpoint: ${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
