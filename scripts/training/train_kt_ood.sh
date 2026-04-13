#!/bin/bash
set -o pipefail

# OOD KT Training Pipeline:
# Phase 1: v2 KT from scratch (3000 epochs, EfficientNet-B3)
# Phase 2: GC-ACT KT fine-tuned from v2 OOD (1000 epochs)
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

echo "  OOD KT Training  -- Tissue 6 held out"
echo "  Train: tissues 2,3,4,5,7,8,9 (689 episodes)"
echo "  Val: tissue 10 (2 episodes)"
echo "  Test: tissue 6 (51 episodes)  -- NEVER TOUCHED"
echo ""

# Write start timestamp for monitoring
echo "$(date +%s)" > /tmp/ood_kt_train_start

# Phase 1: v2 KT OOD from scratch
echo "[$(date)] PHASE 1: v2 KT OOD training (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_kt_v2_ood

python imitate_episodes.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_v2_ood \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 5e-4 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 3000 \
    --seed 0 \
    --use_amp \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_kt_v2_ood.log

V2_EXIT=$?
echo "[$(date)] Phase 1 DONE (exit code: $V2_EXIT)"

if [ $V2_EXIT -ne 0 ]; then
    echo "ERROR: v2 KT OOD training failed. Stopping."
    exit 1
fi

# Phase 2: GC-ACT KT OOD fine-tuned from v2 OOD
V2_CKPT="${CKPT_BASE}/act_kt_v2_ood/policy_best.ckpt"
if [ ! -f "$V2_CKPT" ]; then
    V2_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2_ood/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$V2_CKPT" ] || [ ! -f "$V2_CKPT" ]; then
    echo "ERROR: No v2 OOD checkpoint found. Cannot proceed to Phase 2."
    exit 1
fi

echo ""
echo "[$(date)] PHASE 2: GC-ACT KT OOD fine-tuning (1000 epochs) from $V2_CKPT..."
mkdir -p ${CKPT_BASE}/act_kt_gcact_ood

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

GCACT_EXIT=$?
echo "[$(date)] Phase 2 DONE (exit code: $GCACT_EXIT)"

echo ""
echo "  OOD KT TRAINING COMPLETE"
echo "  v2 OOD:    ${CKPT_BASE}/act_kt_v2_ood/policy_best.ckpt"
echo "  GC-ACT OOD: ${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
