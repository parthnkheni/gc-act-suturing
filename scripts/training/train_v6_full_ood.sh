#!/bin/bash
set -o pipefail

# V6: Full OOD Training Pipeline (v2 from SCRATCH, not fine-tuned from all-tissue v2)
# Phase 1a: v2 KT OOD from scratch (3000 epochs)
# Phase 1b: GC-ACT KT OOD fine-tuned from v2 OOD (1000 epochs)
# Phase 2a: v2 NT OOD from scratch (3000 epochs)
# Phase 2b: GC-ACT NT OOD fine-tuned from v2 OOD (1000 epochs)
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

echo "$(date +%s)" > /tmp/v6_train_start

echo "  V6: Full OOD Training  -- v2 from scratch + GC-ACT"
echo "  Train: tissues 2,3,4,5,7,8,9"
echo "  Val: tissue 10"
echo "  Test: tissue 6  -- NEVER TOUCHED"
echo "  Estimated: ~36hr total (KT ~18hr + NT ~18hr)"
echo ""

# KT

# Phase 1a: v2 KT OOD from scratch
echo "[$(date)] PHASE 1a: v2 KT OOD from scratch (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_kt_v2_ood_scratch

python imitate_episodes.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_v2_ood_scratch \
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
    2>&1 | tee ${LOG_DIR}/train_v6_kt_v2_ood.log

V2_KT_EXIT=$?
echo "[$(date)] Phase 1a DONE (exit code: $V2_KT_EXIT)"

if [ $V2_KT_EXIT -ne 0 ]; then
    echo "ERROR: v2 KT OOD training failed. Stopping."
    exit 1
fi

# Phase 1b: GC-ACT KT OOD fine-tuned from v2 OOD scratch
V2_KT_CKPT="${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_KT_CKPT" ]; then
    V2_KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$V2_KT_CKPT" ] || [ ! -f "$V2_KT_CKPT" ]; then
    echo "ERROR: No v2 KT OOD checkpoint found. Skipping GC-ACT KT."
else
    echo ""
    echo "[$(date)] PHASE 1b: GC-ACT KT OOD fine-tuning (1000 epochs) from $V2_KT_CKPT..."
    mkdir -p ${CKPT_BASE}/act_kt_gcact_ood_scratch

    python imitate_episodes.py \
        --task_name knot_tying_ood \
        --ckpt_dir ${CKPT_BASE}/act_kt_gcact_ood_scratch \
        --resume_ckpt "$V2_KT_CKPT" \
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
        2>&1 | tee ${LOG_DIR}/train_v6_kt_gcact_ood.log

    GCACT_KT_EXIT=$?
    echo "[$(date)] Phase 1b DONE (exit code: $GCACT_KT_EXIT)"
fi

# NT

# Phase 2a: v2 NT OOD from scratch
echo ""
echo "[$(date)] PHASE 2a: v2 NT OOD from scratch (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_nt_v2_ood_scratch

python imitate_episodes.py \
    --task_name needle_throw_ood \
    --ckpt_dir ${CKPT_BASE}/act_nt_v2_ood_scratch \
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
    2>&1 | tee ${LOG_DIR}/train_v6_nt_v2_ood.log

V2_NT_EXIT=$?
echo "[$(date)] Phase 2a DONE (exit code: $V2_NT_EXIT)"

if [ $V2_NT_EXIT -ne 0 ]; then
    echo "ERROR: v2 NT OOD training failed. Skipping GC-ACT NT."
else
    # Phase 2b: GC-ACT NT OOD fine-tuned from v2 OOD scratch
    V2_NT_CKPT="${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best.ckpt"
    if [ ! -f "$V2_NT_CKPT" ]; then
        V2_NT_CKPT=$(ls -t ${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
    fi
    if [ -z "$V2_NT_CKPT" ] || [ ! -f "$V2_NT_CKPT" ]; then
        echo "ERROR: No v2 NT OOD checkpoint found. Skipping GC-ACT NT."
    else
        echo ""
        echo "[$(date)] PHASE 2b: GC-ACT NT OOD fine-tuning (1000 epochs) from $V2_NT_CKPT..."
        mkdir -p ${CKPT_BASE}/act_nt_gcact_ood_scratch

        python imitate_episodes.py \
            --task_name needle_throw_ood \
            --ckpt_dir ${CKPT_BASE}/act_nt_gcact_ood_scratch \
            --resume_ckpt "$V2_NT_CKPT" \
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
            2>&1 | tee ${LOG_DIR}/train_v6_nt_gcact_ood.log

        GCACT_NT_EXIT=$?
        echo "[$(date)] Phase 2b DONE (exit code: $GCACT_NT_EXIT)"
    fi
fi

echo ""
echo "  V6 FULL OOD TRAINING COMPLETE"
echo "  KT v2 OOD:     ${CKPT_BASE}/act_kt_v2_ood_scratch/"
echo "  KT GC-ACT OOD: ${CKPT_BASE}/act_kt_gcact_ood_scratch/"
echo "  NT v2 OOD:     ${CKPT_BASE}/act_nt_v2_ood_scratch/"
echo "  NT GC-ACT OOD: ${CKPT_BASE}/act_nt_gcact_ood_scratch/"
