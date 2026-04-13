#!/bin/bash
set -o pipefail

#  V5b OOD Augmented Training  -- WITH gesture flags to preserve checkpoint arch
#
#  v5a FAILED: dropping --use_gesture changed additional_pos_embed shape [3,512]
#  -> [2,512], randomly re-initializing it and destroying OOD performance.
#
#  v5b FIX: keep --use_gesture --gesture_dim 10 so checkpoint loads cleanly.
#  Gesture input has zero effect (ablation Mar 13) but architecture must match.
#
#  400 epochs (matching GC-ACT aug recipe), LR=1e-5

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

# Configurable parameters
NUM_EPOCHS=400
LR="1e-5"
BATCH_SIZE=32
GRAD_ACCUM=8

COMMON_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} --image_encoder efficientnet_b3 --num_epochs ${NUM_EPOCHS} --seed 0 --use_amp"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    exit 1
fi

echo "  V5b OOD AUGMENTED TRAINING (gesture flags preserved)"
echo "  Fix: --use_gesture kept to preserve checkpoint pos_embed"
echo "  Config: bs=${BATCH_SIZE}x${GRAD_ACCUM}accum(=$(($BATCH_SIZE * $GRAD_ACCUM))), EfficientNet-B3, AMP"
echo "  Epochs: ${NUM_EPOCHS} per subtask"
echo "  LR:     ${LR}"
echo "  Augment: AggressiveDataAug (train) / MinimalDataAug (val)"
echo "  Mode:   Sequential (KT -> NT)"
echo "  Start:  $(date)"
echo ""

#  1. KNOT TYING  -- OOD augmented
KT_SRC="${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
KT_DST="${CKPT_BASE}/act_kt_v5b_ood_aug"
if [ ! -f "$KT_SRC" ]; then
    echo "ERROR: KT v4 OOD checkpoint not found at $KT_SRC"
    exit 1
fi

echo "[$(date)] Starting KNOT TYING v5b OOD augmented training..."
echo "  Source checkpoint: $KT_SRC"
echo "  Output directory:  $KT_DST"
echo ""
python finetune_augmented_wrapper.py \
    --task_name knot_tying_ood \
    --ckpt_dir "$KT_DST" \
    --resume_ckpt "$KT_SRC" \
    --lr ${LR} \
    --gpu 0 \
    --use_gesture --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_v5b_kt_ood_aug.log
KT_EXIT=$?
echo "[$(date)] KT v5b OOD aug DONE (exit code: $KT_EXIT)"
echo ""

if [ $KT_EXIT -ne 0 ]; then
    echo "ERROR: KT training failed. Check ${LOG_DIR}/train_v5b_kt_ood_aug.log"
    exit 1
fi

#  2. NEEDLE THROW  -- OOD augmented
NT_SRC="${CKPT_BASE}/act_nt_gcact_ood/policy_best.ckpt"
NT_DST="${CKPT_BASE}/act_nt_v5b_ood_aug"
if [ ! -f "$NT_SRC" ]; then
    echo "ERROR: NT v4 OOD checkpoint not found at $NT_SRC"
    exit 1
fi

echo "[$(date)] Starting NEEDLE THROW v5b OOD augmented training..."
echo "  Source checkpoint: $NT_SRC"
echo "  Output directory:  $NT_DST"
echo ""
python finetune_augmented_wrapper.py \
    --task_name needle_throw_ood \
    --ckpt_dir "$NT_DST" \
    --resume_ckpt "$NT_SRC" \
    --lr ${LR} \
    --gpu 0 \
    --use_gesture --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_v5b_nt_ood_aug.log
NT_EXIT=$?
echo "[$(date)] NT v5b OOD aug DONE (exit code: $NT_EXIT)"
echo ""

if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: NT training failed. Check ${LOG_DIR}/train_v5b_nt_ood_aug.log"
    exit 1
fi

#  Summary
echo ""
echo "  V5b OOD AUGMENTED TRAINING COMPLETE"
echo "  Finished: $(date)"
echo "  Checkpoints:"
echo "    KT: ${KT_DST}/policy_best.ckpt"
echo "    NT: ${NT_DST}/policy_best.ckpt"
echo "  Logs:"
echo "    KT: ${LOG_DIR}/train_v5b_kt_ood_aug.log"
echo "    NT: ${LOG_DIR}/train_v5b_nt_ood_aug.log"
echo "  Next: run OOD eval, compare vs v4 (KT=0.904mm, NT=0.853mm)"
