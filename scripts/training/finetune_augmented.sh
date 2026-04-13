#!/bin/bash
set -o pipefail  # capture Python exit codes through tee pipes

#  Augmented Fine-Tuning for Domain Robustness
#
#  Purpose:  Fine-tune GC-ACT (NT, KT) and v2 (NP) checkpoints with
#            aggressive image augmentations to improve robustness to
#            visual domain shift at JHU's dVRK setup (different lighting,
#            camera angles, background, etc.)
#
#  Strategy:
#    - Load best existing checkpoints as starting point
#    - Training images get AggressiveDataAug (strong color jitter,
#      Gaussian blur/noise, tighter crops, rotation, perspective, cutout)
#    - Validation images get MinimalDataAug (center crop only, no augmentation)
#    - Low learning rate (1/10th of original) to avoid catastrophic forgetting
#    - Short training (300-500 epochs, not full retraining)
#
#  Does NOT modify original training code  -- uses finetune_augmented_wrapper.py
#  which monkey-patches the augmentation pipeline at runtime.
#

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

# Configurable parameters
NUM_EPOCHS=400           # Short fine-tuning (not full retraining)
LR_GCACT="1e-5"         # 1/10th of GC-ACT lr (1e-4)
LR_V2="5e-5"            # 1/10th of v2 lr (5e-4)
BATCH_SIZE=32
GRAD_ACCUM=8             # Effective batch size = 32*8 = 256
SAVE_FREQ=100            # Save every 100 epochs

# Common args shared by all subtasks
COMMON_ARGS_BASE="--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} --image_encoder efficientnet_b3 --num_epochs ${NUM_EPOCHS} --seed 0 --use_amp"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    echo ""
    echo "Fine-tuning needs the full GPU. Kill other processes first or wait."
    exit 1
fi

echo "  AUGMENTED FINE-TUNING  -- Domain Robustness for JHU dVRK"
echo "  Config: bs=${BATCH_SIZE}x${GRAD_ACCUM}accum(=$(($BATCH_SIZE * $GRAD_ACCUM))), EfficientNet-B3, AMP"
echo "  Epochs: ${NUM_EPOCHS} per subtask"
echo "  LR:     GC-ACT=${LR_GCACT}, v2=${LR_V2} (1/10th of original)"
echo "  Augment: AggressiveDataAug (train) / MinimalDataAug (val)"
echo "  Mode:   Sequential (NT -> KT -> NP)"
echo ""

#  1. NEEDLE THROW (GC-ACT)
NT_SRC="${CKPT_BASE}/act_nt_gcact/policy_best.ckpt"
NT_DST="${CKPT_BASE}/act_nt_gcact_aug"
if [ ! -f "$NT_SRC" ]; then
    echo "ERROR: NT GC-ACT checkpoint not found at $NT_SRC"
    echo "Run GC-ACT training first (train_gcact.sh)"
    exit 1
fi

echo "[$(date)] Starting NEEDLE THROW augmented fine-tuning..."
echo "  Source checkpoint: $NT_SRC"
echo "  Output directory:  $NT_DST"
echo ""
python finetune_augmented_wrapper.py \
    --task_name needle_throw_all \
    --ckpt_dir "$NT_DST" \
    --resume_ckpt "$NT_SRC" \
    --lr ${LR_GCACT} \
    --gpu 0 \
    --use_gesture --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    $COMMON_ARGS_BASE 2>&1 | tee ${LOG_DIR}/finetune_nt_aug.log
NT_EXIT=$?
echo "[$(date)] NT augmented fine-tuning DONE (exit code: $NT_EXIT)"
echo ""

if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: NT fine-tuning failed. Check ${LOG_DIR}/finetune_nt_aug.log"
    exit 1
fi

#  2. KNOT TYING (GC-ACT)
KT_SRC="${CKPT_BASE}/act_kt_gcact/policy_best.ckpt"
KT_DST="${CKPT_BASE}/act_kt_gcact_aug"
if [ ! -f "$KT_SRC" ]; then
    echo "ERROR: KT GC-ACT checkpoint not found at $KT_SRC"
    exit 1
fi

echo "[$(date)] Starting KNOT TYING augmented fine-tuning..."
echo "  Source checkpoint: $KT_SRC"
echo "  Output directory:  $KT_DST"
echo ""
python finetune_augmented_wrapper.py \
    --task_name knot_tying_all \
    --ckpt_dir "$KT_DST" \
    --resume_ckpt "$KT_SRC" \
    --lr ${LR_GCACT} \
    --gpu 0 \
    --use_gesture --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    $COMMON_ARGS_BASE 2>&1 | tee ${LOG_DIR}/finetune_kt_aug.log
KT_EXIT=$?
echo "[$(date)] KT augmented fine-tuning DONE (exit code: $KT_EXIT)"
echo ""

if [ $KT_EXIT -ne 0 ]; then
    echo "ERROR: KT fine-tuning failed. Check ${LOG_DIR}/finetune_kt_aug.log"
    exit 1
fi

#  3. NEEDLE PICKUP (v2, no gesture conditioning)
NP_SRC="${CKPT_BASE}/act_np_v2/policy_best.ckpt"
NP_DST="${CKPT_BASE}/act_np_v2_aug"
if [ ! -f "$NP_SRC" ]; then
    echo "WARNING: NP v2 checkpoint not found at $NP_SRC  -- skipping NP"
else
    echo "[$(date)] Starting NEEDLE PICKUP augmented fine-tuning..."
    echo "  Source checkpoint: $NP_SRC"
    echo "  Output directory:  $NP_DST"
    echo ""
    python finetune_augmented_wrapper.py \
        --task_name needle_pickup_all \
        --ckpt_dir "$NP_DST" \
        --resume_ckpt "$NP_SRC" \
        --lr ${LR_V2} \
        --gpu 0 \
        $COMMON_ARGS_BASE 2>&1 | tee ${LOG_DIR}/finetune_np_aug.log
    NP_EXIT=$?
    echo "[$(date)] NP augmented fine-tuning DONE (exit code: $NP_EXIT)"
    echo ""

    if [ $NP_EXIT -ne 0 ]; then
        echo "WARNING: NP fine-tuning failed. Check ${LOG_DIR}/finetune_np_aug.log"
        echo "Continuing (NP is optional  -- already achieves 9/10)."
    fi
fi

#  Summary
echo ""
echo "  AUGMENTED FINE-TUNING COMPLETE"
echo "  Augmented checkpoints:"
echo "    NT (GC-ACT): ${NT_DST}/policy_best.ckpt"
echo "    KT (GC-ACT): ${KT_DST}/policy_best.ckpt"
if [ -f "$NP_SRC" ]; then
echo "    NP (v2):     ${NP_DST}/policy_best.ckpt"
fi
echo ""
echo "  Logs:"
echo "    NT: ${LOG_DIR}/finetune_nt_aug.log"
echo "    KT: ${LOG_DIR}/finetune_kt_aug.log"
if [ -f "$NP_SRC" ]; then
echo "    NP: ${LOG_DIR}/finetune_np_aug.log"
fi
echo ""
echo "  To use augmented models in inference, update checkpoint paths in:"
echo "    ~/SutureBot/src/act/chained_dvrk_inference_gcact.py"
