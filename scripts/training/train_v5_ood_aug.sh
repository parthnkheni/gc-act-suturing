#!/bin/bash
set -o pipefail

#  V5 OOD Augmented Training  -- Fine-tune v4 OOD checkpoints with augmentation
#
#  What: Augmented fine-tuning on top of existing v4 OOD checkpoints
#  Why:  v4 OOD models are already trained without tissue 6. The improvement
#        from "GC-ACT" was actually from augmentation + extra epochs, NOT
#        gesture conditioning. So v5 = v4 OOD + 1400 aug epochs.
#
#  Resume from:
#    KT: ~/checkpoints/act_kt_gcact_ood/policy_best.ckpt (0.904mm OOD)
#    NT: ~/checkpoints/act_nt_gcact_ood/policy_best.ckpt (0.853mm OOD)
#
#  No --use_gesture flag (proven useless by ablation, Mar 13 2026)
#
#  Estimated time: KT ~5hr + NT ~4hr = ~9hr total

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

# Configurable parameters
NUM_EPOCHS=1400
LR="1e-5"               # 1/10th of GC-ACT lr (1e-4)
BATCH_SIZE=32
GRAD_ACCUM=8             # Effective batch size = 32*8 = 256
SAVE_FREQ=100

COMMON_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} --image_encoder efficientnet_b3 --num_epochs ${NUM_EPOCHS} --seed 0 --use_amp"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    echo ""
    echo "Training needs the full GPU. Kill other processes first or wait."
    exit 1
fi

echo "  V5 OOD AUGMENTED TRAINING"
echo "  Building on v4 OOD checkpoints (tissue 6 held out)"
echo "  Config: bs=${BATCH_SIZE}x${GRAD_ACCUM}accum(=$(($BATCH_SIZE * $GRAD_ACCUM))), EfficientNet-B3, AMP"
echo "  Epochs: ${NUM_EPOCHS} per subtask"
echo "  LR:     ${LR} (1/10th of GC-ACT lr)"
echo "  Augment: AggressiveDataAug (train) / MinimalDataAug (val)"
echo "  Gesture: DISABLED (ablation proved zero effect)"
echo "  Mode:   Sequential (KT -> NT)"
echo "  Start:  $(date)"
echo ""

#  1. KNOT TYING  -- OOD augmented
KT_SRC="${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
KT_DST="${CKPT_BASE}/act_kt_v5_ood_aug"
if [ ! -f "$KT_SRC" ]; then
    echo "ERROR: KT v4 OOD checkpoint not found at $KT_SRC"
    exit 1
fi

echo "[$(date)] Starting KNOT TYING v5 OOD augmented training..."
echo "  Source checkpoint: $KT_SRC"
echo "  Output directory:  $KT_DST"
echo "  Task config:       knot_tying_ood"
echo ""
python finetune_augmented_wrapper.py \
    --task_name knot_tying_ood \
    --ckpt_dir "$KT_DST" \
    --resume_ckpt "$KT_SRC" \
    --lr ${LR} \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_v5_kt_ood_aug.log
KT_EXIT=$?
echo "[$(date)] KT v5 OOD aug DONE (exit code: $KT_EXIT)"
echo ""

if [ $KT_EXIT -ne 0 ]; then
    echo "ERROR: KT training failed. Check ${LOG_DIR}/train_v5_kt_ood_aug.log"
    exit 1
fi

#  2. NEEDLE THROW  -- OOD augmented
NT_SRC="${CKPT_BASE}/act_nt_gcact_ood/policy_best.ckpt"
NT_DST="${CKPT_BASE}/act_nt_v5_ood_aug"
if [ ! -f "$NT_SRC" ]; then
    echo "ERROR: NT v4 OOD checkpoint not found at $NT_SRC"
    exit 1
fi

echo "[$(date)] Starting NEEDLE THROW v5 OOD augmented training..."
echo "  Source checkpoint: $NT_SRC"
echo "  Output directory:  $NT_DST"
echo "  Task config:       needle_throw_ood"
echo ""
python finetune_augmented_wrapper.py \
    --task_name needle_throw_ood \
    --ckpt_dir "$NT_DST" \
    --resume_ckpt "$NT_SRC" \
    --lr ${LR} \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_v5_nt_ood_aug.log
NT_EXIT=$?
echo "[$(date)] NT v5 OOD aug DONE (exit code: $NT_EXIT)"
echo ""

if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: NT training failed. Check ${LOG_DIR}/train_v5_nt_ood_aug.log"
    exit 1
fi

#  Summary
echo ""
echo "  V5 OOD AUGMENTED TRAINING COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Checkpoints:"
echo "    KT: ${KT_DST}/policy_best.ckpt"
echo "    NT: ${NT_DST}/policy_best.ckpt"
echo ""
echo "  Logs:"
echo "    KT: ${LOG_DIR}/train_v5_kt_ood_aug.log"
echo "    NT: ${LOG_DIR}/train_v5_nt_ood_aug.log"
echo ""
echo "  Next steps:"
echo "    1. Run OOD eval on tissue 6:"
echo "       python ~/scripts/evaluation/offline_eval.py --task knot_tying_ood --ckpt ${KT_DST}/policy_best.ckpt --ensemble"
echo "       python ~/scripts/evaluation/offline_eval.py --task needle_throw_ood --ckpt ${NT_DST}/policy_best.ckpt --ensemble"
echo "    2. Compare against v4 OOD: KT=0.904mm, NT=0.853mm"
echo "    3. Update deploy_package with new checkpoints if improved"
