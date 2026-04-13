#!/bin/bash
# train_v2.sh  -- Train ACT v2 Models (All 3 Subtasks)
# Trains the ACT v2 models matching the SutureBot paper configuration:
# EfficientNet-B3 backbone, effective batch size 256, 3000 epochs per subtask.
# Trains needle pickup, needle throw, and knot tying sequentially on all 10
# tissues. Takes ~24 hours total on an A100 GPU. The resulting checkpoints
# serve as the baseline and also as the starting point for GC-ACT fine-tuning.

# ACT v2 Training: Matching SutureBot paper config
# Effective bs=256 (bs=128 x 2 grad accum), lr=5e-4, kl=10, EfficientNet-B3, AMP, 3000 epochs
# SEQUENTIAL training (one model at a time)
#
# Changes from v1 (train_10t.sh):
#   batch_size:    16 -> 128 (x2 grad accum = effective 256)
#   lr:            1e-5 -> 5e-4
#   kl_weight:     1 -> 10
#   image_encoder: resnet18 -> efficientnet_b3
#   num_epochs:    2000 -> 3000
#   AMP:           off -> on
#   Execution:     concurrent -> sequential

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

COMMON_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --lr 5e-4 --batch_size 32 --grad_accum_steps 8 --image_encoder efficientnet_b3 --num_epochs 3000 --seed 0 --use_amp"

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
mkdir -p "$LOG_DIR"

echo "  ACT v2 Training  -- SutureBot Paper Config"
echo "  Config: bs=32x8accum(=256), lr=5e-4, kl=10, EfficientNet-B3, AMP"
echo "  Epochs: 3000 per subtask"
echo "  Mode:   Sequential (one model at a time)"
echo ""

# Needle Pickup
echo "[$(date)] Starting NEEDLE PICKUP training..."
python imitate_episodes.py \
    --task_name needle_pickup_all \
    --ckpt_dir ${CKPT_BASE}/act_np_v2 \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_np_v2.log
NP_EXIT=$?
echo "[$(date)] NP training DONE (exit code: $NP_EXIT)"
echo ""

if [ $NP_EXIT -ne 0 ]; then
    echo "ERROR: NP training failed. Stopping."
    exit 1
fi

# Needle Throw
echo "[$(date)] Starting NEEDLE THROW training..."
python imitate_episodes.py \
    --task_name needle_throw_all \
    --ckpt_dir ${CKPT_BASE}/act_nt_v2 \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_nt_v2.log
NT_EXIT=$?
echo "[$(date)] NT training DONE (exit code: $NT_EXIT)"
echo ""

if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: NT training failed. Stopping."
    exit 1
fi

# Knot Tying
echo "[$(date)] Starting KNOT TYING training..."
python imitate_episodes.py \
    --task_name knot_tying_all \
    --ckpt_dir ${CKPT_BASE}/act_kt_v2 \
    --gpu 0 \
    $COMMON_ARGS 2>&1 | tee ${LOG_DIR}/train_kt_v2.log
KT_EXIT=$?
echo "[$(date)] KT training DONE (exit code: $KT_EXIT)"
echo ""

echo "  ALL 3 SUBTASK MODELS TRAINED (ACT v2)"
echo "  Checkpoints:"
echo "    NP: ${CKPT_BASE}/act_np_v2/policy_best.ckpt"
echo "    NT: ${CKPT_BASE}/act_nt_v2/policy_best.ckpt"
echo "    KT: ${CKPT_BASE}/act_kt_v2/policy_best.ckpt"
