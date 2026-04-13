#!/bin/bash
# Train 3 per-subtask ACT models on all 10 tissues with KL=1
# A100-SXM4-40GB, sequential training

set -e

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

COMMON_ARGS="--policy_class ACT --kl_weight 1 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --batch_size 8 --image_encoder resnet18 --num_epochs 2000 --seed 0 --gpu 0"

echo "  Training Needle Pickup (NP) - all 10 tissues, KL=1"
python imitate_episodes.py \
    --task_name needle_pickup_all \
    --ckpt_dir /home/exouser/checkpoints/act_np_all10_kl1 \
    $COMMON_ARGS

echo ""
echo "  Training Needle Throw (NT) - all 10 tissues, KL=1"
python imitate_episodes.py \
    --task_name needle_throw_all \
    --ckpt_dir /home/exouser/checkpoints/act_nt_all10_kl1 \
    $COMMON_ARGS

echo ""
echo "  Training Knot Tying (KT) - all 10 tissues, KL=1"
python imitate_episodes.py \
    --task_name knot_tying_all \
    --ckpt_dir /home/exouser/checkpoints/act_kt_all10_kl1 \
    $COMMON_ARGS

echo ""
echo "  All 3 subtask models trained successfully!"
