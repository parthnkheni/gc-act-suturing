#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

COMMON="--data_dir /home/exouser/data --tissue_ids 7 --max_episodes 50 --step_stride 1"

echo "  Offline Eval  -- Augmented Fine-tuned Models (50 episodes)"

echo ""
echo "[1/4] GC-ACT Augmented  -- no ensemble..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact_aug/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact_aug/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_50ep_gcact_aug_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    $COMMON

echo ""
echo "[2/4] GC-ACT Augmented  -- with ensemble..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact_aug/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact_aug/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_50ep_gcact_aug_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 \
    $COMMON

echo ""
echo "[3/4] v2 Augmented NP  -- no ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2_aug/policy_best.ckpt \
    --subtasks needle_pickup \
    --output_dir /home/exouser/offline_eval_results_50ep_v2_aug_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    $COMMON

echo ""
echo "[4/4] v2 Augmented NP  -- with ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2_aug/policy_best.ckpt \
    --subtasks needle_pickup \
    --output_dir /home/exouser/offline_eval_results_50ep_v2_aug_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 \
    $COMMON

echo ""
echo "  ALL AUGMENTED EVALS COMPLETE"
