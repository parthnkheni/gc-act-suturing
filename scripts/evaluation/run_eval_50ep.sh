#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

COMMON="--data_dir /home/exouser/data --tissue_ids 7 --max_episodes 50 --step_stride 1"

echo "  Offline Eval  -- 50 episodes per subtask"

echo ""
echo "[1/5] GC-ACT  -- no ensemble..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_50ep_gcact_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    $COMMON

echo ""
echo "[2/5] GC-ACT  -- with temporal ensemble..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_50ep_gcact_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

echo ""
echo "[3/5] v2  -- no ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_50ep_v2_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    $COMMON

echo ""
echo "[4/5] v2  -- with temporal ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_50ep_v2_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

echo ""
echo "[5/5] v1  -- with temporal ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_10t_kl1/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_10t_kl1/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_50ep_v1_ensemble \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

echo ""
echo "  DONE  -- Results in ~/offline_eval_results_50ep_*/"
