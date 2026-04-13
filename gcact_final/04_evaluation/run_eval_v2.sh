#!/bin/bash
# run_eval_v2.sh  -- Run ACT v2 Evaluation
# Shell script that runs offline_eval.py specifically for ACT v2, comparing
# raw predictions vs temporal ensembling (averaging overlapping action chunks
# for smoother output). Also runs v1 ensemble for baseline comparison.

# Offline evaluation commands  -- run after v2 training completes
# Compares: v2 (no ensemble) vs v2 (with ensemble) vs v1 (with ensemble)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

cd /home/exouser

COMMON="--data_dir /home/exouser/data --tissue_ids 7 --max_episodes 5 --step_stride 1"

echo "  Offline Eval  -- ACT v2 vs v1 Comparison"

# 1. v2 without temporal ensembling (baseline)
echo ""
echo "[1/3] v2  -- no ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v2_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    $COMMON

# 2. v2 with temporal ensembling (best settings from v1 sweep)
echo ""
echo "[2/3] v2  -- with temporal ensemble (k=0.01, h=20)..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v2_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

# 3. v1 with temporal ensembling (our previous best, for comparison)
echo ""
echo "[3/3] v1  -- with temporal ensemble (k=0.01, h=20) [reference]..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_10t_kl1/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_10t_kl1/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v1_ensemble \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

echo ""
echo "  Compare results:"
echo "    v2 raw:      ~/offline_eval_results_v2_raw/decision.json"
echo "    v2 ensemble:  ~/offline_eval_results_v2_ensemble/decision.json"
echo "    v1 ensemble:  ~/offline_eval_results_v1_ensemble/decision.json"
