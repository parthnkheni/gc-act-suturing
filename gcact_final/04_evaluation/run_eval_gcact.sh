#!/bin/bash
# run_eval_gcact.sh  -- Run All Model Evaluations
# Shell script that runs offline_eval.py for all model versions (v1, v2,
# GC-ACT) with and without temporal ensembling. Produces the comparison
# numbers used in the paper tables. One command to generate all results.

# Offline evaluation  -- Full comparison: v1 vs v2 vs GC-ACT
# Run after both v2 and GC-ACT training complete
# Evaluates on tissue 7 (validation tissue, included in ACT training)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

cd /home/exouser

COMMON="--data_dir /home/exouser/data --tissue_ids 7 --max_episodes 5 --step_stride 1"

echo "  Offline Eval  -- ACT v1 vs v2 vs GC-ACT Comparison"

# 1. GC-ACT without temporal ensembling
echo ""
echo "[1/5] GC-ACT  -- no ensemble..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_gcact_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    $COMMON

# 2. GC-ACT with temporal ensembling
echo ""
echo "[2/5] GC-ACT  -- with temporal ensemble (k=0.01, h=20)..."
python offline_eval.py \
    --ckpt_nt /home/exouser/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_gcact/policy_best.ckpt \
    --subtasks needle_throw knot_tying \
    --output_dir /home/exouser/offline_eval_results_gcact_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

# 3. v2 without temporal ensembling
echo ""
echo "[3/5] v2  -- no ensemble..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v2_raw \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    $COMMON

# 4. v2 with temporal ensembling
echo ""
echo "[4/5] v2  -- with temporal ensemble (k=0.01, h=20)..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_v2/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v2_ensemble \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

# 5. v1 with temporal ensembling (reference baseline)
echo ""
echo "[5/5] v1  -- with temporal ensemble (k=0.01, h=20) [reference]..."
python offline_eval.py \
    --ckpt_np /home/exouser/checkpoints/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /home/exouser/checkpoints/act_nt_10t_kl1/policy_best.ckpt \
    --ckpt_kt /home/exouser/checkpoints/act_kt_10t_kl1/policy_best.ckpt \
    --output_dir /home/exouser/offline_eval_results_v1_ensemble \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    $COMMON

echo ""
echo "  Compare results:"
echo "    GC-ACT raw:     ~/offline_eval_results_gcact_raw/decision.json"
echo "    GC-ACT ensemble: ~/offline_eval_results_gcact_ensemble/decision.json"
echo "    v2 raw:          ~/offline_eval_results_v2_raw/decision.json"
echo "    v2 ensemble:     ~/offline_eval_results_v2_ensemble/decision.json"
echo "    v1 ensemble:     ~/offline_eval_results_v1_ensemble/decision.json"
echo ""
echo "  Note: GC-ACT only has NT and KT (no NP gesture labels)."
echo "  For NP, use v2 checkpoint."
