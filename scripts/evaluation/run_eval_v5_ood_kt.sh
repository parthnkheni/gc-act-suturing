#!/bin/bash
# V5 OOD KT Evaluation  -- Evaluate v5 aug KT on held-out tissue 6
# No gesture conditioning (v5 drops it)
# Designed to run concurrently with NT training (eval is lighter on GPU)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

CKPT="${HOME}/checkpoints/act_kt_v5_ood_aug/policy_best.ckpt"
EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/v5_ood_kt"

# Wait for checkpoint to exist (KT training may still be finishing)
echo "Waiting for v5 KT checkpoint..."
while [ ! -f "$CKPT" ]; do
    sleep 30
done
echo "Checkpoint found: $CKPT"

# Also grab the best epoch checkpoint if available
BEST_EPOCH_CKPT=$(ls -t ${HOME}/checkpoints/act_kt_v5_ood_aug/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
if [ -n "$BEST_EPOCH_CKPT" ]; then
    CKPT="$BEST_EPOCH_CKPT"
    echo "Using best epoch checkpoint: $CKPT"
fi

mkdir -p "$RESULTS_BASE"

echo "  V5 OOD KT Evaluation  -- Tissue 6 (held out)"
echo "  Checkpoint: $CKPT"
echo "  No gesture conditioning"
echo "  Baseline to beat: v4 OOD KT = 0.904mm (ensemble)"
echo ""

# 1. OOD Test: ALL tissue 6 KT episodes, NO ensemble
echo "[1/4] OOD Test  -- Tissue 6 KT, raw (no ensemble)..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/tissue6_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 51 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""

# 2. OOD Test: ALL tissue 6 KT episodes, WITH ensemble
echo "[2/4] OOD Test  -- Tissue 6 KT, temporal ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/tissue6_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 51 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""

# 3. In-distribution comparison: random 50 from train tissues, raw
echo "[3/4] In-distribution  -- Train tissues KT, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""

# 4. In-distribution comparison: random 50 from train tissues, ensemble
echo "[4/4] In-distribution  -- Train tissues KT, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""
echo "  V5 OOD KT EVAL COMPLETE"
echo "  Results: ${RESULTS_BASE}/"

# Print summary
echo ""
echo "=== SUMMARY (v5 vs v4 baseline: 0.904mm ensemble OOD) ==="
for dir in tissue6_raw tissue6_ensemble train_tissues_raw train_tissues_ensemble; do
    dec="${RESULTS_BASE}/${dir}/decision.json"
    if [ -f "$dec" ]; then
        echo ""
        echo "--- $dir ---"
        python3 -c "
import json
with open('$dec') as f:
    d = json.load(f)
for subtask, metrics in d.items():
    print(f'  {subtask}: pos={metrics[\"pos_l2_mm\"]:.3f}mm  rot={metrics[\"rot_err_deg\"]:.2f}°  jaw={metrics[\"jaw_acc_pct\"]:.1f}%  ({metrics[\"status\"]})')
"
    fi
done
