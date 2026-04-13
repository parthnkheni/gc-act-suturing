#!/bin/bash
# OOD KT Evaluation  -- Evaluate GC-ACT KT OOD on held-out tissue 6
# Runs ALL tissue 6 KT episodes (51 total) with both raw and ensemble
# Also evaluates on a random sample across train tissues for comparison

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

CKPT="${HOME}/checkpoints/act_kt_gcact_ood/policy_best.ckpt"
EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/ood_kt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: OOD KT checkpoint not found at $CKPT"
    echo "Training may not have completed yet."
    exit 1
fi

mkdir -p "$RESULTS_BASE"

echo "  OOD KT Evaluation  -- Tissue 6 (held out)"
echo "  Checkpoint: $CKPT"
echo "  Test tissue: 6 (NEVER seen during training)"
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
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood

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
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood

echo ""

# 3. In-distribution comparison: random 50 from train tissues, raw
echo "[3/4] In-distribution  -- Train tissues KT, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood

echo ""

# 4. In-distribution comparison: random 50 from train tissues, ensemble
echo "[4/4] In-distribution  -- Train tissues KT, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood

echo ""
echo "  OOD KT EVAL COMPLETE"
echo "  Results: ${RESULTS_BASE}/"

# Print summary
echo ""
echo "=== SUMMARY ==="
for dir in tissue6_raw tissue6_ensemble train_tissues_raw train_tissues_ensemble; do
    summary="${RESULTS_BASE}/${dir}/summary.json"
    if [ -f "$summary" ]; then
        echo ""
        echo "--- $dir ---"
        python3 -c "
import json
with open('$summary') as f:
    d = json.load(f)
for subtask, metrics in d.items():
    pos = metrics.get('pos_l2_mean_mm', 'N/A')
    rot = metrics.get('rot_err_mean_deg', 'N/A')
    jaw = metrics.get('jaw_acc_mean_pct', 'N/A')
    if isinstance(pos, dict):
        print(f'  {subtask}: pos={pos[\"mean\"]:.3f}±{pos[\"std\"]:.3f}mm  rot={rot[\"mean\"]:.2f}°  jaw={jaw[\"mean\"]:.1f}%')
    else:
        print(f'  {subtask}: pos={pos:.3f}mm  rot={rot:.2f}°  jaw={jaw:.1f}%')
"
    fi
done
