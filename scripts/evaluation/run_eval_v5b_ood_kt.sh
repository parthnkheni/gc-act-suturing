#!/bin/bash
# V5b OOD KT Evaluation  -- WITH gesture flags preserved

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

CKPT="${HOME}/checkpoints/act_kt_v5b_ood_aug/policy_best.ckpt"
EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/v5b_ood_kt"

BEST_EPOCH_CKPT=$(ls -t ${HOME}/checkpoints/act_kt_v5b_ood_aug/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
if [ -n "$BEST_EPOCH_CKPT" ]; then
    CKPT="$BEST_EPOCH_CKPT"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: v5b KT checkpoint not found"
    exit 1
fi

mkdir -p "$RESULTS_BASE"

echo "  V5b OOD KT Evaluation  -- Tissue 6 (held out)"
echo "  Checkpoint: $CKPT"
echo "  Gesture flags: YES (preserved for checkpoint compat)"
echo "  Baseline: v4 OOD KT = 0.904mm (ensemble)"
echo ""

echo "[1/4] OOD Test  -- Tissue 6 KT, raw..."
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
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""
echo "[2/4] OOD Test  -- Tissue 6 KT, ensemble..."
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
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""
echo "[3/4] In-dist  -- Train tissues KT, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""
echo "[4/4] In-dist  -- Train tissues KT, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood \
    --no_plots

echo ""
echo "=== SUMMARY (v4 baseline: 0.904mm ensemble OOD) ==="
for dir in tissue6_raw tissue6_ensemble train_tissues_raw train_tissues_ensemble; do
    dec="${RESULTS_BASE}/${dir}/decision.json"
    if [ -f "$dec" ]; then
        echo "--- $dir ---"
        python3 -c "
import json
with open('$dec') as f:
    d = json.load(f)
for s, m in d.items():
    print(f'  {s}: pos={m[\"pos_l2_mm\"]:.3f}mm  rot={m[\"rot_err_deg\"]:.2f}°  jaw={m[\"jaw_acc_pct\"]:.1f}%  ({m[\"status\"]})')
"
    fi
done
