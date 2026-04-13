#!/bin/bash
# V6 Full OOD Eval  -- Truly OOD models (v2 from scratch without tissue 6)
# KT + NT, raw + ensemble, all tissue 6 episodes

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/v6_ood"

KT_CKPT="${HOME}/checkpoints/act_kt_gcact_ood_scratch/policy_best.ckpt"
NT_CKPT="${HOME}/checkpoints/act_nt_gcact_ood_scratch/policy_best.ckpt"

mkdir -p "$RESULTS_BASE"

echo "  V6 Full OOD Eval  -- Tissue 6 (truly held out)"
echo "  KT: $KT_CKPT"
echo "  NT: $NT_CKPT"
echo ""

# 1. KT tissue 6, raw
echo "[$(date)] [1/8] KT  -- tissue 6, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/gcact_kt_tissue6_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 51 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood

echo ""

# 2. KT tissue 6, ensemble
echo "[$(date)] [2/8] KT  -- tissue 6, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/gcact_kt_tissue6_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 51 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood

echo ""

# 3. KT train tissues, raw
echo "[$(date)] [3/8] KT  -- train tissues, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/gcact_kt_train_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood

echo ""

# 4. KT train tissues, ensemble
echo "[$(date)] [4/8] KT  -- train tissues, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/gcact_kt_train_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood

echo ""

# 5. NT tissue 6, raw
echo "[$(date)] [5/8] NT  -- tissue 6, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$NT_CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/gcact_nt_tissue6_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 17 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key needle_throw_ood

echo ""

# 6. NT tissue 6, ensemble
echo "[$(date)] [6/8] NT  -- tissue 6, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$NT_CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/gcact_nt_tissue6_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 17 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key needle_throw_ood

echo ""

# 7. NT train tissues, raw
echo "[$(date)] [7/8] NT  -- train tissues, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$NT_CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/gcact_nt_train_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key needle_throw_ood

echo ""

# 8. NT train tissues, ensemble
echo "[$(date)] [8/8] NT  -- train tissues, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$NT_CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/gcact_nt_train_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 8 9 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key needle_throw_ood

echo ""
echo "  V6 OOD EVAL COMPLETE"

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for dir in gcact_kt_tissue6_raw gcact_kt_tissue6_ensemble gcact_kt_train_raw gcact_kt_train_ensemble gcact_nt_tissue6_raw gcact_nt_tissue6_ensemble gcact_nt_train_raw gcact_nt_train_ensemble; do
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
