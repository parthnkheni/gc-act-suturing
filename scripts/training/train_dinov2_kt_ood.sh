#!/bin/bash
set -o pipefail

# DINOv2 ACT  -- KT OOD Training
# Frozen DINOv2-S/14 backbone + trainable ACT transformer
# No gesture conditioning (proven zero effect)
# Tissue 6 = TEST, Tissue 10 = VAL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
EVAL_SCRIPT=/home/exouser/scripts/evaluation/offline_eval.py
RESULTS_BASE=/home/exouser/eval_results/dinov2_ood

mkdir -p "$LOG_DIR" "$RESULTS_BASE"

# Pre-flight: wait for GPU to be free
while true; do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
        break
    fi
    echo "[$(date)] GPU busy ($GPU_PROCS processes). Waiting 60s..."
    sleep 60
done

echo "  DINOv2 ACT  -- KT OOD Training"
echo "  Backbone: DINOv2-S/14 (frozen)"
echo "  Train: tissues 2,3,4,5,7,8,9"
echo "  Val: tissue 10 | Test: tissue 6"
echo "  No gesture conditioning"
echo ""

# Train KT with DINOv2
echo "[$(date)] Training KT with DINOv2 backbone (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_kt_dinov2_ood

python imitate_episodes.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_dinov2_ood \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 5e-4 \
    --batch_size 8 \
    --grad_accum_steps 32 \
    --image_encoder dinov2_vits14 \
    --num_epochs 3000 \
    --seed 0 \
    --use_amp \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_dinov2_kt_ood.log

TRAIN_EXIT=$?
echo "[$(date)] Training DONE (exit: $TRAIN_EXIT)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: DINOv2 KT training failed."
    exit 1
fi

# Eval on tissue 6
KT_CKPT="${CKPT_BASE}/act_kt_dinov2_ood/policy_best.ckpt"
if [ ! -f "$KT_CKPT" ]; then
    KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_dinov2_ood/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -n "$KT_CKPT" ] && [ -f "$KT_CKPT" ]; then
    echo ""
    echo "[$(date)] Evaluating DINOv2 KT on tissue 6..."
    cd /home/exouser

    # Raw
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_tissue6_raw" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 51 \
        --step_stride 1 \
        --image_encoder dinov2_vits14 --kl_weight 10 \
        --norm_stats_key knot_tying_ood \
        2>&1 | tee ${LOG_DIR}/eval_dinov2_kt_raw.log

    # Ensemble
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_tissue6_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 51 \
        --step_stride 1 \
        --image_encoder dinov2_vits14 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key knot_tying_ood \
        2>&1 | tee ${LOG_DIR}/eval_dinov2_kt_ensemble.log

    # Train tissues for comparison
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_train_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 2 3 4 5 7 8 9 \
        --max_episodes 50 \
        --step_stride 1 \
        --image_encoder dinov2_vits14 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key knot_tying_ood \
        2>&1 | tee ${LOG_DIR}/eval_dinov2_kt_train_ensemble.log

    echo ""
    echo "=== DINOV2 KT OOD RESULTS ==="
    for dir in kt_tissue6_raw kt_tissue6_ensemble kt_train_ensemble; do
        summary="${RESULTS_BASE}/${dir}/summary.json"
        if [ -f "$summary" ]; then
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
        print(f'  {subtask}: pos={pos[\"mean\"]:.3f}+/-{pos[\"std\"]:.3f}mm  rot={rot[\"mean\"]:.2f}deg  jaw={jaw[\"mean\"]:.1f}%')
    else:
        print(f'  {subtask}: pos={pos:.3f}mm  rot={rot:.2f}deg  jaw={jaw:.1f}%')
"
        fi
    done
fi

echo ""
echo "  DINOv2 KT OOD COMPLETE  -- $(date)"
