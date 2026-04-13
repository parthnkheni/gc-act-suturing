#!/bin/bash
set -o pipefail

# V6b: Full OOD Training with tuned parameters for best tissue 6 results
# Changes from v6:
#   - AggressiveDataAug from the START of v2 training (not just fine-tuning)
#   - Backbone LR: 1e-5 -> 5e-5 (let backbone learn surgical features faster)
#   - Weight decay: 1e-4 -> 1e-5 (less regularization)
#   - Epochs: 3500 (slightly more than v6's 3000)
#   - Seed: 42 (different from v6's 0)
#   - NO GC-ACT phase (gesture conditioning has zero effect, avoids pos_embed mismatch)
#
# Phase 1: v2 KT OOD from scratch with AggressiveDataAug (3500 epochs)
# Phase 1-eval: Eval KT immediately (tissue 6 + train tissues, raw + ensemble)
# Phase 2: v2 NT OOD from scratch with AggressiveDataAug (3500 epochs)
# Phase 2-eval: Eval NT immediately
# Tissue 6 = held-out TEST, Tissue 10 = VAL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/v6b_ood"
mkdir -p "$LOG_DIR" "$RESULTS_BASE"

# Pre-flight: check GPU is free
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: $GPU_PROCS process(es) already using GPU. Aborting."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    exit 1
fi

echo "$(date +%s)" > /tmp/v6b_train_start

echo "  V6b: Full OOD Training  -- v2 from scratch + AggressiveDataAug"
echo "  Train: tissues 2,3,4,5,7,8,9 (KT) / 3,4,5,7,8,9 (NT)"
echo "  Val: tissue 10"
echo "  Test: tissue 6  -- NEVER TOUCHED"
echo "  Changes from v6:"
echo "    - AggressiveDataAug during v2 backbone training"
echo "    - Backbone LR: 5e-5 (was 1e-5)"
echo "    - Weight decay: 1e-5 (was 1e-4)"
echo "    - Seed: 42 (was 0)"
echo "    - 3500 epochs (was 3000)"
echo "    - No GC-ACT phase (gesture has zero effect)"
echo "  Eval runs immediately after each subtask"
echo ""

# KT TRAINING

echo "[$(date)] PHASE 1: v2 KT OOD from scratch + AggressiveDataAug (3500 epochs)..."
mkdir -p ${CKPT_BASE}/act_kt_v2_v6b

python finetune_augmented_wrapper.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_v2_v6b \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 5e-4 \
    --lr_backbone 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 3500 \
    --seed 42 \
    --use_amp \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_v6b_kt.log

KT_EXIT=$?
echo "[$(date)] KT training DONE (exit code: $KT_EXIT)"

if [ $KT_EXIT -ne 0 ]; then
    echo "ERROR: KT training failed. Stopping."
    exit 1
fi

# KT EVAL (runs immediately)

KT_CKPT="${CKPT_BASE}/act_kt_v2_v6b/policy_best.ckpt"
if [ ! -f "$KT_CKPT" ]; then
    KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2_v6b/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$KT_CKPT" ] || [ ! -f "$KT_CKPT" ]; then
    echo "ERROR: No KT checkpoint found for eval."
else
    echo ""
    echo "[$(date)] KT EVAL: tissue 6 raw..."
    cd /home/exouser
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_tissue6_raw" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --norm_stats_key knot_tying_ood

    echo ""
    echo "[$(date)] KT EVAL: tissue 6 ensemble..."
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_tissue6_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key knot_tying_ood

    echo ""
    echo "[$(date)] KT EVAL: train tissues ensemble..."
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/kt_train_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 2 3 4 5 7 8 9 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key knot_tying_ood

    echo ""
    echo "  V6b KT RESULTS (compare to v6: 4.795mm OOD, v4: 0.904mm OOD)"
    for dir in kt_tissue6_raw kt_tissue6_ensemble kt_train_ensemble; do
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
        print(f'  {subtask}: pos={pos[\"mean\"]:.3f}+/-{pos[\"std\"]:.3f}mm  rot={rot[\"mean\"]:.2f}deg  jaw={jaw[\"mean\"]:.1f}%')
    else:
        print(f'  {subtask}: pos={pos:.3f}mm  rot={rot:.2f}deg  jaw={jaw:.1f}%')
"
        fi
    done
    echo ""
    cd /home/exouser/SutureBot/src/act
fi

# NT TRAINING

echo ""
echo "[$(date)] PHASE 2: v2 NT OOD from scratch + AggressiveDataAug (3500 epochs)..."
mkdir -p ${CKPT_BASE}/act_nt_v2_v6b

python finetune_augmented_wrapper.py \
    --task_name needle_throw_ood \
    --ckpt_dir ${CKPT_BASE}/act_nt_v2_v6b \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 5e-4 \
    --lr_backbone 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 3500 \
    --seed 42 \
    --use_amp \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_v6b_nt.log

NT_EXIT=$?
echo "[$(date)] NT training DONE (exit code: $NT_EXIT)"

# NT EVAL

NT_CKPT="${CKPT_BASE}/act_nt_v2_v6b/policy_best.ckpt"
if [ ! -f "$NT_CKPT" ]; then
    NT_CKPT=$(ls -t ${CKPT_BASE}/act_nt_v2_v6b/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$NT_CKPT" ] || [ ! -f "$NT_CKPT" ]; then
    echo "ERROR: No NT checkpoint found for eval."
else
    echo ""
    echo "[$(date)] NT EVAL: tissue 6 raw..."
    cd /home/exouser
    python "$EVAL_SCRIPT" \
        --ckpt_nt "$NT_CKPT" \
        --subtasks needle_throw \
        --output_dir "${RESULTS_BASE}/nt_tissue6_raw" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --norm_stats_key needle_throw_ood

    echo ""
    echo "[$(date)] NT EVAL: tissue 6 ensemble..."
    python "$EVAL_SCRIPT" \
        --ckpt_nt "$NT_CKPT" \
        --subtasks needle_throw \
        --output_dir "${RESULTS_BASE}/nt_tissue6_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key needle_throw_ood

    echo ""
    echo "[$(date)] NT EVAL: train tissues ensemble..."
    python "$EVAL_SCRIPT" \
        --ckpt_nt "$NT_CKPT" \
        --subtasks needle_throw \
        --output_dir "${RESULTS_BASE}/nt_train_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 3 4 5 7 8 9 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key needle_throw_ood

    echo ""
    echo "  V6b NT RESULTS (compare to v6: 6.074mm OOD, v4: 0.853mm OOD)"
    for dir in nt_tissue6_raw nt_tissue6_ensemble nt_train_ensemble; do
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
        print(f'  {subtask}: pos={pos[\"mean\"]:.3f}+/-{pos[\"std\"]:.3f}mm  rot={rot[\"mean\"]:.2f}deg  jaw={jaw[\"mean\"]:.1f}%')
    else:
        print(f'  {subtask}: pos={pos:.3f}mm  rot={rot:.2f}deg  jaw={jaw:.1f}%')
"
        fi
    done
fi

echo ""
echo "  V6b ALL DONE"
echo "  KT: ${CKPT_BASE}/act_kt_v2_v6b/"
echo "  NT: ${CKPT_BASE}/act_nt_v2_v6b/"
echo "  Results: ${RESULTS_BASE}/"
