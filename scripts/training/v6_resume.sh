#!/bin/bash
set -o pipefail

# V6 Resume: Phase 1b + 2a + eval + 2b
# Phase 1a (v2 KT) already complete. Eval already done.
# Restarting from Phase 1b after OOM failure.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
EVAL_SCRIPT=/home/exouser/scripts/evaluation/offline_eval.py
RESULTS_BASE=/home/exouser/eval_results/v6_ood

mkdir -p "$LOG_DIR" "$RESULTS_BASE"

# Clear any leftover GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

echo "  V6 Resume  -- Phase 1b, 2a, eval, 2b"
echo "  GPU should be free. Starting $(date)"

# Phase 1b: GC-ACT KT OOD
V2_KT_CKPT="${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_KT_CKPT" ]; then
    V2_KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

echo "[$(date)] PHASE 1b: GC-ACT KT OOD (1000 epochs) from $V2_KT_CKPT..."
mkdir -p ${CKPT_BASE}/act_kt_gcact_ood_scratch

cd /home/exouser/SutureBot/src/act

python imitate_episodes.py \
    --task_name knot_tying_ood \
    --ckpt_dir ${CKPT_BASE}/act_kt_gcact_ood_scratch \
    --resume_ckpt "$V2_KT_CKPT" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 1e-4 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 1000 \
    --seed 0 \
    --use_amp \
    --use_gesture \
    --gesture_dim 10 \
    --labels_dir /home/exouser/data/labels \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_v6_kt_gcact_ood.log

echo "[$(date)] Phase 1b DONE (exit: $?)."

# Phase 2a: v2 NT OOD from scratch
echo ""
echo "[$(date)] PHASE 2a: v2 NT OOD from scratch (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_nt_v2_ood_scratch

python imitate_episodes.py \
    --task_name needle_throw_ood \
    --ckpt_dir ${CKPT_BASE}/act_nt_v2_ood_scratch \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --lr 5e-4 \
    --batch_size 32 \
    --grad_accum_steps 8 \
    --image_encoder efficientnet_b3 \
    --num_epochs 3000 \
    --seed 0 \
    --use_amp \
    --gpu 0 \
    2>&1 | tee ${LOG_DIR}/train_v6_nt_v2_ood.log

echo "[$(date)] Phase 2a DONE (exit: $?)."

# Quick eval v2 NT on tissue 6
V2_NT_CKPT="${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_NT_CKPT" ]; then
    V2_NT_CKPT=$(ls -t ${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -n "$V2_NT_CKPT" ] && [ -f "$V2_NT_CKPT" ]; then
    echo ""
    echo "[$(date)] Quick Eval: v2 NT OOD on tissue 6 (3 episodes)..."
    cd /home/exouser

    python "$EVAL_SCRIPT" \
        --ckpt_nt "$V2_NT_CKPT" \
        --subtasks needle_throw \
        --output_dir "${RESULTS_BASE}/v2_nt_tissue6_raw" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --norm_stats_key needle_throw_ood \
        2>&1 | tee ${LOG_DIR}/v6_eval_v2_nt_raw.log

    python "$EVAL_SCRIPT" \
        --ckpt_nt "$V2_NT_CKPT" \
        --subtasks needle_throw \
        --output_dir "${RESULTS_BASE}/v2_nt_tissue6_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key needle_throw_ood \
        2>&1 | tee ${LOG_DIR}/v6_eval_v2_nt_ensemble.log

    echo "[$(date)] v2 NT eval done."

    for dir in v2_nt_tissue6_raw v2_nt_tissue6_ensemble; do
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
        print(f'  {subtask}: pos={pos[\"mean\"]:.3f}±{pos[\"std\"]:.3f}mm  rot={rot[\"mean\"]:.2f}°  jaw={jaw[\"mean\"]:.1f}%')
    else:
        print(f'  {subtask}: pos={pos:.3f}mm  rot={rot:.2f}°  jaw={jaw:.1f}%')
"
        fi
    done

    cd /home/exouser/SutureBot/src/act
fi

# Phase 2b: GC-ACT NT OOD
V2_NT_CKPT_GCACT="${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_NT_CKPT_GCACT" ]; then
    V2_NT_CKPT_GCACT=$(ls -t ${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -n "$V2_NT_CKPT_GCACT" ] && [ -f "$V2_NT_CKPT_GCACT" ]; then
    echo ""
    echo "[$(date)] PHASE 2b: GC-ACT NT OOD (1000 epochs) from $V2_NT_CKPT_GCACT..."
    mkdir -p ${CKPT_BASE}/act_nt_gcact_ood_scratch

    python imitate_episodes.py \
        --task_name needle_throw_ood \
        --ckpt_dir ${CKPT_BASE}/act_nt_gcact_ood_scratch \
        --resume_ckpt "$V2_NT_CKPT_GCACT" \
        --policy_class ACT \
        --kl_weight 10 \
        --chunk_size 60 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --lr 1e-4 \
        --batch_size 32 \
        --grad_accum_steps 8 \
        --image_encoder efficientnet_b3 \
        --num_epochs 1000 \
        --seed 0 \
        --use_amp \
        --use_gesture \
        --gesture_dim 10 \
        --labels_dir /home/exouser/data/labels \
        --gpu 0 \
        2>&1 | tee ${LOG_DIR}/train_v6_nt_gcact_ood.log

    echo "[$(date)] Phase 2b DONE (exit: $?)."
fi

echo ""
echo "  V6 RESUME COMPLETE  -- $(date)"
echo "  KT GC-ACT OOD: ${CKPT_BASE}/act_kt_gcact_ood_scratch/"
echo "  NT v2 OOD:     ${CKPT_BASE}/act_nt_v2_ood_scratch/"
echo "  NT GC-ACT OOD: ${CKPT_BASE}/act_nt_gcact_ood_scratch/"
echo "  Eval results:  ${RESULTS_BASE}/"
