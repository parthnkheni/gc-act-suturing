#!/bin/bash
set -o pipefail

# V6 Watchdog: Intercepts the original v6 script after Phase 1a,
# runs quick eval on v2 checkpoints, then continues with GC-ACT phases.
#
# Flow:
#   1. Wait for Phase 1a (v2 KT) to finish
#   2. Kill original v6 script (PID 6209) before it starts Phase 1b
#   3. Quick eval: v2 KT on tissue 6 (3 eps, raw + ensemble)
#   4. Phase 1b: GC-ACT KT fine-tune
#   5. Phase 2a: v2 NT from scratch
#   6. Quick eval: v2 NT on tissue 6 (3 eps, raw + ensemble)
#   7. Phase 2b: GC-ACT NT fine-tune

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
EVAL_SCRIPT=/home/exouser/scripts/evaluation/offline_eval.py
RESULTS_BASE=/home/exouser/eval_results/v6_ood

ORIGINAL_PID=6209
TRAIN_PID=6265

mkdir -p "$LOG_DIR" "$RESULTS_BASE"

echo "  V6 Watchdog  -- Eval between phases"
echo "  Monitoring training PID: $TRAIN_PID"
echo "  Will kill original script PID: $ORIGINAL_PID"
echo ""

# Wait for Phase 1a to finish
echo "[$(date)] Waiting for Phase 1a (v2 KT, PID $TRAIN_PID) to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Phase 1a training process exited."

# Kill the original v6 script before it starts Phase 1b
if kill -0 $ORIGINAL_PID 2>/dev/null; then
    echo "[$(date)] Killing original v6 script (PID $ORIGINAL_PID)..."
    kill $ORIGINAL_PID 2>/dev/null
    sleep 2
    # Make sure it's dead
    kill -9 $ORIGINAL_PID 2>/dev/null
    echo "[$(date)] Original script killed."
fi

# Wait for GPU to free up
sleep 10

# Quick eval v2 KT on tissue 6
V2_KT_CKPT="${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_KT_CKPT" ]; then
    V2_KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$V2_KT_CKPT" ] || [ ! -f "$V2_KT_CKPT" ]; then
    echo "ERROR: No v2 KT OOD checkpoint found. Skipping eval."
else
    echo ""
    echo "  Quick Eval: v2 KT OOD on tissue 6 (3 episodes)"
    echo "  Checkpoint: $V2_KT_CKPT"

    cd /home/exouser

    # Raw (no ensemble)
    echo "[$(date)] Eval v2 KT  -- tissue 6, raw, 3 episodes..."
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$V2_KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/v2_kt_tissue6_raw" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --norm_stats_key knot_tying_ood \
        2>&1 | tee ${LOG_DIR}/v6_eval_v2_kt_raw.log

    echo ""

    # Ensemble
    echo "[$(date)] Eval v2 KT  -- tissue 6, ensemble, 3 episodes..."
    python "$EVAL_SCRIPT" \
        --ckpt_kt "$V2_KT_CKPT" \
        --subtasks knot_tying \
        --output_dir "${RESULTS_BASE}/v2_kt_tissue6_ensemble" \
        --data_dir /home/exouser/data \
        --tissue_ids 6 \
        --max_episodes 3 \
        --step_stride 1 \
        --image_encoder efficientnet_b3 --kl_weight 10 \
        --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
        --norm_stats_key knot_tying_ood \
        2>&1 | tee ${LOG_DIR}/v6_eval_v2_kt_ensemble.log

    echo ""
    echo "[$(date)] v2 KT eval complete. Results in ${RESULTS_BASE}/v2_kt_tissue6_*"

    # Print summary
    for dir in v2_kt_tissue6_raw v2_kt_tissue6_ensemble; do
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
fi

# Phase 1b  -- GC-ACT KT OOD
V2_KT_CKPT_FOR_GCACT="${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_KT_CKPT_FOR_GCACT" ]; then
    V2_KT_CKPT_FOR_GCACT=$(ls -t ${CKPT_BASE}/act_kt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$V2_KT_CKPT_FOR_GCACT" ] || [ ! -f "$V2_KT_CKPT_FOR_GCACT" ]; then
    echo "ERROR: No v2 KT OOD checkpoint. Skipping Phase 1b."
else
    echo ""
    echo "[$(date)] PHASE 1b: GC-ACT KT OOD fine-tuning (1000 epochs) from $V2_KT_CKPT_FOR_GCACT..."
    mkdir -p ${CKPT_BASE}/act_kt_gcact_ood_scratch

    cd /home/exouser/SutureBot/src/act

    python imitate_episodes.py \
        --task_name knot_tying_ood \
        --ckpt_dir ${CKPT_BASE}/act_kt_gcact_ood_scratch \
        --resume_ckpt "$V2_KT_CKPT_FOR_GCACT" \
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

    echo "[$(date)] Phase 1b DONE."
fi

# Phase 2a  -- v2 NT OOD from scratch
echo ""
echo "[$(date)] PHASE 2a: v2 NT OOD from scratch (3000 epochs)..."
mkdir -p ${CKPT_BASE}/act_nt_v2_ood_scratch

cd /home/exouser/SutureBot/src/act

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

V2_NT_EXIT=$?
echo "[$(date)] Phase 2a DONE (exit code: $V2_NT_EXIT)"

# Quick eval v2 NT on tissue 6
V2_NT_CKPT="${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_NT_CKPT" ]; then
    V2_NT_CKPT=$(ls -t ${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$V2_NT_CKPT" ] || [ ! -f "$V2_NT_CKPT" ]; then
    echo "ERROR: No v2 NT OOD checkpoint. Skipping eval."
else
    echo ""
    echo "  Quick Eval: v2 NT OOD on tissue 6 (3 episodes)"
    echo "  Checkpoint: $V2_NT_CKPT"

    cd /home/exouser

    # Raw
    echo "[$(date)] Eval v2 NT  -- tissue 6, raw, 3 episodes..."
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

    echo ""

    # Ensemble
    echo "[$(date)] Eval v2 NT  -- tissue 6, ensemble, 3 episodes..."
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

    echo ""
    echo "[$(date)] v2 NT eval complete. Results in ${RESULTS_BASE}/v2_nt_tissue6_*"

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
fi

# Phase 2b  -- GC-ACT NT OOD
V2_NT_CKPT_FOR_GCACT="${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best.ckpt"
if [ ! -f "$V2_NT_CKPT_FOR_GCACT" ]; then
    V2_NT_CKPT_FOR_GCACT=$(ls -t ${CKPT_BASE}/act_nt_v2_ood_scratch/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$V2_NT_CKPT_FOR_GCACT" ] || [ ! -f "$V2_NT_CKPT_FOR_GCACT" ]; then
    echo "ERROR: No v2 NT OOD checkpoint. Skipping Phase 2b."
else
    echo ""
    echo "[$(date)] PHASE 2b: GC-ACT NT OOD fine-tuning (1000 epochs) from $V2_NT_CKPT_FOR_GCACT..."
    mkdir -p ${CKPT_BASE}/act_nt_gcact_ood_scratch

    cd /home/exouser/SutureBot/src/act

    python imitate_episodes.py \
        --task_name needle_throw_ood \
        --ckpt_dir ${CKPT_BASE}/act_nt_gcact_ood_scratch \
        --resume_ckpt "$V2_NT_CKPT_FOR_GCACT" \
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

    echo "[$(date)] Phase 2b DONE."
fi

echo ""
echo "  V6 WATCHDOG COMPLETE  -- All phases + evals done"
echo "  KT v2 OOD:     ${CKPT_BASE}/act_kt_v2_ood_scratch/"
echo "  KT GC-ACT OOD: ${CKPT_BASE}/act_kt_gcact_ood_scratch/"
echo "  NT v2 OOD:     ${CKPT_BASE}/act_nt_v2_ood_scratch/"
echo "  NT GC-ACT OOD: ${CKPT_BASE}/act_nt_gcact_ood_scratch/"
echo "  Eval results:  ${RESULTS_BASE}/"
