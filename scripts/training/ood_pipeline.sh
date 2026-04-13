#!/bin/bash
set -o pipefail

# OOD Pipeline: Wait for KT -> Eval KT on tissue 6 -> Train NT if KT is good
# Designed to run overnight (~7 hours total)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

CKPT_BASE=/home/exouser/checkpoints
LOG_DIR=/home/exouser/logs
EVAL_SCRIPT=/home/exouser/scripts/evaluation/offline_eval.py
RESULTS_BASE=/home/exouser/eval_results/ood_kt

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

mkdir -p "$LOG_DIR" "$RESULTS_BASE"

echo "  OOD OVERNIGHT PIPELINE"
echo "  Started: $(date)"

# STEP 1: Wait for KT OOD training to finish
KT_CKPT="${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
echo ""
echo "[STEP 1] Waiting for KT OOD training to complete..."

while true; do
    # Check if training process is still running
    if ! pgrep -f "train_kt_gcact_ood_only.sh" > /dev/null 2>&1 && \
       ! pgrep -f "imitate_episodes.*knot_tying_ood" > /dev/null 2>&1; then
        echo "  Training process finished at $(date)"
        break
    fi
    sleep 60
done

# Check if checkpoint exists
if [ ! -f "$KT_CKPT" ]; then
    # Try best epoch checkpoint
    KT_CKPT=$(ls -t ${CKPT_BASE}/act_kt_gcact_ood/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$KT_CKPT" ] || [ ! -f "$KT_CKPT" ]; then
    echo "ERROR: No KT OOD checkpoint found. Training may have failed."
    echo "Check: ${LOG_DIR}/train_kt_gcact_ood.log"
    exit 1
fi
echo "  KT checkpoint: $KT_CKPT"

# STEP 2: Evaluate KT OOD on held-out tissue 6 (ALL episodes)
echo ""
echo "[STEP 2] Evaluating KT OOD on tissue 6 (held-out test set)..."
echo "  $(date)"

# 2a: Tissue 6  -- raw (no ensemble)
echo "  [2a] Tissue 6, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/tissue6_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 51 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood \
    2>&1 | tee ${LOG_DIR}/eval_kt_ood_tissue6_raw.log

# 2b: Tissue 6  -- with ensemble
echo ""
echo "  [2b] Tissue 6, temporal ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
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
    2>&1 | tee ${LOG_DIR}/eval_kt_ood_tissue6_ensemble.log

# 2c: In-distribution train tissues  -- raw
echo ""
echo "  [2c] In-distribution (train tissues), raw..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key knot_tying_ood \
    2>&1 | tee ${LOG_DIR}/eval_kt_ood_train_raw.log

# 2d: In-distribution train tissues  -- ensemble
echo ""
echo "  [2d] In-distribution (train tissues), ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_kt "$KT_CKPT" \
    --subtasks knot_tying \
    --output_dir "${RESULTS_BASE}/train_tissues_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 2 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key knot_tying_ood \
    2>&1 | tee ${LOG_DIR}/eval_kt_ood_train_ensemble.log

# STEP 3: Check results  -- proceed to NT only if KT is good
echo ""
echo "[STEP 3] Checking KT OOD results..."

# Extract tissue 6 ensemble position error
KT_RESULT=$(python3 -c "
import json, sys
try:
    with open('${RESULTS_BASE}/tissue6_ensemble/summary.json') as f:
        d = json.load(f)
    kt = d.get('knot_tying', {})
    pos = kt.get('pos_l2_mean_mm', {})
    if isinstance(pos, dict):
        val = pos['mean']
    else:
        val = pos
    print(f'{val:.4f}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print('999')
" 2>&1)

echo "  Tissue 6 KT ensemble position error: ${KT_RESULT}mm"

# Decision threshold: 1.5mm (relaxed for OOD  -- in-distribution best was 0.736mm)
KT_GOOD=$(python3 -c "print('yes' if float('${KT_RESULT}') < 1.5 else 'no')")

if [ "$KT_GOOD" != "yes" ]; then
    echo ""
    echo "  KT OOD result ${KT_RESULT}mm > 1.5mm threshold."
    echo "  SKIPPING NT OOD training. Investigate KT results first."
    echo "  Results saved in: ${RESULTS_BASE}/"
    echo ""
    echo "  PIPELINE COMPLETE (KT only)"
    echo "  $(date)"
    exit 0
fi

echo "  KT OOD PASSED (${KT_RESULT}mm < 1.5mm)  -- proceeding to NT OOD training"

# STEP 4: Train GC-ACT NT OOD
echo ""
echo "[STEP 4] Starting GC-ACT NT OOD training..."
echo "  $(date)"

cd /home/exouser/SutureBot/src/act

NT_V2_CKPT="${CKPT_BASE}/act_nt_v2/policy_best.ckpt"
if [ ! -f "$NT_V2_CKPT" ]; then
    NT_V2_CKPT=$(ls -t ${CKPT_BASE}/act_nt_v2/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$NT_V2_CKPT" ] || [ ! -f "$NT_V2_CKPT" ]; then
    echo "ERROR: No NT v2 checkpoint found. Cannot fine-tune NT."
    exit 1
fi

echo "  Base checkpoint: $NT_V2_CKPT"

mkdir -p ${CKPT_BASE}/act_nt_gcact_ood

# Write new start timestamp for monitor
echo "$(date +%s)" > /tmp/ood_kt_train_start

python imitate_episodes.py \
    --task_name needle_throw_ood \
    --ckpt_dir ${CKPT_BASE}/act_nt_gcact_ood \
    --resume_ckpt "$NT_V2_CKPT" \
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
    2>&1 | tee ${LOG_DIR}/train_nt_gcact_ood.log

NT_EXIT=$?

echo ""
echo "  OOD OVERNIGHT PIPELINE COMPLETE"
echo "  $(date)"
echo "  KT OOD checkpoint: ${CKPT_BASE}/act_kt_gcact_ood/policy_best.ckpt"
echo "  KT OOD tissue 6 result: ${KT_RESULT}mm"
echo "  NT OOD checkpoint: ${CKPT_BASE}/act_nt_gcact_ood/policy_best.ckpt"
echo "  NT OOD exit code: $NT_EXIT"
echo "  Results: ${RESULTS_BASE}/"
