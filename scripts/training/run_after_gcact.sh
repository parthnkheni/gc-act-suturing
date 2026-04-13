#!/bin/bash
# Post-GC-ACT training pipeline: eval + visual validation for all model variants
# Run this after train_gcact.sh completes (or let it wait for training to finish)

set -e

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

cd /home/exouser

echo "  Post-Training Pipeline: Eval + Visual Validation"
echo "  $(date)"

# Wait for GC-ACT training if still running
GCACT_PID=$(pgrep -f "train_gcact.sh" || true)
if [ -n "$GCACT_PID" ]; then
    echo "GC-ACT training still running (PID: $GCACT_PID). Waiting..."
    while kill -0 "$GCACT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "GC-ACT training finished. Continuing..."
    sleep 5
fi

# Verify GC-ACT checkpoints exist
if [ ! -f ~/checkpoints/act_nt_gcact/policy_best.ckpt ]; then
    echo "ERROR: GC-ACT NT checkpoint not found. Training may have failed."
    echo "Check ~/logs/train_nt_gcact.log"
    exit 1
fi
if [ ! -f ~/checkpoints/act_kt_gcact/policy_best.ckpt ]; then
    echo "ERROR: GC-ACT KT checkpoint not found. Training may have failed."
    echo "Check ~/logs/train_kt_gcact.log"
    exit 1
fi
echo "All checkpoints verified."

# STEP 1: Run full offline eval (v1 vs v2 vs GC-ACT)
echo ""
echo "  STEP 1: Offline Evaluation"

bash ~/run_eval_gcact.sh

# STEP 2: Visual validation  -- v2 (side-by-side)
echo ""
echo "  STEP 2: Visual Validation  -- v2"

mkdir -p ~/paper_results/v2_visual_validation

python generate_visual_validation_sidebyside.py \
    --eval_dir ~/offline_eval_results_v2_raw \
    --output_dir ~/paper_results/v2_visual_validation \
    --video

echo "v2 visual validation saved to ~/paper_results/v2_visual_validation/"

# STEP 3: Visual validation  -- GC-ACT (side-by-side)
echo ""
echo "  STEP 3: Visual Validation  -- GC-ACT"

mkdir -p ~/paper_results/gcact_visual_validation

python generate_visual_validation_sidebyside.py \
    --eval_dir ~/offline_eval_results_gcact_raw \
    --output_dir ~/paper_results/gcact_visual_validation \
    --video

echo "GC-ACT visual validation saved to ~/paper_results/gcact_visual_validation/"

# STEP 4: Print summary
echo ""
echo "  ALL DONE  -- $(date)"
echo ""
echo "  Eval results:"
echo "    v1 ensemble:     ~/offline_eval_results_v1_ensemble/decision.json"
echo "    v2 raw:          ~/offline_eval_results_v2_raw/decision.json"
echo "    v2 ensemble:     ~/offline_eval_results_v2_ensemble/decision.json"
echo "    GC-ACT raw:      ~/offline_eval_results_gcact_raw/decision.json"
echo "    GC-ACT ensemble: ~/offline_eval_results_gcact_ensemble/decision.json"
echo ""
echo "  Visual validation:"
echo "    v1:    ~/paper_results/v1_visual_validation/"
echo "    v2:    ~/paper_results/v2_visual_validation/"
echo "    GC-ACT: ~/paper_results/gcact_visual_validation/"
echo ""
echo "  Next: fill in paper table placeholders with these numbers"
