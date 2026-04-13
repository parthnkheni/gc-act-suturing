#!/bin/bash
# OOD NT Evaluation  -- Evaluate GC-ACT NT OOD on held-out tissue 6
# Saves: per-episode JSON, ensemble summary, in-distribution summary

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
cd /home/exouser

CKPT="${HOME}/checkpoints/act_nt_gcact_ood/policy_best.ckpt"
EVAL_SCRIPT="${HOME}/scripts/evaluation/offline_eval.py"
RESULTS_BASE="${HOME}/eval_results/ood_nt"

if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -t ${HOME}/checkpoints/act_nt_gcact_ood/policy_best_epoch_*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: No NT OOD checkpoint found"
    exit 1
fi

mkdir -p "$RESULTS_BASE"

echo "  OOD NT Evaluation  -- Tissue 6 (held out)"
echo "  Checkpoint: $CKPT"
echo ""

# 1. Tissue 6  -- raw
echo "[1/4] OOD Test  -- Tissue 6 NT, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/tissue6_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 39 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key needle_throw_ood

echo ""

# 2. Tissue 6  -- ensemble
echo "[2/4] OOD Test  -- Tissue 6 NT, temporal ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/tissue6_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 6 \
    --max_episodes 39 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key needle_throw_ood

echo ""

# 3. Train tissues  -- raw
echo "[3/4] In-distribution  -- Train tissues NT, raw..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/train_tissues_raw" \
    --data_dir /home/exouser/data \
    --tissue_ids 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --norm_stats_key needle_throw_ood

echo ""

# 4. Train tissues  -- ensemble
echo "[4/4] In-distribution  -- Train tissues NT, ensemble..."
python "$EVAL_SCRIPT" \
    --ckpt_nt "$CKPT" \
    --subtasks needle_throw \
    --output_dir "${RESULTS_BASE}/train_tissues_ensemble" \
    --data_dir /home/exouser/data \
    --tissue_ids 3 4 5 7 \
    --max_episodes 50 \
    --step_stride 1 \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --use_gesture --gesture_dim 10 --labels_dir /home/exouser/data/labels \
    --temporal_ensemble_k 0.01 --ensemble_horizon 20 \
    --norm_stats_key needle_throw_ood

echo ""

# Generate per-episode JSONs from results.csv
echo "Generating per-episode JSON files..."
python3 << 'PYEOF'
import os, json, csv

RESULTS_BASE = os.path.expanduser("~/eval_results/ood_nt")

for config_dir in ["tissue6_raw", "tissue6_ensemble", "train_tissues_raw", "train_tissues_ensemble"]:
    csv_path = os.path.join(RESULTS_BASE, config_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping {config_dir}: no results.csv")
        continue

    episodes_dir = os.path.join(RESULTS_BASE, config_dir, "per_episode")
    os.makedirs(episodes_dir, exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        ep_id = row.get("episode_id", "unknown")
        tissue = row.get("tissue", "unknown")
        out = {k: float(v) if k not in ("episode_path", "tissue", "episode_id", "subtask") else v
               for k, v in row.items()}
        out_path = os.path.join(episodes_dir, f"{tissue}_{ep_id}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"  {config_dir}: {len(rows)} per-episode JSONs saved to {episodes_dir}/")

# Also do the same for KT results
RESULTS_KT = os.path.expanduser("~/eval_results/ood_kt")
for config_dir in ["tissue6_raw", "tissue6_ensemble", "train_tissues_raw", "train_tissues_ensemble"]:
    csv_path = os.path.join(RESULTS_KT, config_dir, "results.csv")
    if not os.path.exists(csv_path):
        continue

    episodes_dir = os.path.join(RESULTS_KT, config_dir, "per_episode")
    os.makedirs(episodes_dir, exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        ep_id = row.get("episode_id", "unknown")
        tissue = row.get("tissue", "unknown")
        out = {k: float(v) if k not in ("episode_path", "tissue", "episode_id", "subtask") else v
               for k, v in row.items()}
        out_path = os.path.join(episodes_dir, f"{tissue}_{ep_id}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"  KT {config_dir}: {len(rows)} per-episode JSONs saved")

print("\nDone.")
PYEOF

echo ""
echo "  OOD NT EVAL COMPLETE"
echo "  Results: ${RESULTS_BASE}/"

# Print summary table
echo ""
echo "=== NT OOD SUMMARY ==="
for dir in tissue6_raw tissue6_ensemble train_tissues_raw train_tissues_ensemble; do
    dec="${RESULTS_BASE}/${dir}/decision.json"
    if [ -f "$dec" ]; then
        echo ""
        echo "--- $dir ---"
        python3 -c "
import json
with open('$dec') as f:
    d = json.load(f)
for subtask, m in d.items():
    print(f'  pos={m[\"pos_l2_mm\"]:.3f}mm  rot={m[\"rot_err_deg\"]:.2f}deg  jaw={m[\"jaw_acc_pct\"]:.1f}%  n={m[\"n_episodes\"]}')
"
    fi
done

echo ""
echo "=== KT OOD SUMMARY (for reference) ==="
for dir in tissue6_raw tissue6_ensemble train_tissues_raw train_tissues_ensemble; do
    dec="${HOME}/eval_results/ood_kt/${dir}/decision.json"
    if [ -f "$dec" ]; then
        echo ""
        echo "--- $dir ---"
        python3 -c "
import json
with open('$dec') as f:
    d = json.load(f)
for subtask, m in d.items():
    print(f'  pos={m[\"pos_l2_mm\"]:.3f}mm  rot={m[\"rot_err_deg\"]:.2f}deg  jaw={m[\"jaw_acc_pct\"]:.1f}%  n={m[\"n_episodes\"]}')
"
    fi
done
