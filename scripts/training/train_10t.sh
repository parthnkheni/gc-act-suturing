#!/bin/bash
# Train 3 per-subtask ACT models on ALL 10 TISSUES concurrently on A100-40GB
# Previous training used 9 tissues (held out tissue 7). This uses all 10 for max coverage.
# Norm stats are unchanged (already computed on all 10 tissues).

source ~/miniforge3/etc/profile.d/conda.sh
conda activate act

export PATH_TO_DATASET=/home/exouser/data
export PATH_TO_SUTUREBOT=/home/exouser/SutureBot

cd /home/exouser/SutureBot/src/act

COMMON_ARGS="--policy_class ACT --kl_weight 1 --chunk_size 60 --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --batch_size 16 --image_encoder resnet18 --num_epochs 2000 --seed 0"

echo "[$(date)] Starting all 3 training runs (10 tissues) concurrently..."

# Needle Pickup on GPU 0
python imitate_episodes.py \
    --task_name needle_pickup_all \
    --ckpt_dir /home/exouser/checkpoints/act_np_10t_kl1 \
    --gpu 0 \
    $COMMON_ARGS > /home/exouser/logs/train_np_10t.log 2>&1 &
PID_NP=$!
echo "[$(date)] NP started (PID: $PID_NP)"

# Needle Throw on GPU 0
python imitate_episodes.py \
    --task_name needle_throw_all \
    --ckpt_dir /home/exouser/checkpoints/act_nt_10t_kl1 \
    --gpu 0 \
    $COMMON_ARGS > /home/exouser/logs/train_nt_10t.log 2>&1 &
PID_NT=$!
echo "[$(date)] NT started (PID: $PID_NT)"

# Knot Tying on GPU 0
python imitate_episodes.py \
    --task_name knot_tying_all \
    --ckpt_dir /home/exouser/checkpoints/act_kt_10t_kl1 \
    --gpu 0 \
    $COMMON_ARGS > /home/exouser/logs/train_kt_10t.log 2>&1 &
PID_KT=$!
echo "[$(date)] KT started (PID: $PID_KT)"

echo ""
echo "All 3 training runs launched. PIDs: NP=$PID_NP, NT=$PID_NT, KT=$PID_KT"
echo "Monitor with: tail -f ~/logs/train_np_10t.log ~/logs/train_nt_10t.log ~/logs/train_kt_10t.log"

# Wait for all to complete
wait $PID_NP
echo "[$(date)] NP training DONE (exit code: $?)"
wait $PID_NT
echo "[$(date)] NT training DONE (exit code: $?)"
wait $PID_KT
echo "[$(date)] KT training DONE (exit code: $?)"

echo "[$(date)] All 3 subtask models trained (10 tissues)!"
