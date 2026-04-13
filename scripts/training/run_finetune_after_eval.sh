#!/bin/bash
# Wait for the 50-episode eval to finish, then launch augmented fine-tuning

echo "[$(date)] Waiting for 50-episode eval (run_eval_50ep.sh) to finish..."

while pgrep -f "run_eval_50ep.sh" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Eval finished. Starting augmented fine-tuning in 10 seconds..."
sleep 10

bash /home/exouser/finetune_augmented.sh
