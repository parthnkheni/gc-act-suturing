# JHU dVRK Deployment — Quickstart Guide

## Step 1: Transfer Files to JHU Machine

From this Jetstream instance:
```bash
bash ~/deploy_package/transfer_files.sh
```
This copies: checkpoints (v2 + GC-ACT + gesture classifier), inference scripts, ACT source code, and dependencies.

## Step 2: Install Dependencies on JHU Machine

The JHU machine should already have ROS and dVRK CRTK. Install Python deps:
```bash
pip install torch torchvision einops timm
```

## Step 3: Check ROS Topics

This is the #1 thing that breaks on a new dVRK setup. Run:
```bash
rostopic list | grep -E "psm|image|jaw"
```

The scripts expect these topics (check `rostopics.py`):
- Left endoscope: `/jhu_daVinci/left/image_raw/compressed`
- PSM1 wrist cam: something like `/endo_psm1/image_raw/compressed`
- PSM2 wrist cam: something like `/endo_psm2/image_raw/compressed`
- PSM1 pose: `/PSM1/setpoint_cp`
- PSM2 pose: `/PSM2/setpoint_cp`
- PSM1 jaw: `/PSM1/jaw/setpoint_js`
- PSM2 jaw: `/PSM2/jaw/setpoint_js`

If topic names differ, edit `rostopics.py` to match.

## Step 4: Dry Run (No Robot Movement)

Verifies models load and produce valid actions:
```bash
# ACT v2
python chained_dvrk_inference.py \
    --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt ~/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt ~/checkpoints/act_kt_v2/policy_best.ckpt \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --dry_run

# GC-ACT
python chained_dvrk_inference_gcact.py \
    --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt ~/checkpoints/act_kt_gcact/policy_best.ckpt \
    --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt \
    --dry_run
```

Both should print action chunk shapes and position ranges without errors.

## Step 5: First Real Trial — Single Subtask, Slow Speed

Start with just needle pickup at half speed:
```bash
python chained_dvrk_inference.py \
    --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
    --subtasks needle_pickup \
    --image_encoder efficientnet_b3 --kl_weight 10 \
    --sleep_rate 0.2
```

Watch the robot. If it moves smoothly toward the needle, you're good. If it jerks or moves in the wrong direction, STOP (Ctrl+C) and check coordinate frames.

## Step 6: Full Pipeline

Once single subtasks work:
```bash
# ACT v2 — full suture
python chained_dvrk_inference.py \
    --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt ~/checkpoints/act_nt_v2/policy_best.ckpt \
    --ckpt_kt ~/checkpoints/act_kt_v2/policy_best.ckpt \
    --image_encoder efficientnet_b3 --kl_weight 10

# GC-ACT — full suture
python chained_dvrk_inference_gcact.py \
    --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
    --ckpt_nt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
    --ckpt_kt ~/checkpoints/act_kt_gcact/policy_best.ckpt \
    --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt
```

## Step 7: Run 10 Trials Per Model

For each trial, record:
- Success/failure per subtask (NP, NT, KT)
- Failure mode if failed (grasp miss, tissue miss, incomplete knot, etc.)
- The scripts auto-save action logs to `~/action_log.json` and gesture logs to `~/gesture_inference_log.json`

Rename logs between trials:
```bash
mv ~/action_log.json ~/action_log_v2_trial1.json
mv ~/gesture_inference_log.json ~/gesture_log_gcact_trial1.json
```

## Troubleshooting — What to Send Back

If something goes wrong, collect these (in order of usefulness):

1. **Terminal output** — copy-paste or screenshot any errors
2. **Action log** — `~/action_log.json` (auto-saved by inference scripts)
3. **Gesture log** — `~/gesture_inference_log.json` (GC-ACT only)
4. **Video** — record endoscope view during trials (phone camera works)
5. **ROS topic names** — `rostopic list | grep -E "psm|image|jaw"`

### Common Issues and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Wrong topic names | "No left camera image" error | Edit `rostopics.py` subscriber names |
| Camera format | Garbled/black images | Check compressed vs raw: `rostopic info <topic>` |
| Coordinate frame | Robot moves opposite direction | Check dVRK Si vs Xi base frame convention |
| Model won't load | CUDA out of memory | Check no other GPU processes: `nvidia-smi` |
| Sequential arms slow | Visibly choppy movement | Replace `run_full_pose_goal` with `servo_cp` (see below) |

### Speed Fix (servo_cp)

If arm movement looks choppy because of sequential execution, try replacing in the main loop:
```python
# BEFORE (sequential, ~5Hz effective):
ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])

# AFTER (simultaneous, ~10Hz):
psm1_app.servo_cp(wp_psm1[j])
psm2_app.servo_cp(wp_psm2[j])
```
Test this on a single subtask first — the API may differ on your dVRK version.

## Trial Logging Template

Copy this for each trial:
```
Trial: [v2/gcact]_trial_[N]
Date: ____
Tissue: ____

NP: [PASS/FAIL] - notes: ____
NT: [PASS/FAIL] - notes: ____
KT: [PASS/FAIL] - notes: ____
E2E: [PASS/FAIL]

Failure mode (if any): ____
Video saved: [yes/no]
Logs saved: [yes/no]
```
