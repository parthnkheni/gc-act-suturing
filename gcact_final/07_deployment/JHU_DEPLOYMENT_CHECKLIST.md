# JHU dVRK Deployment Checklist

## 1. Pre-Trip Preparation (before leaving)

### Checkpoints
- [ ] Verify 10-tissue training completed (all 3 runs exit code 0)
- [ ] Run offline eval on multiple tissues to confirm model quality
- [ ] Checkpoints located at:
  - `~/checkpoints/act_np_10t_kl1/policy_best.ckpt`
  - `~/checkpoints/act_nt_10t_kl1/policy_best.ckpt`
  - `~/checkpoints/act_kt_10t_kl1/policy_best.ckpt`

### Scripts
- [ ] `chained_dvrk_inference.py` tested with `--dry_run`
- [ ] Transfer checkpoints + scripts to deployment machine (laptop or JHU workstation)

### Files to transfer
```
~/checkpoints/act_np_10t_kl1/policy_best.ckpt   # ~406MB
~/checkpoints/act_nt_10t_kl1/policy_best.ckpt   # ~406MB
~/checkpoints/act_kt_10t_kl1/policy_best.ckpt   # ~406MB
~/SutureBot/src/act/chained_dvrk_inference.py    # Inference script
~/SutureBot/src/act/dvrk_scripts/               # Robot control
~/SutureBot/src/act/rostopics.py                # ROS topics
~/SutureBot/src/act/policy.py                   # ACT policy
~/SutureBot/src/act/detr/                       # Model architecture
```

### Pack list
- [ ] Deployment laptop (with GPU) or USB drive with checkpoints
- [ ] Power cables
- [ ] This checklist (printed)

---

## 2. On-Site Hardware Setup

### Robot
- [ ] Power on dVRK Si system
- [ ] Home PSM1, PSM2, ECM
- [ ] Verify arms move freely, no errors on console
- [ ] E-stop accessible and tested

### Cameras
- [ ] Left endoscope connected and publishing
- [ ] PSM1 wrist camera (endo_psm1) connected
- [ ] PSM2 wrist camera (endo_psm2) connected
- [ ] Verify image quality: lighting similar to training data, no heavy reflections

### Tissue
- [ ] Place tissue pad on work surface
- [ ] Position within camera field of view (all 3 cameras can see workspace)
- [ ] Place suture needle (thread attached) in accessible position for PSM

---

## 3. Software Verification

```bash
# Activate environment
conda activate act
export PATH_TO_SUTUREBOT=/path/to/SutureBot

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify ROS topics are publishing
rostopic list | grep -E "PSM|image|jaw"
# Expected:
#   /jhu_daVinci/left/image_raw/compressed
#   /PSM1/endoscope_img/compressed
#   /PSM2/endoscope_img/compressed
#   /PSM1/setpoint_cp
#   /PSM2/setpoint_cp
#   PSM1/jaw/measured_js
#   PSM2/jaw/measured_js

# Check camera images are arriving
rostopic hz /jhu_daVinci/left/image_raw/compressed
# Should show ~30Hz

# Check robot state
rostopic echo /PSM1/setpoint_cp -n 1
# Should show position x,y,z and orientation x,y,z,w

# Load checkpoints without moving robot
python chained_dvrk_inference.py \
    --ckpt_np /path/to/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /path/to/act_nt_10t_kl1/policy_best.ckpt \
    --ckpt_kt /path/to/act_kt_10t_kl1/policy_best.ckpt \
    --dry_run
```

---

## 4. Pre-Flight Safety Checks

- [ ] Arms at home/safe position
- [ ] Workspace clear of obstacles
- [ ] E-stop within reach
- [ ] Someone watching the robot at all times
- [ ] Camera recording running (for later analysis)
- [ ] Start with single subtask first (not full chain)

---

## 5. Execution — Incremental Testing

### Step A: Needle Pickup only
```bash
python chained_dvrk_inference.py \
    --ckpt_np /path/to/act_np_10t_kl1/policy_best.ckpt \
    --subtasks needle_pickup \
    --np_steps 300 \
    --action_horizon 20 \
    --sleep_rate 0.1
```
**What to watch for:**
- Does PSM move toward the needle?
- Is the motion smooth or jerky?
- Does the gripper close at the right time?
- Press Ctrl+C if anything looks wrong

### Step B: Needle Pickup + Throw
```bash
python chained_dvrk_inference.py \
    --ckpt_np /path/to/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /path/to/act_nt_10t_kl1/policy_best.ckpt \
    --subtasks needle_pickup needle_throw \
    --np_steps 300 --nt_steps 600
```

### Step C: Full chain
```bash
python chained_dvrk_inference.py \
    --ckpt_np /path/to/act_np_10t_kl1/policy_best.ckpt \
    --ckpt_nt /path/to/act_nt_10t_kl1/policy_best.ckpt \
    --ckpt_kt /path/to/act_kt_10t_kl1/policy_best.ckpt \
    --np_steps 300 --nt_steps 600 --kt_steps 320
```

---

## 6. Tuning Parameters

| Parameter | Default | If jerky | If too slow | If overshooting |
|-----------|---------|----------|-------------|-----------------|
| `--sleep_rate` | 0.1 | Increase to 0.15-0.2 | Decrease to 0.05 | Increase |
| `--action_horizon` | 20 | Decrease to 10 | Increase to 30 | Decrease to 5-10 |
| `--np_steps` | 300 | — | Decrease | — |

---

## 7. Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Robot doesn't move | ROS connection | Check `rostopic list`, verify CRTK |
| Jerky motion | Action horizon too large | Reduce `--action_horizon` to 10 |
| Wrong direction | Camera mapping wrong | Verify left=endoscope, left_wrist=PSM2, right_wrist=PSM1 |
| Positions way off | Norm stats mismatch | Verify checkpoints match task configs |
| Gripper never closes | Jaw angle sign wrong | Check jaw range in `--dry_run` output |
| Script crashes on load | Missing dependency | `pip install scipy scikit-learn einops` |
| Images look different | Lighting/camera change | Adjust lighting to match training conditions |

---

## 8. Data Collection During Deployment

- [ ] Record video of every run (phone or screen recording)
- [ ] Save terminal output (tee to log file)
- [ ] Note which parameters worked best
- [ ] Save any captured frames for qualitative comparison with training data
