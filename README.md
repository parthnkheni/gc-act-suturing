# GC-ACT: Gesture-Conditioned ACT for Autonomous Suturing

Gesture-conditioned Action Chunking Transformer (ACT) policies for autonomous
suturing on the dVRK Si platform. Trained on the SutureBot dataset (1,890
episodes, 10 tissues) across three subtasks: needle pickup (NP), needle throw
(NT), and knot tying (KT).

## Repository Layout

```
gcact_final/                Canonical, organized codebase
  01_model_architecture/    DETR / ACT model + backbones (ResNet, EfficientNet)
  02_training/              Training loop, dataset loader, augmentations
  03_inference/             dVRK inference + chained NP->NT->KT pipeline
  04_evaluation/            Offline evaluation harness
  05_visualization/         Architecture diagrams, replay videos
  06_data_processing/       Per-subtask normalization stats
  07_deployment/            JHU deployment package (setup, transfer, verify)
  08_gesture_classifier/    ResNet-18 gesture classifier (G1-G16)
  09_config_and_utils/      Constants and helpers

scripts/                    Working scripts used across the project
  training/                 Training shell scripts (v1, v2, GC-ACT, OOD)
  evaluation/               Eval runners and 50-episode definitive runs
  inference/                Multi-stitch, parameter sweep, ORBIT chained
  tools/                    GEARS quality monitor, wound detection, utilities
  visualization/            Architecture diagrams, Gantt charts, replay videos

run.py                      Unified evaluation entry point
check_training.py           Training progress monitor
```

## Best Models

| Subtask | Model              | KT Error (mm) |
|---------|--------------------|---------------|
| NP      | v2 ensemble        | 0.809         |
| NT      | GC-ACT aug ensemble| 0.803         |
| KT      | GC-ACT aug ensemble| 0.707         |

## Environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate act
```

Trained on a single A100 (40 GB). Inference target: JHU dVRK Si.

## Dataset

SutureBot dataset (CC-BY-4.0): https://huggingface.co/datasets/jchen396/SutureBot
- 1,890 episodes, ~30 Hz, 4 cameras + 149-column kinematics CSV
- Three subtasks per episode: NP, NT, KT
- Gesture labels (JIGSAWS G1-G16) added for NT and KT across all 10 tissues

## Status

All training and offline evaluation complete. Next milestone: real-robot
testing on the JHU dVRK Si.
