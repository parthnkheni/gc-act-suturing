#!/usr/bin/env python3
"""
Quick test: Run GC-ACT OOD eval with PREDICTED gestures (from classifier)
instead of ground truth labels. Compares both on the same episodes.

Usage:
    conda activate act
    python ~/scripts/evaluation/eval_predicted_gestures.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms

# Setup paths
sys.path.insert(0, os.path.expanduser('~/SutureBot/src/act'))
sys.path.insert(0, os.path.expanduser('~/SutureBot/src'))
sys.path.insert(0, os.path.expanduser('~/scripts/evaluation'))

# Import the existing eval machinery
from offline_eval import (
    ACTPolicyOffline, find_episodes, load_episode_data,
    load_images, extract_qpos_20d, extract_gt_action_20d,
    compute_metrics, load_gesture_labels_for_episode,
    get_gesture_at_frame, gesture_to_onehot, GESTURE_TO_IDX
)

# Gesture classifier (copied from multi_stitch.py)

GESTURE_LABELS = ["G2", "G3", "G6", "G7", "G10", "G11", "G13", "G14", "G15", "G16"]
IDX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_LABELS)}
NUM_GESTURE_CLASSES = len(GESTURE_LABELS)

class GestureClassifier(nn.Module):
    def __init__(self, num_classes=NUM_GESTURE_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

GESTURE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_gesture(classifier, image_rgb, device="cuda"):
    img_tensor = GESTURE_TRANSFORM(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    gesture_str = IDX_TO_GESTURE[pred_idx.item()]
    gesture_onehot = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
    gesture_onehot[pred_idx.item()] = 1.0
    return gesture_str, gesture_onehot, conf.item()

# Evaluate one episode with either GT or predicted gestures

def evaluate_episode_with_gesture_source(policy, episode_info, gesture_source,
                                          gesture_classifier=None, labels_dir=None,
                                          temporal_ensemble_k=0.01, ensemble_horizon=20):
    """
    gesture_source: 'gt' or 'predicted' or 'none'
    Returns metrics dict.
    """
    ep_path = episode_info['path']
    subtask = episode_info['subtask']
    df = load_episode_data(ep_path)
    T = len(df)
    if T < 10:
        return None

    # Load GT labels for comparison even when using predicted
    episode_gestures = None
    if labels_dir:
        episode_gestures = load_gesture_labels_for_episode(labels_dir, episode_info)

    chunk_size = policy.chunk_size
    horizon = min(ensemble_horizon or chunk_size, chunk_size)
    action_dim = policy.action_dim
    action_buffer = np.zeros((T + chunk_size, action_dim), dtype=np.float64)
    weight_buffer = np.zeros(T + chunk_size, dtype=np.float64)
    valid_timesteps = []

    gesture_matches = 0
    gesture_total = 0

    for t in range(0, T, 1):
        images = load_images(ep_path, t)
        if images is None:
            continue

        row = df.iloc[t]
        qpos = extract_qpos_20d(row)

        # Get gesture embedding based on source
        gesture_emb = None
        if gesture_source == 'gt' and episode_gestures is not None:
            gesture_str = get_gesture_at_frame(episode_gestures, t)
            gesture_emb = gesture_to_onehot(gesture_str, policy.gesture_dim)
        elif gesture_source == 'predicted' and gesture_classifier is not None:
            left_img = images['left']  # BGR from cv2
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            pred_str, gesture_emb, conf = predict_gesture(gesture_classifier, left_rgb)
            # Track accuracy vs GT
            if episode_gestures is not None:
                gt_str = get_gesture_at_frame(episode_gestures, t)
                if gt_str is not None:
                    gesture_total += 1
                    if pred_str == gt_str:
                        gesture_matches += 1
        elif gesture_source == 'none':
            gesture_emb = np.zeros(policy.gesture_dim, dtype=np.float32)

        chunk_normalized = policy.predict_chunk(images, qpos, gesture_embedding=gesture_emb)

        for j in range(horizon):
            if t + j >= T + chunk_size:
                break
            action_raw = policy.denormalize_action(chunk_normalized[j])
            w = np.exp(-temporal_ensemble_k * j)
            action_buffer[t + j] += w * action_raw
            weight_buffer[t + j] += w

        valid_timesteps.append(t)

    pred_actions_raw = []
    gt_actions_raw = []
    for t in valid_timesteps:
        if weight_buffer[t] > 0:
            ensembled_action = action_buffer[t] / weight_buffer[t]
            gt_20d = extract_gt_action_20d(df.iloc[t], use_setpoint=True)
            pred_actions_raw.append(ensembled_action)
            gt_actions_raw.append(gt_20d)

    if len(pred_actions_raw) < 5:
        return None

    pred_actions_raw = np.array(pred_actions_raw)
    gt_actions_raw = np.array(gt_actions_raw)

    metrics = compute_metrics(pred_actions_raw, gt_actions_raw, subtask)
    metrics['episode_path'] = ep_path
    metrics['tissue'] = episode_info['tissue']
    metrics['episode_id'] = episode_info['episode_id']
    metrics['num_frames'] = len(valid_timesteps)
    if gesture_total > 0:
        metrics['gesture_accuracy'] = gesture_matches / gesture_total
    return metrics


# MAIN

def run_ablation(model_label, ckpt_path, norm_key, subtask, tissue_ids,
                 classifier, labels_dir, data_dir, num_episodes=3):
    """Run GT vs Predicted vs Zero gesture ablation for one model config."""

    print(f"\n{'#'*70}")
    print(f"  {model_label}")
    print(f"{'#'*70}")

    print(f"\nLoading model from {ckpt_path}...")
    policy = ACTPolicyOffline(
        ckpt_path, norm_key, device="cuda",
        image_encoder="efficientnet_b3", use_gesture=True,
        gesture_dim=10, kl_weight=10
    )

    episodes = find_episodes(data_dir, subtask, max_episodes=num_episodes, tissue_ids=tissue_ids)
    print(f"Found {len(episodes)} episodes, using {min(num_episodes, len(episodes))}")
    episodes = episodes[:num_episodes]

    summary = {}  # source_key -> {pos, rot, jaw, gesture_acc}

    for source_label, source_key in [("GT labels", "gt"), ("Predicted", "predicted"), ("Zero (no gesture)", "none")]:
        print(f"\n{'='*70}")
        print(f"  CONDITION: {source_label}")
        print(f"{'='*70}")

        all_pos = []
        all_rot = []
        all_jaw = []
        all_gesture_acc = []

        for i, ep in enumerate(episodes):
            ep['subtask'] = subtask
            t0 = time.time()
            metrics = evaluate_episode_with_gesture_source(
                policy, ep, gesture_source=source_key,
                gesture_classifier=classifier,
                labels_dir=labels_dir,
                temporal_ensemble_k=0.01, ensemble_horizon=20
            )
            dt = time.time() - t0

            if metrics is None:
                print(f"  [{i+1}/{len(episodes)}] SKIPPED")
                continue

            pos = metrics['pos_l2_mean_mm']
            rot = metrics['rot_err_mean_deg']
            jaw = metrics['jaw_acc_mean_pct']
            all_pos.append(pos)
            all_rot.append(rot)
            all_jaw.append(jaw)

            extra = ""
            if 'gesture_accuracy' in metrics:
                gacc = metrics['gesture_accuracy'] * 100
                all_gesture_acc.append(gacc)
                extra = f", gesture_acc={gacc:.1f}%"

            print(f"  [{i+1}/{len(episodes)}] {ep['episode_id']}: "
                  f"pos={pos:.3f}mm, rot={rot:.2f}°, jaw={jaw:.1f}%{extra}  ({dt:.0f}s)")

        if all_pos:
            summary[source_key] = {
                'pos_mean': np.mean(all_pos), 'pos_std': np.std(all_pos),
                'rot_mean': np.mean(all_rot), 'rot_std': np.std(all_rot),
                'jaw_mean': np.mean(all_jaw), 'jaw_std': np.std(all_jaw),
                'gesture_acc': np.mean(all_gesture_acc) if all_gesture_acc else None,
                'n': len(all_pos),
            }
            print(f"\n  --- {source_label} Summary ({len(all_pos)} episodes) ---")
            print(f"  Position:  {np.mean(all_pos):.3f} +/- {np.std(all_pos):.3f} mm")
            print(f"  Rotation:  {np.mean(all_rot):.2f} +/- {np.std(all_rot):.2f} deg")
            print(f"  Jaw:       {np.mean(all_jaw):.1f} +/- {np.std(all_jaw):.1f}%")
            if all_gesture_acc:
                print(f"  Gesture accuracy vs GT: {np.mean(all_gesture_acc):.1f}%")

    return summary


def main():
    GESTURE_CKPT = os.path.expanduser("~/checkpoints/gesture_classifier/gesture_best.ckpt")
    DATA_DIR = os.path.expanduser("~/data")
    LABELS_DIR = os.path.expanduser("~/data/labels")
    NUM_EPISODES = 3

    # Load gesture classifier once
    print("Loading gesture classifier...")
    classifier = GestureClassifier(num_classes=NUM_GESTURE_CLASSES)
    gc_ckpt = torch.load(GESTURE_CKPT, map_location="cuda", weights_only=False)
    if "model_state_dict" in gc_ckpt:
        classifier.load_state_dict(gc_ckpt["model_state_dict"])
    else:
        classifier.load_state_dict(gc_ckpt)
    classifier.cuda().eval()
    print(f"  Gesture classifier loaded from {GESTURE_CKPT}")

    # Define all 4 configs: (label, ckpt, norm_key, subtask, tissue_ids)
    configs = [
        ("GC-ACT Aug KT (in-dist, tissue 7)",
         os.path.expanduser("~/checkpoints/act_kt_gcact_aug/policy_best.ckpt"),
         "knot_tying", "knot_tying", [7]),

        ("GC-ACT Aug NT (in-dist, tissue 7)",
         os.path.expanduser("~/checkpoints/act_nt_gcact_aug/policy_best.ckpt"),
         "needle_throw", "needle_throw", [7]),

        ("GC-ACT OOD KT (tissue 6)",
         os.path.expanduser("~/checkpoints/act_kt_gcact_ood/policy_best.ckpt"),
         "knot_tying_ood", "knot_tying", [6]),

        ("GC-ACT OOD NT (tissue 6)",
         os.path.expanduser("~/checkpoints/act_nt_gcact_ood/policy_best.ckpt"),
         "needle_throw_ood", "needle_throw", [6]),
    ]

    all_summaries = {}

    for label, ckpt_path, norm_key, subtask, tissue_ids in configs:
        if not os.path.exists(ckpt_path):
            print(f"\nSKIPPING {label}  -- checkpoint not found: {ckpt_path}")
            continue
        summary = run_ablation(label, ckpt_path, norm_key, subtask, tissue_ids,
                               classifier, LABELS_DIR, DATA_DIR, NUM_EPISODES)
        all_summaries[label] = summary

    # Final summary table
    print(f"\n\n{'#'*70}")
    print("  GESTURE ABLATION  -- FINAL SUMMARY TABLE")
    print(f"{'#'*70}")
    print(f"\n{'Model':<40} {'Condition':<15} {'Pos (mm)':<15} {'Rot (deg)':<15} {'Jaw (%)':<12} {'GestAcc'}")
    print("-" * 110)
    for model_label, summary in all_summaries.items():
        for source_key, source_label in [("gt", "GT labels"), ("predicted", "Predicted"), ("none", "Zero")]:
            if source_key in summary:
                s = summary[source_key]
                gacc = f"{s['gesture_acc']:.1f}%" if s['gesture_acc'] is not None else " -- "
                print(f"{model_label:<40} {source_label:<15} "
                      f"{s['pos_mean']:.3f}±{s['pos_std']:.3f}   "
                      f"{s['rot_mean']:.2f}±{s['rot_std']:.2f}     "
                      f"{s['jaw_mean']:.1f}±{s['jaw_std']:.1f}    "
                      f"{gacc}")
        print()

    print("DONE")


if __name__ == "__main__":
    main()
