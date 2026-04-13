#!/usr/bin/env python3
"""
run.py  -- Unified entry point for the GC-ACT suturing project.
=============================================================

One script to run any model on any tissue/episode. No scattered configs,
no monkey-patching, no guessing which flags to use.

Usage:
    # Run best GC-ACT model on tissue 7, episode 0 (knot tying)
    python run.py --tissue 7 --episode 0 --subtask kt

    # Run v2 model on tissue 5, needle throw, with temporal ensembling
    python run.py --tissue 5 --episode 3 --subtask nt --model v2 --ensemble

    # Run v1 model, no ensembling, save plots
    python run.py --tissue 7 --episode 10 --subtask np --model v1 --save_plots

    # Evaluate across all episodes in tissue 7 (batch mode)
    python run.py --tissue 7 --subtask kt --all_episodes --max_episodes 50

    # List available episodes for a tissue/subtask
    python run.py --tissue 7 --subtask nt --list

    # Use a custom checkpoint
    python run.py --tissue 7 --episode 0 --subtask kt --ckpt ~/checkpoints/some_custom/policy_best.ckpt
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch

# ALL CONFIG IN ONE PLACE  -- no external files needed

PROJECT_ROOT = os.path.expanduser("~")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
SUTUREBOT_SRC = os.path.join(PROJECT_ROOT, "SutureBot", "src")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "run_outputs")

# Subtask name mapping
SUBTASK_ALIASES = {
    "np": "needle_pickup",
    "nt": "needle_throw",
    "kt": "knot_tying",
    "needle_pickup": "needle_pickup",
    "needle_throw": "needle_throw",
    "knot_tying": "knot_tying",
}

SUBTASK_FOLDERS = {
    "needle_pickup": "1_needle_pickup",
    "needle_throw": "2_needle_throw",
    "knot_tying": "3_knot_tying",
}

# MODEL REGISTRY  -- every model version, its checkpoint, and its config

MODEL_CONFIGS = {
    "v1": {
        "description": "ACT v1  -- ResNet18, bs=16, LR=1e-5, KL=1",
        "image_encoder": "resnet18",
        "kl_weight": 1,
        "use_gesture": False,
        "checkpoints": {
            "needle_pickup": os.path.join(CHECKPOINTS_DIR, "act_np_10t_kl1", "policy_best.ckpt"),
            "needle_throw": os.path.join(CHECKPOINTS_DIR, "act_nt_10t_kl1", "policy_best.ckpt"),
            "knot_tying": os.path.join(CHECKPOINTS_DIR, "act_kt_10t_kl1", "policy_best.ckpt"),
        },
    },
    "v2": {
        "description": "ACT v2  -- EfficientNet-B3, bs=256, LR=5e-4, KL=10",
        "image_encoder": "efficientnet_b3",
        "kl_weight": 10,
        "use_gesture": False,
        "checkpoints": {
            "needle_pickup": os.path.join(CHECKPOINTS_DIR, "act_np_v2", "policy_best.ckpt"),
            "needle_throw": os.path.join(CHECKPOINTS_DIR, "act_nt_v2", "policy_best.ckpt"),
            "knot_tying": os.path.join(CHECKPOINTS_DIR, "act_kt_v2", "policy_best.ckpt"),
        },
    },
    "v2_aug": {
        "description": "ACT v2 augmented  -- fine-tuned with aggressive augmentations",
        "image_encoder": "efficientnet_b3",
        "kl_weight": 10,
        "use_gesture": False,
        "checkpoints": {
            "needle_pickup": os.path.join(CHECKPOINTS_DIR, "act_np_v2_aug", "policy_best.ckpt"),
        },
    },
    "gcact": {
        "description": "GC-ACT  -- gesture-conditioned, fine-tuned from v2",
        "image_encoder": "efficientnet_b3",
        "kl_weight": 10,
        "use_gesture": True,
        "checkpoints": {
            "needle_throw": os.path.join(CHECKPOINTS_DIR, "act_nt_gcact", "policy_best.ckpt"),
            "knot_tying": os.path.join(CHECKPOINTS_DIR, "act_kt_gcact", "policy_best.ckpt"),
        },
    },
    "gcact_aug": {
        "description": "GC-ACT augmented  -- best model for NT+KT (0.803mm / 0.707mm)",
        "image_encoder": "efficientnet_b3",
        "kl_weight": 10,
        "use_gesture": True,
        "checkpoints": {
            "needle_throw": os.path.join(CHECKPOINTS_DIR, "act_nt_gcact_aug", "policy_best.ckpt"),
            "knot_tying": os.path.join(CHECKPOINTS_DIR, "act_kt_gcact_aug", "policy_best.ckpt"),
        },
    },
    "best": {
        "description": "Best config for each subtask  -- NP:v2, NT:gcact_aug, KT:gcact_aug",
        "image_encoder": "efficientnet_b3",
        "kl_weight": 10,
        "use_gesture": True,  # for NT/KT; overridden for NP
        "checkpoints": {
            "needle_pickup": os.path.join(CHECKPOINTS_DIR, "act_np_v2", "policy_best.ckpt"),
            "needle_throw": os.path.join(CHECKPOINTS_DIR, "act_nt_gcact_aug", "policy_best.ckpt"),
            "knot_tying": os.path.join(CHECKPOINTS_DIR, "act_kt_gcact_aug", "policy_best.ckpt"),
        },
        # Special: NP uses v2 (no gesture), NT/KT use gcact_aug (gesture)
        "_per_subtask_overrides": {
            "needle_pickup": {"use_gesture": False},
        },
    },
}

# Normalization stats per subtask (computed over all 10 tissues)
NORM_STATS = {
    "needle_pickup": {
        "mean": np.array([ 0.02117121, -0.01061033,  0.04433738,  0.22251957,  0.1971502,
                          -0.90597265, -0.17570475,  0.91686429,  0.13694032,  0.08506552,
                          -0.01144972,  0.00619323,  0.04760892, -0.16096897,  0.45806132,
                          -0.77968459, -0.19713571, -0.78929362, -0.41484009,  0.03692078]),
        "std":  np.array([0.01028275, 0.01613198, 0.03551102, 0.14531227, 0.24641732,
                          0.09483275, 0.19915273, 0.10352518, 0.24363046, 0.41263211,
                          0.01,       0.01569558, 0.02975965, 0.18217486, 0.32791292,
                          0.12508477, 0.21523964, 0.13445343, 0.31883376, 0.33758943]),
    },
    "needle_throw": {
        "mean": np.array([ 0.03055553, -0.0079745,   0.03942896,  0.20950523,  0.2459589,
                          -0.83687713,  0.02511587,  0.84121708,  0.25258415, -0.19935053,
                          -0.00144373,  0.0049653,   0.04490225, -0.16504303,  0.40365707,
                          -0.88094752, -0.16531958, -0.89206828, -0.38058983,  0.10116849]),
        "std":  np.array([0.01,       0.01,       0.01,       0.18217966, 0.37271865,
                          0.1521178,  0.22397943, 0.15792516, 0.39091832, 0.38089521,
                          0.01,       0.01,       0.01,       0.10402391, 0.13901681,
                          0.06005486, 0.0981895,  0.06387606, 0.13532799, 0.11621434]),
    },
    "knot_tying": {
        "mean": np.array([ 0.0249953,  -0.01066993,  0.03099244,  0.35583106,  0.09714196,
                          -0.88176762,  0.25547242,  0.85854408,  0.2132891,  -0.3488589,
                          -0.00550976,  0.00859079,  0.04786375, -0.11541289,  0.52012943,
                          -0.81531284, -0.27246644, -0.80284463, -0.4707253,   0.00315498]),
        "std":  np.array([0.01,       0.01,       0.01,       0.20154828, 0.1873807,
                          0.10344235, 0.31842194, 0.1565692,  0.1619791,  0.01224801,
                          0.01,       0.01,       0.01,       0.15112859, 0.13838071,
                          0.09706259, 0.1765505,  0.10569934, 0.13144507, 0.33264002]),
    },
}

# Gesture labels
GESTURE_LABELS = ["G2", "G3", "G6", "G7", "G10", "G11", "G13", "G14", "G15", "G16"]
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURE_LABELS)}

# CSV column names
ACTION_COLS_PSM1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                    "psm1_sp.orientation.x", "psm1_sp.orientation.y",
                    "psm1_sp.orientation.z", "psm1_sp.orientation.w", "psm1_jaw_sp"]
ACTION_COLS_PSM2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                    "psm2_sp.orientation.x", "psm2_sp.orientation.y",
                    "psm2_sp.orientation.z", "psm2_sp.orientation.w", "psm2_jaw_sp"]


# DATA FUNCTIONS

def find_episodes(tissue_id, subtask_name):
    """Find all episode directories for a tissue/subtask. Returns sorted list of dicts."""
    tissue_dir = os.path.join(DATA_DIR, f"tissue_{tissue_id}")
    folder_name = SUBTASK_FOLDERS[subtask_name]
    subtask_dir = os.path.join(tissue_dir, folder_name)

    if not os.path.isdir(subtask_dir):
        return []

    episodes = []
    for i, ep_dir in enumerate(sorted(os.listdir(subtask_dir))):
        ep_path = os.path.join(subtask_dir, ep_dir)
        if not os.path.isdir(ep_path):
            continue
        csv_path = os.path.join(ep_path, "ee_csv.csv")
        left_dir = os.path.join(ep_path, "left_img_dir")
        if os.path.exists(csv_path) and os.path.isdir(left_dir):
            episodes.append({
                "index": i,
                "path": ep_path,
                "tissue": f"tissue_{tissue_id}",
                "tissue_id": tissue_id,
                "subtask": subtask_name,
                "episode_id": ep_dir,
            })
    return episodes


def load_episode_csv(episode_path):
    """Load ee_csv.csv with column name fallback handling."""
    import pandas as pd
    csv_path = os.path.join(episode_path, "ee_csv.csv")
    df = pd.read_csv(csv_path)

    # Handle alternative column names
    required = ACTION_COLS_PSM1 + ACTION_COLS_PSM2
    missing = [c for c in required if c not in df.columns]
    if missing:
        alt_map = {}
        for col in missing:
            if "_sp." in col:
                alt = col.replace("_sp.", "_sp_pose.")
                if alt in df.columns:
                    alt_map[col] = alt
            elif "_jaw_sp" in col:
                alt = col.replace("_jaw_sp", "_sp_jaw")
                if alt in df.columns:
                    alt_map[col] = alt
        if alt_map:
            df = df.rename(columns={v: k for k, v in alt_map.items()})
    return df


def load_images(episode_path, frame_idx):
    """Load 3 camera images for a frame. Returns dict or None."""
    import cv2
    cam_map = {
        "left":        ("left_img_dir", f"frame{frame_idx:06d}_left.jpg"),
        "right_wrist": ("endo_psm1",    f"frame{frame_idx:06d}_psm1.jpg"),
        "left_wrist":  ("endo_psm2",    f"frame{frame_idx:06d}_psm2.jpg"),
    }
    images = {}
    for act_name, (subdir, fname) in cam_map.items():
        path = os.path.join(episode_path, subdir, fname)
        img = cv2.imread(path)
        if img is None:
            return None
        images[act_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images


def quat_to_rot6d(quat_xyzw):
    """Convert quaternion (xyzw) to 6D rotation."""
    from scipy.spatial.transform import Rotation as Rot
    mat = Rot.from_quat(quat_xyzw).as_matrix()
    return mat[:, :2].T.flatten()


def extract_gt_action_20d(row):
    """Extract 20D ground truth from CSV row (setpoint columns)."""
    action = np.zeros(20, dtype=np.float64)
    # PSM1
    pos1 = np.array([row[c] for c in ACTION_COLS_PSM1[:3]])
    quat1 = np.array([row[c] for c in ACTION_COLS_PSM1[3:7]])
    action[0:3] = pos1
    action[3:9] = quat_to_rot6d(quat1)
    action[9] = row[ACTION_COLS_PSM1[7]]
    # PSM2
    pos2 = np.array([row[c] for c in ACTION_COLS_PSM2[:3]])
    quat2 = np.array([row[c] for c in ACTION_COLS_PSM2[3:7]])
    action[10:13] = pos2
    action[13:19] = quat_to_rot6d(quat2)
    action[19] = row[ACTION_COLS_PSM2[7]]
    return action


def normalize_action(action_20d, subtask_name):
    """Normalize positions + jaw. Rotations stay raw."""
    mean = NORM_STATS[subtask_name]["mean"]
    std = NORM_STATS[subtask_name]["std"]
    normalized = (action_20d - mean) / std
    normalized[3:9] = action_20d[3:9]
    normalized[13:19] = action_20d[13:19]
    return normalized


def denormalize_action(action_normalized, subtask_name):
    """Reverse normalization."""
    mean = NORM_STATS[subtask_name]["mean"]
    std = NORM_STATS[subtask_name]["std"]
    raw = action_normalized * std + mean
    raw[3:9] = action_normalized[3:9]
    raw[13:19] = action_normalized[13:19]
    return raw


# GESTURE LABELS

def load_gesture_labels(episode_info):
    """Load gesture labels for an episode. Returns list of (start, end, gesture) or None."""
    tissue = episode_info["tissue"]
    subtask = episode_info["subtask"]
    episode_id = episode_info["episode_id"]
    folder_name = SUBTASK_FOLDERS[subtask]

    for variant in [folder_name, folder_name + "_recovery"]:
        path = os.path.join(LABELS_DIR, tissue, variant, f"{episode_id}_labels.txt")
        if os.path.exists(path):
            gestures = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        gestures.append((int(parts[0]), int(parts[1]), parts[2]))
            return gestures
    return None


def get_gesture_at_frame(gestures, frame_idx):
    """Find gesture active at frame. Returns string or None."""
    if gestures is None:
        return None
    for start, end, gesture in gestures:
        if start <= frame_idx <= end:
            return gesture
    return None


def gesture_to_onehot(gesture_str, dim=10):
    """Convert gesture string to one-hot vector."""
    onehot = np.zeros(dim, dtype=np.float32)
    if gesture_str and gesture_str in GESTURE_TO_IDX:
        onehot[GESTURE_TO_IDX[gesture_str]] = 1.0
    return onehot


# MODEL LOADING

def load_model(model_name, subtask_name, ckpt_override=None, device="cuda"):
    """Load an ACT model. Returns (policy, config_dict).

    model_name: 'v1', 'v2', 'gcact', 'gcact_aug', 'best', or 'custom'
    """
    import cv2  # ensure available

    if model_name == "custom":
        if ckpt_override is None:
            raise ValueError("--ckpt required when --model custom")
        # Custom: must also pass --image_encoder and --use_gesture flags
        raise ValueError("For custom checkpoints, use offline_eval.py directly")

    config = MODEL_CONFIGS[model_name]

    # Handle per-subtask overrides (e.g., 'best' uses v2 for NP, gcact for NT/KT)
    use_gesture = config["use_gesture"]
    image_encoder = config["image_encoder"]
    kl_weight = config["kl_weight"]
    overrides = config.get("_per_subtask_overrides", {})
    if subtask_name in overrides:
        for k, v in overrides[subtask_name].items():
            if k == "use_gesture":
                use_gesture = v
            elif k == "image_encoder":
                image_encoder = v
            elif k == "kl_weight":
                kl_weight = v

    # Get checkpoint path
    if ckpt_override:
        ckpt_path = ckpt_override
    else:
        ckpts = config["checkpoints"]
        if subtask_name not in ckpts:
            available = list(ckpts.keys())
            raise ValueError(
                f"Model '{model_name}' doesn't have a {subtask_name} checkpoint. "
                f"Available: {available}"
            )
        ckpt_path = ckpts[subtask_name]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build and load ACT policy
    saved_argv = sys.argv
    argv = [
        "act", "--task_name", "sim_needle_pickup",
        "--ckpt_dir", "/tmp", "--policy_class", "ACT",
        "--seed", "0", "--num_epochs", "1",
        "--kl_weight", str(kl_weight), "--chunk_size", "60",
        "--hidden_dim", "512", "--dim_feedforward", "3200",
        "--lr", "1e-5", "--batch_size", "8",
        "--image_encoder", image_encoder,
        "--policy_level", "low",
    ]
    if use_gesture:
        argv.extend(["--use_gesture", "--gesture_dim", "10"])
    sys.argv = argv

    sys.path.insert(0, os.path.join(SUTUREBOT_SRC, "act"))
    sys.path.insert(0, SUTUREBOT_SRC)
    from policy import ACTPolicy

    policy_config = {
        "lr": 1e-5, "num_queries": 60,
        "action_dim": 20, "kl_weight": kl_weight,
        "hidden_dim": 512, "dim_feedforward": 3200,
        "lr_backbone": 1e-5, "backbone": image_encoder,
        "enc_layers": 4, "dec_layers": 7, "nheads": 8,
        "camera_names": ["left", "left_wrist", "right_wrist"],
        "multi_gpu": False,
    }
    if use_gesture:
        policy_config["use_gesture"] = True
        policy_config["gesture_dim"] = 10

    policy = ACTPolicy(policy_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    sys.argv = saved_argv

    return policy, {
        "model_name": model_name,
        "subtask": subtask_name,
        "image_encoder": image_encoder,
        "kl_weight": kl_weight,
        "use_gesture": use_gesture,
        "ckpt_path": ckpt_path,
    }


def preprocess_image(image):
    """Resize, scale, HWC->CHW."""
    import cv2
    img = cv2.resize(image, (480, 360))
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def predict_chunk(policy, images, qpos, gesture_emb=None, use_gesture=False):
    """Run one forward pass. Returns (60, 20) normalized action chunk."""
    with torch.no_grad():
        imgs = np.stack([preprocess_image(images[c])
                         for c in ["left", "left_wrist", "right_wrist"]])
        image_t = torch.from_numpy(imgs).float().cuda().unsqueeze(0)
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        kwargs = {}
        if gesture_emb is not None and use_gesture:
            kwargs["gesture_embedding"] = torch.from_numpy(gesture_emb).float().cuda().unsqueeze(0)

        chunk = policy(qpos_t, image_t, **kwargs)
        return chunk.cpu().numpy()[0]


# METRICS

def rotation_error_degrees(rot6d_a, rot6d_b):
    """Geodesic rotation error in degrees."""
    def to_matrix(r6):
        c1 = r6[0:3] / (np.linalg.norm(r6[0:3]) + 1e-8)
        c2 = r6[3:6] - np.dot(r6[3:6], c1) * c1
        c2 = c2 / (np.linalg.norm(c2) + 1e-8)
        return np.stack([c1, c2, np.cross(c1, c2)], axis=1)

    R_diff = to_matrix(rot6d_a).T @ to_matrix(rot6d_b)
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def compute_episode_metrics(pred_raw, gt_raw):
    """Compute position/rotation/jaw metrics. Both are (T, 20) arrays."""
    T = len(pred_raw)
    pos1 = np.linalg.norm(pred_raw[:, 0:3] - gt_raw[:, 0:3], axis=1) * 1000
    pos2 = np.linalg.norm(pred_raw[:, 10:13] - gt_raw[:, 10:13], axis=1) * 1000
    rot1 = np.array([rotation_error_degrees(pred_raw[t, 3:9], gt_raw[t, 3:9]) for t in range(T)])
    rot2 = np.array([rotation_error_degrees(pred_raw[t, 13:19], gt_raw[t, 13:19]) for t in range(T)])
    jaw1 = ((pred_raw[:, 9] > 0) == (gt_raw[:, 9] > 0)).mean() * 100
    jaw2 = ((pred_raw[:, 19] > 0) == (gt_raw[:, 19] > 0)).mean() * 100

    return {
        "pos_mm": (pos1.mean() + pos2.mean()) / 2,
        "pos_psm1_mm": pos1.mean(),
        "pos_psm2_mm": pos2.mean(),
        "rot_deg": (rot1.mean() + rot2.mean()) / 2,
        "jaw_acc_pct": (jaw1 + jaw2) / 2,
        "num_frames": T,
        "_pos1": pos1, "_pos2": pos2,
        "_rot1": rot1, "_rot2": rot2,
    }


# MAIN EVALUATION LOGIC

def run_episode(policy, model_cfg, episode_info, ensemble=False, ensemble_k=0.01):
    """Run model on a single episode. Returns metrics dict + arrays."""
    ep_path = episode_info["path"]
    subtask = episode_info["subtask"]
    use_gesture = model_cfg["use_gesture"]

    df = load_episode_csv(ep_path)
    T = len(df)
    if T < 10:
        print(f"  Skipping: only {T} frames")
        return None

    # Load gesture labels if needed
    gestures = None
    if use_gesture:
        gestures = load_gesture_labels(episode_info)

    qpos = np.zeros(20, dtype=np.float32)  # training uses zeros

    if ensemble:
        # Temporal ensembling
        chunk_size = 60
        action_buf = np.zeros((T + chunk_size, 20), dtype=np.float64)
        weight_buf = np.zeros(T + chunk_size, dtype=np.float64)
        valid_ts = []

        for t in range(T):
            images = load_images(ep_path, t)
            if images is None:
                continue

            gesture_emb = None
            if use_gesture:
                g = get_gesture_at_frame(gestures, t)
                gesture_emb = gesture_to_onehot(g)

            chunk = predict_chunk(policy, images, qpos, gesture_emb, use_gesture)

            for j in range(chunk_size):
                if t + j >= T + chunk_size:
                    break
                raw = denormalize_action(chunk[j], subtask)
                w = np.exp(-ensemble_k * j)
                action_buf[t + j] += w * raw
                weight_buf[t + j] += w
            valid_ts.append(t)

        pred_list, gt_list = [], []
        for t in valid_ts:
            if weight_buf[t] > 0:
                pred_list.append(action_buf[t] / weight_buf[t])
                gt_list.append(extract_gt_action_20d(df.iloc[t]))
    else:
        # No ensembling  -- first action from each chunk
        pred_list, gt_list = [], []
        for t in range(T):
            images = load_images(ep_path, t)
            if images is None:
                continue

            gesture_emb = None
            if use_gesture:
                g = get_gesture_at_frame(gestures, t)
                gesture_emb = gesture_to_onehot(g)

            chunk = predict_chunk(policy, images, qpos, gesture_emb, use_gesture)
            pred_list.append(denormalize_action(chunk[0], subtask))
            gt_list.append(extract_gt_action_20d(df.iloc[t]))

    if len(pred_list) < 5:
        print(f"  Skipping: only {len(pred_list)} valid frames")
        return None

    pred = np.array(pred_list)
    gt = np.array(gt_list)
    metrics = compute_episode_metrics(pred, gt)
    metrics["pred_raw"] = pred
    metrics["gt_raw"] = gt
    return metrics


def save_plots(metrics, episode_info, model_cfg, output_dir):
    """Save trajectory comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = metrics["pred_raw"]
    gt = metrics["gt_raw"]
    T = len(pred)
    ts = np.arange(T)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    title = (f"{model_cfg['model_name'].upper()} | {episode_info['tissue']} | "
             f"{episode_info['subtask']} | ep {episode_info['episode_id']}\n"
             f"Position: {metrics['pos_mm']:.2f}mm | Rotation: {metrics['rot_deg']:.1f}deg | "
             f"Jaw: {metrics['jaw_acc_pct']:.0f}%")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for i, label in enumerate(["X", "Y", "Z"]):
        axes[0, 0].plot(ts, gt[:, i]*1000, "-", label=f"gt {label}", alpha=0.8)
        axes[0, 0].plot(ts, pred[:, i]*1000, "--", label=f"pred {label}", alpha=0.8)
    axes[0, 0].set_title("PSM1 Position (mm)")
    axes[0, 0].legend(fontsize=7)

    for i, label in enumerate(["X", "Y", "Z"]):
        axes[0, 1].plot(ts, gt[:, 10+i]*1000, "-", label=f"gt {label}", alpha=0.8)
        axes[0, 1].plot(ts, pred[:, 10+i]*1000, "--", label=f"pred {label}", alpha=0.8)
    axes[0, 1].set_title("PSM2 Position (mm)")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(ts, metrics["_pos1"], "r-")
    axes[1, 0].axhline(1.0, color="g", ls="--", alpha=0.5)
    axes[1, 0].set_title("PSM1 Position Error (mm)")

    axes[1, 1].plot(ts, metrics["_pos2"], "b-")
    axes[1, 1].axhline(1.0, color="g", ls="--", alpha=0.5)
    axes[1, 1].set_title("PSM2 Position Error (mm)")

    axes[2, 0].plot(ts, metrics["_rot1"], "r-")
    axes[2, 0].axhline(5.0, color="g", ls="--", alpha=0.5)
    axes[2, 0].set_title("PSM1 Rotation Error (deg)")

    axes[2, 1].plot(ts, metrics["_rot2"], "b-")
    axes[2, 1].axhline(5.0, color="g", ls="--", alpha=0.5)
    axes[2, 1].set_title("PSM2 Rotation Error (deg)")

    plt.tight_layout()
    fname = f"{model_cfg['model_name']}_{episode_info['tissue']}_{episode_info['subtask']}_ep{episode_info['index']:03d}.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {os.path.join(output_dir, fname)}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Unified GC-ACT runner  -- one command for any model, tissue, episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --tissue 7 --episode 0 --subtask kt
  python run.py --tissue 7 --episode 0 --subtask kt --model v1
  python run.py --tissue 5 --subtask nt --all_episodes --ensemble
  python run.py --tissue 7 --subtask kt --list
  python run.py --models
        """,
    )
    parser.add_argument("--tissue", type=int, help="Tissue number (1-10)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index (0-based, sorted by timestamp)")
    parser.add_argument("--subtask", type=str, default=None,
                        choices=list(SUBTASK_ALIASES.keys()),
                        help="Subtask: np, nt, kt (or full names)")
    parser.add_argument("--model", type=str, default="best",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model version (default: best)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable temporal ensembling (k=0.01)")
    parser.add_argument("--ensemble_k", type=float, default=0.01,
                        help="Ensembling decay factor (default: 0.01)")
    parser.add_argument("--all_episodes", action="store_true",
                        help="Run on all episodes (batch mode)")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="Max episodes in batch mode (default: 50)")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save trajectory comparison plots")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="List available episodes and exit")
    parser.add_argument("--models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # --models: just print model registry
    if args.models:
        print("\nAvailable models:\n")
        for name, cfg in MODEL_CONFIGS.items():
            print(f"  {name:12s}  {cfg['description']}")
            for st, cp in cfg["checkpoints"].items():
                exists = "OK" if os.path.exists(cp) else "MISSING"
                print(f"               {st:16s} [{exists}] {cp}")
        print()
        return

    # Validate required args
    if args.tissue is None:
        parser.error("--tissue is required (use --models to list model configs)")
    if args.subtask is None:
        parser.error("--subtask is required (np, nt, or kt)")

    subtask_name = SUBTASK_ALIASES[args.subtask]

    # --list: show episodes and exit
    if args.list:
        episodes = find_episodes(args.tissue, subtask_name)
        print(f"\nTissue {args.tissue} / {subtask_name}: {len(episodes)} episodes\n")
        for ep in episodes:
            # Count frames
            csv_path = os.path.join(ep["path"], "ee_csv.csv")
            try:
                import pandas as pd
                nframes = len(pd.read_csv(csv_path))
            except Exception:
                nframes = "?"
            print(f"  [{ep['index']:3d}]  {ep['episode_id']}  ({nframes} frames)")
        print()
        return

    # Find the target episode(s)
    episodes = find_episodes(args.tissue, subtask_name)
    if not episodes:
        print(f"\nNo episodes found for tissue {args.tissue} / {subtask_name}")
        print(f"Check: {os.path.join(DATA_DIR, f'tissue_{args.tissue}', SUBTASK_FOLDERS[subtask_name])}")
        sys.exit(1)

    if args.all_episodes:
        target_episodes = episodes[:args.max_episodes]
    elif args.episode is not None:
        if args.episode >= len(episodes):
            print(f"\nEpisode {args.episode} out of range. {len(episodes)} episodes available (0-{len(episodes)-1}).")
            print(f"Use --list to see them.")
            sys.exit(1)
        target_episodes = [episodes[args.episode]]
    else:
        parser.error("Specify --episode N or --all_episodes")

    # Load model
    print(f"\n{'='*60}")
    print(f"  Model:    {args.model} ({MODEL_CONFIGS[args.model]['description']})")
    print(f"  Subtask:  {subtask_name}")
    print(f"  Tissue:   {args.tissue}")
    print(f"  Episodes: {len(target_episodes)}")
    print(f"  Ensemble: {'YES (k=' + str(args.ensemble_k) + ')' if args.ensemble else 'NO'}")
    print(f"{'='*60}\n")

    policy, model_cfg = load_model(args.model, subtask_name,
                                   ckpt_override=args.ckpt, device=args.device)

    # Run
    all_metrics = []
    for i, ep in enumerate(target_episodes):
        label = f"[{i+1}/{len(target_episodes)}] tissue_{args.tissue} / {ep['episode_id']}"
        print(f"  {label}")

        metrics = run_episode(policy, model_cfg, ep,
                              ensemble=args.ensemble, ensemble_k=args.ensemble_k)
        if metrics is None:
            continue

        print(f"    Position: {metrics['pos_mm']:.3f} mm  |  "
              f"Rotation: {metrics['rot_deg']:.1f} deg  |  "
              f"Jaw: {metrics['jaw_acc_pct']:.0f}%  |  "
              f"Frames: {metrics['num_frames']}")

        if args.save_plots:
            save_plots(metrics, ep, model_cfg, os.path.join(args.output_dir, "plots"))

        all_metrics.append({
            "tissue": ep["tissue"],
            "episode_id": ep["episode_id"],
            "episode_index": ep["index"],
            "pos_mm": metrics["pos_mm"],
            "pos_psm1_mm": metrics["pos_psm1_mm"],
            "pos_psm2_mm": metrics["pos_psm2_mm"],
            "rot_deg": metrics["rot_deg"],
            "jaw_acc_pct": metrics["jaw_acc_pct"],
            "num_frames": metrics["num_frames"],
        })

    # Summary
    if len(all_metrics) > 1:
        pos_vals = [m["pos_mm"] for m in all_metrics]
        rot_vals = [m["rot_deg"] for m in all_metrics]
        print(f"\n{'='*60}")
        print(f"  SUMMARY ({len(all_metrics)} episodes)")
        print(f"  Position: {np.mean(pos_vals):.3f} +/- {np.std(pos_vals):.3f} mm")
        print(f"  Rotation: {np.mean(rot_vals):.1f} +/- {np.std(rot_vals):.1f} deg")
        print(f"{'='*60}")

    # Save results JSON
    if all_metrics:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        summary = {
            "model": args.model,
            "subtask": subtask_name,
            "tissue": args.tissue,
            "ensemble": args.ensemble,
            "ensemble_k": args.ensemble_k if args.ensemble else None,
            "ckpt": model_cfg["ckpt_path"],
            "episodes": all_metrics,
        }
        if len(all_metrics) > 1:
            summary["mean_pos_mm"] = float(np.mean(pos_vals))
            summary["std_pos_mm"] = float(np.std(pos_vals))
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved: {results_path}")

    print()


if __name__ == "__main__":
    main()
