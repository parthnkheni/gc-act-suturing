#!/usr/bin/env python3
"""
Offline Evaluation  -- Validate ACT Models on Real SutureBot Episodes
====================================================================
Loads real SutureBot images + kinematics from ee_csv.csv, runs each subtask
model, and compares predicted actions to ground truth.

This is the critical validation step before driving to the real dVRK.

Usage:
    conda run -n orbitsurgical python offline_eval.py \
        --data_dir ~/suturebot_data \
        --ckpt_np ~/checkpoints/act_np_all10_kl1/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_all10_kl1/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_all10_kl1/policy_best.ckpt \
        --output_dir ~/offline_eval_results

    # Single subtask:
    conda run -n orbitsurgical python offline_eval.py \
        --data_dir ~/suturebot_data \
        --ckpt_kt ~/checkpoints/act_kt_all10_kl1/policy_best.ckpt \
        --subtasks knot_tying --max_episodes 5
"""

import os
import sys
import argparse
import json
import time
import pickle
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from collections import defaultdict

# NORM_STATS  -- must match act_orbit_chained.py exactly
# These are updated by verify_checkpoints.py if they differ
NORM_STATS = {
    'needle_pickup': {
        'mean': np.array([ 0.02117121, -0.01061033,  0.04433738,  0.22251957,  0.1971502 ,
                          -0.90597265, -0.17570475,  0.91686429,  0.13694032,  0.08506552,
                          -0.01144972,  0.00619323,  0.04760892, -0.16096897,  0.45806132,
                          -0.77968459, -0.19713571, -0.78929362, -0.41484009,  0.03692078]),
        'std':  np.array([0.01028275, 0.01613198, 0.03551102, 0.14531227, 0.24641732,
                          0.09483275, 0.19915273, 0.10352518, 0.24363046, 0.41263211,
                          0.01      , 0.01569558, 0.02975965, 0.18217486, 0.32791292,
                          0.12508477, 0.21523964, 0.13445343, 0.31883376, 0.33758943]),
    },
    'needle_throw': {
        'mean': np.array([ 0.03055553, -0.0079745 ,  0.03942896,  0.20950523,  0.2459589 ,
                          -0.83687713,  0.02511587,  0.84121708,  0.25258415, -0.19935053,
                          -0.00144373,  0.0049653 ,  0.04490225, -0.16504303,  0.40365707,
                          -0.88094752, -0.16531958, -0.89206828, -0.38058983,  0.10116849]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.18217966, 0.37271865,
                          0.1521178 , 0.22397943, 0.15792516, 0.39091832, 0.38089521,
                          0.01      , 0.01      , 0.01      , 0.10402391, 0.13901681,
                          0.06005486, 0.0981895 , 0.06387606, 0.13532799, 0.11621434]),
    },
    'knot_tying': {
        'mean': np.array([ 0.0249953 , -0.01066993,  0.03099244,  0.35583106,  0.09714196,
                          -0.88176762,  0.25547242,  0.85854408,  0.2132891 , -0.3488589 ,
                          -0.00550976,  0.00859079,  0.04786375, -0.11541289,  0.52012943,
                          -0.81531284, -0.27246644, -0.80284463, -0.4707253 ,  0.00315498]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.20154828, 0.1873807 ,
                          0.10344235, 0.31842194, 0.1565692 , 0.1619791 , 0.01224801,
                          0.01      , 0.01      , 0.01      , 0.15112859, 0.13838071,
                          0.09706259, 0.1765505 , 0.10569934, 0.13144507, 0.33264002]),
    },
    # OOD splits (tissue 6 held out, tissue 10 val)
    'knot_tying_ood': {
        'mean': np.array([ 0.02498467, -0.01036659,  0.03075692,  0.36082841,  0.09967446,
                          -0.87832993,  0.22352721,  0.87200901,  0.20849486, -0.34882886,
                          -0.00576851,  0.00857972,  0.04765369, -0.11135938,  0.51709467,
                          -0.81725226, -0.27512077, -0.80333178, -0.46751891, -0.0065732 ]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.20359851, 0.1887441 ,
                          0.10643166, 0.31239903, 0.14864324, 0.16272003, 0.01310612,
                          0.01      , 0.01      , 0.01      , 0.15349879, 0.13855457,
                          0.09773277, 0.17803928, 0.10714334, 0.13120596, 0.32292322]),
    },
    'needle_throw_ood': {
        'mean': np.array([ 0.03050782, -0.00796649,  0.03955676,  0.21758988,  0.24469944,
                          -0.83469772,  0.02110674,  0.84074499,  0.25187499, -0.20501632,
                          -0.00171012,  0.00478567,  0.04517474, -0.1601474 ,  0.40466053,
                          -0.88177077, -0.16610946, -0.89150309, -0.38208756,  0.10131299]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.1816347 , 0.3736032 ,
                          0.15325153, 0.223059  , 0.15726658, 0.39341233, 0.37509713,
                          0.01      , 0.01      , 0.01      , 0.10534669, 0.13638394,
                          0.05817613, 0.09948507, 0.06285573, 0.13338243, 0.11248928]),
    },
}

# Subtask folder name mapping
SUBTASK_FOLDERS = {
    'needle_pickup': '1_needle_pickup',
    'needle_throw': '2_needle_throw',
    'knot_tying': '3_knot_tying',
}

# Gesture labels (GC-ACT)  -- 10 classes matching gesture classifier
GESTURE_LABELS = ['G2', 'G3', 'G6', 'G7', 'G10', 'G11', 'G13', 'G14', 'G15', 'G16']
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURE_LABELS)}

# CSV column names from SutureBot ee_csv.csv
QPOS_COLS_PSM1 = [f'psm1_js[{i}]' for i in range(6)] + ['psm1_jaw']
QPOS_COLS_PSM2 = [f'psm2_js[{i}]' for i in range(6)] + ['psm2_jaw']

# Setpoint columns (used as ground truth actions  -- what was commanded)
ACTION_COLS_PSM1 = ['psm1_sp.position.x', 'psm1_sp.position.y', 'psm1_sp.position.z',
                    'psm1_sp.orientation.x', 'psm1_sp.orientation.y',
                    'psm1_sp.orientation.z', 'psm1_sp.orientation.w', 'psm1_jaw_sp']
ACTION_COLS_PSM2 = ['psm2_sp.position.x', 'psm2_sp.position.y', 'psm2_sp.position.z',
                    'psm2_sp.orientation.x', 'psm2_sp.orientation.y',
                    'psm2_sp.orientation.z', 'psm2_sp.orientation.w', 'psm2_jaw_sp']

# Measured pose columns (actual EE state)
POSE_COLS_PSM1 = ['psm1_pose.position.x', 'psm1_pose.position.y', 'psm1_pose.position.z',
                  'psm1_pose.orientation.x', 'psm1_pose.orientation.y',
                  'psm1_pose.orientation.z', 'psm1_pose.orientation.w', 'psm1_jaw']
POSE_COLS_PSM2 = ['psm2_pose.position.x', 'psm2_pose.position.y', 'psm2_pose.position.z',
                  'psm2_pose.orientation.x', 'psm2_pose.orientation.y',
                  'psm2_pose.orientation.z', 'psm2_pose.orientation.w', 'psm2_jaw']


# LIGHTWEIGHT ACT INFERENCE (no Isaac Sim dependency)

class ACTPolicyOffline:
    """ACT model for offline evaluation  -- no sim, no temporal ensembling.

    Supports v1 (resnet18), v2 (efficientnet_b3), and GC-ACT (efficientnet_b3 + gesture).
    """

    def __init__(self, ckpt_path, subtask_name, device="cuda",
                 image_encoder="resnet18", use_gesture=False, gesture_dim=10,
                 kl_weight=1):
        self.device = device
        self.chunk_size = 60
        self.action_dim = 20
        self.state_dim = 20
        self.camera_names = ['left', 'left_wrist', 'right_wrist']
        self.subtask_name = subtask_name
        self.image_encoder = image_encoder
        self.use_gesture = use_gesture
        self.gesture_dim = gesture_dim
        self.kl_weight = kl_weight

        self.action_mean = NORM_STATS[subtask_name]['mean']
        self.action_std = NORM_STATS[subtask_name]['std']

        self._build_and_load(ckpt_path)
        mode = "GC-ACT" if use_gesture else "ACT"
        print(f"[{mode}-Offline] {subtask_name} loaded from {ckpt_path} ({image_encoder})")

    def _build_and_load(self, ckpt_path):
        saved_argv = sys.argv
        argv = [
            'act', '--task_name', 'sim_needle_pickup',
            '--ckpt_dir', '/tmp', '--policy_class', 'ACT',
            '--seed', '0', '--num_epochs', '1',
            '--kl_weight', str(self.kl_weight), '--chunk_size', '60',
            '--hidden_dim', '512', '--dim_feedforward', '3200',
            '--lr', '1e-5', '--batch_size', '8',
            '--image_encoder', self.image_encoder,
            '--policy_level', 'low',
        ]
        if self.use_gesture:
            argv.extend(['--use_gesture', '--gesture_dim', str(self.gesture_dim)])
        sys.argv = argv

        sys.path.insert(0, os.path.expanduser('~/SutureBot/src/act'))
        sys.path.insert(0, os.path.expanduser('~/SutureBot/src'))
        from policy import ACTPolicy

        policy_config = {
            "lr": 1e-5, "num_queries": self.chunk_size,
            "action_dim": self.action_dim, "kl_weight": self.kl_weight,
            "hidden_dim": 512, "dim_feedforward": 3200,
            "lr_backbone": 1e-5, "backbone": self.image_encoder,
            "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": self.camera_names, "multi_gpu": False,
        }
        if self.use_gesture:
            policy_config["use_gesture"] = True
            policy_config["gesture_dim"] = self.gesture_dim

        self.policy = ACTPolicy(policy_config)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.policy.load_state_dict(state_dict)
        self.policy.cuda()
        self.policy.eval()
        sys.argv = saved_argv

    def preprocess_image(self, image):
        """Preprocess image: resize, /255, HWC->CHW. ImageNet norm is inside ACTPolicy."""
        img = cv2.resize(image, (480, 360))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img

    def predict_chunk(self, images, qpos, gesture_embedding=None):
        """Run single forward pass, return full action chunk (60, 20) in NORMALIZED space.

        gesture_embedding: optional (gesture_dim,) numpy array (one-hot) for GC-ACT.
        """
        with torch.no_grad():
            imgs = []
            for cam_name in self.camera_names:
                imgs.append(self.preprocess_image(images[cam_name]))
            image_data = np.stack(imgs, axis=0)
            image_tensor = torch.from_numpy(image_data).float().cuda().unsqueeze(0)
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            kwargs = {}
            if gesture_embedding is not None and self.use_gesture:
                gesture_tensor = torch.from_numpy(gesture_embedding).float().cuda().unsqueeze(0)
                kwargs['gesture_embedding'] = gesture_tensor

            action_chunk = self.policy(qpos_tensor, image_tensor, **kwargs)
            return action_chunk.cpu().numpy()[0]  # (60, 20) normalized

    def denormalize_action(self, action_normalized):
        """Denormalize: positions (0:3, 10:13) and jaw (9, 19). Rotations stay raw."""
        raw = action_normalized * self.action_std + self.action_mean
        # Rotations are NOT normalized, so restore originals
        raw[3:9] = action_normalized[3:9]
        raw[13:19] = action_normalized[13:19]
        return raw


# DATA LOADING

def find_episodes(data_dir, subtask_name, max_episodes=None, tissue_ids=None):
    """Find all episode directories for a given subtask."""
    folder_name = SUBTASK_FOLDERS[subtask_name]
    episodes = []

    # Walk tissue directories
    for tissue_dir in sorted(glob.glob(os.path.join(data_dir, 'tissue_*'))):
        tissue_name = os.path.basename(tissue_dir)
        if tissue_ids is not None:
            tissue_num = int(tissue_name.split('_')[1])
            if tissue_num not in tissue_ids:
                continue

        subtask_dir = os.path.join(tissue_dir, folder_name)
        if not os.path.isdir(subtask_dir):
            continue

        for ep_dir in sorted(os.listdir(subtask_dir)):
            ep_path = os.path.join(subtask_dir, ep_dir)
            if not os.path.isdir(ep_path):
                continue
            csv_path = os.path.join(ep_path, 'ee_csv.csv')
            left_dir = os.path.join(ep_path, 'left_img_dir')
            if os.path.exists(csv_path) and os.path.isdir(left_dir):
                episodes.append({
                    'path': ep_path,
                    'tissue': tissue_name,
                    'subtask': subtask_name,
                    'episode_id': ep_dir,
                })

    if max_episodes is not None and len(episodes) > max_episodes:
        # Sample uniformly across tissues
        np.random.seed(42)
        indices = np.random.choice(len(episodes), max_episodes, replace=False)
        episodes = [episodes[i] for i in sorted(indices)]

    return episodes


def load_episode_data(episode_path):
    """Load images and kinematics for a single episode."""
    csv_path = os.path.join(episode_path, 'ee_csv.csv')
    df = pd.read_csv(csv_path)

    # Verify expected columns exist
    required_cols = ACTION_COLS_PSM1 + ACTION_COLS_PSM2 + POSE_COLS_PSM1 + POSE_COLS_PSM2
    # Some columns may use slightly different names - check with fallback
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Try without 'sp.' prefix  -- check actual column names
        print(f"  Warning: Missing columns: {missing[:5]}...")
        # Try mapping: psm1_sp.position.x -> psm1_sp_pose.position.x
        alt_map = {}
        for col in missing:
            if '_sp.' in col:
                alt = col.replace('_sp.', '_sp_pose.')
                if alt in df.columns:
                    alt_map[col] = alt
            elif '_jaw_sp' in col:
                alt = col.replace('_jaw_sp', '_sp_jaw')
                if alt in df.columns:
                    alt_map[col] = alt
        if alt_map:
            print(f"  Found alternatives: {alt_map}")
            df = df.rename(columns={v: k for k, v in alt_map.items()})
            missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  Still missing: {missing}")

    return df


def quat_to_rot6d(quat_xyzw):
    """Convert quaternion (xyzw) to 6D rotation representation."""
    r = Rot.from_quat(quat_xyzw)
    mat = r.as_matrix()  # (3, 3)
    # First two columns, transposed and flattened: [col1(3), col2(3)]
    rot6d = mat[:, :2].T.flatten()  # (6,)
    return rot6d


def extract_gt_action_20d(row, use_setpoint=True):
    """Extract 20D ground truth action from a CSV row.

    Returns: 20D array [PSM1_pos(3) + PSM1_rot6d(6) + PSM1_jaw(1) +
                         PSM2_pos(3) + PSM2_rot6d(6) + PSM2_jaw(1)]
    """
    if use_setpoint:
        cols1 = ACTION_COLS_PSM1
        cols2 = ACTION_COLS_PSM2
    else:
        cols1 = POSE_COLS_PSM1
        cols2 = POSE_COLS_PSM2

    action_20d = np.zeros(20, dtype=np.float64)

    # PSM1
    pos1 = np.array([row[cols1[0]], row[cols1[1]], row[cols1[2]]])
    quat1 = np.array([row[cols1[3]], row[cols1[4]], row[cols1[5]], row[cols1[6]]])  # xyzw
    jaw1 = row[cols1[7]]
    rot6d1 = quat_to_rot6d(quat1)
    action_20d[0:3] = pos1
    action_20d[3:9] = rot6d1
    action_20d[9] = jaw1

    # PSM2
    pos2 = np.array([row[cols2[0]], row[cols2[1]], row[cols2[2]]])
    quat2 = np.array([row[cols2[3]], row[cols2[4]], row[cols2[5]], row[cols2[6]]])  # xyzw
    jaw2 = row[cols2[7]]
    rot6d2 = quat_to_rot6d(quat2)
    action_20d[10:13] = pos2
    action_20d[13:19] = rot6d2
    action_20d[19] = jaw2

    return action_20d


def normalize_action(action_20d, subtask_name):
    """Normalize a 20D action the same way training does: positions + jaw only."""
    mean = NORM_STATS[subtask_name]['mean']
    std = NORM_STATS[subtask_name]['std']
    normalized = (action_20d - mean) / std
    # Rotations stay raw (unnormalized)
    normalized[3:9] = action_20d[3:9]
    normalized[13:19] = action_20d[13:19]
    return normalized


def extract_qpos_20d(row):
    """Extract 20D qpos  -- returns zeros to match training (generic_dataset.py line 593).

    Training always uses qpos = np.zeros(20), so inference must do the same.
    Feeding real joint states confuses the model and causes systematic position bias.
    """
    qpos = np.zeros(20, dtype=np.float32)
    return qpos


def load_images(episode_path, frame_idx):
    """Load 3 camera images for a given frame index."""
    images = {}

    # Camera mapping: ACT name -> (subdirectory, filename pattern)
    cam_map = {
        'left': ('left_img_dir', f'frame{frame_idx:06d}_left.jpg'),
        'right_wrist': ('endo_psm1', f'frame{frame_idx:06d}_psm1.jpg'),
        'left_wrist': ('endo_psm2', f'frame{frame_idx:06d}_psm2.jpg'),
    }

    for act_name, (subdir, fname) in cam_map.items():
        path = os.path.join(episode_path, subdir, fname)
        img = cv2.imread(path)
        if img is None:
            return None  # Missing frame
        images[act_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return images


# METRICS

def rot6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 rotation matrix."""
    col1 = rot6d[0:3]
    col2 = rot6d[3:6]
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = col2 - np.dot(col2, col1) * col1
    col2 = col2 / (np.linalg.norm(col2) + 1e-8)
    col3 = np.cross(col1, col2)
    return np.stack([col1, col2, col3], axis=1)


def rotation_error_degrees(rot6d_pred, rot6d_gt):
    """Geodesic rotation error in degrees between two 6D rotations."""
    R_pred = rot6d_to_matrix(rot6d_pred)
    R_gt = rot6d_to_matrix(rot6d_gt)
    R_diff = R_pred.T @ R_gt
    # Clamp trace to valid range
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def compute_metrics(pred_actions, gt_actions, subtask_name):
    """Compute all evaluation metrics.

    Both inputs are arrays of shape (T, 20) in RAW (denormalized) action space.
    """
    T = len(pred_actions)
    assert len(gt_actions) == T

    metrics = {}

    # Position L2 per arm (in mm)
    pos_err_psm1 = np.linalg.norm(pred_actions[:, 0:3] - gt_actions[:, 0:3], axis=1) * 1000
    pos_err_psm2 = np.linalg.norm(pred_actions[:, 10:13] - gt_actions[:, 10:13], axis=1) * 1000
    metrics['pos_l2_psm1_mm'] = pos_err_psm1.mean()
    metrics['pos_l2_psm2_mm'] = pos_err_psm2.mean()
    metrics['pos_l2_mean_mm'] = (pos_err_psm1.mean() + pos_err_psm2.mean()) / 2

    # Position RMSE per arm
    metrics['pos_rmse_psm1_mm'] = np.sqrt((pos_err_psm1**2).mean())
    metrics['pos_rmse_psm2_mm'] = np.sqrt((pos_err_psm2**2).mean())

    # Rotation error per arm (degrees)
    rot_err_psm1 = np.array([rotation_error_degrees(pred_actions[t, 3:9], gt_actions[t, 3:9]) for t in range(T)])
    rot_err_psm2 = np.array([rotation_error_degrees(pred_actions[t, 13:19], gt_actions[t, 13:19]) for t in range(T)])
    metrics['rot_err_psm1_deg'] = rot_err_psm1.mean()
    metrics['rot_err_psm2_deg'] = rot_err_psm2.mean()
    metrics['rot_err_mean_deg'] = (rot_err_psm1.mean() + rot_err_psm2.mean()) / 2

    # Jaw accuracy (sign match)
    # Jaw open > 0, closed <= 0
    jaw_thresh = 0.0
    jaw_match_psm1 = ((pred_actions[:, 9] > jaw_thresh) == (gt_actions[:, 9] > jaw_thresh)).mean() * 100
    jaw_match_psm2 = ((pred_actions[:, 19] > jaw_thresh) == (gt_actions[:, 19] > jaw_thresh)).mean() * 100
    metrics['jaw_acc_psm1_pct'] = jaw_match_psm1
    metrics['jaw_acc_psm2_pct'] = jaw_match_psm2
    metrics['jaw_acc_mean_pct'] = (jaw_match_psm1 + jaw_match_psm2) / 2

    # Trajectory RMSE (all 20D)
    full_err = np.linalg.norm(pred_actions - gt_actions, axis=1)
    metrics['traj_rmse_20d'] = np.sqrt((full_err**2).mean())

    # Drift: compare position error at end vs start (last 10% vs first 10%)
    n10 = max(1, T // 10)
    drift_psm1 = pos_err_psm1[-n10:].mean() - pos_err_psm1[:n10].mean()
    drift_psm2 = pos_err_psm2[-n10:].mean() - pos_err_psm2[:n10].mean()
    metrics['drift_psm1_mm'] = drift_psm1
    metrics['drift_psm2_mm'] = drift_psm2

    # Action magnitude comparison
    metrics['pred_magnitude_mean'] = np.linalg.norm(pred_actions[:, :3], axis=1).mean()
    metrics['gt_magnitude_mean'] = np.linalg.norm(gt_actions[:, :3], axis=1).mean()
    metrics['magnitude_ratio'] = metrics['pred_magnitude_mean'] / (metrics['gt_magnitude_mean'] + 1e-8)

    # Store per-timestep errors for plotting
    metrics['_pos_err_psm1'] = pos_err_psm1
    metrics['_pos_err_psm2'] = pos_err_psm2
    metrics['_rot_err_psm1'] = rot_err_psm1
    metrics['_rot_err_psm2'] = rot_err_psm2
    metrics['_jaw_pred_psm1'] = pred_actions[:, 9]
    metrics['_jaw_gt_psm1'] = gt_actions[:, 9]
    metrics['_jaw_pred_psm2'] = pred_actions[:, 19]
    metrics['_jaw_gt_psm2'] = gt_actions[:, 19]

    return metrics


# GESTURE LABEL LOADING (GC-ACT)

def load_gesture_labels_for_episode(labels_dir, episode_info):
    """Load gesture labels for a specific episode.

    Returns list of (start_frame, end_frame, gesture_str) or None if no labels.
    """
    tissue = episode_info['tissue']       # e.g., 'tissue_7'
    subtask = episode_info['subtask']     # e.g., 'needle_throw'
    episode_id = episode_info['episode_id']  # e.g., '20250117-120051-073008'

    folder_name = SUBTASK_FOLDERS[subtask]  # e.g., '2_needle_throw'
    # Also check recovery variant
    label_path = os.path.join(labels_dir, tissue, folder_name,
                              f"{episode_id}_labels.txt")

    if not os.path.exists(label_path):
        # Try recovery folder variant
        recovery_folder = folder_name + '_recovery'
        label_path = os.path.join(labels_dir, tissue, recovery_folder,
                                  f"{episode_id}_labels.txt")
        if not os.path.exists(label_path):
            return None

    gestures = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = int(parts[0])
                end = int(parts[1])
                gesture = parts[2]
                gestures.append((start, end, gesture))
    return gestures


def get_gesture_at_frame(gestures, frame_idx):
    """Find which gesture is active at the given frame."""
    if gestures is None:
        return None
    for start, end, gesture in gestures:
        if start <= frame_idx <= end:
            return gesture
    return None


def gesture_to_onehot(gesture_str, gesture_dim=10):
    """Convert gesture string to one-hot vector."""
    onehot = np.zeros(gesture_dim, dtype=np.float32)
    if gesture_str is not None and gesture_str in GESTURE_TO_IDX:
        onehot[GESTURE_TO_IDX[gesture_str]] = 1.0
    return onehot


# EVALUATION LOOP

def evaluate_episode(policy, episode_info, step_stride=1, temporal_ensemble_k=None,
                     ensemble_horizon=None, use_gesture=False, labels_dir=None):
    """Evaluate a single episode. Returns metrics dict + per-timestep predictions.

    If temporal_ensemble_k is set, uses exponential temporal ensembling:
    at each timestep, the prediction is a weighted average of overlapping
    chunk predictions, with weight exp(-k * age) where age is how many
    steps ago the chunk was queried. This smooths drift by letting newer
    predictions correct older ones.
    """
    ep_path = episode_info['path']
    subtask = episode_info['subtask']

    df = load_episode_data(ep_path)
    T = len(df)
    if T < 10:
        print(f"    Skipping {ep_path}: only {T} timesteps")
        return None

    # Load gesture labels if GC-ACT mode
    episode_gestures = None
    if use_gesture and labels_dir:
        episode_gestures = load_gesture_labels_for_episode(labels_dir, episode_info)

    pred_actions_raw = []
    gt_actions_raw = []
    timestamps_used = []

    if temporal_ensemble_k is not None:
        # Temporal ensembling: accumulate weighted predictions from overlapping chunks
        chunk_size = policy.chunk_size
        horizon = min(ensemble_horizon or chunk_size, chunk_size)
        action_dim = policy.action_dim
        action_buffer = np.zeros((T + chunk_size, action_dim), dtype=np.float64)
        weight_buffer = np.zeros(T + chunk_size, dtype=np.float64)

        valid_timesteps = []

        for t in range(0, T, step_stride):
            images = load_images(ep_path, t)
            if images is None:
                continue

            row = df.iloc[t]
            qpos = extract_qpos_20d(row)

            # Get gesture embedding for this frame (GC-ACT)
            gesture_emb = None
            if use_gesture:
                gesture_str = get_gesture_at_frame(episode_gestures, t)
                gesture_emb = gesture_to_onehot(gesture_str, policy.gesture_dim)

            # Get full 60-step chunk (normalized)
            chunk_normalized = policy.predict_chunk(images, qpos, gesture_embedding=gesture_emb)

            # Denormalize actions and accumulate with exponential weights
            for j in range(horizon):
                if t + j >= T + chunk_size:
                    break
                action_raw = policy.denormalize_action(chunk_normalized[j])
                w = np.exp(-temporal_ensemble_k * j)
                action_buffer[t + j] += w * action_raw
                weight_buffer[t + j] += w

            valid_timesteps.append(t)

        # Extract ensembled predictions at each valid timestep
        for t in valid_timesteps:
            if weight_buffer[t] > 0:
                ensembled_action = action_buffer[t] / weight_buffer[t]
                gt_20d = extract_gt_action_20d(df.iloc[t], use_setpoint=True)
                pred_actions_raw.append(ensembled_action)
                gt_actions_raw.append(gt_20d)
                timestamps_used.append(t)
    else:
        # Original path: no temporal ensembling, use first action from each chunk
        for t in range(0, T, step_stride):
            images = load_images(ep_path, t)
            if images is None:
                continue

            row = df.iloc[t]
            qpos = extract_qpos_20d(row)
            gt_20d = extract_gt_action_20d(row, use_setpoint=True)

            # Get gesture embedding for this frame (GC-ACT)
            gesture_emb = None
            if use_gesture:
                gesture_str = get_gesture_at_frame(episode_gestures, t)
                gesture_emb = gesture_to_onehot(gesture_str, policy.gesture_dim)

            chunk_normalized = policy.predict_chunk(images, qpos, gesture_embedding=gesture_emb)
            pred_normalized = chunk_normalized[0]
            pred_raw = policy.denormalize_action(pred_normalized)

            pred_actions_raw.append(pred_raw)
            gt_actions_raw.append(gt_20d)
            timestamps_used.append(t)

    if len(pred_actions_raw) < 5:
        print(f"    Skipping {ep_path}: only {len(pred_actions_raw)} valid frames")
        return None

    pred_actions_raw = np.array(pred_actions_raw)
    gt_actions_raw = np.array(gt_actions_raw)

    metrics = compute_metrics(pred_actions_raw, gt_actions_raw, subtask)
    metrics['episode_path'] = ep_path
    metrics['tissue'] = episode_info['tissue']
    metrics['episode_id'] = episode_info['episode_id']
    metrics['num_frames'] = len(timestamps_used)
    metrics['_timestamps'] = np.array(timestamps_used)
    metrics['_pred_raw'] = pred_actions_raw
    metrics['_gt_raw'] = gt_actions_raw

    return metrics


# VISUALIZATION

def plot_episode_trajectory(metrics, output_dir, episode_idx):
    """Plot predicted vs actual trajectory for a single episode."""
    pred = metrics['_pred_raw']
    gt = metrics['_gt_raw']
    ts = metrics['_timestamps']
    subtask = metrics.get('subtask', 'unknown')
    tissue = metrics.get('tissue', 'unknown')

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"{subtask} | {tissue} | {metrics['episode_id']}\n"
                 f"Pos L2: {metrics['pos_l2_mean_mm']:.2f}mm | "
                 f"Rot: {metrics['rot_err_mean_deg']:.1f}deg | "
                 f"Jaw: {metrics['jaw_acc_mean_pct']:.0f}%",
                 fontsize=14, fontweight='bold')

    # Row 0: PSM1 position XYZ
    labels = ['X', 'Y', 'Z']
    for i, label in enumerate(labels):
        axes[0, 0].plot(ts, pred[:, i] * 1000, '--', label=f'pred {label}', alpha=0.8)
        axes[0, 0].plot(ts, gt[:, i] * 1000, '-', label=f'gt {label}', alpha=0.8)
    axes[0, 0].set_title('PSM1 Position (mm)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel('mm')

    # Row 0: PSM2 position XYZ
    for i, label in enumerate(labels):
        axes[0, 1].plot(ts, pred[:, 10+i] * 1000, '--', label=f'pred {label}', alpha=0.8)
        axes[0, 1].plot(ts, gt[:, 10+i] * 1000, '-', label=f'gt {label}', alpha=0.8)
    axes[0, 1].set_title('PSM2 Position (mm)')
    axes[0, 1].legend(fontsize=8)

    # Row 1: Position error over time
    axes[1, 0].plot(ts, metrics['_pos_err_psm1'], 'r-', label='PSM1')
    axes[1, 0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1mm threshold')
    axes[1, 0].set_title('PSM1 Position Error (mm)')
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('mm')

    axes[1, 1].plot(ts, metrics['_pos_err_psm2'], 'b-', label='PSM2')
    axes[1, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1mm threshold')
    axes[1, 1].set_title('PSM2 Position Error (mm)')
    axes[1, 1].legend()

    # Row 2: Rotation error over time
    axes[2, 0].plot(ts, metrics['_rot_err_psm1'], 'r-', label='PSM1')
    axes[2, 0].axhline(y=5.0, color='g', linestyle='--', alpha=0.5, label='5deg threshold')
    axes[2, 0].set_title('PSM1 Rotation Error (deg)')
    axes[2, 0].legend()
    axes[2, 0].set_ylabel('degrees')

    axes[2, 1].plot(ts, metrics['_rot_err_psm2'], 'b-', label='PSM2')
    axes[2, 1].axhline(y=5.0, color='g', linestyle='--', alpha=0.5, label='5deg threshold')
    axes[2, 1].set_title('PSM2 Rotation Error (deg)')
    axes[2, 1].legend()

    # Row 3: Jaw timeline
    axes[3, 0].plot(ts, metrics['_jaw_pred_psm1'], 'r--', label='pred', alpha=0.8)
    axes[3, 0].plot(ts, metrics['_jaw_gt_psm1'], 'r-', label='gt', alpha=0.8)
    axes[3, 0].axhline(y=0, color='k', linestyle=':', alpha=0.3)
    axes[3, 0].set_title('PSM1 Jaw')
    axes[3, 0].legend()
    axes[3, 0].set_xlabel('Timestep')
    axes[3, 0].set_ylabel('rad')

    axes[3, 1].plot(ts, metrics['_jaw_pred_psm2'], 'b--', label='pred', alpha=0.8)
    axes[3, 1].plot(ts, metrics['_jaw_gt_psm2'], 'b-', label='gt', alpha=0.8)
    axes[3, 1].axhline(y=0, color='k', linestyle=':', alpha=0.3)
    axes[3, 1].set_title('PSM2 Jaw')
    axes[3, 1].legend()
    axes[3, 1].set_xlabel('Timestep')

    plt.tight_layout()
    fname = f"episode_{episode_idx:03d}_{tissue}_{subtask}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches='tight')
    plt.close()


def plot_3d_trajectory(metrics, output_dir, episode_idx):
    """3D scatter of predicted vs GT positions for both arms."""
    pred = metrics['_pred_raw']
    gt = metrics['_gt_raw']
    tissue = metrics.get('tissue', 'unknown')
    subtask = metrics.get('subtask', 'unknown')

    fig = plt.figure(figsize=(14, 6))

    for arm_idx, (arm_name, offset) in enumerate([('PSM1', 0), ('PSM2', 10)]):
        ax = fig.add_subplot(1, 2, arm_idx + 1, projection='3d')
        ax.plot(gt[:, offset] * 1000, gt[:, offset+1] * 1000, gt[:, offset+2] * 1000,
                'g-', alpha=0.6, linewidth=2, label='GT')
        ax.plot(pred[:, offset] * 1000, pred[:, offset+1] * 1000, pred[:, offset+2] * 1000,
                'r--', alpha=0.6, linewidth=2, label='Pred')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{arm_name}')
        ax.legend()

    fig.suptitle(f"{subtask} | {tissue} | 3D Trajectory", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"traj3d_{episode_idx:03d}_{tissue}_{subtask}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches='tight')
    plt.close()


def plot_aggregate_summary(all_metrics, subtask_name, output_dir):
    """Bar chart summary across all episodes for a subtask."""
    if not all_metrics:
        return

    scalar_keys = ['pos_l2_mean_mm', 'rot_err_mean_deg', 'jaw_acc_mean_pct',
                   'pos_rmse_psm1_mm', 'pos_rmse_psm2_mm',
                   'drift_psm1_mm', 'drift_psm2_mm']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{subtask_name.upper()}  -- Aggregate ({len(all_metrics)} episodes)",
                 fontsize=14, fontweight='bold')

    # Position L2 distribution
    pos_l2 = [m['pos_l2_mean_mm'] for m in all_metrics]
    axes[0, 0].hist(pos_l2, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=1.0, color='g', linestyle='--', label='1mm threshold')
    axes[0, 0].set_title(f'Position L2 (mm)  -- mean={np.mean(pos_l2):.2f}')
    axes[0, 0].legend()

    # Rotation error distribution
    rot_err = [m['rot_err_mean_deg'] for m in all_metrics]
    axes[0, 1].hist(rot_err, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=5.0, color='g', linestyle='--', label='5deg threshold')
    axes[0, 1].set_title(f'Rotation Error (deg)  -- mean={np.mean(rot_err):.1f}')
    axes[0, 1].legend()

    # Jaw accuracy distribution
    jaw_acc = [m['jaw_acc_mean_pct'] for m in all_metrics]
    axes[1, 0].hist(jaw_acc, bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=90.0, color='g', linestyle='--', label='90% threshold')
    axes[1, 0].set_title(f'Jaw Accuracy (%)  -- mean={np.mean(jaw_acc):.0f}')
    axes[1, 0].legend()

    # Per-tissue breakdown
    tissue_metrics = defaultdict(list)
    for m in all_metrics:
        tissue_metrics[m['tissue']].append(m['pos_l2_mean_mm'])
    tissues = sorted(tissue_metrics.keys())
    tissue_means = [np.mean(tissue_metrics[t]) for t in tissues]
    tissue_stds = [np.std(tissue_metrics[t]) for t in tissues]
    axes[1, 1].bar(range(len(tissues)), tissue_means, yerr=tissue_stds,
                   color='teal', edgecolor='black', alpha=0.7, capsize=3)
    axes[1, 1].set_xticks(range(len(tissues)))
    axes[1, 1].set_xticklabels([t.replace('tissue_', 'T') for t in tissues], rotation=45)
    axes[1, 1].axhline(y=2.0, color='g', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Position L2 by Tissue')
    axes[1, 1].set_ylabel('mm')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{subtask_name}.png'), dpi=120, bbox_inches='tight')
    plt.close()


# GO/NO-GO DECISION

THRESHOLDS = {
    'pos_l2_mean_mm': 1.0,      # <1mm mean position error (matches SutureBot paper SOTA)
    'rot_err_mean_deg': 5.0,     # <5 degree mean rotation error
    'jaw_acc_mean_pct': 90.0,    # >90% jaw accuracy
    'drift_mm': 1.0,             # <1mm drift (end vs start error growth)
}


def go_nogo_decision(all_metrics_by_subtask):
    """Print GO/NO-GO summary."""
    print("\n" + "="*70)
    print("  GO / NO-GO DECISION")
    print("="*70)

    overall_go = True
    results = {}

    for subtask_name, metrics_list in all_metrics_by_subtask.items():
        if not metrics_list:
            print(f"\n  {subtask_name.upper()}: NO DATA")
            overall_go = False
            continue

        pos_l2 = np.mean([m['pos_l2_mean_mm'] for m in metrics_list])
        rot_err = np.mean([m['rot_err_mean_deg'] for m in metrics_list])
        jaw_acc = np.mean([m['jaw_acc_mean_pct'] for m in metrics_list])
        # Only penalize positive drift (error growing). Negative drift (error
        # shrinking / model self-correcting) is good behavior, not a failure.
        drift = np.mean([max(max(m['drift_psm1_mm'], 0), max(m['drift_psm2_mm'], 0)) for m in metrics_list])

        pos_ok = pos_l2 < THRESHOLDS['pos_l2_mean_mm']
        rot_ok = rot_err < THRESHOLDS['rot_err_mean_deg']
        jaw_ok = jaw_acc > THRESHOLDS['jaw_acc_mean_pct']
        drift_ok = drift < THRESHOLDS['drift_mm']

        subtask_go = pos_ok and rot_ok and jaw_ok and drift_ok

        status = "GO" if subtask_go else "NO-GO"
        print(f"\n  {subtask_name.upper()}: [{status}]")
        print(f"    Position L2:  {pos_l2:.2f} mm  {'[OK]' if pos_ok else '[FAIL]'} (threshold: <{THRESHOLDS['pos_l2_mean_mm']}mm)")
        print(f"    Rotation:     {rot_err:.1f} deg  {'[OK]' if rot_ok else '[FAIL]'} (threshold: <{THRESHOLDS['rot_err_mean_deg']}deg)")
        print(f"    Jaw accuracy: {jaw_acc:.0f}%     {'[OK]' if jaw_ok else '[FAIL]'} (threshold: >{THRESHOLDS['jaw_acc_mean_pct']}%)")
        print(f"    Drift:        {drift:.2f} mm  {'[OK]' if drift_ok else '[FAIL]'} (threshold: <{THRESHOLDS['drift_mm']}mm)")

        if not subtask_go:
            overall_go = False

        results[subtask_name] = {
            'status': status,
            'pos_l2_mm': pos_l2,
            'rot_err_deg': rot_err,
            'jaw_acc_pct': jaw_acc,
            'drift_mm': drift,
            'n_episodes': len(metrics_list),
        }

    print(f"\n  {'='*50}")
    if overall_go:
        print(f"  OVERALL: GO  -- Models pass all thresholds. Safe to deploy on real dVRK.")
    else:
        print(f"  OVERALL: NO-GO  -- Fix failing subtasks before real dVRK deployment.")
    print(f"  {'='*50}\n")

    return results


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Offline evaluation of ACT models on SutureBot data")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory of SutureBot data (contains tissue_N/ folders)')
    parser.add_argument('--ckpt_np', type=str, default=None,
                        help='Needle pickup checkpoint')
    parser.add_argument('--ckpt_nt', type=str, default=None,
                        help='Needle throw checkpoint')
    parser.add_argument('--ckpt_kt', type=str, default=None,
                        help='Knot tying checkpoint')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/offline_eval_results'),
                        help='Output directory for results')
    parser.add_argument('--subtasks', nargs='+', default=None,
                        choices=['needle_pickup', 'needle_throw', 'knot_tying'],
                        help='Which subtasks to evaluate (default: all with checkpoints)')
    parser.add_argument('--max_episodes', type=int, default=10,
                        help='Max episodes per subtask (default: 10)')
    parser.add_argument('--step_stride', type=int, default=1,
                        help='Evaluate every Nth timestep (default: 1 = every frame)')
    parser.add_argument('--tissue_ids', nargs='+', type=int, default=None,
                        help='Specific tissue IDs to evaluate (default: all available)')
    parser.add_argument('--temporal_ensemble_k', type=float, default=None,
                        help='Temporal ensembling decay factor (e.g. 0.01). '
                             'Averages overlapping chunk predictions with exponential '
                             'weights exp(-k*age) to reduce drift. None=disabled.')
    parser.add_argument('--ensemble_horizon', type=int, default=None,
                        help='Max chunk steps to use for ensembling (default: full chunk_size=60). '
                             'Lower values (e.g. 20) discard later, less-accurate predictions.')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--image_encoder', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b3', 'dinov2_vits14'],
                        help='Backbone (resnet18=v1, efficientnet_b3=v2/GC-ACT, dinov2_vits14=DINOv2)')
    parser.add_argument('--kl_weight', type=int, default=1,
                        help='KL weight (1=v1, 10=v2/GC-ACT)')
    parser.add_argument('--use_gesture', action='store_true',
                        help='Enable gesture conditioning (GC-ACT mode)')
    parser.add_argument('--gesture_dim', type=int, default=10,
                        help='Gesture embedding dimension (default: 10)')
    parser.add_argument('--labels_dir', type=str,
                        default=os.path.expanduser('~/data/labels'),
                        help='Directory containing gesture label files')
    parser.add_argument('--norm_stats_key', type=str, default=None,
                        help='Override norm stats key (e.g. knot_tying_ood). '
                             'If set, uses this key for all subtasks instead of the subtask name.')

    args = parser.parse_args()

    # Determine which subtasks to evaluate
    ckpt_map = {
        'needle_pickup': args.ckpt_np,
        'needle_throw': args.ckpt_nt,
        'knot_tying': args.ckpt_kt,
    }

    if args.subtasks:
        subtasks_to_eval = args.subtasks
    else:
        subtasks_to_eval = [s for s, c in ckpt_map.items() if c is not None]

    if not subtasks_to_eval:
        print("ERROR: No checkpoints provided. Specify at least one of --ckpt_np, --ckpt_nt, --ckpt_kt")
        sys.exit(1)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    mode = "GC-ACT" if args.use_gesture else "ACT"
    print("="*70)
    print(f"  OFFLINE EVALUATION  -- {mode} on SutureBot")
    print("="*70)
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Subtasks:     {subtasks_to_eval}")
    print(f"  Max episodes: {args.max_episodes}")
    print(f"  Step stride:  {args.step_stride}")
    print(f"  Tissue IDs:   {args.tissue_ids or 'all'}")
    print(f"  Backbone:     {args.image_encoder}")
    print(f"  KL weight:    {args.kl_weight}")
    if args.use_gesture:
        print(f"  Gesture:      ENABLED (dim={args.gesture_dim})")
        print(f"  Labels dir:   {args.labels_dir}")
    if args.temporal_ensemble_k is not None:
        print(f"  Temporal ensembling: k={args.temporal_ensemble_k}")

    # Verify data directory
    if not os.path.isdir(args.data_dir):
        print(f"\nERROR: Data directory not found: {args.data_dir}")
        print("Download SutureBot data first. See download_suturebot.py")
        sys.exit(1)

    all_metrics_by_subtask = {}

    for subtask_name in subtasks_to_eval:
        ckpt_path = ckpt_map[subtask_name]
        if ckpt_path is None:
            print(f"\n  Skipping {subtask_name}: no checkpoint provided")
            continue

        print(f"\n{'='*70}")
        print(f"  EVALUATING: {subtask_name.upper()}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*70}")

        # Load model  -- use OOD norm stats key if provided
        norm_key = args.norm_stats_key if args.norm_stats_key else subtask_name
        policy = ACTPolicyOffline(ckpt_path, norm_key, device=args.device,
                                  image_encoder=args.image_encoder,
                                  use_gesture=args.use_gesture,
                                  gesture_dim=args.gesture_dim,
                                  kl_weight=args.kl_weight)

        # Find episodes
        episodes = find_episodes(args.data_dir, subtask_name,
                                 max_episodes=args.max_episodes,
                                 tissue_ids=args.tissue_ids)
        if not episodes:
            print(f"  No episodes found for {subtask_name}!")
            all_metrics_by_subtask[subtask_name] = []
            continue

        print(f"  Found {len(episodes)} episodes")
        tissues_found = set(ep['tissue'] for ep in episodes)
        print(f"  Tissues: {sorted(tissues_found)}")

        # Evaluate each episode
        subtask_metrics = []
        for ep_idx, ep_info in enumerate(episodes):
            print(f"\n  [{ep_idx+1}/{len(episodes)}] {ep_info['tissue']}/{ep_info['episode_id']}", end=" ")
            t0 = time.time()
            ep_info['subtask'] = subtask_name

            metrics = evaluate_episode(policy, ep_info, step_stride=args.step_stride,
                                       temporal_ensemble_k=args.temporal_ensemble_k,
                                       ensemble_horizon=args.ensemble_horizon,
                                       use_gesture=args.use_gesture,
                                       labels_dir=args.labels_dir)

            if metrics is None:
                print("-- SKIPPED")
                continue

            dt = time.time() - t0
            print(f"-- {metrics['num_frames']} frames, "
                  f"pos={metrics['pos_l2_mean_mm']:.2f}mm, "
                  f"rot={metrics['rot_err_mean_deg']:.1f}deg, "
                  f"jaw={metrics['jaw_acc_mean_pct']:.0f}%  "
                  f"({dt:.1f}s)")

            metrics['subtask'] = subtask_name
            subtask_metrics.append(metrics)

            # Plot individual episode
            if not args.no_plots and ep_idx < 5:  # Plot first 5 episodes
                plot_episode_trajectory(metrics, plots_dir, ep_idx)
                plot_3d_trajectory(metrics, plots_dir, ep_idx)

        all_metrics_by_subtask[subtask_name] = subtask_metrics

        # Print subtask summary
        if subtask_metrics:
            print(f"\n  --- {subtask_name.upper()} SUMMARY ({len(subtask_metrics)} episodes) ---")
            for key in ['pos_l2_mean_mm', 'rot_err_mean_deg', 'jaw_acc_mean_pct',
                        'pos_rmse_psm1_mm', 'pos_rmse_psm2_mm',
                        'drift_psm1_mm', 'drift_psm2_mm', 'magnitude_ratio']:
                vals = [m[key] for m in subtask_metrics]
                print(f"    {key:25s}: {np.mean(vals):8.3f} +/- {np.std(vals):.3f}  "
                      f"[min={np.min(vals):.3f}, max={np.max(vals):.3f}]")

            # Plot aggregate
            if not args.no_plots:
                plot_aggregate_summary(subtask_metrics, subtask_name, plots_dir)

        # Cleanup GPU memory
        del policy
        torch.cuda.empty_cache()

    # Save all results as CSV
    csv_rows = []
    for subtask_name, metrics_list in all_metrics_by_subtask.items():
        for m in metrics_list:
            row = {k: v for k, v in m.items() if not k.startswith('_')}
            row['subtask'] = subtask_name
            csv_rows.append(row)

    if csv_rows:
        results_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(args.output_dir, 'results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results saved to {csv_path}")

    # Save full metrics (including arrays) as npz
    npz_path = os.path.join(args.output_dir, 'results_full.npz')
    npz_data = {}
    for subtask_name, metrics_list in all_metrics_by_subtask.items():
        for i, m in enumerate(metrics_list):
            prefix = f"{subtask_name}_{i}"
            npz_data[f"{prefix}_pred_raw"] = m.get('_pred_raw', np.array([]))
            npz_data[f"{prefix}_gt_raw"] = m.get('_gt_raw', np.array([]))
            npz_data[f"{prefix}_timestamps"] = m.get('_timestamps', np.array([]))
    np.savez_compressed(npz_path, **npz_data)
    print(f"  Full results saved to {npz_path}")

    # GO/NO-GO
    decision = go_nogo_decision(all_metrics_by_subtask)

    # Save decision as JSON
    decision_path = os.path.join(args.output_dir, 'decision.json')
    with open(decision_path, 'w') as f:
        json.dump(decision, f, indent=2)
    print(f"  Decision saved to {decision_path}")


if __name__ == '__main__':
    main()
