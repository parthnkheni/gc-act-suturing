#!/usr/bin/env python3
"""
Single-Subtask Test Mode  -- Run ONE subtask at a time on the real dVRK.

Designed for JHU one-day deployment: verify each subtask (NP, NT, KT)
individually before chaining them together.

Safety features:
  - Pre-move confirmation prompt (press Enter to start, 'q' to abort)
  - --dry_run mode: load model, print predictions, no robot commands
  - Real-time console feedback (positions, gestures, confidence)
  - All predictions logged to CSV for post-analysis
  - Ctrl+C gracefully stops at any time

Usage:
    # Dry run needle pickup with v2 model:
    python test_single_subtask.py --subtask np --model_version v2 --dry_run

    # Dry run needle throw with GC-ACT:
    python test_single_subtask.py --subtask nt --model_version gcact --dry_run

    # Real robot needle pickup (v2):
    python test_single_subtask.py --subtask np --model_version v2

    # Real robot knot tying (GC-ACT):
    python test_single_subtask.py --subtask kt --model_version gcact

    # Custom checkpoint paths:
    python test_single_subtask.py --subtask nt --model_version gcact \
        --ckpt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
        --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt

    # Adjust steps and action horizon:
    python test_single_subtask.py --subtask kt --model_version v2 \
        --max_steps 500 --action_horizon 10 --sleep_rate 0.05
"""

import os
import sys
import argparse
import time
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from torchvision import models, transforms

# Path setup  -- find SutureBot source regardless of where this script lives
path_to_suturebot = os.getenv(
    "PATH_TO_SUTUREBOT",
    os.path.join(os.path.expanduser("~"), "SutureBot"),
)
sys.path.insert(0, os.path.join(path_to_suturebot, "src", "act"))
sys.path.insert(0, os.path.join(path_to_suturebot, "src"))

from policy import ACTPolicy
from dvrk_scripts.constants_dvrk import TASK_CONFIGS


# CONSTANTS

GESTURE_LABELS = ["G2", "G3", "G6", "G7", "G10", "G11", "G13", "G14", "G15", "G16"]
IDX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_LABELS)}
NUM_GESTURE_CLASSES = len(GESTURE_LABELS)

# Maps the short CLI name to the internal subtask key
SUBTASK_ALIAS = {
    "np": "needle_pickup",
    "nt": "needle_throw",
    "kt": "knot_tying",
}

SUBTASK_CONFIGS = {
    "needle_pickup": {
        "task_name": "needle_pickup_all",
        "default_steps": 300,
        "use_gesture": False,  # NP has no gesture labels
    },
    "needle_throw": {
        "task_name": "needle_throw_all",
        "default_steps": 600,
        "use_gesture": True,
    },
    "knot_tying": {
        "task_name": "knot_tying_all",
        "default_steps": 320,
        "use_gesture": True,
    },
}

# Default checkpoint paths (match project conventions)
DEFAULT_CKPTS = {
    "v2": {
        "needle_pickup": "~/checkpoints/act_np_v2/policy_best.ckpt",
        "needle_throw": "~/checkpoints/act_nt_v2/policy_best.ckpt",
        "knot_tying":   "~/checkpoints/act_kt_v2/policy_best.ckpt",
    },
    "gcact": {
        "needle_pickup": "~/checkpoints/act_np_v2/policy_best.ckpt",  # NP always uses v2
        "needle_throw": "~/checkpoints/act_nt_gcact/policy_best.ckpt",
        "knot_tying":   "~/checkpoints/act_kt_gcact/policy_best.ckpt",
    },
}

DEFAULT_GESTURE_CKPT = "~/checkpoints/gesture_classifier/gesture_best.ckpt"


# GESTURE CLASSIFIER

class GestureClassifier(nn.Module):
    """ResNet18 + FC head for gesture classification."""

    def __init__(self, num_classes=NUM_GESTURE_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_gesture_classifier(ckpt_path, device="cuda"):
    """Load the trained gesture classifier."""
    model = GestureClassifier(num_classes=NUM_GESTURE_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"  [GestureClassifier] Loaded from {ckpt_path}")
    return model


GESTURE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_gesture(classifier, image_rgb, device="cuda"):
    """Predict gesture from a single RGB image (H, W, 3) uint8.

    Returns:
        gesture_str: e.g. 'G3'
        gesture_onehot: (10,) numpy float32
        confidence: float in [0, 1]
    """
    img_tensor = GESTURE_TRANSFORM(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    gesture_str = IDX_TO_GESTURE[pred_idx.item()]
    gesture_onehot = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
    gesture_onehot[pred_idx.item()] = 1.0
    return gesture_str, gesture_onehot, conf.item()


# MODEL LOADING

def load_act_model(ckpt_path, use_gesture=False, image_encoder="efficientnet_b3",
                   kl_weight=10, gesture_dim=10, device="cuda"):
    """Load a single ACT or GC-ACT model from checkpoint."""
    saved_argv = sys.argv
    argv = [
        "act", "--task_name", "needle_pickup_all",
        "--ckpt_dir", "/tmp", "--policy_class", "ACT",
        "--seed", "0", "--num_epochs", "1",
        "--kl_weight", str(kl_weight), "--chunk_size", "60",
        "--hidden_dim", "512", "--dim_feedforward", "3200",
        "--lr", "1e-5", "--batch_size", "8",
        "--image_encoder", image_encoder,
    ]
    if use_gesture:
        argv.extend(["--use_gesture", "--gesture_dim", str(gesture_dim)])
    sys.argv = argv

    policy_config = {
        "lr": 1e-5,
        "num_queries": 60,
        "action_dim": 20,
        "kl_weight": kl_weight,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": image_encoder,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": ["left", "left_wrist", "right_wrist"],
        "multi_gpu": False,
    }
    if use_gesture:
        policy_config["use_gesture"] = True
        policy_config["gesture_dim"] = gesture_dim

    policy = ACTPolicy(policy_config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    sys.argv = saved_argv

    mode = "GC-ACT" if use_gesture else "ACT v2"
    print(f"  [{mode}] {image_encoder} loaded from {ckpt_path}")
    return policy


# ACTION CONVERSION

def unnormalize_action(action_chunk, mean, std):
    """Unnormalize positions and jaw; rotations stay raw."""
    unnormalized = action_chunk * std + mean
    unnormalized[:, 3:9] = action_chunk[:, 3:9]    # PSM1 rotation raw
    unnormalized[:, 13:19] = action_chunk[:, 13:19]  # PSM2 rotation raw
    return unnormalized


def convert_6d_to_quat(rot6d_chunk, current_quat_xyzw):
    """Convert 6D rotation predictions to quaternion targets (xyzw)."""
    c1 = rot6d_chunk[:, 0:3]
    c2 = rot6d_chunk[:, 3:6]
    c1 = normalize(c1, axis=1)
    dot_product = np.sum(c1 * c2, axis=1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product * c1, axis=1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3))
    rots = R.from_matrix(r_mat)
    rot_init = R.from_quat(current_quat_xyzw)
    composed = (rot_init * rots).as_quat()
    return composed


def actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2):
    """Convert unnormalized 20D action chunk to dVRK waypoints.

    Returns:
        wp_psm1: (chunk_size, 8)  -- [x,y,z, qx,qy,qz,qw, jaw]
        wp_psm2: (chunk_size, 8)  -- [x,y,z, qx,qy,qz,qw, jaw]
    """
    chunk_size = action_chunk.shape[0]
    action = action_chunk.copy()

    # Absolute -> delta (subtract first timestep)
    action[:, 0:3] = action[:, 0:3] - action[0, 0:3]
    action[:, 10:13] = action[:, 10:13] - action[0, 10:13]

    wp_psm1 = np.zeros((chunk_size, 8))
    wp_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3]
    wp_psm1[:, 3:7] = convert_6d_to_quat(action[:, 3:9], qpos_psm1[3:7])
    wp_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)

    wp_psm2 = np.zeros((chunk_size, 8))
    wp_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13]
    wp_psm2[:, 3:7] = convert_6d_to_quat(action[:, 13:19], qpos_psm2[3:7])
    wp_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)

    return wp_psm1, wp_psm2


# IMAGE PROCESSING

def process_compressed_image(compressed_data):
    """Decode compressed ROS image to RGB numpy array (360, 480, 3)."""
    img = np.frombuffer(compressed_data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (480, 360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_camera_images(rt):
    """Get 3 camera images from ROS topics.

    Returns:
        image_tensor: (1, 3, 3, 360, 480) float32 CUDA tensor
        left_rgb: (360, 480, 3) uint8 numpy (for gesture classifier)
    """
    if rt.usb_image_left is None or rt.endo_cam_psm1 is None or rt.endo_cam_psm2 is None:
        return None, None

    left = process_compressed_image(rt.usb_image_left.data)
    left_rgb = left.copy()

    left_chw = np.transpose(left, (2, 0, 1))
    lw = np.transpose(process_compressed_image(rt.endo_cam_psm2.data), (2, 0, 1))
    rw = np.transpose(process_compressed_image(rt.endo_cam_psm1.data), (2, 0, 1))

    image_data = np.stack([left_chw, lw, rw], axis=0)
    image_tensor = torch.from_numpy(image_data / 255.0).float().cuda().unsqueeze(0)
    return image_tensor, left_rgb


def get_robot_state(rt):
    """Get current PSM1 and PSM2 state from ROS topics.

    Returns:
        qpos_psm1: (8,)  -- [x,y,z, qx,qy,qz,qw, jaw]
        qpos_psm2: (8,)  -- [x,y,z, qx,qy,qz,qw, jaw]
    """
    qpos_psm1 = np.array([
        rt.psm1_pose.position.x, rt.psm1_pose.position.y, rt.psm1_pose.position.z,
        rt.psm1_pose.orientation.x, rt.psm1_pose.orientation.y,
        rt.psm1_pose.orientation.z, rt.psm1_pose.orientation.w,
        rt.psm1_jaw,
    ])
    qpos_psm2 = np.array([
        rt.psm2_pose.position.x, rt.psm2_pose.position.y, rt.psm2_pose.position.z,
        rt.psm2_pose.orientation.x, rt.psm2_pose.orientation.y,
        rt.psm2_pose.orientation.z, rt.psm2_pose.orientation.w,
        rt.psm2_jaw,
    ])
    return qpos_psm1, qpos_psm2


# CSV LOGGER

CSV_COLUMNS = [
    "timestamp",
    "step",
    "chunk_id",
    "chunk_step",
    "subtask",
    "model_version",
    "gesture_pred",
    "gesture_conf",
    "gesture_used",
    "psm1_x", "psm1_y", "psm1_z",
    "psm1_qx", "psm1_qy", "psm1_qz", "psm1_qw",
    "psm1_jaw",
    "psm2_x", "psm2_y", "psm2_z",
    "psm2_qx", "psm2_qy", "psm2_qz", "psm2_qw",
    "psm2_jaw",
]


def init_csv_logger(log_path):
    """Create CSV file with header. Returns (file_handle, csv_writer)."""
    fh = open(log_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    return fh, writer


def log_step(writer, step, chunk_id, chunk_step, subtask, model_version,
             gesture_pred, gesture_conf, gesture_used, wp_psm1, wp_psm2):
    """Write one row to the CSV log."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "step": step,
        "chunk_id": chunk_id,
        "chunk_step": chunk_step,
        "subtask": subtask,
        "model_version": model_version,
        "gesture_pred": gesture_pred,
        "gesture_conf": f"{gesture_conf:.4f}" if gesture_conf is not None else "",
        "gesture_used": gesture_used,
        "psm1_x": f"{wp_psm1[0]:.6f}",
        "psm1_y": f"{wp_psm1[1]:.6f}",
        "psm1_z": f"{wp_psm1[2]:.6f}",
        "psm1_qx": f"{wp_psm1[3]:.6f}",
        "psm1_qy": f"{wp_psm1[4]:.6f}",
        "psm1_qz": f"{wp_psm1[5]:.6f}",
        "psm1_qw": f"{wp_psm1[6]:.6f}",
        "psm1_jaw": f"{wp_psm1[7]:.6f}",
        "psm2_x": f"{wp_psm2[0]:.6f}",
        "psm2_y": f"{wp_psm2[1]:.6f}",
        "psm2_z": f"{wp_psm2[2]:.6f}",
        "psm2_qx": f"{wp_psm2[3]:.6f}",
        "psm2_qy": f"{wp_psm2[4]:.6f}",
        "psm2_qz": f"{wp_psm2[5]:.6f}",
        "psm2_qw": f"{wp_psm2[6]:.6f}",
        "psm2_jaw": f"{wp_psm2[7]:.6f}",
    }
    writer.writerow(row)


# DRY RUN

def run_dry_run(policy, subtask_name, model_version, norm_mean, norm_std,
                use_gesture, gesture_classifier, max_steps, action_horizon,
                csv_writer):
    """Run inference with dummy data. No ROS/robot needed."""

    print("\n  [DRY RUN] Generating dummy camera images and robot state...")
    dummy_images = torch.randn(1, 3, 3, 360, 480).float().cuda()
    dummy_left_rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
    qpos_zero_tensor = torch.zeros(1, 20).float().cuda()

    # Dummy robot state for waypoint conversion
    dummy_qpos = np.array([0.02, -0.01, 0.04, 0.0, 0.0, 0.0, 1.0, 0.0])

    # Test gesture classifier if applicable
    gesture_str, gesture_onehot, gesture_conf = "", None, None
    if use_gesture and gesture_classifier is not None:
        gesture_str, gesture_onehot, gesture_conf = predict_gesture(
            gesture_classifier, dummy_left_rgb)
        print(f"  [DRY RUN] Gesture classifier test: {gesture_str} "
              f"(conf={gesture_conf:.3f})")

    print(f"\n  [DRY RUN] Running {max_steps} steps "
          f"(action_horizon={action_horizon}, chunk_size=60)...\n")

    step = 0
    chunk_id = 0
    while step < max_steps:
        # Gesture inference
        g_str, g_conf, g_used = "", None, ""
        kwargs = {}
        if use_gesture and gesture_classifier is not None:
            g_str, g_onehot, g_conf = predict_gesture(
                gesture_classifier, dummy_left_rgb)
            g_used = g_conf >= 0.5
            if g_conf < 0.5:
                g_onehot = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
            gesture_tensor = torch.from_numpy(g_onehot).float().cuda().unsqueeze(0)
            kwargs["gesture_embedding"] = gesture_tensor

        # Model inference
        with torch.inference_mode():
            action_chunk = policy(
                qpos_zero_tensor, dummy_images, **kwargs
            ).cpu().numpy().squeeze()

        action_chunk = unnormalize_action(action_chunk, norm_mean, norm_std)
        wp_psm1, wp_psm2 = actions_to_waypoints(
            action_chunk, dummy_qpos, dummy_qpos)

        steps_to_execute = min(action_horizon, max_steps - step,
                               action_chunk.shape[0])

        # Print chunk summary
        print(f"  Chunk {chunk_id:3d} | steps {step:4d}-{step+steps_to_execute-1:4d} | "
              f"PSM1 pos=({wp_psm1[0,0]:+.4f}, {wp_psm1[0,1]:+.4f}, {wp_psm1[0,2]:+.4f}) | "
              f"PSM2 pos=({wp_psm2[0,0]:+.4f}, {wp_psm2[0,1]:+.4f}, {wp_psm2[0,2]:+.4f})",
              end="")
        if use_gesture and g_str:
            print(f" | gesture={g_str}({g_conf:.2f})", end="")
        print()

        # Log every step in the executed portion of the chunk
        for j in range(steps_to_execute):
            log_step(
                csv_writer,
                step=step,
                chunk_id=chunk_id,
                chunk_step=j,
                subtask=subtask_name,
                model_version=model_version,
                gesture_pred=g_str,
                gesture_conf=g_conf,
                gesture_used=g_used,
                wp_psm1=wp_psm1[j],
                wp_psm2=wp_psm2[j],
            )
            step += 1

        chunk_id += 1

    # Print final summary statistics from first and last chunks
    print(f"\n  [DRY RUN] Completed {step} steps across {chunk_id} chunks.")
    print(f"  Action chunk shape: {action_chunk.shape}")
    print(f"  PSM1 pos range: x=[{wp_psm1[:,0].min():.4f}, {wp_psm1[:,0].max():.4f}], "
          f"y=[{wp_psm1[:,1].min():.4f}, {wp_psm1[:,1].max():.4f}], "
          f"z=[{wp_psm1[:,2].min():.4f}, {wp_psm1[:,2].max():.4f}]")
    print(f"  PSM2 pos range: x=[{wp_psm2[:,0].min():.4f}, {wp_psm2[:,0].max():.4f}], "
          f"y=[{wp_psm2[:,1].min():.4f}, {wp_psm2[:,1].max():.4f}], "
          f"z=[{wp_psm2[:,2].min():.4f}, {wp_psm2[:,2].max():.4f}]")
    print(f"  PSM1 jaw range: [{wp_psm1[:,7].min():.3f}, {wp_psm1[:,7].max():.3f}]")
    print(f"  PSM2 jaw range: [{wp_psm2[:,7].min():.3f}, {wp_psm2[:,7].max():.3f}]")


# REAL ROBOT RUN

def run_real_robot(policy, subtask_name, model_version, norm_mean, norm_std,
                   use_gesture, gesture_classifier, max_steps, action_horizon,
                   sleep_rate, gesture_conf_threshold, csv_writer):
    """Run inference on the real dVRK robot with live cameras."""

    import rospy
    import crtk
    from rostopics import ros_topics
    from dvrk_scripts.dvrk_control import example_application

    rospy.init_node("test_single_subtask", anonymous=True)
    rt = ros_topics()
    ral = crtk.ral("test_single_subtask")
    psm1_app = example_application(ral, "PSM1", 1)
    psm2_app = example_application(ral, "PSM2", 1)

    print("\n  Waiting for camera and robot data...")
    time.sleep(2.0)

    # Verify data availability
    if rt.usb_image_left is None:
        print("  ERROR: No left camera image. "
              "Check /jhu_daVinci/left/image_raw/compressed")
        sys.exit(1)
    if rt.endo_cam_psm1 is None:
        print("  ERROR: No PSM1 wrist camera. "
              "Check endo camera topics.")
        sys.exit(1)
    if rt.endo_cam_psm2 is None:
        print("  ERROR: No PSM2 wrist camera. "
              "Check endo camera topics.")
        sys.exit(1)
    if rt.psm1_pose is None:
        print("  ERROR: No PSM1 pose. Check /PSM1/setpoint_cp")
        sys.exit(1)
    if rt.psm2_pose is None:
        print("  ERROR: No PSM2 pose. Check /PSM2/setpoint_cp")
        sys.exit(1)
    print("  Camera and robot data OK.")

    # Read initial state for sanity display
    qpos_psm1, qpos_psm2 = get_robot_state(rt)
    print(f"\n  Current PSM1 pos: ({qpos_psm1[0]:+.4f}, {qpos_psm1[1]:+.4f}, "
          f"{qpos_psm1[2]:+.4f})  jaw: {qpos_psm1[7]:.3f}")
    print(f"  Current PSM2 pos: ({qpos_psm2[0]:+.4f}, {qpos_psm2[1]:+.4f}, "
          f"{qpos_psm2[2]:+.4f})  jaw: {qpos_psm2[7]:.3f}")

    # Run one inference to show the first predicted positions
    image_tensor, left_rgb = get_camera_images(rt)
    if image_tensor is None:
        print("  ERROR: Camera images returned None on first read.")
        sys.exit(1)

    qpos_zero_tensor = torch.zeros(1, 20).float().cuda()

    # Preview first chunk
    print("\n  -- Preview: first chunk predictions --")
    g_str_preview, g_conf_preview = "", None
    kwargs = {}
    if use_gesture and gesture_classifier is not None:
        g_str_preview, g_onehot_preview, g_conf_preview = predict_gesture(
            gesture_classifier, left_rgb)
        used = g_conf_preview >= gesture_conf_threshold
        if not used:
            g_onehot_preview = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
        gesture_tensor = torch.from_numpy(g_onehot_preview).float().cuda().unsqueeze(0)
        kwargs["gesture_embedding"] = gesture_tensor
        print(f"  Gesture prediction: {g_str_preview} "
              f"(conf={g_conf_preview:.3f}, used={used})")

    with torch.inference_mode():
        preview_chunk = policy(
            qpos_zero_tensor, image_tensor, **kwargs
        ).cpu().numpy().squeeze()

    preview_chunk = unnormalize_action(preview_chunk, norm_mean, norm_std)
    wp1_preview, wp2_preview = actions_to_waypoints(
        preview_chunk, qpos_psm1, qpos_psm2)

    print(f"  First waypoint PSM1: ({wp1_preview[0,0]:+.4f}, "
          f"{wp1_preview[0,1]:+.4f}, {wp1_preview[0,2]:+.4f})")
    print(f"  First waypoint PSM2: ({wp2_preview[0,0]:+.4f}, "
          f"{wp2_preview[0,1]:+.4f}, {wp2_preview[0,2]:+.4f})")
    delta1 = np.linalg.norm(wp1_preview[0, 0:3] - qpos_psm1[0:3])
    delta2 = np.linalg.norm(wp2_preview[0, 0:3] - qpos_psm2[0:3])
    print(f"  Delta from current: PSM1={delta1:.4f}m, PSM2={delta2:.4f}m")

    # Safety confirmation
    print("\n" + "=" * 60)
    print("  SAFETY CHECK: The robot is about to move.")
    print(f"  Subtask: {subtask_name.upper()}")
    print(f"  Max steps: {max_steps}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  Sleep rate: {sleep_rate}s ({1.0/sleep_rate:.0f} Hz)")
    print("=" * 60)
    user_input = input("  Press ENTER to start, or 'q' to abort: ").strip().lower()
    if user_input == "q":
        print("  Aborted by user.")
        return

    # Main control loop
    step = 0
    chunk_id = 0
    print(f"\n  Running {subtask_name.upper()}... (Ctrl+C to stop)\n")

    try:
        while step < max_steps:
            image_tensor, left_rgb = get_camera_images(rt)
            if image_tensor is None:
                print("  Warning: missing camera frame, retrying...")
                time.sleep(0.1)
                continue

            qpos_psm1, qpos_psm2 = get_robot_state(rt)

            # Gesture classification
            g_str, g_conf, g_used = "", None, ""
            kwargs = {}
            if use_gesture and gesture_classifier is not None:
                g_str, g_onehot, g_conf = predict_gesture(
                    gesture_classifier, left_rgb)
                g_used = g_conf >= gesture_conf_threshold
                if not g_used:
                    g_onehot = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
                gesture_tensor = torch.from_numpy(g_onehot).float().cuda().unsqueeze(0)
                kwargs["gesture_embedding"] = gesture_tensor

            # Model inference
            with torch.inference_mode():
                action_chunk = policy(
                    qpos_zero_tensor, image_tensor, **kwargs
                ).cpu().numpy().squeeze()

            action_chunk = unnormalize_action(action_chunk, norm_mean, norm_std)
            wp_psm1, wp_psm2 = actions_to_waypoints(
                action_chunk, qpos_psm1, qpos_psm2)

            steps_to_execute = min(action_horizon, max_steps - step,
                                   action_chunk.shape[0])

            # Print chunk info
            gesture_info = ""
            if use_gesture and g_str:
                gesture_info = f" | gesture={g_str}({g_conf:.2f}, {'ON' if g_used else 'OFF'})"
            print(f"  Chunk {chunk_id:3d} | steps {step:4d}-{step+steps_to_execute-1:4d}"
                  f"/{max_steps}{gesture_info}")

            # Execute waypoints
            for j in range(steps_to_execute):
                ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
                ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])
                time.sleep(sleep_rate)

                log_step(
                    csv_writer,
                    step=step,
                    chunk_id=chunk_id,
                    chunk_step=j,
                    subtask=subtask_name,
                    model_version=model_version,
                    gesture_pred=g_str,
                    gesture_conf=g_conf,
                    gesture_used=g_used,
                    wp_psm1=wp_psm1[j],
                    wp_psm2=wp_psm2[j],
                )
                step += 1

                # Periodic detailed print (every 50 steps)
                if step % 50 == 0:
                    print(f"    step {step:4d}/{max_steps} | "
                          f"PSM1=({wp_psm1[j,0]:+.4f},{wp_psm1[j,1]:+.4f},"
                          f"{wp_psm1[j,2]:+.4f}) jaw={wp_psm1[j,7]:.3f} | "
                          f"PSM2=({wp_psm2[j,0]:+.4f},{wp_psm2[j,1]:+.4f},"
                          f"{wp_psm2[j,2]:+.4f}) jaw={wp_psm2[j,7]:.3f}")

            chunk_id += 1

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted at step {step}/{max_steps} (chunk {chunk_id})")

    print(f"\n  {subtask_name.upper()} finished: {step} steps, {chunk_id} chunks.")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Single-subtask test mode for GC-ACT dVRK deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run needle pickup (v2):
  python test_single_subtask.py --subtask np --model_version v2 --dry_run

  # Dry run needle throw (GC-ACT):
  python test_single_subtask.py --subtask nt --model_version gcact --dry_run

  # Real robot knot tying (GC-ACT):
  python test_single_subtask.py --subtask kt --model_version gcact
        """,
    )

    parser.add_argument(
        "--subtask", type=str, required=True,
        choices=["np", "nt", "kt"],
        help="Which subtask to run: np=needle_pickup, nt=needle_throw, kt=knot_tying",
    )
    parser.add_argument(
        "--model_version", type=str, required=True,
        choices=["v2", "gcact"],
        help="Model version: v2 (ACT EfficientNet-B3) or gcact (gesture-conditioned)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load model and print predictions without sending commands to robot",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Override checkpoint path (default: auto-selected based on subtask+version)",
    )
    parser.add_argument(
        "--gesture_ckpt", type=str, default=None,
        help="Override gesture classifier checkpoint path",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max inference steps (default: NP=300, NT=600, KT=320)",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=20,
        help="Steps of the 60-step chunk to execute before re-querying (default: 20)",
    )
    parser.add_argument(
        "--sleep_rate", type=float, default=0.1,
        help="Seconds between action steps (default: 0.1 = 10Hz)",
    )
    parser.add_argument(
        "--gesture_conf_threshold", type=float, default=0.5,
        help="Gesture classifier confidence threshold; below this, zero vector is used (default: 0.5)",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Directory for CSV log file (default: ~/test_subtask_logs/)",
    )

    args = parser.parse_args()

    # Resolve subtask
    subtask_name = SUBTASK_ALIAS[args.subtask]
    subtask_cfg = SUBTASK_CONFIGS[subtask_name]
    model_version = args.model_version

    # Determine if gesture classifier is needed
    # GC-ACT uses gesture for NT and KT; NP always uses plain v2
    use_gesture = (model_version == "gcact" and subtask_cfg["use_gesture"])

    # Resolve checkpoint path
    if args.ckpt:
        ckpt_path = os.path.expanduser(args.ckpt)
    else:
        ckpt_path = os.path.expanduser(DEFAULT_CKPTS[model_version][subtask_name])

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Resolve gesture checkpoint
    gesture_ckpt_path = None
    if use_gesture:
        gesture_ckpt_path = os.path.expanduser(
            args.gesture_ckpt if args.gesture_ckpt else DEFAULT_GESTURE_CKPT
        )
        if not os.path.isfile(gesture_ckpt_path):
            print(f"ERROR: Gesture checkpoint not found: {gesture_ckpt_path}")
            sys.exit(1)

    # Resolve max_steps
    max_steps = args.max_steps if args.max_steps else subtask_cfg["default_steps"]

    # Resolve log directory and file
    log_dir = os.path.expanduser(args.log_dir if args.log_dir else "~/test_subtask_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "dry" if args.dry_run else "live"
    log_filename = f"{args.subtask}_{model_version}_{mode_tag}_{timestamp}.csv"
    log_path = os.path.join(log_dir, log_filename)

    # Banner
    print("=" * 60)
    print("  SINGLE-SUBTASK TEST MODE")
    print("=" * 60)
    print(f"  Subtask:       {subtask_name.upper()} ({args.subtask})")
    print(f"  Model version: {model_version}")
    print(f"  Use gesture:   {use_gesture}")
    print(f"  Checkpoint:    {ckpt_path}")
    if gesture_ckpt_path:
        print(f"  Gesture ckpt:  {gesture_ckpt_path}")
    print(f"  Max steps:     {max_steps}")
    print(f"  Action horizon:{args.action_horizon}")
    print(f"  Sleep rate:    {args.sleep_rate}s ({1.0/args.sleep_rate:.0f} Hz)")
    print(f"  Dry run:       {args.dry_run}")
    print(f"  Log file:      {log_path}")
    if use_gesture:
        print(f"  Gesture conf:  {args.gesture_conf_threshold}")
    print("=" * 60)

    # Load models
    print("\n  Loading models...")

    # For NP with gcact flag, we still use plain v2 (no gesture labels for NP)
    if model_version == "gcact" and subtask_name == "needle_pickup":
        print("  NOTE: NP has no gesture labels; loading as plain ACT v2.")

    # Determine image encoder: v2 and gcact both use efficientnet_b3
    image_encoder = "efficientnet_b3"

    policy = load_act_model(
        ckpt_path,
        use_gesture=use_gesture,
        image_encoder=image_encoder,
        kl_weight=10,
        gesture_dim=NUM_GESTURE_CLASSES,
    )

    gesture_classifier = None
    if use_gesture and gesture_ckpt_path:
        gesture_classifier = load_gesture_classifier(gesture_ckpt_path)

    # Load normalization stats
    task_config = TASK_CONFIGS[subtask_cfg["task_name"]]
    norm_mean = task_config["action_mode"][1]["mean"]
    norm_std = task_config["action_mode"][1]["std"]

    print(f"\n  Norm stats from task config: {subtask_cfg['task_name']}")
    print(f"  Action mean (PSM1 xyz): ({norm_mean[0]:.4f}, {norm_mean[1]:.4f}, {norm_mean[2]:.4f})")
    print(f"  Action mean (PSM2 xyz): ({norm_mean[10]:.4f}, {norm_mean[11]:.4f}, {norm_mean[12]:.4f})")

    # Initialize CSV logger
    csv_fh, csv_writer = init_csv_logger(log_path)

    # Run
    try:
        if args.dry_run:
            run_dry_run(
                policy=policy,
                subtask_name=subtask_name,
                model_version=model_version,
                norm_mean=norm_mean,
                norm_std=norm_std,
                use_gesture=use_gesture,
                gesture_classifier=gesture_classifier,
                max_steps=max_steps,
                action_horizon=args.action_horizon,
                csv_writer=csv_writer,
            )
        else:
            run_real_robot(
                policy=policy,
                subtask_name=subtask_name,
                model_version=model_version,
                norm_mean=norm_mean,
                norm_std=norm_std,
                use_gesture=use_gesture,
                gesture_classifier=gesture_classifier,
                max_steps=max_steps,
                action_horizon=args.action_horizon,
                sleep_rate=args.sleep_rate,
                gesture_conf_threshold=args.gesture_conf_threshold,
                csv_writer=csv_writer,
            )
    finally:
        csv_fh.close()
        print(f"\n  CSV log saved to: {log_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
