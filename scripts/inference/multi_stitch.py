#!/usr/bin/env python3
"""
Multi-Stitch Autonomous Suturing  -- GC-ACT Pipeline on the real dVRK.

Executes 2-3 continuous sutures without human intervention by looping the
learned subtask chain (NP -> NT -> KT) and inserting a scripted needle
repositioning phase between stitches.

Pipeline per stitch:
    Stitch N: Needle Pickup -> Needle Throw -> Knot Tying -> [Needle Reposition]

The Needle Reposition phase is NOT learned. It uses scripted servo_cp
waypoints to:
    1. LIFT    -- move PSM1 straight up ~2cm to clear tissue
    2. TRANSLATE  -- move laterally to the next insertion point
    3. LOWER   -- descend to tissue surface height
    4. ORIENT  -- rotate needle to correct insertion angle
    5. RELEASE  -- open gripper to place needle
    6. RETRACT  -- pull PSM1 up slightly, ready for next NP

Insertion points can be specified manually (--insertion_points) or
auto-generated as evenly spaced points along a line (default).

Usage:
    # Dry run  -- verify models load, print full plan, no robot commands:
    python multi_stitch.py --dry_run

    # Dry run with 2 stitches and 4mm spacing:
    python multi_stitch.py --num_stitches 2 --stitch_spacing_mm 4 --dry_run

    # Real robot  -- 3 stitches with safety pauses:
    python multi_stitch.py --num_stitches 3

    # Real robot  -- skip safety pauses between stitches:
    python multi_stitch.py --num_stitches 3 --no_pause

    # Custom insertion points (x,y in meters, comma-separated pairs):
    python multi_stitch.py --insertion_points "0.03,-0.01;0.035,-0.01;0.04,-0.01"

    # Custom checkpoint paths:
    python multi_stitch.py \
        --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_gcact/policy_best.ckpt \
        --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt
"""

import os
import sys
import argparse
import time
import json
import datetime
import csv
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from torchvision import models, transforms

# Path setup
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

SUBTASK_CONFIGS = {
    "needle_pickup": {
        "task_name": "needle_pickup_all",
        "default_steps": 300,
        "use_gesture": False,
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

DEFAULT_CKPTS = {
    "needle_pickup": "~/checkpoints/act_np_v2/policy_best.ckpt",
    "needle_throw":  "~/checkpoints/act_nt_gcact/policy_best.ckpt",
    "knot_tying":    "~/checkpoints/act_kt_gcact/policy_best.ckpt",
}
DEFAULT_GESTURE_CKPT = "~/checkpoints/gesture_classifier/gesture_best.ckpt"

# Repositioning parameters (meters)
REPOSITION_LIFT_HEIGHT = 0.02       # 2cm lift above current z
REPOSITION_SURFACE_Z = 0.04        # approximate tissue surface z in task frame
REPOSITION_CLEARANCE_Z = 0.06      # safe clearance z for lateral moves
REPOSITION_RELEASE_JAW = 0.5       # jaw angle to open gripper (radians)
REPOSITION_CLOSE_JAW = -0.2        # jaw angle for closed gripper
REPOSITION_RETRACT_HEIGHT = 0.015  # pull up after releasing needle

# Default needle insertion orientation (xyzw quaternion)  -- roughly downward
# This is an approximate needle-insertion orientation for the dVRK PSM1.
# Adjust based on your specific tissue/pad setup.
DEFAULT_NEEDLE_ORIENT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


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
    """Predict gesture from a single RGB image (H, W, 3) uint8."""
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
    unnormalized[:, 3:9] = action_chunk[:, 3:9]
    unnormalized[:, 13:19] = action_chunk[:, 13:19]
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
        wp_psm1: (chunk_size, 8) -- [x,y,z, qx,qy,qz,qw, jaw]
        wp_psm2: (chunk_size, 8) -- [x,y,z, qx,qy,qz,qw, jaw]
    """
    chunk_size = action_chunk.shape[0]
    action = action_chunk.copy()

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
        qpos_psm1: (8,) -- [x,y,z, qx,qy,qz,qw, jaw]
        qpos_psm2: (8,) -- [x,y,z, qx,qy,qz,qw, jaw]
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
    "timestamp", "stitch_idx", "phase", "step", "chunk_id", "chunk_step",
    "subtask", "model_version", "gesture_pred", "gesture_conf", "gesture_used",
    "psm1_x", "psm1_y", "psm1_z",
    "psm1_qx", "psm1_qy", "psm1_qz", "psm1_qw", "psm1_jaw",
    "psm2_x", "psm2_y", "psm2_z",
    "psm2_qx", "psm2_qy", "psm2_qz", "psm2_qw", "psm2_jaw",
]


def init_csv_logger(log_path):
    """Create CSV file with header. Returns (file_handle, csv_writer)."""
    fh = open(log_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    return fh, writer


def log_step(writer, stitch_idx, phase, step, chunk_id, chunk_step,
             subtask, model_version, gesture_pred, gesture_conf, gesture_used,
             wp_psm1, wp_psm2):
    """Write one row to the CSV log."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "stitch_idx": stitch_idx,
        "phase": phase,
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


# INSERTION POINT GENERATION

def generate_insertion_points(num_stitches, spacing_m, start_xy=None, direction=None):
    """Generate evenly spaced insertion points along a line.

    Args:
        num_stitches: number of stitches
        spacing_m: distance between stitches in meters
        start_xy: (x, y) starting position, default based on dVRK mean workspace
        direction: (dx, dy) unit direction for stitch line, default along +x

    Returns:
        list of (x, y) tuples in meters (task frame)
    """
    if start_xy is None:
        # Default: near the center of the dVRK PSM1 workspace
        # Based on the normalization mean for PSM1 xyz: (0.026, -0.011, 0.036)
        start_xy = (0.026, -0.011)

    if direction is None:
        direction = (1.0, 0.0)  # along +x axis

    dx, dy = direction
    mag = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / mag, dy / mag

    points = []
    for i in range(num_stitches):
        x = start_xy[0] + i * spacing_m * dx
        y = start_xy[1] + i * spacing_m * dy
        points.append((x, y))

    return points


def parse_insertion_points(points_str):
    """Parse insertion points from string format 'x1,y1;x2,y2;x3,y3'.

    Returns:
        list of (x, y) tuples in meters
    """
    points = []
    for pair in points_str.strip().split(";"):
        parts = pair.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point format: '{pair}'. Expected 'x,y'.")
        x, y = float(parts[0]), float(parts[1])
        points.append((x, y))
    return points


# NEEDLE REPOSITIONING (SCRIPTED)

def interpolate_waypoints(start_xyz, end_xyz, num_steps):
    """Generate linearly interpolated xyz waypoints between two positions.

    Args:
        start_xyz: (3,) start position
        end_xyz: (3,) end position
        num_steps: number of intermediate waypoints (including endpoints)

    Returns:
        (num_steps, 3) array of xyz positions
    """
    if num_steps < 2:
        return np.array([end_xyz])
    t = np.linspace(0.0, 1.0, num_steps).reshape(-1, 1)
    waypoints = start_xyz + t * (end_xyz - start_xyz)
    return waypoints


def execute_repositioning(psm1_app, ral, current_psm1_state, next_insertion_xy,
                          reposition_speed, logger_info=None, csv_writer=None):
    """Execute the scripted needle repositioning trajectory on PSM1.

    Moves the needle from its current position (after KT) to the next
    insertion point so the next stitch's NP can begin.

    Steps:
        1. LIFT   -- move up to clearance height
        2. TRANSLATE -- move laterally to next insertion point
        3. LOWER  -- descend to tissue surface
        4. ORIENT -- rotate to needle insertion orientation
        5. RELEASE -- open gripper
        6. RETRACT -- pull up slightly

    Args:
        psm1_app: example_application for PSM1
        ral: crtk.ral instance
        current_psm1_state: (8,) [x,y,z, qx,qy,qz,qw, jaw]
        next_insertion_xy: (x, y) target insertion point in meters
        reposition_speed: seconds between waypoints
        logger_info: dict with stitch_idx etc. for logging
        csv_writer: CSV writer for logging
    """
    import PyKDL

    cur_pos = current_psm1_state[0:3].copy()
    cur_quat = current_psm1_state[3:7].copy()  # xyzw
    cur_jaw = current_psm1_state[7]

    next_x, next_y = next_insertion_xy

    # Phase positions
    lift_pos = np.array([cur_pos[0], cur_pos[1], REPOSITION_CLEARANCE_Z])
    translate_pos = np.array([next_x, next_y, REPOSITION_CLEARANCE_Z])
    lower_pos = np.array([next_x, next_y, REPOSITION_SURFACE_Z])
    retract_pos = np.array([next_x, next_y, REPOSITION_SURFACE_Z + REPOSITION_RETRACT_HEIGHT])

    # Target orientation for needle placement
    target_quat = DEFAULT_NEEDLE_ORIENT_XYZW.copy()

    def send_psm1_waypoint(xyz, quat_xyzw, jaw_angle):
        """Send a single Cartesian + jaw command to PSM1."""
        goal = PyKDL.Frame()
        goal.p = PyKDL.Vector(xyz[0], xyz[1], xyz[2])
        goal.M = PyKDL.Rotation.Quaternion(quat_xyzw[0], quat_xyzw[1],
                                            quat_xyzw[2], quat_xyzw[3])
        psm1_app.arm.move_cp(goal).wait()
        jaw_pos = psm1_app.arm.jaw.setpoint_jp()
        jaw_pos[0] = jaw_angle
        psm1_app.arm.jaw.servo_jp(jaw_pos)

    def execute_phase(phase_name, start_xyz, end_xyz, quat_xyzw, jaw_angle,
                      num_steps=10):
        """Execute one phase of the repositioning with interpolated moves."""
        waypoints = interpolate_waypoints(start_xyz, end_xyz, num_steps)
        print(f"      {phase_name}: "
              f"({start_xyz[0]:.4f},{start_xyz[1]:.4f},{start_xyz[2]:.4f}) -> "
              f"({end_xyz[0]:.4f},{end_xyz[1]:.4f},{end_xyz[2]:.4f}) "
              f"[{num_steps} steps]")
        for i, wp_xyz in enumerate(waypoints):
            send_psm1_waypoint(wp_xyz, quat_xyzw, jaw_angle)
            time.sleep(reposition_speed)

            # Log if writer provided
            if csv_writer is not None and logger_info is not None:
                wp_full = np.concatenate([wp_xyz, quat_xyzw, [jaw_angle]])
                dummy_psm2 = np.zeros(8)
                log_step(
                    csv_writer,
                    stitch_idx=logger_info.get("stitch_idx", -1),
                    phase=f"reposition_{phase_name}",
                    step=logger_info.get("global_step", 0),
                    chunk_id=-1,
                    chunk_step=i,
                    subtask="reposition",
                    model_version="scripted",
                    gesture_pred="",
                    gesture_conf=None,
                    gesture_used="",
                    wp_psm1=wp_full,
                    wp_psm2=dummy_psm2,
                )

    # Execute the 6-phase repositioning
    # 1. LIFT: current position -> clearance height (keep current orientation, keep jaw closed)
    execute_phase("LIFT", cur_pos, lift_pos, cur_quat, REPOSITION_CLOSE_JAW, num_steps=10)

    # 2. TRANSLATE: lateral move at clearance height
    dist = np.linalg.norm(translate_pos - lift_pos)
    translate_steps = max(10, int(dist / 0.001))  # ~1mm per step
    execute_phase("TRANSLATE", lift_pos, translate_pos, cur_quat,
                  REPOSITION_CLOSE_JAW, num_steps=translate_steps)

    # 3. LOWER: descend to tissue surface
    execute_phase("LOWER", translate_pos, lower_pos, cur_quat,
                  REPOSITION_CLOSE_JAW, num_steps=10)

    # 4. ORIENT: rotate to needle insertion orientation (in place)
    # We do a single move_cp with target orientation
    print(f"      ORIENT: rotating to insertion orientation")
    send_psm1_waypoint(lower_pos, target_quat, REPOSITION_CLOSE_JAW)
    time.sleep(reposition_speed * 3)  # extra settling time for rotation

    # 5. RELEASE: open gripper to place needle
    print(f"      RELEASE: opening gripper (jaw={REPOSITION_RELEASE_JAW:.2f})")
    send_psm1_waypoint(lower_pos, target_quat, REPOSITION_RELEASE_JAW)
    time.sleep(reposition_speed * 5)  # hold open for needle to settle

    # 6. RETRACT: pull up slightly
    execute_phase("RETRACT", lower_pos, retract_pos, target_quat,
                  REPOSITION_RELEASE_JAW, num_steps=5)

    print(f"      Repositioning complete.")


def print_repositioning_plan(current_xy, next_xy, stitch_idx):
    """Print the repositioning plan without executing (for dry run)."""
    dist = np.sqrt((next_xy[0] - current_xy[0])**2 + (next_xy[1] - current_xy[1])**2)
    print(f"      [DRY RUN] Repositioning plan (stitch {stitch_idx} -> {stitch_idx+1}):")
    print(f"        Current (x,y):  ({current_xy[0]:.4f}, {current_xy[1]:.4f})")
    print(f"        Next (x,y):     ({next_xy[0]:.4f}, {next_xy[1]:.4f})")
    print(f"        Lateral dist:   {dist*1000:.2f} mm")
    print(f"        1. LIFT:      z -> {REPOSITION_CLEARANCE_Z:.4f} m")
    print(f"        2. TRANSLATE: ({current_xy[0]:.4f},{current_xy[1]:.4f}) -> "
          f"({next_xy[0]:.4f},{next_xy[1]:.4f})")
    print(f"        3. LOWER:     z -> {REPOSITION_SURFACE_Z:.4f} m")
    print(f"        4. ORIENT:    rotate to insertion angle")
    print(f"        5. RELEASE:   jaw -> {REPOSITION_RELEASE_JAW:.2f} rad")
    print(f"        6. RETRACT:   z -> {REPOSITION_SURFACE_Z + REPOSITION_RETRACT_HEIGHT:.4f} m")


# LEARNED SUBTASK EXECUTION

def run_subtask(policy, subtask_name, norm_mean, norm_std, use_gesture,
                gesture_classifier, max_steps, action_horizon, sleep_rate,
                gesture_conf_threshold, stitch_idx, csv_writer,
                # ROS objects (None for dry run):
                rt=None, ral=None, psm1_app=None, psm2_app=None,
                dry_run=False):
    """Run a single learned subtask (NP, NT, or KT).

    Returns:
        (success, steps_executed, final_psm1_state)
        success: True if subtask completed normally, False if error/interrupted
        steps_executed: number of steps actually run
        final_psm1_state: (8,) last PSM1 state or dummy for dry run
    """
    mode = "GC-ACT" if use_gesture else "ACT v2"
    print(f"    Running {subtask_name.upper()} ({mode}, {max_steps} steps)...")

    qpos_zero_tensor = torch.zeros(1, 20).float().cuda()
    step = 0
    chunk_id = 0
    final_psm1 = np.zeros(8)
    final_psm2 = np.zeros(8)

    if dry_run:
        dummy_images = torch.randn(1, 3, 3, 360, 480).float().cuda()
        dummy_left_rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        dummy_qpos = np.array([0.02, -0.01, 0.04, 0.0, 0.0, 0.0, 1.0, 0.0])

    try:
        while step < max_steps:
            # Get images
            if dry_run:
                image_tensor = torch.randn(1, 3, 3, 360, 480).float().cuda()
                left_rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
                qpos_psm1, qpos_psm2 = dummy_qpos.copy(), dummy_qpos.copy()
            else:
                image_tensor, left_rgb = get_camera_images(rt)
                if image_tensor is None:
                    time.sleep(0.1)
                    continue
                qpos_psm1, qpos_psm2 = get_robot_state(rt)

            # Gesture classification
            g_str, g_conf, g_used = "", None, ""
            kwargs = {}
            if use_gesture and gesture_classifier is not None:
                g_str, g_onehot, g_conf = predict_gesture(gesture_classifier, left_rgb)
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
            wp_psm1, wp_psm2 = actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2)

            steps_to_execute = min(action_horizon, max_steps - step,
                                   action_chunk.shape[0])

            # Execute waypoints
            for j in range(steps_to_execute):
                if not dry_run:
                    ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
                    ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])
                    time.sleep(sleep_rate)

                if csv_writer is not None:
                    log_step(
                        csv_writer,
                        stitch_idx=stitch_idx,
                        phase="learned",
                        step=step,
                        chunk_id=chunk_id,
                        chunk_step=j,
                        subtask=subtask_name,
                        model_version="gcact" if use_gesture else "v2",
                        gesture_pred=g_str,
                        gesture_conf=g_conf,
                        gesture_used=g_used,
                        wp_psm1=wp_psm1[j],
                        wp_psm2=wp_psm2[j],
                    )

                final_psm1 = wp_psm1[j]
                final_psm2 = wp_psm2[j]
                step += 1

            # Periodic progress
            if step % 100 == 0 or step == max_steps:
                gesture_info = ""
                if use_gesture and g_str:
                    gesture_info = f" gesture={g_str}({g_conf:.2f})"
                print(f"      step {step}/{max_steps}{gesture_info}")

            chunk_id += 1

    except KeyboardInterrupt:
        print(f"\n      Interrupted at step {step}/{max_steps}")
        return False, step, final_psm1

    except Exception as e:
        print(f"      ERROR during {subtask_name}: {e}")
        return False, step, final_psm1

    print(f"    {subtask_name.upper()} complete ({step} steps, {chunk_id} chunks)")
    return True, step, final_psm1


# MAIN MULTI-STITCH LOOP

def main():
    parser = argparse.ArgumentParser(
        description="Multi-stitch autonomous suturing with GC-ACT on real dVRK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no robot):
  python multi_stitch.py --dry_run

  # 3 stitches, 5mm spacing:
  python multi_stitch.py --num_stitches 3 --stitch_spacing_mm 5

  # Custom insertion points:
  python multi_stitch.py --insertion_points "0.03,-0.01;0.035,-0.01;0.04,-0.01"

  # Skip safety pauses:
  python multi_stitch.py --num_stitches 3 --no_pause
        """,
    )

    # Stitch parameters
    parser.add_argument("--num_stitches", type=int, default=3,
                        help="Number of stitches to perform (default: 3)")
    parser.add_argument("--stitch_spacing_mm", type=float, default=5.0,
                        help="Distance between insertion points in mm (default: 5)")
    parser.add_argument("--insertion_points", type=str, default=None,
                        help="Manual insertion points as 'x1,y1;x2,y2;...' in meters. "
                             "Overrides --num_stitches and --stitch_spacing_mm.")
    parser.add_argument("--stitch_direction", type=str, default="x",
                        choices=["x", "y", "-x", "-y"],
                        help="Direction for auto-generated insertion points (default: x)")

    # Checkpoint paths
    parser.add_argument("--ckpt_np", type=str, default=None,
                        help="Needle pickup checkpoint (ACT v2)")
    parser.add_argument("--ckpt_nt", type=str, default=None,
                        help="Needle throw checkpoint (GC-ACT)")
    parser.add_argument("--ckpt_kt", type=str, default=None,
                        help="Knot tying checkpoint (GC-ACT)")
    parser.add_argument("--gesture_ckpt", type=str, default=None,
                        help="Gesture classifier checkpoint")

    # Subtask step limits
    parser.add_argument("--np_steps", type=int, default=300,
                        help="Max steps for needle pickup (default: 300)")
    parser.add_argument("--nt_steps", type=int, default=600,
                        help="Max steps for needle throw (default: 600)")
    parser.add_argument("--kt_steps", type=int, default=320,
                        help="Max steps for knot tying (default: 320)")

    # Execution parameters
    parser.add_argument("--action_horizon", type=int, default=20,
                        help="Steps of 60-step chunk to execute before re-querying (default: 20)")
    parser.add_argument("--sleep_rate", type=float, default=0.1,
                        help="Seconds between learned action steps (default: 0.1 = 10Hz)")
    parser.add_argument("--reposition_speed", type=float, default=0.05,
                        help="Seconds between repositioning waypoints (default: 0.05 = 20Hz)")
    parser.add_argument("--gesture_conf_threshold", type=float, default=0.5,
                        help="Gesture confidence threshold (default: 0.5)")

    # Modes
    parser.add_argument("--dry_run", action="store_true",
                        help="Load models, print plan, no robot commands")
    parser.add_argument("--no_pause", action="store_true",
                        help="Skip safety pauses between stitches")
    parser.add_argument("--retry_on_fail", action="store_true",
                        help="Retry a failed subtask once before skipping (default: enabled)")
    parser.add_argument("--no_retry", action="store_true",
                        help="Skip to next stitch on any subtask failure")

    # Logging
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for log files (default: ~/multi_stitch_logs/)")

    args = parser.parse_args()

    # Resolve insertion points
    direction_map = {
        "x": (1.0, 0.0), "-x": (-1.0, 0.0),
        "y": (0.0, 1.0), "-y": (0.0, -1.0),
    }

    if args.insertion_points:
        insertion_points = parse_insertion_points(args.insertion_points)
        num_stitches = len(insertion_points)
    else:
        num_stitches = args.num_stitches
        spacing_m = args.stitch_spacing_mm / 1000.0
        direction = direction_map[args.stitch_direction]
        insertion_points = generate_insertion_points(num_stitches, spacing_m,
                                                     direction=direction)

    # Resolve checkpoint paths
    ckpt_np = os.path.expanduser(args.ckpt_np or DEFAULT_CKPTS["needle_pickup"])
    ckpt_nt = os.path.expanduser(args.ckpt_nt or DEFAULT_CKPTS["needle_throw"])
    ckpt_kt = os.path.expanduser(args.ckpt_kt or DEFAULT_CKPTS["knot_tying"])
    gesture_ckpt = os.path.expanduser(args.gesture_ckpt or DEFAULT_GESTURE_CKPT)

    # Verify checkpoint files exist
    for name, path in [("NP", ckpt_np), ("NT", ckpt_nt), ("KT", ckpt_kt),
                       ("Gesture", gesture_ckpt)]:
        if not os.path.isfile(path):
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    steps_map = {
        "needle_pickup": args.np_steps,
        "needle_throw": args.nt_steps,
        "knot_tying": args.kt_steps,
    }

    retry_enabled = not args.no_retry

    # Setup logging
    log_dir = os.path.expanduser(args.log_dir or "~/multi_stitch_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "dry" if args.dry_run else "live"
    log_filename = f"multi_stitch_{num_stitches}x_{mode_tag}_{timestamp}.csv"
    log_path = os.path.join(log_dir, log_filename)

    # Also save the full execution plan as JSON
    plan_path = os.path.join(log_dir, f"plan_{num_stitches}x_{mode_tag}_{timestamp}.json")

    # Banner
    print("=" * 70)
    print("  MULTI-STITCH AUTONOMOUS SUTURING")
    print("=" * 70)
    print(f"  Stitches:          {num_stitches}")
    print(f"  Spacing:           {args.stitch_spacing_mm:.1f} mm")
    print(f"  Direction:         {args.stitch_direction}")
    print(f"  Dry run:           {args.dry_run}")
    print(f"  Safety pauses:     {not args.no_pause}")
    print(f"  Retry on failure:  {retry_enabled}")
    print(f"  Action horizon:    {args.action_horizon}")
    print(f"  Sleep rate:        {args.sleep_rate}s ({1.0/args.sleep_rate:.0f} Hz)")
    print(f"  Reposition speed:  {args.reposition_speed}s ({1.0/args.reposition_speed:.0f} Hz)")
    print(f"  Gesture threshold: {args.gesture_conf_threshold}")
    print()
    print(f"  Checkpoints:")
    print(f"    NP:      {ckpt_np}")
    print(f"    NT:      {ckpt_nt}")
    print(f"    KT:      {ckpt_kt}")
    print(f"    Gesture: {gesture_ckpt}")
    print()
    print(f"  Insertion points:")
    for i, (x, y) in enumerate(insertion_points):
        print(f"    Stitch {i+1}: ({x:.4f}, {y:.4f}) m  "
              f"= ({x*1000:.2f}, {y*1000:.2f}) mm")
    print()
    print(f"  Pipeline per stitch:")
    for i in range(num_stitches):
        subtask_str = f"NP({args.np_steps}) -> NT({args.nt_steps}) -> KT({args.kt_steps})"
        if i < num_stitches - 1:
            subtask_str += " -> [Reposition]"
        print(f"    Stitch {i+1}: {subtask_str}")

    total_learned_steps = num_stitches * (args.np_steps + args.nt_steps + args.kt_steps)
    est_time_min = total_learned_steps * args.sleep_rate / 60.0
    print(f"\n  Total learned steps: {total_learned_steps}")
    print(f"  Estimated time:     {est_time_min:.1f} min (learned phases only)")
    print(f"  Log file:           {log_path}")
    print("=" * 70)

    # Save plan
    plan = {
        "timestamp": timestamp,
        "num_stitches": num_stitches,
        "stitch_spacing_mm": args.stitch_spacing_mm,
        "insertion_points": [(x, y) for x, y in insertion_points],
        "steps": {"np": args.np_steps, "nt": args.nt_steps, "kt": args.kt_steps},
        "action_horizon": args.action_horizon,
        "sleep_rate": args.sleep_rate,
        "reposition_speed": args.reposition_speed,
        "dry_run": args.dry_run,
        "retry_enabled": retry_enabled,
        "checkpoints": {
            "np": ckpt_np, "nt": ckpt_nt, "kt": ckpt_kt, "gesture": gesture_ckpt
        },
    }
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\n  Plan saved to: {plan_path}")

    # Load models
    print("\n  Loading models...")
    gesture_classifier = load_gesture_classifier(gesture_ckpt)

    models_dict = {}
    norm_stats = {}
    for subtask_name in ["needle_pickup", "needle_throw", "knot_tying"]:
        subtask_cfg = SUBTASK_CONFIGS[subtask_name]
        task_config = TASK_CONFIGS[subtask_cfg["task_name"]]
        use_gesture = subtask_cfg["use_gesture"]

        ckpt_path = {"needle_pickup": ckpt_np, "needle_throw": ckpt_nt,
                     "knot_tying": ckpt_kt}[subtask_name]

        print(f"  Loading {subtask_name} "
              f"({'GC-ACT' if use_gesture else 'ACT v2'})...")
        models_dict[subtask_name] = load_act_model(
            ckpt_path,
            use_gesture=use_gesture,
            image_encoder="efficientnet_b3",
            kl_weight=10,
            gesture_dim=NUM_GESTURE_CLASSES,
        )
        norm_stats[subtask_name] = {
            "mean": task_config["action_mode"][1]["mean"],
            "std": task_config["action_mode"][1]["std"],
        }

    print("  All models loaded.\n")

    # Initialize ROS (real robot only)
    rt, ral, psm1_app, psm2_app = None, None, None, None
    if not args.dry_run:
        import rospy
        import crtk
        from rostopics import ros_topics
        from dvrk_scripts.dvrk_control import example_application

        rospy.init_node("multi_stitch_suturing", anonymous=True)
        rt = ros_topics()
        ral = crtk.ral("multi_stitch_suturing")
        psm1_app = example_application(ral, "PSM1", 1)
        psm2_app = example_application(ral, "PSM2", 1)

        print("  Waiting for camera and robot data...")
        time.sleep(2.0)

        if rt.usb_image_left is None:
            print("  ERROR: No left camera image.")
            sys.exit(1)
        if rt.psm1_pose is None:
            print("  ERROR: No PSM1 pose.")
            sys.exit(1)
        if rt.psm2_pose is None:
            print("  ERROR: No PSM2 pose.")
            sys.exit(1)
        print("  Camera and robot data OK.\n")

    # Initialize CSV logger
    csv_fh, csv_writer = init_csv_logger(log_path)

    # Main multi-stitch loop
    subtask_chain = ["needle_pickup", "needle_throw", "knot_tying"]
    stitch_results = []
    global_step = 0
    t_start = time.time()

    try:
        for stitch_idx in range(num_stitches):
            stitch_start = time.time()
            stitch_success = True
            stitch_steps = 0

            print("=" * 70)
            print(f"  STITCH {stitch_idx + 1}/{num_stitches}")
            print(f"  Insertion point: ({insertion_points[stitch_idx][0]:.4f}, "
                  f"{insertion_points[stitch_idx][1]:.4f}) m")
            print("=" * 70)

            # Safety pause before each stitch (except first or if disabled)
            if stitch_idx > 0 and not args.no_pause and not args.dry_run:
                print("\n  --- SAFETY PAUSE ---")
                print(f"  About to start stitch {stitch_idx + 1}.")
                print("  Check: Is the needle placed correctly?")
                print("  Check: Are the arms in a safe configuration?")
                user_input = input("  Press ENTER to continue, or 'q' to abort: ").strip().lower()
                if user_input == "q":
                    print("  Aborted by user.")
                    break

            # Run the 3 subtasks
            for subtask_name in subtask_chain:
                subtask_cfg = SUBTASK_CONFIGS[subtask_name]
                use_gesture = subtask_cfg["use_gesture"]
                max_steps = steps_map[subtask_name]

                success, steps_done, final_psm1 = run_subtask(
                    policy=models_dict[subtask_name],
                    subtask_name=subtask_name,
                    norm_mean=norm_stats[subtask_name]["mean"],
                    norm_std=norm_stats[subtask_name]["std"],
                    use_gesture=use_gesture,
                    gesture_classifier=gesture_classifier if use_gesture else None,
                    max_steps=max_steps,
                    action_horizon=args.action_horizon,
                    sleep_rate=args.sleep_rate,
                    gesture_conf_threshold=args.gesture_conf_threshold,
                    stitch_idx=stitch_idx,
                    csv_writer=csv_writer,
                    rt=rt, ral=ral, psm1_app=psm1_app, psm2_app=psm2_app,
                    dry_run=args.dry_run,
                )

                stitch_steps += steps_done
                global_step += steps_done

                if not success:
                    if retry_enabled:
                        print(f"\n    RETRY: Re-running {subtask_name.upper()}...")
                        success2, steps_done2, final_psm1 = run_subtask(
                            policy=models_dict[subtask_name],
                            subtask_name=subtask_name,
                            norm_mean=norm_stats[subtask_name]["mean"],
                            norm_std=norm_stats[subtask_name]["std"],
                            use_gesture=use_gesture,
                            gesture_classifier=gesture_classifier if use_gesture else None,
                            max_steps=max_steps,
                            action_horizon=args.action_horizon,
                            sleep_rate=args.sleep_rate,
                            gesture_conf_threshold=args.gesture_conf_threshold,
                            stitch_idx=stitch_idx,
                            csv_writer=csv_writer,
                            rt=rt, ral=ral, psm1_app=psm1_app, psm2_app=psm2_app,
                            dry_run=args.dry_run,
                        )
                        stitch_steps += steps_done2
                        global_step += steps_done2
                        if not success2:
                            print(f"    FAILED: {subtask_name.upper()} failed on retry. "
                                  f"Skipping rest of stitch {stitch_idx + 1}.")
                            stitch_success = False
                            break
                    else:
                        print(f"    FAILED: {subtask_name.upper()} failed. "
                              f"Skipping rest of stitch {stitch_idx + 1}.")
                        stitch_success = False
                        break

            stitch_duration = time.time() - stitch_start

            # Record stitch result
            stitch_results.append({
                "stitch": stitch_idx + 1,
                "success": stitch_success,
                "steps": stitch_steps,
                "duration_s": stitch_duration,
                "insertion_point": insertion_points[stitch_idx],
            })

            print(f"\n  Stitch {stitch_idx + 1} "
                  f"{'COMPLETED' if stitch_success else 'FAILED'} "
                  f"({stitch_steps} steps, {stitch_duration:.1f}s)")

            # Needle repositioning between stitches
            if stitch_idx < num_stitches - 1:
                next_xy = insertion_points[stitch_idx + 1]
                current_xy = insertion_points[stitch_idx]

                print(f"\n    --- Needle Repositioning ---")
                if args.dry_run:
                    print_repositioning_plan(current_xy, next_xy, stitch_idx + 1)
                else:
                    # Get current PSM1 state for repositioning
                    if rt is not None and rt.psm1_pose is not None:
                        current_psm1_state, _ = get_robot_state(rt)
                    else:
                        # Fallback to last known state
                        current_psm1_state = final_psm1

                    execute_repositioning(
                        psm1_app=psm1_app,
                        ral=ral,
                        current_psm1_state=current_psm1_state,
                        next_insertion_xy=next_xy,
                        reposition_speed=args.reposition_speed,
                        logger_info={
                            "stitch_idx": stitch_idx,
                            "global_step": global_step,
                        },
                        csv_writer=csv_writer,
                    )

    except KeyboardInterrupt:
        print(f"\n\n  Multi-stitch interrupted at stitch {stitch_idx + 1}, "
              f"global step {global_step}")

    # Close CSV logger
    csv_fh.close()

    # Summary
    total_duration = time.time() - t_start
    successful_stitches = sum(1 for r in stitch_results if r["success"])

    print("\n" + "=" * 70)
    print("  MULTI-STITCH SUMMARY")
    print("=" * 70)
    print(f"  Total stitches attempted: {len(stitch_results)}/{num_stitches}")
    print(f"  Successful stitches:      {successful_stitches}/{len(stitch_results)}")
    print(f"  Total steps (learned):    {global_step}")
    print(f"  Total duration:           {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print()

    for result in stitch_results:
        status = "OK" if result["success"] else "FAIL"
        pt = result["insertion_point"]
        print(f"  Stitch {result['stitch']}: [{status}] "
              f"{result['steps']} steps, {result['duration_s']:.1f}s "
              f"@ ({pt[0]:.4f}, {pt[1]:.4f})")

    print(f"\n  Log file: {log_path}")
    print(f"  Plan file: {plan_path}")

    # Save summary JSON
    summary_path = os.path.join(log_dir, f"summary_{num_stitches}x_{mode_tag}_{timestamp}.json")
    summary = {
        "timestamp": timestamp,
        "num_stitches_planned": num_stitches,
        "num_stitches_attempted": len(stitch_results),
        "num_stitches_successful": successful_stitches,
        "total_steps": global_step,
        "total_duration_s": total_duration,
        "dry_run": args.dry_run,
        "stitch_results": [
            {
                "stitch": r["stitch"],
                "success": r["success"],
                "steps": r["steps"],
                "duration_s": r["duration_s"],
                "insertion_point_x": r["insertion_point"][0],
                "insertion_point_y": r["insertion_point"][1],
            }
            for r in stitch_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
