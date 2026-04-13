#!/usr/bin/env python3
"""
Parameter Sweep for JHU dVRK Deployment Day
=============================================
Quickly test different parameter configurations for the GC-ACT autonomous
suturing pipeline on the real dVRK. Designed for time-constrained deployment.

Sweepable parameters:
    1. action_scale    -- motion scaling factor (default: 3.0)
    2. chunk_length    -- steps of the 60-step chunk to execute before re-querying (default: 60)
    3. ensemble_k      -- temporal ensemble exponential weight (default: 0.01)
    4. gesture_thresh   -- gesture classifier confidence threshold (default: 0.5)

Usage:
    # Dry run  -- just print the configs:
    python param_sweep.py --sweep action_scale --dry_run

    # Sweep action scale on needle_pickup only (10 inference steps each):
    python param_sweep.py --sweep action_scale --subtask needle_pickup --steps 10

    # Sweep chunk length with custom values:
    python param_sweep.py --sweep chunk_length --values 15 30 45 60

    # Sweep gesture threshold on knot_tying:
    python param_sweep.py --sweep gesture_thresh --subtask knot_tying --steps 10

    # Full pipeline run with specific parameter values:
    python param_sweep.py --sweep action_scale --values 2.0 3.0 --subtask needle_throw --steps 20

    # On the real robot (not dry run):
    python param_sweep.py --sweep action_scale --subtask needle_pickup --steps 10 --real
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime
from collections import OrderedDict

# PARAMETER DEFINITIONS

SWEEP_PARAMS = {
    'action_scale': {
        'description': 'Motion scaling factor applied to position deltas',
        'default': 3.0,
        'values': [1.0, 2.0, 3.0, 4.0],
        'unit': 'x',
    },
    'chunk_length': {
        'description': 'Steps of the 60-step action chunk to execute before re-querying',
        'default': 60,
        'values': [20, 30, 45, 60],
        'unit': 'steps',
    },
    'ensemble_k': {
        'description': 'Temporal ensemble exponential weight (lower = more smoothing)',
        'default': 0.01,
        'values': [0.005, 0.01, 0.02, 0.05],
        'unit': '',
    },
    'gesture_thresh': {
        'description': 'Gesture classifier confidence threshold (below = zero vector fallback)',
        'default': 0.5,
        'values': [0.3, 0.5, 0.7],
        'unit': '',
    },
}

# Default checkpoint paths
DEFAULT_CKPTS = {
    'ckpt_np': os.path.expanduser('~/checkpoints/act_np_v2/policy_best.ckpt'),
    'ckpt_nt': os.path.expanduser('~/checkpoints/act_nt_gcact/policy_best.ckpt'),
    'ckpt_kt': os.path.expanduser('~/checkpoints/act_kt_gcact/policy_best.ckpt'),
    'gesture_ckpt': os.path.expanduser('~/checkpoints/gesture_classifier/gesture_best.ckpt'),
}

SUBTASK_CONFIGS = {
    'needle_pickup': {
        'task_name': 'needle_pickup_all',
        'default_steps': 300,
        'use_gesture': False,
    },
    'needle_throw': {
        'task_name': 'needle_throw_all',
        'default_steps': 600,
        'use_gesture': True,
    },
    'knot_tying': {
        'task_name': 'knot_tying_all',
        'default_steps': 320,
        'use_gesture': True,
    },
}


# METRICS COLLECTION

class SweepMetrics:
    """Collect and summarize metrics for a single parameter configuration."""

    def __init__(self, param_name, param_value, subtask):
        self.param_name = param_name
        self.param_value = param_value
        self.subtask = subtask
        self.inference_times = []
        self.position_deltas_psm1 = []
        self.position_deltas_psm2 = []
        self.jaw_values_psm1 = []
        self.jaw_values_psm2 = []
        self.gesture_predictions = []
        self.gesture_confidences = []
        self.gesture_used = []
        self.action_magnitudes = []
        self.chunk_smoothness = []  # std of consecutive action diffs
        self.total_time = 0.0
        self.num_steps = 0

    def record_inference(self, duration, action_chunk, gesture_str=None,
                         gesture_conf=None, gesture_was_used=None):
        """Record metrics for a single inference step."""
        self.inference_times.append(duration)
        self.num_steps += 1

        # Position deltas (magnitude of displacement across the chunk)
        psm1_pos = action_chunk[:, 0:3]
        psm2_pos = action_chunk[:, 10:13]
        self.position_deltas_psm1.append(np.linalg.norm(psm1_pos[-1] - psm1_pos[0]))
        self.position_deltas_psm2.append(np.linalg.norm(psm2_pos[-1] - psm2_pos[0]))

        # Jaw values (mean across chunk)
        self.jaw_values_psm1.append(np.mean(action_chunk[:, 9]))
        self.jaw_values_psm2.append(np.mean(action_chunk[:, 19]))

        # Action magnitude (mean L2 norm of position component per step)
        pos_all = np.concatenate([psm1_pos, psm2_pos], axis=1)
        self.action_magnitudes.append(np.mean(np.linalg.norm(pos_all, axis=1)))

        # Smoothness: mean L2 of consecutive position differences
        if len(psm1_pos) > 1:
            diffs = np.diff(psm1_pos, axis=0)
            self.chunk_smoothness.append(np.mean(np.linalg.norm(diffs, axis=1)))

        # Gesture info
        if gesture_str is not None:
            self.gesture_predictions.append(gesture_str)
            self.gesture_confidences.append(gesture_conf)
            self.gesture_used.append(gesture_was_used)

    def summary(self):
        """Return a dict of summary statistics."""
        s = OrderedDict()
        s['param'] = self.param_name
        s['value'] = self.param_value
        s['subtask'] = self.subtask
        s['steps'] = self.num_steps

        if self.inference_times:
            s['inf_time_mean_ms'] = np.mean(self.inference_times) * 1000
            s['inf_time_std_ms'] = np.std(self.inference_times) * 1000
            s['inf_hz'] = 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0

        if self.position_deltas_psm1:
            s['psm1_delta_mean_mm'] = np.mean(self.position_deltas_psm1) * 1000
            s['psm2_delta_mean_mm'] = np.mean(self.position_deltas_psm2) * 1000

        if self.action_magnitudes:
            s['action_mag_mean'] = np.mean(self.action_magnitudes)

        if self.chunk_smoothness:
            s['smoothness_mean_mm'] = np.mean(self.chunk_smoothness) * 1000

        if self.jaw_values_psm1:
            s['psm1_jaw_mean'] = np.mean(self.jaw_values_psm1)
            s['psm2_jaw_mean'] = np.mean(self.jaw_values_psm2)

        if self.gesture_confidences:
            s['gesture_conf_mean'] = np.mean(self.gesture_confidences)
            s['gesture_used_pct'] = np.mean(self.gesture_used) * 100
            # Most common gesture
            from collections import Counter
            gesture_counts = Counter(self.gesture_predictions)
            s['top_gesture'] = gesture_counts.most_common(1)[0][0]

        return s


# TEMPORAL ENSEMBLE

class TemporalEnsemble:
    """Temporal ensemble that blends overlapping action chunks.

    When chunk_length < 60, consecutive chunks overlap. This class
    blends them using exponential weighting: newer predictions get
    weight exp(-k * age).
    """

    def __init__(self, action_dim=20, chunk_size=60, k=0.01):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.k = k
        self.buffer = {}  # {target_step: [(weight, action)]}
        self.step_counter = 0

    def add_chunk(self, action_chunk, current_step):
        """Add a new predicted chunk starting at current_step."""
        for i in range(action_chunk.shape[0]):
            target_step = current_step + i
            weight = np.exp(-self.k * i)
            if target_step not in self.buffer:
                self.buffer[target_step] = []
            self.buffer[target_step].append((weight, action_chunk[i].copy()))

    def get_action(self, step):
        """Get the ensembled action for a specific step."""
        if step not in self.buffer or len(self.buffer[step]) == 0:
            return None
        entries = self.buffer[step]
        weights = np.array([w for w, _ in entries])
        actions = np.array([a for _, a in entries])
        weights = weights / weights.sum()
        ensembled = np.sum(weights[:, None] * actions, axis=0)
        return ensembled

    def clear_before(self, step):
        """Remove buffer entries for steps already executed."""
        to_remove = [s for s in self.buffer if s < step]
        for s in to_remove:
            del self.buffer[s]


# CORE SWEEP RUNNER (DRY / SYNTHETIC MODE)

def run_synthetic_sweep(param_name, values, subtask, num_steps, args):
    """Run sweep using synthetic (random) inputs -- no robot, no ROS.

    This is the primary mode for pre-deployment parameter exploration.
    Uses real model weights but random camera images.
    """
    import torch

    # Lazy imports to avoid slow load on --dry_run
    path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT",
                                   os.path.expanduser("~/SutureBot"))
    sys.path.insert(0, os.path.join(path_to_suturebot, 'src', 'act'))
    sys.path.insert(0, os.path.join(path_to_suturebot, 'src'))

    from policy import ACTPolicy
    from dvrk_scripts.constants_dvrk import TASK_CONFIGS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    subtask_cfg = SUBTASK_CONFIGS[subtask]
    task_config = TASK_CONFIGS[subtask_cfg['task_name']]
    use_gesture = subtask_cfg['use_gesture']
    mean = task_config['action_mode'][1]['mean']
    std = task_config['action_mode'][1]['std']

    print(f"\n  Loading {subtask} model...")
    model = _load_model(
        subtask, use_gesture=use_gesture, device=device,
        ckpt_np=args.ckpt_np, ckpt_nt=args.ckpt_nt, ckpt_kt=args.ckpt_kt,
    )

    # Load gesture classifier if needed
    gesture_classifier = None
    if use_gesture:
        gesture_classifier = _load_gesture_classifier(args.gesture_ckpt, device)

    print(f"  Model loaded. Running {len(values)} configurations x {num_steps} steps each.\n")

    all_metrics = []

    for vi, val in enumerate(values):
        print(f"  [{vi+1}/{len(values)}] {param_name} = {val}")

        metrics = SweepMetrics(param_name, val, subtask)

        # Configure parameter for this run
        action_scale = 3.0
        chunk_length = 60
        ensemble_k = 0.01
        gesture_thresh = 0.5

        if param_name == 'action_scale':
            action_scale = val
        elif param_name == 'chunk_length':
            chunk_length = int(val)
        elif param_name == 'ensemble_k':
            ensemble_k = val
        elif param_name == 'gesture_thresh':
            gesture_thresh = val

        # Set up temporal ensemble
        ensemble = TemporalEnsemble(action_dim=20, chunk_size=60, k=ensemble_k)

        # Synthetic inputs
        dummy_images = torch.randn(1, 3, 3, 360, 480).float().to(device)
        qpos_zero = torch.zeros(1, 20).float().to(device)
        dummy_left_rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)

        global_step = 0
        while global_step < num_steps:
            t0 = time.time()

            # Gesture prediction
            gesture_str = None
            gesture_conf = None
            gesture_was_used = None
            kwargs = {}

            if use_gesture and gesture_classifier is not None:
                gesture_str, gesture_onehot, gesture_conf = _predict_gesture(
                    gesture_classifier, dummy_left_rgb, device)
                gesture_was_used = gesture_conf >= gesture_thresh
                if not gesture_was_used:
                    gesture_onehot = np.zeros(10, dtype=np.float32)
                gesture_tensor = torch.from_numpy(gesture_onehot).float().to(device).unsqueeze(0)
                kwargs['gesture_embedding'] = gesture_tensor

            # Forward pass
            with torch.inference_mode():
                raw_chunk = model(qpos_zero, dummy_images, **kwargs).cpu().numpy().squeeze()

            # Unnormalize
            action_chunk = _unnormalize_action(raw_chunk, mean, std)

            # Apply action scaling to position components
            action_chunk[:, 0:3] *= action_scale
            action_chunk[:, 10:13] *= action_scale

            # Add to temporal ensemble
            ensemble.add_chunk(action_chunk, global_step)

            t1 = time.time()
            duration = t1 - t0

            # Record metrics
            metrics.record_inference(
                duration, action_chunk,
                gesture_str=gesture_str,
                gesture_conf=gesture_conf,
                gesture_was_used=gesture_was_used,
            )

            # Advance by chunk_length steps
            steps_this_chunk = min(chunk_length, num_steps - global_step)
            global_step += steps_this_chunk
            ensemble.clear_before(global_step)

        metrics.total_time = sum(metrics.inference_times)
        all_metrics.append(metrics)

    return all_metrics


# CORE SWEEP RUNNER (REAL ROBOT MODE)

def run_real_sweep(param_name, values, subtask, num_steps, args):
    """Run sweep on the real dVRK robot.

    Each configuration runs num_steps inference+execution cycles on one subtask.
    Pauses between configs for operator to reset if needed.
    """
    import torch
    import rospy
    import crtk

    path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT",
                                   os.path.expanduser("~/SutureBot"))
    sys.path.insert(0, os.path.join(path_to_suturebot, 'src', 'act'))
    sys.path.insert(0, os.path.join(path_to_suturebot, 'src'))

    from policy import ACTPolicy
    from dvrk_scripts.constants_dvrk import TASK_CONFIGS
    from rostopics import ros_topics
    from dvrk_scripts.dvrk_control import example_application

    device = "cuda" if torch.cuda.is_available() else "cpu"

    subtask_cfg = SUBTASK_CONFIGS[subtask]
    task_config = TASK_CONFIGS[subtask_cfg['task_name']]
    use_gesture = subtask_cfg['use_gesture']
    mean = task_config['action_mode'][1]['mean']
    std = task_config['action_mode'][1]['std']

    print(f"\n  Loading {subtask} model...")
    model = _load_model(
        subtask, use_gesture=use_gesture, device=device,
        ckpt_np=args.ckpt_np, ckpt_nt=args.ckpt_nt, ckpt_kt=args.ckpt_kt,
    )

    gesture_classifier = None
    if use_gesture:
        gesture_classifier = _load_gesture_classifier(args.gesture_ckpt, device)

    # ROS setup
    rospy.init_node('param_sweep', anonymous=True)
    rt = ros_topics()
    ral = crtk.ral('param_sweep')
    psm1_app = example_application(ral, "PSM1", 1)
    psm2_app = example_application(ral, "PSM2", 1)

    print("  Waiting for camera and robot data...")
    time.sleep(2.0)
    if rt.usb_image_left is None:
        print("ERROR: No left camera image.")
        sys.exit(1)
    if rt.psm1_pose is None:
        print("ERROR: No PSM1 pose.")
        sys.exit(1)
    print("  Camera and robot data OK.\n")

    qpos_zero = torch.zeros(1, 20).float().to(device)
    all_metrics = []

    for vi, val in enumerate(values):
        print(f"\n{'='*60}")
        print(f"  [{vi+1}/{len(values)}] {param_name} = {val}")
        print(f"{'='*60}")

        if vi > 0:
            print("\n  >>> PAUSE: Reset the robot/tissue if needed.")
            print("  >>> Press ENTER to continue, or Ctrl+C to stop.")
            try:
                input()
            except KeyboardInterrupt:
                print("\n  Stopped by operator.")
                break

        metrics = SweepMetrics(param_name, val, subtask)

        action_scale = 3.0
        chunk_length = 60
        ensemble_k = 0.01
        gesture_thresh = 0.5

        if param_name == 'action_scale':
            action_scale = val
        elif param_name == 'chunk_length':
            chunk_length = int(val)
        elif param_name == 'ensemble_k':
            ensemble_k = val
        elif param_name == 'gesture_thresh':
            gesture_thresh = val

        ensemble = TemporalEnsemble(action_dim=20, chunk_size=60, k=ensemble_k)
        global_step = 0

        try:
            while global_step < num_steps:
                # Get camera images
                image_tensor, left_rgb = _get_camera_images_real(rt, device)
                if image_tensor is None:
                    time.sleep(0.1)
                    continue

                qpos_psm1, qpos_psm2 = _get_robot_state(rt)

                t0 = time.time()

                # Gesture
                gesture_str = None
                gesture_conf = None
                gesture_was_used = None
                kwargs = {}

                if use_gesture and gesture_classifier is not None:
                    gesture_str, gesture_onehot, gesture_conf = _predict_gesture(
                        gesture_classifier, left_rgb, device)
                    gesture_was_used = gesture_conf >= gesture_thresh
                    if not gesture_was_used:
                        gesture_onehot = np.zeros(10, dtype=np.float32)
                    gesture_tensor = torch.from_numpy(gesture_onehot).float().to(device).unsqueeze(0)
                    kwargs['gesture_embedding'] = gesture_tensor

                with torch.inference_mode():
                    raw_chunk = model(qpos_zero, image_tensor, **kwargs).cpu().numpy().squeeze()

                action_chunk = _unnormalize_action(raw_chunk, mean, std)
                action_chunk[:, 0:3] *= action_scale
                action_chunk[:, 10:13] *= action_scale

                ensemble.add_chunk(action_chunk, global_step)

                t1 = time.time()

                metrics.record_inference(
                    t1 - t0, action_chunk,
                    gesture_str=gesture_str,
                    gesture_conf=gesture_conf,
                    gesture_was_used=gesture_was_used,
                )

                # Execute chunk_length steps
                from scipy.spatial.transform import Rotation as R
                from sklearn.preprocessing import normalize

                wp_psm1, wp_psm2 = _actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2)

                steps_this_chunk = min(chunk_length, num_steps - global_step,
                                       action_chunk.shape[0])
                for j in range(steps_this_chunk):
                    ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
                    ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])
                    time.sleep(args.sleep_rate)
                    global_step += 1

                ensemble.clear_before(global_step)

                if global_step % 5 == 0:
                    g_info = ""
                    if gesture_str:
                        g_info = f" [{gesture_str} conf={gesture_conf:.2f}]"
                    print(f"    step {global_step}/{num_steps}{g_info}")

        except KeyboardInterrupt:
            print(f"\n  Config interrupted at step {global_step}")

        metrics.total_time = sum(metrics.inference_times)
        all_metrics.append(metrics)

    return all_metrics


# SHARED HELPER FUNCTIONS

def _load_model(subtask, use_gesture, device, ckpt_np, ckpt_nt, ckpt_kt):
    """Load the appropriate ACT/GC-ACT model for a subtask."""
    from policy import ACTPolicy

    ckpt_map = {
        'needle_pickup': ckpt_np,
        'needle_throw': ckpt_nt,
        'knot_tying': ckpt_kt,
    }
    ckpt_path = ckpt_map[subtask]

    image_encoder = 'efficientnet_b3'  # v2 and GC-ACT both use efficientnet_b3
    kl_weight = 10

    saved_argv = sys.argv
    argv = [
        'act', '--task_name', 'needle_pickup_all',
        '--ckpt_dir', '/tmp', '--policy_class', 'ACT',
        '--seed', '0', '--num_epochs', '1',
        '--kl_weight', str(kl_weight), '--chunk_size', '60',
        '--hidden_dim', '512', '--dim_feedforward', '3200',
        '--lr', '1e-5', '--batch_size', '8',
        '--image_encoder', image_encoder,
    ]
    if use_gesture:
        argv.extend(['--use_gesture', '--gesture_dim', '10'])
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
        "camera_names": ['left', 'left_wrist', 'right_wrist'],
        "multi_gpu": False,
    }
    if use_gesture:
        policy_config["use_gesture"] = True
        policy_config["gesture_dim"] = 10

    import torch
    policy = ACTPolicy(policy_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    sys.argv = saved_argv

    mode = "GC-ACT" if use_gesture else "ACT v2"
    print(f"  [{mode}] Loaded {subtask} from {ckpt_path}")
    return policy


def _load_gesture_classifier(ckpt_path, device):
    """Load gesture classifier."""
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    class GestureClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        def forward(self, x):
            return self.backbone(x)

    model = GestureClassifier(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"  [GestureClassifier] Loaded from {ckpt_path}")
    return model


def _predict_gesture(classifier, image_rgb, device):
    """Predict gesture from a single RGB image."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    GESTURE_LABELS = ['G2', 'G3', 'G6', 'G7', 'G10', 'G11', 'G13', 'G14', 'G15', 'G16']

    img_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    gesture_str = GESTURE_LABELS[pred_idx.item()]
    gesture_onehot = np.zeros(10, dtype=np.float32)
    gesture_onehot[pred_idx.item()] = 1.0
    return gesture_str, gesture_onehot, conf.item()


def _unnormalize_action(action_chunk, mean, std):
    """Unnormalize positions and jaw; rotations stay raw."""
    unnormalized = action_chunk * std + mean
    unnormalized[:, 3:9] = action_chunk[:, 3:9]
    unnormalized[:, 13:19] = action_chunk[:, 13:19]
    return unnormalized


def _actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2):
    """Convert unnormalized 20D action chunk to dVRK waypoints."""
    from scipy.spatial.transform import Rotation as R
    from sklearn.preprocessing import normalize

    chunk_size = action_chunk.shape[0]
    action = action_chunk.copy()

    action[:, 0:3] = action[:, 0:3] - action[0, 0:3]
    action[:, 10:13] = action[:, 10:13] - action[0, 10:13]

    def convert_6d_to_quat(rot6d, current_quat_xyzw):
        c1 = rot6d[:, 0:3]
        c2 = rot6d[:, 3:6]
        c1 = normalize(c1, axis=1)
        dot = np.sum(c1 * c2, axis=1).reshape(-1, 1)
        c2 = normalize(c2 - dot * c1, axis=1)
        c3 = np.cross(c1, c2)
        r_mat = np.dstack((c1, c2, c3))
        rots = R.from_matrix(r_mat)
        rot_init = R.from_quat(current_quat_xyzw)
        return (rot_init * rots).as_quat()

    wp_psm1 = np.zeros((chunk_size, 8))
    wp_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3]
    wp_psm1[:, 3:7] = convert_6d_to_quat(action[:, 3:9], qpos_psm1[3:7])
    wp_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)

    wp_psm2 = np.zeros((chunk_size, 8))
    wp_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13]
    wp_psm2[:, 3:7] = convert_6d_to_quat(action[:, 13:19], qpos_psm2[3:7])
    wp_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)

    return wp_psm1, wp_psm2


def _get_camera_images_real(rt, device):
    """Get 3 camera images from ROS topics."""
    import torch
    import cv2

    if rt.usb_image_left is None or rt.endo_cam_psm1 is None or rt.endo_cam_psm2 is None:
        return None, None

    def process(compressed_data):
        img = np.frombuffer(compressed_data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (480, 360))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    left = process(rt.usb_image_left.data)
    left_rgb = left.copy()
    left_chw = np.transpose(left, (2, 0, 1))
    lw = np.transpose(process(rt.endo_cam_psm2.data), (2, 0, 1))
    rw = np.transpose(process(rt.endo_cam_psm1.data), (2, 0, 1))

    image_data = np.stack([left_chw, lw, rw], axis=0)
    image_tensor = torch.from_numpy(image_data / 255.0).float().to(device).unsqueeze(0)
    return image_tensor, left_rgb


def _get_robot_state(rt):
    """Get current PSM1 and PSM2 state from ROS topics."""
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


# OUTPUT FORMATTING

def print_summary_table(all_metrics, param_name):
    """Print a formatted summary table of sweep results."""
    if not all_metrics:
        print("\n  No results to display.")
        return

    summaries = [m.summary() for m in all_metrics]

    # Determine which columns to show based on what's available
    always_cols = ['value', 'steps', 'inf_time_mean_ms', 'inf_hz',
                   'psm1_delta_mean_mm', 'psm2_delta_mean_mm', 'smoothness_mean_mm']
    gesture_cols = ['gesture_conf_mean', 'gesture_used_pct', 'top_gesture']

    has_gesture = any('gesture_conf_mean' in s for s in summaries)

    cols = always_cols[:]
    if has_gesture:
        cols.extend(gesture_cols)

    # Column display names and widths
    col_names = {
        'value': param_name,
        'steps': 'Steps',
        'inf_time_mean_ms': 'Inf (ms)',
        'inf_hz': 'Hz',
        'psm1_delta_mean_mm': 'PSM1 d(mm)',
        'psm2_delta_mean_mm': 'PSM2 d(mm)',
        'smoothness_mean_mm': 'Smooth(mm)',
        'gesture_conf_mean': 'G.Conf',
        'gesture_used_pct': 'G.Used%',
        'top_gesture': 'TopGest',
    }

    col_widths = {
        'value': max(len(param_name), 10),
        'steps': 6,
        'inf_time_mean_ms': 10,
        'inf_hz': 8,
        'psm1_delta_mean_mm': 11,
        'psm2_delta_mean_mm': 11,
        'smoothness_mean_mm': 11,
        'gesture_conf_mean': 7,
        'gesture_used_pct': 8,
        'top_gesture': 8,
    }

    # Header
    header = "  "
    for c in cols:
        name = col_names.get(c, c)
        w = col_widths.get(c, 10)
        header += f"{name:>{w}}  "
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    # Rows
    for s in summaries:
        row = "  "
        for c in cols:
            w = col_widths.get(c, 10)
            val = s.get(c, '-')
            if isinstance(val, float):
                row += f"{val:>{w}.3f}  "
            elif isinstance(val, int):
                row += f"{val:>{w}d}  "
            else:
                row += f"{str(val):>{w}}  "
        print(row)

    print(f"{'='*len(header)}")


def save_results(all_metrics, param_name, output_dir):
    """Save sweep results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"sweep_{param_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    results = {
        'param_name': param_name,
        'timestamp': timestamp,
        'configs': [m.summary() for m in all_metrics],
    }

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = json.loads(json.dumps(results, default=convert))

    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n  Results saved to {filepath}")
    return filepath


# DRY RUN MODE

def print_dry_run(param_name, values, subtask, num_steps):
    """Print the configurations that would be tested without executing."""
    print(f"\n{'='*60}")
    print(f"  PARAMETER SWEEP  -- DRY RUN")
    print(f"{'='*60}")
    print(f"  Parameter: {param_name}")
    print(f"  Description: {SWEEP_PARAMS[param_name]['description']}")
    print(f"  Default value: {SWEEP_PARAMS[param_name]['default']}")
    print(f"  Subtask: {subtask}")
    print(f"  Steps per config: {num_steps}")
    print(f"  Total configs: {len(values)}")
    print(f"  Estimated time: ~{len(values) * num_steps * 0.15:.0f}s (synthetic)")
    print()

    use_gesture = SUBTASK_CONFIGS[subtask]['use_gesture']

    for vi, val in enumerate(values):
        is_default = (val == SWEEP_PARAMS[param_name]['default'])
        marker = " (DEFAULT)" if is_default else ""
        unit = SWEEP_PARAMS[param_name]['unit']

        print(f"  Config {vi+1}: {param_name} = {val}{unit}{marker}")

        # Show effective settings for this config
        settings = {
            'action_scale': 3.0,
            'chunk_length': 60,
            'ensemble_k': 0.01,
            'gesture_thresh': 0.5,
        }
        settings[param_name] = val
        print(f"    action_scale={settings['action_scale']}, "
              f"chunk_length={settings['chunk_length']}, "
              f"ensemble_k={settings['ensemble_k']}, "
              f"gesture_thresh={settings['gesture_thresh']}")
        print(f"    use_gesture={use_gesture}")

    # Checkpoint info
    print(f"\n  Checkpoints that will be loaded:")
    if subtask == 'needle_pickup':
        print(f"    NP (ACT v2): {DEFAULT_CKPTS['ckpt_np']}")
    elif subtask == 'needle_throw':
        print(f"    NT (GC-ACT): {DEFAULT_CKPTS['ckpt_nt']}")
        print(f"    Gesture:     {DEFAULT_CKPTS['gesture_ckpt']}")
    elif subtask == 'knot_tying':
        print(f"    KT (GC-ACT): {DEFAULT_CKPTS['ckpt_kt']}")
        print(f"    Gesture:     {DEFAULT_CKPTS['gesture_ckpt']}")

    print(f"\n  To run this sweep:")
    vals_str = ' '.join(str(v) for v in values)
    print(f"    python param_sweep.py --sweep {param_name} --subtask {subtask} "
          f"--steps {num_steps} --values {vals_str}")
    print(f"\n  To run on the real robot:")
    print(f"    python param_sweep.py --sweep {param_name} --subtask {subtask} "
          f"--steps {num_steps} --values {vals_str} --real")
    print()


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for GC-ACT dVRK deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See what a sweep would do (no model loading):
  python param_sweep.py --sweep action_scale --dry_run

  # Run action_scale sweep on needle_pickup (synthetic, 10 steps):
  python param_sweep.py --sweep action_scale --subtask needle_pickup --steps 10

  # Run gesture_thresh sweep on knot_tying with custom values:
  python param_sweep.py --sweep gesture_thresh --subtask knot_tying --values 0.3 0.5 0.7 0.9

  # Sweep all parameters quickly (list available sweeps):
  python param_sweep.py --list

  # Run on real robot:
  python param_sweep.py --sweep chunk_length --subtask needle_pickup --steps 10 --real
""")

    parser.add_argument('--sweep', type=str, choices=list(SWEEP_PARAMS.keys()),
                        help='Which parameter to sweep')
    parser.add_argument('--values', nargs='+', type=float, default=None,
                        help='Custom values to sweep (overrides defaults)')
    parser.add_argument('--subtask', type=str, default='needle_pickup',
                        choices=['needle_pickup', 'needle_throw', 'knot_tying'],
                        help='Subtask to test (default: needle_pickup)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Inference steps per configuration (default: 10)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configs without loading models or executing')
    parser.add_argument('--real', action='store_true',
                        help='Run on real robot (requires ROS + dVRK)')
    parser.add_argument('--list', action='store_true',
                        help='List all sweepable parameters and exit')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/sweep_results'),
                        help='Directory for sweep result JSON files')
    parser.add_argument('--sleep_rate', type=float, default=0.1,
                        help='Seconds between action steps on real robot (default: 0.1)')

    # Checkpoint overrides
    parser.add_argument('--ckpt_np', type=str, default=DEFAULT_CKPTS['ckpt_np'],
                        help='Needle pickup checkpoint')
    parser.add_argument('--ckpt_nt', type=str, default=DEFAULT_CKPTS['ckpt_nt'],
                        help='Needle throw checkpoint')
    parser.add_argument('--ckpt_kt', type=str, default=DEFAULT_CKPTS['ckpt_kt'],
                        help='Knot tying checkpoint')
    parser.add_argument('--gesture_ckpt', type=str, default=DEFAULT_CKPTS['gesture_ckpt'],
                        help='Gesture classifier checkpoint')

    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"\n{'='*60}")
        print(f"  SWEEPABLE PARAMETERS")
        print(f"{'='*60}")
        for name, cfg in SWEEP_PARAMS.items():
            unit = f" ({cfg['unit']})" if cfg['unit'] else ""
            vals_str = ', '.join(str(v) for v in cfg['values'])
            print(f"\n  {name}{unit}")
            print(f"    {cfg['description']}")
            print(f"    Default: {cfg['default']}")
            print(f"    Sweep values: [{vals_str}]")
        print()
        return

    if args.sweep is None:
        parser.print_help()
        print("\n  ERROR: --sweep is required (or use --list to see options)")
        sys.exit(1)

    param_name = args.sweep
    values = args.values if args.values is not None else SWEEP_PARAMS[param_name]['values']
    subtask = args.subtask
    num_steps = args.steps

    # Validation: gesture_thresh sweep only makes sense on gesture-conditioned subtasks
    if param_name == 'gesture_thresh' and subtask == 'needle_pickup':
        print("  WARNING: gesture_thresh has no effect on needle_pickup (no gesture conditioning).")
        print("  Switching to needle_throw.")
        subtask = 'needle_throw'

    # Dry run
    if args.dry_run:
        print_dry_run(param_name, values, subtask, num_steps)
        return

    # Run sweep
    print(f"\n{'='*60}")
    print(f"  PARAMETER SWEEP: {param_name}")
    print(f"  Subtask: {subtask} | Steps: {num_steps} | Configs: {len(values)}")
    print(f"  Mode: {'REAL ROBOT' if args.real else 'SYNTHETIC (no robot)'}")
    print(f"{'='*60}")

    t_start = time.time()

    if args.real:
        all_metrics = run_real_sweep(param_name, values, subtask, num_steps, args)
    else:
        all_metrics = run_synthetic_sweep(param_name, values, subtask, num_steps, args)

    t_total = time.time() - t_start

    # Print results
    print_summary_table(all_metrics, param_name)
    print(f"\n  Total sweep time: {t_total:.1f}s")

    # Save results
    filepath = save_results(all_metrics, param_name, args.output_dir)

    # Recommendations
    print(f"\n  QUICK ANALYSIS:")
    summaries = [m.summary() for m in all_metrics]

    if param_name == 'action_scale':
        # Recommend based on smoothness vs displacement tradeoff
        best = min(summaries, key=lambda s: s.get('smoothness_mean_mm', float('inf')))
        print(f"    Smoothest: {param_name}={best['value']} "
              f"(smoothness={best.get('smoothness_mean_mm', 0):.3f}mm)")
        biggest = max(summaries, key=lambda s: s.get('psm1_delta_mean_mm', 0))
        print(f"    Largest motion: {param_name}={biggest['value']} "
              f"(PSM1 delta={biggest.get('psm1_delta_mean_mm', 0):.3f}mm)")

    elif param_name == 'chunk_length':
        # Recommend based on inference frequency
        fastest = max(summaries, key=lambda s: s.get('inf_hz', 0))
        print(f"    Fastest re-query: {param_name}={fastest['value']} "
              f"({fastest.get('inf_hz', 0):.1f} Hz)")

    elif param_name == 'ensemble_k':
        best = min(summaries, key=lambda s: s.get('smoothness_mean_mm', float('inf')))
        print(f"    Smoothest: {param_name}={best['value']} "
              f"(smoothness={best.get('smoothness_mean_mm', 0):.3f}mm)")

    elif param_name == 'gesture_thresh':
        if any('gesture_used_pct' in s for s in summaries):
            for s in summaries:
                pct = s.get('gesture_used_pct', 0)
                print(f"    thresh={s['value']}: gesture used {pct:.1f}% of the time "
                      f"(mean conf={s.get('gesture_conf_mean', 0):.3f})")

    print(f"\n  Done. Results at {filepath}")


if __name__ == '__main__':
    main()
