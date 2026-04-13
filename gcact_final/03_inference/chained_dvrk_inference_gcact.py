#!/usr/bin/env python3
# chained_dvrk_inference_gcact.py  -- GC-ACT Robot Deployment Script
# Same as chained_dvrk_inference.py but with our novel contribution: real-time
# gesture classification. A separate classifier looks at the camera feed and
# predicts what surgical gesture is happening (e.g., "pushing needle through
# tissue"). This gesture label is fed into the ACT model as extra context,
# helping it produce more accurate actions for each phase of the suture.
# Needle Pickup uses ACT v2 (no gesture labels exist for that subtask).
# Needle Throw and Knot Tying use GC-ACT.
"""
GC-ACT Chained dVRK Inference  -- Gesture-Conditioned ACT on the real dVRK.

Same as chained_dvrk_inference.py but with real-time gesture classification
that conditions the ACT policy at each inference step.

Architecture:
    Camera image -> Gesture Classifier -> gesture one-hot (10D)
    Camera images + gesture one-hot -> GC-ACT -> action chunk (60, 20)

Sequences: Needle Pickup (v2) -> Needle Throw (GC-ACT) -> Knot Tying (GC-ACT)
NP uses v2 checkpoint (no gesture labels for NP).

Usage:
    python chained_dvrk_inference_gcact.py \
        --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_gcact/policy_best.ckpt \
        --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt

    # Dry run:
    python chained_dvrk_inference_gcact.py \
        --ckpt_np ~/checkpoints/act_np_v2/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_gcact/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_gcact/policy_best.ckpt \
        --gesture_ckpt ~/checkpoints/gesture_classifier/gesture_best.ckpt \
        --dry_run
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from torchvision import models, transforms

path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT",
                               os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(path_to_suturebot, 'src', 'act'))
sys.path.insert(0, os.path.join(path_to_suturebot, 'src'))

from policy import ACTPolicy
from dvrk_scripts.constants_dvrk import TASK_CONFIGS


# CONSTANTS

GESTURE_LABELS = ['G2', 'G3', 'G6', 'G7', 'G10', 'G11', 'G13', 'G14', 'G15', 'G16']
IDX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_LABELS)}
NUM_GESTURE_CLASSES = len(GESTURE_LABELS)

SUBTASK_CONFIGS = {
    'needle_pickup': {
        'task_name': 'needle_pickup_all',
        'default_steps': 300,
        'use_gesture': False,  # NP has no gesture labels
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


# GESTURE CLASSIFIER

class GestureClassifier(nn.Module):
    """ResNet18 + FC head for gesture classification (matches train_gesture_classifier.py)."""

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
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"[GestureClassifier] Loaded from {ckpt_path}")
    return model


# Gesture classifier preprocessing (matches training transforms)
GESTURE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_gesture(classifier, image_rgb, device="cuda"):
    """Predict gesture from a single RGB image.

    Args:
        classifier: GestureClassifier model
        image_rgb: (H, W, 3) numpy array, RGB, uint8

    Returns:
        gesture_str: predicted gesture label (e.g., 'G3')
        gesture_onehot: (10,) numpy array
        confidence: float
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

def load_act_model(ckpt_path, use_gesture=False, image_encoder="resnet18",
                   kl_weight=1, gesture_dim=10, device="cuda"):
    """Load a single ACT or GC-ACT model from checkpoint."""
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
        argv.extend(['--use_gesture', '--gesture_dim', str(gesture_dim)])
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
        policy_config["gesture_dim"] = gesture_dim

    policy = ACTPolicy(policy_config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    sys.argv = saved_argv

    mode = "GC-ACT" if use_gesture else "ACT"
    print(f"[{mode}] {image_encoder} loaded from {ckpt_path}")
    return policy


# ACTION CONVERSION (same as chained_dvrk_inference.py)

def unnormalize_action(action_chunk, mean, std):
    """Unnormalize positions and jaw; rotations stay raw."""
    unnormalized = action_chunk * std + mean
    unnormalized[:, 3:9] = action_chunk[:, 3:9]
    unnormalized[:, 13:19] = action_chunk[:, 13:19]
    return unnormalized


def convert_6d_to_quat(rot6d_chunk, current_quat_xyzw):
    """Convert 6D rotation predictions to quaternion targets."""
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
    """Convert unnormalized 20D action chunk to dVRK waypoints."""
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
        image_tensor: (1, 3, 3, 360, 480) float32 tensor for ACT
        left_rgb: (360, 480, 3) uint8 numpy array for gesture classifier
    """
    if rt.usb_image_left is None or rt.endo_cam_psm1 is None or rt.endo_cam_psm2 is None:
        return None, None

    left = process_compressed_image(rt.usb_image_left.data)
    left_rgb = left.copy()  # Keep for gesture classifier (HWC, RGB)

    left_chw = np.transpose(left, (2, 0, 1))
    lw = np.transpose(process_compressed_image(rt.endo_cam_psm2.data), (2, 0, 1))
    rw = np.transpose(process_compressed_image(rt.endo_cam_psm1.data), (2, 0, 1))

    image_data = np.stack([left_chw, lw, rw], axis=0)
    image_tensor = torch.from_numpy(image_data / 255.0).float().cuda().unsqueeze(0)
    return image_tensor, left_rgb


def get_robot_state(rt):
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


# MAIN

def main():
    parser = argparse.ArgumentParser(description="GC-ACT Chained inference on real dVRK")
    parser.add_argument('--ckpt_np', type=str, default=None,
                        help='Needle pickup checkpoint (v2, no gesture)')
    parser.add_argument('--ckpt_nt', type=str, default=None,
                        help='Needle throw GC-ACT checkpoint')
    parser.add_argument('--ckpt_kt', type=str, default=None,
                        help='Knot tying GC-ACT checkpoint')
    parser.add_argument('--gesture_ckpt', type=str, required=True,
                        help='Gesture classifier checkpoint')
    parser.add_argument('--subtasks', nargs='+', default=None,
                        choices=['needle_pickup', 'needle_throw', 'knot_tying'],
                        help='Which subtasks to run (default: all with checkpoints)')
    parser.add_argument('--np_steps', type=int, default=300)
    parser.add_argument('--nt_steps', type=int, default=600)
    parser.add_argument('--kt_steps', type=int, default=320)
    parser.add_argument('--action_horizon', type=int, default=20,
                        help='Steps of the 60-step chunk to execute before re-querying')
    parser.add_argument('--sleep_rate', type=float, default=0.1,
                        help='Seconds between action steps (default: 0.1 = 10Hz)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Load models and print actions without ROS/robot')
    parser.add_argument('--image_encoder_np', type=str, default='efficientnet_b3',
                        help='Backbone for NP (default: efficientnet_b3 for v2)')
    parser.add_argument('--image_encoder_gcact', type=str, default='efficientnet_b3',
                        help='Backbone for GC-ACT models (default: efficientnet_b3)')
    args = parser.parse_args()

    ckpt_map = {
        'needle_pickup': args.ckpt_np,
        'needle_throw': args.ckpt_nt,
        'knot_tying': args.ckpt_kt,
    }
    steps_map = {
        'needle_pickup': args.np_steps,
        'needle_throw': args.nt_steps,
        'knot_tying': args.kt_steps,
    }

    if args.subtasks:
        subtask_sequence = args.subtasks
    else:
        subtask_sequence = [s for s in ['needle_pickup', 'needle_throw', 'knot_tying']
                           if ckpt_map[s] is not None]

    if not subtask_sequence:
        print("ERROR: No checkpoints provided.")
        sys.exit(1)

    # Load models
    print("=" * 60)
    print("  GC-ACT CHAINED dVRK INFERENCE")
    print("=" * 60)

    # Load gesture classifier
    gesture_classifier = load_gesture_classifier(args.gesture_ckpt)

    models_dict = {}
    norm_stats = {}
    for subtask_name in subtask_sequence:
        ckpt_path = ckpt_map[subtask_name]
        if ckpt_path is None:
            print(f"ERROR: No checkpoint for {subtask_name}")
            sys.exit(1)

        subtask_cfg = SUBTASK_CONFIGS[subtask_name]
        task_config = TASK_CONFIGS[subtask_cfg['task_name']]
        use_gesture = subtask_cfg['use_gesture']

        if use_gesture:
            image_encoder = args.image_encoder_gcact
        else:
            image_encoder = args.image_encoder_np

        print(f"  Loading {subtask_name} ({'GC-ACT' if use_gesture else 'ACT v2'})...")
        models_dict[subtask_name] = load_act_model(
            ckpt_path,
            use_gesture=use_gesture,
            image_encoder=image_encoder,
            kl_weight=10,
            gesture_dim=NUM_GESTURE_CLASSES,
        )
        norm_stats[subtask_name] = {
            'mean': task_config['action_mode'][1]['mean'],
            'std': task_config['action_mode'][1]['std'],
        }

    chain = [(s, steps_map[s]) for s in subtask_sequence]
    total_steps = sum(steps for _, steps in chain)
    print(f"\n  Subtask chain: {' -> '.join(s.upper() for s, _ in chain)}")
    print(f"  Steps: {' + '.join(f'{s}={n}' for s, n in chain)} = {total_steps} total")
    print(f"  Action horizon: {args.action_horizon}")
    print(f"  Dry run: {args.dry_run}")

    # Dry run mode
    if args.dry_run:
        print("\n  [DRY RUN] Testing model inference without ROS...")
        dummy_images = torch.randn(1, 3, 3, 360, 480).float().cuda()
        dummy_left_rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        qpos_zero = torch.zeros(1, 20).float().cuda()

        # Test gesture classifier
        gesture_str, gesture_onehot, conf = predict_gesture(
            gesture_classifier, dummy_left_rgb)
        print(f"\n  Gesture classifier test: {gesture_str} (conf={conf:.3f})")

        for subtask_name in subtask_sequence:
            policy = models_dict[subtask_name]
            subtask_cfg = SUBTASK_CONFIGS[subtask_name]
            use_gesture = subtask_cfg['use_gesture']

            with torch.inference_mode():
                kwargs = {}
                if use_gesture:
                    gesture_tensor = torch.from_numpy(gesture_onehot).float().cuda().unsqueeze(0)
                    kwargs['gesture_embedding'] = gesture_tensor

                action_chunk = policy(qpos_zero, dummy_images, **kwargs).cpu().numpy().squeeze()

            unnorm = unnormalize_action(action_chunk,
                                        norm_stats[subtask_name]['mean'],
                                        norm_stats[subtask_name]['std'])
            mode = "GC-ACT" if use_gesture else "ACT v2"
            print(f"\n  {subtask_name} ({mode}):")
            print(f"    Action chunk shape: {action_chunk.shape}")
            print(f"    PSM1 pos range: x=[{unnorm[:,0].min():.4f}, {unnorm[:,0].max():.4f}]")
            print(f"    PSM2 pos range: x=[{unnorm[:,10].min():.4f}, {unnorm[:,10].max():.4f}]")

        print("\n  [DRY RUN] All models loaded and inference OK.")
        return

    # Real robot mode
    import rospy
    import crtk
    from rostopics import ros_topics
    from dvrk_scripts.dvrk_control import example_application

    rospy.init_node('gcact_dvrk_inference', anonymous=True)
    rt = ros_topics()
    ral = crtk.ral('gcact_dvrk_inference')
    psm1_app = example_application(ral, "PSM1", 1)
    psm2_app = example_application(ral, "PSM2", 1)

    print("\n  Waiting for camera and robot data...")
    time.sleep(2.0)

    if rt.usb_image_left is None:
        print("ERROR: No left camera image.")
        sys.exit(1)
    if rt.psm1_pose is None:
        print("ERROR: No PSM1 pose.")
        sys.exit(1)
    print("  Camera and robot data OK.")

    qpos_zero = torch.zeros(1, 20).float().cuda()
    global_step = 0
    gesture_log = []

    print("\n  Press Ctrl+C to stop at any time.")
    print("=" * 60)

    try:
        for subtask_idx, (subtask_name, max_steps) in enumerate(chain):
            policy = models_dict[subtask_name]
            mean = norm_stats[subtask_name]['mean']
            std = norm_stats[subtask_name]['std']
            use_gesture = SUBTASK_CONFIGS[subtask_name]['use_gesture']

            mode = "GC-ACT" if use_gesture else "ACT v2"
            print(f"\n  [{subtask_idx+1}/{len(chain)}] Starting {subtask_name.upper()} "
                  f"({mode}, {max_steps} steps)")

            subtask_step = 0
            while subtask_step < max_steps:
                image_tensor, left_rgb = get_camera_images(rt)
                if image_tensor is None:
                    time.sleep(0.1)
                    continue

                qpos_psm1, qpos_psm2 = get_robot_state(rt)

                # Classify gesture (for GC-ACT subtasks)
                gesture_emb = None
                if use_gesture:
                    gesture_str, gesture_onehot, conf = predict_gesture(
                        gesture_classifier, left_rgb)
                    # Fall back to zero vector if classifier is uncertain
                    if conf < 0.5:
                        gesture_onehot = np.zeros(NUM_GESTURE_CLASSES, dtype=np.float32)
                    gesture_emb = torch.from_numpy(gesture_onehot).float().cuda().unsqueeze(0)
                    gesture_log.append({
                        'subtask': subtask_name,
                        'step': subtask_step,
                        'gesture': gesture_str,
                        'confidence': conf,
                        'used': conf >= 0.5,
                    })

                # Run inference
                with torch.inference_mode():
                    kwargs = {}
                    if gesture_emb is not None:
                        kwargs['gesture_embedding'] = gesture_emb
                    action_chunk = policy(qpos_zero, image_tensor, **kwargs).cpu().numpy().squeeze()

                action_chunk = unnormalize_action(action_chunk, mean, std)
                wp_psm1, wp_psm2 = actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2)

                steps_to_execute = min(args.action_horizon,
                                       max_steps - subtask_step,
                                       action_chunk.shape[0])

                for j in range(steps_to_execute):
                    ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
                    ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])
                    time.sleep(args.sleep_rate)
                    subtask_step += 1
                    global_step += 1

                # Log actions for this chunk
                gesture_log.append({
                    'subtask': subtask_name,
                    'step': subtask_step,
                    'type': 'action',
                    'wp_psm1_first': wp_psm1[0].tolist(),
                    'wp_psm2_first': wp_psm2[0].tolist(),
                    'steps_executed': steps_to_execute,
                })

                if subtask_step % 50 == 0:
                    gesture_info = ""
                    if use_gesture and gesture_log:
                        last = gesture_log[-1]
                        gesture_info = f" gesture={last['gesture']}({last['confidence']:.2f})"
                    print(f"    {subtask_name}: step {subtask_step}/{max_steps}"
                          f" (global: {global_step}/{total_steps}){gesture_info}")

            print(f"  {subtask_name.upper()} complete ({subtask_step} steps)")

    except KeyboardInterrupt:
        print(f"\n  Interrupted at global step {global_step}")

    # Save gesture log
    if gesture_log:
        import json
        log_path = os.path.expanduser('~/gesture_inference_log.json')
        with open(log_path, 'w') as f:
            json.dump(gesture_log, f, indent=2)
        print(f"\n  Gesture log saved to {log_path} ({len(gesture_log)} entries)")

    print(f"\n  Done. Total steps executed: {global_step}")


if __name__ == '__main__':
    main()
