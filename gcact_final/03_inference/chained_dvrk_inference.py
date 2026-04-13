#!/usr/bin/env python3
# chained_dvrk_inference.py  -- ACT v2 Robot Deployment Script
# Runs the trained ACT v2 models on the real da Vinci robot. Chains three
# subtask models in sequence: Needle Pickup -> Needle Throw -> Knot Tying.
# Each model takes live camera images, predicts a chunk of future actions,
# and sends position/rotation/gripper commands to the robot arms.
# Includes a --dry_run mode for testing without moving the robot.
# This is one of the two scripts that ships to JHU for real-robot trials.
"""
Chained dVRK Inference  -- Run 3 per-subtask ACT models sequentially on the real dVRK.

Sequences: Needle Pickup -> Needle Throw -> Knot Tying
Each subtask uses its own checkpoint and normalization stats.

Usage:
    python chained_dvrk_inference.py \
        --ckpt_np ~/checkpoints/act_np_10t_kl1/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_10t_kl1/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_10t_kl1/policy_best.ckpt

    # Dry run (no ROS/robot, just verify models load and produce actions):
    python chained_dvrk_inference.py \
        --ckpt_np ~/checkpoints/act_np_10t_kl1/policy_best.ckpt \
        --ckpt_nt ~/checkpoints/act_nt_10t_kl1/policy_best.ckpt \
        --ckpt_kt ~/checkpoints/act_kt_10t_kl1/policy_best.ckpt \
        --dry_run

    # Single subtask only:
    python chained_dvrk_inference.py \
        --ckpt_np ~/checkpoints/act_np_10t_kl1/policy_best.ckpt \
        --subtasks needle_pickup
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT",
                               os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(path_to_suturebot, 'src', 'act'))
sys.path.insert(0, os.path.join(path_to_suturebot, 'src'))

from policy import ACTPolicy
from dvrk_scripts.constants_dvrk import TASK_CONFIGS


# NORM STATS per subtask (from constants_dvrk.py task configs)

SUBTASK_CONFIGS = {
    'needle_pickup': {
        'task_name': 'needle_pickup_all',
        'default_steps': 300,
    },
    'needle_throw': {
        'task_name': 'needle_throw_all',
        'default_steps': 600,
    },
    'knot_tying': {
        'task_name': 'knot_tying_all',
        'default_steps': 320,
    },
}


# MODEL LOADING

def load_act_model(ckpt_path, device="cuda", image_encoder="resnet18", kl_weight=1):
    """Load a single ACT model from checkpoint."""
    # Temporarily override sys.argv so detr/main.py's argparse doesn't crash
    saved_argv = sys.argv
    sys.argv = [
        'act', '--task_name', 'needle_pickup_all',
        '--ckpt_dir', '/tmp', '--policy_class', 'ACT',
        '--seed', '0', '--num_epochs', '1',
        '--kl_weight', str(kl_weight), '--chunk_size', '60',
        '--hidden_dim', '512', '--dim_feedforward', '3200',
        '--lr', '1e-5', '--batch_size', '8',
        '--image_encoder', image_encoder,
    ]

    policy_config = {
        "lr": 1e-5,
        "num_queries": 60,  # chunk_size
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
    policy = ACTPolicy(policy_config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    sys.argv = saved_argv
    print(f"[ACT] {image_encoder} loaded from {ckpt_path}")
    return policy


# ACTION CONVERSION (from low_level_policy_suturing.py)

def unnormalize_action(action_chunk, mean, std):
    """Unnormalize positions and jaw; rotations stay raw.
    action_chunk: (chunk_size, 20)
    """
    unnormalized = action_chunk * std + mean
    unnormalized[:, 3:9] = action_chunk[:, 3:9]    # PSM1 rotation raw
    unnormalized[:, 13:19] = action_chunk[:, 13:19]  # PSM2 rotation raw
    return unnormalized


def convert_6d_to_quat(rot6d_chunk, current_quat_xyzw):
    """Convert 6D rotation predictions to quaternion targets.

    Uses Gram-Schmidt orthogonalization then composes with current orientation.
    This matches low_level_policy_suturing.py convert_delta_6d_to_taskspace_quat().

    rot6d_chunk: (N, 6)  -- 6D rotation from model
    current_quat_xyzw: (4,)  -- current EE orientation as xyzw quaternion
    Returns: (N, 4) quaternion targets in xyzw format
    """
    c1 = rot6d_chunk[:, 0:3]
    c2 = rot6d_chunk[:, 3:6]
    c1 = normalize(c1, axis=1)
    dot_product = np.sum(c1 * c2, axis=1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product * c1, axis=1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3))  # (N, 3, 3)
    rots = R.from_matrix(r_mat)
    rot_init = R.from_quat(current_quat_xyzw)
    composed = (rot_init * rots).as_quat()  # xyzw
    return composed


def actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2):
    """Convert unnormalized 20D action chunk to dVRK waypoints.

    Follows the exact pattern from low_level_policy_suturing.py lines 490-513:
    1. Subtract first action position (absolute -> delta)
    2. Add current measured pose (delta -> absolute target)
    3. Convert 6D rotation -> quaternion composed with current orientation
    4. Clip jaw to safe range

    action_chunk: (chunk_size, 20)  -- unnormalized actions
    qpos_psm1: (8,)  -- [x,y,z, qx,qy,qz,qw, jaw] current PSM1 state
    qpos_psm2: (8,)  -- [x,y,z, qx,qy,qz,qw, jaw] current PSM2 state

    Returns:
        wp_psm1: (chunk_size, 8)  -- [x,y,z, qx,qy,qz,qw, jaw] targets for PSM1
        wp_psm2: (chunk_size, 8)  -- [x,y,z, qx,qy,qz,qw, jaw] targets for PSM2
    """
    chunk_size = action_chunk.shape[0]
    action = action_chunk.copy()

    # Convert absolute positions to deltas (subtract first timestep)
    action[:, 0:3] = action[:, 0:3] - action[0, 0:3]
    action[:, 10:13] = action[:, 10:13] - action[0, 10:13]

    # PSM1 waypoints
    wp_psm1 = np.zeros((chunk_size, 8))
    wp_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3]
    wp_psm1[:, 3:7] = convert_6d_to_quat(action[:, 3:9], qpos_psm1[3:7])
    wp_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)

    # PSM2 waypoints
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

    Returns dict of {camera_name: (3, 360, 480) float32 tensor} or None if any missing.
    """
    if rt.usb_image_left is None or rt.endo_cam_psm1 is None or rt.endo_cam_psm2 is None:
        return None

    left = process_compressed_image(rt.usb_image_left.data)
    left = np.transpose(left, (2, 0, 1))  # CHW

    lw = process_compressed_image(rt.endo_cam_psm2.data)
    lw = np.transpose(lw, (2, 0, 1))

    rw = process_compressed_image(rt.endo_cam_psm1.data)
    rw = np.transpose(rw, (2, 0, 1))

    # Stack: [left, left_wrist, right_wrist] matching camera_names order
    image_data = np.stack([left, lw, rw], axis=0)  # (3, 3, 360, 480)
    image_tensor = torch.from_numpy(image_data / 255.0).float().cuda().unsqueeze(0)
    return image_tensor


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


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Chained ACT inference on real dVRK")
    parser.add_argument('--ckpt_np', type=str, default=None, help='Needle pickup checkpoint')
    parser.add_argument('--ckpt_nt', type=str, default=None, help='Needle throw checkpoint')
    parser.add_argument('--ckpt_kt', type=str, default=None, help='Knot tying checkpoint')
    parser.add_argument('--subtasks', nargs='+', default=None,
                        choices=['needle_pickup', 'needle_throw', 'knot_tying'],
                        help='Which subtasks to run (default: all with checkpoints)')
    parser.add_argument('--np_steps', type=int, default=300, help='Steps for needle pickup')
    parser.add_argument('--nt_steps', type=int, default=600, help='Steps for needle throw')
    parser.add_argument('--kt_steps', type=int, default=320, help='Steps for knot tying')
    parser.add_argument('--action_horizon', type=int, default=20,
                        help='How many steps of the 60-step chunk to execute before re-querying')
    parser.add_argument('--sleep_rate', type=float, default=0.1,
                        help='Seconds between action steps (default: 0.1 = 10Hz)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Load models and print actions without ROS/robot')
    parser.add_argument('--image_encoder', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b3'],
                        help='Backbone (resnet18=v1, efficientnet_b3=v2)')
    parser.add_argument('--kl_weight', type=int, default=1,
                        help='KL weight (1=v1, 10=v2)')
    args = parser.parse_args()

    # Determine subtask sequence
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
        print("ERROR: No checkpoints provided. Use --ckpt_np, --ckpt_nt, --ckpt_kt")
        sys.exit(1)

    # Load models
    print("=" * 60)
    print("  CHAINED dVRK INFERENCE")
    print("=" * 60)

    models = {}
    norm_stats = {}
    for subtask_name in subtask_sequence:
        ckpt_path = ckpt_map[subtask_name]
        if ckpt_path is None:
            print(f"ERROR: No checkpoint for {subtask_name}")
            sys.exit(1)

        task_config_name = SUBTASK_CONFIGS[subtask_name]['task_name']
        task_config = TASK_CONFIGS[task_config_name]

        print(f"  Loading {subtask_name} from {ckpt_path}...")
        models[subtask_name] = load_act_model(ckpt_path,
                                               image_encoder=args.image_encoder,
                                               kl_weight=args.kl_weight)
        norm_stats[subtask_name] = {
            'mean': task_config['action_mode'][1]['mean'],
            'std': task_config['action_mode'][1]['std'],
        }

    chain = [(s, steps_map[s]) for s in subtask_sequence]
    total_steps = sum(steps for _, steps in chain)
    print(f"\n  Subtask chain: {' -> '.join(s.upper() for s, _ in chain)}")
    print(f"  Steps: {' + '.join(f'{s}={n}' for s, n in chain)} = {total_steps} total")
    print(f"  Action horizon: {args.action_horizon}")
    print(f"  Sleep rate: {args.sleep_rate}s")
    print(f"  Dry run: {args.dry_run}")

    # Dry run mode
    if args.dry_run:
        print("\n  [DRY RUN] Testing model inference without ROS...")
        dummy_images = torch.randn(1, 3, 3, 360, 480).float().cuda()
        qpos_zero = torch.zeros(1, 20).float().cuda()

        for subtask_name in subtask_sequence:
            policy = models[subtask_name]
            with torch.inference_mode():
                action_chunk = policy(qpos_zero, dummy_images).cpu().numpy().squeeze()

            print(f"\n  {subtask_name}:")
            print(f"    Action chunk shape: {action_chunk.shape}")
            unnorm = unnormalize_action(action_chunk,
                                        norm_stats[subtask_name]['mean'],
                                        norm_stats[subtask_name]['std'])
            print(f"    Unnormalized PSM1 pos range: "
                  f"x=[{unnorm[:,0].min():.4f}, {unnorm[:,0].max():.4f}], "
                  f"y=[{unnorm[:,1].min():.4f}, {unnorm[:,1].max():.4f}], "
                  f"z=[{unnorm[:,2].min():.4f}, {unnorm[:,2].max():.4f}]")
            print(f"    Unnormalized PSM2 pos range: "
                  f"x=[{unnorm[:,10].min():.4f}, {unnorm[:,10].max():.4f}], "
                  f"y=[{unnorm[:,11].min():.4f}, {unnorm[:,11].max():.4f}], "
                  f"z=[{unnorm[:,12].min():.4f}, {unnorm[:,12].max():.4f}]")
            print(f"    PSM1 jaw range: [{unnorm[:,9].min():.3f}, {unnorm[:,9].max():.3f}]")
            print(f"    PSM2 jaw range: [{unnorm[:,19].min():.3f}, {unnorm[:,19].max():.3f}]")

        print("\n  [DRY RUN] All models loaded and inference OK.")
        return

    # Real robot mode
    import rospy
    import crtk
    from rostopics import ros_topics
    from dvrk_scripts.dvrk_control import example_application

    rospy.init_node('chained_dvrk_inference', anonymous=True)
    rt = ros_topics()
    ral = crtk.ral('chained_dvrk_inference')
    psm1_app = example_application(ral, "PSM1", 1)
    psm2_app = example_application(ral, "PSM2", 1)

    print("\n  Waiting for camera and robot data...")
    time.sleep(2.0)

    # Verify data is available
    if rt.usb_image_left is None:
        print("ERROR: No left camera image. Check /jhu_daVinci/left/image_raw/compressed")
        sys.exit(1)
    if rt.psm1_pose is None:
        print("ERROR: No PSM1 pose. Check /PSM1/setpoint_cp")
        sys.exit(1)
    print("  Camera and robot data OK.")

    qpos_zero = torch.zeros(1, 20).float().cuda()
    global_step = 0
    action_log = []

    print("\n  Press Ctrl+C to stop at any time.")
    print("=" * 60)

    try:
        for subtask_idx, (subtask_name, max_steps) in enumerate(chain):
            policy = models[subtask_name]
            mean = norm_stats[subtask_name]['mean']
            std = norm_stats[subtask_name]['std']

            print(f"\n  [{subtask_idx+1}/{len(chain)}] Starting {subtask_name.upper()} "
                  f"({max_steps} steps)")

            subtask_step = 0
            while subtask_step < max_steps:
                # Get camera images
                image_tensor = get_camera_images(rt)
                if image_tensor is None:
                    print("  Warning: missing camera frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Get current robot state
                qpos_psm1, qpos_psm2 = get_robot_state(rt)

                # Run inference
                with torch.inference_mode():
                    action_chunk = policy(qpos_zero, image_tensor).cpu().numpy().squeeze()

                # Unnormalize
                action_chunk = unnormalize_action(action_chunk, mean, std)

                # Convert to dVRK waypoints
                wp_psm1, wp_psm2 = actions_to_waypoints(action_chunk, qpos_psm1, qpos_psm2)

                # Execute action_horizon steps
                steps_to_execute = min(args.action_horizon,
                                       max_steps - subtask_step,
                                       action_chunk.shape[0])

                for j in range(steps_to_execute):
                    ral.spin_and_execute(psm1_app.run_full_pose_goal, wp_psm1[j])
                    ral.spin_and_execute(psm2_app.run_full_pose_goal, wp_psm2[j])
                    time.sleep(args.sleep_rate)

                    action_log.append({
                        'step': global_step,
                        'subtask': subtask_name,
                        'wp_psm1': wp_psm1[j].tolist(),
                        'wp_psm2': wp_psm2[j].tolist(),
                    })

                    subtask_step += 1
                    global_step += 1

                if subtask_step % 50 == 0:
                    print(f"    {subtask_name}: step {subtask_step}/{max_steps} "
                          f"(global: {global_step}/{total_steps})")

            print(f"  {subtask_name.upper()} complete ({subtask_step} steps)")

    except KeyboardInterrupt:
        print(f"\n  Interrupted at global step {global_step}")
        print(f"  Subtask was: {subtask_name}, step {subtask_step}")

    # Save action log
    if action_log:
        import json
        log_path = os.path.expanduser('~/action_log.json')
        with open(log_path, 'w') as f:
            json.dump(action_log, f, indent=2)
        print(f"\n  Action log saved to {log_path} ({len(action_log)} entries)")

    print(f"\n  Done. Total steps executed: {global_step}")


if __name__ == '__main__':
    main()
