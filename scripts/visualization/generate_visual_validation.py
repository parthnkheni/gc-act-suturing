#!/usr/bin/env python3
"""
Visual Validation  -- Overlay predicted vs GT trajectories on real endoscope frames.

Generates:
1. Video (MP4) with predicted (red) and GT (green) EE positions drawn on endoscope frames
2. Side-by-side comparison frames (left endoscope + trajectory plot)
3. Summary montage of key frames

Uses pre-computed offline eval results (no GPU needed).

Usage:
    python generate_visual_validation.py \
        --eval_dir ~/offline_eval_results_10t_final \
        --data_dir ~/data \
        --output_dir ~/paper_results/visual_validation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque


def project_3d_to_2d(pos_3d, cam_matrix=None):
    """Simple projection of 3D EE position to 2D image coordinates.

    Since we don't have exact camera calibration, we use an approximate
    affine mapping learned from the data range to image coordinates.
    The endoscope roughly looks down at the workspace.

    pos_3d: (3,) array [x, y, z] in dVRK task frame (meters)
    Returns: (2,) array [u, v] in image pixels (640x480)
    """
    # dVRK workspace ranges (from norm stats):
    # PSM1 x: ~0.015-0.040, y: ~-0.025-0.005, z: ~0.025-0.055
    # Map to image: x->u (horizontal), y->v (vertical, inverted)
    # These are approximate  -- good enough for visualization
    x, y, z = pos_3d

    # Linear mapping: workspace range -> image pixel range
    # x: [0.00, 0.06] -> [100, 540]  (horizontal)
    # y: [-0.03, 0.02] -> [380, 100]  (vertical, inverted)
    u = int(100 + (x - 0.00) / 0.06 * 440)
    v = int(380 + (y - (-0.03)) / 0.05 * (-280))

    # Clamp to image bounds
    u = max(10, min(629, u))
    v = max(10, min(469, v))
    return np.array([u, v])


def draw_trajectory_on_frame(frame, pred_history, gt_history, pred_current, gt_current,
                              frame_idx, total_frames, metrics_text=""):
    """Draw predicted and GT trajectories on a single frame.

    pred_history: list of (u, v) for past predicted positions
    gt_history: list of (u, v) for past GT positions
    pred_current: (u, v) current predicted position
    gt_current: (u, v) current GT position
    """
    vis = frame.copy()

    # Draw trajectory trails (fading)
    for i in range(1, len(gt_history)):
        alpha = 0.3 + 0.7 * (i / len(gt_history))
        thickness = max(1, int(2 * alpha))
        # GT trail (green)
        pt1 = tuple(gt_history[i-1].astype(int))
        pt2 = tuple(gt_history[i].astype(int))
        cv2.line(vis, pt1, pt2, (0, int(255*alpha), 0), thickness)

    for i in range(1, len(pred_history)):
        alpha = 0.3 + 0.7 * (i / len(pred_history))
        thickness = max(1, int(2 * alpha))
        # Pred trail (red)
        pt1 = tuple(pred_history[i-1].astype(int))
        pt2 = tuple(pred_history[i].astype(int))
        cv2.line(vis, pt1, pt2, (0, 0, int(255*alpha)), thickness)

    # Draw current positions
    if gt_current is not None:
        cv2.circle(vis, tuple(gt_current.astype(int)), 6, (0, 255, 0), -1)  # Green = GT
        cv2.circle(vis, tuple(gt_current.astype(int)), 8, (0, 200, 0), 2)
    if pred_current is not None:
        cv2.circle(vis, tuple(pred_current.astype(int)), 6, (0, 0, 255), -1)  # Red = Pred
        cv2.circle(vis, tuple(pred_current.astype(int)), 8, (0, 0, 200), 2)

    # Legend
    cv2.rectangle(vis, (10, 10), (200, 70), (0, 0, 0), -1)
    cv2.putText(vis, "GT (ground truth)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "Predicted (ACT)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Frame counter
    cv2.rectangle(vis, (430, 10), (630, 35), (0, 0, 0), -1)
    cv2.putText(vis, f"Frame {frame_idx}/{total_frames}", (435, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Metrics text
    if metrics_text:
        cv2.rectangle(vis, (10, 440), (630, 475), (0, 0, 0), -1)
        cv2.putText(vis, metrics_text, (15, 462),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


def generate_episode_video(episode_path, pred_raw, gt_raw, timestamps,
                            output_path, subtask, tissue, episode_id, metrics):
    """Generate an MP4 video with trajectory overlay for one episode."""

    # Check if images exist
    left_dir = os.path.join(episode_path, 'left_img_dir')
    if not os.path.isdir(left_dir):
        print(f"  No images at {left_dir}")
        return False

    # Set up video writer
    fps = 15  # Half real-time for visibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Trail length (how many past points to show)
    trail_len = 60

    pred_trail_psm1 = deque(maxlen=trail_len)
    gt_trail_psm1 = deque(maxlen=trail_len)
    pred_trail_psm2 = deque(maxlen=trail_len)
    gt_trail_psm2 = deque(maxlen=trail_len)

    writer = None
    frames_written = 0

    metrics_text = (f"{subtask} | {tissue} | "
                    f"Pos: {metrics['pos_l2_mean_mm']:.2f}mm | "
                    f"Rot: {metrics['rot_err_mean_deg']:.1f}deg | "
                    f"Jaw: {metrics['jaw_acc_mean_pct']:.0f}%")

    for idx, t in enumerate(timestamps):
        # Load frame
        fname = f"frame{t:06d}_left.jpg"
        img_path = os.path.join(left_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        if writer is None:
            h, w = frame.shape[:2]
            # Create side-by-side canvas: image (640) + trajectory plot (640)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Project 3D positions to 2D
        pred_psm1_2d = project_3d_to_2d(pred_raw[idx, 0:3])
        gt_psm1_2d = project_3d_to_2d(gt_raw[idx, 0:3])
        pred_psm2_2d = project_3d_to_2d(pred_raw[idx, 10:13])
        gt_psm2_2d = project_3d_to_2d(gt_raw[idx, 10:13])

        pred_trail_psm1.append(pred_psm1_2d)
        gt_trail_psm1.append(gt_psm1_2d)
        pred_trail_psm2.append(pred_psm2_2d)
        gt_trail_psm2.append(gt_psm2_2d)

        # Draw PSM1 trajectory (primary arm for most tasks)
        vis = draw_trajectory_on_frame(
            frame,
            list(pred_trail_psm1), list(gt_trail_psm1),
            pred_psm1_2d, gt_psm1_2d,
            t, len(timestamps),
            metrics_text=metrics_text,
        )

        # Also draw PSM2 with different markers
        for i in range(1, len(gt_trail_psm2)):
            alpha = 0.3 + 0.7 * (i / len(gt_trail_psm2))
            pt1 = tuple(list(gt_trail_psm2)[i-1].astype(int))
            pt2 = tuple(list(gt_trail_psm2)[i].astype(int))
            cv2.line(vis, pt1, pt2, (0, int(180*alpha), 0), 1)
        for i in range(1, len(pred_trail_psm2)):
            alpha = 0.3 + 0.7 * (i / len(pred_trail_psm2))
            pt1 = tuple(list(pred_trail_psm2)[i-1].astype(int))
            pt2 = tuple(list(pred_trail_psm2)[i].astype(int))
            cv2.line(vis, pt1, pt2, (0, 0, int(180*alpha)), 1)

        # PSM2 current positions (smaller markers)
        cv2.circle(vis, tuple(gt_psm2_2d.astype(int)), 4, (0, 200, 0), -1)
        cv2.circle(vis, tuple(pred_psm2_2d.astype(int)), 4, (0, 0, 200), -1)

        writer.write(vis)
        frames_written += 1

    if writer is not None:
        writer.release()

    return frames_written > 0


def generate_side_by_side_figure(episode_path, pred_raw, gt_raw, timestamps,
                                  output_path, subtask, tissue, episode_id, metrics):
    """Generate a publication-quality figure with key frames + trajectory."""

    left_dir = os.path.join(episode_path, 'left_img_dir')
    T = len(timestamps)

    # Pick 4 key frames: start, 1/3, 2/3, end
    key_indices = [0, T//3, 2*T//3, T-1]

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)

    fig.suptitle(f"{subtask.replace('_', ' ').title()}  -- {tissue}\n"
                 f"Pos L2: {metrics['pos_l2_mean_mm']:.2f}mm | "
                 f"Rot: {metrics['rot_err_mean_deg']:.1f}deg | "
                 f"Jaw: {metrics['jaw_acc_mean_pct']:.0f}%",
                 fontsize=14, fontweight='bold')

    # Row 0: Key endoscope frames with trajectory overlay
    for col, ki in enumerate(key_indices):
        t = timestamps[ki]
        fname = f"frame{t:06d}_left.jpg"
        img_path = os.path.join(left_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw trajectory up to this point on the frame
        overlay = frame_rgb.copy()
        for i in range(max(0, ki-30), ki+1):
            pred_pt = project_3d_to_2d(pred_raw[i, 0:3])
            gt_pt = project_3d_to_2d(gt_raw[i, 0:3])
            cv2.circle(overlay, tuple(gt_pt.astype(int)), 2, (0, 255, 0), -1)
            cv2.circle(overlay, tuple(pred_pt.astype(int)), 2, (255, 0, 0), -1)

        # Current position (larger)
        pred_now = project_3d_to_2d(pred_raw[ki, 0:3])
        gt_now = project_3d_to_2d(gt_raw[ki, 0:3])
        cv2.circle(overlay, tuple(gt_now.astype(int)), 6, (0, 255, 0), -1)
        cv2.circle(overlay, tuple(pred_now.astype(int)), 6, (255, 0, 0), -1)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(overlay)
        pct = int(100 * ki / max(T-1, 1))
        ax.set_title(f"Frame {t} ({pct}%)", fontsize=10)
        ax.axis('off')

    # Row 1: 3D trajectory comparison (PSM1 and PSM2)
    for arm_idx, (arm_name, offset) in enumerate([('PSM1 (right)', 0), ('PSM2 (left)', 10)]):
        ax = fig.add_subplot(gs[1, arm_idx*2:arm_idx*2+2], projection='3d')
        ax.plot(gt_raw[:, offset]*1000, gt_raw[:, offset+1]*1000, gt_raw[:, offset+2]*1000,
                'g-', linewidth=2, alpha=0.8, label='Ground Truth')
        ax.plot(pred_raw[:, offset]*1000, pred_raw[:, offset+1]*1000, pred_raw[:, offset+2]*1000,
                'r--', linewidth=2, alpha=0.8, label='Predicted')
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.set_zlabel('Z (mm)', fontsize=8)
        ax.set_title(arm_name, fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=7)

    # Row 2: Error over time + jaw
    # Position error
    ax_err = fig.add_subplot(gs[2, 0:2])
    pos_err_psm1 = np.linalg.norm(pred_raw[:, 0:3] - gt_raw[:, 0:3], axis=1) * 1000
    pos_err_psm2 = np.linalg.norm(pred_raw[:, 10:13] - gt_raw[:, 10:13], axis=1) * 1000
    ax_err.plot(timestamps, pos_err_psm1, 'r-', alpha=0.8, label='PSM1')
    ax_err.plot(timestamps, pos_err_psm2, 'b-', alpha=0.8, label='PSM2')
    ax_err.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='2mm threshold')
    ax_err.set_xlabel('Frame')
    ax_err.set_ylabel('Position Error (mm)')
    ax_err.set_title('Position Error Over Time')
    ax_err.legend(fontsize=8)
    # Mark key frames
    for ki in key_indices:
        ax_err.axvline(x=timestamps[ki], color='gray', linestyle=':', alpha=0.3)

    # Jaw comparison
    ax_jaw = fig.add_subplot(gs[2, 2:4])
    ax_jaw.plot(timestamps, gt_raw[:, 9], 'g-', label='GT PSM1', alpha=0.8)
    ax_jaw.plot(timestamps, pred_raw[:, 9], 'r--', label='Pred PSM1', alpha=0.8)
    ax_jaw.plot(timestamps, gt_raw[:, 19], 'g-', label='GT PSM2', alpha=0.5, linewidth=1)
    ax_jaw.plot(timestamps, pred_raw[:, 19], 'r--', label='Pred PSM2', alpha=0.5, linewidth=1)
    ax_jaw.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax_jaw.set_xlabel('Frame')
    ax_jaw.set_ylabel('Jaw Angle (rad)')
    ax_jaw.set_title('Jaw Open/Close')
    ax_jaw.legend(fontsize=8)
    for ki in key_indices:
        ax_jaw.axvline(x=timestamps[ki], color='gray', linestyle=':', alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate visual validation of ACT models")
    parser.add_argument('--eval_dir', type=str,
                        default=os.path.expanduser('~/offline_eval_results_10t_final'),
                        help='Directory with offline eval results')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/data'),
                        help='Root data directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/paper_results/visual_validation'),
                        help='Output directory')
    parser.add_argument('--max_episodes', type=int, default=5,
                        help='Max episodes to visualize')
    parser.add_argument('--video', action='store_true',
                        help='Also generate MP4 videos (slower)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load eval results
    npz_path = os.path.join(args.eval_dir, 'results_full.npz')
    csv_path = os.path.join(args.eval_dir, 'results.csv')

    if not os.path.exists(npz_path) or not os.path.exists(csv_path):
        print(f"ERROR: Eval results not found at {args.eval_dir}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("  Visual Validation  -- ACT Trajectory Overlay")
    print("=" * 60)
    print(f"  Eval dir:   {args.eval_dir}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Episodes:   {len(df)}")
    print(f"  Video:      {'yes' if args.video else 'no (figures only)'}")
    print()

    for idx, row in df.iterrows():
        if idx >= args.max_episodes:
            break

        subtask = row['subtask']
        tissue = row['tissue']
        episode_id = row['episode_id']

        # Map subtask to folder
        folder_map = {
            'needle_pickup': '1_needle_pickup',
            'needle_throw': '2_needle_throw',
            'knot_tying': '3_knot_tying',
        }
        episode_path = os.path.join(args.data_dir, tissue,
                                     folder_map[subtask], episode_id)

        if not os.path.isdir(episode_path):
            print(f"  [{idx}] SKIP  -- episode not found: {episode_path}")
            continue

        # Get trajectory data
        prefix = f"{subtask}_{idx}"
        pred_key = f"{prefix}_pred_raw"
        gt_key = f"{prefix}_gt_raw"
        ts_key = f"{prefix}_timestamps"

        if pred_key not in data or gt_key not in data:
            print(f"  [{idx}] SKIP  -- no trajectory data for {prefix}")
            continue

        pred_raw = data[pred_key]
        gt_raw = data[gt_key]
        timestamps = data[ts_key].astype(int)

        metrics = {
            'pos_l2_mean_mm': row['pos_l2_mean_mm'],
            'rot_err_mean_deg': row['rot_err_mean_deg'],
            'jaw_acc_mean_pct': row['jaw_acc_mean_pct'],
        }

        print(f"  [{idx}] {subtask} | {tissue}/{episode_id} "
              f"({len(timestamps)} frames, pos={metrics['pos_l2_mean_mm']:.2f}mm)")

        # Generate side-by-side figure
        fig_path = os.path.join(args.output_dir,
                                 f"validation_{idx:02d}_{tissue}_{subtask}.png")
        ok = generate_side_by_side_figure(
            episode_path, pred_raw, gt_raw, timestamps,
            fig_path, subtask, tissue, episode_id, metrics)
        if ok:
            print(f"       Figure: {fig_path}")

        # Generate video (optional)
        if args.video:
            vid_path = os.path.join(args.output_dir,
                                     f"validation_{idx:02d}_{tissue}_{subtask}.mp4")
            ok = generate_episode_video(
                episode_path, pred_raw, gt_raw, timestamps,
                vid_path, subtask, tissue, episode_id, metrics)
            if ok:
                print(f"       Video:  {vid_path}")

    print()
    print(f"  Done. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
