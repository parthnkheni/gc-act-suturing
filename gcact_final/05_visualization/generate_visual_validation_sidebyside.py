#!/usr/bin/env python3
# generate_visual_validation_sidebyside.py  -- Paper Figure: Trajectory Plots
# Generates side-by-side comparison figures for the paper. Left panel shows
# real endoscope camera footage. Right panel shows an animated 3D plot of the
# predicted arm trajectory (red) vs the ground truth trajectory (green).
# Produces figures for each subtask (NP, NT, KT) and each model version
# (v2, GC-ACT) so readers can visually assess prediction quality.
"""
Visual Validation (Side-by-Side)  -- Endoscope video + animated 3D trajectory.

Left panel:  Real endoscope footage with frame counter
Right panel: Animated 3D trajectory plot (green=GT, red=predicted) with
             a moving marker showing current position

No camera calibration needed  -- the 3D plot uses actual robot coordinates.

Usage:
    python generate_visual_validation_sidebyside.py \
        --eval_dir ~/offline_eval_results_10t_final \
        --data_dir ~/data \
        --output_dir ~/paper_results/visual_validation_sidebyside

    # Figures only (faster):
    python generate_visual_validation_sidebyside.py \
        --eval_dir ~/offline_eval_results_10t_final \
        --output_dir ~/paper_results/visual_validation_sidebyside
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
from mpl_toolkits.mplot3d import Axes3D
import io


def render_3d_plot_to_image(pred_raw, gt_raw, current_idx, timestamps,
                             subtask, metrics, width=640, height=480):
    """Render an animated 3D trajectory plot as a numpy image.

    Shows trajectory up to current_idx with a moving marker.
    """
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)

    # Two 3D subplots side by side for PSM1 and PSM2
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for ax, arm_name, offset in [(ax1, 'PSM1', 0), (ax2, 'PSM2', 10)]:
        # Full trajectory (faded)
        ax.plot(gt_raw[:, offset]*1000, gt_raw[:, offset+1]*1000, gt_raw[:, offset+2]*1000,
                'g-', linewidth=1, alpha=0.2, label='GT')
        ax.plot(pred_raw[:, offset]*1000, pred_raw[:, offset+1]*1000, pred_raw[:, offset+2]*1000,
                'r-', linewidth=1, alpha=0.2, label='Pred')

        # Trajectory up to now (solid)
        i = current_idx + 1
        ax.plot(gt_raw[:i, offset]*1000, gt_raw[:i, offset+1]*1000, gt_raw[:i, offset+2]*1000,
                'g-', linewidth=2, alpha=0.8)
        ax.plot(pred_raw[:i, offset]*1000, pred_raw[:i, offset+1]*1000, pred_raw[:i, offset+2]*1000,
                'r--', linewidth=2, alpha=0.8)

        # Current position markers
        ax.scatter(*[gt_raw[current_idx, offset+j]*1000 for j in range(3)],
                   c='green', s=80, marker='o', zorder=5, edgecolors='darkgreen')
        ax.scatter(*[pred_raw[current_idx, offset+j]*1000 for j in range(3)],
                   c='red', s=80, marker='o', zorder=5, edgecolors='darkred')

        ax.set_xlabel('X (mm)', fontsize=7, labelpad=1)
        ax.set_ylabel('Y (mm)', fontsize=7, labelpad=1)
        ax.set_zlabel('Z (mm)', fontsize=7, labelpad=1)
        ax.set_title(arm_name, fontsize=10)
        ax.tick_params(labelsize=6)

        # Fixed axis limits for stable animation
        for dim, off in enumerate([0, 1, 2]):
            all_vals = np.concatenate([gt_raw[:, offset+off], pred_raw[:, offset+off]]) * 1000
            margin = max(1.0, (all_vals.max() - all_vals.min()) * 0.1)
            lo, hi = all_vals.min() - margin, all_vals.max() + margin
            if dim == 0:
                ax.set_xlim(lo, hi)
            elif dim == 1:
                ax.set_ylim(lo, hi)
            else:
                ax.set_zlim(lo, hi)

        if offset == 0:
            ax.legend(fontsize=7, loc='upper left')

    # Position error for current frame
    pos_err = np.linalg.norm(pred_raw[current_idx, 0:3] - gt_raw[current_idx, 0:3]) * 1000
    t = timestamps[current_idx]
    pct = int(100 * current_idx / max(len(timestamps)-1, 1))

    fig.suptitle(f"{subtask.replace('_', ' ').title()}  -- Frame {t} ({pct}%)\n"
                 f"Current error: {pos_err:.2f}mm | "
                 f"Mean: {metrics['pos_l2_mean_mm']:.2f}mm",
                 fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=100)
    buf.seek(0)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = img.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    plt.close(fig)

    # RGBA -> BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img_bgr = cv2.resize(img_bgr, (width, height))
    return img_bgr


def generate_sidebyside_video(episode_path, pred_raw, gt_raw, timestamps,
                               output_path, subtask, tissue, episode_id, metrics,
                               frame_step=3):
    """Generate side-by-side MP4: endoscope video (left) + 3D trajectory (right).

    frame_step: render every Nth frame for speed (3D plotting is slow).
    """
    left_dir = os.path.join(episode_path, 'left_img_dir')
    if not os.path.isdir(left_dir):
        print(f"  No images at {left_dir}")
        return False

    fps = 10
    canvas_w, canvas_h = 1280, 480  # 640+640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    T = len(timestamps)
    frames_written = 0

    for idx in range(0, T, frame_step):
        t = timestamps[idx]
        fname = f"frame{t:06d}_left.jpg"
        img_path = os.path.join(left_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Resize endoscope frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        # Add info overlay on endoscope frame
        cv2.rectangle(frame, (10, 10), (250, 50), (0, 0, 0), -1)
        pct = int(100 * idx / max(T-1, 1))
        cv2.putText(frame, f"Endoscope | Frame {t} ({pct}%)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Render 3D trajectory plot
        plot_img = render_3d_plot_to_image(pred_raw, gt_raw, idx, timestamps,
                                            subtask, metrics, width=640, height=480)

        # Stitch side by side
        canvas = np.hstack([frame, plot_img])
        writer.write(canvas)
        frames_written += 1

        if frames_written % 50 == 0:
            print(f"       {frames_written}/{T//frame_step} frames rendered...")

    writer.release()
    return frames_written > 0


def generate_sidebyside_figure(episode_path, pred_raw, gt_raw, timestamps,
                                output_path, subtask, tissue, episode_id, metrics):
    """Generate a static figure with 4 key moments shown side by side."""
    left_dir = os.path.join(episode_path, 'left_img_dir')
    T = len(timestamps)
    key_indices = [0, T//3, 2*T//3, T-1]

    fig = plt.figure(figsize=(24, 14))

    fig.suptitle(f"{subtask.replace('_', ' ').title()}  -- {tissue}  -- {episode_id}\n"
                 f"Pos L2: {metrics['pos_l2_mean_mm']:.2f}mm | "
                 f"Rot: {metrics['rot_err_mean_deg']:.1f}deg | "
                 f"Jaw: {metrics['jaw_acc_mean_pct']:.0f}%",
                 fontsize=14, fontweight='bold')

    for col, ki in enumerate(key_indices):
        t = timestamps[ki]
        pct = int(100 * ki / max(T-1, 1))

        # Top row: endoscope frame
        ax_img = fig.add_subplot(3, 4, col + 1)
        fname = f"frame{t:06d}_left.jpg"
        img_path = os.path.join(left_dir, fname)
        frame = cv2.imread(img_path)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax_img.imshow(frame_rgb)
        ax_img.set_title(f"Frame {t} ({pct}%)", fontsize=10)
        ax_img.axis('off')

        # Middle row: PSM1 3D trajectory up to this point
        ax_3d = fig.add_subplot(3, 4, col + 5, projection='3d')
        i = ki + 1
        ax_3d.plot(gt_raw[:, 0]*1000, gt_raw[:, 1]*1000, gt_raw[:, 2]*1000,
                   'g-', linewidth=1, alpha=0.2)
        ax_3d.plot(gt_raw[:i, 0]*1000, gt_raw[:i, 1]*1000, gt_raw[:i, 2]*1000,
                   'g-', linewidth=2, alpha=0.8, label='GT')
        ax_3d.plot(pred_raw[:, 0]*1000, pred_raw[:, 1]*1000, pred_raw[:, 2]*1000,
                   'r-', linewidth=1, alpha=0.2)
        ax_3d.plot(pred_raw[:i, 0]*1000, pred_raw[:i, 1]*1000, pred_raw[:i, 2]*1000,
                   'r--', linewidth=2, alpha=0.8, label='Pred')
        ax_3d.scatter(*[gt_raw[ki, j]*1000 for j in range(3)],
                      c='green', s=60, zorder=5)
        ax_3d.scatter(*[pred_raw[ki, j]*1000 for j in range(3)],
                      c='red', s=60, zorder=5)
        ax_3d.set_xlabel('X', fontsize=7)
        ax_3d.set_ylabel('Y', fontsize=7)
        ax_3d.set_zlabel('Z', fontsize=7)
        ax_3d.set_title(f'PSM1 @ {pct}%', fontsize=9)
        ax_3d.tick_params(labelsize=6)
        if col == 0:
            ax_3d.legend(fontsize=7)

        # Set consistent axis limits
        for dim in range(3):
            all_vals = np.concatenate([gt_raw[:, dim], pred_raw[:, dim]]) * 1000
            margin = max(1.0, (all_vals.max() - all_vals.min()) * 0.1)
            lims = (all_vals.min() - margin, all_vals.max() + margin)
            if dim == 0: ax_3d.set_xlim(lims)
            elif dim == 1: ax_3d.set_ylim(lims)
            else: ax_3d.set_zlim(lims)

    # Bottom row: error plots spanning full width
    ax_err = fig.add_subplot(3, 2, 5)
    pos_err_psm1 = np.linalg.norm(pred_raw[:, 0:3] - gt_raw[:, 0:3], axis=1) * 1000
    pos_err_psm2 = np.linalg.norm(pred_raw[:, 10:13] - gt_raw[:, 10:13], axis=1) * 1000
    ax_err.plot(timestamps, pos_err_psm1, 'r-', alpha=0.8, label='PSM1')
    ax_err.plot(timestamps, pos_err_psm2, 'b-', alpha=0.8, label='PSM2')
    ax_err.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='2mm threshold')
    for ki in key_indices:
        ax_err.axvline(x=timestamps[ki], color='gray', linestyle=':', alpha=0.4)
    ax_err.set_xlabel('Frame')
    ax_err.set_ylabel('Position Error (mm)')
    ax_err.set_title('Position Error Over Time')
    ax_err.legend(fontsize=8)

    ax_jaw = fig.add_subplot(3, 2, 6)
    ax_jaw.plot(timestamps, gt_raw[:, 9], 'g-', linewidth=2, label='GT PSM1', alpha=0.8)
    ax_jaw.plot(timestamps, pred_raw[:, 9], 'r--', linewidth=2, label='Pred PSM1', alpha=0.8)
    ax_jaw.plot(timestamps, gt_raw[:, 19], color='limegreen', linestyle='-',
                linewidth=1, label='GT PSM2', alpha=0.6)
    ax_jaw.plot(timestamps, pred_raw[:, 19], color='salmon', linestyle='--',
                linewidth=1, label='Pred PSM2', alpha=0.6)
    ax_jaw.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    for ki in key_indices:
        ax_jaw.axvline(x=timestamps[ki], color='gray', linestyle=':', alpha=0.4)
    ax_jaw.set_xlabel('Frame')
    ax_jaw.set_ylabel('Jaw Angle (rad)')
    ax_jaw.set_title('Jaw Open/Close')
    ax_jaw.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side visual validation: endoscope + 3D trajectory")
    parser.add_argument('--eval_dir', type=str,
                        default=os.path.expanduser('~/offline_eval_results_10t_final'),
                        help='Directory with offline eval results')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/data'),
                        help='Root data directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/paper_results/visual_validation_sidebyside'),
                        help='Output directory')
    parser.add_argument('--max_episodes', type=int, default=5,
                        help='Max episodes to visualize')
    parser.add_argument('--video', action='store_true',
                        help='Generate MP4 videos (much slower due to 3D rendering)')
    parser.add_argument('--frame_step', type=int, default=3,
                        help='Render every Nth frame in videos (default: 3 for speed)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npz_path = os.path.join(args.eval_dir, 'results_full.npz')
    csv_path = os.path.join(args.eval_dir, 'results.csv')

    if not os.path.exists(npz_path) or not os.path.exists(csv_path):
        print(f"ERROR: Eval results not found at {args.eval_dir}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("  Visual Validation (Side-by-Side)")
    print("  Endoscope Video + 3D Trajectory")
    print("=" * 60)
    print(f"  Eval dir:    {args.eval_dir}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Episodes:    {len(df)}")
    print(f"  Video:       {'yes (frame_step=' + str(args.frame_step) + ')' if args.video else 'no (figures only)'}")
    print()

    folder_map = {
        'needle_pickup': '1_needle_pickup',
        'needle_throw': '2_needle_throw',
        'knot_tying': '3_knot_tying',
    }

    # Track per-subtask index for npz key lookup
    subtask_counters = {}
    global_idx = 0

    for idx, row in df.iterrows():
        subtask = row['subtask']
        tissue = row['tissue']
        episode_id = row['episode_id']

        # Per-subtask counter (npz keys are like needle_throw_0, needle_throw_1, ...)
        if subtask not in subtask_counters:
            subtask_counters[subtask] = 0
        sub_idx = subtask_counters[subtask]
        subtask_counters[subtask] += 1

        episode_path = os.path.join(args.data_dir, tissue,
                                     folder_map[subtask], episode_id)

        if not os.path.isdir(episode_path):
            print(f"  [{global_idx}] SKIP  -- episode not found")
            global_idx += 1
            continue

        prefix = f"{subtask}_{sub_idx}"
        pred_key = f"{prefix}_pred_raw"
        gt_key = f"{prefix}_gt_raw"
        ts_key = f"{prefix}_timestamps"

        if pred_key not in data or gt_key not in data:
            print(f"  [{global_idx}] SKIP  -- no trajectory data (key={pred_key})")
            global_idx += 1
            continue

        pred_raw = data[pred_key]
        gt_raw = data[gt_key]
        timestamps = data[ts_key].astype(int)

        metrics = {
            'pos_l2_mean_mm': row['pos_l2_mean_mm'],
            'rot_err_mean_deg': row['rot_err_mean_deg'],
            'jaw_acc_mean_pct': row['jaw_acc_mean_pct'],
        }

        print(f"  [{global_idx}] {subtask} | {tissue}/{episode_id} "
              f"({len(timestamps)} frames, pos={metrics['pos_l2_mean_mm']:.2f}mm)")

        # Static figure
        fig_path = os.path.join(args.output_dir,
                                 f"sidebyside_{global_idx:02d}_{tissue}_{subtask}.png")
        ok = generate_sidebyside_figure(
            episode_path, pred_raw, gt_raw, timestamps,
            fig_path, subtask, tissue, episode_id, metrics)
        if ok:
            print(f"       Figure: {fig_path}")

        # Video
        if args.video:
            vid_path = os.path.join(args.output_dir,
                                     f"sidebyside_{global_idx:02d}_{tissue}_{subtask}.mp4")
            print(f"       Rendering video (this is slow due to 3D plotting)...")
            ok = generate_sidebyside_video(
                episode_path, pred_raw, gt_raw, timestamps,
                vid_path, subtask, tissue, episode_id, metrics,
                frame_step=args.frame_step)
            if ok:
                print(f"       Video:  {vid_path}")

        global_idx += 1

    print()
    print(f"  Done. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
