# generate_suture_replay.py  -- Demo Video: Predicted vs Actual Trajectories
# Creates a video showing a complete suture attempt (needle pickup -> needle
# throw -> knot tying) with the model's predicted arm positions overlaid on
# the real endoscope footage alongside the ground truth positions. Lets you
# visually see how closely the model tracks the human surgeon's movements.
"""
Autonomous Suture Replay Video v2
Shows complete suture (NP -> NT -> KT) with 4 distinct markers:
  - PSM1 predicted (green crosshair) vs GT (cyan circle)
  - PSM2 predicted (magenta crosshair) vs GT (yellow circle)
  - Error lines connecting GT->predicted
Uses calibrated 3D-to-pixel projection from dVRK task frame.
"""
import numpy as np
import cv2
import os
import csv

# Config
DATA_DIR = '/home/exouser/data'
V2_RESULTS = '/home/exouser/offline_eval_results_v2_raw/results_full.npz'
GCACT_RESULTS = '/home/exouser/offline_eval_results_gcact_raw/results_full.npz'
V2_CSV = '/home/exouser/offline_eval_results_v2_raw/results.csv'
GCACT_CSV = '/home/exouser/offline_eval_results_gcact_raw/results.csv'
OUTPUT_PATH = '/home/exouser/paper_results/autonomous_suture_replay.mp4'
FPS = 15
FRAME_W, FRAME_H = 960, 540  # Output video resolution
EPISODE_IDX = 0

# Calibrated projection
# Fit from known arm positions in endoscope images:
# PSM1 at NP frame 50: xyz=(0.025,-0.014,0.036) -> pixel≈(250,240) in 960x480
# PSM2 at NP frame 50: xyz=(-0.019,0.009,0.055) -> pixel≈(620,200) in 960x480
def project_3d_to_pixel(pos_3d, img_w=960, img_h=480):
    x, y, z = pos_3d[0], pos_3d[1], pos_3d[2]
    px = 460 - 8400 * x
    py = 287 - 500 * y - 1500 * z
    # Scale to output resolution
    px = px * (img_w / 960)
    py = py * (img_h / 480)
    return (int(np.clip(px, 5, img_w - 5)), int(np.clip(py, 5, img_h - 5)))


def get_episode_paths(csv_path, subtask):
    episodes = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['subtask'] == subtask:
                episodes.append(r['episode_path'])
    return episodes


def load_frame(episode_path, frame_idx):
    fpath = os.path.join(episode_path, 'left_img_dir', f'frame{frame_idx:06d}_left.jpg')
    return cv2.imread(fpath) if os.path.exists(fpath) else None


def draw_crosshair(frame, pt, color, size=14, thickness=2):
    """Large crosshair for predicted positions."""
    px, py = pt
    cv2.line(frame, (px - size, py), (px + size, py), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (px, py - size), (px, py + size), color, thickness, cv2.LINE_AA)


def draw_circle(frame, pt, color, radius=7, thickness=2):
    """Circle for ground truth positions."""
    cv2.circle(frame, pt, radius, color, thickness, cv2.LINE_AA)


def draw_error_line(frame, gt_pt, pred_pt, color):
    """Thin line from GT to predicted showing error vector."""
    cv2.line(frame, gt_pt, pred_pt, color, 1, cv2.LINE_AA)


def draw_trail(frame, positions, color, max_trail=40):
    n = len(positions)
    start = max(0, n - max_trail)
    for i in range(start, n - 1):
        alpha = (i - start) / max_trail
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, positions[i], positions[i + 1], c, 1, cv2.LINE_AA)


# Color scheme - 4 distinct colors
COL_PSM1_PRED = (0, 255, 0)       # Bright green
COL_PSM1_GT   = (255, 255, 0)     # Cyan (BGR)
COL_PSM2_PRED = (255, 0, 255)     # Magenta
COL_PSM2_GT   = (0, 165, 255)     # Orange (BGR)

# Subtask title colors
SUBTASK_COLORS = {
    'Needle Pickup': (136, 255, 0),
    'Needle Throw': (68, 221, 255),
    'Knot Tying': (102, 136, 255),
}


def draw_info_panel(frame, subtask_name, model_name, frame_num, total_frames,
                    err_psm1_mm, err_psm2_mm):
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = SUBTASK_COLORS.get(subtask_name, (255, 255, 255))
    cv2.putText(frame, subtask_name, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'Model: {model_name}', (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f'{frame_num}/{total_frames}', (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Per-arm errors
    e1_col = (0, 255, 0) if err_psm1_mm < 1.5 else (0, 200, 255) if err_psm1_mm < 2.5 else (0, 0, 255)
    e2_col = (0, 255, 0) if err_psm2_mm < 1.5 else (0, 200, 255) if err_psm2_mm < 2.5 else (0, 0, 255)
    cv2.putText(frame, f'PSM1: {err_psm1_mm:.1f}mm', (w - 280, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, e1_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f'PSM2: {err_psm2_mm:.1f}mm', (w - 280, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, e2_col, 1, cv2.LINE_AA)

    # Progress bar
    bar_w = int((frame_num / total_frames) * w)
    cv2.rectangle(frame, (0, h - 6), (bar_w, h), color, -1)


def draw_legend(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    lx, ly = w - 250, h - 120
    cv2.rectangle(overlay, (lx - 10, ly - 10), (w - 5, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    items = [
        ('PSM1 Predicted', COL_PSM1_PRED, 'cross'),
        ('PSM1 Ground Truth', COL_PSM1_GT, 'circle'),
        ('PSM2 Predicted', COL_PSM2_PRED, 'cross'),
        ('PSM2 Ground Truth', COL_PSM2_GT, 'circle'),
    ]
    for i, (label, color, shape) in enumerate(items):
        y = ly + 5 + i * 22
        if shape == 'cross':
            draw_crosshair(frame, (lx + 8, y), color, size=6, thickness=2)
        else:
            draw_circle(frame, (lx + 8, y), color, radius=5, thickness=2)
        cv2.putText(frame, label, (lx + 22, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)


def make_title_card(text_lines, duration_s):
    """Generate title/transition frames."""
    frames = []
    card = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    y_start = FRAME_H // 2 - len(text_lines) * 25
    for i, (text, size, color, thickness) in enumerate(text_lines):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0][0]
        x = (FRAME_W - tw) // 2
        y = y_start + i * 55
        cv2.putText(card, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
    return [card.copy()] * int(FPS * duration_s)


def main():
    print("Loading evaluation results...")
    v2_data = np.load(V2_RESULTS, allow_pickle=True)
    gcact_data = np.load(GCACT_RESULTS, allow_pickle=True)

    np_episodes = get_episode_paths(V2_CSV, 'needle_pickup')
    nt_episodes = get_episode_paths(GCACT_CSV, 'needle_throw')
    kt_episodes = get_episode_paths(GCACT_CSV, 'knot_tying')

    segments = [
        {
            'name': 'Needle Pickup', 'model': 'ACT v2',
            'pred': v2_data[f'needle_pickup_{EPISODE_IDX}_pred_raw'],
            'gt': v2_data[f'needle_pickup_{EPISODE_IDX}_gt_raw'],
            'timestamps': v2_data[f'needle_pickup_{EPISODE_IDX}_timestamps'],
            'episode_path': np_episodes[EPISODE_IDX],
        },
        {
            'name': 'Needle Throw', 'model': 'GC-ACT',
            'pred': gcact_data[f'needle_throw_{EPISODE_IDX}_pred_raw'],
            'gt': gcact_data[f'needle_throw_{EPISODE_IDX}_gt_raw'],
            'timestamps': gcact_data[f'needle_throw_{EPISODE_IDX}_timestamps'],
            'episode_path': nt_episodes[EPISODE_IDX],
        },
        {
            'name': 'Knot Tying', 'model': 'GC-ACT',
            'pred': gcact_data[f'knot_tying_{EPISODE_IDX}_pred_raw'],
            'gt': gcact_data[f'knot_tying_{EPISODE_IDX}_gt_raw'],
            'timestamps': gcact_data[f'knot_tying_{EPISODE_IDX}_timestamps'],
            'episode_path': kt_episodes[EPISODE_IDX],
        },
    ]

    total_frames = sum(len(s['pred']) for s in segments)
    print(f"Total frames: {total_frames} ({total_frames/FPS:.1f}s at {FPS}fps)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (FRAME_W, FRAME_H))

    # Title card
    for f in make_title_card([
        ('Autonomous Suture Replay', 1.2, (255, 255, 255), 2),
        ('GC-ACT: Gesture-Conditioned Action Chunking Transformer', 0.6, (150, 200, 255), 1),
        ('', 0.1, (0,0,0), 1),
        ('Needle Pickup (v2)  >  Needle Throw (GC-ACT)  >  Knot Tying (GC-ACT)', 0.55, (200, 200, 200), 1),
        (f'Tissue 7  |  {total_frames} frames  |  {FPS}fps', 0.45, (140, 140, 140), 1),
    ], 3.0):
        out.write(f)

    global_frame = 0
    for seg_idx, seg in enumerate(segments):
        pred = seg['pred']
        gt = seg['gt']
        timestamps = seg['timestamps']
        episode_path = seg['episode_path']
        n_frames = len(pred)

        print(f"\n[{seg_idx+1}/3] {seg['name']}  -- {n_frames} frames from {os.path.basename(episode_path)}")

        # Transition card
        if seg_idx > 0:
            color = SUBTASK_COLORS[seg['name']]
            for f in make_title_card([
                (f'Phase {seg_idx+1}: {seg["name"]}', 1.0, color, 2),
                (f'Model: {seg["model"]}', 0.6, (200, 200, 200), 1),
            ], 1.5):
                out.write(f)

        # Trails
        trails = {k: [] for k in ['p1_pred', 'p2_pred', 'p1_gt', 'p2_gt']}

        for i in range(n_frames):
            frame = load_frame(episode_path, timestamps[i])
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # Extract 3D positions
            pred_p1 = pred[i, 0:3]
            pred_p2 = pred[i, 10:13]
            gt_p1 = gt[i, 0:3]
            gt_p2 = gt[i, 10:13]

            # Project to 2D
            pred_p1_2d = project_3d_to_pixel(pred_p1, FRAME_W, FRAME_H)
            pred_p2_2d = project_3d_to_pixel(pred_p2, FRAME_W, FRAME_H)
            gt_p1_2d = project_3d_to_pixel(gt_p1, FRAME_W, FRAME_H)
            gt_p2_2d = project_3d_to_pixel(gt_p2, FRAME_W, FRAME_H)

            # Accumulate trails
            trails['p1_pred'].append(pred_p1_2d)
            trails['p2_pred'].append(pred_p2_2d)
            trails['p1_gt'].append(gt_p1_2d)
            trails['p2_gt'].append(gt_p2_2d)

            # Draw trails (GT dimmer, pred brighter)
            draw_trail(frame, trails['p1_gt'], (128, 128, 0), max_trail=30)
            draw_trail(frame, trails['p2_gt'], (0, 80, 128), max_trail=30)
            draw_trail(frame, trails['p1_pred'], (0, 200, 0), max_trail=30)
            draw_trail(frame, trails['p2_pred'], (200, 0, 200), max_trail=30)

            # Draw error lines (GT -> pred)
            draw_error_line(frame, gt_p1_2d, pred_p1_2d, (255, 255, 255))
            draw_error_line(frame, gt_p2_2d, pred_p2_2d, (255, 255, 255))

            # Draw GT markers (circles, behind)
            draw_circle(frame, gt_p1_2d, COL_PSM1_GT, radius=8, thickness=2)
            draw_circle(frame, gt_p2_2d, COL_PSM2_GT, radius=8, thickness=2)

            # Draw predicted markers (crosshairs, on top)
            draw_crosshair(frame, pred_p1_2d, COL_PSM1_PRED, size=14, thickness=2)
            draw_crosshair(frame, pred_p2_2d, COL_PSM2_PRED, size=14, thickness=2)

            # Add small labels near markers
            cv2.putText(frame, 'P1', (pred_p1_2d[0] + 16, pred_p1_2d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_PSM1_PRED, 1, cv2.LINE_AA)
            cv2.putText(frame, 'P2', (pred_p2_2d[0] + 16, pred_p2_2d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_PSM2_PRED, 1, cv2.LINE_AA)

            # Errors
            err_p1 = np.linalg.norm(pred_p1 - gt_p1) * 1000
            err_p2 = np.linalg.norm(pred_p2 - gt_p2) * 1000

            # Info panel + legend
            draw_info_panel(frame, seg['name'], seg['model'], i + 1, n_frames, err_p1, err_p2)
            draw_legend(frame)

            out.write(frame)
            global_frame += 1

            if i % 100 == 0:
                print(f"  Frame {i}/{n_frames}  -- PSM1: {err_p1:.2f}mm, PSM2: {err_p2:.2f}mm")

    # End card with stats
    stats_lines = [('Suture Complete', 1.2, (0, 255, 0), 2),
                   ('', 0.1, (0,0,0), 1)]
    for seg in segments:
        err1 = np.mean(np.linalg.norm(seg['pred'][:, 0:3] - seg['gt'][:, 0:3], axis=1)) * 1000
        err2 = np.mean(np.linalg.norm(seg['pred'][:, 10:13] - seg['gt'][:, 10:13], axis=1)) * 1000
        stats_lines.append((f'{seg["name"]}:  PSM1 {err1:.2f}mm  |  PSM2 {err2:.2f}mm',
                           0.55, (200, 200, 200), 1))
    stats_lines.append(('', 0.1, (0,0,0), 1))
    stats_lines.append(('Green + = predicted  |  Cyan/Orange O = ground truth  |  White line = error',
                        0.45, (150, 150, 150), 1))

    for f in make_title_card(stats_lines, 4.0):
        out.write(f)

    out.release()
    duration = (global_frame + FPS * 8.5) / FPS
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Duration: {duration:.1f}s, Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f}MB")


if __name__ == '__main__':
    main()
