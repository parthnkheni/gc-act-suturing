#!/usr/bin/env python3
"""
Verify A100 Checkpoints + Compare NORM_STATS
=============================================
Loads each checkpoint to verify it's valid, loads dataset_stats.pkl,
and compares against the NORM_STATS hardcoded in act_orbit_chained.py.

Usage:
    conda run -n orbitsurgical python verify_checkpoints.py
"""

import os
import sys
import pickle
import numpy as np
import torch

# Per-subtask norm stats currently hardcoded in act_orbit_chained.py
NORM_STATS_HARDCODED = {
    'needle_pickup': {
        'mean': np.array([0.021, -0.011, 0.044, 0.223, 0.197, -0.906, -0.176, 0.917, 0.137, 0.085,
                          -0.011, 0.006, 0.048, -0.161, 0.458, -0.780, -0.197, -0.789, -0.415, 0.037]),
        'std':  np.array([0.010, 0.016, 0.036, 0.145, 0.246, 0.095, 0.199, 0.104, 0.244, 0.413,
                          0.010, 0.016, 0.030, 0.182, 0.328, 0.125, 0.215, 0.134, 0.319, 0.338]),
    },
    'needle_throw': {
        'mean': np.array([0.031, -0.008, 0.039, 0.210, 0.246, -0.837, 0.025, 0.841, 0.253, -0.199,
                          -0.001, 0.005, 0.045, -0.165, 0.404, -0.881, -0.165, -0.892, -0.381, 0.101]),
        'std':  np.array([0.010, 0.010, 0.010, 0.182, 0.373, 0.152, 0.224, 0.158, 0.391, 0.381,
                          0.010, 0.010, 0.010, 0.104, 0.139, 0.060, 0.098, 0.064, 0.135, 0.116]),
    },
    'knot_tying': {
        'mean': np.array([0.025, -0.011, 0.031, 0.356, 0.097, -0.882, 0.255, 0.859, 0.213, -0.349,
                          -0.006, 0.009, 0.048, -0.115, 0.520, -0.815, -0.272, -0.803, -0.471, 0.003]),
        'std':  np.array([0.010, 0.010, 0.010, 0.202, 0.187, 0.103, 0.318, 0.157, 0.162, 0.012,
                          0.010, 0.010, 0.010, 0.151, 0.138, 0.097, 0.177, 0.106, 0.131, 0.333]),
    },
}

CHECKPOINT_DIRS = {
    'needle_pickup': os.path.expanduser('~/checkpoints/act_np_all10_kl1'),
    'needle_throw':  os.path.expanduser('~/checkpoints/act_nt_all10_kl1'),
    'knot_tying':    os.path.expanduser('~/checkpoints/act_kt_all10_kl1'),
}


def verify_checkpoint(subtask_name, ckpt_dir):
    """Verify a single checkpoint + dataset_stats.pkl."""
    print(f"\n{'='*70}")
    print(f"  {subtask_name.upper()}")
    print(f"  Dir: {ckpt_dir}")
    print(f"{'='*70}")

    # Check directory exists
    if not os.path.isdir(ckpt_dir):
        print(f"  [MISSING] Checkpoint directory not found!")
        return False

    # Find checkpoint file
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        # Try pattern: policy_best_epoch_*.ckpt
        import glob
        pattern = os.path.join(ckpt_dir, 'policy_best_epoch_*.ckpt')
        matches = glob.glob(pattern)
        if matches:
            ckpt_path = matches[0]
        else:
            print(f"  [MISSING] No checkpoint file found!")
            return False

    # Load checkpoint
    ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"\n  Checkpoint: {os.path.basename(ckpt_path)} ({ckpt_size_mb:.1f} MB)")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"  [OK] Checkpoint loaded successfully")

        # Extract info
        if 'epoch' in ckpt:
            print(f"  Best epoch: {ckpt['epoch']}")
        if 'model_state_dict' in ckpt:
            n_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
            print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
            # Check action dim from final layer
            for key in ckpt['model_state_dict']:
                if 'action_head' in key or 'additional_pos_embed' in key:
                    shape = ckpt['model_state_dict'][key].shape
                    print(f"  Key '{key}': {shape}")
        else:
            print(f"  Keys in checkpoint: {list(ckpt.keys())}")
    except Exception as e:
        print(f"  [ERROR] Failed to load checkpoint: {e}")
        return False

    # Load dataset_stats.pkl
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        if stats is None:
            print(f"\n  dataset_stats.pkl exists but contains None")
            print(f"  (This means normalization was done via constants_dvrk.py task config, not dataset_stats)")
        else:
            print(f"\n  dataset_stats.pkl keys: {list(stats.keys())}")
            if 'action_mean' in stats:
                print(f"  action_mean shape: {stats['action_mean'].shape}")
                print(f"  action_std shape:  {stats['action_std'].shape}")
    else:
        print(f"\n  [MISSING] dataset_stats.pkl not found")

    # Compare with hardcoded NORM_STATS
    print(f"\n  --- Comparison with hardcoded NORM_STATS ---")
    hardcoded = NORM_STATS_HARDCODED[subtask_name]

    if stats is not None and 'action_mean' in stats:
        pkl_mean = stats['action_mean']
        pkl_std = stats['action_std']

        # Position indices (normalized)
        pos_idx = [0, 1, 2, 10, 11, 12]
        # Jaw indices (normalized)
        jaw_idx = [9, 19]
        # Rotation indices (raw, should have similar mean/std as ground truth)
        rot_idx = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]

        print(f"\n  MEAN comparison (pkl vs hardcoded):")
        mean_diff = np.abs(pkl_mean - hardcoded['mean'])
        for i in range(20):
            tag = "pos" if i in pos_idx else ("jaw" if i in jaw_idx else "rot")
            match = "OK" if mean_diff[i] < 0.01 else "DIFF"
            print(f"    [{i:2d}] {tag:3s}: pkl={pkl_mean[i]:+.6f}  hc={hardcoded['mean'][i]:+.6f}  "
                  f"diff={mean_diff[i]:.6f}  [{match}]")

        print(f"\n  STD comparison (pkl vs hardcoded):")
        std_diff = np.abs(pkl_std - hardcoded['std'])
        for i in range(20):
            tag = "pos" if i in pos_idx else ("jaw" if i in jaw_idx else "rot")
            match = "OK" if std_diff[i] < 0.01 else "DIFF"
            print(f"    [{i:2d}] {tag:3s}: pkl={pkl_std[i]:+.6f}  hc={hardcoded['std'][i]:+.6f}  "
                  f"diff={std_diff[i]:.6f}  [{match}]")

        # Summary
        max_mean_diff = mean_diff.max()
        max_std_diff = std_diff.max()
        if max_mean_diff < 0.01 and max_std_diff < 0.01:
            print(f"\n  [MATCH] NORM_STATS match dataset_stats.pkl (max diff: mean={max_mean_diff:.6f}, std={max_std_diff:.6f})")
        else:
            print(f"\n  [MISMATCH] NORM_STATS differ from dataset_stats.pkl!")
            print(f"  Max mean diff: {max_mean_diff:.6f} (at idx {np.argmax(mean_diff)})")
            print(f"  Max std diff:  {max_std_diff:.6f} (at idx {np.argmax(std_diff)})")
            print(f"\n  --> UPDATE act_orbit_chained.py NORM_STATS with these values:")
            print(f"  '{subtask_name}': {{")
            print(f"      'mean': np.array({np.array2string(pkl_mean, separator=', ', max_line_width=120)}),")
            print(f"      'std':  np.array({np.array2string(pkl_std, separator=', ', max_line_width=120)}),")
            print(f"  }},")
        return True

    else:
        print(f"\n  No dataset_stats.pkl with action stats available.")
        print(f"  Check the A100's constants_dvrk.py for the actual mean/std used during training.")
        print(f"  If you copied it to ~/checkpoints/a100_constants_dvrk.py, inspect it manually.")

        # Check if a100_constants exists
        a100_const = os.path.expanduser('~/checkpoints/a100_constants_dvrk.py')
        if os.path.exists(a100_const):
            print(f"\n  Found {a100_const} -- inspect manually for task configs with per-subtask mean/std.")
        return True


def main():
    print("="*70)
    print("  A100 CHECKPOINT VERIFICATION")
    print("="*70)

    all_ok = True
    for subtask_name, ckpt_dir in CHECKPOINT_DIRS.items():
        ok = verify_checkpoint(subtask_name, ckpt_dir)
        if not ok:
            all_ok = False

    print(f"\n{'='*70}")
    if all_ok:
        print("  ALL CHECKPOINTS VERIFIED")
    else:
        print("  SOME CHECKPOINTS MISSING OR INVALID")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
