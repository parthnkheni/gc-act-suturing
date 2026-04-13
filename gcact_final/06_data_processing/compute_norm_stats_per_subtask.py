# compute_norm_stats_per_subtask.py  -- Dataset Normalization Statistics
# Before training, we need to know the average and spread of all the robot
# measurements (positions, rotations, gripper angles) so we can normalize
# them to a standard range. This script reads all 1,890 episodes across all
# 10 tissues and computes the mean and standard deviation for each of the 20
# action dimensions, separately for each subtask. These statistics are then
# hardcoded into constants_dvrk.py and used during both training and inference.
"""
Compute per-subtask normalization stats (20D, 6D rotation representation)
for all 10 tissues. Matches the quat_to_axis_angle_action() in generic_dataset.py
which actually converts to 6D rotation (first 2 cols of rotation matrix).

Output: mean and std arrays (20D) for each subtask.
"""
import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

DATA_DIR = os.getenv("PATH_TO_DATASET", "/home/exouser/data")

SUBTASKS = {
    "needle_pickup": ["1_"],
    "needle_throw": ["2_"],
    "knot_tying": ["3_"],
}

TISSUE_IDS = list(range(1, 11))  # all 10 tissues

SP_PSM1 = [
    "psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
    "psm1_sp.orientation.x", "psm1_sp.orientation.y",
    "psm1_sp.orientation.z", "psm1_sp.orientation.w",
    "psm1_jaw_sp",
]
SP_PSM2 = [
    "psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
    "psm2_sp.orientation.x", "psm2_sp.orientation.y",
    "psm2_sp.orientation.z", "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]


def quat_to_6d_action(action_8d):
    """
    Convert (N, 8) action [x,y,z, qx,qy,qz,qw, jaw] to (N, 10)
    [x,y,z, r11,r12,r13,r21,r22,r23, jaw] using 6D rotation representation.
    Matches generic_dataset.py quat_to_axis_angle_action().
    """
    quat = action_8d[:, 3:7]  # xyzw convention
    rot_mat = R.from_quat(quat).as_matrix()  # (N, 3, 3)
    # First two columns of rotation matrix, flattened
    rot_6d = rot_mat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)

    out = np.zeros((action_8d.shape[0], 10))
    out[:, 0:3] = action_8d[:, 0:3]   # position
    out[:, 3:9] = rot_6d               # 6D rotation
    out[:, 9] = action_8d[:, 7]        # jaw
    return out


def collect_actions(tissue_ids, data_dir, phase_prefixes):
    """Collect all action samples for given tissues and phase prefixes."""
    all_actions = []
    episode_count = 0

    for tid in tissue_ids:
        tissue_dir = os.path.join(data_dir, f"tissue_{tid}")
        if not os.path.isdir(tissue_dir):
            continue

        phases = [
            d for d in os.listdir(tissue_dir)
            if os.path.isdir(os.path.join(tissue_dir, d))
            and any(d.startswith(pfx) for pfx in phase_prefixes)
        ]

        for phase in natsorted(phases):
            phase_dir = os.path.join(tissue_dir, phase)
            demos = [
                d for d in os.listdir(phase_dir)
                if os.path.isdir(os.path.join(phase_dir, d)) and d != "Corrections"
            ]
            episode_count += len(demos)

            for demo in demos:
                csv_path = os.path.join(phase_dir, demo, "ee_csv.csv")
                if not os.path.exists(csv_path):
                    continue
                csv = pd.read_csv(csv_path)

                action_psm1 = csv[SP_PSM1].to_numpy()
                action_psm2 = csv[SP_PSM2].to_numpy()

                act_6d_psm1 = quat_to_6d_action(action_psm1)
                act_6d_psm2 = quat_to_6d_action(action_psm2)

                stacked = np.column_stack((act_6d_psm1, act_6d_psm2))  # (T, 20)
                all_actions.append(stacked)

    return all_actions, episode_count


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Tissues: {TISSUE_IDS}\n")

    for subtask_name, prefixes in SUBTASKS.items():
        print(f"=== {subtask_name} (prefixes: {prefixes}) ===")
        actions, n_episodes = collect_actions(TISSUE_IDS, DATA_DIR, prefixes)

        if not actions:
            print(f"  No episodes found!\n")
            continue

        all_data = np.concatenate(actions, axis=0)  # (total_timesteps, 20)
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0).clip(1e-2, 10)

        print(f"  Episodes: {n_episodes}")
        print(f"  Timesteps: {all_data.shape[0]}")
        print(f"  Mean: {np.array2string(mean, separator=', ', max_line_width=200)}")
        print(f"  Std:  {np.array2string(std, separator=', ', max_line_width=200)}")
        print()

        # Save to npz
        out_path = os.path.join(
            os.path.dirname(__file__), f"norm_stats_{subtask_name}_all10.npz"
        )
        np.savez(out_path, mean=mean, std=std)
        print(f"  Saved to {out_path}\n")


if __name__ == "__main__":
    main()
