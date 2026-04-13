"""
ACT Task Chaining Inference Script
===================================
Chains 3 subtask models (needle_pickup -> needle_throw -> knot_tying) in a single
continuous ORBIT-Surgical environment for full suturing demo.

Models: All 3 subtasks use 20D ACT policies (trained on real SutureBot data with
3 cameras: endoscope + PSM1 wrist + PSM2 wrist). Per-subtask denormalization via
NORM_STATS.

Switching: fixed step counts (configurable via CLI), no env reset between subtasks.
Temporal ensembling state is reset at each subtask boundary.

Usage:
    python act_orbit_chained.py --headless --enable_cameras \
      --ckpt_np ~/checkpoints/act_real_np/policy_best.ckpt \
      --ckpt_nt ~/checkpoints/act_real_nt/policy_best.ckpt \
      --ckpt_kt ~/checkpoints/act_real_kt/policy_best.ckpt

Note: Classes and utilities are inlined from act_orbit_agent_v5.py because that
script has module-level side effects (argparse/AppLauncher) that prevent direct import.
"""
import os
os.environ['ENABLE_CAMERAS'] = '1'

import sys
import time
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="ACT Task Chaining: NP -> NT -> KT")
parser.add_argument('--disable_fabric', action='store_true', default=False)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--task', type=str, default='Isaac-Handover-Needle-Dual-PSM-IK-Abs-v0')
parser.add_argument('--temporal_weight', type=float, default=0.01,
                    help='Temporal ensembling weight k: exp(-k*age). Lower=smoother, higher=more responsive.')
parser.add_argument('--motion_scale', type=float, default=None,
                    help='Override MOTION_SCALE (default: 3.0). Higher values amplify model output.')

# Subtask checkpoints (all 20D)
parser.add_argument('--ckpt_np', type=str, required=True,
                    help='Needle pickup checkpoint (20D)')
parser.add_argument('--ckpt_nt', type=str, required=True,
                    help='Needle throw checkpoint (20D)')
parser.add_argument('--ckpt_kt', type=str, required=True,
                    help='Knot tying checkpoint (20D)')

# Subtask step counts
parser.add_argument('--steps_np', type=int, default=300, help='Steps for needle pickup phase (data mean=299)')
parser.add_argument('--steps_nt', type=int, default=600, help='Steps for needle throw phase (data mean=582)')
parser.add_argument('--steps_kt', type=int, default=320, help='Steps for knot tying phase (data mean=309)')

from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli, window_width=1920, window_height=1080)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg
import orbit.surgical.tasks

from scipy.spatial.transform import Rotation as R
import cv2


# COORDINATE CALIBRATION (Centered Actions  -- from v5)
EE_HOME = np.array([0.0, 0.0, -0.098])
MOTION_SCALE = 3.0   # must match generate_sim_dataset.py and act_orbit_agent_v5.py
DELTA_SCALE = 10.0   # scale factor for delta mode (3.0 = data-matched, 10 = amplified for visibility)

# Pre-positioning targets: where the arms should be before model control starts.
# These are in robot base frame. Each robot base is at ±[0.2, 0, 0.15] in world.
# To reach the suture pad at [0, 0, 0.005], arms need to extend inward and down.
PREPOSITION_R1 = np.array([-0.15, 0.0, -0.14])   # R1 reaches toward center
PREPOSITION_R2 = np.array([0.15, 0.0, -0.14])     # R2 reaches toward center
PREPOSITION_QUAT = np.array([1.0, 0.0, 0.0, 0.0]) # neutral orientation (wxyz)
PREPOSITION_STEPS = 80  # steps to settle into position

REAL_QPOS_MEAN = np.array([ 0.215052, -1.113195,  0.202571,  0.108202, -0.890289,  0.547698,
        0.088633, -0.071789, -1.168704,  0.186194, -0.453821, -0.925601,
       -0.031915,  0.037388])
REAL_QPOS_STD = np.array([0.195979, 0.100092, 0.006403, 0.513928, 0.165657, 0.379074,
       0.402738, 0.314738, 0.062025, 0.007033, 0.536599, 0.222125,
       0.446234, 0.339993])
SIM_QPOS_HOME = np.array([0.01, 0.01, 0.07, 0.01, 0.01, 0.01, -0.09,
                           0.01, 0.01, 0.07, 0.01, 0.01, 0.01, -0.09])

# Per-subtask normalization stats (20D: positions 0:3,10:13 and jaw 9,19 are normalized; rotations are raw)
NORM_STATS = {
    'needle_pickup': {
        'mean': np.array([ 0.02117121, -0.01061033,  0.04433738,  0.22251957,  0.1971502 ,
                          -0.90597265, -0.17570475,  0.91686429,  0.13694032,  0.08506552,
                          -0.01144972,  0.00619323,  0.04760892, -0.16096897,  0.45806132,
                          -0.77968459, -0.19713571, -0.78929362, -0.41484009,  0.03692078]),
        'std':  np.array([0.01028275, 0.01613198, 0.03551102, 0.14531227, 0.24641732,
                          0.09483275, 0.19915273, 0.10352518, 0.24363046, 0.41263211,
                          0.01      , 0.01569558, 0.02975965, 0.18217486, 0.32791292,
                          0.12508477, 0.21523964, 0.13445343, 0.31883376, 0.33758943]),
    },
    'needle_throw': {
        'mean': np.array([ 0.03055553, -0.0079745 ,  0.03942896,  0.20950523,  0.2459589 ,
                          -0.83687713,  0.02511587,  0.84121708,  0.25258415, -0.19935053,
                          -0.00144373,  0.0049653 ,  0.04490225, -0.16504303,  0.40365707,
                          -0.88094752, -0.16531958, -0.89206828, -0.38058983,  0.10116849]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.18217966, 0.37271865,
                          0.1521178 , 0.22397943, 0.15792516, 0.39091832, 0.38089521,
                          0.01      , 0.01      , 0.01      , 0.10402391, 0.13901681,
                          0.06005486, 0.0981895 , 0.06387606, 0.13532799, 0.11621434]),
    },
    'knot_tying': {
        'mean': np.array([ 0.0249953 , -0.01066993,  0.03099244,  0.35583106,  0.09714196,
                          -0.88176762,  0.25547242,  0.85854408,  0.2132891 , -0.3488589 ,
                          -0.00550976,  0.00859079,  0.04786375, -0.11541289,  0.52012943,
                          -0.81531284, -0.27246644, -0.80284463, -0.4707253 ,  0.00315498]),
        'std':  np.array([0.01      , 0.01      , 0.01      , 0.20154828, 0.1873807 ,
                          0.10344235, 0.31842194, 0.1565692 , 0.1619791 , 0.01224801,
                          0.01      , 0.01      , 0.01      , 0.15112859, 0.13838071,
                          0.09706259, 0.1765505 , 0.10569934, 0.13144507, 0.33264002]),
    },
}


# SECTION 1: ACT MODEL LOADING (from v5)

class ACTPolicyInference:
    """Single 20D ACT policy for both arms combined."""

    def __init__(self, ckpt_path, device="cuda", temporal_weight=0.01, action_mean=None, action_std=None):
        self.device = device
        self.chunk_size = 60
        self.action_dim = 20
        self.state_dim = 20
        self.camera_names = ['left', 'left_wrist', 'right_wrist']
        self.action_mean = action_mean
        self.action_std = action_std

        self._build_and_load(ckpt_path)

        self.temporal_agg = temporal_weight > 0
        self.temporal_weight = temporal_weight
        self.all_actions = np.zeros((self.chunk_size, self.chunk_size, self.action_dim))
        self.all_actions_filled = np.zeros(self.chunk_size, dtype=bool)
        self.timestep = 0
        self.current_chunk = None

        print(f"[ACT-20D] Model loaded from {ckpt_path}")
        print(f"[ACT-20D] Temporal ensembling: k={self.temporal_weight}"
              f"{' (disabled)' if not self.temporal_agg else ''}")

    def _build_and_load(self, ckpt_path):
        saved_argv = sys.argv
        sys.argv = [
            'act', '--task_name', 'sim_needle_pickup',
            '--ckpt_dir', '/tmp', '--policy_class', 'ACT',
            '--seed', '0', '--num_epochs', '1',
            '--kl_weight', '1', '--chunk_size', '60',
            '--hidden_dim', '512', '--dim_feedforward', '3200',
            '--lr', '1e-5', '--batch_size', '8',
            '--image_encoder', 'resnet18',
            '--policy_level', 'low',
        ]
        sys.path.insert(0, os.path.expanduser('~/src/act'))
        sys.path.insert(0, os.path.expanduser('~/src'))
        from policy import ACTPolicy

        policy_config = {
            "lr": 1e-5, "num_queries": self.chunk_size,
            "action_dim": self.action_dim, "kl_weight": 1,
            "hidden_dim": 512, "dim_feedforward": 3200,
            "lr_backbone": 1e-5, "backbone": "resnet18",
            "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": self.camera_names, "multi_gpu": False,
        }
        self.policy = ACTPolicy(policy_config)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.policy.load_state_dict(state_dict)
        self.policy.cuda()
        self.policy.eval()
        sys.argv = saved_argv

    def reset_temporal_state(self):
        """Reset temporal ensembling state (call at subtask boundaries)."""
        self.all_actions = np.zeros((self.chunk_size, self.chunk_size, self.action_dim))
        self.all_actions_filled = np.zeros(self.chunk_size, dtype=bool)
        self.timestep = 0
        self.current_chunk = None

    def preprocess_image(self, image):
        img = cv2.resize(image, (480, 360))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def hybrid_to_pose(self, hybrid_10d):
        pos = hybrid_10d[0:3]
        rot6d = hybrid_10d[3:9]
        jaw = hybrid_10d[9]
        col1 = rot6d[0:3]
        col2 = rot6d[3:6]
        col1 = col1 / (np.linalg.norm(col1) + 1e-8)
        col2 = col2 - np.dot(col2, col1) * col1
        col2 = col2 / (np.linalg.norm(col2) + 1e-8)
        col3 = np.cross(col1, col2)
        rot_matrix = np.stack([col1, col2, col3], axis=1)
        r = R.from_matrix(rot_matrix)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz, jaw

    def predict(self, kinematics, images=None):
        chunk_idx = self.timestep % self.chunk_size
        if chunk_idx == 0 or self.current_chunk is None:
            qpos = self._extract_qpos(kinematics)
            self.current_chunk = self._run_inference_raw(images, qpos=qpos)

        if self.temporal_agg:
            qpos = self._extract_qpos(kinematics)
            raw_chunk = self._run_inference_raw(images, qpos=qpos)
            buf_idx = self.timestep % self.chunk_size
            self.all_actions[buf_idx] = raw_chunk
            self.all_actions_filled[buf_idx] = True
            weights = []
            actions = []
            for age in range(self.chunk_size):
                src_idx = (self.timestep - age) % self.chunk_size
                if not self.all_actions_filled[src_idx]:
                    continue
                w = np.exp(-self.temporal_weight * age)
                weights.append(w)
                actions.append(self.all_actions[src_idx][age])
            weights = np.array(weights)
            actions = np.array(actions)
            action_20d = np.average(actions, axis=0, weights=weights)
            self.timestep += 1
        else:
            action_20d = self.current_chunk[chunk_idx]
            self.timestep += 1

        if self.action_mean is not None:
            # Denormalize positions (0:3, 10:13) and jaw (9, 19); rotations stay raw
            raw_20d = action_20d * self.action_std + self.action_mean
            raw_20d[3:9] = action_20d[3:9]
            raw_20d[13:19] = action_20d[13:19]
        else:
            raw_20d = action_20d

        psm1_pos, psm1_quat, psm1_jaw = self.hybrid_to_pose(raw_20d[0:10])
        psm2_pos, psm2_quat, psm2_jaw = self.hybrid_to_pose(raw_20d[10:20])

        if self.action_mean is not None:
            psm1_pos_sim = EE_HOME + MOTION_SCALE * (psm1_pos - self.action_mean[0:3])
            psm2_pos_sim = EE_HOME + MOTION_SCALE * (psm2_pos - self.action_mean[10:13])
        else:
            psm1_pos_sim = EE_HOME + MOTION_SCALE * psm1_pos
            psm2_pos_sim = EE_HOME + MOTION_SCALE * psm2_pos

        psm1_grip = np.clip(psm1_jaw * 5.0, -1.0, 1.0)
        psm2_grip = np.clip(psm2_jaw * 5.0, -1.0, 1.0)

        return {
            "robot1_ee_target": np.concatenate([psm1_pos_sim, psm1_quat]),
            "robot2_ee_target": np.concatenate([psm2_pos_sim, psm2_quat]),
            "robot1_gripper": float(psm1_grip),
            "robot2_gripper": float(psm2_grip),
            "r1_dvrk_pos": psm1_pos.copy(),
            "r2_dvrk_pos": psm2_pos.copy(),
        }

    def _extract_qpos(self, kinematics):
        qpos = np.zeros(20, dtype=np.float32)
        r1_jp = kinematics.get("robot1_joint_pos", np.zeros(8))
        r2_jp = kinematics.get("robot2_joint_pos", np.zeros(8))
        sim_qpos = np.zeros(14, dtype=np.float32)
        sim_qpos[:7] = r1_jp[:7]
        sim_qpos[7:14] = r2_jp[:7]
        sim_displacement = sim_qpos - SIM_QPOS_HOME
        sim_range = np.array([0.5, 0.5, 0.1, 1.0, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.1, 1.0, 0.5, 0.5, 0.5])
        normalized = sim_displacement / sim_range
        mapped_qpos = REAL_QPOS_MEAN + normalized * REAL_QPOS_STD
        qpos[:14] = mapped_qpos.astype(np.float32)
        return qpos

    def _run_inference_raw(self, images=None, qpos=None):
        with torch.no_grad():
            if images is not None:
                imgs = []
                for cam_name in self.camera_names:
                    imgs.append(self.preprocess_image(images[cam_name]))
                image_data = np.stack(imgs, axis=0)
            else:
                image_data = np.zeros((3, 3, 360, 480), dtype=np.float32)
            image_tensor = torch.from_numpy(image_data).float().cuda().unsqueeze(0)
            if qpos is not None:
                qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            else:
                qpos_tensor = torch.zeros(1, self.state_dim).float().cuda()
            action_chunk = self.policy(qpos_tensor, image_tensor)
            return action_chunk.cpu().numpy()[0]


class DualArmACTPolicyInference:
    """Loads two separate 10D ACT models (one per arm) and combines their outputs."""

    def __init__(self, ckpt_arm1, ckpt_arm2, device="cuda", temporal_weight=0.01):
        self.device = device
        self.chunk_size = 60
        self.camera_names = ['left', 'left_wrist', 'right_wrist']
        self.temporal_weight = temporal_weight
        self.temporal_agg = temporal_weight > 0
        self.timestep = 0
        self.current_chunk = None
        self.all_actions = np.zeros((self.chunk_size, self.chunk_size, 20))
        self.all_actions_filled = np.zeros(self.chunk_size, dtype=bool)

        self._build_dual(ckpt_arm1, ckpt_arm2)

        print(f"[DualArm] arm1 loaded from {ckpt_arm1}")
        print(f"[DualArm] arm2 loaded from {ckpt_arm2}")
        print(f"[DualArm] Temporal ensembling: k={self.temporal_weight}"
              f"{' (disabled)' if not self.temporal_agg else ''}")

    def _build_dual(self, ckpt_arm1, ckpt_arm2):
        saved_argv = sys.argv
        sys.argv = [
            'act', '--task_name', 'sim_needle_pickup_arm1',
            '--ckpt_dir', '/tmp', '--policy_class', 'ACT',
            '--seed', '0', '--num_epochs', '1',
            '--kl_weight', '1', '--chunk_size', '60',
            '--hidden_dim', '512', '--dim_feedforward', '3200',
            '--lr', '1e-5', '--batch_size', '8',
            '--image_encoder', 'resnet18', '--policy_level', 'low',
        ]
        sys.path.insert(0, os.path.expanduser('~/src/act'))
        sys.path.insert(0, os.path.expanduser('~/src'))
        from policy import ACTPolicy

        policy_config_10d = {
            "lr": 1e-5, "num_queries": self.chunk_size,
            "action_dim": 10, "kl_weight": 1,
            "hidden_dim": 512, "dim_feedforward": 3200,
            "lr_backbone": 1e-5, "backbone": "resnet18",
            "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": self.camera_names, "multi_gpu": False,
        }

        self.policy_arm1 = ACTPolicy(policy_config_10d)
        ckpt1 = torch.load(ckpt_arm1, map_location=self.device)
        self.policy_arm1.load_state_dict(ckpt1['model_state_dict'])
        self.policy_arm1.cuda()
        self.policy_arm1.eval()

        self.policy_arm2 = ACTPolicy(policy_config_10d)
        ckpt2 = torch.load(ckpt_arm2, map_location=self.device)
        self.policy_arm2.load_state_dict(ckpt2['model_state_dict'])
        self.policy_arm2.cuda()
        self.policy_arm2.eval()

        sys.argv = saved_argv

    def reset_temporal_state(self):
        """Reset temporal ensembling state (call at subtask boundaries)."""
        self.all_actions = np.zeros((self.chunk_size, self.chunk_size, 20))
        self.all_actions_filled = np.zeros(self.chunk_size, dtype=bool)
        self.timestep = 0
        self.current_chunk = None

    def preprocess_image(self, image):
        img = cv2.resize(image, (480, 360))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def hybrid_to_pose(self, hybrid_10d):
        pos = hybrid_10d[0:3]
        rot6d = hybrid_10d[3:9]
        jaw = hybrid_10d[9]
        col1 = rot6d[0:3]
        col2 = rot6d[3:6]
        col1 = col1 / (np.linalg.norm(col1) + 1e-8)
        col2 = col2 - np.dot(col2, col1) * col1
        col2 = col2 / (np.linalg.norm(col2) + 1e-8)
        col3 = np.cross(col1, col2)
        rot_matrix = np.stack([col1, col2, col3], axis=1)
        r = R.from_matrix(rot_matrix)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz, jaw

    def _extract_qpos_per_arm(self, kinematics, arm_id):
        r1_jp = kinematics.get("robot1_joint_pos", np.zeros(8))
        r2_jp = kinematics.get("robot2_joint_pos", np.zeros(8))
        if arm_id == 1:
            sim_qpos_7 = r1_jp[:7].astype(np.float32)
            home_7 = SIM_QPOS_HOME[:7]
            mean_7 = REAL_QPOS_MEAN[:7]
            std_7 = REAL_QPOS_STD[:7]
        else:
            sim_qpos_7 = r2_jp[:7].astype(np.float32)
            home_7 = SIM_QPOS_HOME[7:14]
            mean_7 = REAL_QPOS_MEAN[7:14]
            std_7 = REAL_QPOS_STD[7:14]
        sim_range = np.array([0.5, 0.5, 0.1, 1.0, 0.5, 0.5, 0.5])
        normalized = (sim_qpos_7 - home_7) / sim_range
        mapped = mean_7 + normalized * std_7
        qpos = np.zeros(10, dtype=np.float32)
        qpos[:7] = mapped
        return qpos

    def _run_dual_inference(self, images, kinematics):
        with torch.no_grad():
            if images is not None:
                imgs = []
                for cam_name in self.camera_names:
                    imgs.append(self.preprocess_image(images[cam_name]))
                image_data = np.stack(imgs, axis=0)
            else:
                image_data = np.zeros((3, 3, 360, 480), dtype=np.float32)
            image_tensor = torch.from_numpy(image_data).float().cuda().unsqueeze(0)

            qpos1 = self._extract_qpos_per_arm(kinematics, 1)
            qpos1_t = torch.from_numpy(qpos1).float().cuda().unsqueeze(0)
            chunk1 = self.policy_arm1(qpos1_t, image_tensor).cpu().numpy()[0]

            qpos2 = self._extract_qpos_per_arm(kinematics, 2)
            qpos2_t = torch.from_numpy(qpos2).float().cuda().unsqueeze(0)
            chunk2 = self.policy_arm2(qpos2_t, image_tensor).cpu().numpy()[0]

            return np.concatenate([chunk1, chunk2], axis=1)

    def predict(self, kinematics, images=None):
        chunk_idx = self.timestep % self.chunk_size
        if chunk_idx == 0 or self.current_chunk is None:
            self.current_chunk = self._run_dual_inference(images, kinematics)

        if self.temporal_agg:
            raw_chunk = self._run_dual_inference(images, kinematics)
            buf_idx = self.timestep % self.chunk_size
            self.all_actions[buf_idx] = raw_chunk
            self.all_actions_filled[buf_idx] = True
            weights = []
            actions = []
            for age in range(self.chunk_size):
                src_idx = (self.timestep - age) % self.chunk_size
                if not self.all_actions_filled[src_idx]:
                    continue
                w = np.exp(-self.temporal_weight * age)
                weights.append(w)
                actions.append(self.all_actions[src_idx][age])
            weights = np.array(weights)
            actions = np.array(actions)
            action_20d = np.average(actions, axis=0, weights=weights)
            self.timestep += 1
        else:
            action_20d = self.current_chunk[chunk_idx]
            self.timestep += 1

        raw_20d = action_20d
        psm1_pos, psm1_quat, psm1_jaw = self.hybrid_to_pose(raw_20d[0:10])
        psm2_pos, psm2_quat, psm2_jaw = self.hybrid_to_pose(raw_20d[10:20])

        psm1_pos_sim = EE_HOME + MOTION_SCALE * psm1_pos
        psm2_pos_sim = EE_HOME + MOTION_SCALE * psm2_pos

        psm1_grip = np.clip(psm1_jaw * 5.0, -1.0, 1.0)
        psm2_grip = np.clip(psm2_jaw * 5.0, -1.0, 1.0)

        return {
            "robot1_ee_target": np.concatenate([psm1_pos_sim, psm1_quat]),
            "robot2_ee_target": np.concatenate([psm2_pos_sim, psm2_quat]),
            "robot1_gripper": float(psm1_grip),
            "robot2_gripper": float(psm2_grip),
            "r1_dvrk_pos": psm1_pos.copy(),
            "r2_dvrk_pos": psm2_pos.copy(),
        }


# SECTION 2: SUBTASK CHAINER (NEW)

class SubtaskChainer:
    """Chains multiple ACT subtask policies in sequence.

    Each subtask runs for a fixed number of steps. When a subtask completes,
    the chainer advances to the next one and resets temporal ensembling state
    so the new policy starts with a clean prediction buffer.

    Attributes:
        subtasks: Ordered list of dicts with keys: name, policy, steps.
        current_idx: Index of the active subtask.
        step_in_subtask: Step counter within the current subtask.
        global_step: Total steps across all subtasks.
        total_steps: Sum of all subtask step counts.
    """

    def __init__(self, subtasks):
        """Initialize the chainer with an ordered list of subtask configs.

        Args:
            subtasks: List of dicts, each with:
                - name (str): Human-readable subtask name.
                - policy: An ACTPolicyInference or DualArmACTPolicyInference instance.
                - steps (int): Number of steps to run this subtask.
        """
        self.subtasks = subtasks
        self.current_idx = 0
        self.step_in_subtask = 0
        self.global_step = 0
        self.total_steps = sum(s["steps"] for s in subtasks)

        print(f"\n[SubtaskChainer] Loaded {len(subtasks)} subtasks:")
        for i, s in enumerate(subtasks):
            step_range_start = sum(st["steps"] for st in subtasks[:i])
            step_range_end = step_range_start + s["steps"] - 1
            print(f"  {i+1}. {s['name']:20s} | {s['steps']:4d} steps | global [{step_range_start}-{step_range_end}]")
        print(f"  Total: {self.total_steps} steps\n")

    @property
    def current_subtask(self):
        return self.subtasks[self.current_idx]

    @property
    def current_name(self):
        return self.current_subtask["name"]

    @property
    def current_policy(self):
        return self.current_subtask["policy"]

    @property
    def done(self):
        return self.current_idx >= len(self.subtasks)

    def step(self, kinematics, images=None):
        """Run one step of the active subtask policy.

        Returns:
            action_dict: The policy output (same format as predict()).
            metadata: Dict with subtask_name, subtask_idx, step_in_subtask, global_step, switched.
        """
        if self.done:
            raise RuntimeError("All subtasks completed  -- no more steps to run.")

        switched = False

        # Check if current subtask is exhausted -> advance
        if self.step_in_subtask >= self.current_subtask["steps"]:
            self.current_idx += 1
            self.step_in_subtask = 0
            switched = True
            if self.done:
                raise RuntimeError("All subtasks completed  -- no more steps to run.")
            # Reset temporal ensembling on the new policy
            self.current_policy.reset_temporal_state()
            print(f"\n{'='*60}")
            print(f"[SWITCH] Advancing to subtask: {self.current_name} (idx={self.current_idx})")
            print(f"{'='*60}\n")

        action_dict = self.current_policy.predict(kinematics, images=images)

        metadata = {
            "subtask_name": self.current_name,
            "subtask_idx": self.current_idx,
            "step_in_subtask": self.step_in_subtask,
            "global_step": self.global_step,
            "switched": switched,
        }

        self.step_in_subtask += 1
        self.global_step += 1

        return action_dict, metadata


# SECTION 3: ENVIRONMENT SETUP (from v5)

def make_env_with_camera(args_cli):
    # Fabric MUST stay enabled for camera rendering to reflect physics updates.
    # Cameras are manually initialized so fabric compatibility is fine.
    use_fabric = not args_cli.disable_fabric
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=use_fabric
    )
    env_cfg.scene.overhead_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCamera",
        update_period=0.0, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0, horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    env_cfg.scene.viz_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VizCamera",
        update_period=0.0, height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955,
            focus_distance=400.0, clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.35, 0.35, 0.55),
            rot=(0.3573, 0.1371, 0.3310, 0.8626),
            convention="opengl",
        ),
    )
    # Endoscope camera  -- static scene view (maps to 'left' in ACT)
    env_cfg.scene.endoscope_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/EndoscopeCamera",
        update_period=0.0, height=540, width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.10, 0.18),
            rot=(0.9665, 0.2567, 0.0, 0.0),
            convention="opengl",
        ),
    )
    # PSM1 wrist camera  -- attached to Robot_1 tool tip (maps to 'right_wrist' in ACT)
    env_cfg.scene.psm1_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link/WristCam1",
        update_period=0.0, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, horizontal_aperture=20.955,
            clipping_range=(0.001, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.035),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    # PSM2 wrist camera  -- attached to Robot_2 tool tip (maps to 'left_wrist' in ACT)
    env_cfg.scene.psm2_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link/WristCam2",
        update_period=0.0, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, horizontal_aperture=20.955,
            clipping_range=(0.001, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.035),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    env_cfg.scene.suture_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SuturePad",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.08, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.25, 0.25),
                roughness=0.75,
                metallic=0.05,
            ),
        ),
    )
    env_cfg.scene.drape = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Drape",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.001)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.18, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.35, 0.55),
                roughness=0.8,
                metallic=0.02,
            ),
        ),
    )
    env_cfg.viewer.eye = (0.0, 0.5, 0.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.05)
    env = gym.make(args_cli.task, cfg=env_cfg)
    cam = env.unwrapped.scene["overhead_camera"]
    cam._initialize_impl()
    cam._is_initialized = True
    viz_cam = env.unwrapped.scene["viz_camera"]
    viz_cam._initialize_impl()
    viz_cam._is_initialized = True
    # Init model cameras
    for cam_name in ["endoscope_camera", "psm1_wrist_camera", "psm2_wrist_camera"]:
        c = env.unwrapped.scene[cam_name]
        c._initialize_impl()
        c._is_initialized = True
    return env


def extract_kinematics(unwrapped):
    robot1 = unwrapped.scene["robot_1"]
    robot2 = unwrapped.scene["robot_2"]
    ee1 = unwrapped.scene["ee_1_frame"]
    ee2 = unwrapped.scene["ee_2_frame"]
    return {
        "robot1_joint_pos": robot1.data.joint_pos[0].cpu().numpy(),
        "robot1_joint_vel": robot1.data.joint_vel[0].cpu().numpy(),
        "robot1_ee_pos":    ee1.data.target_pos_w[0, 0].cpu().numpy(),
        "robot1_ee_quat":   ee1.data.target_quat_w[0, 0].cpu().numpy(),
        "robot2_joint_pos": robot2.data.joint_pos[0].cpu().numpy(),
        "robot2_joint_vel": robot2.data.joint_vel[0].cpu().numpy(),
        "robot2_ee_pos":    ee2.data.target_pos_w[0, 0].cpu().numpy(),
        "robot2_ee_quat":   ee2.data.target_quat_w[0, 0].cpu().numpy(),
    }


def get_camera_frame(unwrapped):
    cam = unwrapped.scene["overhead_camera"]
    rgb = cam.data.output["rgb"]
    frame = rgb[0].cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    return frame


def get_viz_camera_frame(unwrapped):
    cam = unwrapped.scene["viz_camera"]
    rgb = cam.data.output["rgb"]
    frame = rgb[0].cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    return frame


def get_camera_frames(unwrapped):
    """Capture RGB frames from all 3 model cameras.

    Returns dict mapping ACT camera names to numpy frames:
        'left' -> endoscope, 'left_wrist' -> PSM2 wrist, 'right_wrist' -> PSM1 wrist
    """
    cam_mapping = {
        'left': 'endoscope_camera',
        'left_wrist': 'psm2_wrist_camera',    # PSM2 = Robot_2 = left arm
        'right_wrist': 'psm1_wrist_camera',    # PSM1 = Robot_1 = right arm
    }
    frames = {}
    for act_name, scene_name in cam_mapping.items():
        cam = unwrapped.scene[scene_name]
        rgb = cam.data.output["rgb"]
        frame = rgb[0].cpu().numpy()
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frames[act_name] = frame
    return frames


def format_action(model_output, env, device):
    action = torch.zeros(env.action_space.shape, device=device)
    action[0, 0:7] = torch.tensor(model_output["robot1_ee_target"], device=device, dtype=torch.float32)
    action[0, 7] = model_output["robot1_gripper"]
    action[0, 8:15] = torch.tensor(model_output["robot2_ee_target"], device=device, dtype=torch.float32)
    action[0, 15] = model_output["robot2_gripper"]
    return action


# SECTION 4: MAIN LOOP

def main():
    global MOTION_SCALE
    if args_cli.motion_scale is not None:
        MOTION_SCALE = args_cli.motion_scale
        print(f"[CONFIG] MOTION_SCALE overridden to {MOTION_SCALE}")

    env = make_env_with_camera(args_cli)
    unwrapped = env.unwrapped

    tw = args_cli.temporal_weight

    # Load all 3 models upfront (20D each, with per-subtask denormalization)
    print("\n[Loading models...]")
    print("  NP (needle pickup)  -- 20D")
    np_policy = ACTPolicyInference(
        args_cli.ckpt_np, device=args_cli.device, temporal_weight=tw,
        action_mean=NORM_STATS['needle_pickup']['mean'],
        action_std=NORM_STATS['needle_pickup']['std'])

    print("  NT (needle throw)  -- 20D")
    nt_policy = ACTPolicyInference(
        args_cli.ckpt_nt, device=args_cli.device, temporal_weight=tw,
        action_mean=NORM_STATS['needle_throw']['mean'],
        action_std=NORM_STATS['needle_throw']['std'])

    print("  KT (knot tying)  -- 20D")
    kt_policy = ACTPolicyInference(
        args_cli.ckpt_kt, device=args_cli.device, temporal_weight=tw,
        action_mean=NORM_STATS['knot_tying']['mean'],
        action_std=NORM_STATS['knot_tying']['std'])

    # Build subtask sequence
    subtasks = [
        {"name": "needle_pickup", "policy": np_policy, "steps": args_cli.steps_np},
        {"name": "needle_throw",  "policy": nt_policy, "steps": args_cli.steps_nt},
        {"name": "knot_tying",    "policy": kt_policy, "steps": args_cli.steps_kt},
    ]
    chainer = SubtaskChainer(subtasks)

    # Output directories
    frames_dir = os.path.expanduser("~/chained_frames")
    viz_frames_dir = os.path.expanduser("~/chained_viz_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(viz_frames_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ACT Task Chaining: Needle Pickup -> Needle Throw -> Knot Tying")
    print(f"  Task:       {args_cli.task}")
    print(f"  Subtasks:   NP({args_cli.steps_np}) -> NT({args_cli.steps_nt}) -> KT({args_cli.steps_kt})")
    print(f"  Total:      {chainer.total_steps} steps")
    print(f"  Temporal k: {tw}")
    print(f"  Frames:     {frames_dir}/")
    print(f"  Viz:        {viz_frames_dir}/")
    print(f"{'='*60}\n")

    obs, info = env.reset()
    trajectory_log = []

    has_gui = unwrapped.sim.has_gui()
    step_dt = 1.0 / 30.0

    # Log initial EE positions
    init_kin = extract_kinematics(unwrapped)
    print(f"[INIT] R1 EE: {init_kin['robot1_ee_pos'].round(4)}")
    print(f"[INIT] R2 EE: {init_kin['robot2_ee_pos'].round(4)}")
    print(f"[INIT] MOTION_SCALE={MOTION_SCALE}, temporal_weight={tw}")

    if has_gui:
        print(f"\nRunning {chainer.total_steps} steps with GUI (real-time pacing)...\n")
    else:
        print(f"\nRunning {chainer.total_steps} steps (headless)...\n")

    current_subtask_idx = -1

    global_step = 0
    while simulation_app.is_running() and global_step < chainer.total_steps:
        step_start = time.time()
        with torch.inference_mode():
            kinematics = extract_kinematics(unwrapped)
            images = get_camera_frames(unwrapped)

            model_output, meta = chainer.step(kinematics, images=images)

            # Absolute mode (same as v5): model predict() already computes
            # pos_sim = EE_HOME + MOTION_SCALE * model_pos in robot base frame.
            # No delta override needed.
            if meta["subtask_idx"] != current_subtask_idx:
                current_subtask_idx = meta["subtask_idx"]
                r1_tgt = model_output["robot1_ee_target"][:3]
                r2_tgt = model_output["robot2_ee_target"][:3]
                print(f"  [SUBTASK {current_subtask_idx}] First target: "
                      f"R1={r1_tgt.round(4)}, R2={r2_tgt.round(4)}")

            action = format_action(model_output, env, unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory_log.append({
                "step": global_step,
                "subtask": meta["subtask_name"],
                "subtask_idx": meta["subtask_idx"],
                "step_in_subtask": meta["step_in_subtask"],
                "r1_ee_pos": kinematics["robot1_ee_pos"].copy(),
                "r2_ee_pos": kinematics["robot2_ee_pos"].copy(),
                "r1_target_sim": model_output["robot1_ee_target"][:3].copy(),
                "r2_target_sim": model_output["robot2_ee_target"][:3].copy(),
                "r1_dvrk_pos": model_output["r1_dvrk_pos"],
                "r2_dvrk_pos": model_output["r2_dvrk_pos"],
                "r1_grip": model_output["robot1_gripper"],
                "r2_grip": model_output["robot2_gripper"],
                "r1_qpos": kinematics["robot1_joint_pos"][:7].copy(),
                "r2_qpos": kinematics["robot2_joint_pos"][:7].copy(),
                "reward": reward.item(),
            })

            # Save frames every 5 steps
            if global_step % 5 == 0:
                from PIL import Image as PILImage
                # Save endoscope (left) as the main overhead debug frame
                img = PILImage.fromarray(images['left'].astype(np.uint8))
                img.save(os.path.join(frames_dir, f"frame_{global_step:04d}.png"))
                viz_frame = get_viz_camera_frame(unwrapped)
                viz_img = PILImage.fromarray(viz_frame.astype(np.uint8))
                viz_img.save(os.path.join(viz_frames_dir, f"frame_{global_step:04d}.png"))
                # Save all 3 model camera views
                for cam_name in ['left', 'left_wrist', 'right_wrist']:
                    cam_dir = os.path.join(frames_dir, cam_name)
                    os.makedirs(cam_dir, exist_ok=True)
                    cam_img = PILImage.fromarray(images[cam_name].astype(np.uint8))
                    cam_img.save(os.path.join(cam_dir, f"frame_{global_step:04d}.png"))

            # Progress logging every 20 steps
            if global_step % 20 == 0:
                r1_ee = kinematics["robot1_ee_pos"].round(4)
                r1_tgt = model_output["robot1_ee_target"][:3].round(4)
                grip = model_output["robot1_gripper"]
                phase = meta["subtask_name"]
                sub_step = meta["step_in_subtask"]
                print(f"  [{phase:15s} {sub_step:3d}] Step {global_step:4d}: "
                      f"R1 sim={r1_ee} | target={r1_tgt} | grip={grip:.2f}")

        if has_gui:
            elapsed = time.time() - step_start
            sleep_time = step_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        global_step += 1

    # Save trajectory
    save_path = os.path.expanduser("~/chained_trajectory.npz")
    subtask_names = [d["subtask"] for d in trajectory_log]
    # Encode subtask names as integers for npz compatibility
    subtask_name_map = {name: i for i, name in enumerate(["needle_pickup", "needle_throw", "knot_tying"])}
    subtask_ids = np.array([subtask_name_map[n] for n in subtask_names])

    np.savez(save_path,
        steps=np.array([d["step"] for d in trajectory_log]),
        subtask_ids=subtask_ids,
        subtask_steps=np.array([d["step_in_subtask"] for d in trajectory_log]),
        r1_ee_pos=np.array([d["r1_ee_pos"] for d in trajectory_log]),
        r2_ee_pos=np.array([d["r2_ee_pos"] for d in trajectory_log]),
        r1_target_sim=np.array([d["r1_target_sim"] for d in trajectory_log]),
        r2_target_sim=np.array([d["r2_target_sim"] for d in trajectory_log]),
        r1_dvrk_pos=np.array([d["r1_dvrk_pos"] for d in trajectory_log]),
        r2_dvrk_pos=np.array([d["r2_dvrk_pos"] for d in trajectory_log]),
        r1_grip=np.array([d["r1_grip"] for d in trajectory_log]),
        r2_grip=np.array([d["r2_grip"] for d in trajectory_log]),
        r1_qpos=np.array([d["r1_qpos"] for d in trajectory_log]),
        r2_qpos=np.array([d["r2_qpos"] for d in trajectory_log]),
        rewards=np.array([d["reward"] for d in trajectory_log]),
        subtask_name_map=np.array(["needle_pickup", "needle_throw", "knot_tying"]),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Chained execution complete: {global_step} total steps")
    print(f"{'='*60}")
    print(f"[SAVED] Trajectory: {save_path}")
    print(f"[SAVED] Overhead frames: {frames_dir}/ ({len(os.listdir(frames_dir))} frames)")
    print(f"[SAVED] Viz frames: {viz_frames_dir}/ ({len(os.listdir(viz_frames_dir))} frames)")

    print(f"\n--- Per-Subtask Summary ---")
    for st in subtasks:
        name = st["name"]
        st_entries = [d for d in trajectory_log if d["subtask"] == name]
        if not st_entries:
            print(f"  {name}: (no steps recorded)")
            continue
        r1_pos = np.array([d["r1_ee_pos"] for d in st_entries])
        r1_tgt = np.array([d["r1_target_sim"] for d in st_entries])
        r1_disp = np.linalg.norm(r1_pos[-1] - r1_pos[0])
        grips = np.array([d["r1_grip"] for d in st_entries])
        print(f"  {name:20s}: {len(st_entries):4d} steps | "
              f"R1 displacement={r1_disp:.4f}m | "
              f"grip range=[{grips.min():.2f}, {grips.max():.2f}]")

    # Transition continuity check
    print(f"\n--- Transition Continuity ---")
    for i in range(len(subtasks) - 1):
        name_a = subtasks[i]["name"]
        name_b = subtasks[i + 1]["name"]
        entries_a = [d for d in trajectory_log if d["subtask"] == name_a]
        entries_b = [d for d in trajectory_log if d["subtask"] == name_b]
        if entries_a and entries_b:
            last_pos = entries_a[-1]["r1_ee_pos"]
            first_pos = entries_b[0]["r1_ee_pos"]
            jump = np.linalg.norm(last_pos - first_pos)
            last_grip = entries_a[-1]["r1_grip"]
            first_grip = entries_b[0]["r1_grip"]
            print(f"  {name_a} -> {name_b}: "
                  f"R1 position jump={jump:.4f}m | "
                  f"grip {last_grip:.2f} -> {first_grip:.2f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
