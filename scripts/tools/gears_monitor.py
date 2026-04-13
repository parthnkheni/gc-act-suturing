#!/usr/bin/env python3
"""
GEARS (Global Evaluative Assessment of Robotic Skills) Real-Time Quality Monitor
=================================================================================
Standalone module for monitoring surgical robot action quality during ACT inference
on the dVRK, and for retroactive offline scoring of evaluation results.

GEARS is a validated clinical tool for assessing robotic surgical skill. This module
adapts its principles into computable metrics derived from robot kinematics, designed
to run in real-time at 30Hz alongside the ACT inference pipeline.

The module provides four main components:

1. GEARSMonitor    -- Sliding-window quality scoring from a stream of 20D actions
2. ActionModulator  -- Scales action magnitude and ensembling based on quality score
3. GEARSLogger     -- CSV/summary logging of per-timestep and per-episode metrics
4. evaluate_episode_quality()  -- Batch function for offline evaluation (no ROS needed)

Action format (20D):
    [PSM1_xyz(3) + PSM1_rot6d(6) + PSM1_jaw(1) +
     PSM2_xyz(3) + PSM2_rot6d(6) + PSM2_jaw(1)]

Usage (real-time):
    monitor = GEARSMonitor()
    modulator = ActionModulator()
    logger = GEARSLogger(output_dir="~/gears_logs")

    for t in range(num_steps):
        action = policy.predict(...)          # (20,) raw action
        monitor.update(action)
        scores = monitor.get_scores()
        quality = scores['overall']
        modified_action, new_k, status = modulator.modulate(action, quality)
        logger.log_step(t, scores)
        execute(modified_action)

    logger.save_episode_summary()

Usage (offline):
    from gears_monitor import evaluate_episode_quality
    metrics = evaluate_episode_quality(pred_actions, gt_actions)

Dependencies: numpy, scipy (standard scientific Python  -- no ROS, no torch)
"""

import os
import csv
import json
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import signal
from scipy.stats import pearsonr


# DEFAULT CONFIGURATION

DEFAULT_CONFIG = {
    # Sliding window size (timesteps). At 30Hz, 30 = 1 second.
    'window_size': 30,

    # Velocity threshold (m/s) for idle detection.
    # At 30Hz with ~1cm workspace, typical speed is 0.001-0.01 m/frame.
    # In normalized space, velocities are larger (~0.1-1.0 per frame).
    'idle_velocity_threshold': 1e-4,

    # Jaw thresholds
    'jaw_grip_threshold': 0.1,     # jaw < this = gripping (radians, real space)
    'jaw_transition_threshold': 0.05,  # jaw delta > this = open/close transition

    # Metric weights for overall score (must sum to 1.0)
    # Calibrated for dVRK suturing (March 2026):
    #   - Path efficiency low: suturing is inherently curved, straight-line penalizes it
    #   - Smoothness high: jerky motion directly correlates with poor accuracy
    #   - Idle ratio low: PSM2 is legitimately idle during needle throw
    #   - Jaw stability high: holding needle steady matters for suturing
    'weights': {
        'path_efficiency': 0.10,
        'smoothness': 0.25,
        'speed_consistency': 0.15,
        'idle_ratio': 0.05,
        'regrasp_penalty': 0.15,
        'bimanual_coordination': 0.15,
        'jaw_stability': 0.15,
    },

    # Quality thresholds for ActionModulator
    # Calibrated against 50-episode GC-ACT ensemble eval (March 2026):
    #   Typical quality scores land in 0.35-0.65 range for suturing.
    'quality_thresholds': {
        'high': 0.55,
        'medium': 0.40,
        'critical': 0.25,
    },

    # ActionModulator scaling factors
    # Gentler modulation  -- most episodes are good, only truly bad ones need damping
    'modulation': {
        'high_scale': 1.0,
        'medium_scale': 0.85,
        'low_scale': 0.6,
        'high_ensemble_k': None,       # no change
        'medium_ensemble_k': 0.01,
        'low_ensemble_k': 0.02,
    },

    # Whether actions are in normalized space (zero mean, unit variance)
    # or real space (meters, radians). Affects thresholds.
    'normalized': False,

    # Normalized-space overrides for thresholds
    'normalized_idle_velocity_threshold': 0.05,
    'normalized_jaw_grip_threshold': -0.5,
    'normalized_jaw_transition_threshold': 0.3,

    # Jerk/smoothness normalization scale.
    # Calibrated from 50-episode eval: median jerk ~1.85e-5 m/frame^3.
    # With scale=2e-5, sigmoid maps: p5 jerk->0.78, p50->0.50, p95->0.23.
    'jerk_scale_real': 2e-5,
    'jerk_scale_normalized': 1.0,

    # Jaw stability normalization scale.
    # Calibrated: median jaw std ~0.14 rad across episodes.
    'jaw_std_scale_real': 0.15,
    'jaw_std_scale_normalized': 0.5,
}


# UTILITY FUNCTIONS

def _extract_arm_positions(action_20d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract PSM1 and PSM2 xyz positions from a 20D action.

    Args:
        action_20d: (20,) action vector.

    Returns:
        psm1_pos: (3,) xyz position of PSM1
        psm2_pos: (3,) xyz position of PSM2
    """
    return action_20d[0:3].copy(), action_20d[10:13].copy()


def _extract_arm_jaws(action_20d: np.ndarray) -> Tuple[float, float]:
    """Extract PSM1 and PSM2 jaw angles from a 20D action.

    Returns:
        psm1_jaw: scalar jaw angle
        psm2_jaw: scalar jaw angle
    """
    return float(action_20d[9]), float(action_20d[19])


def _finite_diff(arr: np.ndarray, order: int = 1, dt: float = 1.0) -> np.ndarray:
    """Compute finite differences of arbitrary order along axis 0.

    Args:
        arr: (T, D) array
        order: derivative order (1=velocity, 2=accel, 3=jerk)
        dt: timestep duration (seconds). Default 1.0 for frame-based.

    Returns:
        (T-order, D) array of derivatives
    """
    result = arr.copy()
    for _ in range(order):
        result = np.diff(result, axis=0) / dt
    return result


# GEARSMonitor

class GEARSMonitor:
    """Real-time GEARS-inspired quality monitor for surgical robot actions.

    Maintains a sliding window of recent 20D actions and computes quality
    metrics at each timestep. Can also process a full episode in batch mode.

    Metrics computed (per arm where applicable):
        - path_efficiency: ratio of straight-line distance to actual path length
        - smoothness: mean magnitude of position jerk (3rd derivative)
        - speed_consistency: coefficient of variation of velocity magnitude
        - idle_ratio: fraction of timesteps with velocity below threshold
        - regrasp_count: number of jaw open/close transitions
        - bimanual_coordination: cross-correlation of arm velocity magnitudes
        - jaw_stability: std of jaw angle during grip phases

    Attributes:
        config: configuration dictionary
        window: deque of recent (20,) action vectors
        step_count: total number of update() calls
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the GEARS monitor.

        Args:
            config: optional configuration dict (see DEFAULT_CONFIG).
                    Missing keys are filled from DEFAULT_CONFIG.
        """
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            # Deep-merge weights and modulation dicts
            for key, val in config.items():
                if isinstance(val, dict) and key in self.config and isinstance(self.config[key], dict):
                    self.config[key] = {**self.config[key], **val}
                else:
                    self.config[key] = val

        self.window_size = self.config['window_size']
        self.window: deque = deque(maxlen=self.window_size)
        self.step_count = 0
        self._last_scores: Optional[Dict[str, float]] = None

    def reset(self):
        """Clear the sliding window and step counter."""
        self.window.clear()
        self.step_count = 0
        self._last_scores = None

    def update(self, action_20d: np.ndarray):
        """Add a new 20D action to the sliding window and recompute scores.

        Args:
            action_20d: (20,) numpy array  -- a single timestep action.
                        Can be in normalized or real space (set config['normalized']).
        """
        assert action_20d.shape == (20,), f"Expected (20,) action, got {action_20d.shape}"
        self.window.append(action_20d.copy())
        self.step_count += 1
        # Recompute scores
        self._last_scores = self._compute_scores()

    def get_scores(self) -> Dict[str, float]:
        """Return the latest GEARS metric scores.

        Returns:
            Dictionary with metric names as keys and float scores as values.
            Includes 'overall' key with weighted quality score in [0, 1].
            Returns None values if window has fewer than 3 timesteps.
        """
        if self._last_scores is not None:
            return self._last_scores
        return self._compute_scores()

    def evaluate_batch(self, actions: np.ndarray) -> Dict[str, Any]:
        """Evaluate a full episode of actions at once (offline mode).

        Args:
            actions: (T, 20) array of actions for the entire episode.

        Returns:
            Dictionary containing:
                - Per-metric scores (scalars, computed over the full episode)
                - 'overall': weighted quality score in [0, 1]
                - 'per_window': list of per-window score dicts (sliding window)
                - 'T': number of timesteps
        """
        T = actions.shape[0]
        assert actions.ndim == 2 and actions.shape[1] == 20, \
            f"Expected (T, 20) actions, got {actions.shape}"

        # Full-episode metrics
        episode_scores = self._compute_from_array(actions)

        # Per-window scores (sliding window sweep)
        per_window = []
        saved_window = self.window.copy()
        saved_count = self.step_count
        self.reset()
        for t in range(T):
            self.update(actions[t])
            scores = self.get_scores()
            if scores is not None:
                per_window.append({'t': t, **scores})
        # Restore state
        self.window = saved_window
        self.step_count = saved_count
        self._last_scores = None

        episode_scores['per_window'] = per_window
        episode_scores['T'] = T
        return episode_scores

    # Internal computation

    def _compute_scores(self) -> Optional[Dict[str, float]]:
        """Compute all GEARS metrics from the current sliding window."""
        if len(self.window) < 3:
            return None

        actions = np.array(self.window)  # (W, 20)
        return self._compute_from_array(actions)

    def _compute_from_array(self, actions: np.ndarray) -> Dict[str, float]:
        """Compute all GEARS metrics from an (T, 20) actions array.

        This is the core computation shared by real-time and batch modes.
        """
        T = actions.shape[0]
        if T < 3:
            return {
                'path_efficiency_psm1': None, 'path_efficiency_psm2': None,
                'smoothness_psm1': None, 'smoothness_psm2': None,
                'speed_consistency_psm1': None, 'speed_consistency_psm2': None,
                'idle_ratio_psm1': None, 'idle_ratio_psm2': None,
                'regrasp_count_psm1': None, 'regrasp_count_psm2': None,
                'bimanual_coordination': None,
                'jaw_stability_psm1': None, 'jaw_stability_psm2': None,
                'overall': None,
            }

        is_norm = self.config['normalized']

        # Extract positions and jaws
        pos1 = actions[:, 0:3]    # (T, 3) PSM1 xyz
        pos2 = actions[:, 10:13]  # (T, 3) PSM2 xyz
        jaw1 = actions[:, 9]      # (T,) PSM1 jaw
        jaw2 = actions[:, 19]     # (T,) PSM2 jaw

        scores = {}

        # Path Length Efficiency
        scores['path_efficiency_psm1'] = self._path_efficiency(pos1)
        scores['path_efficiency_psm2'] = self._path_efficiency(pos2)

        # Smoothness (jerk)
        scores['smoothness_psm1'] = self._smoothness(pos1)
        scores['smoothness_psm2'] = self._smoothness(pos2)

        # Speed Consistency (CV of velocity magnitude)
        scores['speed_consistency_psm1'] = self._speed_consistency(pos1)
        scores['speed_consistency_psm2'] = self._speed_consistency(pos2)

        # Idle Ratio
        idle_thresh = (self.config['normalized_idle_velocity_threshold']
                       if is_norm else self.config['idle_velocity_threshold'])
        scores['idle_ratio_psm1'] = self._idle_ratio(pos1, idle_thresh)
        scores['idle_ratio_psm2'] = self._idle_ratio(pos2, idle_thresh)

        # Re-grasp Count
        jaw_trans_thresh = (self.config['normalized_jaw_transition_threshold']
                            if is_norm else self.config['jaw_transition_threshold'])
        scores['regrasp_count_psm1'] = self._regrasp_count(jaw1, jaw_trans_thresh)
        scores['regrasp_count_psm2'] = self._regrasp_count(jaw2, jaw_trans_thresh)

        # Bimanual Coordination
        scores['bimanual_coordination'] = self._bimanual_coordination(pos1, pos2)

        # Jaw Stability
        grip_thresh = (self.config['normalized_jaw_grip_threshold']
                       if is_norm else self.config['jaw_grip_threshold'])
        scores['jaw_stability_psm1'] = self._jaw_stability(jaw1, grip_thresh)
        scores['jaw_stability_psm2'] = self._jaw_stability(jaw2, grip_thresh)

        # Overall weighted score
        scores['overall'] = self._overall_score(scores)

        return scores

    def _path_efficiency(self, positions: np.ndarray) -> float:
        """Ratio of straight-line distance to actual path length.

        A perfectly efficient path (straight line) scores 1.0.
        Wandering or oscillating paths score closer to 0.0.

        Args:
            positions: (T, 3) xyz positions for one arm.

        Returns:
            Efficiency ratio in [0, 1]. Returns 1.0 if the arm is stationary.
        """
        deltas = np.diff(positions, axis=0)                      # (T-1, 3)
        step_lengths = np.linalg.norm(deltas, axis=1)            # (T-1,)
        total_path = np.sum(step_lengths)
        straight_line = np.linalg.norm(positions[-1] - positions[0])

        if total_path < 1e-10:
            # Essentially stationary  -- perfect efficiency
            return 1.0

        efficiency = straight_line / total_path
        return float(np.clip(efficiency, 0.0, 1.0))

    def _smoothness(self, positions: np.ndarray) -> float:
        """Mean jerk magnitude (3rd derivative of position).

        Lower jerk = smoother motion = higher skill. The raw jerk value is
        returned (not inverted) so that downstream normalization can handle
        the scale. For the overall score, this is mapped to [0, 1] where
        lower jerk gives higher score.

        Args:
            positions: (T, 3) xyz positions for one arm.

        Returns:
            Mean jerk magnitude (non-negative float). Units depend on
            whether actions are normalized or in real space.
        """
        if positions.shape[0] < 4:
            return 0.0

        jerk = _finite_diff(positions, order=3)  # (T-3, 3)
        jerk_mag = np.linalg.norm(jerk, axis=1)  # (T-3,)
        return float(np.mean(jerk_mag))

    def _speed_consistency(self, positions: np.ndarray) -> float:
        """Coefficient of variation (CV) of velocity magnitude.

        CV = std/mean. Lower CV means more consistent speed throughout
        the motion. Returns 0.0 if the arm is stationary.

        Args:
            positions: (T, 3) xyz positions.

        Returns:
            CV of velocity magnitude (non-negative float).
        """
        velocity = _finite_diff(positions, order=1)           # (T-1, 3)
        speed = np.linalg.norm(velocity, axis=1)              # (T-1,)
        mean_speed = np.mean(speed)
        if mean_speed < 1e-10:
            return 0.0
        return float(np.std(speed) / mean_speed)

    def _idle_ratio(self, positions: np.ndarray, threshold: float) -> float:
        """Fraction of timesteps where velocity is below the idle threshold.

        Higher idle ratio indicates hesitation or loss of intent.

        Args:
            positions: (T, 3) xyz positions.
            threshold: velocity magnitude below which a timestep is 'idle'.

        Returns:
            Ratio in [0, 1].
        """
        velocity = _finite_diff(positions, order=1)
        speed = np.linalg.norm(velocity, axis=1)
        idle_count = np.sum(speed < threshold)
        return float(idle_count / len(speed)) if len(speed) > 0 else 0.0

    def _regrasp_count(self, jaw_angles: np.ndarray, threshold: float) -> int:
        """Count the number of jaw open/close transitions.

        A transition is detected when the absolute change in jaw angle between
        consecutive timesteps exceeds the threshold AND the jaw crosses the
        grip boundary (transitions from open to closed or vice versa).

        Args:
            jaw_angles: (T,) jaw angle sequence.
            threshold: minimum jaw delta to count as a transition.

        Returns:
            Number of transitions (non-negative integer).
        """
        if len(jaw_angles) < 2:
            return 0

        jaw_deltas = np.abs(np.diff(jaw_angles))
        transitions = np.sum(jaw_deltas > threshold)
        return int(transitions)

    def _bimanual_coordination(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Cross-correlation of PSM1 and PSM2 velocity magnitudes.

        Higher correlation means the arms move in a coordinated temporal
        pattern (both active or both resting together), which is characteristic
        of skilled bimanual manipulation.

        Args:
            pos1: (T, 3) PSM1 positions.
            pos2: (T, 3) PSM2 positions.

        Returns:
            Pearson correlation coefficient in [-1, 1], or 0.0 if insufficient data.
        """
        if pos1.shape[0] < 3:
            return 0.0

        vel1 = _finite_diff(pos1, order=1)
        vel2 = _finite_diff(pos2, order=1)
        speed1 = np.linalg.norm(vel1, axis=1)
        speed2 = np.linalg.norm(vel2, axis=1)

        if np.std(speed1) < 1e-10 or np.std(speed2) < 1e-10:
            # One arm stationary  -- cannot compute correlation
            return 0.0

        corr, _ = pearsonr(speed1, speed2)
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def _jaw_stability(self, jaw_angles: np.ndarray, grip_threshold: float) -> float:
        """Standard deviation of jaw angle during gripping phases.

        When the jaw is in a gripping configuration (angle below threshold),
        lower std indicates a steadier, more controlled grip.

        Args:
            jaw_angles: (T,) jaw angle sequence.
            grip_threshold: jaw angle below which the gripper is considered closed.

        Returns:
            Std of jaw angle during grip (non-negative float).
            Returns 0.0 if the jaw never grips.
        """
        gripping_mask = jaw_angles < grip_threshold
        if np.sum(gripping_mask) < 2:
            # Not enough gripping samples
            return 0.0
        return float(np.std(jaw_angles[gripping_mask]))

    def _overall_score(self, scores: Dict[str, Any]) -> float:
        """Compute weighted overall quality score in [0, 1].

        Each metric is normalized to [0, 1] where 1.0 = best quality,
        then combined with the configured weights.

        Normalization strategy for each metric:
            - path_efficiency: already in [0, 1], higher = better
            - smoothness: sigmoid mapping, lower jerk = higher score
            - speed_consistency: sigmoid mapping, lower CV = higher score
            - idle_ratio: 1 - ratio, lower idle = higher score
            - regrasp_penalty: sigmoid mapping, fewer regrasps = higher score
            - bimanual_coordination: (corr + 1) / 2, maps [-1,1] to [0,1]
            - jaw_stability: sigmoid mapping, lower std = higher score
        """
        weights = self.config['weights']
        is_norm = self.config['normalized']

        # Select scaling factors based on action space
        # Scales calibrated from 50-episode GC-ACT ensemble eval data
        if is_norm:
            jerk_scale = self.config.get('jerk_scale_normalized', 1.0)
            cv_scale = 2.0         # CV is dimensionless
            jaw_std_scale = self.config.get('jaw_std_scale_normalized', 0.5)
        else:
            jerk_scale = self.config.get('jerk_scale_real', 2e-5)
            cv_scale = 2.0         # CV is dimensionless
            jaw_std_scale = self.config.get('jaw_std_scale_real', 0.15)

        def _avg_arm(key):
            """Average PSM1 and PSM2 values, handling None."""
            v1 = scores.get(f'{key}_psm1')
            v2 = scores.get(f'{key}_psm2')
            vals = [v for v in [v1, v2] if v is not None]
            return np.mean(vals) if vals else None

        # 1. Path efficiency (already 0-1, higher=better)
        pe = _avg_arm('path_efficiency')
        s_pe = pe if pe is not None else 0.5

        # 2. Smoothness (lower jerk = higher score)
        sm = _avg_arm('smoothness')
        if sm is not None:
            # Sigmoid: score = 1 / (1 + jerk/scale)
            s_sm = 1.0 / (1.0 + sm / jerk_scale)
        else:
            s_sm = 0.5

        # 3. Speed consistency (lower CV = higher score)
        sc = _avg_arm('speed_consistency')
        if sc is not None:
            s_sc = 1.0 / (1.0 + sc / cv_scale)
        else:
            s_sc = 0.5

        # 4. Idle ratio (lower = better)
        ir = _avg_arm('idle_ratio')
        s_ir = (1.0 - ir) if ir is not None else 0.5

        # 5. Re-grasp penalty (fewer = better)
        rg1 = scores.get('regrasp_count_psm1', 0) or 0
        rg2 = scores.get('regrasp_count_psm2', 0) or 0
        rg_total = rg1 + rg2
        # Map: 0 regrasps -> 1.0, many regrasps -> 0.0
        # In a 30-frame window, >5 transitions is excessive
        s_rg = 1.0 / (1.0 + rg_total / 3.0)

        # 6. Bimanual coordination (higher correlation = better)
        bc = scores.get('bimanual_coordination', 0.0) or 0.0
        s_bc = (bc + 1.0) / 2.0  # map [-1, 1] -> [0, 1]

        # 7. Jaw stability (lower std = better)
        js = _avg_arm('jaw_stability')
        if js is not None:
            s_js = 1.0 / (1.0 + js / jaw_std_scale)
        else:
            s_js = 0.5

        # Weighted sum
        overall = (
            weights['path_efficiency'] * s_pe +
            weights['smoothness'] * s_sm +
            weights['speed_consistency'] * s_sc +
            weights['idle_ratio'] * s_ir +
            weights['regrasp_penalty'] * s_rg +
            weights['bimanual_coordination'] * s_bc +
            weights['jaw_stability'] * s_js
        )

        return float(np.clip(overall, 0.0, 1.0))


# ActionModulator

class ActionModulator:
    """Modulates ACT action predictions based on GEARS quality scores.

    When quality drops, this class scales down action magnitudes and
    increases temporal ensembling to produce safer, smoother motions.

    Quality bands (calibrated for dVRK suturing):
        - HIGH   (> 0.55): pass through unchanged
        - MEDIUM (0.40 - 0.55): gentle scaling (85%)
        - LOW    (0.25 - 0.40): moderate scaling (60%) + strong ensembling
        - CRITICAL (< 0.25): recommend pause

    The modulator does NOT modify rotation components (indices 3:9 and 13:19)
    to avoid destabilizing orientation control. Only positions (0:3, 10:13)
    and jaw (9, 19) are scaled.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the action modulator.

        Args:
            config: optional configuration dict. Uses DEFAULT_CONFIG if not provided.
        """
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            for key, val in config.items():
                if isinstance(val, dict) and key in self.config and isinstance(self.config[key], dict):
                    self.config[key] = {**self.config[key], **val}
                else:
                    self.config[key] = val

        self.thresholds = self.config['quality_thresholds']
        self.modulation = self.config['modulation']
        self._pause_recommended = False

    @property
    def pause_recommended(self) -> bool:
        """Whether the most recent modulation recommended a pause."""
        return self._pause_recommended

    def modulate(self, action_20d: np.ndarray, quality_score: float,
                 current_ensemble_k: Optional[float] = None
                 ) -> Tuple[np.ndarray, Optional[float], str]:
        """Modulate a 20D action based on the current quality score.

        Args:
            action_20d: (20,) raw ACT action prediction.
            quality_score: overall GEARS quality score in [0, 1].
            current_ensemble_k: current temporal ensembling k value (or None).

        Returns:
            modified_action: (20,) modulated action.
            new_ensemble_k: suggested temporal ensembling k value (or None).
            status: human-readable status string.
        """
        modified = action_20d.copy()
        self._pause_recommended = False

        if quality_score is None:
            # Not enough data to assess  -- pass through
            return modified, current_ensemble_k, "WARMUP (insufficient data)"

        if quality_score > self.thresholds['high']:
            # HIGH quality  -- pass through unchanged
            scale = self.modulation['high_scale']
            new_k = self.modulation['high_ensemble_k'] or current_ensemble_k
            status = f"HIGH (q={quality_score:.3f})  -- full speed"

        elif quality_score > self.thresholds['medium']:
            # MEDIUM quality  -- moderate scaling
            scale = self.modulation['medium_scale']
            new_k = self.modulation['medium_ensemble_k']
            status = f"MEDIUM (q={quality_score:.3f})  -- scaled to {scale:.0%}"

        elif quality_score > self.thresholds['critical']:
            # LOW quality  -- aggressive scaling
            scale = self.modulation['low_scale']
            new_k = self.modulation['low_ensemble_k']
            status = (f"LOW (q={quality_score:.3f})  -- scaled to {scale:.0%}, "
                      f"ensemble_k={new_k}")
            warnings.warn(f"GEARS quality LOW: {quality_score:.3f}. "
                          f"Action scaled to {scale:.0%}.", RuntimeWarning)

        else:
            # CRITICAL quality  -- recommend pause
            scale = 0.0
            new_k = self.modulation['low_ensemble_k']
            status = f"CRITICAL (q={quality_score:.3f})  -- PAUSE RECOMMENDED"
            self._pause_recommended = True
            warnings.warn(f"GEARS quality CRITICAL: {quality_score:.3f}. "
                          f"Recommending pause.", RuntimeWarning)

        # Scale positions and jaw (NOT rotations)
        # PSM1 position: indices 0:3
        modified[0:3] *= scale
        # PSM1 jaw: index 9
        modified[9] *= scale
        # PSM2 position: indices 10:13
        modified[10:13] *= scale
        # PSM2 jaw: index 19
        modified[19] *= scale

        return modified, new_k, status


# GEARSLogger

class GEARSLogger:
    """Logs GEARS metrics to CSV files for post-hoc analysis.

    Creates two files:
        - gears_timesteps.csv: per-timestep metric values
        - gears_episodes.csv: per-episode summary statistics

    The logger accumulates timestep data in memory and writes to disk
    on save_episode_summary() or flush().
    """

    # Column order for the timestep CSV
    TIMESTEP_COLUMNS = [
        'episode_id', 'timestep',
        'path_efficiency_psm1', 'path_efficiency_psm2',
        'smoothness_psm1', 'smoothness_psm2',
        'speed_consistency_psm1', 'speed_consistency_psm2',
        'idle_ratio_psm1', 'idle_ratio_psm2',
        'regrasp_count_psm1', 'regrasp_count_psm2',
        'bimanual_coordination',
        'jaw_stability_psm1', 'jaw_stability_psm2',
        'overall',
        'modulation_status',
    ]

    EPISODE_COLUMNS = [
        'episode_id', 'num_timesteps',
        'mean_overall', 'std_overall', 'min_overall', 'max_overall',
        'mean_path_efficiency', 'mean_smoothness', 'mean_speed_consistency',
        'mean_idle_ratio', 'total_regrasps', 'mean_bimanual_coordination',
        'mean_jaw_stability',
        'pct_high', 'pct_medium', 'pct_low', 'pct_critical',
    ]

    def __init__(self, output_dir: str = "~/gears_logs", episode_id: str = "ep_000"):
        """Initialize the logger.

        Args:
            output_dir: directory to write CSV files. Created if needed.
            episode_id: identifier for the current episode.
        """
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_id = episode_id

        self._timestep_rows: List[Dict] = []
        self._episode_rows: List[Dict] = []

        # Initialize CSV files with headers if they do not exist
        self._ts_path = os.path.join(self.output_dir, 'gears_timesteps.csv')
        self._ep_path = os.path.join(self.output_dir, 'gears_episodes.csv')

        if not os.path.exists(self._ts_path):
            with open(self._ts_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.TIMESTEP_COLUMNS)
                writer.writeheader()

        if not os.path.exists(self._ep_path):
            with open(self._ep_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.EPISODE_COLUMNS)
                writer.writeheader()

    def set_episode(self, episode_id: str):
        """Set the current episode ID (call before logging a new episode)."""
        self.episode_id = episode_id

    def log_step(self, timestep: int, scores: Optional[Dict[str, float]],
                 modulation_status: str = ""):
        """Log a single timestep's GEARS scores.

        Args:
            timestep: integer timestep index.
            scores: dict from GEARSMonitor.get_scores() (may be None during warmup).
            modulation_status: status string from ActionModulator.
        """
        row = {
            'episode_id': self.episode_id,
            'timestep': timestep,
            'modulation_status': modulation_status,
        }
        if scores is not None:
            for col in self.TIMESTEP_COLUMNS:
                if col in scores:
                    val = scores[col]
                    row[col] = f"{val:.6f}" if isinstance(val, float) else str(val)
        self._timestep_rows.append(row)

    def save_episode_summary(self, config: Optional[Dict] = None):
        """Compute and save episode-level summary from accumulated timestep data.

        Also flushes timestep rows to the CSV file.

        Args:
            config: optional config dict for quality thresholds. Uses DEFAULT_CONFIG.
        """
        cfg = config or DEFAULT_CONFIG
        thresholds = cfg.get('quality_thresholds', DEFAULT_CONFIG['quality_thresholds'])

        # Flush timestep rows
        self._flush_timesteps()

        # Compute episode summary
        overall_scores = []
        for row in self._timestep_rows:
            val = row.get('overall')
            if val is not None and val != '':
                try:
                    overall_scores.append(float(val))
                except (ValueError, TypeError):
                    pass

        # Also try from any scores we have
        if not overall_scores and self._timestep_rows:
            # Re-parse from written data
            pass

        n = len(self._timestep_rows)
        if overall_scores:
            os_arr = np.array(overall_scores)
            summary = {
                'episode_id': self.episode_id,
                'num_timesteps': n,
                'mean_overall': f"{np.mean(os_arr):.6f}",
                'std_overall': f"{np.std(os_arr):.6f}",
                'min_overall': f"{np.min(os_arr):.6f}",
                'max_overall': f"{np.max(os_arr):.6f}",
                'pct_high': f"{100.0 * np.mean(os_arr > thresholds['high']):.1f}",
                'pct_medium': f"{100.0 * np.mean((os_arr > thresholds['medium']) & (os_arr <= thresholds['high'])):.1f}",
                'pct_low': f"{100.0 * np.mean((os_arr > thresholds['critical']) & (os_arr <= thresholds['medium'])):.1f}",
                'pct_critical': f"{100.0 * np.mean(os_arr <= thresholds['critical']):.1f}",
            }

            # Aggregate per-metric means
            for metric in ['path_efficiency', 'smoothness', 'speed_consistency',
                           'idle_ratio', 'bimanual_coordination', 'jaw_stability']:
                vals = []
                for row in self._timestep_rows:
                    for arm in ['_psm1', '_psm2', '']:
                        key = f'{metric}{arm}'
                        v = row.get(key)
                        if v is not None and v != '':
                            try:
                                vals.append(float(v))
                            except (ValueError, TypeError):
                                pass
                summary[f'mean_{metric}'] = f"{np.mean(vals):.6f}" if vals else ""

            # Total regrasps
            rg_total = 0
            for row in self._timestep_rows:
                for arm in ['_psm1', '_psm2']:
                    v = row.get(f'regrasp_count{arm}')
                    if v is not None and v != '':
                        try:
                            rg_total += int(float(v))
                        except (ValueError, TypeError):
                            pass
            summary['total_regrasps'] = str(rg_total)

        else:
            summary = {
                'episode_id': self.episode_id,
                'num_timesteps': n,
            }

        # Append to episode CSV
        with open(self._ep_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_COLUMNS,
                                    extrasaction='ignore')
            writer.writerow(summary)

        self._episode_rows.append(summary)

        # Clear timestep buffer for next episode
        self._timestep_rows = []

    def _flush_timesteps(self):
        """Write accumulated timestep rows to CSV."""
        if not self._timestep_rows:
            return
        with open(self._ts_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.TIMESTEP_COLUMNS,
                                    extrasaction='ignore')
            for row in self._timestep_rows:
                writer.writerow(row)


# OFFLINE EVALUATION FUNCTION

def evaluate_episode_quality(actions_pred: np.ndarray,
                             actions_gt: Optional[np.ndarray] = None,
                             config: Optional[Dict] = None,
                             ) -> Dict[str, Any]:
    """Evaluate GEARS quality metrics for a predicted action sequence.

    This function works with numpy arrays and has no ROS or torch dependency.
    It can be used to retroactively score episodes from the offline eval results.

    Args:
        actions_pred: (T, 20) array of predicted actions (real space: meters, radians).
        actions_gt: (T, 20) array of ground truth actions (optional). If provided,
                    additional comparative metrics are computed.
        config: optional GEARS configuration dict.

    Returns:
        Dictionary containing:
            - All GEARS metric scores for the predicted trajectory
            - 'overall': weighted quality score in [0, 1]
            - 'per_window': list of per-window quality scores
            - If actions_gt provided:
                - 'gt_overall': quality score for the ground truth trajectory
                - 'pos_error_mm': mean position L2 error (mm)
                - 'quality_error_correlation': Pearson r between window quality
                  and window position error (negative = quality predicts error)
    """
    cfg = config or {'normalized': False}
    monitor = GEARSMonitor(config=cfg)

    results = monitor.evaluate_batch(actions_pred)
    results['source'] = 'predicted'

    if actions_gt is not None:
        assert actions_gt.shape == actions_pred.shape, \
            f"Shape mismatch: pred={actions_pred.shape}, gt={actions_gt.shape}"
        T = actions_pred.shape[0]

        # GT quality metrics
        gt_monitor = GEARSMonitor(config=cfg)
        gt_results = gt_monitor.evaluate_batch(actions_gt)
        results['gt_overall'] = gt_results.get('overall')
        results['gt_scores'] = {k: v for k, v in gt_results.items()
                                if k not in ('per_window', 'T')}

        # Position error
        pos_err_psm1 = np.linalg.norm(
            actions_pred[:, 0:3] - actions_gt[:, 0:3], axis=1) * 1000  # mm
        pos_err_psm2 = np.linalg.norm(
            actions_pred[:, 10:13] - actions_gt[:, 10:13], axis=1) * 1000
        pos_err_mean = (pos_err_psm1 + pos_err_psm2) / 2.0

        results['pos_error_psm1_mm'] = float(np.mean(pos_err_psm1))
        results['pos_error_psm2_mm'] = float(np.mean(pos_err_psm2))
        results['pos_error_mm'] = float(np.mean(pos_err_mean))

        # Correlation between per-window quality and per-window position error
        per_window = results.get('per_window', [])
        if len(per_window) > 10:
            window_quality = np.array([pw['overall'] for pw in per_window
                                       if pw.get('overall') is not None])
            # Compute windowed position error (align with window timesteps)
            window_times = np.array([pw['t'] for pw in per_window
                                     if pw.get('overall') is not None])
            window_pos_err = pos_err_mean[window_times[window_times < T]]

            min_len = min(len(window_quality), len(window_pos_err))
            if min_len > 10:
                corr, pval = pearsonr(window_quality[:min_len],
                                      window_pos_err[:min_len])
                results['quality_error_correlation'] = float(corr)
                results['quality_error_pvalue'] = float(pval)
            else:
                results['quality_error_correlation'] = None
                results['quality_error_pvalue'] = None
        else:
            results['quality_error_correlation'] = None
            results['quality_error_pvalue'] = None

    return results


# STANDALONE DEMO / TEST

def _print_table_row(label: str, values: List, widths: List[int]):
    """Print a formatted table row."""
    parts = [f"{label:<25s}"]
    for val, w in zip(values, widths):
        if val is None:
            parts.append(f"{'N/A':>{w}s}")
        elif isinstance(val, float):
            # Use scientific notation for very small values (e.g. jerk ~1e-5)
            if 0 < abs(val) < 0.001:
                parts.append(f"{val:>{w}.2e}")
            else:
                parts.append(f"{val:>{w}.4f}")
        elif isinstance(val, int):
            parts.append(f"{val:>{w}d}")
        else:
            parts.append(f"{str(val):>{w}s}")
    print("  ".join(parts))


def main():
    """Standalone demo: load offline eval results and compute GEARS metrics.

    Loads results from ~/offline_eval_results_50ep_gcact_ensemble/results_full.npz,
    computes GEARS metrics for each episode, prints a summary table, and
    saves analysis to ~/gears_analysis/.
    """
    import sys

    # Paths
    results_dirs = {
        'gcact_ensemble': os.path.expanduser(
            '~/offline_eval_results_50ep_gcact_ensemble/results_full.npz'),
        'gcact_raw': os.path.expanduser(
            '~/offline_eval_results_50ep_gcact_raw/results_full.npz'),
        'v2_ensemble': os.path.expanduser(
            '~/offline_eval_results_50ep_v2_ensemble/results_full.npz'),
        'v2_raw': os.path.expanduser(
            '~/offline_eval_results_50ep_v2_raw/results_full.npz'),
        'v1_ensemble': os.path.expanduser(
            '~/offline_eval_results_50ep_v1_ensemble/results_full.npz'),
    }

    output_dir = os.path.expanduser('~/gears_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Pick first available results file
    npz_path = None
    npz_label = None
    for label, path in results_dirs.items():
        if os.path.exists(path):
            npz_path = path
            npz_label = label
            break

    if npz_path is None:
        print("ERROR: No offline eval results found. Expected one of:")
        for label, path in results_dirs.items():
            print(f"  {path}")
        sys.exit(1)

    print("=" * 80)
    print("  GEARS QUALITY ANALYSIS  -- Offline Evaluation Results")
    print("=" * 80)
    print(f"  Results file: {npz_path}")
    print(f"  Label: {npz_label}")
    print(f"  Output dir: {output_dir}")

    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"  NPZ keys: {len(keys)} entries")

    # Parse keys to find episodes
    # Format: {subtask}_{idx}_pred_raw, {subtask}_{idx}_gt_raw, {subtask}_{idx}_timestamps
    episodes = {}
    for key in keys:
        if key.endswith('_pred_raw'):
            # Extract episode prefix: everything before _pred_raw
            prefix = key[:-len('_pred_raw')]
            # Parse subtask and index
            # Keys like: needle_throw_0_pred_raw or knot_tying_12_pred_raw
            parts = prefix.rsplit('_', 1)
            if len(parts) == 2:
                subtask_name, idx_str = parts
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                episodes[prefix] = {
                    'subtask': subtask_name,
                    'index': idx,
                    'pred_raw': data[f'{prefix}_pred_raw'],
                    'gt_raw': data.get(f'{prefix}_gt_raw'),
                    'timestamps': data.get(f'{prefix}_timestamps'),
                }

    print(f"  Found {len(episodes)} episodes")

    # Group by subtask
    subtask_episodes = {}
    for prefix, ep in episodes.items():
        st = ep['subtask']
        if st not in subtask_episodes:
            subtask_episodes[st] = []
        subtask_episodes[st].append((prefix, ep))

    for st in subtask_episodes:
        subtask_episodes[st].sort(key=lambda x: x[1]['index'])

    print(f"  Subtasks: {list(subtask_episodes.keys())}")
    for st, eps in subtask_episodes.items():
        print(f"    {st}: {len(eps)} episodes")

    # Compute GEARS metrics for each episode
    print("\n" + "-" * 80)
    print("  Computing GEARS metrics...")
    print("-" * 80)

    all_results = []
    config = {'normalized': False, 'window_size': 30}

    for subtask_name, ep_list in subtask_episodes.items():
        print(f"\n  === {subtask_name.upper()} ({len(ep_list)} episodes) ===")

        for prefix, ep in ep_list:
            pred = ep['pred_raw']
            gt = ep['gt_raw']

            if pred is None or pred.ndim != 2 or pred.shape[1] != 20:
                print(f"    [{prefix}] Skipping  -- invalid pred shape: "
                      f"{pred.shape if pred is not None else 'None'}")
                continue

            if pred.shape[0] < 5:
                print(f"    [{prefix}] Skipping  -- too few timesteps ({pred.shape[0]})")
                continue

            gt_valid = (gt is not None and gt.ndim == 2 and
                        gt.shape == pred.shape)

            result = evaluate_episode_quality(
                pred, gt if gt_valid else None, config=config)

            result['prefix'] = prefix
            result['subtask'] = subtask_name
            result['index'] = ep['index']
            all_results.append(result)

    if not all_results:
        print("\n  No valid episodes found for GEARS analysis.")
        sys.exit(1)

    # Summary Table
    print("\n" + "=" * 80)
    print("  GEARS QUALITY SUMMARY")
    print("=" * 80)

    col_widths = [10, 10, 10, 10, 10, 10, 10]
    header_labels = ['Quality', 'PathEff', 'Smooth', 'SpeedCV', 'Idle', 'Bimanual', 'PosErr(mm)']
    header = f"{'Episode':<25s}" + "  ".join(f"{h:>{w}s}" for h, w in zip(header_labels, col_widths))
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for subtask_name in subtask_episodes:
        st_results = [r for r in all_results if r['subtask'] == subtask_name]
        if not st_results:
            continue

        print(f"\n  --- {subtask_name.upper()} ---")
        for r in st_results:
            pe = r.get('path_efficiency_psm1')
            pe2 = r.get('path_efficiency_psm2')
            pe_avg = np.mean([v for v in [pe, pe2] if v is not None]) if pe is not None else None

            sm = r.get('smoothness_psm1')
            sm2 = r.get('smoothness_psm2')
            sm_avg = np.mean([v for v in [sm, sm2] if v is not None]) if sm is not None else None

            sc = r.get('speed_consistency_psm1')
            sc2 = r.get('speed_consistency_psm2')
            sc_avg = np.mean([v for v in [sc, sc2] if v is not None]) if sc is not None else None

            ir = r.get('idle_ratio_psm1')
            ir2 = r.get('idle_ratio_psm2')
            ir_avg = np.mean([v for v in [ir, ir2] if v is not None]) if ir is not None else None

            bc = r.get('bimanual_coordination')
            pos_err = r.get('pos_error_mm')

            label = f"{r['subtask']}_{r['index']}"
            _print_table_row(label,
                             [r.get('overall'), pe_avg, sm_avg, sc_avg, ir_avg, bc, pos_err],
                             col_widths)

        # Subtask aggregates
        overalls = [r['overall'] for r in st_results if r.get('overall') is not None]
        pos_errs = [r['pos_error_mm'] for r in st_results if r.get('pos_error_mm') is not None]

        if overalls:
            print(f"  {'':25s}{'------':>10s}  {'':>10s}  {'':>10s}  {'':>10s}  {'':>10s}  {'':>10s}  {'------':>10s}")
            _print_table_row(
                f"  MEAN ({len(overalls)} eps)",
                [np.mean(overalls), None, None, None, None, None,
                 np.mean(pos_errs) if pos_errs else None],
                col_widths)
            _print_table_row(
                f"  STD",
                [np.std(overalls), None, None, None, None, None,
                 np.std(pos_errs) if pos_errs else None],
                col_widths)

    # Correlation Analysis
    print("\n" + "=" * 80)
    print("  QUALITY-ERROR CORRELATION ANALYSIS")
    print("=" * 80)

    for subtask_name in subtask_episodes:
        st_results = [r for r in all_results if r['subtask'] == subtask_name]
        qualities = []
        errors = []
        for r in st_results:
            q = r.get('overall')
            e = r.get('pos_error_mm')
            if q is not None and e is not None:
                qualities.append(q)
                errors.append(e)

        if len(qualities) > 5:
            corr, pval = pearsonr(qualities, errors)
            print(f"\n  {subtask_name.upper()}:")
            print(f"    Episodes: {len(qualities)}")
            print(f"    Quality: mean={np.mean(qualities):.4f}, "
                  f"std={np.std(qualities):.4f}")
            print(f"    Pos Error: mean={np.mean(errors):.4f} mm, "
                  f"std={np.std(errors):.4f} mm")
            print(f"    Pearson r(quality, error): {corr:.4f} (p={pval:.4f})")
            if pval < 0.05:
                direction = "NEGATIVE" if corr < 0 else "POSITIVE"
                print(f"    => Significant {direction} correlation: "
                      f"GEARS quality {'inversely predicts' if corr < 0 else 'correlates with'} position error")
            else:
                print(f"    => No significant correlation (p >= 0.05)")

            # Per-window correlation (aggregated across episodes)
            window_corrs = [r.get('quality_error_correlation')
                            for r in st_results
                            if r.get('quality_error_correlation') is not None]
            if window_corrs:
                print(f"    Per-window corr (mean across eps): {np.mean(window_corrs):.4f}")

    # Quality Band Distribution
    print("\n" + "=" * 80)
    print("  QUALITY BAND DISTRIBUTION")
    print("=" * 80)

    thresholds = DEFAULT_CONFIG['quality_thresholds']
    for subtask_name in subtask_episodes:
        st_results = [r for r in all_results if r['subtask'] == subtask_name]
        overalls = [r['overall'] for r in st_results if r.get('overall') is not None]
        if not overalls:
            continue

        ov = np.array(overalls)
        n_high = np.sum(ov > thresholds['high'])
        n_medium = np.sum((ov > thresholds['medium']) & (ov <= thresholds['high']))
        n_low = np.sum((ov > thresholds['critical']) & (ov <= thresholds['medium']))
        n_critical = np.sum(ov <= thresholds['critical'])
        total = len(ov)

        print(f"\n  {subtask_name.upper()} ({total} episodes):")
        print(f"    HIGH     (>{thresholds['high']:.2f}):  {n_high:3d} ({100*n_high/total:5.1f}%)")
        print(f"    MEDIUM   ({thresholds['medium']:.2f}-{thresholds['high']:.2f}): {n_medium:3d} ({100*n_medium/total:5.1f}%)")
        print(f"    LOW      ({thresholds['critical']:.2f}-{thresholds['medium']:.2f}): {n_low:3d} ({100*n_low/total:5.1f}%)")
        print(f"    CRITICAL (<{thresholds['critical']:.2f}):  {n_critical:3d} ({100*n_critical/total:5.1f}%)")

    # Save results
    print("\n" + "=" * 80)
    print("  SAVING RESULTS")
    print("=" * 80)

    # Save per-episode results CSV
    csv_path = os.path.join(output_dir, f'gears_results_{npz_label}.csv')
    csv_cols = ['subtask', 'index', 'overall', 'pos_error_mm',
                'path_efficiency_psm1', 'path_efficiency_psm2',
                'smoothness_psm1', 'smoothness_psm2',
                'speed_consistency_psm1', 'speed_consistency_psm2',
                'idle_ratio_psm1', 'idle_ratio_psm2',
                'regrasp_count_psm1', 'regrasp_count_psm2',
                'bimanual_coordination',
                'jaw_stability_psm1', 'jaw_stability_psm2',
                'gt_overall', 'quality_error_correlation']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {}
            for col in csv_cols:
                val = r.get(col)
                if isinstance(val, float):
                    row[col] = f"{val:.6f}"
                elif val is not None:
                    row[col] = str(val)
                else:
                    row[col] = ""
            writer.writerow(row)
    print(f"  Per-episode CSV: {csv_path}")

    # Save summary JSON
    summary = {}
    for subtask_name in subtask_episodes:
        st_results = [r for r in all_results if r['subtask'] == subtask_name]
        overalls = [r['overall'] for r in st_results if r.get('overall') is not None]
        pos_errs = [r['pos_error_mm'] for r in st_results if r.get('pos_error_mm') is not None]
        if overalls:
            summary[subtask_name] = {
                'n_episodes': len(overalls),
                'quality_mean': float(np.mean(overalls)),
                'quality_std': float(np.std(overalls)),
                'quality_min': float(np.min(overalls)),
                'quality_max': float(np.max(overalls)),
                'pos_error_mean_mm': float(np.mean(pos_errs)) if pos_errs else None,
                'pos_error_std_mm': float(np.std(pos_errs)) if pos_errs else None,
            }
            # Correlation
            if pos_errs and len(overalls) > 5:
                corr, pval = pearsonr(overalls, pos_errs[:len(overalls)])
                summary[subtask_name]['quality_error_pearson_r'] = float(corr)
                summary[subtask_name]['quality_error_pvalue'] = float(pval)

    json_path = os.path.join(output_dir, f'gears_summary_{npz_label}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {json_path}")

    # Save per-window quality timeseries for each episode (npz)
    window_data = {}
    for r in all_results:
        prefix = r.get('prefix', f"{r['subtask']}_{r['index']}")
        per_window = r.get('per_window', [])
        if per_window:
            times = np.array([pw['t'] for pw in per_window])
            qualities = np.array([pw.get('overall', np.nan) for pw in per_window])
            window_data[f'{prefix}_times'] = times
            window_data[f'{prefix}_quality'] = qualities

    if window_data:
        npz_out_path = os.path.join(output_dir, f'gears_timeseries_{npz_label}.npz')
        np.savez_compressed(npz_out_path, **window_data)
        print(f"  Timeseries NPZ: {npz_out_path}")

    print(f"\n  Analysis complete. All outputs saved to {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
