#!/usr/bin/env python3
"""Smoke test for GC-ACT: verify model construction, dataset, and forward pass."""
import sys
import os
# Override argv for DETR argparse (required args must be present)
sys.argv = ['test', '--ckpt_dir', '/tmp/test_gcact', '--policy_class', 'ACT',
            '--task_name', 'test', '--seed', '0', '--num_epochs', '1',
            '--use_gesture', '--gesture_dim', '10']
os.environ['PATH_TO_SUTUREBOT'] = os.path.expanduser('~/SutureBot')
os.environ['PATH_TO_DATASET'] = os.path.expanduser('~/data')

sys.path.insert(0, os.path.expanduser('~/SutureBot/src/act'))
sys.path.insert(0, os.path.expanduser('~/SutureBot/src'))

import torch
import numpy as np

print("=" * 60)
print("  GC-ACT Smoke Test")
print("=" * 60)

# Test 1: Model construction with gesture conditioning
print("\n[Test 1] Model construction with use_gesture=True...")
from policy import ACTPolicy

policy_config = {
    "lr": 1e-4,
    "num_queries": 60,
    "action_dim": 20,
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",  # Use resnet18 for faster test
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["left", "left_wrist", "right_wrist"],
    "multi_gpu": False,
    "use_gesture": True,
    "gesture_dim": 10,
    # Required args for DETR parser
    "ckpt_dir": "/tmp/test_gcact",
    "policy_class": "ACT",
    "task_name": ["test"],
    "seed": 0,
    "num_epochs": 1,
}

policy = ACTPolicy(policy_config)
print(f"  Model created successfully")

# Check that gesture layers exist
model = policy.model
assert hasattr(model, 'use_gesture') and model.use_gesture, "use_gesture not set"
assert hasattr(model, 'gesture_embed_proj'), "gesture_embed_proj missing"
print(f"  gesture_embed_proj: {model.gesture_embed_proj}")
print(f"  additional_pos_embed: {model.additional_pos_embed.weight.shape} (should be [3, 512])")
assert model.additional_pos_embed.weight.shape[0] == 3, f"Expected 3 pos embeddings, got {model.additional_pos_embed.weight.shape[0]}"
print("  PASSED")

# Test 2: Forward pass with gesture embedding
print("\n[Test 2] Forward pass with gesture embedding...")
policy.cuda()
bs = 2
qpos = torch.zeros(bs, 20).cuda()
image = torch.randn(bs, 3, 3, 360, 480).cuda()  # 3 cameras
actions = torch.randn(bs, 60, 20).cuda()
is_pad = torch.zeros(bs, 60).bool().cuda()

# Create a one-hot gesture embedding (G3 = index 1)
gesture_emb = torch.zeros(bs, 10).cuda()
gesture_emb[:, 1] = 1.0  # G3

loss_dict = policy(qpos, image, actions, is_pad, gesture_embedding=gesture_emb)
print(f"  Loss: {loss_dict['loss'].item():.4f}")
print(f"  L1:   {loss_dict['l1'].item():.4f}")
print(f"  KL:   {loss_dict['kl'].item():.4f}")
print("  PASSED")

# Test 3: Inference with gesture embedding (no actions)
print("\n[Test 3] Inference with gesture embedding...")
with torch.no_grad():
    a_hat = policy(qpos, image, gesture_embedding=gesture_emb)
print(f"  Output shape: {a_hat.shape} (should be [{bs}, 60, 20])")
assert a_hat.shape == (bs, 60, 20), f"Wrong shape: {a_hat.shape}"
print("  PASSED")

# Test 4: Forward pass without gesture (backward compatible)
print("\n[Test 4] Forward pass without gesture (zeros)...")
loss_dict_no_gesture = policy(qpos, image, actions, is_pad)
print(f"  Loss: {loss_dict_no_gesture['loss'].item():.4f}")
print("  PASSED")

# Test 5: Dataset gesture label loading
print("\n[Test 5] Dataset gesture label loading...")
from generic_dataset import EpisodicDatasetDvrkGeneric, GESTURE_LABELS, GESTURE_TO_IDX

print(f"  GESTURE_LABELS: {GESTURE_LABELS}")
print(f"  GESTURE_TO_IDX: {GESTURE_TO_IDX}")

# Test the parsing methods
labels_dir = os.path.expanduser('~/data/labels')
if os.path.exists(labels_dir):
    # Create a temporary dataset instance just to test label loading
    dataset = EpisodicDatasetDvrkGeneric.__new__(EpisodicDatasetDvrkGeneric)
    dataset.gesture_labels = {}
    dataset._load_gesture_labels(labels_dir)
    print(f"  Loaded {len(dataset.gesture_labels)} episode gesture labels")

    # Show a sample
    if dataset.gesture_labels:
        sample_key = list(dataset.gesture_labels.keys())[0]
        sample_gestures = dataset.gesture_labels[sample_key]
        print(f"  Sample: {sample_key}")
        print(f"  Gestures: {sample_gestures[:3]}...")

        # Test frame lookup
        frame_gesture = dataset._get_gesture_at_frame(sample_gestures, sample_gestures[0][0])
        print(f"  Gesture at frame {sample_gestures[0][0]}: {frame_gesture}")

        # Test one-hot
        onehot = dataset._gesture_to_onehot(frame_gesture)
        print(f"  One-hot: {onehot}")
        assert onehot.sum() == 1.0, "One-hot should sum to 1"
    print("  PASSED")
else:
    print(f"  SKIPPED (labels_dir not found at {labels_dir})")

# Test 6: Load v2 checkpoint with shape-mismatch handling
print("\n[Test 6] Load v2 checkpoint with shape-mismatch handling...")
v2_ckpt_path = os.path.expanduser("~/checkpoints/act_nt_v2/policy_best.ckpt")
if os.path.exists(v2_ckpt_path):
    checkpoint = torch.load(v2_ckpt_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Create a fresh GC-ACT policy with EfficientNet-B3 (matches v2 ckpt)
    policy_config_effnet = dict(policy_config)
    policy_config_effnet["backbone"] = "efficientnet_b3"
    policy2 = ACTPolicy(policy_config_effnet)

    # Filter out shape mismatches (same logic as imitate_episodes.py)
    model_state = policy2.state_dict()
    filtered_state = {}
    skipped_keys = []
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped_keys.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
        else:
            filtered_state[k] = v
    if skipped_keys:
        print(f"  Skipped keys: {skipped_keys}")

    loading_status = policy2.load_state_dict(filtered_state, strict=False)
    print(f"  Missing keys: {loading_status.missing_keys}")
    print(f"  Unexpected keys: {loading_status.unexpected_keys}")

    # The missing keys should include gesture layers and any skipped shape-mismatch keys
    expected_missing = ['model.gesture_embed_proj.weight', 'model.gesture_embed_proj.bias']
    for key in expected_missing:
        assert key in loading_status.missing_keys, f"Expected {key} in missing keys"

    # Copy first 2 rows of additional_pos_embed from checkpoint
    if 'model.additional_pos_embed.weight' in state_dict:
        old_pos_embed = state_dict['model.additional_pos_embed.weight']
        if old_pos_embed.shape[0] == 2:
            with torch.no_grad():
                policy2.model.additional_pos_embed.weight[:2] = old_pos_embed
            print("  Copied 2/3 rows of additional_pos_embed from checkpoint")
    print("  PASSED")
else:
    print(f"  SKIPPED (v2 checkpoint not found at {v2_ckpt_path})")

print("\n" + "=" * 60)
print("  ALL SMOKE TESTS PASSED")
print("=" * 60)
