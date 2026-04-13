#!/usr/bin/env python3
# verify_deployment.py  -- Pre-Flight Check for JHU Deployment
# Run this on the JHU machine BEFORE attempting any robot trials. Performs a
# 10-point verification: checks GPU availability, loads all 3 model checkpoints,
# runs a dummy inference to verify output shapes, and reports timing. If all
# checks pass, the system is ready for real robot trials. No ROS or robot
# connection needed  -- this is a pure software check.
"""
verify_deployment.py

Run on the JHU deployment machine to verify that the SutureBot ACT inference
system is correctly set up and ready for use on the real dVRK.

This script does NOT require ROS  -- it is designed for pre-robot verification.
It checks:
  1. Conda environment exists and is active
  2. GPU is available and has sufficient memory
  3. All required Python packages can be imported
  4. All required files exist (checkpoints, scripts, model code)
  5. Each checkpoint loads successfully and has the expected structure
  6. A full dummy inference pass works end-to-end (random images -> actions)

Usage:
    conda activate act
    python verify_deployment.py

    # Specify a custom deploy directory:
    python verify_deployment.py --deploy_dir /path/to/suturebot_deploy

    # Verbose mode (print model details):
    python verify_deployment.py --verbose
"""

import os
import sys
import argparse
import time
import traceback


# Test framework

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.message = ""
        self.warnings = []

    def pass_(self, msg=""):
        self.passed = True
        self.message = msg

    def fail(self, msg):
        self.passed = False
        self.message = msg

    def warn(self, msg):
        self.warnings.append(msg)


results = []


def run_test(name, func, *args, **kwargs):
    """Run a test function and capture results."""
    result = TestResult(name)
    try:
        func(result, *args, **kwargs)
    except Exception as e:
        result.fail(f"Exception: {e}\n{traceback.format_exc()}")
    results.append(result)

    status = "PASS" if result.passed else "FAIL"
    symbol = " [+]" if result.passed else " [-]"
    print(f"  {symbol} {name}: {status}")
    if result.message:
        for line in result.message.split("\n"):
            if line.strip():
                print(f"       {line}")
    for w in result.warnings:
        print(f"       WARNING: {w}")

    return result.passed


# Individual tests

def test_conda_env(result):
    """Check that we are running inside the 'act' conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    if conda_env == "act":
        result.pass_(f"Active environment: {conda_env} ({conda_prefix})")
    elif conda_env:
        result.warn(f"Active environment is '{conda_env}', expected 'act'")
        result.pass_(f"Conda is active but environment is '{conda_env}' (not 'act')")
    else:
        result.fail("No conda environment is active. Run: conda activate act")


def test_python_version(result):
    """Check Python version is 3.8.x."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor == 8:
        result.pass_(f"Python {version_str}")
    elif v.major == 3 and v.minor >= 7:
        result.warn(f"Python {version_str} (expected 3.8.x, should still work)")
        result.pass_(f"Python {version_str}")
    else:
        result.fail(f"Python {version_str}  -- requires Python 3.8.x")


def test_gpu(result):
    """Check that CUDA GPU is available."""
    try:
        import torch
    except ImportError:
        result.fail("Cannot import torch")
        return

    if not torch.cuda.is_available():
        result.fail("torch.cuda.is_available() == False. Check NVIDIA driver and CUDA.")
        return

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    cuda_version = torch.version.cuda

    result.pass_(f"{gpu_name} ({gpu_mem:.1f} GB), CUDA {cuda_version}, {gpu_count} GPU(s)")

    # ACT models need ~2 GB GPU memory for inference
    if gpu_mem < 2.0:
        result.warn(f"GPU has only {gpu_mem:.1f} GB  -- may be insufficient for 3 models")


def test_imports(result):
    """Check all required Python packages can be imported."""
    required = {
        "torch": None,
        "torchvision": None,
        "numpy": None,
        "scipy": None,
        "scipy.spatial.transform": None,
        "sklearn.preprocessing": None,
        "cv2": None,
        "einops": None,
        "matplotlib": None,
        "IPython": None,
    }

    missing = []
    versions = []
    for pkg_name in required:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "?")
            versions.append(f"{pkg_name}={ver}")
        except ImportError as e:
            missing.append(f"{pkg_name}: {e}")

    if missing:
        result.fail("Missing packages:\n" + "\n".join(f"  {m}" for m in missing))
    else:
        result.pass_(", ".join(versions[:5]) + " ...")


def test_files_exist(result, deploy_dir):
    """Check all required files exist in the deploy directory."""
    required_files = [
        # Checkpoints
        "checkpoints/act_np/policy_best.ckpt",
        "checkpoints/act_nt/policy_best.ckpt",
        "checkpoints/act_kt/policy_best.ckpt",
        # Main inference script
        "src/act/chained_dvrk_inference.py",
        # Policy
        "src/act/policy.py",
        # DETR model code
        "src/act/detr/main.py",
        "src/act/detr/models/__init__.py",
        "src/act/detr/models/detr_vae.py",
        "src/act/detr/models/backbone.py",
        "src/act/detr/models/transformer.py",
        "src/act/detr/models/position_encoding.py",
        # dVRK scripts
        "src/act/dvrk_scripts/constants_dvrk.py",
        "src/act/dvrk_scripts/dvrk_control.py",
        # ROS topics
        "src/act/rostopics.py",
    ]

    missing = []
    found = 0
    for f in required_files:
        full_path = os.path.join(deploy_dir, f)
        if os.path.exists(full_path):
            found += 1
        else:
            missing.append(f)

    if missing:
        result.fail(
            f"{found}/{len(required_files)} files found. Missing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )
    else:
        result.pass_(f"All {found} required files present")


def test_checkpoint_load(result, ckpt_path, label):
    """Load a single checkpoint and verify its structure."""
    import torch

    if not os.path.exists(ckpt_path):
        result.fail(f"Checkpoint not found: {ckpt_path}")
        return

    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        ckpt_type = "model_state_dict"
    else:
        state_dict = ckpt
        ckpt_type = "raw state_dict"

    num_params = sum(p.numel() for p in state_dict.values())
    num_keys = len(state_dict)

    # Verify expected keys are present (backbone, transformer, etc.)
    key_prefixes = set()
    for k in state_dict.keys():
        prefix = k.split(".")[0]
        key_prefixes.add(prefix)

    expected_prefixes = {"backbones", "transformer", "encoder"}
    found_prefixes = expected_prefixes.intersection(key_prefixes)

    if len(found_prefixes) < 2:
        result.warn(
            f"Expected key prefixes {expected_prefixes}, found {key_prefixes}"
        )

    result.pass_(
        f"{label}: {file_size_mb:.0f} MB, {num_keys} keys, "
        f"{num_params/1e6:.1f}M params ({ckpt_type})"
    )


def test_model_build(result, deploy_dir):
    """Test that ACTPolicy can be instantiated from the deployed code."""
    import torch

    # Add deploy source to path
    src_act_dir = os.path.join(deploy_dir, "src", "act")
    if src_act_dir not in sys.path:
        sys.path.insert(0, src_act_dir)

    # Override sys.argv for detr argparser
    saved_argv = sys.argv
    sys.argv = [
        "act", "--task_name", "needle_pickup_all",
        "--ckpt_dir", "/tmp", "--policy_class", "ACT",
        "--seed", "0", "--num_epochs", "1",
        "--kl_weight", "1", "--chunk_size", "60",
        "--hidden_dim", "512", "--dim_feedforward", "3200",
        "--lr", "1e-5", "--batch_size", "8",
        "--image_encoder", "resnet18",
    ]

    try:
        from policy import ACTPolicy

        policy_config = {
            "lr": 1e-5,
            "num_queries": 60,
            "action_dim": 20,
            "kl_weight": 1,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["left", "left_wrist", "right_wrist"],
            "multi_gpu": False,
        }
        policy = ACTPolicy(policy_config)
        num_params = sum(p.numel() for p in policy.parameters())
        result.pass_(f"ACTPolicy built successfully ({num_params/1e6:.1f}M parameters)")
    finally:
        sys.argv = saved_argv


def test_dummy_inference(result, deploy_dir):
    """Run a full dummy inference pass: random images -> action chunk."""
    import torch
    import numpy as np

    src_act_dir = os.path.join(deploy_dir, "src", "act")
    if src_act_dir not in sys.path:
        sys.path.insert(0, src_act_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Override sys.argv for detr argparser
    saved_argv = sys.argv
    sys.argv = [
        "act", "--task_name", "needle_pickup_all",
        "--ckpt_dir", "/tmp", "--policy_class", "ACT",
        "--seed", "0", "--num_epochs", "1",
        "--kl_weight", "1", "--chunk_size", "60",
        "--hidden_dim", "512", "--dim_feedforward", "3200",
        "--lr", "1e-5", "--batch_size", "8",
        "--image_encoder", "resnet18",
    ]

    try:
        from policy import ACTPolicy

        policy_config = {
            "lr": 1e-5,
            "num_queries": 60,
            "action_dim": 20,
            "kl_weight": 1,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["left", "left_wrist", "right_wrist"],
            "multi_gpu": False,
        }

        # Build model
        policy = ACTPolicy(policy_config)
        policy.to(device)
        policy.eval()

        # Create dummy inputs matching real inference:
        #   images: (1, 3_cameras, 3_channels, 360, 480)
        #   qpos:   (1, 20)
        dummy_images = torch.randn(1, 3, 3, 360, 480).float().to(device)
        qpos_zero = torch.zeros(1, 20).float().to(device)

        # Forward pass
        t0 = time.time()
        with torch.inference_mode():
            action_chunk = policy(qpos_zero, dummy_images)
        t1 = time.time()

        action_np = action_chunk.cpu().numpy().squeeze()
        expected_shape = (60, 20)  # chunk_size=60, action_dim=20

        if action_np.shape != expected_shape:
            result.fail(
                f"Output shape {action_np.shape}, expected {expected_shape}"
            )
            return

        inference_ms = (t1 - t0) * 1000
        result.pass_(
            f"Output shape: {action_np.shape}, "
            f"range: [{action_np.min():.3f}, {action_np.max():.3f}], "
            f"inference time: {inference_ms:.0f} ms"
        )

        if inference_ms > 500:
            result.warn(
                f"Inference took {inference_ms:.0f} ms (>500 ms). "
                "This may be too slow for 10 Hz control."
            )
    finally:
        sys.argv = saved_argv


def test_checkpoint_inference(result, deploy_dir):
    """Load all 3 real checkpoints and run inference with each."""
    import torch
    import numpy as np

    src_act_dir = os.path.join(deploy_dir, "src", "act")
    if src_act_dir not in sys.path:
        sys.path.insert(0, src_act_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    subtasks = {
        "needle_pickup": os.path.join(deploy_dir, "checkpoints", "act_np", "policy_best.ckpt"),
        "needle_throw": os.path.join(deploy_dir, "checkpoints", "act_nt", "policy_best.ckpt"),
        "knot_tying": os.path.join(deploy_dir, "checkpoints", "act_kt", "policy_best.ckpt"),
    }

    # Check all checkpoints exist first
    missing = [name for name, path in subtasks.items() if not os.path.exists(path)]
    if missing:
        result.fail(f"Missing checkpoints for: {', '.join(missing)}")
        return

    saved_argv = sys.argv
    sys.argv = [
        "act", "--task_name", "needle_pickup_all",
        "--ckpt_dir", "/tmp", "--policy_class", "ACT",
        "--seed", "0", "--num_epochs", "1",
        "--kl_weight", "1", "--chunk_size", "60",
        "--hidden_dim", "512", "--dim_feedforward", "3200",
        "--lr", "1e-5", "--batch_size", "8",
        "--image_encoder", "resnet18",
    ]

    try:
        from policy import ACTPolicy

        policy_config = {
            "lr": 1e-5,
            "num_queries": 60,
            "action_dim": 20,
            "kl_weight": 1,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["left", "left_wrist", "right_wrist"],
            "multi_gpu": False,
        }

        dummy_images = torch.randn(1, 3, 3, 360, 480).float().to(device)
        qpos_zero = torch.zeros(1, 20).float().to(device)

        subtask_results = []
        for name, ckpt_path in subtasks.items():
            policy = ACTPolicy(policy_config)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            policy.load_state_dict(state_dict)
            policy.to(device)
            policy.eval()

            t0 = time.time()
            with torch.inference_mode():
                action_chunk = policy(qpos_zero, dummy_images)
            t1 = time.time()

            action_np = action_chunk.cpu().numpy().squeeze()
            inference_ms = (t1 - t0) * 1000
            subtask_results.append(
                f"{name}: shape={action_np.shape}, "
                f"range=[{action_np.min():.3f}, {action_np.max():.3f}], "
                f"{inference_ms:.0f}ms"
            )

            # Free memory
            del policy, ckpt, state_dict
            torch.cuda.empty_cache()

        result.pass_("All 3 checkpoints loaded and inferred:\n" +
                     "\n".join(f"    {r}" for r in subtask_results))
    finally:
        sys.argv = saved_argv


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Verify SutureBot ACT deployment readiness"
    )
    parser.add_argument(
        "--deploy_dir", type=str,
        default=os.path.expanduser("~/suturebot_deploy"),
        help="Path to deployment directory (default: ~/suturebot_deploy)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra details"
    )
    args = parser.parse_args()

    deploy_dir = os.path.abspath(args.deploy_dir)

    print("=" * 60)
    print("  SutureBot ACT Deployment Verification")
    print("=" * 60)
    print(f"  Deploy directory: {deploy_dir}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print()

    # Run tests
    print("--- Environment ---")
    run_test("Conda environment", test_conda_env)
    run_test("Python version", test_python_version)
    run_test("GPU availability", test_gpu)
    run_test("Python imports", test_imports)
    print()

    print("--- Files ---")
    run_test("Required files", test_files_exist, deploy_dir)
    print()

    print("--- Checkpoints ---")
    for label, subdir in [("Needle Pickup", "act_np"),
                          ("Needle Throw", "act_nt"),
                          ("Knot Tying", "act_kt")]:
        ckpt_path = os.path.join(deploy_dir, "checkpoints", subdir, "policy_best.ckpt")
        run_test(f"Load {label} checkpoint", test_checkpoint_load, ckpt_path, label)
    print()

    print("--- Model Pipeline ---")
    run_test("Build ACTPolicy from source", test_model_build, deploy_dir)
    run_test("Dummy inference (random weights)", test_dummy_inference, deploy_dir)
    run_test("Checkpoint inference (all 3 models)", test_checkpoint_inference, deploy_dir)
    print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    warnings = sum(len(r.warnings) for r in results)

    print("=" * 60)
    if failed == 0:
        print(f"  PASS   -- {passed}/{passed + failed} tests passed", end="")
        if warnings:
            print(f" ({warnings} warning(s))")
        else:
            print()
        print()
        print("  The deployment is ready for robot testing.")
        print("  To run inference on the real dVRK:")
        print(f"    cd {deploy_dir}/src/act")
        print(f"    source /opt/ros/noetic/setup.bash")
        print(f"    source ~/catkin_ws/devel/setup.bash")
        print(f"    conda activate act")
        print(f"    python chained_dvrk_inference.py \\")
        print(f"      --ckpt_np {deploy_dir}/checkpoints/act_np/policy_best.ckpt \\")
        print(f"      --ckpt_nt {deploy_dir}/checkpoints/act_nt/policy_best.ckpt \\")
        print(f"      --ckpt_kt {deploy_dir}/checkpoints/act_kt/policy_best.ckpt")
    else:
        print(f"  FAIL   -- {failed} test(s) failed, {passed} passed")
        print()
        print("  Failed tests:")
        for r in results:
            if not r.passed:
                print(f"    [-] {r.name}: {r.message.split(chr(10))[0]}")
        print()
        print("  Fix the issues above and re-run this script.")

    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
