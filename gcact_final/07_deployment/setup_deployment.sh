#!/usr/bin/env bash
# setup_deployment.sh  -- Set Up JHU Robot Machine
# Run this on the JHU da Vinci workstation to install all Python dependencies
# (PyTorch, scipy, einops, etc.) needed to run the inference scripts. Creates
# a conda environment and verifies that all imports work. Only needs to be
# run once when first setting up the deployment machine.

# setup_deployment.sh
#
# Run this script on the JHU deployment machine to prepare the environment
# for running the SutureBot ACT chained inference system on the real dVRK.
#
# What it does:
#   1. Creates a conda environment "act" with Python 3.8
#   2. Installs PyTorch 1.13.1 + CUDA 11.7 (matches training server)
#   3. Installs all Python dependencies
#   4. Creates the expected directory structure
#   5. Runs import verification tests
#
# Prerequisites:
#   - conda (miniconda or anaconda) is installed and on PATH
#   - NVIDIA driver + CUDA toolkit installed (nvidia-smi should work)
#   - ROS Noetic is installed system-wide (for rospy, sensor_msgs, etc.)
#
# Usage:
#   chmod +x setup_deployment.sh
#   ./setup_deployment.sh
#
# After running this script, also run verify_deployment.py for a full check.

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  SutureBot ACT Deployment  -- Environment Setup${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# 0. Pre-flight checks
echo -e "${BOLD}[0/5] Pre-flight checks...${NC}"

if ! command -v conda &>/dev/null; then
    echo -e "${RED}ERROR: conda not found. Install Miniconda first:${NC}"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}WARNING: nvidia-smi not found. GPU may not be available.${NC}"
else
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1 | sed 's/^/    /'
fi

# Check for ROS
if [ -d "/opt/ros/noetic" ]; then
    echo "  ROS Noetic found at /opt/ros/noetic"
    ROS_FOUND=1
elif [ -d "/opt/ros/melodic" ]; then
    echo "  ROS Melodic found at /opt/ros/melodic"
    ROS_FOUND=1
else
    echo -e "${YELLOW}  WARNING: ROS not found in /opt/ros/. ROS packages (rospy, sensor_msgs,"
    echo -e "  geometry_msgs, cv_bridge) must come from the system ROS install.${NC}"
    echo -e "${YELLOW}  The conda env will NOT include ROS packages  -- source your ROS setup.bash${NC}"
    echo -e "${YELLOW}  before running inference on the robot.${NC}"
    ROS_FOUND=0
fi

echo ""

# 1. Create conda environment
echo -e "${BOLD}[1/5] Creating conda environment 'act' (Python 3.8)...${NC}"

if conda env list | grep -qw "^act "; then
    echo -e "${YELLOW}  Environment 'act' already exists. Updating...${NC}"
    EXISTING_ENV=1
else
    conda create -y -n act python=3.8
    EXISTING_ENV=0
fi

# Activate the environment
# (eval is needed because conda activate doesn't work in subshells by default)
eval "$(conda shell.bash hook)"
conda activate act

echo "  Python: $(python --version)"
echo "  Env location: $(conda info --envs | grep 'act ' | awk '{print $NF}')"
echo ""

# 2. Install PyTorch with CUDA
echo -e "${BOLD}[2/5] Installing PyTorch with CUDA support...${NC}"

# PyTorch 1.13.1 + CUDA 11.7  -- matches the A100 training server
# Adjust if deployment machine has a different CUDA version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

echo ""

# 3. Install Python dependencies
echo -e "${BOLD}[3/5] Installing Python dependencies...${NC}"

pip install \
    scipy \
    scikit-learn \
    einops \
    opencv-python \
    matplotlib \
    numpy \
    ipython \
    pandas \
    keyboard

# Note: The following packages are needed for real robot operation but come
# from the system ROS install, NOT pip:
#   - rospy
#   - sensor_msgs
#   - geometry_msgs
#   - std_msgs
#   - cv_bridge
#   - dynamic_reconfigure
#
# The following are dVRK-specific and typically installed separately:
#   - crtk  (Collaborative Robotics Toolkit)
#   - dvrk  (da Vinci Research Kit Python client)
#   - PyKDL (KDL kinematics library)
#
# These are NOT installed by this script. Ensure they are available in your
# ROS workspace or system Python path. When running inference, do:
#   source /opt/ros/noetic/setup.bash
#   source ~/catkin_ws/devel/setup.bash  # (your dVRK workspace)
#   conda activate act

echo ""

# 4. Create directory structure
echo -e "${BOLD}[4/5] Creating directory structure...${NC}"

DEPLOY_BASE="${HOME}/suturebot_deploy"
mkdir -p "${DEPLOY_BASE}/checkpoints/act_np"
mkdir -p "${DEPLOY_BASE}/checkpoints/act_nt"
mkdir -p "${DEPLOY_BASE}/checkpoints/act_kt"
mkdir -p "${DEPLOY_BASE}/src/act/detr/models"
mkdir -p "${DEPLOY_BASE}/src/act/detr/util"
mkdir -p "${DEPLOY_BASE}/src/act/dvrk_scripts"
mkdir -p "${DEPLOY_BASE}/logs"

echo "  Created directory tree at ${DEPLOY_BASE}:"
echo "    checkpoints/"
echo "      act_np/       <- needle pickup checkpoint"
echo "      act_nt/       <- needle throw checkpoint"
echo "      act_kt/       <- knot tying checkpoint"
echo "    src/act/         <- inference scripts"
echo "      detr/          <- transformer model code"
echo "      dvrk_scripts/  <- dVRK constants and control"
echo "    logs/            <- inference logs"
echo ""

# 5. Verify installation
echo -e "${BOLD}[5/5] Verifying installation...${NC}"

IMPORT_ERRORS=0

python -c "
import sys
errors = []

# Core ML
try:
    import torch
    print(f'  torch {torch.__version__}  -- CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'    GPU: {torch.cuda.get_device_name(0)}')
        print(f'    CUDA version: {torch.version.cuda}')
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import torchvision
    print(f'  torchvision {torchvision.__version__}')
except ImportError as e:
    errors.append(f'torchvision: {e}')

# Scientific
try:
    import scipy
    print(f'  scipy {scipy.__version__}')
except ImportError as e:
    errors.append(f'scipy: {e}')

try:
    import sklearn
    print(f'  scikit-learn {sklearn.__version__}')
except ImportError as e:
    errors.append(f'scikit-learn: {e}')

try:
    import cv2
    print(f'  opencv {cv2.__version__}')
except ImportError as e:
    errors.append(f'opencv: {e}')

try:
    import einops
    print(f'  einops {einops.__version__}')
except ImportError as e:
    errors.append(f'einops: {e}')

try:
    import matplotlib
    print(f'  matplotlib {matplotlib.__version__}')
except ImportError as e:
    errors.append(f'matplotlib: {e}')

try:
    import numpy
    print(f'  numpy {numpy.__version__}')
except ImportError as e:
    errors.append(f'numpy: {e}')

try:
    import IPython
    print(f'  IPython {IPython.__version__}')
except ImportError as e:
    errors.append(f'IPython: {e}')

# ROS (informational  -- may not be available in conda alone)
try:
    import rospy
    print(f'  rospy (available)')
except ImportError:
    print(f'  rospy (NOT available  -- source ROS setup.bash before robot use)')

if errors:
    print()
    print('IMPORT ERRORS:')
    for err in errors:
        print(f'  FAIL: {err}')
    sys.exit(1)
else:
    print()
    print('All required packages imported successfully.')
    sys.exit(0)
" && IMPORT_ERRORS=0 || IMPORT_ERRORS=1

echo ""

# Summary
echo -e "${BOLD}============================================================${NC}"
if [ ${IMPORT_ERRORS} -eq 0 ]; then
    echo -e "${GREEN}  SETUP COMPLETE${NC}"
    echo ""
    echo "  Next steps:"
    echo "    1. Run transfer_files.sh on the training server to copy checkpoints"
    echo "    2. Run verify_deployment.py to confirm everything works end-to-end"
    echo "    3. Before robot use, source ROS and dVRK workspace:"
    echo "         source /opt/ros/noetic/setup.bash"
    echo "         source ~/catkin_ws/devel/setup.bash"
    echo "         conda activate act"
    echo "    4. Run inference:"
    echo "         cd ${DEPLOY_BASE}/src/act"
    echo "         python chained_dvrk_inference.py \\"
    echo "           --ckpt_np ${DEPLOY_BASE}/checkpoints/act_np/policy_best.ckpt \\"
    echo "           --ckpt_nt ${DEPLOY_BASE}/checkpoints/act_nt/policy_best.ckpt \\"
    echo "           --ckpt_kt ${DEPLOY_BASE}/checkpoints/act_kt/policy_best.ckpt"
else
    echo -e "${RED}  SETUP FAILED  -- see errors above${NC}"
    exit 1
fi
echo -e "${BOLD}============================================================${NC}"
