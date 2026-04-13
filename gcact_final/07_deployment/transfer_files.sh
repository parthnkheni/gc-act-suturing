#!/usr/bin/env bash
# transfer_files.sh  -- Copy Trained Models to JHU Machine
# Run this on the A100 training server to copy all necessary files to the JHU
# da Vinci workstation: model checkpoints (~1.2GB each), inference scripts,
# model source code, and the deployment verification script. Uses rsync so
# it can be re-run safely if the transfer is interrupted.

# transfer_files.sh
#
# Run this script on the A100 TRAINING SERVER to transfer checkpoints and
# inference code to the JHU deployment machine.
#
# Usage:
#   ./transfer_files.sh user@jhu-machine:/home/user/suturebot_deploy
#   ./transfer_files.sh --dry-run user@jhu-machine:/home/user/suturebot_deploy
#
# What gets transferred:
#   - 3 ACT model checkpoints (~406 MB each, ~1.2 GB total)
#   - Inference scripts (chained_dvrk_inference.py, policy.py, rostopics.py)
#   - DETR model code (detr/ directory)
#   - dVRK scripts (dvrk_scripts/ directory)
#   - Deployment checklist
#   - This deploy_package/ itself (setup + verify scripts)
#
# Checkpoint priority:
#   Prefers 10-tissue models (act_{np,nt,kt}_10t_kl1) if available.
#   Falls back to 9-tissue models (act_{np,nt,kt}_all10_kl1) otherwise.
#
# NOTE: The 10-tissue models may still be training. Run this script AFTER
# training completes, or use --dry-run to see what would be transferred.

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
DRY_RUN=0
DESTINATION=""

for arg in "$@"; do
    case "$arg" in
        --dry-run|--dry_run)
            DRY_RUN=1
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] user@host:/path/to/suturebot_deploy"
            echo ""
            echo "  --dry-run    Show what would be transferred without actually doing it"
            echo "  DESTINATION  rsync-style destination (e.g., user@jhu-machine:/home/user/suturebot_deploy)"
            exit 0
            ;;
        *)
            DESTINATION="$arg"
            ;;
    esac
done

if [ -z "${DESTINATION}" ]; then
    echo -e "${RED}ERROR: No destination specified.${NC}"
    echo "Usage: $0 [--dry-run] user@host:/path/to/suturebot_deploy"
    echo ""
    echo "Example:"
    echo "  $0 alice@dvrk-workstation.jhu.edu:/home/alice/suturebot_deploy"
    echo "  $0 --dry-run alice@dvrk-workstation.jhu.edu:/home/alice/suturebot_deploy"
    exit 1
fi

# Paths on this (training) machine
HOME_DIR="${HOME}"
CKPT_DIR="${HOME_DIR}/checkpoints"
SRC_DIR="${HOME_DIR}/SutureBot/src/act"
DEPLOY_PKG="${HOME_DIR}/deploy_package"

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  SutureBot ACT  -- File Transfer to Deployment Machine${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo "  Source (this machine):  $(hostname)"
echo "  Destination:            ${DESTINATION}"
if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "  Mode:                   ${YELLOW}DRY RUN (no files will be transferred)${NC}"
fi
echo ""

# Resolve checkpoints (prefer v2, fall back to 10-tissue, then 9-tissue)
echo -e "${BOLD}[1/4] Resolving checkpoints...${NC}"

resolve_checkpoint() {
    local subtask_label="$1"    # e.g., "needle pickup"
    local v2="$2"               # e.g., act_np_v2
    local primary="$3"          # e.g., act_np_10t_kl1
    local fallback="$4"         # e.g., act_np_all10_kl1
    local dest_subdir="$5"      # e.g., act_np

    local ckpt_path=""
    local ckpt_source=""

    if [ -f "${CKPT_DIR}/${v2}/policy_best.ckpt" ]; then
        ckpt_path="${CKPT_DIR}/${v2}/policy_best.ckpt"
        ckpt_source="${v2} (v2 EfficientNet-B3)"
    elif [ -f "${CKPT_DIR}/${primary}/policy_best.ckpt" ]; then
        ckpt_path="${CKPT_DIR}/${primary}/policy_best.ckpt"
        ckpt_source="${primary} (v1 10-tissue)"
    elif [ -f "${CKPT_DIR}/${fallback}/policy_best.ckpt" ]; then
        ckpt_path="${CKPT_DIR}/${fallback}/policy_best.ckpt"
        ckpt_source="${fallback} (9-tissue fallback)"
    else
        echo -e "  ${RED}MISSING: ${subtask_label}  -- no checkpoint found${NC}"
        echo "    Checked: ${CKPT_DIR}/${v2}/policy_best.ckpt"
        echo "    Checked: ${CKPT_DIR}/${primary}/policy_best.ckpt"
        echo "    Checked: ${CKPT_DIR}/${fallback}/policy_best.ckpt"
        return 1
    fi

    local size=$(du -sh "${ckpt_path}" | cut -f1)
    echo -e "  ${GREEN}${subtask_label}${NC}: ${ckpt_source} (${size})"
    echo "    ${ckpt_path}"

    # Export for use by transfer step
    eval "CKPT_${dest_subdir}=${ckpt_path}"
    return 0
}

CKPT_ERRORS=0

resolve_checkpoint "Needle Pickup" "act_np_v2" "act_np_10t_kl1" "act_np_all10_kl1" "act_np" || ((CKPT_ERRORS++))
resolve_checkpoint "Needle Throw"  "act_nt_v2" "act_nt_10t_kl1" "act_nt_all10_kl1" "act_nt" || ((CKPT_ERRORS++))
resolve_checkpoint "Knot Tying"    "act_kt_v2" "act_kt_10t_kl1" "act_kt_all10_kl1" "act_kt" || ((CKPT_ERRORS++))

# GC-ACT checkpoints (NT and KT only  -- NP uses v2)
echo ""
echo -e "${BOLD}  GC-ACT checkpoints:${NC}"
GCACT_NT_CKPT=""
GCACT_KT_CKPT=""
GESTURE_CKPT=""

if [ -f "${CKPT_DIR}/act_nt_gcact/policy_best.ckpt" ]; then
    GCACT_NT_CKPT="${CKPT_DIR}/act_nt_gcact/policy_best.ckpt"
    local_size=$(du -sh "${GCACT_NT_CKPT}" | cut -f1)
    echo -e "  ${GREEN}NT GC-ACT${NC}: ${local_size}"
else
    echo -e "  ${YELLOW}NT GC-ACT: not yet trained${NC}"
fi

if [ -f "${CKPT_DIR}/act_kt_gcact/policy_best.ckpt" ]; then
    GCACT_KT_CKPT="${CKPT_DIR}/act_kt_gcact/policy_best.ckpt"
    local_size=$(du -sh "${GCACT_KT_CKPT}" | cut -f1)
    echo -e "  ${GREEN}KT GC-ACT${NC}: ${local_size}"
else
    echo -e "  ${YELLOW}KT GC-ACT: not yet trained${NC}"
fi

if [ -f "${CKPT_DIR}/gesture_classifier/gesture_best.ckpt" ]; then
    GESTURE_CKPT="${CKPT_DIR}/gesture_classifier/gesture_best.ckpt"
    local_size=$(du -sh "${GESTURE_CKPT}" | cut -f1)
    echo -e "  ${GREEN}Gesture Classifier${NC}: ${local_size}"
else
    echo -e "  ${YELLOW}Gesture Classifier: not found${NC}"
fi

if [ ${CKPT_ERRORS} -gt 0 ]; then
    echo ""
    echo -e "${RED}ERROR: ${CKPT_ERRORS} checkpoint(s) missing. Training may still be in progress.${NC}"
    echo "  Re-run this script after training completes."
    if [ ${DRY_RUN} -eq 0 ]; then
        echo "  Or use --dry-run to see what else would be transferred."
        exit 1
    fi
    echo -e "${YELLOW}  Continuing in dry-run mode...${NC}"
fi

echo ""

# Check source files exist
echo -e "${BOLD}[2/4] Verifying source files...${NC}"

MISSING=0
check_file() {
    if [ -e "$1" ]; then
        echo -e "  ${GREEN}OK${NC}   $1"
    else
        echo -e "  ${RED}MISSING${NC} $1"
        ((MISSING++))
    fi
}

check_file "${SRC_DIR}/chained_dvrk_inference.py"
check_file "${SRC_DIR}/chained_dvrk_inference_gcact.py"
check_file "${SRC_DIR}/policy.py"
check_file "${SRC_DIR}/rostopics.py"
check_file "${SRC_DIR}/detr/main.py"
check_file "${SRC_DIR}/detr/models/__init__.py"
check_file "${SRC_DIR}/detr/models/detr_vae.py"
check_file "${SRC_DIR}/detr/models/backbone.py"
check_file "${SRC_DIR}/detr/models/transformer.py"
check_file "${SRC_DIR}/detr/models/transformer_bert.py"
check_file "${SRC_DIR}/detr/models/position_encoding.py"
check_file "${SRC_DIR}/detr/models/resnet.py"
check_file "${SRC_DIR}/detr/models/resnet_film.py"
check_file "${SRC_DIR}/detr/models/efficientnet.py"
check_file "${SRC_DIR}/detr/util/__init__.py"
check_file "${SRC_DIR}/detr/util/misc.py"
check_file "${SRC_DIR}/dvrk_scripts/constants_dvrk.py"
check_file "${SRC_DIR}/dvrk_scripts/dvrk_control.py"
check_file "${HOME_DIR}/JHU_DEPLOYMENT_CHECKLIST.md"
check_file "${DEPLOY_PKG}/setup_deployment.sh"
check_file "${DEPLOY_PKG}/verify_deployment.py"

if [ ${MISSING} -gt 0 ]; then
    echo ""
    echo -e "${RED}ERROR: ${MISSING} source file(s) missing.${NC}"
    if [ ${DRY_RUN} -eq 0 ]; then
        exit 1
    fi
fi

echo ""

# Estimate total transfer size
echo -e "${BOLD}[3/4] Estimating transfer size...${NC}"

TOTAL_SIZE=0

estimate_size() {
    if [ -e "$1" ]; then
        local bytes=$(du -sb "$1" 2>/dev/null | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + bytes))
    fi
}

# Checkpoints
estimate_size "${CKPT_act_np:-/dev/null}"
estimate_size "${CKPT_act_nt:-/dev/null}"
estimate_size "${CKPT_act_kt:-/dev/null}"

# Source code
estimate_size "${SRC_DIR}/chained_dvrk_inference.py"
estimate_size "${SRC_DIR}/policy.py"
estimate_size "${SRC_DIR}/rostopics.py"
estimate_size "${SRC_DIR}/detr"
estimate_size "${SRC_DIR}/dvrk_scripts"

# Deploy package
estimate_size "${DEPLOY_PKG}"

# Checklist
estimate_size "${HOME_DIR}/JHU_DEPLOYMENT_CHECKLIST.md"

# Convert to human-readable
if [ ${TOTAL_SIZE} -gt 1073741824 ]; then
    SIZE_HUMAN="$(echo "scale=1; ${TOTAL_SIZE}/1073741824" | bc) GB"
elif [ ${TOTAL_SIZE} -gt 1048576 ]; then
    SIZE_HUMAN="$(echo "scale=1; ${TOTAL_SIZE}/1048576" | bc) MB"
else
    SIZE_HUMAN="${TOTAL_SIZE} bytes"
fi

echo "  Estimated total transfer size: ${SIZE_HUMAN}"
echo "    Checkpoints:  ~1.2 GB (3 x ~406 MB)"
echo "    Source code:   ~1 MB"
echo "    Deploy tools:  ~50 KB"
echo ""

# Transfer files
echo -e "${BOLD}[4/4] Transferring files...${NC}"

RSYNC_FLAGS="-avz --progress"
if [ ${DRY_RUN} -eq 1 ]; then
    RSYNC_FLAGS="${RSYNC_FLAGS} --dry-run"
    echo -e "${YELLOW}  DRY RUN  -- showing what would be transferred:${NC}"
fi

echo ""

# Helper function
do_transfer() {
    local src="$1"
    local dest="$2"
    local label="$3"

    echo -e "${CYAN}--- ${label} ---${NC}"
    if [ -e "${src}" ]; then
        rsync ${RSYNC_FLAGS} "${src}" "${dest}"
    else
        echo -e "  ${YELLOW}SKIPPED (source not found): ${src}${NC}"
    fi
    echo ""
}

# 4a. Checkpoints
do_transfer "${CKPT_act_np:-}" "${DESTINATION}/checkpoints/act_np/policy_best.ckpt" "Needle Pickup checkpoint"
do_transfer "${CKPT_act_nt:-}" "${DESTINATION}/checkpoints/act_nt/policy_best.ckpt" "Needle Throw checkpoint"
do_transfer "${CKPT_act_kt:-}" "${DESTINATION}/checkpoints/act_kt/policy_best.ckpt" "Knot Tying checkpoint"

# 4b. Main inference scripts (ACT + GC-ACT)
do_transfer "${SRC_DIR}/chained_dvrk_inference.py" "${DESTINATION}/src/act/" "Chained inference script (ACT)"
do_transfer "${SRC_DIR}/chained_dvrk_inference_gcact.py" "${DESTINATION}/src/act/" "Chained inference script (GC-ACT)"

# 4c. Policy module
do_transfer "${SRC_DIR}/policy.py" "${DESTINATION}/src/act/" "ACT policy module"

# 4d. ROS topics module
do_transfer "${SRC_DIR}/rostopics.py" "${DESTINATION}/src/act/" "ROS topics module"

# 4e. DETR directory (entire model codebase)
do_transfer "${SRC_DIR}/detr/" "${DESTINATION}/src/act/detr/" "DETR model code"

# 4f. dVRK scripts directory
do_transfer "${SRC_DIR}/dvrk_scripts/" "${DESTINATION}/src/act/dvrk_scripts/" "dVRK scripts (constants + control)"

# 4g. Deployment checklist
do_transfer "${HOME_DIR}/JHU_DEPLOYMENT_CHECKLIST.md" "${DESTINATION}/" "Deployment checklist"

# 4h. Deploy package (setup + verify scripts)
do_transfer "${DEPLOY_PKG}/" "${DESTINATION}/deploy_package/" "Deploy package (setup + verify scripts)"

# 4i. GC-ACT checkpoints (if available)
if [ -n "${GCACT_NT_CKPT}" ]; then
    do_transfer "${GCACT_NT_CKPT}" "${DESTINATION}/checkpoints/act_nt_gcact/policy_best.ckpt" "NT GC-ACT checkpoint"
fi
if [ -n "${GCACT_KT_CKPT}" ]; then
    do_transfer "${GCACT_KT_CKPT}" "${DESTINATION}/checkpoints/act_kt_gcact/policy_best.ckpt" "KT GC-ACT checkpoint"
fi

# 4j. Gesture classifier (if available)
if [ -n "${GESTURE_CKPT}" ]; then
    do_transfer "${GESTURE_CKPT}" "${DESTINATION}/checkpoints/gesture_classifier/gesture_best.ckpt" "Gesture classifier checkpoint"
fi

# Summary
echo -e "${BOLD}============================================================${NC}"
if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}  DRY RUN COMPLETE  -- no files were transferred${NC}"
    echo ""
    echo "  To actually transfer, run without --dry-run:"
    echo "    $0 ${DESTINATION}"
else
    echo -e "${GREEN}  TRANSFER COMPLETE${NC}"
    echo ""
    echo "  On the deployment machine, run:"
    echo "    1. cd ${DESTINATION##*:}/deploy_package"
    echo "    2. bash setup_deployment.sh       # set up conda env"
    echo "    3. conda activate act"
    echo "    4. python verify_deployment.py    # verify everything works"
fi
echo -e "${BOLD}============================================================${NC}"
