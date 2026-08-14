#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

BASE_MODEL=${BASE_MODEL:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/Qwen3-VL-30B-A3B-Instruct}
BASE_PATH=${BASE_PATH:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/MRCL}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/checkpoints/Qwen3-VL-30B-A3B-GSPO-Experts}
RESULTS_ROOT=${RESULTS_ROOT:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/results/Qwen3-VL-30B-A3B-GSPO-Experts}
TASK_IDS=${TASK_IDS:-"1 2 3 4 5"}
DO_EVAL=${DO_EVAL:-true}
EVAL_BASE=${EVAL_BASE:-true}
CONDA_ROOT=${CONDA_ROOT:-/mnt/project_modelware_roce/zhaojian/miniconda3}
TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-trlQwen}
EVAL_CONDA_ENV=${EVAL_CONDA_ENV:-vllmQwen}

export CHECKPOINT_ROOT RESULTS_ROOT

DATASETS=(
    MedBookVQA
    Navigation
    We-Math2
    Puzzle
    FinMME
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}


initialize_conda() {
    local conda_sh=${CONDA_ROOT}/etc/profile.d/conda.sh
    [ -f "$conda_sh" ] || \
        die "Conda initialization script not found: ${conda_sh}. Set CONDA_ROOT to the correct Miniconda root."

    CONDA_ROOT=$(cd "$CONDA_ROOT" && pwd -P)
    export CONDA_ROOT
    export PATH="${CONDA_ROOT}/condabin:${PATH}"
    hash -r

    # Initialize conda inside this script; shell functions from a caller are not
    # exported reliably to `bash run_experts.sh`.
    set +u
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    set -u

    local detected_root
    detected_root=$(conda info --base)
    detected_root=$(cd "$detected_root" && pwd -P)
    [ "$detected_root" = "$CONDA_ROOT" ] || \
        die "Expected Conda root ${CONDA_ROOT}, but conda resolved to ${detected_root}."
}

activate_conda_env() {
    local env_name=$1
    conda activate "$env_name" || die "Failed to activate Conda environment '${env_name}'."

    # A caller may already have another Conda installation activated. Put this
    # environment first so its python/accelerate executables cannot be shadowed.
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r

    local python_exe
    python_exe=$(command -v python) || die "python is unavailable after activating '${env_name}'."
    case "$python_exe" in
        "${CONDA_PREFIX}/"*) ;;
        *) die "Environment '${env_name}' is active, but python resolved outside it: ${python_exe}." ;;
    esac
    echo "Activated Conda environment: ${env_name} (${CONDA_PREFIX})"
    echo "Python executable: ${python_exe}"
}


initialize_conda

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available on PATH."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    mapfile -t DETECTED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DETECTED_GPUS[*]}")
    export CUDA_VISIBLE_DEVICES
else
    IFS=',' read -ra DETECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
fi

if [ "${#DETECTED_GPUS[@]}" -ne 8 ]; then
    die "P0 requires exactly 8 visible GPUs; found ${#DETECTED_GPUS[@]} (${CUDA_VISIBLE_DEVICES:-none})."
fi

[ -f "${BASE_MODEL}/config.json" ] || die "Base model config not found: ${BASE_MODEL}/config.json"

read -r -a SELECTED_TASK_IDS <<< "$TASK_IDS"

mkdir -p "${CHECKPOINT_ROOT}/logs" "${RESULTS_ROOT}/logs"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MAIN_LOG=${CHECKPOINT_ROOT}/logs/run-${RUN_STAMP}-${BASHPID}.log
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Qwen3-VL-30B-A3B independent GSPO experts"
echo "Repository: ${REPO_ROOT}"
echo "Base model: ${BASE_MODEL}"
echo "MRCL data: ${BASE_PATH}"
echo "Selected task IDs: ${SELECTED_TASK_IDS[*]}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Training: temperature=0.8, P0 batch=4 x 8 GPUs x accumulation 8"
echo "Conda root: ${CONDA_ROOT}"
echo "Training environment: ${TRAIN_CONDA_ENV}"
echo "Evaluation environment: ${EVAL_CONDA_ENV}"
echo "Inference: temperature=0.0 (defined in src/eval/inference.py)"
echo "Main log: ${MAIN_LOG}"
echo "======================================"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ "$EVAL_BASE" = true ]; then
    activate_conda_env "$EVAL_CONDA_ENV"
    BASE_EVAL_LOG=${RESULTS_ROOT}/logs/base.log
    bash "${SCRIPT_DIR}/eval_expert.sh" \
        "$BASE_MODEL" base "$BASE_PATH" "${DATASETS[@]}" 2>&1 | tee -a "$BASE_EVAL_LOG"
fi

for TASK_ID in "${SELECTED_TASK_IDS[@]}"; do
    CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

    activate_conda_env "$TRAIN_CONDA_ENV"
    TRAIN_LOG=${CHECKPOINT_ROOT}/logs/${CUR_DATASET}.log
    bash "${SCRIPT_DIR}/gspo_expert.sh" \
        "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}" 2>&1 | tee -a "$TRAIN_LOG"

    if [ "$DO_EVAL" = true ]; then
        activate_conda_env "$EVAL_CONDA_ENV"
        EVAL_LOG=${RESULTS_ROOT}/logs/${CUR_DATASET}.log
        MODEL_NAME=${CHECKPOINT_ROOT}/training/${CUR_DATASET}
        bash "${SCRIPT_DIR}/eval_expert.sh" \
            "$MODEL_NAME" "$CUR_DATASET" "$BASE_PATH" "${DATASETS[@]}" 2>&1 | tee -a "$EVAL_LOG"
    fi
done

echo "======================================"
echo "Finished task IDs: ${SELECTED_TASK_IDS[*]}"
echo "======================================"
