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
CONDA_PATH=${CONDA_PATH:-/mnt/project_modelware_roce/zhaojian/miniconda3}
export CONDA_ENVS_PATH="${CONDA_PATH%/*}/conda_envs:${CONDA_PATH}/envs${CONDA_ENVS_PATH:+:${CONDA_ENVS_PATH}}"

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


# shellcheck source=/dev/null
source "${CONDA_PATH}/etc/profile.d/conda.sh"

activate_env() {
    conda activate "$1"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
}

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
echo "Conda path: ${CONDA_PATH}"
echo "Training environment: trlQwenVllm"
echo "Evaluation environment: vllmQwen"
echo "Inference: temperature=0.0 (defined in src/eval/inference.py)"
echo "Main log: ${MAIN_LOG}"
echo "======================================"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ "$EVAL_BASE" = true ]; then
    activate_env vllmQwen
    BASE_EVAL_LOG=${RESULTS_ROOT}/logs/base.log
    bash "${SCRIPT_DIR}/eval_expert.sh" \
        "$BASE_MODEL" base "$BASE_PATH" "${DATASETS[@]}" 2>&1 | tee -a "$BASE_EVAL_LOG"
fi

for TASK_ID in "${SELECTED_TASK_IDS[@]}"; do
    CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

    activate_env trlQwenVllm
    TRAIN_LOG=${CHECKPOINT_ROOT}/logs/${CUR_DATASET}.log
    bash "${SCRIPT_DIR}/gspo_expert.sh" \
        "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}" 2>&1 | tee -a "$TRAIN_LOG"

    if [ "$DO_EVAL" = true ]; then
        activate_env vllmQwen
        EVAL_LOG=${RESULTS_ROOT}/logs/${CUR_DATASET}.log
        MODEL_NAME=${CHECKPOINT_ROOT}/training/${CUR_DATASET}
        bash "${SCRIPT_DIR}/eval_expert.sh" \
            "$MODEL_NAME" "$CUR_DATASET" "$BASE_PATH" "${DATASETS[@]}" 2>&1 | tee -a "$EVAL_LOG"
    fi
done

echo "======================================"
echo "Finished task IDs: ${SELECTED_TASK_IDS[*]}"
echo "======================================"
