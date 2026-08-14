#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 MODEL_NAME MODEL_LABEL BASE_PATH DATASET..." >&2
    exit 2
fi

MODEL_NAME=$1
MODEL_LABEL=$2
BASE_PATH=$3
shift 3
DATASETS=("$@")

if ! [[ "$MODEL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "MODEL_LABEL may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi
if [ ! -f "${MODEL_NAME}/config.json" ]; then
    echo "Model config not found: ${MODEL_NAME}/config.json" >&2
    exit 1
fi

BASE_RESULTS_DIR=${RESULTS_ROOT:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/results/Qwen3-VL-30B-A3B-GSPO-Experts}
BATCH_SIZE=512
DISABLE_FLASH_ATTN2=false

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
fi

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPU_LIST[@]}
if [ "$CHUNKS" -ne 8 ]; then
    echo "30B evaluation expects exactly 8 visible GPUs; found ${CHUNKS}." >&2
    exit 1
fi

echo "======================================"
echo "Evaluating model: ${MODEL_LABEL}"
echo "Model path: ${MODEL_NAME}"
echo "Test tasks: ${#DATASETS[@]}"
echo "Tensor parallel size: ${CHUNKS}"
echo "======================================"

for TEST_DATASET in "${DATASETS[@]}"; do
    TEST_FILE=${BASE_PATH}/${TEST_DATASET}/jsons/test/data.json
    MEDIA_DIR=${BASE_PATH}/${TEST_DATASET}/images
    RESULTS_DIR=${BASE_RESULTS_DIR}/${MODEL_LABEL}/${TEST_DATASET}

    if [ ! -f "$TEST_FILE" ]; then
        echo "Test data not found: ${TEST_FILE}" >&2
        exit 1
    fi
    if [ ! -d "$MEDIA_DIR" ]; then
        echo "Image directory not found: ${MEDIA_DIR}" >&2
        exit 1
    fi

    if [ -f "${RESULTS_DIR}/evaluation_stats.txt" ]; then
        echo "Skipping completed evaluation: ${MODEL_LABEL}/${TEST_DATASET}"
        continue
    fi

    mkdir -p "$RESULTS_DIR"

    FLASH_ATTN_FLAG=()
    if [ "$DISABLE_FLASH_ATTN2" = true ]; then
        FLASH_ATTN_FLAG=(--disable_flash_attn2)
    fi

    echo "--------------------------------------"
    echo "Test dataset: ${TEST_DATASET}"
    echo "Results: ${RESULTS_DIR}"
    echo "--------------------------------------"

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python src/eval/inference.py \
        --base_model "$MODEL_NAME" \
        --test_file "$TEST_FILE" \
        --media_dir "$MEDIA_DIR" \
        --output_dir "$RESULTS_DIR" \
        --prompts_file src/dataset/prompts_2.yaml \
        --max_completion_length 4096 \
        --tensor_parallel_size "$CHUNKS" \
        --batch_size "$BATCH_SIZE" \
        "${FLASH_ATTN_FLAG[@]}"

    python src/eval/eval.py \
        --dataset_name "$TEST_DATASET" \
        --merged_file "${RESULTS_DIR}/merge.jsonl" \
        --output_dir "$RESULTS_DIR"
done

echo "======================================"
echo "Evaluation completed: ${MODEL_LABEL}"
echo "======================================"
