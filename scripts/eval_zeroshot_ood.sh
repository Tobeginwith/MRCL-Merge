#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 MODEL_NAME MODEL_LABEL TEST_FILE_BASE RESULTS_DIR_BASE [TASK_IDS]" >&2
    echo "TASK_IDS is a quoted, space- or comma-separated list from 1 to 10; default: all tasks" >&2
    exit 2
fi

MODEL_NAME=$1
MODEL_LABEL=$2
TEST_FILE_BASE=$3
RESULTS_DIR_BASE=$4
TASK_IDS=${5-"1 2 3 4 5 6 7 8 9 10"}

DATA=(
    "MMMU_pro"
    "MathVerse"
    "RealworldQA"
    "MMStar"
    "CountBenchQA"
    "DocVQA"
    "Charxiv"
    "POPE"
    "MathVision"
    "MathVista"
)

# Task IDs are one-based indices into DATA; accept "1 4 6" or "1,4,6".
read -r -a SELECTED_TASK_IDS <<< "${TASK_IDS//,/ }"
if [ "${#SELECTED_TASK_IDS[@]}" -eq 0 ]; then
    echo "TASK_IDS must contain at least one task ID." >&2
    exit 2
fi

SELECTED_DATASETS=()
for TASK_ID in "${SELECTED_TASK_IDS[@]}"; do
    if ! [[ "$TASK_ID" =~ ^([1-9]|10)$ ]]; then
        echo "TASK_IDS only accepts values from 1 to 10: ${TASK_IDS}" >&2
        exit 2
    fi
    SELECTED_DATASETS+=("${DATA[$((TASK_ID - 1))]}")
done

if ! [[ "$MODEL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$MODEL_LABEL" == "." || "$MODEL_LABEL" == ".." ]]; then
    echo "MODEL_LABEL must be a directory name containing only letters, digits, dot, underscore, and hyphen (not . or ..)." >&2
    exit 2
fi

LOG_FILE="${RESULTS_DIR_BASE}/${MODEL_LABEL}/eval.log"
mkdir -p "${RESULTS_DIR_BASE}/${MODEL_LABEL}"
exec > >(tee -a "$LOG_FILE") 2>&1

BATCH_SIZE=2048
DISABLE_FLASH_ATTN2=true
for dataset in "${SELECTED_DATASETS[@]}"; do
    if [ "${dataset}" == "POPE" ]; then
        TEST_FILE="${TEST_FILE_BASE}/POPE/coco_pope.json"
        MEDIA_DIR="${TEST_FILE_BASE}/POPE/coco/image"
    elif [[ "${dataset}" =~ ^(We-Math2|Chemistry|Coding|Navigation|CVQA|FinMME|MedBookVQA|InstructFollow)$ ]]; then
        TEST_FILE="${TEST_FILE_BASE}/${dataset}/jsons/test/data.json"
        MEDIA_DIR="${TEST_FILE_BASE}/${dataset}/images"
    else
        # parquet-based HuggingFace datasets (MMMU_pro, MathVerse, RealworldQA, MMStar,
        # CountBenchQA, DocVQA, Charxiv, MathVision, MathVista, …)
        TEST_FILE="${TEST_FILE_BASE}/${dataset}"
        MEDIA_DIR=""
    fi
    RESULTS_DIR="${RESULTS_DIR_BASE}/${MODEL_LABEL}/${dataset}"
    if [ "${dataset}" == "MMMU_pro" ]; then
        Options=(
                "standard (4 options)"
                "standard (10 options)"
                "vision"
                )
        for options in "${Options[@]}"; do
            TEST_FILE_OPTION="${TEST_FILE}/${options}"
            RESULTS_DIR_OPTION="${RESULTS_DIR}/${options}"
            echo "========================================"
            echo "Processing dataset: ${dataset}"
            echo "========================================"
            if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
                    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
                fi

            IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
            CHUNKS=${#GPULIST[@]}

            echo "======================================"
            echo "Starting evaluation"
            echo "======================================"
            echo "Model: ${MODEL_NAME}"
            echo "Test file: ${TEST_FILE_OPTION}"
            echo "Results directory: ${RESULTS_DIR_OPTION}"
            echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
            echo "GPU count: ${CHUNKS}"
            echo "======================================"

            mkdir -p "${RESULTS_DIR_OPTION}"
            echo "======================================"
            FLASH_ATTN_FLAG=""
            if [ "${DISABLE_FLASH_ATTN2}" = "true" ]; then
                FLASH_ATTN_FLAG="--disable_flash_attn2"
            fi

            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            python src/eval/inference.py \
                --base_model "${MODEL_NAME}" \
                --test_file "${TEST_FILE_OPTION}" \
                --media_dir "${MEDIA_DIR}" \
                --output_dir "${RESULTS_DIR_OPTION}" \
                --prompts_file "src/dataset/prompts_ood.yaml" \
                --max_completion_length 16384 \
                --tensor_parallel_size ${CHUNKS} \
                --batch_size ${BATCH_SIZE} \
                ${FLASH_ATTN_FLAG}

            OUTPUT_FILE="${RESULTS_DIR_OPTION}/merge.jsonl"

            echo "Merging completed!"
            echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"
            echo "======================================"

            python src/eval/eval.py \
                --dataset_name "${dataset}" \
                --merged_file "${OUTPUT_FILE}" \
                --output_dir "${RESULTS_DIR_OPTION}"
        done
    else
        echo "========================================"
        echo "Processing dataset: ${dataset}"
        echo "========================================"
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
                export CUDA_VISIBLE_DEVICES=$ALL_GPUS
            fi

        IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
        CHUNKS=${#GPULIST[@]}

        echo "======================================"
        echo "Starting evaluation"
        echo "======================================"
        echo "Model: ${MODEL_NAME}"
        echo "Test file: ${TEST_FILE}"
        echo "Results directory: ${RESULTS_DIR}"
        echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
        echo "GPU count: ${CHUNKS}"
        echo "======================================"

        mkdir -p "${RESULTS_DIR}"
        echo "======================================"
        FLASH_ATTN_FLAG=""
        if [ "${DISABLE_FLASH_ATTN2}" = "true" ]; then
            FLASH_ATTN_FLAG="--disable_flash_attn2"
        fi

        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        python src/eval/inference.py \
            --base_model "${MODEL_NAME}" \
            --test_file "${TEST_FILE}" \
            --media_dir "${MEDIA_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --prompts_file "src/dataset/prompts_ood.yaml" \
            --max_completion_length 16384 \
            --tensor_parallel_size ${CHUNKS} \
            --batch_size ${BATCH_SIZE} \
            ${FLASH_ATTN_FLAG}

        OUTPUT_FILE="${RESULTS_DIR}/merge.jsonl"

        echo "Merging completed!"
        echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"
        echo "======================================"
        
        python src/eval/eval.py \
            --dataset_name "${dataset}" \
            --merged_file "${OUTPUT_FILE}" \
            --output_dir "${RESULTS_DIR}"
        
    fi
done
