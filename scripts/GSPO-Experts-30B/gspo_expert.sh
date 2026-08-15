#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 BASE_MODEL BASE_PATH TASK_ID DATASET..." >&2
    exit 2
fi

BASE_MODEL=$1
BASE_PATH=$2
TASK_ID=$3
shift 3
DATASETS=("$@")


unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
export PYTHONPATH=src:${PYTHONPATH:-}

# Fixed P0 configuration for a single 8 x H100 80GB node.
EXPECTED_NUM_DEVICES=8
NUM_GENERATIONS=8
BATCH_PER_DEVICE=2
GRAD_ACCUM_STEPS=16
STEPS_PER_GENERATION=16
MAX_COMPLETION_LENGTH=4096
LR=5e-6
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.50
VLLM_MAX_MODEL_LENGTH=8192

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_DEVICES=${#GPU_LIST[@]}
else
    NUM_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi

if [ "$NUM_DEVICES" -ne "$EXPECTED_NUM_DEVICES" ]; then
    echo "P0 requires exactly ${EXPECTED_NUM_DEVICES} visible GPUs; found ${NUM_DEVICES}." >&2
    exit 1
fi

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
MAX_STEPS=600
if [ "$CUR_DATASET" = "MedBookVQA" ]; then
    MAX_STEPS=300
fi

if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then
    if ! [[ "$MAX_STEPS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_STEPS_OVERRIDE must be a positive integer." >&2
        exit 2
    fi
    MAX_STEPS=$MAX_STEPS_OVERRIDE
fi

SAVE_STEPS=100
if [ "$MAX_STEPS" -lt "$SAVE_STEPS" ]; then
    SAVE_STEPS=$MAX_STEPS
fi

if [ "$CUR_DATASET" = "We-Math2" ] || [ "$CUR_DATASET" = "Puzzle" ] || [ "$CUR_DATASET" = "FinMME" ]; then
    IMAGE_MIN_PIXELS=$((64 * 32 * 32))
    IMAGE_MAX_PIXELS=$((256 * 32 * 32))
else
    IMAGE_MIN_PIXELS=$((128 * 32 * 32))
    IMAGE_MAX_PIXELS=$((512 * 32 * 32))
fi

# Every task specialist is initialized independently from the same base model.
MODEL_NAME=$BASE_MODEL
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/checkpoints/Qwen3-VL-30B-A3B-GSPO-Experts}
DATA_PATH=${BASE_PATH}/${CUR_DATASET}/jsons/train/data.json
MEDIA_DIR=${BASE_PATH}/${CUR_DATASET}/images
OUTPUT_DIR=${CHECKPOINT_ROOT}/training/${CUR_DATASET}

if [ ! -f "${MODEL_NAME}/config.json" ]; then
    echo "Base model config not found: ${MODEL_NAME}/config.json" >&2
    exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "Training data not found: ${DATA_PATH}" >&2
    exit 1
fi
if [ ! -d "$MEDIA_DIR" ]; then
    echo "Image directory not found: ${MEDIA_DIR}" >&2
    echo "Extract ${BASE_PATH}/${CUR_DATASET}/images.zip before training." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "Training independent GSPO expert"
echo "Dataset: ${CUR_DATASET}"
echo "Task ID: ${TASK_ID}"
echo "Using model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_DEVICES}"
echo "P0: per-device batch=${BATCH_PER_DEVICE}, gradient accumulation=${GRAD_ACCUM_STEPS}"
echo "Rollout: vLLM colocate, tensor parallel=${VLLM_TENSOR_PARALLEL_SIZE}, GPU memory utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Generation: max completion length=${MAX_COMPLETION_LENGTH}, steps per generation=${STEPS_PER_GENERATION}"
echo "Max steps: ${MAX_STEPS}"
echo "======================================"

deepspeed src/train/train_grpo.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$MEDIA_DIR" \
    --prompt_path src/dataset/prompts_2.yaml \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_model_impl vllm \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_length "$VLLM_MAX_MODEL_LENGTH" \
    --vllm_enable_sleep_mode True \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --learning_rate "$LR" \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \
    --beta 0.0 \
    --temperature 0.8 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --steps_per_generation "$STEPS_PER_GENERATION" \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --importance_sampling_level sequence \
    --loss_type grpo
