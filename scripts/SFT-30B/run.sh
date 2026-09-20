#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONDA_SH=${CONDA_SH:-/mnt/project_modelware/zhaojian/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-trlQwen}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_MODEL=${BASE_MODEL:-/mnt/project_modelware/zhaojian/zhc/blockdata/Qwen3-VL-30B-A3B-Instruct}
BASE_PATH=${BASE_PATH:-/mnt/project_modelware/zhaojian/zhc/blockdata/MRCL}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/project_modelware/zhaojian/zhc/blockdata/checkpoints/Qwen3-VL-30B-A3B-SFT}
# Let DeepSpeed honor explicit GPU IDs without overriding them with --num_gpus.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${NUM_GPUS:-${#GPU_IDS[@]}}
    [[ "$NUM_GPUS" == "${#GPU_IDS[@]}" ]] || {
        echo "NUM_GPUS must match the number of CUDA_VISIBLE_DEVICES entries" >&2
        exit 2
    }
    LAUNCH_ARGS=()
else
    NUM_GPUS=${NUM_GPUS:-4}
    LAUNCH_ARGS=(--num_gpus "$NUM_GPUS")
fi
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
for value in "$NUM_GPUS" "$GLOBAL_BATCH_SIZE" "$BATCH_PER_DEVICE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "GPU and batch sizes must be positive integers" >&2; exit 2; }
done
DENOMINATOR=$((NUM_GPUS * BATCH_PER_DEVICE))
(( GLOBAL_BATCH_SIZE >= DENOMINATOR && GLOBAL_BATCH_SIZE % DENOMINATOR == 0 )) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * BATCH_PER_DEVICE" >&2
    exit 2
}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / DENOMINATOR))
PROMPT_PATH="$REPO_ROOT/src/dataset/prompts_1.yaml"
read -ra SELECTED_TASKS <<< "${TASK_IDS:-1}"

for TASK_ID in "${SELECTED_TASKS[@]}"; do
    case "$TASK_ID" in
        1) TASK=MedBookVQA; EPOCHS=2; MIN_PIXELS=$((128*32*32)); MAX_PIXELS=$((512*32*32)) ;;
        2) TASK=Navigation; EPOCHS=3; MIN_PIXELS=$((128*32*32)); MAX_PIXELS=$((512*32*32)) ;;
        3) TASK=We-Math2; EPOCHS=1; MIN_PIXELS=$((64*32*32)); MAX_PIXELS=$((256*32*32)) ;;
        4) TASK=Puzzle; EPOCHS=1; MIN_PIXELS=$((64*32*32)); MAX_PIXELS=$((256*32*32)) ;;
        5) TASK=FinMME; EPOCHS=1; MIN_PIXELS=$((64*32*32)); MAX_PIXELS=$((256*32*32)) ;;
        *) echo "TASK_IDS only accepts 1..5" >&2; exit 2 ;;
    esac
    OUTPUT_DIR="$CHECKPOINT_ROOT/$TASK"
    LOG_DIR="$CHECKPOINT_ROOT/logs/$TASK"
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
    echo "SFT $TASK: base=$BASE_MODEL, GPUs=$NUM_GPUS, global_batch=$GLOBAL_BATCH_SIZE, accumulation=$GRAD_ACCUM_STEPS"
    echo "Template=$PROMPT_PATH, output=$OUTPUT_DIR (existing checkpoints resume automatically)"
    deepspeed "${LAUNCH_ARGS[@]}" \
        "$REPO_ROOT/src/train/train_sft.py" \
        --deepspeed "$SCRIPT_DIR/zero3_optimizer_offload.json" \
        --model_id "$BASE_MODEL" \
        --data_path "$BASE_PATH/$TASK/jsons/train/data.json" \
        --image_folder "$BASE_PATH/$TASK/images" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --logging_dir "$LOG_DIR/tensorboard" \
        --use_liger_kernel False --lora_enable False \
        --freeze_vision_tower False --freeze_llm False --freeze_merger False \
        --bf16 True --fp16 False --tf32 True --disable_flash_attn2 False \
        --gradient_checkpointing True \
        --per_device_train_batch_size "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
        --num_train_epochs "$EPOCHS" \
        --max_seq_length 4096 \
        --image_min_pixels "$MIN_PIXELS" --image_max_pixels "$MAX_PIXELS" \
        --learning_rate 5e-6 --weight_decay 0.1 \
        --warmup_ratio 0.03 --lr_scheduler_type cosine \
        --remove_unused_columns False --lazy_preprocess True \
        --logging_steps 1 --report_to tensorboard \
        --save_strategy steps --save_steps 200 --save_total_limit 2 \
        --dataloader_num_workers 4 \
        "$@" 2>&1 | tee "$LOG_DIR/train.log"
done
