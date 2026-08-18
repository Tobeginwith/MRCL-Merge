#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-/mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh}
ROBUSTMERGE_ENV=${ROBUSTMERGE_ENV:-/mnt/project_modelware_roce_hs/zhaojian/zhc/conda_envs/robustmerge-moe}
TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-/mnt/project_modelware_roce_hs/zhaojian/zhc/conda_envs/te-build}
VERL_SRC=${VERL_SRC:-/mnt/project_modelware_roce_hs/zhaojian/zhc/MOPD/verl-v0.8.0-mrcl}

source "$CONDA_SH"
conda activate "$ROBUSTMERGE_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ -d "$TOOLCHAIN_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$TOOLCHAIN_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$VERL_SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0

if [[ ${1:-} == prepare ]]; then
    shift
    exec python "$SCRIPT_DIR/data_and_reward.py" "$@"
fi

BASE_MODEL=${BASE_MODEL:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/Qwen3-VL-30B-A3B-Instruct}
BASE_PATH=${BASE_PATH:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/MRCL}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/project_modelware_roce_hs/zhaojian/zhc/blockdata/checkpoints/Qwen3-VL-30B-A3B-GSPO-Experts-verl}
PROJECT_NAME=${PROJECT_NAME:-qwen3_vl_30b_a3b_gspo_experts_verl}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}

USE_DYNAMIC_BSZ=false
BALANCE_BATCH=false
PPO_MAX_TOKEN_LEN_PER_GPU=8192
ROLLOUT_GPU_MEMORY_UTILIZATION=0.50

for TASK_ID in ${TASK_IDS:-1}; do
    case "$TASK_ID" in
        1) TASK=MedBookVQA; MAX_STEPS=150; IMAGE_MIN_PIXELS=$((128*32*32)); IMAGE_MAX_PIXELS=$((512*32*32)) ;;
        2) TASK=Navigation; MAX_STEPS=150; IMAGE_MIN_PIXELS=$((128*32*32)); IMAGE_MAX_PIXELS=$((512*32*32)) ;;
        3) TASK=We-Math2; MAX_STEPS=600; IMAGE_MIN_PIXELS=$((64*32*32)); IMAGE_MAX_PIXELS=$((256*32*32)) ;;
        4) TASK=Puzzle; MAX_STEPS=600; IMAGE_MIN_PIXELS=$((64*32*32)); IMAGE_MAX_PIXELS=$((256*32*32)) ;;
        5) TASK=FinMME; MAX_STEPS=600; IMAGE_MIN_PIXELS=$((64*32*32)); IMAGE_MAX_PIXELS=$((256*32*32)) ;;
        *) echo "TASK_IDS only accepts 1..5" >&2; exit 2 ;;
    esac
    MAX_STEPS=${MAX_STEPS_OVERRIDE:-$MAX_STEPS}
    TRAIN_FILE="$BASE_PATH/$TASK/train.parquet"
    TEST_FILE="$BASE_PATH/$TASK/test.parquet"
    OUTPUT_DIR="$CHECKPOINT_ROOT/$TASK"
    LOG_DIR="$CHECKPOINT_ROOT/logs/$TASK"
    export TENSORBOARD_DIR="$CHECKPOINT_ROOT/tensorboard/$TASK"
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"

    echo "Starting $TASK: steps=$MAX_STEPS, train=$TRAIN_FILE, val=$TEST_FILE, output=$OUTPUT_DIR"
    DATA=(
        algorithm.adv_estimator=grpo
        algorithm.norm_adv_by_std_in_grpo=True
        algorithm.use_kl_in_reward=False
        data.train_files="$TRAIN_FILE"
        data.val_files="$TEST_FILE"
        data.train_batch_size=32
        data.max_prompt_length=1024
        data.max_response_length=4096
        data.filter_overlong_prompts=False
        data.truncation=error
        data.image_patch_size=16
        +data.mm_processor_kwargs.min_pixels="$IMAGE_MIN_PIXELS"
        +data.mm_processor_kwargs.max_pixels="$IMAGE_MAX_PIXELS"
    )

    MODEL=(
        actor_rollout_ref.model.path="$BASE_MODEL"
        actor_rollout_ref.model.use_remove_padding=True
    )

    ACTOR=(
        actor_rollout_ref.actor.freeze_vision_tower=False
        actor_rollout_ref.actor.policy_loss.loss_mode=gspo
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
        actor_rollout_ref.actor.clip_ratio_low=0.0003
        actor_rollout_ref.actor.clip_ratio_high=0.0004
        actor_rollout_ref.actor.entropy_coeff=0.0
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.ppo_mini_batch_size=32
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ"
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
        actor_rollout_ref.actor.optim.lr=0.000005
        actor_rollout_ref.actor.optim.weight_decay=0.1
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
        actor_rollout_ref.actor.optim.lr_decay_style=cosine
        actor_rollout_ref.actor.optim.min_lr=0.0
        actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
        actor_rollout_ref.actor.optim.clip_grad=1.0
        actor_rollout_ref.actor.optim.total_training_steps="$MAX_STEPS"
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
        actor_rollout_ref.actor.megatron.context_parallel_size=2
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
        actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
        actor_rollout_ref.actor.megatron.use_mbridge=True
        actor_rollout_ref.actor.megatron.vanilla_mbridge=True
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.grad_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=none
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.0
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    )

    ROLLOUT=(
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size=4
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
        actor_rollout_ref.rollout.max_model_len=8192
        actor_rollout_ref.rollout.n=8
        actor_rollout_ref.rollout.do_sample=True
        actor_rollout_ref.rollout.temperature=0.8
        actor_rollout_ref.rollout.top_p=1.0
        actor_rollout_ref.rollout.top_k=-1
        +actor_rollout_ref.rollout.repetition_penalty=1.0
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ"
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
        actor_rollout_ref.rollout.val_kwargs.n=4
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0
        actor_rollout_ref.rollout.val_kwargs.top_k=-1
    )

    REF=()


    REWARD=(
        reward.custom_reward_function.path="$SCRIPT_DIR/data_and_reward.py"
        reward.custom_reward_function.name=compute_score
        reward.reward_manager.name=dapo
        +reward.reward_kwargs.overlong_buffer_cfg.enable=True
        +reward.reward_kwargs.overlong_buffer_cfg.len=2048
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
        +reward.reward_kwargs.overlong_buffer_cfg.log=False
        +reward.reward_kwargs.max_resp_len=4096
    )

    EXTRA=(
        model_engine=megatron
    )

    TRAINER=(
        trainer.balance_batch="$BALANCE_BATCH"
        trainer.logger='[console,tensorboard]'
        trainer.project_name="$PROJECT_NAME"
        trainer.experiment_name="$TASK"
        trainer.nnodes=1
        trainer.n_gpus_per_node=8
        trainer.total_training_steps="$MAX_STEPS"
        trainer.val_before_train=True
        trainer.test_freq=50
        trainer.save_freq=50
        trainer.default_local_dir="$OUTPUT_DIR"
        trainer.log_val_generations=8
        trainer.resume_mode="$RESUME_MODE"
        trainer.resume_from_path="$RESUME_FROM_PATH"
    )

    python -m verl.trainer.main_ppo \
        "${DATA[@]}" \
        "${MODEL[@]}" \
        "${ACTOR[@]}" \
        "${ROLLOUT[@]}" \
        "${REF[@]}" \
        "${REWARD[@]}" \
        "${TRAINER[@]}" \
        "${EXTRA[@]}" \
        "$@" 2>&1 | tee "$LOG_DIR/train.log"
done
