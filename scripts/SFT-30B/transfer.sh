#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/project_modelware/zhaojian/zhc/blockdata
SOURCE_ROOT="$BASE/checkpoints/Qwen3-VL-30B-A3B-SFT"

copy_for_vllm() {
    local src="$1"
    local dst="$2"

    # Check the source before rsync removes excluded files from the destination.
    test -f "$src/config.json"
    test -f "$src/model.safetensors.index.json"
    test ! -L "$dst"
    mkdir -p "$dst"

    echo "Copying $src -> $dst"
    # Keep inference files only. Original checkpoints remain untouched.
    # --partial allows an interrupted transfer to reuse partially copied files.
    rsync -ah --partial --info=progress2 \
        --delete --delete-excluded \
        --include='/config.json' \
        --include='/generation_config.json' \
        --include='/model*.safetensors' \
        --include='/model.safetensors.index.json' \
        --include='/tokenizer*.json' \
        --include='/special_tokens_map.json' \
        --include='/added_tokens.json' \
        --include='/vocab.json' \
        --include='/merges.txt' \
        --include='/preprocessor_config.json' \
        --include='/video_preprocessor_config.json' \
        --include='/processor_config.json' \
        --include='/chat_template*' \
        --exclude='*' \
        "$src/" "$dst/"

    echo "Completed: $dst"
    du -sh "$dst"
}

# Final checkpoints from the completed one-epoch runs.
# copy_for_vllm \
#     "$SOURCE_ROOT/MedBookVQA/checkpoint-172" \
#     "$BASE/Qwen3-VL-30B-A3B-Instruct-MedVQA-SFTEpoch1"

# copy_for_vllm \
#     "$SOURCE_ROOT/Navigation/checkpoint-364" \
#     "$BASE/Qwen3-VL-30B-A3B-Instruct-Navigation-SFTEpoch2"

# copy_for_vllm \
#     "$SOURCE_ROOT/We-Math2/checkpoint-716" \
#     "$BASE/Qwen3-VL-30B-A3B-Instruct-WeMath2-SFTEpoch1"

copy_for_vllm \
    "$SOURCE_ROOT/Puzzle/checkpoint-750" \
    "$BASE/Qwen3-VL-30B-A3B-Instruct-Puzzle-SFTEpoch1"

copy_for_vllm \
    "$SOURCE_ROOT/FinMME/checkpoint-1005" \
    "$BASE/Qwen3-VL-30B-A3B-Instruct-FinMME-SFTEpoch1"
