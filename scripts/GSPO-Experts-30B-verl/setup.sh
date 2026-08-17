#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-/mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh}
ROBUSTMERGE_ENV=${ROBUSTMERGE_ENV:-/mnt/project_modelware_roce_hs/zhaojian/zhc/conda_envs/robustmerge-moe}
TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-/mnt/project_modelware_roce_hs/zhaojian/zhc/conda_envs/te-build}
VERL_SRC=${VERL_SRC:-/mnt/project_modelware_roce_hs/zhaojian/zhc/MOPD/verl-v0.8.0-mrcl}
VERL_REPOSITORY=${VERL_REPOSITORY:-https://github.com/verl-project/verl.git}
VERL_COMMIT=3e4edb6e3d6872ad8aa21af83e98ee6e1bea19a2
MBRIDGE_COMMIT=a61943d7fcb34a190471cfeb0a0eb8bbda621ddf

source "$CONDA_SH"
conda activate "$ROBUSTMERGE_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ -d "$TOOLCHAIN_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$TOOLCHAIN_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ ! -d "$VERL_SRC/.git" ]]; then
    mkdir -p "$(dirname "$VERL_SRC")"
    git clone --filter=blob:none --branch release/v0.8.0 "$VERL_REPOSITORY" "$VERL_SRC"
fi
git -C "$VERL_SRC" diff --quiet
git -C "$VERL_SRC" diff --cached --quiet
git -C "$VERL_SRC" fetch origin "$VERL_COMMIT"
git -C "$VERL_SRC" checkout --detach "$VERL_COMMIT"

python -m pip install --no-deps --upgrade --force-reinstall \
    qwen-vl-utils==0.0.14 \
    math-verify==0.8.0 \
    latex2sympy2-extended==1.10.2 \
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_COMMIT"

MBRIDGE_PACKAGE_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("mbridge").submodule_search_locations)))')
sed -i 's/packed_seq_params\.cp_group is not None/getattr(packed_seq_params, "cp_group", None) is not None/' \
    "$MBRIDGE_PACKAGE_DIR/models/qwen3_vl/model.py"

echo "Ready: verl=$VERL_SRC, MBridge=$MBRIDGE_COMMIT, env=$CONDA_PREFIX"
