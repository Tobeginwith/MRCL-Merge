# Qwen3-VL-30B-A3B verl GSPO

该实现复用现有 `robustmerge-moe` 环境，只包含 4 个文件。固定源码版本为：

- verl `release/v0.8.0`：`3e4edb6e3d6872ad8aa21af83e98ee6e1bea19a2`
- MBridge：`a61943d7fcb34a190471cfeb0a0eb8bbda621ddf`

安装依赖、准备 verl 源码并安装固定 MBridge 提交：

```bash
bash scripts/GSPO-Experts-30B-verl/setup.sh
```

如果不能访问 GitHub，可通过 `VERL_REPOSITORY` 指向本地 verl Git 仓库。MBridge 直接安装进现有环境，不再需要单独设置源码路径。脚本不会安装另一套 PyTorch、Megatron 或 vLLM，也不会克隆 Conda 环境。

转换全部数据；验证集直接使用完整 `test.parquet`：

```bash
bash scripts/GSPO-Experts-30B-verl/run.sh prepare --tasks all
```

训练一个专家或依次训练全部专家：

```bash
TASK_IDS=1 bash scripts/GSPO-Experts-30B-verl/run.sh
TASK_IDS="1 2 3 4 5" bash scripts/GSPO-Experts-30B-verl/run.sh
```

编号依次为 MedBookVQA、Navigation、We-Math2、Puzzle、FinMME，训练参数固定使用 strict 配置。

训练使用 32 prompt × 8 completion、GSPO、actor TP2/CP2/EP8、rollout TP4、CPU
parameter/gradient/optimizer offload。训练前、每 100 step 和结束时使用完整 test 集验证；每
100 step 和结束时保存 checkpoint，并只保留最新一个。默认 `RESUME_MODE=auto`，也可显式续跑：

```bash
TASK_IDS=1 RESUME_MODE=resume_path \
RESUME_FROM_PATH=/absolute/path/global_step_100 \
bash scripts/GSPO-Experts-30B-verl/run.sh
```

模型、MRCL 数据根目录和输出根目录可分别通过 `BASE_MODEL`、`BASE_PATH`、`CHECKPOINT_ROOT` 覆盖；
额外的 Hydra 参数可直接追加到 `run.sh` 命令末尾。
