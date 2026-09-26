#!/usr/bin/env python3
"""
Merge multiple teacher checkpoints into one HuggingFace-compatible model.

Supported methods:
  - ta: Task Arithmetic
        θ = θ_base + Σ λ_i · (θ_teacher_i − θ_base)
  - ties: TIES Merging
  - tsvm: Task Singular Vector Merging with randomized low-rank task SVD and rank truncation
  - wudi: WUDI linear-weight merging with streamed Gram statistics and Adam
  - ram_plus: RAM+ with global overlap-aware unique update scaling
  - iso_c: Isotropic Merging in Common Subspace (Iso-C)
  - dc: DC-Merge (FFT), randomized low-rank task SVD and TIES in shared cover space

Example:
    python src/merge/merge.py ta \\
        --base /path/Qwen3-VL-30B-A3B-Instruct \\
        --teachers /path/math /path/code /path/logic \\
        --scale 1.0 \\
        --output /path/merged_ta

    python src/merge/merge.py ties \\
        --base /path/Qwen3-VL-30B-A3B-Instruct \\
        --teachers /path/math /path/code /path/logic \\
        --ties-density 0.2 \\
        --scale 1.0 \\
        --output /path/merged_ties

    python src/merge/merge.py tsvm \\
        --base /path/Qwen3-VL-30B-A3B-Instruct \\
        --teachers /path/math /path/code /path/logic \\
        --device cuda:0 --scale 1.0 \\
        --output /path/merged_tsvm

TSVM follows https://arxiv.org/abs/2412.00081 and the orthogonalization variant:
https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/utils/TSVM_utils.py
Task SVDs use torch.svd_lowrank with q=min(k+oversampling,min(m,n)),
niter=2 and oversampling=8 by default. Factor orthogonalization continues to
use standard SVD (full_matrices=False).
Task factors are then truncated to floor(min(m,n)/num_teachers) components. Non-matrix floating tensors use the mean task delta.
GPU work is limited to one matrix at a time. CUDA OOM propagates to the caller.

WUDI follows https://arxiv.org/abs/2503.08099 and
https://github.com/nathanielyvo/WUDI-Merging/blob/main/vit/wudi_main.py
Use: python src/merge/merge.py wudi --base BASE --teachers T1 T2
     --device cuda:0 --wudi-steps 300 --wudi-lr 1e-5 --output OUTPUT
Only linear weights are optimized; embeddings and other parameters retain base
values. Packed Qwen projections are transposed to [output,input] for WUDI.

DC-Merge follows https://arxiv.org/abs/2603.06242 (official FFT recipe).
Use: python src/merge/merge.py dc --base BASE --teachers T1 T2
     --device cuda:0 --dc-density 0.001 --scale 1.0 --output OUTPUT
All floating matrices, including embeddings, use randomized low-rank task SVD then truncation;
cover bases use reduced standard SVD. Nonmatrix floats use mean task delta.
No explicit LoRA energy smoothing is applied. OOM propagates without retry.

Iso-C follows https://arxiv.org/abs/2502.04959 and
https://github.com/danielm1405/iso-merging/blob/main/src/utils/iso.py
Use: python src/merge/merge.py iso_c --base BASE --teachers T1 T2
     --device cuda:0 --scale 1.0 --output OUTPUT
All floating matrices (including text_projection) use the summed task matrix's
reduced SVD, replacing every singular value by their mean, without truncation.
Nonmatrix floats use the CPU mean task delta, as in TSVM/DC. Matrix computation
requires CUDA; OOM propagates without retry or CPU fallback.

RAM+ follows the official arm-r-v2 code (not the paper's different scaling rule):
https://github.com/xiangchi-yuan/mrl/blob/main/ram-main.py
Use: python src/merge/merge.py ram_plus --base BASE --teachers T1 T2
     --device cuda:0 --ram-threshold 1e-5 --ram-rescale-factor 1.2 --output OUTPUT
All floating tensors use active-only averaging in shared regions and task-specific
scaling in unique regions. Global shared/unique counts determine each scale as
1 + (max(1,rescale_factor)-1) * min(shared/unique,1). With no unique updates,
the ratio is 1 if shared updates exist, otherwise 0. Statistics and merging run
on CUDA in separate streaming passes. OOM propagates without CPU fallback.

Loading strategy:
  - Index safetensors metadata without loading complete checkpoints.
  - Read one base/teacher tensor at a time and merge it in float32.
  - Buffer only one output shard, write it directly, then save the processor.

Packed Qwen experts are merged independently per expert and projection (gate,
up, down). TIES trims each such projection separately; other tensors are trimmed
as a whole. Output keys and packed layouts are preserved.

Requires Python 3.10+ (tested with 3.12).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections.abc import Callable, Iterable
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor


def _shard_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)-of-\d+\.", path.name)
    if match:
        return int(match.group(1)), path.name
    return 0, path.name


def _find_safetensors_index(model_dir: Path) -> Path | None:
    candidate = model_dir / "model.safetensors.index.json"
    if candidate.is_file():
        return candidate
    matches = sorted(model_dir.glob("*.safetensors.index.json"))
    return matches[0] if matches else None


class SafetensorCheckpoint:
    """Metadata index plus lazy, per-tensor access to one HF safetensors checkpoint."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.weight_map: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self._ignored_keys: set[str] = set()
        self._stack = ExitStack()
        self._handles: dict[str, object] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        index_path = _find_safetensors_index(self.model_dir)
        if index_path is not None:
            with index_path.open(encoding="utf-8") as f:
                index = json.load(f)
            raw_weight_map = index.get("weight_map")
            if not isinstance(raw_weight_map, dict) or not raw_weight_map:
                raise ValueError(f"Invalid or empty weight_map in {index_path}")
            self.weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
            shard_names = sorted(set(self.weight_map.values()))
        else:
            single_path = self.model_dir / "model.safetensors"
            if single_path.is_file():
                shard_names = [single_path.name]
            else:
                shard_paths = sorted(
                    self.model_dir.glob("model-*-of-*.safetensors"),
                    key=_shard_sort_key,
                )
                shard_names = [path.name for path in shard_paths]
            if not shard_names:
                raise FileNotFoundError(
                    f"No safetensors weights found under {self.model_dir}. Expected "
                    "model.safetensors or model-*-of-*.safetensors with an optional index."
                )

        discovered_weight_map: dict[str, str] = {}
        for shard_name in shard_names:
            shard_path = self.model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing shard file referenced by checkpoint: {shard_path}"
                )
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in discovered_weight_map:
                        raise ValueError(
                            f"Duplicate tensor key {key!r} in {self.model_dir}: "
                            f"{discovered_weight_map[key]} and {shard_name}"
                        )
                    tensor_slice = handle.get_slice(key)
                    discovered_weight_map[key] = shard_name
                    self.shapes[key] = tuple(tensor_slice.get_shape())
                    self.dtypes[key] = str(tensor_slice.get_dtype())

        if self.weight_map:
            indexed_keys = set(self.weight_map)
            discovered_keys = set(discovered_weight_map)
            if indexed_keys != discovered_keys:
                missing = sorted(indexed_keys - discovered_keys)
                extra = sorted(discovered_keys - indexed_keys)
                raise ValueError(
                    f"Safetensors index mismatch in {self.model_dir}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            for key, actual_shard in discovered_weight_map.items():
                indexed_shard = self.weight_map[key]
                if indexed_shard != actual_shard:
                    raise ValueError(
                        f"Safetensors index maps {key!r} to {indexed_shard}, "
                        f"but it is stored in {actual_shard}"
                    )
        else:
            self.weight_map = discovered_weight_map

        print(
            f"  indexed {len(self.weight_map)} tensors from "
            f"{len(set(self.weight_map.values()))} safetensors shard(s)"
        )

    def keys(self) -> list[str]:
        return [key for key in self.weight_map if key not in self._ignored_keys]

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map and key not in self._ignored_keys

    def get_shape(self, key: str) -> tuple[int, ...]:
        if key not in self:
            raise KeyError(key)
        return self.shapes[key]

    def get_dtype(self, key: str) -> str:
        if key not in self:
            raise KeyError(key)
        return self.dtypes[key]

    def _get_handle(self, key: str):
        if key not in self:
            raise KeyError(key)
        shard_name = self.weight_map[key]
        handle = self._handles.get(shard_name)
        if handle is None:
            handle = self._stack.enter_context(
                safe_open(
                    str(self.model_dir / shard_name),
                    framework="pt",
                    device="cpu",
                )
            )
            self._handles[shard_name] = handle
        return handle

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._get_handle(key).get_tensor(key)

    def get_slice(self, key: str, index: tuple[slice, ...]) -> torch.Tensor:
        return self._get_handle(key).get_slice(key)[index]

    def discard(self, key: str) -> None:
        if key in self.weight_map:
            self._ignored_keys.add(key)

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()


def load_state_dict(model_dir: str | Path) -> SafetensorCheckpoint:
    """Index a HF safetensors checkpoint without loading its tensors into RAM."""
    return SafetensorCheckpoint(model_dir)


def _read_tie_word_embeddings(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    return bool(config.get("tie_word_embeddings", False))


def _strip_redundant_lm_head(
    reference: SafetensorCheckpoint | dict[str, torch.Tensor],
    state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    names: list[str],
) -> None:
    """Drop lm_head.weight when it duplicates embed_tokens (common in verl FSDP exports)."""
    if "lm_head.weight" in reference:
        return

    embed_key = "model.embed_tokens.weight"
    if len(names) != len(state_dicts):
        raise ValueError(f"Expected {len(state_dicts)} state dict names, got {len(names)}")
    for name, sd in zip(names, state_dicts):
        if "lm_head.weight" not in sd:
            continue
        if embed_key not in sd:
            raise ValueError(f"{name} has lm_head.weight but missing {embed_key}")
        embed_tensor = _get_state_tensor(sd, embed_key)
        lm_head_tensor = _get_state_tensor(sd, "lm_head.weight")
        if not torch.equal(embed_tensor, lm_head_tensor):
            raise ValueError(
                f"{name}: lm_head.weight != {embed_key}; refusing to strip non-tied lm_head"
            )
        if isinstance(sd, SafetensorCheckpoint):
            sd.discard("lm_head.weight")
        else:
            sd.pop("lm_head.weight")
        print(f"  stripped redundant lm_head.weight from {name}")


def _state_keys(
    state: SafetensorCheckpoint | dict[str, torch.Tensor],
) -> list[str]:
    return state.keys() if isinstance(state, SafetensorCheckpoint) else list(state.keys())


def _get_state_tensor(
    state: SafetensorCheckpoint | dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    return state.get_tensor(key) if isinstance(state, SafetensorCheckpoint) else state[key]


def _get_state_shape(
    state: SafetensorCheckpoint | dict[str, torch.Tensor],
    key: str,
) -> tuple[int, ...]:
    if isinstance(state, SafetensorCheckpoint):
        return state.get_shape(key)
    return tuple(state[key].shape)


def _validate_compatible_state_dicts(
    state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    names: list[str],
) -> None:
    if not state_dicts:
        raise ValueError("At least one state dict is required.")
    if len(state_dicts) != len(names):
        raise ValueError(
            f"Expected {len(state_dicts)} state dict names, got {len(names)}"
        )

    ref_keys = set(_state_keys(state_dicts[0]))
    for name, sd in zip(names[1:], state_dicts[1:]):
        keys = set(_state_keys(sd))
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            msg = [f"State dict mismatch for {name}."]
            if missing:
                msg.append(f"  missing keys ({len(missing)}): {missing[:5]}...")
            if extra:
                msg.append(f"  extra keys ({len(extra)}): {extra[:5]}...")
            raise ValueError("\n".join(msg))

    ref = state_dicts[0]
    for name, sd in zip(names[1:], state_dicts[1:]):
        for key in _state_keys(ref):
            ref_shape = _get_state_shape(ref, key)
            other_shape = _get_state_shape(sd, key)
            if ref_shape != other_shape:
                raise ValueError(
                    f"Shape mismatch at {key}: {names[0]} {ref_shape} "
                    f"vs {name} {other_shape}"
                )


def _merge_ta_tensor(
    base_tensor: torch.Tensor,
    teacher_tensors: Iterable[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    base_float = base_tensor.to(dtype=torch.float32, copy=True)
    merged_delta = torch.zeros_like(base_float)
    for teacher_tensor in teacher_tensors:
        delta = teacher_tensor.to(dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        merged_delta.add_(delta)
        del delta, teacher_tensor
    base_float.add_(merged_delta, alpha=scale)
    del merged_delta
    return base_float.to(base_tensor.dtype)


def merge_ta(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """Apply standard Task Arithmetic to every tensor."""
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError("Streaming TA requires safetensors checkpoint readers for all teachers")
        return _StreamingMergedStateDict(
            base_state_dict,
            teacher_state_dicts,
            method="ta",
            scale=scale,
        )

    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "ta", scale, 1.0, packed,
        )
        for key in base_state_dict
    }


def _topk_magnitude_mask(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the largest-magnitude entries of one task vector."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"ties density must be in (0, 1], got {density}")
    if delta.numel() == 0:
        return torch.zeros_like(delta, dtype=torch.bool)
    if density == 1.0:
        return torch.ones_like(delta, dtype=torch.bool)

    k = max(1, int(delta.numel() * density))
    flat_abs = delta.abs().reshape(-1)
    threshold_index = flat_abs.numel() - k + 1
    threshold = torch.kthvalue(flat_abs, threshold_index).values
    return delta.abs() >= threshold


def _merge_ties_tensor(
    base_tensor: torch.Tensor,
    teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
    density: float,
    scale: float,
) -> torch.Tensor:
    if not torch.is_floating_point(base_tensor):
        return base_tensor.clone()

    base_float = base_tensor.to(dtype=torch.float32, copy=True)
    sign_votes = torch.zeros_like(base_float)

    # First pass: trim each task vector and elect the dominant sign.
    for teacher_tensor in teacher_tensor_factory():
        delta = teacher_tensor.to(dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        trim_mask = _topk_magnitude_mask(delta, density)
        delta.masked_fill_(~trim_mask, 0.0)
        sign_votes.add_(delta)
        del delta, trim_mask, teacher_tensor

    elected_sign = torch.sign(sign_votes)
    del sign_votes
    aligned_sum = torch.zeros_like(base_float)
    aligned_count = torch.zeros_like(base_float)

    # Second pass: reread one teacher tensor at a time and reduce aligned updates.
    for teacher_tensor in teacher_tensor_factory():
        delta = teacher_tensor.to(dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        trim_mask = _topk_magnitude_mask(delta, density)
        aligned_mask = (
            trim_mask
            & (torch.sign(delta) == elected_sign)
            & (elected_sign != 0)
        )
        aligned_count.add_(aligned_mask.to(dtype=aligned_count.dtype))
        delta.masked_fill_(~aligned_mask, 0.0)
        aligned_sum.add_(delta)
        del delta, trim_mask, aligned_mask, teacher_tensor

    # aligned_sum is already zero where aligned_count is zero, so clamping the
    # divisor produces the same result without allocating another full tensor.
    aligned_count.clamp_min_(1.0)
    aligned_sum.div_(aligned_count)
    base_float.add_(aligned_sum, alpha=scale)
    del aligned_sum, aligned_count, elected_sign
    return base_float.to(base_tensor.dtype)


def merge_ties(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    density: float = 0.2,
    scale: float = 1.0,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """
    TIES Merging:
      1. build task vectors teacher_i - base
      2. trim each expert projection (or ordinary tensor) independently
      3. elect the dominant sign at each parameter by summing trimmed vectors
      4. average only trimmed updates that match the elected sign
      5. add scale * merged_delta back to the base model

    Defaults follow the no-validation recipe from the TIES-Merging paper:
    top-20% task-vector entries, mass-based sign election, disjoint mean, and
    lambda/scale = 1.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"ties density must be in (0, 1], got {density}")
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError(
                "Streaming TIES requires safetensors checkpoint readers for all teachers"
            )
        return _StreamingMergedStateDict(
            base_state_dict,
            teacher_state_dicts,
            method="ties",
            density=density,
            scale=scale,
        )

    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "ties", scale, density, packed,
        )
        for key in base_state_dict
    }



@contextmanager
def _tsvm_fp32_matmul():
    """Disable TF32 locally without changing precision for subsequent callers."""
    previous = torch.get_float32_matmul_precision()
    try:
        if previous != "highest":
            torch.set_float32_matmul_precision("highest")
        yield
    finally:
        if previous != "highest":
            torch.set_float32_matmul_precision(previous)


def _tsvm_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes factor, including the official zero padding."""
    left, _, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


@torch.inference_mode()
def _merge_tsvm_tensor(
    base_tensor: torch.Tensor,
    teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
    *,
    num_teachers: int,
    scale: float,
    device: torch.device,
    identity: str,
    niter: int = 2,
    oversampling: int = 8,
) -> torch.Tensor:
    """Return CPU storage weights without modifying base or teacher tensors.

    GPU intermediates live only within this call; allocation failures propagate.
    """
    if not torch.is_floating_point(base_tensor):
        return base_tensor.detach().to(device="cpu", copy=True)
    if not torch.isfinite(base_tensor).all():
        raise FloatingPointError("Non-finite base weights")
    is_matrix = base_tensor.ndim == 2
    rank = min(base_tensor.shape) if is_matrix else 0
    keep = rank // num_teachers
    if scale == 0 or base_tensor.numel() == 0 or (is_matrix and keep == 0):
        return base_tensor.detach().to(device="cpu", copy=True)

    base_float = base_tensor.to(device=device, dtype=torch.float32, copy=True)
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite base weights after FP32 conversion")
    if is_matrix:
        m, n = base_tensor.shape
        # Match the official padded factor dimensions even when rank % T != 0.
        sum_u = torch.zeros((m, rank), device=device, dtype=torch.float32)
        sum_s = torch.zeros(rank, device=device, dtype=torch.float32)
        sum_vh = torch.zeros((rank, n), device=device, dtype=torch.float32)
    else:
        mean_delta = torch.zeros_like(base_float)

    count = 0
    for teacher_index, teacher_tensor in enumerate(teacher_tensor_factory()):
        if teacher_index >= num_teachers:
            raise ValueError("Too many teacher tensors")
        if tuple(teacher_tensor.shape) != tuple(base_tensor.shape):
            raise ValueError(f"Teacher {teacher_index}: incompatible tensor shape")
        delta = teacher_tensor.to(device=device, dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        if not torch.isfinite(delta).all():
            raise FloatingPointError(f"Teacher {teacher_index}: non-finite FP32 task delta")
        if is_matrix:
            u, singular, v = torch.svd_lowrank(
                delta, q=min(keep + oversampling, rank), niter=niter,
            )
            vh = v.mT
            start = teacher_index * keep
            # Copy into independent storage; sliced views must not keep full
            # full SVD decompositions alive across teacher iterations.
            sum_u[:, start:start + keep].copy_(u[:, :keep])
            sum_s[start:start + keep].copy_(singular[:keep])
            sum_vh[start:start + keep].copy_(vh[:keep, :])
            del u, singular, v, vh
        else:
            # Match the official online mean for non-matrix task vectors.
            delta.sub_(mean_delta).div_(teacher_index + 1)
            mean_delta.add_(delta)
        del delta, teacher_tensor
        count += 1
    if count != num_teachers:
        raise ValueError(f"Expected {num_teachers} teacher tensors, got {count}")

    if is_matrix:
        sum_u = _tsvm_polar(sum_u)
        sum_vh = _tsvm_polar(sum_vh)
        sum_u.mul_(sum_s.unsqueeze(0))
        merged_delta = sum_u @ sum_vh
        del sum_u, sum_s, sum_vh
    else:
        merged_delta = mean_delta
    base_float.add_(merged_delta, alpha=scale)
    del merged_delta
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite TSVM merged weights")
    stored = base_float.to(device="cpu", dtype=base_tensor.dtype)
    if not torch.isfinite(stored).all():
        raise FloatingPointError(f"TSVM merged weights overflow storage dtype {base_tensor.dtype}")
    return stored


class _TSVMRuntime:
    """Configuration and scalar diagnostics; never retain merged tensor weights."""

    def __init__(self, *, num_teachers: int, scale: float, device: str | torch.device,
                 niter: int = 2, oversampling: int = 8):
        if num_teachers <= 0:
            raise ValueError("TSVM requires at least one teacher")
        if not math.isfinite(scale):
            raise ValueError("TSVM scale must be finite")
        for name, value in (("niter", niter), ("oversampling", oversampling)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TSVM {name} must be a nonnegative integer")
        self.niter = niter
        self.oversampling = oversampling
        if str(device) == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type not in ("cpu", "cuda"):
            raise ValueError("TSVM device must be auto, cpu, or a CUDA device")
        if resolved.type == "cpu" and resolved.index is not None:
            raise ValueError("Use cpu without a device index")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("TSVM requested CUDA, but CUDA is unavailable")
            index = torch.cuda.current_device() if resolved.index is None else resolved.index
            if index >= torch.cuda.device_count():
                raise ValueError(f"Invalid CUDA device index: {index}")
            resolved = torch.device("cuda", index)
            # Initialize CUDA before resetting allocator statistics in a fresh process.
            torch.cuda.get_device_properties(resolved)
            torch.cuda.reset_peak_memory_stats(resolved)
        self.device = resolved
        self.num_teachers = num_teachers
        self.scale = scale
        self.completed_parts = 0
        self.cuda_parts = 0
        self.cpu_parts = 0
        self.elapsed_seconds = 0.0

    def merge(self, base_tensor: torch.Tensor,
              teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
              identity: str) -> torch.Tensor:
        # Biases, norms, convolution tensors, and copy-only cases stay on CPU.
        use_svd = (torch.is_floating_point(base_tensor) and base_tensor.ndim == 2
                   and min(base_tensor.shape) // self.num_teachers > 0 and self.scale != 0)
        device = self.device if use_svd else torch.device("cpu")
        started = time.perf_counter()
        kwargs = dict(num_teachers=self.num_teachers, scale=self.scale,
                      identity=identity, niter=self.niter, oversampling=self.oversampling)
        try:
            with _tsvm_fp32_matmul():
                result = _merge_tsvm_tensor(
                    base_tensor, teacher_tensor_factory, device=device, **kwargs,
                )
        except torch.cuda.OutOfMemoryError as exc:
            # Preserve the original exception type and traceback; never retry.
            exc.args = (f"TSVM OOM at {identity}, shape={tuple(base_tensor.shape)}, "
                        f"device={device}: {exc}",)
            raise
        except Exception as exc:
            raise RuntimeError(f"TSVM failed at {identity}, shape={tuple(base_tensor.shape)}, "
                               f"device={device}: {exc}") from exc
        self.completed_parts += 1
        self.cuda_parts += int(device.type == "cuda")
        self.cpu_parts += int(device.type == "cpu")
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def summary(self) -> dict:
        cuda = self.device.type == "cuda"
        return {
            "method": "tsvm", "status": "complete", "task_svd": "randomized_lowrank",
            "orthogonalization": "reduced_svd_procrustes",
            "num_teachers": self.num_teachers, "scale": self.scale,
            "rank_rule": "k=floor(min(m,n)/num_teachers); zero pad factors to min(m,n)",
            "task_svd_full_matrices": False,
            "niter": self.niter, "oversampling": self.oversampling,
            "task_svd_q_rule": "min(k + oversampling, min(m,n))",
            "device": str(self.device), "compute_dtype": "float32", "tf32": False,
            "nonmatrix_rule": "base + scale * mean(teacher-base)",
            "nonfloating_rule": "copy_base", "output_dtype": "base_storage_dtype",
            "completed_parts": self.completed_parts, "cuda_parts": self.cuda_parts,
            "cpu_parts": self.cpu_parts, "elapsed_compute_seconds": self.elapsed_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else 0,
            "memory_statistics_scope": "PyTorch allocator peaks since TSVM runtime initialization",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "reference": "https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/utils/TSVM_utils.py",
        }


def merge_tsvm(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
    niter: int = 2,
    oversampling: int = 8,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """Merge task singular vectors with randomized low-rank task SVD and polar whitening.

    Matrices retain floor(min(m,n)/T) singular components per task. Ordinary
    non-matrix floating parameters use the official mean of task differences.
    Packed experts are merged separately per expert and gate/up/down projection.
    """
    runtime = _TSVMRuntime(num_teachers=len(teacher_state_dicts), scale=scale,
                           device=device, niter=niter, oversampling=oversampling)
    _validate_compatible_state_dicts(
        [base_state_dict, *teacher_state_dicts],
        ["base", *(f"teacher_{i}" for i in range(len(teacher_state_dicts)))],
    )
    print(f"TSVM device={runtime.device}; scale={scale}; task_svd=randomized_lowrank; niter={niter}; oversampling={oversampling}", flush=True)
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError("Streaming TSVM requires safetensors checkpoint readers for all teachers")
        return _StreamingMergedStateDict(
            base_state_dict, teacher_state_dicts, method="tsvm", scale=scale, tsvm=runtime,
        )
    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "tsvm", scale, 1.0, packed, tsvm=runtime,
        )
        for key in base_state_dict
    }


@torch.no_grad()
def _merge_wudi_tensor(
    base_tensor: torch.Tensor,
    teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
    *,
    num_teachers: int,
    scale: float,
    device: torch.device,
    steps: int,
    lr: float,
    transpose: bool = False,
) -> torch.Tensor:
    """Stream exact WUDI statistics, then optimize with analytic Adam gradients.

    For output-by-input D: A=sum(D.T D / ||D||²), B=sum(D (D.T D) / ||D||²).
    The official loss has gradient 2(X A-B). Reassociation changes rounding,
    but requires neither resident teacher deltas nor output-by-output matrices.
    """
    if not torch.isfinite(base_tensor).all():
        raise FloatingPointError("Non-finite base weights")
    if scale == 0 or base_tensor.numel() == 0:
        return base_tensor.detach().to(device="cpu", copy=True)
    base_float = base_tensor.to(device=device, dtype=torch.float32, copy=True)
    if transpose:
        base_float = base_float.mT
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite FP32 base weights")
    merged = torch.zeros_like(base_float, memory_format=torch.contiguous_format)
    inputs = base_float.shape[1]
    gram_sum = torch.zeros((inputs, inputs), device=device, dtype=torch.float32) if steps else None
    target = torch.zeros_like(merged) if steps else None
    count = active = 0
    for teacher_index, teacher_tensor in enumerate(teacher_tensor_factory()):
        if teacher_index >= num_teachers:
            raise ValueError("Too many teacher tensors")
        if teacher_tensor.shape != base_tensor.shape:
            raise ValueError(f"Teacher {teacher_index}: incompatible tensor shape")
        delta = teacher_tensor.to(device=device, dtype=torch.float32, copy=True)
        if transpose:
            delta = delta.mT
        delta.sub_(base_float)
        if not torch.isfinite(delta).all():
            raise FloatingPointError(f"Teacher {teacher_index}: non-finite task delta")
        merged.add_(delta)
        if steps:
            norm_sq = torch.linalg.vector_norm(delta).square()
            if not torch.isfinite(norm_sq):
                raise FloatingPointError(f"Teacher {teacher_index}: non-finite squared norm")
            if norm_sq.item() > 0:
                gram = (delta.mT @ delta).div_(norm_sq)
                gram_sum.add_(gram)
                target.addmm_(delta, gram)
                active += 1
                del gram
            elif torch.count_nonzero(delta).item():
                raise FloatingPointError(f"Teacher {teacher_index}: squared norm underflow")
            del norm_sq
        count += 1
        del delta, teacher_tensor
    if count != num_teachers:
        raise ValueError(f"Expected {num_teachers} teachers, got {count}")
    if not torch.isfinite(merged).all():
        raise FloatingPointError("Non-finite initial merged task vector")
    if steps:
        if not torch.isfinite(gram_sum).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite WUDI sufficient statistics")
        # With zero/one nonzero task, initialization already has zero loss.
        if active > 1:
            optimizer = torch.optim.Adam(
                [merged], lr=lr, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0, foreach=False, fused=False,
            )
            gradient = torch.empty_like(merged)
            for _ in range(steps):
                torch.addmm(target, merged, gram_sum, beta=-2, alpha=2, out=gradient)
                merged.grad = gradient
                optimizer.step()
            state = optimizer.state[merged]
            if any(not torch.isfinite(value).all() for value in (
                gradient, state["exp_avg"], state["exp_avg_sq"],
            )):
                raise FloatingPointError("Non-finite WUDI gradient or Adam state")
            del state
            merged.grad = None
            del optimizer, gradient
        del gram_sum, target
    base_float.add_(merged, alpha=scale)
    del merged
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite WUDI merged weights")
    if transpose:
        base_float = base_float.mT
    stored = base_float.to(device="cpu", dtype=base_tensor.dtype)
    if not torch.isfinite(stored).all():
        raise FloatingPointError(f"WUDI merged weights overflow {base_tensor.dtype}")
    return stored


class _WUDIRuntime:
    """WUDI configuration and scalar diagnostics; never retain GPU tensors."""

    def __init__(self, *, num_teachers: int, scale: float, device: str | torch.device,
                 steps: int, lr: float):
        if num_teachers <= 0:
            raise ValueError("WUDI requires at least one teacher")
        if not math.isfinite(scale):
            raise ValueError("WUDI scale must be finite")
        if type(steps) is not int or steps < 0:
            raise ValueError("WUDI steps must be a nonnegative integer")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("WUDI learning rate must be finite and positive")
        if str(device) == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("WUDI device must be auto, cpu, or a CUDA device")
        if self.device.type == "cpu" and self.device.index is not None:
            raise ValueError("Use cpu without a device index")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("WUDI requested CUDA, but CUDA is unavailable")
            index = torch.cuda.current_device() if self.device.index is None else self.device.index
            if index >= torch.cuda.device_count():
                raise ValueError(f"Invalid CUDA device index: {index}")
            self.device = torch.device("cuda", index)
            torch.cuda.get_device_properties(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.num_teachers, self.scale, self.steps, self.lr = num_teachers, scale, steps, lr
        self.completed_parts = self.copied_base_tensors = 0
        self.elapsed_seconds = 0.0

    def merge(self, base_tensor: torch.Tensor,
              teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
              identity: str, *, transpose: bool = False) -> torch.Tensor:
        started = time.perf_counter()
        try:
            with _tsvm_fp32_matmul():
                result = _merge_wudi_tensor(
                    base_tensor, teacher_tensor_factory, num_teachers=self.num_teachers,
                    scale=self.scale, device=self.device, steps=self.steps, lr=self.lr,
                    transpose=transpose,
                )
        except torch.cuda.OutOfMemoryError as exc:
            exc.args = (f"WUDI OOM at {identity}, shape={tuple(base_tensor.shape)}, "
                        f"device={self.device}: {exc}",)
            raise
        except Exception as exc:
            raise RuntimeError(f"WUDI failed at {identity}, shape={tuple(base_tensor.shape)}, "
                               f"device={self.device}: {exc}") from exc
        self.completed_parts += 1
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def summary(self) -> dict:
        cuda = self.device.type == "cuda"
        return {
            "method": "wudi", "status": "complete", "scale": self.scale,
            "steps": self.steps, "lr": self.lr, "num_teachers": self.num_teachers,
            "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0, "foreach": False, "fused": False,
            "gradient": "2*(X*A-B), A=sum(D.T@D/||D||^2), B=sum(D@(D.T@D)/||D||^2)",
            "initialization": "sum_task_deltas", "zero_delta_rule": "skip_loss_term",
            "parameter_scope": "packed gate/up/down; 2D floating .weight except embed_tokens/pos_embed",
            "other_parameters": "copy_base", "packed_orientation": "transpose_to_output_input",
            "device": str(self.device), "compute_dtype": "float32", "tf32": False,
            "output_dtype": "base_storage_dtype", "completed_parts": self.completed_parts,
            "copied_base_tensors": self.copied_base_tensors,
            "elapsed_compute_seconds": self.elapsed_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else 0,
            "memory_statistics_scope": "PyTorch allocator peaks since WUDI runtime initialization",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "reference": "https://github.com/nathanielyvo/WUDI-Merging/blob/main/vit/wudi_main.py",
            "paper": "https://arxiv.org/abs/2503.08099",
        }


def merge_wudi(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
    steps: int = 300,
    lr: float = 1e-5,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """WUDI for Qwen linear weights; retain base embeddings and other parameters."""
    runtime = _WUDIRuntime(num_teachers=len(teacher_state_dicts), scale=scale,
                           device=device, steps=steps, lr=lr)
    _validate_compatible_state_dicts(
        [base_state_dict, *teacher_state_dicts],
        ["base", *(f"teacher_{i}" for i in range(len(teacher_state_dicts)))],
    )
    print(f"WUDI device={runtime.device}; scale={scale}; steps={steps}; lr={lr}", flush=True)
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError("Streaming WUDI requires safetensors readers for all teachers")
        return _StreamingMergedStateDict(
            base_state_dict, teacher_state_dicts, method="wudi", scale=scale, wudi=runtime,
        )
    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "wudi", scale, 1.0, packed, wudi=runtime,
        )
        for key in base_state_dict
    }


def _dc_trim(matrix: torch.Tensor, density: float) -> torch.Tensor:
    """Official cover-space top-k threshold, including all threshold ties."""
    k = max(1, int(density * matrix.numel()))
    threshold = torch.topk(matrix.abs().reshape(-1), k).values.min()
    return matrix.masked_fill(matrix.abs() < threshold, 0)


def _dc_cover_merge(matrices: list[torch.Tensor], density: float) -> torch.Tensor:
    """Trim full cover matrices, elect signs, average, THEN apply block mask."""
    stacked = torch.stack([_dc_trim(matrix, density) for matrix in matrices])
    summed = stacked.sum(dim=0)
    aligned = stacked.sign() == summed.sign().unsqueeze(0)
    merged = (stacked * aligned).sum(dim=0) / aligned.sum(dim=0).clamp_min(1)
    merged.masked_fill_(summed == 0, 0)
    del stacked, summed, aligned
    # Preserve the official zero-padded dimensions when d % T != 0.
    rank = merged.shape[0] // len(matrices)
    masked = torch.zeros_like(merged)
    for task in range(len(matrices)):
        block = slice(task * rank, (task + 1) * rank)
        masked[block, block].copy_(merged[block, block])
    return masked


@torch.inference_mode()
def _merge_dc_tensor(
    base_tensor: torch.Tensor,
    teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
    *,
    num_teachers: int,
    scale: float,
    device: torch.device,
    density: float,
    identity: str,
    niter: int = 2,
    oversampling: int = 8,
) -> torch.Tensor:
    """FFT DC-Merge with randomized low-rank task SVD and factorized cover projection.

    The official implementation reconstructs each truncated task matrix first.
    Here M_i=(Q_U.T U_i diag(s_i)) (V_i.T Q_V) avoids those full matrices.
    Task truncation is FFT's implicit smoothing; no LoRA smoothing is applied.
    """
    if (not torch.is_floating_point(base_tensor) or base_tensor.ndim != 2
            or scale == 0 or base_tensor.numel() == 0
            or min(base_tensor.shape) // num_teachers == 0):
        # Identical copy/online-mean rules to TSVM, with no matrix SVD here.
        return _merge_tsvm_tensor(
            base_tensor, teacher_tensor_factory, num_teachers=num_teachers,
            scale=scale, device=torch.device("cpu"), identity=identity,
        )
    base_float = base_tensor.to(device=device, dtype=torch.float32, copy=True)
    if not torch.isfinite(base_tensor).all() or not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite base weights")
    m, n = base_tensor.shape
    d = min(m, n)
    rank = d // num_teachers
    left = torch.zeros((m, d), device=device, dtype=torch.float32)
    right = torch.zeros((d, n), device=device, dtype=torch.float32)
    singular = torch.zeros(d, device=device, dtype=torch.float32)
    count = 0
    for task, teacher_tensor in enumerate(teacher_tensor_factory()):
        if task >= num_teachers:
            raise ValueError("Too many teacher tensors")
        if teacher_tensor.shape != base_tensor.shape:
            raise ValueError(f"Teacher {task}: incompatible tensor shape")
        delta = teacher_tensor.to(device=device, dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        if not torch.isfinite(delta).all():
            raise FloatingPointError(f"Teacher {task}: non-finite task delta")
        u, s, v = torch.svd_lowrank(
            delta, q=min(rank + oversampling, d), niter=niter,
        )
        vh = v.mT
        block = slice(task * rank, (task + 1) * rank)
        left[:, block].copy_(u[:, :rank])
        right[block, :].copy_(vh[:rank, :])
        singular[block].copy_(s[:rank])
        count += 1
        del delta, teacher_tensor, u, s, v, vh
    if count != num_teachers:
        raise ValueError(f"Expected {num_teachers} teachers, got {count}")
    if any(not torch.isfinite(value).all() for value in (left, right, singular)):
        raise FloatingPointError("Non-finite task SVD factors")
    cover_left = _tsvm_polar(left)
    projected_left = cover_left.mT @ left
    del left
    cover_right = _tsvm_polar(right)
    projected_right = right @ cover_right.mT
    del right
    matrices = []
    for task in range(num_teachers):
        block = slice(task * rank, (task + 1) * rank)
        matrix = (projected_left[:, block] * singular[block]) @ projected_right[block, :]
        if not torch.isfinite(matrix).all():
            raise FloatingPointError(f"Teacher {task}: non-finite cover projection")
        matrices.append(matrix)
    del projected_left, projected_right, singular, matrix
    merged = _dc_cover_merge(matrices, density)
    del matrices
    if not torch.isfinite(merged).all():
        raise FloatingPointError("Non-finite cover-space aggregation")
    delta = (cover_left @ merged) @ cover_right
    del cover_left, cover_right, merged
    base_float.add_(delta, alpha=scale)
    del delta
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite DC merged weights")
    stored = base_float.to(device="cpu", dtype=base_tensor.dtype)
    if not torch.isfinite(stored).all():
        raise FloatingPointError(f"DC merged weights overflow storage dtype {base_tensor.dtype}")
    return stored


class _DCRuntime(_TSVMRuntime):
    """Share SVD configuration/device validation and scalar diagnostics with TSVM."""

    def __init__(self, *, density: float, **kwargs):
        if not math.isfinite(density) or not 0 < density <= 1:
            raise ValueError("DC density must be finite and in (0, 1]")
        try:
            super().__init__(**kwargs)
        except ValueError as exc:
            exc.args = (str(exc).replace("TSVM", "DC"),)
            raise
        self.density = density

    def merge(self, base_tensor: torch.Tensor,
              teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
              identity: str) -> torch.Tensor:
        use_svd = (torch.is_floating_point(base_tensor) and base_tensor.ndim == 2
                   and min(base_tensor.shape) // self.num_teachers > 0 and self.scale != 0)
        device = self.device if use_svd else torch.device("cpu")
        started = time.perf_counter()
        try:
            with _tsvm_fp32_matmul():
                result = _merge_dc_tensor(
                    base_tensor, teacher_tensor_factory, num_teachers=self.num_teachers,
                    scale=self.scale, device=device, density=self.density,
                    identity=identity, niter=self.niter, oversampling=self.oversampling,
                )
        except torch.cuda.OutOfMemoryError as exc:
            exc.args = (f"DC OOM at {identity}, shape={tuple(base_tensor.shape)}, "
                        f"device={device}: {exc}",)
            raise
        except Exception as exc:
            raise RuntimeError(f"DC failed at {identity}, shape={tuple(base_tensor.shape)}, "
                               f"device={device}: {exc}") from exc
        self.completed_parts += 1
        self.cuda_parts += int(device.type == "cuda")
        self.cpu_parts += int(device.type == "cpu")
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def summary(self) -> dict:
        report = super().summary()
        report.update({
            "method": "dc", "variant": "FFT", "density": self.density,
            "smoothing": "task_svd_truncation_only", "projection": "factorized_cover_space",
            "aggregation": "full_matrix_topk_threshold_then_mass_sign_disjoint_mean",
            "post_aggregation_mask": "T diagonal r-by-r blocks; padded remainder zero",
            "parameter_scope": "all floating 2D matrices including embeddings; packed expert projections",
            "memory_statistics_scope": "PyTorch allocator peaks since DC runtime initialization",
            "reference": "https://github.com/Tobeginwith/DC-Merge/blob/089af77a378562272461e0f041217ec440a4a230/vision_fft_merge/src/utils/DC_utils.py",
            "paper": "https://arxiv.org/abs/2603.06242",
        })
        return report


def merge_dc(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
    niter: int = 2,
    oversampling: int = 8,
    density: float = 0.001,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """FFT DC-Merge; nonmatrix floats use mean task delta, integers copy base."""
    runtime = _DCRuntime(num_teachers=len(teacher_state_dicts), scale=scale,
                         device=device, density=density, niter=niter, oversampling=oversampling)
    _validate_compatible_state_dicts(
        [base_state_dict, *teacher_state_dicts],
        ["base", *(f"teacher_{i}" for i in range(len(teacher_state_dicts)))],
    )
    print(f"DC device={runtime.device}; scale={scale}; density={density}; "
          f"task_svd=randomized_lowrank; niter={niter}; oversampling={oversampling}", flush=True)
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError("Streaming DC requires safetensors readers for all teachers")
        return _StreamingMergedStateDict(
            base_state_dict, teacher_state_dicts, method="dc", scale=scale, dc=runtime,
        )
    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "dc", scale, 1.0, packed, dc=runtime,
        )
        for key in base_state_dict
    }


@torch.inference_mode()
def _merge_iso_c_tensor(
    base_tensor: torch.Tensor,
    teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
    *,
    num_teachers: int,
    scale: float,
    device: torch.device,
    identity: str,
) -> torch.Tensor:
    """Flatten the spectrum of the summed task matrix, without rank truncation."""
    if (not torch.is_floating_point(base_tensor) or base_tensor.ndim != 2
            or scale == 0 or base_tensor.numel() == 0):
        return _merge_tsvm_tensor(
            base_tensor, teacher_tensor_factory, num_teachers=num_teachers,
            scale=scale, device=torch.device("cpu"), identity=identity,
        )
    base_float = base_tensor.to(device=device, dtype=torch.float32, copy=True)
    if not torch.isfinite(base_tensor).all() or not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite base weights")
    combined = torch.zeros_like(base_float)
    count = 0
    for task, teacher_tensor in enumerate(teacher_tensor_factory()):
        if task >= num_teachers:
            raise ValueError("Too many teacher tensors")
        if teacher_tensor.shape != base_tensor.shape:
            raise ValueError(f"Teacher {task}: incompatible tensor shape")
        delta = teacher_tensor.to(device=device, dtype=torch.float32, copy=True)
        delta.sub_(base_float)
        if not torch.isfinite(delta).all():
            raise FloatingPointError(f"Teacher {task}: non-finite task delta")
        combined.add_(delta)
        count += 1
        del delta, teacher_tensor
    if count != num_teachers:
        raise ValueError(f"Expected {num_teachers} teachers, got {count}")
    if not torch.isfinite(combined).all():
        raise FloatingPointError("Non-finite summed task matrix")
    u, singular, vh = torch.linalg.svd(combined, full_matrices=False)
    del combined
    # Keep every reduced-SVD component, including zero singular values.
    u.mul_(singular.mean())
    delta = u @ vh
    del u, singular, vh
    base_float.add_(delta, alpha=scale)
    del delta
    if not torch.isfinite(base_float).all():
        raise FloatingPointError("Non-finite Iso-C merged weights")
    stored = base_float.to(device="cpu", dtype=base_tensor.dtype)
    if not torch.isfinite(stored).all():
        raise FloatingPointError(f"Iso-C merged weights overflow storage dtype {base_tensor.dtype}")
    return stored


class _IsoCRuntime(_TSVMRuntime):
    """Reuse SVD runtime diagnostics; matrix computation requires CUDA."""

    def __init__(self, **kwargs):
        device = kwargs.get("device", "auto")
        resolved = torch.device("cuda:0" if str(device) == "auto" else device)
        if resolved.type != "cuda":
            raise ValueError("Iso-C requires a CUDA device for matrix merging")
        if resolved.index is not None and not 0 <= resolved.index < torch.cuda.device_count():
            raise ValueError(f"Invalid Iso-C CUDA device index: {resolved.index}")
        kwargs["device"] = resolved
        try:
            super().__init__(**kwargs)
        except ValueError as exc:
            exc.args = (str(exc).replace("TSVM", "Iso-C"),)
            raise

    def merge(self, base_tensor: torch.Tensor,
              teacher_tensor_factory: Callable[[], Iterable[torch.Tensor]],
              identity: str) -> torch.Tensor:
        use_svd = (torch.is_floating_point(base_tensor) and base_tensor.ndim == 2
                   and base_tensor.numel() > 0 and self.scale != 0)
        device = self.device if use_svd else torch.device("cpu")
        started = time.perf_counter()
        try:
            with _tsvm_fp32_matmul():
                result = _merge_iso_c_tensor(
                    base_tensor, teacher_tensor_factory, num_teachers=self.num_teachers,
                    scale=self.scale, device=device, identity=identity,
                )
        except torch.cuda.OutOfMemoryError as exc:
            exc.args = (f"Iso-C OOM at {identity}, shape={tuple(base_tensor.shape)}, "
                        f"device={device}: {exc}",)
            raise
        except Exception as exc:
            raise RuntimeError(f"Iso-C failed at {identity}, shape={tuple(base_tensor.shape)}, "
                               f"device={device}: {exc}") from exc
        self.completed_parts += 1
        self.cuda_parts += int(device.type == "cuda")
        self.cpu_parts += int(device.type == "cpu")
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def summary(self) -> dict:
        report = super().summary()
        for key in ("task_svd", "task_svd_full_matrices", "orthogonalization", "rank_rule",
                    "niter", "oversampling", "task_svd_q_rule"):
            del report[key]
        report.update({
            "method": "iso_c", "svd_input": "sum(teacher-base)",
            "svd_full_matrices": False, "spectrum": "mean of all reduced singular values",
            "rank_rule": "all min(m,n) components; no truncation",
            "parameter_scope": "all floating 2D matrices; packed expert projections",
            "memory_statistics_scope": "PyTorch allocator peaks since Iso-C runtime initialization",
            "reference": "https://github.com/danielm1405/iso-merging/blob/main/src/utils/iso.py",
            "paper": "https://arxiv.org/abs/2502.04959",
        })
        return report


def merge_iso_c(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """Iso-C on every floating matrix; other floats use the CPU mean task delta."""
    runtime = _IsoCRuntime(num_teachers=len(teacher_state_dicts), scale=scale, device=device)
    _validate_compatible_state_dicts(
        [base_state_dict, *teacher_state_dicts],
        ["base", *(f"teacher_{i}" for i in range(len(teacher_state_dicts)))],
    )
    print(f"Iso-C device={runtime.device}; scale={scale}; summed-task SVD", flush=True)
    if isinstance(base_state_dict, SafetensorCheckpoint):
        if not all(isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
            raise TypeError("Streaming Iso-C requires safetensors readers for all teachers")
        return _StreamingMergedStateDict(
            base_state_dict, teacher_state_dicts, method="iso_c", scale=scale, iso_c=runtime,
        )
    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "iso_c", scale, 1.0, packed, iso_c=runtime,
        )
        for key in base_state_dict
    }


def _expert_part_indices(layout: tuple[int, int, bool]) -> Iterable[tuple[slice, ...]]:
    experts, intermediate, is_gate_up = layout
    for expert in range(experts):
        for projection in range(2 if is_gate_up else 1):
            columns = (slice(projection * intermediate, (projection + 1) * intermediate)
                       if is_gate_up else slice(None))
            yield (slice(expert, expert + 1), slice(None), columns)


def _teacher_parts(teachers, key: str, index: tuple[slice, ...] | None,
                   squeeze: bool = False) -> Iterable[torch.Tensor]:
    for teacher in teachers:
        if index is None:
            tensor = _get_state_tensor(teacher, key)
        elif isinstance(teacher, SafetensorCheckpoint):
            tensor = teacher.get_slice(key, index)
        else:
            tensor = teacher[key][index]
        yield tensor.squeeze(0) if squeeze and index is not None else tensor
        del tensor


class _RAMPlusRuntime(_TSVMRuntime):
    """Global RAM+ counts and scales; never retain checkpoint tensors or masks."""

    def __init__(self, *, num_teachers: int, scale: float, device: str | torch.device,
                 threshold: float, rescale_factor: float):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("RAM+ threshold must be finite and nonnegative")
        if not math.isfinite(rescale_factor):
            raise ValueError("RAM+ rescale_factor must be finite")
        resolved = torch.device("cuda:0" if str(device) == "auto" else device)
        if resolved.type != "cuda":
            raise ValueError("RAM+ requires a CUDA device")
        if resolved.index is not None and not 0 <= resolved.index < torch.cuda.device_count():
            raise ValueError(f"Invalid RAM+ CUDA device index: {resolved.index}")
        try:
            super().__init__(num_teachers=num_teachers, scale=scale, device=resolved)
        except ValueError as exc:
            exc.args = (str(exc).replace("TSVM", "RAM+"),)
            raise
        self.threshold = threshold
        self.requested_rescale_factor = rescale_factor
        self.rescale_factor = max(1.0, rescale_factor)
        self.shared_counts = [0] * num_teachers
        self.unique_counts = [0] * num_teachers
        self.rescales = [1.0] * num_teachers
        self.statistics_complete = False
        self.statistics_parts = 0
        self.statistics_seconds = 0.0

    def _deltas(self, base_float, shape, teacher_tensor_factory):
        count = 0
        for task, tensor in enumerate(teacher_tensor_factory()):
            if task >= self.num_teachers or tuple(tensor.shape) != shape:
                raise ValueError(f"Teacher {task}: incompatible tensor count or shape")
            delta = tensor.to(device=self.device, dtype=torch.float32, copy=True)
            delta.sub_(base_float)
            if not torch.isfinite(delta).all():
                raise FloatingPointError(f"Teacher {task}: non-finite FP32 task delta")
            active = delta.abs() > self.threshold
            yield task, delta, active
            count += 1
            del tensor, delta, active
        if count != self.num_teachers:
            raise ValueError(f"Expected {self.num_teachers} teachers, got {count}")

    @torch.inference_mode()
    def _compute(self, base_tensor, teacher_tensor_factory, *, probing: bool):
        if not torch.is_floating_point(base_tensor):
            return None if probing else base_tensor.detach().to(device="cpu", copy=True)
        if not torch.isfinite(base_tensor).all():
            raise FloatingPointError("Non-finite base weights")
        if self.scale == 0 or base_tensor.numel() == 0:
            return None if probing else base_tensor.detach().to(device="cpu", copy=True)
        base_float = base_tensor.to(device=self.device, dtype=torch.float32, copy=True)
        if not torch.isfinite(base_float).all():
            raise FloatingPointError("Non-finite FP32 base weights")
        shape = tuple(base_tensor.shape)
        counts = torch.zeros(shape, device=self.device, dtype=torch.int64)
        if probing:
            for _, delta, active in self._deltas(base_float, shape, teacher_tensor_factory):
                counts.add_(active)
                del delta, active
            # Reread teachers instead of retaining N masks for this parameter.
            for task, delta, active in self._deltas(base_float, shape, teacher_tensor_factory):
                self.shared_counts[task] += int((active & (counts >= 2)).sum().item())
                self.unique_counts[task] += int((active & (counts == 1)).sum().item())
                del delta, active
            return None
        summed = torch.zeros_like(base_float)
        weighted = torch.zeros_like(base_float)
        for task, delta, active in self._deltas(base_float, shape, teacher_tensor_factory):
            counts.add_(active)
            delta.masked_fill_(~active, 0)
            summed.add_(delta)
            weighted.add_(delta, alpha=self.rescales[task])
            del delta, active
        if not torch.isfinite(summed).all() or not torch.isfinite(weighted).all():
            raise FloatingPointError("Non-finite RAM+ accumulated task delta")
        summed.div_(counts.clamp_min(1))
        merged = torch.where(counts == 1, weighted, summed)
        base_float.add_(merged, alpha=self.scale)
        if not torch.isfinite(base_float).all():
            raise FloatingPointError("Non-finite RAM+ merged weights")
        stored = base_float.to(device="cpu", dtype=base_tensor.dtype)
        if not torch.isfinite(stored).all():
            raise FloatingPointError(f"RAM+ weights overflow storage dtype {base_tensor.dtype}")
        return stored

    def _run(self, base_tensor, teacher_tensor_factory, identity: str, *, probing: bool):
        stage = "statistics" if probing else "merge"
        started = time.perf_counter()
        try:
            result = self._compute(base_tensor, teacher_tensor_factory, probing=probing)
        except torch.cuda.OutOfMemoryError as exc:
            exc.args = (f"RAM+ OOM during {stage} at {identity}, "
                        f"shape={tuple(base_tensor.shape)}, device={self.device}: {exc}",)
            raise
        except Exception as exc:
            raise RuntimeError(f"RAM+ failed during {stage} at {identity}, "
                               f"shape={tuple(base_tensor.shape)}, device={self.device}: {exc}") from exc
        elapsed = time.perf_counter() - started
        if probing:
            self.statistics_parts += 1
            self.statistics_seconds += elapsed
        else:
            self.completed_parts += 1
            self.elapsed_seconds += elapsed
        return result

    def probe(self, base_tensor, teacher_tensor_factory, identity: str) -> None:
        self._run(base_tensor, teacher_tensor_factory, identity, probing=True)

    def finish_statistics(self) -> None:
        for task, (shared, unique) in enumerate(zip(self.shared_counts, self.unique_counts)):
            ratio = min(shared / unique, 1.0) if unique else float(shared > 0)
            self.rescales[task] = 1.0 + (self.rescale_factor - 1.0) * ratio
        self.statistics_complete = True
        print(f"RAM+ global shared={self.shared_counts}; unique={self.unique_counts}; "
              f"rescales={self.rescales}", flush=True)

    def merge(self, base_tensor, teacher_tensor_factory, identity: str) -> torch.Tensor:
        if not self.statistics_complete:
            raise RuntimeError("RAM+ requires completed global statistics before merging")
        return self._run(base_tensor, teacher_tensor_factory, identity, probing=False)

    def summary(self) -> dict:
        return {
            "method": "ram_plus", "variant": "official_arm_r_v2", "status": "complete",
            "num_teachers": self.num_teachers, "scale": self.scale,
            "threshold": self.threshold, "requested_rescale_factor": self.requested_rescale_factor,
            "rescale_factor": self.rescale_factor,
            "rescale_rule": "1 + (max(1,rescale_factor)-1) * min(shared/unique,1)",
            "zero_unique_rule": "ratio=1 if shared>0 else 0",
            "statistics_scope": "all aligned floating parameters, global per teacher",
            "statistics_skipped": self.scale == 0,
            "shared_counts": self.shared_counts, "unique_counts": self.unique_counts,
            "rescales": self.rescales, "device": str(self.device), "compute_dtype": "float32",
            "output_dtype": "base_storage_dtype", "nonfloating_rule": "copy_base",
            "statistics_parts": self.statistics_parts, "completed_parts": self.completed_parts,
            "statistics_seconds": self.statistics_seconds,
            "elapsed_compute_seconds": self.elapsed_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "memory_statistics_scope": "PyTorch allocator peaks since RAM+ runtime initialization",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "reference": "https://github.com/xiangchi-yuan/mrl/blob/main/ram-main.py",
            "paper": "https://aclanthology.org/2026.acl-long.1524.pdf",
        }


def merge_ram_plus(
    base_state_dict: SafetensorCheckpoint | dict[str, torch.Tensor],
    teacher_state_dicts: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
    threshold: float = 1e-5,
    rescale_factor: float = 1.2,
) -> _StreamingMergedStateDict | dict[str, torch.Tensor]:
    """Official RAM+ scaling using global overlap/unique counts, on CUDA."""
    runtime = _RAMPlusRuntime(num_teachers=len(teacher_state_dicts), scale=scale,
                             device=device, threshold=threshold, rescale_factor=rescale_factor)
    _validate_compatible_state_dicts(
        [base_state_dict, *teacher_state_dicts],
        ["base", *(f"teacher_{i}" for i in range(len(teacher_state_dicts)))],
    )
    if isinstance(base_state_dict, SafetensorCheckpoint) and not all(
            isinstance(sd, SafetensorCheckpoint) for sd in teacher_state_dicts):
        raise TypeError("Streaming RAM+ requires safetensors readers for all teachers")
    packed = _packed_expert_layouts(base_state_dict, teacher_state_dicts)
    if scale != 0:
        print(f"RAM+ global statistics on {runtime.device} ...", flush=True)
        for key in _state_keys(base_state_dict):
            base_tensor = _get_state_tensor(base_state_dict, key)
            print(f"RAM+ statistics {key}, shape={tuple(base_tensor.shape)}", flush=True)
            indices = _expert_part_indices(packed[key]) if key in packed else (None,)
            for index in indices:
                part = base_tensor if index is None else base_tensor[index].squeeze(0)
                identity = key
                if index is not None:
                    name = ("gate" if index[2].start == 0 else "up") if packed[key][2] else "down"
                    identity = f"{key}/expert={index[0].start}/projection={name}"
                runtime.probe(part, lambda: _teacher_parts(teacher_state_dicts, key, index, True), identity)
                del part
            del base_tensor
    runtime.finish_statistics()
    if isinstance(base_state_dict, SafetensorCheckpoint):
        return _StreamingMergedStateDict(
            base_state_dict, teacher_state_dicts, method="ram_plus", scale=scale, ram_plus=runtime,
        )
    return {
        key: _merge_state_tensor(
            key, base_state_dict, teacher_state_dicts, "ram_plus", scale, 1.0, packed, ram_plus=runtime,
        )
        for key in base_state_dict
    }


def _packed_expert_layouts(
    base: SafetensorCheckpoint | dict[str, torch.Tensor],
    teachers: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
) -> dict[str, tuple[int, int, bool]]:
    """Validate Qwen [E,H,2M]/[E,M,H] pairs before any output is written."""
    pattern = re.compile(r"^(.*\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")
    prefixes = {
        match.group(1)
        for key in _state_keys(base)
        if (match := pattern.fullmatch(key))
    }
    layouts = {}
    for prefix in sorted(prefixes):
        gate_key, down_key = f"{prefix}.gate_up_proj", f"{prefix}.down_proj"
        if gate_key not in base or down_key not in base:
            raise ValueError(f"Missing paired Qwen expert tensor at {prefix}")
        gate_shape, down_shape = _get_state_shape(base, gate_key), _get_state_shape(base, down_key)
        if (len(gate_shape) != 3 or len(down_shape) != 3
                or any(size <= 0 for size in (*gate_shape, *down_shape))
                or gate_shape[0] != down_shape[0]
                or gate_shape[1] != down_shape[2]
                or gate_shape[2] != 2 * down_shape[1]):
            raise ValueError(
                f"Incompatible Qwen packed expert shapes at {prefix}: "
                f"gate_up={gate_shape}, down={down_shape}"
            )
        for i, teacher in enumerate(teachers):
            for key, expected in ((gate_key, gate_shape), (down_key, down_shape)):
                if key not in teacher or _get_state_shape(teacher, key) != expected:
                    raise ValueError(f"Teacher {i}: missing or incompatible expert tensor {key}")
        experts, intermediate = gate_shape[0], down_shape[1]
        layouts[gate_key] = (experts, intermediate, True)
        layouts[down_key] = (experts, intermediate, False)
    return layouts


def _merge_state_tensor(
    key: str,
    base: SafetensorCheckpoint | dict[str, torch.Tensor],
    teachers: list[SafetensorCheckpoint | dict[str, torch.Tensor]],
    method: str,
    scale: float,
    density: float,
    packed: dict[str, tuple[int, int, bool]],
    *,
    tsvm: _TSVMRuntime | None = None,
    wudi: _WUDIRuntime | None = None,
    dc: _DCRuntime | None = None,
    iso_c: _IsoCRuntime | None = None,
    ram_plus: _RAMPlusRuntime | None = None,
) -> torch.Tensor:
    """Dispatch both in-memory and streaming merges through the same partitions."""
    base_tensor = _get_state_tensor(base, key)
    expected = tuple(base_tensor.shape)
    if method == "wudi":
        if wudi is None:
            raise ValueError("Missing WUDI runtime")
        linear = key in packed or (
            base_tensor.ndim == 2 and key.endswith(".weight")
            and key.rsplit(".", 2)[-2] not in ("embed_tokens", "pos_embed")
        )
        if not torch.is_floating_point(base_tensor) or not linear:
            wudi.copied_base_tensors += 1
            return base_tensor.detach().to(device="cpu", copy=True)
    for teacher in teachers:
        if _get_state_shape(teacher, key) != expected:
            raise ValueError(f"Shape mismatch at {key}: expected {expected}")

    def merge_part(index: tuple[slice, ...] | None) -> torch.Tensor:
        def teacher_parts() -> Iterable[torch.Tensor]:
            return _teacher_parts(teachers, key, index,
                                  method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus"))

        part = base_tensor if index is None else base_tensor[index]
        if method == "ta":
            return _merge_ta_tensor(part, teacher_parts(), scale)
        if method == "ties":
            return _merge_ties_tensor(part, teacher_parts, density, scale)
        if method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus"):
            runtime = {"tsvm": tsvm, "wudi": wudi, "dc": dc, "iso_c": iso_c, "ram_plus": ram_plus}[method]
            if runtime is None:
                raise ValueError(f"Missing {method.upper()} runtime")
            identity = key
            if index is not None:
                name = ("gate" if index[2].start == 0 else "up") if packed[key][2] else "down"
                identity = f"{key}/expert={index[0].start}/projection={name}"
                part = part.squeeze(0)
            if method == "wudi":
                result = runtime.merge(part, teacher_parts, identity, transpose=index is not None)
            else:
                result = runtime.merge(part, teacher_parts, identity)
            return result.unsqueeze(0) if index is not None else result
        raise ValueError(f"Unsupported merge method: {method}")

    if method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus"):
        print(f"{method.upper()} merging {key}, shape={expected}", flush=True)
    if key not in packed:
        return merge_part(None)
    experts, intermediate, is_gate_up = packed[key]
    output = torch.empty_like(base_tensor, device="cpu") if method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus") else torch.empty_like(base_tensor)
    for index in _expert_part_indices(packed[key]):
        expert = index[0].start
        if (method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus") and expert % 16 == 0
                and (not is_gate_up or index[2].start == 0)):
            print(f"  {method.upper()} expert {expert + 1}/{experts}", flush=True)
        output[index].copy_(merge_part(index))
    return output


class _StreamingMergedStateDict:
    """Compute one merged tensor on demand without retaining complete checkpoints."""

    def __init__(
        self,
        base_state_dict: SafetensorCheckpoint,
        teacher_state_dicts: list[SafetensorCheckpoint],
        method: str,
        scale: float,
        density: float = 0.2,
        *,
        tsvm: _TSVMRuntime | None = None,
        wudi: _WUDIRuntime | None = None,
        dc: _DCRuntime | None = None,
        iso_c: _IsoCRuntime | None = None,
        ram_plus: _RAMPlusRuntime | None = None,
    ):
        self.base_state_dict = base_state_dict
        self.teacher_state_dicts = teacher_state_dicts
        self.method = method
        self.scale = scale
        self.density = density
        self.tsvm = tsvm
        self.wudi = wudi
        self.dc = dc
        self.iso_c = iso_c
        self.ram_plus = ram_plus
        self.packed_layouts = _packed_expert_layouts(base_state_dict, teacher_state_dicts)

    def keys(self) -> list[str]:
        return self.base_state_dict.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.base_state_dict.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.base_state_dict.get_dtype(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        return _merge_state_tensor(
            key, self.base_state_dict, self.teacher_state_dicts,
            self.method, self.scale, self.density, self.packed_layouts, tsvm=self.tsvm, wudi=self.wudi, dc=self.dc, iso_c=self.iso_c, ram_plus=self.ram_plus,
        )


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _parse_size_bytes(size: str | int) -> int:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"max shard size must be positive, got {size}")
        return size

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)\s*",
        size.upper(),
    )
    if match is None:
        raise ValueError(
            f"Invalid max shard size {size!r}; examples: 500MB, 5GB, 4GiB"
        )
    value = float(match.group(1))
    unit = match.group(2)
    decimal_units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    binary_units = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    size_bytes = int(value * (decimal_units | binary_units)[unit])
    if size_bytes <= 0:
        raise ValueError(f"max shard size must be positive, got {size}")
    return size_bytes


def _merged_keys(
    merged_state_dict: _StreamingMergedStateDict | dict[str, torch.Tensor],
) -> list[str]:
    return (
        merged_state_dict.keys()
        if isinstance(merged_state_dict, _StreamingMergedStateDict)
        else list(merged_state_dict.keys())
    )


def _merged_tensor_nbytes(
    merged_state_dict: _StreamingMergedStateDict | dict[str, torch.Tensor],
    key: str,
) -> int:
    if isinstance(merged_state_dict, _StreamingMergedStateDict):
        dtype = merged_state_dict.get_dtype(key)
        try:
            element_size = _SAFETENSORS_DTYPE_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} at {key}") from exc
        numel = 1
        for dimension in merged_state_dict.get_shape(key):
            numel *= dimension
        return numel * element_size

    tensor = merged_state_dict[key]
    return tensor.numel() * tensor.element_size()


def _get_merged_tensor(
    merged_state_dict: _StreamingMergedStateDict | dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if isinstance(merged_state_dict, _StreamingMergedStateDict):
        return merged_state_dict.get_tensor(key)
    return merged_state_dict[key]


def _plan_output_shards(
    merged_state_dict: _StreamingMergedStateDict | dict[str, torch.Tensor],
    max_shard_size: str | int,
) -> list[tuple[str, list[str]]]:
    max_bytes = _parse_size_bytes(max_shard_size)
    shard_keys: list[list[str]] = []
    current_keys: list[str] = []
    current_size = 0

    for key in _merged_keys(merged_state_dict):
        tensor_size = _merged_tensor_nbytes(merged_state_dict, key)
        if current_keys and current_size + tensor_size > max_bytes:
            shard_keys.append(current_keys)
            current_keys = []
            current_size = 0
        current_keys.append(key)
        current_size += tensor_size

    if current_keys:
        shard_keys.append(current_keys)
    if not shard_keys:
        raise ValueError("Cannot save an empty merged state dict")

    total_shards = len(shard_keys)
    if total_shards == 1:
        return [("model.safetensors", shard_keys[0])]
    return [
        (f"model-{index:05d}-of-{total_shards:05d}.safetensors", keys)
        for index, keys in enumerate(shard_keys, start=1)
    ]


def _remove_stale_safetensors(output_dir: Path) -> None:
    for path in output_dir.glob("model*.safetensors"):
        if path.is_file():
            path.unlink()
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_path.unlink()


def save_merged_model_hf(
    merged_state_dict: _StreamingMergedStateDict | dict[str, torch.Tensor],
    template_dir: str | Path,
    output_dir: str | Path,
    max_shard_size: str = "5GB",
    trust_remote_code: bool = True,
) -> None:
    """Compute tensors on demand and write a sharded HF safetensors checkpoint."""
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    if (isinstance(merged_state_dict, _StreamingMergedStateDict)
            and (merged_state_dict.wudi is not None or merged_state_dict.dc is not None
                 or merged_state_dict.iso_c is not None or merged_state_dict.ram_plus is not None)):
        _parse_size_bytes(max_shard_size)
        input_paths = [merged_state_dict.base_state_dict.model_dir,
                       *(sd.model_dir for sd in merged_state_dict.teacher_state_dicts)]
        if output_dir.resolve() in {path.resolve() for path in input_paths}:
            raise ValueError(f"{merged_state_dict.method.upper()} output directory must differ from all input checkpoints")
    if template_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the base/template directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_safetensors(output_dir)
    (output_dir / "tsvm_merge_summary.json").unlink(missing_ok=True)
    (output_dir / "wudi_merge_summary.json").unlink(missing_ok=True)
    (output_dir / "dc_merge_summary.json").unlink(missing_ok=True)
    (output_dir / "iso_c_merge_summary.json").unlink(missing_ok=True)
    (output_dir / "ram_plus_merge_summary.json").unlink(missing_ok=True)

    shard_plan = _plan_output_shards(merged_state_dict, max_shard_size)
    weight_map: dict[str, str] = {}
    total_size = 0
    print(
        f"Streaming merged tensors into {len(shard_plan)} output shard(s) "
        f"under {output_dir} ..."
    )
    for shard_index, (shard_name, keys) in enumerate(shard_plan, start=1):
        shard_state: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor = _get_merged_tensor(merged_state_dict, key).contiguous()
            shard_state[key] = tensor
            total_size += tensor.numel() * tensor.element_size()
            weight_map[key] = shard_name

        save_file(
            shard_state,
            str(output_dir / shard_name),
            metadata={"format": "pt"},
        )
        del shard_state
        print(f"  saved shard {shard_index}/{len(shard_plan)}: {shard_name}")

    if len(shard_plan) > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with (output_dir / "model.safetensors.index.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
            f.write("\n")

    for config_name in ("config.json", "generation_config.json"):
        source = template_dir / config_name
        if source.is_file():
            shutil.copy2(source, output_dir / config_name)

    processor = AutoProcessor.from_pretrained(
        str(template_dir),
        trust_remote_code=trust_remote_code,
    )
    processor.save_pretrained(str(output_dir))
    print("Processor saved.")

    runtime = (merged_state_dict.tsvm or merged_state_dict.wudi or merged_state_dict.dc or merged_state_dict.iso_c or merged_state_dict.ram_plus
               if isinstance(merged_state_dict, _StreamingMergedStateDict) else None)
    if runtime is not None:
        report = runtime.summary()
        report["base"] = str(merged_state_dict.base_state_dict.model_dir.resolve())
        report["teachers"] = [str(sd.model_dir.resolve()) for sd in merged_state_dict.teacher_state_dicts]
        with (output_dir / f"{report['method']}_merge_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    print(f"Done. Merged checkpoint -> {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher models (TA / TIES / TSVM / WUDI / DC-Merge / Iso-C / RAM+).")
    parser.add_argument(
        "method",
        choices=["ta", "ties", "tsvm", "wudi", "dc", "iso_c", "ram_plus"],
        help="ta = task arithmetic; ties = TIES merging; tsvm = task singular vector merging; wudi = WUDI linear-weight merging; dc = FFT DC-Merge; iso_c = isotropic common-subspace merging; ram_plus = reinforced agent merging",
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="One or more teacher checkpoint directories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged HF checkpoint",
    )
    parser.add_argument(
        "--base",
        help="Base model directory (required for all methods)",
    )
    parser.add_argument(
        "--ties-density",
        type=float,
        default=0.2,
        help="Fraction of largest-magnitude task-vector entries to keep independently "
             "per expert projection (gate/up/down), or per ordinary tensor (default: 0.2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global scale applied to the summed TA task vector or merged TIES/TSVM/WUDI/DC/Iso-C/RAM+ task vector (default: 1.0)",
    )
    parser.add_argument(
        "--teacher-names",
        nargs="+",
        help="Optional labels used in validation error messages",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output safetensors shard size (default: 5GB)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers (default: True)",
    )
    parser.add_argument("--device", default="auto",
                        help="TSVM/WUDI/DC/Iso-C/RAM+ compute device: auto, cpu, cuda:0, ... (default: auto; Iso-C requires CUDA for matrices; RAM+ requires CUDA; TA/TIES unchanged)")
    parser.add_argument("--ram-threshold", type=float, default=1e-5,
                        help="RAM+ active update threshold, nonnegative (default: 1e-5)")
    parser.add_argument("--ram-rescale-factor", type=float, default=1.2,
                        help="RAM+ unique update rescale bound; values <=1 mean RAM (default: 1.2)")
    parser.add_argument("--niter", type=int, default=4,
                        help="TSVM/DC task low-rank SVD subspace iterations, nonnegative (default: 4)")
    parser.add_argument("--oversampling", type=int, default=8,
                        help="TSVM/DC extra sampled directions beyond retained rank, nonnegative (default: 8)")
    parser.add_argument("--wudi-steps", type=int, default=300,
                        help="Nonnegative WUDI Adam iteration count (default: 300)")
    parser.add_argument("--wudi-lr", type=float, default=1e-5,
                        help="Positive WUDI Adam learning rate (default: 1e-5)")
    parser.add_argument("--dc-density", type=float, default=0.001,
                        help="DC cover-space TIES retained fraction (default: 0.001 = 0.1%%)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.base:
        raise SystemExit(f"{args.method} requires --base (shared pretrained checkpoint).")

    teacher_paths = [Path(p) for p in args.teachers]
    teacher_names = args.teacher_names or [p.name for p in teacher_paths]
    if len(teacher_names) != len(teacher_paths):
        raise SystemExit(
            f"--teacher-names expects {len(teacher_paths)} values, "
            f"got {len(teacher_names)}."
        )
    base_path = Path(args.base)
    if args.method in ("tsvm", "wudi", "dc", "iso_c", "ram_plus"):
        _parse_size_bytes(args.max_shard_size)
        if Path(args.output).resolve() in {base_path.resolve(), *(p.resolve() for p in teacher_paths)}:
            raise ValueError(f"{args.method.upper()} output directory must differ from all input checkpoints")
    teacher_states: list[SafetensorCheckpoint] = []
    base_state: SafetensorCheckpoint | None = None

    try:
        print(f"Indexing {len(teacher_paths)} teacher checkpoint(s)...")
        for path in teacher_paths:
            teacher_states.append(load_state_dict(path))
        _validate_compatible_state_dicts(teacher_states, teacher_names)

        print(f"Indexing base checkpoint from {base_path} ...")
        base_state = load_state_dict(base_path)
        if _read_tie_word_embeddings(base_path):
            print("tie_word_embeddings=True: aligning teacher checkpoints with base ...")
            _strip_redundant_lm_head(base_state, teacher_states, teacher_names)
        _validate_compatible_state_dicts(
            [base_state, *teacher_states],
            ["base", *teacher_names],
        )

        if args.method == "ta":
            print(f"TA scale: {args.scale}")
            merged = merge_ta(
                base_state,
                teacher_states,
                scale=args.scale,
            )
        elif args.method == "ties":
            print(f"TIES density: {args.ties_density}; scale: {args.scale}")
            merged = merge_ties(
                base_state,
                teacher_states,
                density=args.ties_density,
                scale=args.scale,
            )
        elif args.method == "ram_plus":
            merged = merge_ram_plus(
                base_state, teacher_states, scale=args.scale, device=args.device,
                threshold=args.ram_threshold, rescale_factor=args.ram_rescale_factor,
            )
        elif args.method == "iso_c":
            merged = merge_iso_c(
                base_state, teacher_states, scale=args.scale, device=args.device,
            )
        elif args.method == "dc":
            merged = merge_dc(
                base_state, teacher_states, scale=args.scale, device=args.device,
                density=args.dc_density, niter=args.niter, oversampling=args.oversampling,
            )
        elif args.method == "wudi":
            merged = merge_wudi(
                base_state, teacher_states, scale=args.scale, device=args.device,
                steps=args.wudi_steps, lr=args.wudi_lr,
            )
        else:
            merged = merge_tsvm(
                base_state, teacher_states, scale=args.scale, device=args.device,
                niter=args.niter, oversampling=args.oversampling,
            )

        save_merged_model_hf(
            merged,
            template_dir=base_path,
            output_dir=args.output,
            max_shard_size=args.max_shard_size,
            trust_remote_code=args.trust_remote_code,
        )
    finally:
        if base_state is not None:
            base_state.close()
        for teacher_state in teacher_states:
            teacher_state.close()


if __name__ == "__main__":
    main()
