#!/usr/bin/env python3
"""
Merge multiple Qwen3-VL MoE teacher checkpoints into one HuggingFace model.

Supported methods:
  - ta: Task Arithmetic
        theta = theta_base + scale * sum(theta_teacher_i - theta_base)
  - ties: TIES Merging

Example:
    python src/merge/merge.py ta \
        --base /path/Qwen3-VL-30B-A3B-Instruct \
        --teachers /path/math /path/code /path/logic \
        --scale 1.0 \
        --output /path/merged_ta

    python src/merge/merge.py ties \
        --base /path/Qwen3-VL-30B-A3B-Instruct \
        --teachers /path/math /path/code /path/logic \
        --ties-density 0.2 \
        --scale 1.0 \
        --output /path/merged_ties

Loading strategy:
  - Load complete base/teacher state dicts from safetensors on CPU.
  - Merge one parameter key at a time in float32 while retaining the complete
    input and merged state dicts in memory.
  - Load a Qwen3-VL MoE model from the base checkpoint, inject the merged
    weights, then save a sharded HuggingFace checkpoint and processor.

Requires Python 3.10+ and transformers==4.57.3.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen3VLMoeForConditionalGeneration is unavailable. "
        "Install the project requirement transformers==4.57.3."
    ) from exc


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


def _load_safetensors_shards(
    model_dir: Path,
    shard_names: list[str],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    tensor_sources: dict[str, str] = {}

    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing shard file referenced by checkpoint: {shard_path}"
            )
        shard_state = load_file(str(shard_path), device="cpu")
        duplicate_keys = set(state).intersection(shard_state)
        if duplicate_keys:
            key = sorted(duplicate_keys)[0]
            raise ValueError(
                f"Duplicate tensor key {key!r} in {model_dir}: "
                f"{tensor_sources[key]} and {shard_name}"
            )
        state.update(shard_state)
        tensor_sources.update({key: shard_name for key in shard_state})

    print(
        f"  loaded {len(state)} tensors from "
        f"{len(shard_names)} safetensors shard(s)"
    )
    return state


def load_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load one complete HuggingFace safetensors checkpoint on CPU."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    index_path = _find_safetensors_index(model_dir)
    if index_path is not None:
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")

        shard_names = sorted(set(map(str, raw_weight_map.values())))
        state = _load_safetensors_shards(model_dir, shard_names)
        indexed_keys = set(map(str, raw_weight_map))
        loaded_keys = set(state)
        if indexed_keys != loaded_keys:
            missing = sorted(indexed_keys - loaded_keys)
            extra = sorted(loaded_keys - indexed_keys)
            raise ValueError(
                f"Safetensors index mismatch in {model_dir}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        return state

    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return _load_safetensors_shards(model_dir, [single_path.name])

    shard_paths = sorted(
        model_dir.glob("model-*-of-*.safetensors"),
        key=_shard_sort_key,
    )
    if shard_paths:
        return _load_safetensors_shards(
            model_dir,
            [path.name for path in shard_paths],
        )

    raise FileNotFoundError(
        f"No safetensors weights found under {model_dir}. Expected "
        "model.safetensors or model-*-of-*.safetensors with an optional index."
    )


def _validate_compatible_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    names: list[str],
) -> None:
    if not state_dicts:
        raise ValueError("At least one state dict is required.")
    if len(state_dicts) != len(names):
        raise ValueError(
            f"Expected {len(state_dicts)} state dict names, got {len(names)}"
        )

    ref_keys = set(state_dicts[0])
    for name, state_dict in zip(names[1:], state_dicts[1:]):
        keys = set(state_dict)
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            message = [f"State dict mismatch for {name}."]
            if missing:
                message.append(f"  missing keys ({len(missing)}): {missing[:5]}...")
            if extra:
                message.append(f"  extra keys ({len(extra)}): {extra[:5]}...")
            raise ValueError("\n".join(message))

    reference = state_dicts[0]
    for name, state_dict in zip(names[1:], state_dicts[1:]):
        for key, reference_tensor in reference.items():
            other_tensor = state_dict[key]
            if reference_tensor.shape != other_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at {key}: {names[0]} "
                    f"{tuple(reference_tensor.shape)} vs {name} "
                    f"{tuple(other_tensor.shape)}"
                )


def _merge_ta_tensor(
    base_tensor: torch.Tensor,
    teacher_tensors: Iterable[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    if not torch.is_floating_point(base_tensor):
        return base_tensor.clone()

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
    base_state_dict: dict[str, torch.Tensor],
    teacher_state_dicts: list[dict[str, torch.Tensor]],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Task arithmetic: base + scale * sum_i(teacher_i - base)."""
    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        merged[key] = _merge_ta_tensor(
            base_tensor,
            (teacher_state[key] for teacher_state in teacher_state_dicts),
            scale,
        )
    return merged


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

    aligned_count.clamp_min_(1.0)
    aligned_sum.div_(aligned_count)
    base_float.add_(aligned_sum, alpha=scale)
    del aligned_sum, aligned_count, elected_sign
    return base_float.to(base_tensor.dtype)


def merge_ties(
    base_state_dict: dict[str, torch.Tensor],
    teacher_state_dicts: list[dict[str, torch.Tensor]],
    density: float = 0.2,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    TIES Merging:
      1. build task vectors teacher_i - base
      2. trim each vector to its largest-magnitude density fraction
      3. elect the dominant sign at each parameter by summed update mass
      4. average only trimmed updates matching the elected sign
      5. add scale * merged_delta back to the base model
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"ties density must be in (0, 1], got {density}")

    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        merged[key] = _merge_ties_tensor(
            base_tensor,
            lambda key=key: (
                teacher_state[key] for teacher_state in teacher_state_dicts
            ),
            density,
            scale,
        )
    return merged


def save_merged_model_hf(
    merged_state_dict: dict[str, torch.Tensor],
    template_dir: str | Path,
    output_dir: str | Path,
    max_shard_size: str = "5GB",
    trust_remote_code: bool = True,
) -> None:
    """Inject merged weights into Qwen3-VL MoE and save an HF checkpoint."""
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    if template_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the base/template directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen3-VL MoE model from {template_dir} ...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        str(template_dir),
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    model.load_state_dict(merged_state_dict, strict=True)

    print(f"Saving merged model to {output_dir} (max_shard_size={max_shard_size}) ...")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    processor = AutoProcessor.from_pretrained(
        str(template_dir),
        trust_remote_code=trust_remote_code,
    )
    processor.save_pretrained(str(output_dir))
    print("Processor saved.")
    print(f"Done. Merged checkpoint -> {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher models (TA / TIES).")
    parser.add_argument(
        "method",
        choices=["ta", "ties"],
        help="ta = task arithmetic; ties = TIES merging",
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
        help="Output directory",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Shared pretrained base checkpoint directory",
    )
    parser.add_argument(
        "--ties-density",
        type=float,
        default=0.2,
        help="Fraction of largest-magnitude task-vector entries retained by TIES",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global scale applied to the merged task vector",
    )
    parser.add_argument(
        "--teacher-names",
        nargs="+",
        help="Optional labels used in validation error messages",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum HuggingFace output shard size (default: 5GB)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to Transformers (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    teacher_paths = [Path(path) for path in args.teachers]
    teacher_names = args.teacher_names or [path.name for path in teacher_paths]
    if len(teacher_names) != len(teacher_paths):
        raise SystemExit(
            f"--teacher-names expects {len(teacher_paths)} values, "
            f"got {len(teacher_names)}."
        )

    print(f"Loading {len(teacher_paths)} complete teacher checkpoint(s)...")
    teacher_states = [load_state_dict(path) for path in teacher_paths]
    _validate_compatible_state_dicts(teacher_states, teacher_names)

    base_path = Path(args.base)
    print(f"Loading complete base checkpoint from {base_path} ...")
    base_state = load_state_dict(base_path)
    _validate_compatible_state_dicts(
        [base_state, *teacher_states],
        ["base", *teacher_names],
    )

    if args.method == "ta":
        print(f"TA scale: {args.scale}")
        merged = merge_ta(base_state, teacher_states, scale=args.scale)
    else:
        print(f"TIES density: {args.ties_density}; scale: {args.scale}")
        merged = merge_ties(
            base_state,
            teacher_states,
            density=args.ties_density,
            scale=args.scale,
        )

    save_merged_model_hf(
        merged,
        template_dir=base_path,
        output_dir=args.output,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
