#!/usr/bin/env python3
"""
Merge multiple teacher checkpoints into one HuggingFace-compatible model.

Supported methods:
  - wa: Weight Averaging
        θ = Σ w_i · θ_teacher_i
  - ta: Task Arithmetic
        θ = θ_base + Σ λ_i · (θ_teacher_i − θ_base)

Example:
    python merge_recipe/merge.py wa \\
        --teachers /path/math /path/code /path/logic \\
        --output /path/merged_wa

    python merge_recipe/merge.py ta \\
        --base /path/Qwen3-4B \\
        --teachers /path/math /path/code /path/logic \\
        --scale 1.0 \\
        --output /path/merged_ta

    python merge_recipe/merge.py ties \\
        --base /path/Qwen3-4B \\
        --teachers /path/math /path/code /path/logic \\
        --ties-density 0.2 \\
        --scale 1.0 \\
        --output /path/merged_ties

Loading strategy:
  - Teachers/base weights: read safetensors shards directly (memory friendly).
  - Save: one HF model skeleton via from_pretrained + load_state_dict +
    save_pretrained(max_shard_size=...) for sharded output + tokenizer.

Requires Python 3.10+ (tested with 3.12).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_pytorch_bin(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


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


def _load_sharded_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = _find_safetensors_index(model_dir)
    if index_path is not None:
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard file referenced by index: {shard_path}")
            state.update(load_file(str(shard_path), device="cpu"))
        print(f"  loaded {len(state)} tensors from {len(set(weight_map.values()))} shard(s) via {index_path.name}")
        return state

    shard_paths = sorted(model_dir.glob("model-*-of-*.safetensors"), key=_shard_sort_key)
    if not shard_paths:
        return {}

    state = {}
    for shard_path in shard_paths:
        state.update(load_file(str(shard_path), device="cpu"))
    print(f"  loaded {len(state)} tensors from {len(shard_paths)} shard(s): {shard_paths[0].name} ...")
    return state


def _load_sharded_pytorch_bin(model_dir: Path) -> dict[str, torch.Tensor]:
    bin_index_path = model_dir / "pytorch_model.bin.index.json"
    if bin_index_path.is_file():
        with bin_index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state = {}
        for shard_name in sorted(set(weight_map.values())):
            state.update(_load_pytorch_bin(model_dir / shard_name))
        print(f"  loaded {len(state)} tensors from pytorch shards via index")
        return state

    shard_paths = sorted(model_dir.glob("pytorch_model-*-of-*.bin"), key=_shard_sort_key)
    if not shard_paths:
        return {}

    state = {}
    for shard_path in shard_paths:
        state.update(_load_pytorch_bin(shard_path))
    print(f"  loaded {len(state)} tensors from {len(shard_paths)} pytorch shard(s)")
    return state


def load_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load a HF checkpoint state dict from safetensors or pytorch bin."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    single_st_path = model_dir / "model.safetensors"
    if single_st_path.is_file():
        state = dict(load_file(str(single_st_path), device="cpu"))
        print(f"  loaded {len(state)} tensors from model.safetensors")
        return state

    state = _load_sharded_safetensors(model_dir)
    if state:
        return state

    single_bin_path = model_dir / "pytorch_model.bin"
    if single_bin_path.is_file():
        state = _load_pytorch_bin(single_bin_path)
        print(f"  loaded {len(state)} tensors from pytorch_model.bin")
        return state

    state = _load_sharded_pytorch_bin(model_dir)
    if state:
        return state

    raise FileNotFoundError(
        f"No supported weight files found under {model_dir}. "
        "Expected one of: model.safetensors, model-*-of-*.safetensors (+ optional index), "
        "pytorch_model.bin, pytorch_model-*-of-*.bin."
    )


def _read_tie_word_embeddings(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    return bool(config.get("tie_word_embeddings", False))


def _strip_redundant_lm_head(
    reference: dict[str, torch.Tensor],
    state_dicts: list[dict[str, torch.Tensor]],
    names: list[str],
) -> None:
    """Drop lm_head.weight when it duplicates embed_tokens (common in verl FSDP exports)."""
    if "lm_head.weight" in reference:
        return

    embed_key = "model.embed_tokens.weight"
    for name, sd in zip(names, state_dicts, strict=True):
        if "lm_head.weight" not in sd:
            continue
        if embed_key not in sd:
            raise ValueError(f"{name} has lm_head.weight but missing {embed_key}")
        if not torch.equal(sd[embed_key], sd["lm_head.weight"]):
            raise ValueError(
                f"{name}: lm_head.weight != {embed_key}; refusing to strip non-tied lm_head"
            )
        sd.pop("lm_head.weight")
        print(f"  stripped redundant lm_head.weight from {name}")


def _validate_compatible_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    names: list[str],
) -> None:
    if not state_dicts:
        raise ValueError("At least one state dict is required.")

    ref_keys = set(state_dicts[0].keys())
    for name, sd in zip(names[1:], state_dicts[1:], strict=True):
        keys = set(sd.keys())
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
    for name, sd in zip(names[1:], state_dicts[1:], strict=True):
        for key, ref_tensor in ref.items():
            other = sd[key]
            if ref_tensor.shape != other.shape:
                raise ValueError(
                    f"Shape mismatch at {key}: {names[0]} {tuple(ref_tensor.shape)} "
                    f"vs {name} {tuple(other.shape)}"
                )


def _normalize_weights(weights: list[float], n: int, normalize: bool = False) -> list[float]:
    if weights is None:
        if normalize:
            return [1.0 / n] * n
        else:
            return [1.0] * n
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    if normalize:
        weights = [w / sum(weights) for w in weights]
    return weights


def merge_wa(
    teacher_state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Weight averaging over teacher checkpoints."""
    weights = _normalize_weights(weights, len(teacher_state_dicts))
    keys = teacher_state_dicts[0].keys()
    merged: dict[str, torch.Tensor] = {}

    for key in keys:
        acc = None
        out_dtype = teacher_state_dicts[0][key].dtype
        for weight, sd in zip(weights, teacher_state_dicts, strict=True):
            tensor = sd[key].float() * weight
            acc = tensor if acc is None else acc + tensor
        merged[key] = acc.to(out_dtype)

    return merged


def merge_ta(
    base_state_dict: dict[str, torch.Tensor],
    teacher_state_dicts: list[dict[str, torch.Tensor]],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Task arithmetic: base + scale * sum_i(teacher_i - base)."""

    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        base_float = base_tensor.float()
        merged_delta = torch.zeros_like(base_float)
        for teacher_sd in teacher_state_dicts:
            merged_delta.add_(teacher_sd[key].float() - base_float)
        merged[key] = (base_float + scale * merged_delta).to(base_tensor.dtype)

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
      3. elect the dominant sign at each parameter by summing trimmed vectors
      4. average only trimmed updates that match the elected sign
      5. add scale * merged_delta back to the base model

    Defaults follow the no-validation recipe from the TIES-Merging paper:
    top-20% task-vector entries, mass-based sign election, disjoint mean, and
    lambda/scale = 1.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"ties density must be in (0, 1], got {density}")

    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        if not torch.is_floating_point(base_tensor):
            merged[key] = base_tensor.clone()
            continue

        base_float = base_tensor.float()
        sign_votes = torch.zeros_like(base_float)

        for teacher_sd in teacher_state_dicts:
            delta = teacher_sd[key].float() - base_float
            trim_mask = _topk_magnitude_mask(delta, density)
            delta.masked_fill_(~trim_mask, 0.0)
            sign_votes.add_(delta)

        elected_sign = torch.sign(sign_votes)
        aligned_sum = torch.zeros_like(base_float)
        aligned_count = torch.zeros_like(base_float)

        for teacher_sd in teacher_state_dicts:
            delta = teacher_sd[key].float() - base_float
            trim_mask = _topk_magnitude_mask(delta, density)
            aligned_mask = trim_mask & (torch.sign(delta) == elected_sign) & (elected_sign != 0)
            aligned_count.add_(aligned_mask.float())
            delta.masked_fill_(~aligned_mask, 0.0)
            aligned_sum.add_(delta)

        merged_delta = torch.where(
            aligned_count > 0,
            aligned_sum / aligned_count.clamp_min(1.0),
            torch.zeros_like(aligned_sum),
        )
        merged[key] = (base_float + scale * merged_delta).to(base_tensor.dtype)

    return merged


def _ensure_tied_lm_head_for_load(
    state_dict: dict[str, torch.Tensor],
    model_dir: Path,
) -> None:
    """HF models with tie_word_embeddings still expect lm_head.weight at load time."""
    if not _read_tie_word_embeddings(model_dir):
        return
    embed_key = "model.embed_tokens.weight"
    lm_key = "lm_head.weight"
    if lm_key not in state_dict and embed_key in state_dict:
        state_dict[lm_key] = state_dict[embed_key]
        print("  added lm_head.weight from model.embed_tokens.weight for tied load")


def save_merged_model_hf(
    merged_state_dict: dict[str, torch.Tensor],
    template_dir: str | Path,
    output_dir: str | Path,
    max_shard_size: str = "5GB",
    trust_remote_code: bool = True,
) -> None:
    """Load one HF model shell, inject merged weights, save sharded checkpoint."""
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF model skeleton from {template_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(template_dir),
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    _ensure_tied_lm_head_for_load(merged_state_dict, template_dir)
    model.load_state_dict(merged_state_dict, strict=True)

    print(f"Saving merged model to {output_dir} (max_shard_size={max_shard_size}) ...")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(template_dir),
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(str(output_dir))
        print("Tokenizer saved.")
    except Exception as exc:
        print(f"Warning: failed to save tokenizer: {exc}")

    print(f"Done. Merged checkpoint -> {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher models (WA / TA / TIES).")
    parser.add_argument(
        "method",
        choices=["wa", "ta", "ties"],
        help="wa = weight averaging; ta = task arithmetic; ties = TIES merging",
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
        "--normalize",
        action="store_true",
        help="Normalize weight-averaging weights to sum to 1.0 (wa only)",
    )
    parser.add_argument(
        "--base",
        help="Base model directory (required for ta and ties)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Teacher weights for wa",
    )
    parser.add_argument(
        "--ties-density",
        type=float,
        default=0.2,
        help="Fraction of largest-magnitude task-vector entries to keep for TIES (paper default: 0.2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global scale applied to the summed TA task vector or merged TIES task vector (default: 1.0)",
    )
    parser.add_argument(
        "--teacher-names",
        nargs="+",
        help="Optional labels used in validation error messages",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="HF save_pretrained shard size (default: 5GB)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.normalize and args.method != "wa":
        raise SystemExit("--normalize is only valid for weight averaging (method: wa).")

    teacher_paths = [Path(p) for p in args.teachers]
    teacher_names = args.teacher_names or [p.name for p in teacher_paths]

    print(f"Loading {len(teacher_paths)} teacher checkpoint(s)...")
    teacher_states = [load_state_dict(path) for path in teacher_paths]
    _validate_compatible_state_dicts(teacher_states, teacher_names)

    if args.method == "wa":
        weights = _normalize_weights(args.weights, len(teacher_states), args.normalize)
        print(f"WA weights: {dict(zip(teacher_names, weights, strict=True))}")
        merged = merge_wa(teacher_states, weights)
        template_dir = teacher_paths[0]
    elif args.method == "ta":
        if not args.base:
            raise SystemExit("ta requires --base (shared pretrained checkpoint).")
        base_path = Path(args.base)
        print(f"Loading base model from {base_path} ...")
        base_state = load_state_dict(base_path)
        if _read_tie_word_embeddings(base_path):
            print("tie_word_embeddings=True: aligning teacher checkpoints with base ...")
            _strip_redundant_lm_head(base_state, teacher_states, teacher_names)
        _validate_compatible_state_dicts(
            [base_state, *teacher_states],
            ["base", *teacher_names],
        )
        print(f"TA scale: {args.scale}")
        merged = merge_ta(base_state, teacher_states, scale=args.scale)
        template_dir = base_path
    else:
        if not args.base:
            raise SystemExit("ties requires --base (shared pretrained checkpoint).")
        base_path = Path(args.base)
        print(f"Loading base model from {base_path} ...")
        base_state = load_state_dict(base_path)
        if _read_tie_word_embeddings(base_path):
            print("tie_word_embeddings=True: aligning teacher checkpoints with base ...")
            _strip_redundant_lm_head(base_state, teacher_states, teacher_names)
        _validate_compatible_state_dicts(
            [base_state, *teacher_states],
            ["base", *teacher_names],
        )
        print(f"TIES density: {args.ties_density}; scale: {args.scale}")
        merged = merge_ties(
            base_state,
            teacher_states,
            density=args.ties_density,
            scale=args.scale,
        )
        template_dir = base_path

    save_merged_model_hf(
        merged,
        template_dir=template_dir,
        output_dir=args.output,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
