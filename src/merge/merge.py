#!/usr/bin/env python3
"""
Merge multiple teacher checkpoints into one HuggingFace-compatible model.

Supported methods:
  - ta: Task Arithmetic
        θ = θ_base + Σ λ_i · (θ_teacher_i − θ_base)
  - ties: TIES Merging

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

Loading strategy:
  - Index safetensors metadata without loading complete checkpoints.
  - Read one base/teacher tensor at a time and merge it in float32.
  - Buffer only one output shard, write it directly, then save the processor.

Requires Python 3.10+ (tested with 3.12).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Callable, Iterable
from contextlib import ExitStack
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

    def get_tensor(self, key: str) -> torch.Tensor:
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
        return handle.get_tensor(key)

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

    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        merged[key] = _merge_ta_tensor(
            base_tensor,
            (teacher_sd[key] for teacher_sd in teacher_state_dicts),
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

    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state_dict.items():
        merged[key] = _merge_ties_tensor(
            base_tensor,
            lambda key=key: (teacher_sd[key] for teacher_sd in teacher_state_dicts),
            density,
            scale,
        )

    return merged


class _StreamingMergedStateDict:
    """Compute one merged tensor on demand without retaining complete checkpoints."""

    def __init__(
        self,
        base_state_dict: SafetensorCheckpoint,
        teacher_state_dicts: list[SafetensorCheckpoint],
        method: str,
        scale: float,
        density: float = 0.2,
    ):
        self.base_state_dict = base_state_dict
        self.teacher_state_dicts = teacher_state_dicts
        self.method = method
        self.scale = scale
        self.density = density

    def keys(self) -> list[str]:
        return self.base_state_dict.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.base_state_dict.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.base_state_dict.get_dtype(key)

    def _teacher_tensors(
        self,
        key: str,
        expected_shape: tuple[int, ...],
    ) -> Iterable[torch.Tensor]:
        for teacher_state in self.teacher_state_dicts:
            tensor = teacher_state.get_tensor(key)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch at {key}: expected {expected_shape}, "
                    f"got {tuple(tensor.shape)} from {teacher_state.model_dir}"
                )
            yield tensor
            del tensor

    def get_tensor(self, key: str) -> torch.Tensor:
        base_tensor = self.base_state_dict.get_tensor(key)
        expected_shape = tuple(base_tensor.shape)

        if self.method == "ta":
            return _merge_ta_tensor(
                base_tensor,
                self._teacher_tensors(key, expected_shape),
                self.scale,
            )
        if self.method == "ties":
            return _merge_ties_tensor(
                base_tensor,
                lambda: self._teacher_tensors(key, expected_shape),
                self.density,
                self.scale,
            )
        raise ValueError(f"Unsupported streaming merge method: {self.method}")


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
    if template_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the base/template directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_safetensors(output_dir)

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
        help="Output directory for merged HF checkpoint",
    )
    parser.add_argument(
        "--base",
        help="Base model directory (required for ta and ties)",
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
        help="Maximum output safetensors shard size (default: 5GB)",
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
    finally:
        if base_state is not None:
            base_state.close()
        for teacher_state in teacher_states:
            teacher_state.close()


if __name__ == "__main__":
    main()
