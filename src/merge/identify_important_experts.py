#!/usr/bin/env python3
"""
Select task-critical MoE experts with data-free functional drift.

For every MoE layer and routed expert, this script evaluates the complete
SwiGLU expert on a shared set of fixed Rademacher hidden-state probes:

    I[l, e] = mean_r ||f_task(z_r) - f_base(z_r)||_2^2
                       / (||f_base(z_r)||_2^2 + eps)

The lowest-scoring fraction of experts in each layer is restored from the
base checkpoint. All other parameters remain exactly those of the task expert
checkpoint. The result is saved as a complete HuggingFace safetensors model.

The script targets the Qwen3-VL-30B-A3B-Instruct packed expert layout:

    ...mlp.experts.gate_up_proj  [num_experts, 2 * intermediate, hidden]
    ...mlp.experts.down_proj     [num_experts, hidden, intermediate]

Example:

    python src/merge/identify_important_experts.py \
        --base /models/Qwen3-VL-30B-A3B-Instruct \
        --expert /models/MedBookVQA \
        --output /models/MedBookVQA-selected \
        --num-probes 16 \
        --seed 42 \
        --revert-fraction 0.5 \
        --device cuda:0 \
        --compute-dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


_GATE_UP_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts)\."
    r"gate_up_proj$"
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
    """Lazy metadata and tensor/slice access for a local HF checkpoint."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.weight_map: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self._stack = ExitStack()
        self._handles: dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        index_path = _find_safetensors_index(self.model_dir)
        if index_path is not None:
            with index_path.open(encoding="utf-8") as f:
                index = json.load(f)
            raw_weight_map = index.get("weight_map")
            if not isinstance(raw_weight_map, dict) or not raw_weight_map:
                raise ValueError(f"Invalid or empty weight_map in {index_path}")
            indexed_weight_map = {
                str(key): str(shard_name)
                for key, shard_name in raw_weight_map.items()
            }
            shard_names = sorted(set(indexed_weight_map.values()))
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
                    f"No safetensors weights found under {self.model_dir}"
                )
            indexed_weight_map = {}

        discovered: dict[str, str] = {}
        for shard_name in shard_names:
            shard_path = self.model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in discovered:
                        raise ValueError(f"Duplicate tensor key in checkpoint: {key}")
                    tensor_slice = handle.get_slice(key)
                    discovered[key] = shard_name
                    self.shapes[key] = tuple(tensor_slice.get_shape())
                    self.dtypes[key] = str(tensor_slice.get_dtype())

        if indexed_weight_map:
            if set(indexed_weight_map) != set(discovered):
                missing = sorted(set(indexed_weight_map) - set(discovered))
                extra = sorted(set(discovered) - set(indexed_weight_map))
                raise ValueError(
                    f"Safetensors index mismatch in {self.model_dir}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            for key, actual_shard in discovered.items():
                if indexed_weight_map[key] != actual_shard:
                    raise ValueError(
                        f"Index maps {key!r} to {indexed_weight_map[key]}, "
                        f"but it is stored in {actual_shard}"
                    )
            self.weight_map = indexed_weight_map
        else:
            self.weight_map = discovered

        print(
            f"  indexed {len(self.weight_map)} tensors from "
            f"{len(set(self.weight_map.values()))} shard(s): {self.model_dir}"
        )

    def keys(self) -> list[str]:
        return list(self.weight_map)

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.shapes[key]

    def get_dtype(self, key: str) -> str:
        return self.dtypes[key]

    def _get_handle(self, key: str) -> Any:
        try:
            shard_name = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.model_dir}") from exc
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

    def get_slice(self, key: str, index: int | slice) -> torch.Tensor:
        return self._get_handle(key).get_slice(key)[index]

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()

    def __enter__(self) -> "SafetensorCheckpoint":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class ExpertLayerLayout:
    layer_index: int
    prefix: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    gate_up_key: str
    down_key: str

    @property
    def label(self) -> str:
        return self.prefix.removesuffix(".mlp.experts")

    def all_weight_keys(self) -> list[str]:
        return [self.gate_up_key, self.down_key]

    def load_expert_batch(
        self,
        checkpoint: SafetensorCheckpoint,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= start < end <= self.num_experts:
            raise ValueError(f"Invalid expert slice [{start}:{end}]")

        gate_up = checkpoint.get_slice(self.gate_up_key, slice(start, end))
        gate, up = gate_up.chunk(2, dim=1)
        down = checkpoint.get_slice(self.down_key, slice(start, end))
        return gate, up, down


def _make_expert_layout(
    checkpoint: SafetensorCheckpoint,
    layer_index: int,
    prefix: str,
    gate_up_key: str,
) -> ExpertLayerLayout:
    down_key = f"{prefix}.down_proj"
    if down_key not in checkpoint:
        raise ValueError(f"Missing Qwen packed expert tensor: {down_key}")

    gate_up_shape = checkpoint.get_shape(gate_up_key)
    down_shape = checkpoint.get_shape(down_key)
    if len(gate_up_shape) != 3 or len(down_shape) != 3:
        raise ValueError(
            f"Qwen packed expert tensors at {prefix} must be 3D, got "
            f"{gate_up_shape} and {down_shape}"
        )
    num_experts, twice_intermediate, hidden_size = gate_up_shape
    down_experts, down_hidden, intermediate_size = down_shape
    if (
        twice_intermediate != 2 * intermediate_size
        or down_experts != num_experts
        or down_hidden != hidden_size
    ):
        raise ValueError(
            f"Incompatible Qwen packed expert shapes at {prefix}: "
            f"gate_up={gate_up_shape}, down={down_shape}"
        )
    return ExpertLayerLayout(
        layer_index=layer_index,
        prefix=prefix,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_up_key=gate_up_key,
        down_key=down_key,
    )


def discover_expert_layers(
    checkpoint: SafetensorCheckpoint,
) -> list[ExpertLayerLayout]:
    layouts: list[ExpertLayerLayout] = []
    for key in checkpoint.keys():
        match = _GATE_UP_EXPERT_RE.match(key)
        if match:
            layouts.append(
                _make_expert_layout(
                    checkpoint,
                    int(match.group("layer")),
                    match.group("prefix"),
                    key,
                )
            )

    layouts.sort(key=lambda layout: (layout.layer_index, layout.prefix))
    if not layouts:
        raise ValueError(
            "No Qwen3-VL packed routed-expert tensors found. Expected keys "
            "'...layers.0.mlp.experts.gate_up_proj' and matching down_proj."
        )
    return layouts


def validate_base_compatibility(
    base: SafetensorCheckpoint,
    task: SafetensorCheckpoint,
    layouts: list[ExpertLayerLayout],
) -> None:
    hidden_sizes = {layout.hidden_size for layout in layouts}
    if len(hidden_sizes) != 1:
        raise ValueError(
            "All MoE layers must share one hidden size so they can use the same probes, "
            f"got {sorted(hidden_sizes)}"
        )

    for layout in layouts:
        for key in layout.all_weight_keys():
            if key not in base:
                raise ValueError(f"Base checkpoint is missing expert tensor: {key}")
            if base.get_shape(key) != task.get_shape(key):
                raise ValueError(
                    f"Shape mismatch at {key}: base={base.get_shape(key)}, "
                    f"task={task.get_shape(key)}"
                )


def make_rademacher_probes(
    num_probes: int,
    hidden_size: int,
    seed: int,
) -> torch.Tensor:
    if num_probes <= 0:
        raise ValueError(f"num_probes must be positive, got {num_probes}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probes = torch.randint(
        low=0,
        high=2,
        size=(num_probes, hidden_size),
        generator=generator,
        dtype=torch.int8,
    )
    return probes.to(torch.float32).mul_(2.0).sub_(1.0)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device_arg}")
    return device


def _resolve_compute_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_arg]
    if device.type == "cpu" and dtype == torch.float16:
        print("Warning: float16 CPU matmul can be slow or unsupported; float32 is safer.")
    return dtype


def _swiglu_expert_forward(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    gate = gate.to(device=device, dtype=compute_dtype)
    up = up.to(device=device, dtype=compute_dtype)
    down = down.to(device=device, dtype=compute_dtype)

    gate_hidden = torch.einsum("rh,eih->eri", probes, gate)
    up_hidden = torch.einsum("rh,eih->eri", probes, up)
    intermediate = F.silu(gate_hidden).mul_(up_hidden)
    output = torch.einsum("eri,ehi->erh", intermediate, down)
    return output.float()


@torch.inference_mode()
def score_expert_layers(
    base: SafetensorCheckpoint,
    task: SafetensorCheckpoint,
    layouts: list[ExpertLayerLayout],
    probes_cpu: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    expert_batch_size: int,
    eps: float,
) -> dict[str, list[float]]:
    if expert_batch_size <= 0:
        raise ValueError("expert_batch_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    probes = probes_cpu.to(device=device, dtype=compute_dtype)
    scores_by_layer: dict[str, list[float]] = {}

    for layer_number, layout in enumerate(layouts, start=1):
        print(
            f"Scoring layer {layer_number}/{len(layouts)}: {layout.label} "
            f"({layout.num_experts} packed experts)"
        )
        layer_scores: list[float] = []
        for start in range(0, layout.num_experts, expert_batch_size):
            end = min(start + expert_batch_size, layout.num_experts)

            base_gate, base_up, base_down = layout.load_expert_batch(
                base, start, end
            )
            base_output = _swiglu_expert_forward(
                probes,
                base_gate,
                base_up,
                base_down,
                device=device,
                compute_dtype=compute_dtype,
            )
            del base_gate, base_up, base_down

            task_gate, task_up, task_down = layout.load_expert_batch(
                task, start, end
            )
            task_output = _swiglu_expert_forward(
                probes,
                task_gate,
                task_up,
                task_down,
                device=device,
                compute_dtype=compute_dtype,
            )
            del task_gate, task_up, task_down

            numerator = (task_output - base_output).square().sum(dim=-1)
            denominator = base_output.square().sum(dim=-1).add_(eps)
            batch_scores = numerator.div_(denominator).mean(dim=-1)
            if not torch.isfinite(batch_scores).all():
                raise FloatingPointError(
                    f"Non-finite functional-drift score at {layout.label}, "
                    f"experts [{start}:{end}]. Try --compute-dtype float32."
                )
            layer_scores.extend(batch_scores.cpu().tolist())
            del base_output, task_output, numerator, denominator, batch_scores

        scores_by_layer[layout.prefix] = layer_scores
        print(
            "  score min/median/max: "
            f"{min(layer_scores):.6g} / {statistics.median(layer_scores):.6g} / "
            f"{max(layer_scores):.6g}"
        )

    return scores_by_layer


def select_experts_to_revert(
    layouts: list[ExpertLayerLayout],
    scores_by_layer: dict[str, list[float]],
    revert_fraction: float,
) -> dict[str, list[int]]:
    if not 0.0 <= revert_fraction <= 1.0:
        raise ValueError(
            f"revert_fraction must be in [0, 1], got {revert_fraction}"
        )

    selected: dict[str, list[int]] = {}
    for layout in layouts:
        scores = scores_by_layer[layout.prefix]
        num_revert = math.floor(layout.num_experts * revert_fraction)
        ranked = sorted(range(layout.num_experts), key=lambda e: (scores[e], e))
        reverted = sorted(ranked[:num_revert])
        selected[layout.prefix] = reverted
        if reverted:
            highest_reverted = max(scores[e] for e in reverted)
            print(
                f"  {layout.label}: reverting {len(reverted)}/{layout.num_experts} "
                f"experts (bottom score <= {highest_reverted:.6g})"
            )
        else:
            print(f"  {layout.label}: reverting 0/{layout.num_experts} experts")
    return selected


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous + 1))
        start = previous = index
    ranges.append((start, previous + 1))
    return ranges


class SelectedCheckpoint:
    """Task checkpoint with selected expert projection tensors restored from base."""

    def __init__(
        self,
        base: SafetensorCheckpoint,
        task: SafetensorCheckpoint,
        layouts: list[ExpertLayerLayout],
        reverted_experts: dict[str, list[int]],
    ):
        self.base = base
        self.task = task
        self.replacements: dict[str, list[int]] = {}

        for layout in layouts:
            expert_ids = reverted_experts[layout.prefix]
            self.replacements[layout.gate_up_key] = expert_ids
            self.replacements[layout.down_key] = expert_ids

    def keys(self) -> list[str]:
        return self.task.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.task.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.task.get_dtype(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        expert_ids = self.replacements.get(key)
        if expert_ids is None or not expert_ids:
            return self.task.get_tensor(key)

        output = self.task.get_tensor(key).clone()
        for start, end in _contiguous_ranges(expert_ids):
            output[start:end].copy_(self.base.get_slice(key, slice(start, end)))
        return output


def _parse_size_bytes(size: str) -> int:
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
    decimal = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    binary = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    size_bytes = int(value * (decimal | binary)[unit])
    if size_bytes <= 0:
        raise ValueError("max shard size must be positive")
    return size_bytes


def _tensor_nbytes(state: SelectedCheckpoint, key: str) -> int:
    dtype_name = state.get_dtype(key)
    try:
        element_size = _SAFETENSORS_DTYPE_BYTES[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported safetensors dtype {dtype_name!r} at {key}"
        ) from exc
    return math.prod(state.get_shape(key)) * element_size


def _plan_output_shards(
    state: SelectedCheckpoint,
    max_shard_size: str,
) -> list[tuple[str, list[str]]]:
    max_bytes = _parse_size_bytes(max_shard_size)
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in state.keys():
        size = _tensor_nbytes(state, key)
        if current and current_size + size > max_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    if not shards:
        raise ValueError("Cannot save an empty checkpoint")

    if len(shards) == 1:
        return [("model.safetensors", shards[0])]
    total = len(shards)
    return [
        (f"model-{index:05d}-of-{total:05d}.safetensors", keys)
        for index, keys in enumerate(shards, start=1)
    ]


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_dir.glob("model*.safetensors"):
            if path.is_file():
                path.unlink()
        index_path = output_dir / "model.safetensors.index.json"
        if index_path.is_file():
            index_path.unlink()


def _copy_model_configs(source_dir: Path, output_dir: Path) -> None:
    for name in ("config.json", "generation_config.json"):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def _save_processor(
    source_dir: Path,
    output_dir: Path,
    trust_remote_code: bool,
) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(source_dir),
        trust_remote_code=trust_remote_code,
    )
    processor.save_pretrained(str(output_dir))
    print(f"Processor/tokenizer saved from {source_dir}")


def save_selected_checkpoint(
    state: SelectedCheckpoint,
    *,
    expert_dir: Path,
    processor_source: Path,
    output_dir: Path,
    max_shard_size: str,
    save_processor: bool,
    trust_remote_code: bool,
) -> None:
    shard_plan = _plan_output_shards(state, max_shard_size)
    weight_map: dict[str, str] = {}
    total_size = 0
    print(
        f"Saving complete selected checkpoint to {len(shard_plan)} shard(s): "
        f"{output_dir}"
    )
    for shard_index, (shard_name, keys) in enumerate(shard_plan, start=1):
        shard_state: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor = state.get_tensor(key).contiguous()
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
            "w", encoding="utf-8"
        ) as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
            f.write("\n")

    _copy_model_configs(expert_dir, output_dir)
    if save_processor:
        _save_processor(processor_source, output_dir, trust_remote_code)


def save_selection_reports(
    output_dir: Path,
    layouts: list[ExpertLayerLayout],
    scores_by_layer: dict[str, list[float]],
    reverted_experts: dict[str, list[int]],
    *,
    base_dir: Path,
    expert_dir: Path,
    num_probes: int,
    seed: int,
    revert_fraction: float,
    eps: float,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    layers_json: list[dict[str, Any]] = []
    for layout in layouts:
        reverted = set(reverted_experts[layout.prefix])
        layers_json.append(
            {
                "layer_index": layout.layer_index,
                "layer_path": layout.label,
                "layout": "qwen3_vl_packed_gate_up",
                "num_experts": layout.num_experts,
                "hidden_size": layout.hidden_size,
                "intermediate_size": layout.intermediate_size,
                "scores": scores_by_layer[layout.prefix],
                "reverted_expert_ids": sorted(reverted),
                "kept_expert_ids": [
                    e for e in range(layout.num_experts) if e not in reverted
                ],
            }
        )

    report = {
        "method": "data_free_functional_drift",
        "formula": (
            "mean_r(||f_task(z_r)-f_base(z_r)||_2^2 / "
            "(||f_base(z_r)||_2^2+eps))"
        ),
        "probe_distribution": "Rademacher({-1,+1})",
        "base_model": str(base_dir.resolve()),
        "expert_model": str(expert_dir.resolve()),
        "num_probes": num_probes,
        "seed": seed,
        "revert_fraction": revert_fraction,
        "selection_rule": "lowest scores independently within each layer",
        "eps": eps,
        "device": str(device),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "layers": layers_json,
    }
    json_path = output_dir / "functional_drift_selection.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    csv_path = output_dir / "functional_drift_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer_index", "layer_path", "expert_id", "score", "action"]
        )
        for layout in layouts:
            reverted = set(reverted_experts[layout.prefix])
            for expert_id, score in enumerate(scores_by_layer[layout.prefix]):
                writer.writerow(
                    [
                        layout.layer_index,
                        layout.label,
                        expert_id,
                        f"{score:.17g}",
                        "revert_to_base" if expert_id in reverted else "keep_task",
                    ]
                )
    print(f"Selection reports saved: {json_path.name}, {csv_path.name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use data-free SwiGLU functional drift to keep the most changed "
            "internal MoE experts and restore the rest from the base model."
        )
    )
    parser.add_argument("--base", required=True, help="Local base model directory")
    parser.add_argument(
        "--expert", required=True, help="Local task-expert checkpoint directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--num-probes",
        "-R",
        type=int,
        default=16,
        help="Number of shared Rademacher hidden vectors (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Probe RNG seed (default: 42)"
    )
    parser.add_argument(
        "--revert-fraction",
        type=float,
        default=0.5,
        help=(
            "Lowest-scoring fraction restored to base independently per layer "
            "(default: 0.5)"
        ),
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=8,
        help="Experts scored together on the compute device (default: 8)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device, e.g. auto, cuda:0, or cpu (default: auto)",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="SwiGLU scoring dtype (default: auto)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Positive denominator stabilizer (default: 1e-8)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output safetensors shard size (default: 5GB)",
    )
    parser.add_argument(
        "--processor-source",
        help=(
            "Optional tokenizer/processor source directory; defaults to --base "
            "because fine-tuning does not change these artifacts"
        ),
    )
    parser.add_argument(
        "--save-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save processor and tokenizer into the output (default: true)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom processor code when loading local artifacts (default: true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base)
    expert_dir = Path(args.expert)
    output_dir = Path(args.output)
    processor_source = (
        Path(args.processor_source) if args.processor_source else base_dir
    )

    resolved_output = output_dir.resolve()
    if resolved_output in {base_dir.resolve(), expert_dir.resolve()}:
        raise ValueError("Output directory must differ from base and expert directories")
    if args.save_processor and not processor_source.is_dir():
        raise FileNotFoundError(
            f"Processor source directory not found: {processor_source}"
        )
    _prepare_output_dir(output_dir, args.overwrite)

    device = _resolve_device(args.device)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype, device)
    print(f"Scoring device={device}, compute_dtype={compute_dtype}")

    with SafetensorCheckpoint(base_dir) as base, SafetensorCheckpoint(
        expert_dir
    ) as task:
        layouts = discover_expert_layers(task)
        validate_base_compatibility(base, task, layouts)
        hidden_size = layouts[0].hidden_size
        print(
            f"Discovered {len(layouts)} MoE layers; hidden_size={hidden_size}; "
            f"shared probes R={args.num_probes}, seed={args.seed}"
        )

        probes = make_rademacher_probes(args.num_probes, hidden_size, args.seed)
        scores = score_expert_layers(
            base,
            task,
            layouts,
            probes,
            device=device,
            compute_dtype=compute_dtype,
            expert_batch_size=args.expert_batch_size,
            eps=args.eps,
        )
        reverted = select_experts_to_revert(
            layouts, scores, args.revert_fraction
        )
        selected_state = SelectedCheckpoint(base, task, layouts, reverted)
        save_selected_checkpoint(
            selected_state,
            expert_dir=expert_dir,
            processor_source=processor_source,
            output_dir=output_dir,
            max_shard_size=args.max_shard_size,
            save_processor=args.save_processor,
            trust_remote_code=args.trust_remote_code,
        )
        save_selection_reports(
            output_dir,
            layouts,
            scores,
            reverted,
            base_dir=base_dir,
            expert_dir=expert_dir,
            num_probes=args.num_probes,
            seed=args.seed,
            revert_fraction=args.revert_fraction,
            eps=args.eps,
            device=device,
            compute_dtype=compute_dtype,
        )

    print(f"Done. Selected checkpoint -> {output_dir}")


if __name__ == "__main__":
    main()
