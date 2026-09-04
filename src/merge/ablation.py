#!/usr/bin/env python3
"""Merge non-expert parameters while anchoring all routed MoE experts.

All tensors except Qwen packed routed-expert weights are merged across every
teacher with Task Arithmetic or TIES. By default, every routed expert is copied
unchanged from ``--experts-anchor``. With ``--key-anchor-only``, only experts
marked ``keep_task`` in ``--key-experts-csv`` are copied from the anchor; every
other routed expert is copied from the base model.

Vision-tower tensors under ``model.visual.*`` are merged by default. If
``--vision-anchor`` is supplied, the complete vision tower is instead copied
unchanged from that teacher. Router tensors always use the requested merge method.

Example:
    python src/merge/ablation.py ta \
        --base /models/base \
        --teachers /models/med /models/puzzle /models/nav /models/math \
        --experts-anchor /models/math \
        --vision-anchor /models/math \
        --scale 0.5 \
        --output /models/merged-anchor-math
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import torch


# Direct execution adds src/merge, rather than the repository root, to
# sys.path. Add the root explicitly so the existing merge modules can be reused.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.merge.identify_important_experts import (  # noqa: E402
    ExpertLayerLayout,
    SafetensorCheckpoint,
    _prepare_output_dir,
    discover_expert_layers,
    save_selected_checkpoint,
)
from src.merge.merge import _merge_ta_tensor, _merge_ties_tensor  # noqa: E402


def _validate_checkpoint_compatibility(
    base: SafetensorCheckpoint,
    teachers: list[SafetensorCheckpoint],
) -> None:
    """Require identical tensor keys and shapes across all checkpoints."""
    base_keys = set(base.keys())
    for teacher in teachers:
        teacher_keys = set(teacher.keys())
        if teacher_keys != base_keys:
            missing = sorted(base_keys - teacher_keys)
            extra = sorted(teacher_keys - base_keys)
            raise ValueError(
                f"Checkpoint key mismatch for {teacher.model_dir}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for key in base.keys():
            base_shape = base.get_shape(key)
            teacher_shape = teacher.get_shape(key)
            if teacher_shape != base_shape:
                raise ValueError(
                    f"Shape mismatch at {key}: base={base_shape}, "
                    f"{teacher.model_dir}={teacher_shape}"
                )


def _resolve_teacher_index(
    teacher_dirs: list[Path],
    candidate: Path,
    option_name: str,
) -> int:
    """Return the teacher index matching an anchor by canonical path."""
    resolved_candidate = candidate.expanduser().resolve()
    for index, teacher_dir in enumerate(teacher_dirs):
        if teacher_dir.expanduser().resolve() == resolved_candidate:
            return index
    teacher_list = ", ".join(str(path) for path in teacher_dirs)
    raise ValueError(
        f"{option_name} must be one of the paths passed to --teachers. "
        f"Got anchor={candidate}; teachers=[{teacher_list}]"
    )


def _expert_keys(layouts: list[ExpertLayerLayout]) -> frozenset[str]:
    return frozenset(
        key
        for layout in layouts
        for key in (layout.gate_up_key, layout.down_key)
    )


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


def _read_key_expert_slots(
    path: Path,
    layouts: list[ExpertLayerLayout],
) -> frozenset[tuple[int, int]]:
    """Read keep_task slots and require exact layer/expert CSV coverage."""
    if not path.is_file():
        raise FileNotFoundError(f"Key-expert CSV not found: {path}")

    selected: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer_index", "expert_id", "action"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                slot = (int(row["layer_index"]), int(row["expert_id"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid layer_index/expert_id at {path}:{row_number}"
                ) from exc
            if slot in seen:
                raise ValueError(
                    f"Duplicate layer/expert row at {path}:{row_number}: {slot}"
                )
            seen.add(slot)
            if (row.get("action") or "").strip() == "keep_task":
                selected.add(slot)

    expected = {
        (layout.layer_index, expert_id)
        for layout in layouts
        for expert_id in range(layout.num_experts)
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Key-expert CSV coverage mismatch for {path}: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if not selected:
        raise ValueError(f"No experts with action='keep_task' found in {path}")
    return frozenset(selected)


def _vision_keys(base: SafetensorCheckpoint) -> frozenset[str]:
    keys = frozenset(key for key in base.keys() if key.startswith("model.visual."))
    if not keys:
        raise ValueError(
            "Base checkpoint has no vision-tower tensors under 'model.visual.*'"
        )
    return keys


class AnchorExpertsMergedCheckpoint:
    """Stream merged non-experts and anchor/base expert slices."""

    def __init__(
        self,
        base: SafetensorCheckpoint,
        teachers: list[SafetensorCheckpoint],
        anchor: SafetensorCheckpoint,
        layouts: list[ExpertLayerLayout],
        key_anchor_slots: frozenset[tuple[int, int]] | None,
        vision_anchor: SafetensorCheckpoint | None,
        vision_keys: frozenset[str],
        *,
        method: str,
        scale: float,
        ties_density: float,
    ) -> None:
        self.base = base
        self.teachers = teachers
        self.anchor = anchor
        self.expert_keys = _expert_keys(layouts)
        self.expert_key_layout = {
            key: layout
            for layout in layouts
            for key in (layout.gate_up_key, layout.down_key)
        }
        self.key_anchor_slots = key_anchor_slots
        self.vision_anchor = vision_anchor
        self.vision_keys = vision_keys
        self.method = method
        self.scale = scale
        self.ties_density = ties_density

    def keys(self) -> list[str]:
        return self.base.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        source = self._anchored_source(key) or self.base
        return source.get_shape(key)

    def get_dtype(self, key: str) -> str:
        source = self._anchored_source(key) or self.base
        return source.get_dtype(key)

    def _anchored_source(self, key: str) -> SafetensorCheckpoint | None:
        if key in self.expert_keys and self.key_anchor_slots is None:
            return self.anchor
        if self.vision_anchor is not None and key in self.vision_keys:
            return self.vision_anchor
        return None

    def _teacher_tensors(self, key: str) -> Iterable[torch.Tensor]:
        for teacher in self.teachers:
            yield teacher.get_tensor(key)

    def _merge_non_expert_tensor(self, key: str) -> torch.Tensor:
        base_tensor = self.base.get_tensor(key)
        if self.method == "ta":
            return _merge_ta_tensor(
                base_tensor,
                self._teacher_tensors(key),
                self.scale,
            )
        if self.method == "ties":
            return _merge_ties_tensor(
                base_tensor,
                lambda: self._teacher_tensors(key),
                self.ties_density,
                self.scale,
            )
        raise ValueError(f"Unsupported merge method: {self.method}")

    def _copy_key_anchor_experts(
        self,
        key: str,
        layout: ExpertLayerLayout,
    ) -> torch.Tensor:
        output = self.base.get_tensor(key).clone()
        expert_ids = sorted(
            expert_id
            for layer_index, expert_id in self.key_anchor_slots or ()
            if layer_index == layout.layer_index
        )
        for start, end in _contiguous_ranges(expert_ids):
            output[start:end].copy_(
                self.anchor.get_slice(key, slice(start, end))
            )
        return output

    def get_tensor(self, key: str) -> torch.Tensor:
        layout = self.expert_key_layout.get(key)
        if layout is not None and self.key_anchor_slots is None:
            # gate_up_proj is packed as
            # [num_experts, hidden, gate_intermediate || up_intermediate].
            # Copy that complete packed tensor exactly; do not split or merge it.
            return self.anchor.get_tensor(key)
        if layout is not None:
            return self._copy_key_anchor_experts(key, layout)
        if self.vision_anchor is not None and key in self.vision_keys:
            return self.vision_anchor.get_tensor(key)
        return self._merge_non_expert_tensor(key)


def _save_summary(
    output_dir: Path,
    *,
    method: str,
    scale: float,
    ties_density: float,
    base_dir: Path,
    teacher_dirs: list[Path],
    teacher_names: list[str],
    anchor_dir: Path,
    key_anchor_only: bool,
    key_experts_csv: Path | None,
    key_anchor_slots: frozenset[tuple[int, int]] | None,
    vision_anchor_dir: Path | None,
    layouts: list[ExpertLayerLayout],
) -> None:
    total_expert_slots = sum(layout.num_experts for layout in layouts)
    anchored_expert_slots = (
        len(key_anchor_slots) if key_anchor_slots is not None else total_expert_slots
    )
    expert_rule = (
        "copy_keep_task_anchor_slices_else_base"
        if key_anchor_only
        else "copy_all_anchor_experts"
    )
    summary = {
        "method": method,
        "scale": scale,
        "ties_density": ties_density if method == "ties" else None,
        "base": str(base_dir.resolve()),
        "teachers": [
            {"name": name, "path": str(path.resolve())}
            for name, path in zip(teacher_names, teacher_dirs)
        ],
        "experts_anchor": str(anchor_dir.resolve()),
        "key_anchor_only": key_anchor_only,
        "key_experts_csv": (
            str(key_experts_csv.resolve()) if key_experts_csv is not None else None
        ),
        "vision_anchor": (
            str(vision_anchor_dir.resolve())
            if vision_anchor_dir is not None
            else None
        ),
        "num_moe_layers": len(layouts),
        "num_expert_parameter_tensors": 2 * len(layouts),
        "num_total_expert_slots": total_expert_slots,
        "num_anchored_expert_slots": anchored_expert_slots,
        "num_base_expert_slots": total_expert_slots - anchored_expert_slots,
        "semantics": {
            "mlp.experts.gate_up_proj": expert_rule,
            "mlp.experts.down_proj": expert_rule,
            "vision_tower": (
                "copy_complete_vision_anchor"
                if vision_anchor_dir is not None
                else f"{method}_all_teachers"
            ),
            "router": f"{method}_all_teachers",
            "all_other_tensors": f"{method}_all_teachers",
        },
    }
    with (output_dir / "ablation_merge_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge non-expert tensors with TA/TIES while copying all or only "
            "selected Qwen packed MoE experts, and optionally the vision tower, "
            "from anchors."
        )
    )
    parser.add_argument("method", choices=["ta", "ties"])
    parser.add_argument("--base", required=True, help="Local base checkpoint")
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="Teacher checkpoint directories",
    )
    parser.add_argument(
        "--teacher-names",
        nargs="+",
        help="Optional labels in the same order as --teachers",
    )
    parser.add_argument(
        "--experts-anchor",
        required=True,
        help=(
            "Teacher checkpoint whose gate_up_proj/down_proj expert tensors are "
            "copied; must be one of --teachers"
        ),
    )
    parser.add_argument(
        "--key-anchor-only",
        action="store_true",
        help=(
            "Copy only experts marked keep_task in --key-experts-csv from "
            "--experts-anchor; copy every other routed expert from --base"
        ),
    )
    parser.add_argument(
        "--key-experts-csv",
        help=(
            "Functional-drift CSV for --experts-anchor. Required with "
            "--key-anchor-only and rejected otherwise"
        ),
    )
    parser.add_argument(
        "--vision-anchor",
        help=(
            "Optional teacher checkpoint whose complete model.visual.* vision "
            "tower is copied; must be one of --teachers. Vision tensors are "
            "merged normally when omitted"
        ),
    )
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Scale for TA/TIES tensors, including routers and the vision "
            "tower unless --vision-anchor is used (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--ties-density",
        type=float,
        default=0.2,
        help="TIES retained task-vector density (default: 0.2)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output shard size (default: 5GB)",
    )
    parser.add_argument(
        "--processor-source",
        help="Tokenizer/processor source; defaults to --base",
    )
    parser.add_argument(
        "--save-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.ties_density <= 1.0:
        raise ValueError("--ties-density must be in (0, 1]")

    base_dir = Path(args.base)
    teacher_dirs = [Path(path) for path in args.teachers]
    anchor_dir = Path(args.experts_anchor)
    anchor_index = _resolve_teacher_index(
        teacher_dirs, anchor_dir, "--experts-anchor"
    )
    key_experts_csv = (
        Path(args.key_experts_csv) if args.key_experts_csv is not None else None
    )
    if args.key_anchor_only and key_experts_csv is None:
        raise ValueError("--key-experts-csv is required with --key-anchor-only")
    if not args.key_anchor_only and key_experts_csv is not None:
        raise ValueError("--key-experts-csv requires --key-anchor-only")
    vision_anchor_dir = Path(args.vision_anchor) if args.vision_anchor else None
    vision_anchor_index = (
        _resolve_teacher_index(
            teacher_dirs, vision_anchor_dir, "--vision-anchor"
        )
        if vision_anchor_dir is not None
        else None
    )
    output_dir = Path(args.output)
    processor_source = (
        Path(args.processor_source) if args.processor_source else base_dir
    )
    teacher_names = args.teacher_names or [path.name for path in teacher_dirs]
    if len(teacher_names) != len(teacher_dirs):
        raise ValueError(
            f"Expected {len(teacher_dirs)} --teacher-names, "
            f"got {len(teacher_names)}"
        )

    resolved_output = output_dir.resolve()
    forbidden_outputs = {base_dir.resolve(), *(path.resolve() for path in teacher_dirs)}
    if resolved_output in forbidden_outputs:
        raise ValueError("Output directory must differ from base and teacher directories")
    if args.save_processor and not processor_source.is_dir():
        raise FileNotFoundError(
            f"Processor source directory not found: {processor_source}"
        )
    _prepare_output_dir(output_dir, args.overwrite)

    base = SafetensorCheckpoint(base_dir)
    teachers: list[SafetensorCheckpoint] = []
    try:
        teachers = [SafetensorCheckpoint(path) for path in teacher_dirs]
        _validate_checkpoint_compatibility(base, teachers)
        layouts = discover_expert_layers(base)
        expert_keys = _expert_keys(layouts)
        key_anchor_slots = (
            _read_key_expert_slots(key_experts_csv, layouts)
            if key_experts_csv is not None
            else None
        )
        vision_keys = _vision_keys(base)
        anchor = teachers[anchor_index]
        vision_anchor = (
            teachers[vision_anchor_index]
            if vision_anchor_index is not None
            else None
        )

        if key_anchor_slots is None:
            print(
                f"Anchoring {len(expert_keys)} packed expert tensors across "
                f"{len(layouts)} MoE layers from {teacher_dirs[anchor_index]}"
            )
        else:
            total_expert_slots = sum(layout.num_experts for layout in layouts)
            print(
                f"Key-anchor-only: copying {len(key_anchor_slots)}/"
                f"{total_expert_slots} expert slots from "
                f"{teacher_dirs[anchor_index]}; all other expert slots use base"
            )
        if vision_anchor is None:
            print(
                f"Merging all remaining tensors, including vision and routers, "
                f"with {args.method.upper()}: scale={args.scale}"
            )
        else:
            print(
                f"Anchoring {len(vision_keys)} vision-tower tensors from "
                f"{teacher_dirs[vision_anchor_index]}"
            )
            print(
                f"Merging all remaining tensors, including routers, with "
                f"{args.method.upper()}: scale={args.scale}"
            )
        if args.method == "ties":
            print(f"TIES density={args.ties_density}")

        merged_state = AnchorExpertsMergedCheckpoint(
            base,
            teachers,
            anchor,
            layouts,
            key_anchor_slots,
            vision_anchor,
            vision_keys,
            method=args.method,
            scale=args.scale,
            ties_density=args.ties_density,
        )
        _save_summary(
            output_dir,
            method=args.method,
            scale=args.scale,
            ties_density=args.ties_density,
            base_dir=base_dir,
            teacher_dirs=teacher_dirs,
            teacher_names=teacher_names,
            anchor_dir=anchor_dir,
            key_anchor_only=args.key_anchor_only,
            key_experts_csv=key_experts_csv,
            key_anchor_slots=key_anchor_slots,
            vision_anchor_dir=vision_anchor_dir,
            layouts=layouts,
        )
        save_selected_checkpoint(
            merged_state,
            expert_dir=base_dir,
            processor_source=processor_source,
            output_dir=output_dir,
            max_shard_size=args.max_shard_size,
            save_processor=args.save_processor,
            trust_remote_code=args.trust_remote_code,
        )
    finally:
        for teacher in teachers:
            teacher.close()
        base.close()

    print(f"Done. Anchor-experts ablation checkpoint -> {output_dir}")


if __name__ == "__main__":
    main()
