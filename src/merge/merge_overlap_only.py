#!/usr/bin/env python3
"""TA/TIES merging restricted to overlapping Qwen3-VL MoE experts.

Expert slots selected by zero, one, or multiple tasks use the base expert,
copy the sole task expert, or merge only the selecting tasks, respectively.
All other tensors are merged across every task checkpoint.

Usage:
    python src/merge/merge_overlap_only.py ties --base BASE \
        --teachers MED PUZZLE NAV WEMATH --task-names medvqa puzzle navigation wemath2 \
        --importance-dir SCORES --output OUTPUT

``SCORES`` must contain ``<task>_functional_drift_scores.csv`` for each task.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch

try:
    from .identify_important_experts import (
        ExpertLayerLayout,
        SafetensorCheckpoint,
        _prepare_output_dir,
        discover_expert_layers,
        save_selected_checkpoint,
    )
    from .merge import _is_router_weight_key, _merge_ta_tensor, _merge_ties_tensor
except ImportError:
    # Support direct execution: python src/merge/merge_overlap_only.py ...
    from identify_important_experts import (
        ExpertLayerLayout,
        SafetensorCheckpoint,
        _prepare_output_dir,
        discover_expert_layers,
        save_selected_checkpoint,
    )
    from merge import _is_router_weight_key, _merge_ta_tensor, _merge_ties_tensor


Selection = dict[int, set[int]]
ScoreMap = dict[tuple[int, int], float]


def _read_importance_csv(
    path: Path,
    *,
    selected_action: str,
) -> tuple[Selection, ScoreMap, set[tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Importance CSV not found: {path}")

    selected: Selection = {}
    scores: ScoreMap = {}
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer_index", "expert_id", "action"}
        fieldnames = set(reader.fieldnames or [])
        missing_columns = required - fieldnames
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                layer_index = int(row["layer_index"])
                expert_id = int(row["expert_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid layer_index/expert_id at {path}:{row_number}"
                ) from exc

            pair = (layer_index, expert_id)
            if pair in seen:
                raise ValueError(
                    f"Duplicate layer/expert row at {path}:{row_number}: {pair}"
                )
            seen.add(pair)

            score_text = row.get("score", "")
            if score_text not in (None, ""):
                try:
                    scores[pair] = float(score_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid score at {path}:{row_number}: {score_text!r}"
                    ) from exc

            if (row.get("action") or "").strip() == selected_action:
                selected.setdefault(layer_index, set()).add(expert_id)

    if not seen:
        raise ValueError(f"Importance CSV is empty: {path}")
    return selected, scores, seen


def _validate_csv_coverage(
    path: Path,
    seen: set[tuple[int, int]],
    layouts: list[ExpertLayerLayout],
) -> None:
    expected = {
        (layout.layer_index, expert_id)
        for layout in layouts
        for expert_id in range(layout.num_experts)
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Importance CSV coverage mismatch for {path}: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def _validate_checkpoint_compatibility(
    base: SafetensorCheckpoint,
    teachers: list[SafetensorCheckpoint],
) -> None:
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
            if teacher.get_shape(key) != base.get_shape(key):
                raise ValueError(
                    f"Shape mismatch at {key}: base={base.get_shape(key)}, "
                    f"{teacher.model_dir}={teacher.get_shape(key)}"
                )


def _resolve_task_names(
    teacher_dirs: list[Path],
    task_names_arg: list[str] | None,
) -> list[str]:
    if task_names_arg is None:
        names = [path.name for path in teacher_dirs]
    else:
        if len(task_names_arg) != len(teacher_dirs):
            raise ValueError(
                f"Expected {len(teacher_dirs)} --task-names, "
                f"got {len(task_names_arg)}"
            )
        names = task_names_arg
    if len(set(names)) != len(names):
        raise ValueError(f"Task names must be unique, got {names}")
    return names


def _resolve_importance_paths(
    *,
    teacher_dirs: list[Path],
    task_names: list[str],
    importance_dir: str,
) -> list[Path]:
    directory = Path(importance_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Importance directory not found: {directory}")
    paths = [
        directory / f"{task_name}_functional_drift_scores.csv"
        for task_name in task_names
    ]

    for task_name, teacher_dir, csv_path in zip(task_names, teacher_dirs, paths):
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Importance CSV for task {task_name!r} not found: {csv_path} "
                f"(teacher={teacher_dir})"
            )
    return paths


def _build_selected_teachers(
    layouts: list[ExpertLayerLayout],
    selections: list[Selection],
) -> dict[tuple[int, int], tuple[int, ...]]:
    selected_teachers: dict[tuple[int, int], tuple[int, ...]] = {}
    for layout in layouts:
        for expert_id in range(layout.num_experts):
            selected_teachers[(layout.layer_index, expert_id)] = tuple(
                task_index
                for task_index, selection in enumerate(selections)
                if expert_id in selection.get(layout.layer_index, set())
            )
    return selected_teachers


class OverlapOnlyMergedCheckpoint:
    """Compute the overlap-only merged checkpoint one output tensor at a time."""

    def __init__(
        self,
        base: SafetensorCheckpoint,
        teachers: list[SafetensorCheckpoint],
        layouts: list[ExpertLayerLayout],
        selected_teachers: dict[tuple[int, int], tuple[int, ...]],
        *,
        method: str,
        scale: float,
        ties_density: float,
        center_router: bool,
    ):
        self.base = base
        self.teachers = teachers
        self.selected_teachers = selected_teachers
        self.method = method
        self.scale = scale
        self.ties_density = ties_density
        self.center_router = center_router
        self.expert_key_layout: dict[str, ExpertLayerLayout] = {}
        for layout in layouts:
            self.expert_key_layout[layout.gate_up_key] = layout
            self.expert_key_layout[layout.down_key] = layout

    def keys(self) -> list[str]:
        return self.base.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.base.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.base.get_dtype(key)

    def _teacher_full_tensors(
        self,
        key: str,
        task_indices: Iterable[int],
    ) -> Iterable[torch.Tensor]:
        for task_index in task_indices:
            yield self.teachers[task_index].get_tensor(key)

    def _teacher_expert_slices(
        self,
        key: str,
        expert_id: int,
        task_indices: Iterable[int],
    ) -> Iterable[torch.Tensor]:
        for task_index in task_indices:
            yield self.teachers[task_index].get_slice(
                key, slice(expert_id, expert_id + 1)
            )[0]

    def _teacher_gate_up_component_slices(
        self,
        key: str,
        expert_id: int,
        task_indices: Iterable[int],
        component_index: int,
    ) -> Iterable[torch.Tensor]:
        """Yield one unpacked gate/up projection for one expert at a time."""
        for task_index in task_indices:
            packed = self.teachers[task_index].get_slice(
                key, slice(expert_id, expert_id + 1)
            )[0]
            yield packed.chunk(2, dim=-1)[component_index]

    def _merge_selected_tensor(
        self,
        base_tensor: torch.Tensor,
        teacher_tensor_factory,
    ) -> torch.Tensor:
        if self.method == "ta":
            return _merge_ta_tensor(
                base_tensor,
                teacher_tensor_factory(),
                self.scale,
            )
        return _merge_ties_tensor(
            base_tensor,
            teacher_tensor_factory,
            self.ties_density,
            self.scale,
        )

    def _merge_full_tensor(self, key: str) -> torch.Tensor:
        base_tensor = self.base.get_tensor(key)
        all_tasks = tuple(range(len(self.teachers)))
        if self.method == "ta":
            return _merge_ta_tensor(
                base_tensor,
                self._teacher_full_tensors(key, all_tasks),
                self.scale,
                center_expert_rows=(
                    self.center_router and _is_router_weight_key(key)
                ),
            )
        return _merge_ties_tensor(
            base_tensor,
            lambda: self._teacher_full_tensors(key, all_tasks),
            self.ties_density,
            self.scale,
        )

    def _merge_overlap_expert(
        self,
        key: str,
        layout: ExpertLayerLayout,
        expert_id: int,
        task_indices: tuple[int, ...],
    ) -> torch.Tensor:
        base_slice = self.base.get_slice(key, slice(expert_id, expert_id + 1))[0]
        if key == layout.gate_up_key:
            # Qwen stores [hidden, gate_intermediate || up_intermediate].
            # Merge the logical projections independently so that TIES trims
            # each projection with its own magnitude threshold.
            base_gate, base_up = base_slice.chunk(2, dim=-1)
            merged_gate = self._merge_selected_tensor(
                base_gate,
                lambda: self._teacher_gate_up_component_slices(
                    key, expert_id, task_indices, 0
                ),
            )
            merged_up = self._merge_selected_tensor(
                base_up,
                lambda: self._teacher_gate_up_component_slices(
                    key, expert_id, task_indices, 1
                ),
            )
            return torch.cat((merged_gate, merged_up), dim=-1)

        return self._merge_selected_tensor(
            base_slice,
            lambda: self._teacher_expert_slices(key, expert_id, task_indices),
        )

    def _merge_packed_expert_tensor(
        self,
        key: str,
        layout: ExpertLayerLayout,
    ) -> torch.Tensor:
        output = self.base.get_tensor(key).clone()
        for expert_id in range(layout.num_experts):
            task_indices = self.selected_teachers[(layout.layer_index, expert_id)]
            if not task_indices:
                continue
            if len(task_indices) == 1:
                # Private task-critical expert: protect it exactly, without TA/TIES.
                source = self.teachers[task_indices[0]].get_slice(
                    key, slice(expert_id, expert_id + 1)
                )[0]
                output[expert_id].copy_(source)
            else:
                merged = self._merge_overlap_expert(
                    key, layout, expert_id, task_indices
                )
                output[expert_id].copy_(merged)
        return output

    def get_tensor(self, key: str) -> torch.Tensor:
        layout = self.expert_key_layout.get(key)
        if layout is not None:
            return self._merge_packed_expert_tensor(key, layout)
        return self._merge_full_tensor(key)


def _save_overlap_report(
    output_dir: Path,
    *,
    layouts: list[ExpertLayerLayout],
    selected_teachers: dict[tuple[int, int], tuple[int, ...]],
    scores: list[ScoreMap],
    task_names: list[str],
    teacher_dirs: list[Path],
    importance_paths: list[Path],
    method: str,
    scale: float,
    ties_density: float,
    selected_action: str,
) -> None:
    counts = Counter(len(task_indices) for task_indices in selected_teachers.values())
    summary = {
        "method": method,
        "scale": scale,
        "ties_density": ties_density if method == "ties" else None,
        "selected_action": selected_action,
        "semantics": {
            "selected_by_zero": "base",
            "selected_by_one": "copy_private_task_expert",
            "selected_by_multiple": f"{method}_selected_tasks_only",
            "overlap_gate_up_proj": "split_gate_and_up_merge_independently_then_concat",
            "non_expert_tensors": f"{method}_all_tasks",
        },
        "tasks": [
            {
                "name": name,
                "teacher": str(teacher.resolve()),
                "importance_csv": str(csv_path.resolve()),
            }
            for name, teacher, csv_path in zip(
                task_names, teacher_dirs, importance_paths
            )
        ],
        "expert_slot_counts": {
            "base": counts.get(0, 0),
            "private": counts.get(1, 0),
            "overlap": sum(
                count for num_tasks, count in counts.items() if num_tasks >= 2
            ),
        },
    }
    with (output_dir / "overlap_merge_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with (output_dir / "overlap_merge_experts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_index",
                "layer_path",
                "expert_id",
                "num_selected_tasks",
                "selected_tasks",
                "selected_scores",
                "operation",
            ]
        )
        for layout in layouts:
            for expert_id in range(layout.num_experts):
                task_indices = selected_teachers[(layout.layer_index, expert_id)]
                if not task_indices:
                    operation = "base"
                elif len(task_indices) == 1:
                    operation = "copy_private"
                else:
                    operation = f"{method}_overlap"
                writer.writerow(
                    [
                        layout.layer_index,
                        layout.label,
                        expert_id,
                        len(task_indices),
                        ";".join(task_names[index] for index in task_indices),
                        ";".join(
                            f"{scores[index].get((layout.layer_index, expert_id), float('nan')):.17g}"
                            for index in task_indices
                        ),
                        operation,
                    ]
                )
    print("Overlap reports saved: overlap_merge_summary.json, overlap_merge_experts.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge only overlapping task-critical Qwen3-VL internal experts; "
            "merge all non-expert tensors normally."
        )
    )
    parser.add_argument("method", choices=["ta", "ties"])
    parser.add_argument("--base", required=True, help="Local base checkpoint")
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="Task checkpoint directories in task order",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        help=(
            "Optional names in teacher order. Required for convenient "
            "--importance-dir lookup when CSV stems differ from teacher directory names."
        ),
    )
    parser.add_argument(
        "--importance-dir",
        required=True,
        help=(
            "Directory containing one "
            "<task-name>_functional_drift_scores.csv per teacher"
        ),
    )
    parser.add_argument(
        "--selected-action",
        default="keep_task",
        help="CSV action value denoting a selected expert (default: keep_task)",
    )
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="TA/TIES task-vector scale (default: 1.0)",
    )
    parser.add_argument(
        "--ties-density",
        type=float,
        default=0.2,
        help="TIES retained task-vector density (default: 0.2)",
    )
    parser.add_argument(
        "--ta-center-router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For TA, center each full router task vector across expert rows",
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
    if args.method != "ta" and args.ta_center_router:
        raise ValueError("--ta-center-router can only be used with method 'ta'")
    if not 0.0 < args.ties_density <= 1.0:
        raise ValueError("--ties-density must be in (0, 1]")

    base_dir = Path(args.base)
    teacher_dirs = [Path(path) for path in args.teachers]
    output_dir = Path(args.output)
    processor_source = (
        Path(args.processor_source) if args.processor_source else base_dir
    )
    task_names = _resolve_task_names(teacher_dirs, args.task_names)
    importance_paths = _resolve_importance_paths(
        teacher_dirs=teacher_dirs,
        task_names=task_names,
        importance_dir=args.importance_dir,
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

    print("Task/importance mapping:")
    for task_name, teacher_dir, csv_path in zip(
        task_names, teacher_dirs, importance_paths
    ):
        print(f"  {task_name}: teacher={teacher_dir}, csv={csv_path}")

    base = SafetensorCheckpoint(base_dir)
    teachers: list[SafetensorCheckpoint] = []
    try:
        teachers = [SafetensorCheckpoint(path) for path in teacher_dirs]
        _validate_checkpoint_compatibility(base, teachers)
        layouts = discover_expert_layers(base)

        selections: list[Selection] = []
        scores: list[ScoreMap] = []
        for csv_path in importance_paths:
            selection, score_map, seen = _read_importance_csv(
                csv_path,
                selected_action=args.selected_action,
            )
            _validate_csv_coverage(csv_path, seen, layouts)
            selections.append(selection)
            scores.append(score_map)

        selected_teachers = _build_selected_teachers(layouts, selections)
        count_by_tasks = Counter(
            len(indices) for indices in selected_teachers.values()
        )
        total_slots = sum(count_by_tasks.values())
        private_slots = count_by_tasks.get(1, 0)
        overlap_slots = sum(
            count for num_tasks, count in count_by_tasks.items() if num_tasks >= 2
        )
        print(
            f"Expert slots: total={total_slots}, base={count_by_tasks.get(0, 0)}, "
            f"private={private_slots}, overlap={overlap_slots}"
        )
        for num_tasks in sorted(count_by_tasks):
            if num_tasks >= 2:
                print(
                    f"  selected by {num_tasks} tasks: "
                    f"{count_by_tasks[num_tasks]} expert slot(s)"
                )

        merged_state = OverlapOnlyMergedCheckpoint(
            base,
            teachers,
            layouts,
            selected_teachers,
            method=args.method,
            scale=args.scale,
            ties_density=args.ties_density,
            center_router=args.ta_center_router,
        )
        _save_overlap_report(
            output_dir,
            layouts=layouts,
            selected_teachers=selected_teachers,
            scores=scores,
            task_names=task_names,
            teacher_dirs=teacher_dirs,
            importance_paths=importance_paths,
            method=args.method,
            scale=args.scale,
            ties_density=args.ties_density,
            selected_action=args.selected_action,
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

    print(f"Done. Overlap-only merged checkpoint -> {output_dir}")


if __name__ == "__main__":
    main()
