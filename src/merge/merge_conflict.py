#!/usr/bin/env python3
"""Repair MoE experts whose task behavior is lost after model merging.

The script uses a previously merged checkpoint as the context model and
scores the target task's experts in every layer with shared Rademacher hidden-state probes:

    I_task = mean_r ||f_task(z_r) - f_base(z_r)||_2^2
                    / (||f_base(z_r)||_2^2 + eps)

    I_lost = mean_r ||f_task(z_r) - f_context(z_r)||_2^2
                    / (||f_task(z_r)||_2^2 + eps)

    I_repair = sqrt(I_task * I_lost)

Only ``--target-task`` is scored.  It must identify one entry in
``--task-names`` and therefore one teacher in ``--teachers``.  The highest
scoring experts are selected independently in every layer.  Selected packed
``gate_up_proj`` and ``down_proj`` expert slices are copied completely from
the target teacher; all other tensors remain exactly those of the context
checkpoint.

Example:

    python src/merge/merge_conflict.py \
        --base /models/base \
        --context-model /models/TA-alpha0.5 \
        --teachers /models/med /models/puzzle /models/nav /models/math \
        --task-names medvqa puzzle navigation wemath2 \
        --target-task puzzle \
        --repair-fraction 0.125 \
        --selection-score geometric \
        --output /models/TA-alpha0.5-conflict-repair

This is a data-free *expert-local* post-merge score.  The context checkpoint's
expert weights affect I_lost, but attention, routing frequency, and real hidden
state distributions are not evaluated because no model inputs are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


# Direct execution adds src/merge rather than the repository root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.merge.identify_important_experts import (  # noqa: E402
    ExpertLayerLayout,
    SafetensorCheckpoint,
    _prepare_output_dir,
    _resolve_compute_dtype,
    _resolve_device,
    _swiglu_expert_forward,
    discover_expert_layers,
    make_rademacher_probes,
    save_selected_checkpoint,
)

Slot = tuple[int, int]


@dataclass(frozen=True)
class RepairScore:
    """The three relative functional scores for one task/expert slot."""

    i_task: float
    i_lost: float
    geometric_mean: float


TaskScoreMap = dict[Slot, RepairScore]


def _validate_checkpoint_compatibility(
    reference: SafetensorCheckpoint,
    checkpoints: list[SafetensorCheckpoint],
) -> None:
    """Require identical tensor keys and shapes across all checkpoints."""

    reference_keys = set(reference.keys())
    for checkpoint in checkpoints:
        checkpoint_keys = set(checkpoint.keys())
        if checkpoint_keys != reference_keys:
            missing = sorted(reference_keys - checkpoint_keys)
            extra = sorted(checkpoint_keys - reference_keys)
            raise ValueError(
                f"Checkpoint key mismatch for {checkpoint.model_dir}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for key in reference.keys():
            expected = reference.get_shape(key)
            actual = checkpoint.get_shape(key)
            if actual != expected:
                raise ValueError(
                    f"Shape mismatch at {key}: reference={expected}, "
                    f"{checkpoint.model_dir}={actual}"
                )


def _resolve_task_names(
    teacher_dirs: list[Path],
    task_names_arg: list[str] | None,
) -> list[str]:
    names = task_names_arg or [path.name for path in teacher_dirs]
    if len(names) != len(teacher_dirs):
        raise ValueError(
            f"Expected {len(teacher_dirs)} --task-names, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Task names must be unique, got {names}")
    return names


@torch.inference_mode()
def score_conflict_repairs(
    base: SafetensorCheckpoint,
    context: SafetensorCheckpoint,
    teacher: SafetensorCheckpoint,
    layouts: list[ExpertLayerLayout],
    probes_cpu: torch.Tensor,
    *,
    target_task: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    expert_batch_size: int,
    eps: float,
) -> TaskScoreMap:
    """Score target-task drift and the portion lost in the context expert."""

    if expert_batch_size <= 0:
        raise ValueError("expert_batch_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    probes = probes_cpu.to(device=device, dtype=compute_dtype)
    scores: TaskScoreMap = {}

    for layer_number, layout in enumerate(layouts, start=1):
        print(
            f"Scoring layer {layer_number}/{len(layouts)}: {layout.label} "
            f"({layout.num_experts} packed experts)"
        )
        layer_scores: list[RepairScore] = []

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

            context_gate, context_up, context_down = layout.load_expert_batch(
                context, start, end
            )
            context_output = _swiglu_expert_forward(
                probes,
                context_gate,
                context_up,
                context_down,
                device=device,
                compute_dtype=compute_dtype,
            )
            del context_gate, context_up, context_down

            task_gate, task_up, task_down = layout.load_expert_batch(
                teacher, start, end
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

            task_numerator = (task_output - base_output).square().sum(dim=-1)
            lost_numerator = (task_output - context_output).square().sum(dim=-1)
            base_denominator = base_output.square().sum(dim=-1).add_(eps)
            task_denominator = task_output.square().sum(dim=-1).add_(eps)
            i_task = task_numerator.div(base_denominator).mean(dim=-1)
            i_lost = lost_numerator.div(task_denominator).mean(dim=-1)
            geometric = torch.sqrt(
                i_task.clamp_min(0.0) * i_lost.clamp_min(0.0)
            )

            if not (
                torch.isfinite(i_task).all()
                and torch.isfinite(i_lost).all()
                and torch.isfinite(geometric).all()
            ):
                raise FloatingPointError(
                    f"Non-finite repair score for target task {target_task!r} "
                    f"at {layout.label}, experts [{start}:{end}]. "
                    "Try --compute-dtype float32."
                )

            task_values = i_task.cpu().tolist()
            lost_values = i_lost.cpu().tolist()
            geometric_values = geometric.cpu().tolist()
            for offset, expert_id in enumerate(range(start, end)):
                score = RepairScore(
                    i_task=float(task_values[offset]),
                    i_lost=float(lost_values[offset]),
                    geometric_mean=float(geometric_values[offset]),
                )
                scores[(layout.layer_index, expert_id)] = score
                layer_scores.append(score)

            del (
                base_output,
                context_output,
                task_output,
                task_numerator,
                lost_numerator,
                base_denominator,
                task_denominator,
                i_task,
                i_lost,
                geometric,
            )

        repair_values = [score.geometric_mean for score in layer_scores]
        print(
            f"  {target_task}: repair score min/median/max: "
            f"{min(repair_values):.6g} / "
            f"{statistics.median(repair_values):.6g} / "
            f"{max(repair_values):.6g}"
        )

    return scores


def select_repairs(
    layouts: list[ExpertLayerLayout],
    scores: TaskScoreMap,
    *,
    target_task: str,
    repair_fraction: float,
    selection_score: str,
) -> frozenset[Slot]:
    """Select the target task's top requested scores in every layer."""

    if not 0.0 <= repair_fraction <= 1.0:
        raise ValueError(
            f"repair_fraction must be in [0, 1], got {repair_fraction}"
        )
    score_attributes = {
        "geometric": "geometric_mean",
        "i_task": "i_task",
        "i_lost": "i_lost",
    }
    try:
        score_attribute = score_attributes[selection_score]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported selection score: {selection_score!r}"
        ) from exc

    selected: set[Slot] = set()
    print(
        f"Selecting repairs for target task {target_task!r} "
        f"using {selection_score}:"
    )
    for layout in layouts:
        num_repaired = math.floor(layout.num_experts * repair_fraction)
        ranked = sorted(
            range(layout.num_experts),
            key=lambda expert_id: (
                getattr(
                    scores[(layout.layer_index, expert_id)], score_attribute
                ),
                expert_id,
            ),
        )
        repaired = ranked[-num_repaired:] if num_repaired else []
        selected.update(
            (layout.layer_index, expert_id) for expert_id in repaired
        )
        if repaired:
            threshold = min(
                getattr(
                    scores[(layout.layer_index, expert_id)], score_attribute
                )
                for expert_id in repaired
            )
            print(
                f"  {layout.label}: repairing {len(repaired)}/"
                f"{layout.num_experts} experts "
                f"({selection_score} >= {threshold:.6g})"
            )
        else:
            print(f"  {layout.label}: repairing 0/{layout.num_experts} experts")

    return frozenset(selected)


class ConflictRepairCheckpoint:
    """Copy context and replace selected expert slices from one teacher."""

    def __init__(
        self,
        context: SafetensorCheckpoint,
        teacher: SafetensorCheckpoint,
        layouts: list[ExpertLayerLayout],
        repair_slots: frozenset[Slot],
    ) -> None:
        self.context = context
        self.teacher = teacher
        self.repair_slots = repair_slots
        self.expert_key_layout = {
            key: layout
            for layout in layouts
            for key in (layout.gate_up_key, layout.down_key)
        }

    def keys(self) -> list[str]:
        return self.context.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.context.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.context.get_dtype(key)

    def _repair_packed_expert_tensor(
        self,
        key: str,
        layout: ExpertLayerLayout,
    ) -> torch.Tensor:
        output = self.context.get_tensor(key).clone()
        expert_ids = sorted(
            expert_id
            for layer_index, expert_id in self.repair_slots
            if layer_index == layout.layer_index
        )
        for expert_id in expert_ids:
            # Copy the complete packed gate_up or down expert slice exactly.
            source = self.teacher.get_slice(
                key, slice(expert_id, expert_id + 1)
            )[0]
            output[expert_id].copy_(source)
        return output

    def get_tensor(self, key: str) -> torch.Tensor:
        layout = self.expert_key_layout.get(key)
        if layout is None:
            return self.context.get_tensor(key)
        return self._repair_packed_expert_tensor(key, layout)


def _save_reports(
    output_dir: Path,
    *,
    layouts: list[ExpertLayerLayout],
    scores: TaskScoreMap,
    repair_slots: frozenset[Slot],
    target_task: str,
    base_dir: Path,
    context_dir: Path,
    target_teacher_dir: Path,
    repair_fraction: float,
    selection_score: str,
    num_probes: int,
    seed: int,
    eps: float,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    total_slots = sum(layout.num_experts for layout in layouts)
    report = {
        "method": "data_free_single_task_post_merge_conflict_repair",
        "base_model": str(base_dir.resolve()),
        "context_model": str(context_dir.resolve()),
        "target_task": target_task,
        "target_teacher": str(target_teacher_dir.resolve()),
        "formulas": {
            "I_task": (
                "mean_r(||f_task(z_r)-f_base(z_r)||_2^2 / "
                "(||f_base(z_r)||_2^2+eps))"
            ),
            "I_lost": (
                "mean_r(||f_task(z_r)-f_context(z_r)||_2^2 / "
                "(||f_task(z_r)||_2^2+eps))"
            ),
            "repair_score": "sqrt(I_task * I_lost)",
        },
        "probe_distribution": "Rademacher({-1,+1})",
        "num_probes": num_probes,
        "seed": seed,
        "eps": eps,
        "device": str(device),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "repair_fraction": repair_fraction,
        "selection_score": selection_score,
        "selection_rule": (
            f"highest {selection_score} scores within each layer for target task"
        ),
        "expert_slot_counts": {
            "unchanged_context": total_slots - len(repair_slots),
            "repaired_from_target_teacher": len(repair_slots),
        },
        "semantics": {
            "selected_expert": "copy_complete_target_teacher_expert",
            "unselected_expert": "copy_context",
            "non_expert_tensors": "copy_context",
        },
        "limitation": (
            "expert-local synthetic probes; routing frequency, attention, vision, "
            "and real hidden-state distributions are not measured"
        ),
    }
    with (output_dir / "conflict_repair_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with (output_dir / "conflict_repair_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_name",
                "teacher",
                "layer_index",
                "layer_path",
                "expert_id",
                "I_task",
                "I_lost",
                "geometric_mean",
                "action",
            ]
        )
        for layout in layouts:
            for expert_id in range(layout.num_experts):
                slot = (layout.layer_index, expert_id)
                score = scores[slot]
                writer.writerow(
                    [
                        target_task,
                        str(target_teacher_dir.resolve()),
                        layout.layer_index,
                        layout.label,
                        expert_id,
                        f"{score.i_task:.17g}",
                        f"{score.i_lost:.17g}",
                        f"{score.geometric_mean:.17g}",
                        "repair" if slot in repair_slots else "keep_context",
                    ]
                )

    print(
        "Conflict-repair reports saved: conflict_repair_summary.json, "
        "conflict_repair_scores.csv"
    )



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use data-free post-merge functional loss to repair one target "
            "task's Qwen3-VL packed MoE experts."
        )
    )
    parser.add_argument("--base", required=True, help="Local base checkpoint")
    parser.add_argument(
        "--context-model",
        "--context",
        dest="context_model",
        required=True,
        help=(
            "Previously merged checkpoint to score and repair. All unselected "
            "experts and non-expert tensors are copied from this model"
        ),
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="Task checkpoint directories in task order",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        required=True,
        help="Unique task names in the same order as --teachers",
    )
    parser.add_argument(
        "--target-task",
        required=True,
        help="Single task to repair; must be one of --task-names",
    )
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument(
        "--repair-fraction",
        type=float,
        default=0.125,
        help=(
            "Highest selected-score fraction copied from the target teacher "
            "in every layer (default: 0.125)"
        ),
    )
    parser.add_argument(
        "--selection-score",
        choices=["geometric", "i_task", "i_lost"],
        default="geometric",
        help=(
            "Score used to rank experts for repair; all scores are always "
            "computed and reported (default: geometric)"
        ),
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
        "--expert-batch-size",
        type=int,
        default=8,
        help="Experts scored together on the compute device (default: 8)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Scoring device, e.g. auto, cuda:0, or cpu (default: auto)",
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
        help="Positive relative-score denominator stabilizer (default: 1e-8)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output shard size (default: 5GB)",
    )
    parser.add_argument(
        "--processor-source",
        help="Tokenizer/processor source; defaults to --context-model",
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
    if not 0.0 <= args.repair_fraction <= 1.0:
        raise ValueError("--repair-fraction must be in [0, 1]")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if args.expert_batch_size <= 0:
        raise ValueError("--expert-batch-size must be positive")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")

    base_dir = Path(args.base)
    context_dir = Path(args.context_model)
    teacher_dirs = [Path(path) for path in args.teachers]
    task_names = _resolve_task_names(teacher_dirs, args.task_names)
    if args.target_task not in task_names:
        raise ValueError(
            f"--target-task must be one of --task-names. "
            f"Got {args.target_task!r}; choices={task_names}"
        )
    target_index = task_names.index(args.target_task)
    target_teacher_dir = teacher_dirs[target_index]
    output_dir = Path(args.output)
    processor_source = (
        Path(args.processor_source) if args.processor_source else context_dir
    )

    resolved_output = output_dir.resolve()
    forbidden_outputs = {
        base_dir.resolve(),
        context_dir.resolve(),
        *(path.resolve() for path in teacher_dirs),
    }
    if resolved_output in forbidden_outputs:
        raise ValueError(
            "Output directory must differ from base, context, and teacher directories"
        )
    if args.save_processor and not processor_source.is_dir():
        raise FileNotFoundError(
            f"Processor source directory not found: {processor_source}"
        )
    _prepare_output_dir(output_dir, args.overwrite)

    device = _resolve_device(args.device)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype, device)
    print(f"Scoring device={device}, compute_dtype={compute_dtype}")
    print(f"Context model: {context_dir}")
    print(
        f"Target task: {args.target_task}; "
        f"teacher={target_teacher_dir}"
    )

    base = SafetensorCheckpoint(base_dir)
    context: SafetensorCheckpoint | None = None
    target_teacher: SafetensorCheckpoint | None = None
    try:
        context = SafetensorCheckpoint(context_dir)
        target_teacher = SafetensorCheckpoint(target_teacher_dir)
        _validate_checkpoint_compatibility(base, [context, target_teacher])
        layouts = discover_expert_layers(base)
        hidden_sizes = {layout.hidden_size for layout in layouts}
        if len(hidden_sizes) != 1:
            raise ValueError(
                "All MoE layers must share one hidden size for shared probes, "
                f"got {sorted(hidden_sizes)}"
            )
        hidden_size = next(iter(hidden_sizes))
        print(
            f"Discovered {len(layouts)} MoE layers; hidden_size={hidden_size}; "
            f"shared probes R={args.num_probes}, seed={args.seed}"
        )

        probes = make_rademacher_probes(
            args.num_probes, hidden_size, args.seed
        )
        scores = score_conflict_repairs(
            base,
            context,
            target_teacher,
            layouts,
            probes,
            target_task=args.target_task,
            device=device,
            compute_dtype=compute_dtype,
            expert_batch_size=args.expert_batch_size,
            eps=args.eps,
        )
        repair_slots = select_repairs(
            layouts,
            scores,
            target_task=args.target_task,
            repair_fraction=args.repair_fraction,
            selection_score=args.selection_score,
        )

        total_slots = sum(layout.num_experts for layout in layouts)
        print(
            f"Repair slots for {args.target_task}: "
            f"selected={len(repair_slots)}, "
            f"unchanged={total_slots - len(repair_slots)}"
        )
        print(
            "All non-expert tensors and unselected expert slots are copied "
            "unchanged from the context model."
        )

        repaired_state = ConflictRepairCheckpoint(
            context,
            target_teacher,
            layouts,
            repair_slots,
        )
        _save_reports(
            output_dir,
            layouts=layouts,
            scores=scores,
            repair_slots=repair_slots,
            target_task=args.target_task,
            base_dir=base_dir,
            context_dir=context_dir,
            target_teacher_dir=target_teacher_dir,
            repair_fraction=args.repair_fraction,
            selection_score=args.selection_score,
            num_probes=args.num_probes,
            seed=args.seed,
            eps=args.eps,
            device=device,
            compute_dtype=compute_dtype,
        )
        save_selected_checkpoint(
            repaired_state,
            expert_dir=context_dir,
            processor_source=processor_source,
            output_dir=output_dir,
            max_shard_size=args.max_shard_size,
            save_processor=args.save_processor,
            trust_remote_code=args.trust_remote_code,
        )
    finally:
        if target_teacher is not None:
            target_teacher.close()
        if context is not None:
            context.close()
        base.close()

    print(
        f"Done. Conflict-repaired checkpoint for {args.target_task} "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
