#!/usr/bin/env python3
"""Repair shared MoE experts with connection-salient channel updates.

Expert selection and directional coefficients use the original context and
complete teacher outputs, exactly as in merge_conflict.py. Unique selections
copy the full teacher expert. For a shared expert e, only selecting tasks S_e
contribute:

    theta'_e = theta_context,e + sum_t a_t,e * M_t,e * (theta_t,e - theta_base,e)

Each task independently retains floor(intermediate_size * channel_keep_fraction)
channels. A channel mask couples both gate/up columns and the corresponding
down row in the input-major packed checkpoint. There is no rescaling after
masking and no cross-task agreement modulation or iterative pruning.

The expert-local connection proxy is R = sum_j g_j * u_j * d_j, where g/u/d
are teacher L1 connection strengths. Channel saliency is sum_p |delta_p dR/dp|,
evaluated at the teacher, using the zero subgradient of abs at zero. This is a
structural proxy, not an estimate of task loss or real routing importance.
Saliency and task-vector differences are computed in FP32; synthetic forward
precision remains controlled by --compute-dtype. Repair accumulation is FP32
with one final cast to context storage precision.

Example (the channel fraction is an explicit experimental choice):

    python src/merge/merge_conflict_by_connection.py \
        --base /models/base --context-model /models/TA-alpha0.5 \
        --teachers /models/med /models/puzzle /models/nav /models/math \
        --task-names medvqa puzzle navigation wemath2 \
        --selection-score i_task --num-probes 32 --repair-fraction 0.125 \
        --channel-keep-fraction 0.5 --output /models/connection-repair
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.merge.identify_important_experts import (  # noqa: E402
    ExpertLayerLayout,
    SafetensorCheckpoint,
    _prepare_output_dir,
    _resolve_compute_dtype,
    _resolve_device,
    discover_expert_layers,
    make_rademacher_probes,
    save_selected_checkpoint,
)
from src.merge.merge_conflict import (  # noqa: E402
    ConflictRepairCheckpoint,
    Slot,
    TaskScoreMap,
    _resolve_target_tasks,
    _resolve_task_names,
    _summarize_values,
    _validate_checkpoint_compatibility,
    resolve_repair_conflicts,
    score_conflict_repairs,
    select_repairs,
)


@dataclass(frozen=True)
class ChannelSelection:
    """One task's CPU boolean channel mask and its reporting metadata."""

    mask: torch.Tensor
    retained_ids: tuple[int, ...]
    threshold: float | None


ChannelSelections = dict[Slot, dict[str, ChannelSelection]]


@torch.inference_mode()
def connection_channel_scores(
    base_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return [experts, intermediate] saliency for input-major packed weights.

    gate/up: [experts, hidden, intermediate]; down: [experts, intermediate, hidden].
    q_W = sum(|teacher-base| * (teacher != 0)) uses abs'(0) = 0. The indicator
    is necessary to match the defining gradient when a teacher weight is zero.
    """
    if len(base_weights) != 3 or len(teacher_weights) != 3:
        raise ValueError("Expected gate, up, and down for base and teacher")
    strengths, updates = [], []
    gate, up, down = teacher_weights
    if (gate.ndim != 3 or up.shape != gate.shape
            or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])):
        raise ValueError("Invalid input-major gate/up/down shapes")
    for base, teacher, dim in zip(base_weights, teacher_weights, (1, 1, 2)):
        if base.shape != teacher.shape:
            raise ValueError("Base and teacher shapes must match")
        base, teacher = base.float(), teacher.float()
        if not torch.isfinite(base).all() or not torch.isfinite(teacher).all():
            raise FloatingPointError("Non-finite weights in connection scoring")
        strengths.append(teacher.abs().sum(dim=dim))
        updates.append(((teacher - base).abs() * (teacher != 0)).sum(dim=dim))
    g, u, d = strengths
    qg, qu, qd = updates
    scores = u * d * qg + g * d * qu + g * u * qd
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite connection channel scores")
    return scores


def select_connection_channels(
    scores: torch.Tensor, keep_fraction: float,
) -> ChannelSelection:
    """Select an exact top-k; equal scores prefer larger channel indices."""
    if not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
        raise ValueError("channel keep fraction must be finite and in [0, 1]")
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected a finite one-dimensional channel score tensor")
    values = scores.detach().float().cpu().tolist()
    count = math.floor(len(values) * keep_fraction)
    ranked = sorted(range(len(values)), key=lambda j: (values[j], j))
    retained = tuple(sorted(ranked[-count:])) if count else ()
    mask = torch.zeros(len(values), dtype=torch.bool)
    if retained:
        mask[list(retained)] = True
    threshold = min(values[j] for j in retained) if retained else None
    return ChannelSelection(mask, retained, threshold)


@torch.inference_mode()
def build_channel_selections(
    base: SafetensorCheckpoint,
    teachers: dict[str, SafetensorCheckpoint],
    layouts: list[ExpertLayerLayout],
    shared_tasks: dict[Slot, tuple[str, ...]],
    *,
    keep_fraction: float,
    device: torch.device,
    expert_batch_size: int,
) -> ChannelSelections:
    """Read only shared slots selected by each task; retain compact CPU masks."""
    if expert_batch_size <= 0:
        raise ValueError("expert_batch_size must be positive")
    if not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
        raise ValueError("channel keep fraction must be finite and in [0, 1]")
    selections: ChannelSelections = {slot: {} for slot in shared_tasks}
    for layout in layouts:
        for task, teacher in teachers.items():
            ids = sorted(e for (layer, e), tasks in shared_tasks.items()
                         if layer == layout.layer_index and task in tasks)
            if not ids:
                continue
            print(f"Connection scoring {layout.label}, task={task}: {len(ids)} shared experts")
            for offset in range(0, len(ids), expert_batch_size):
                batch_ids = ids[offset:offset + expert_batch_size]
                def load(checkpoint: SafetensorCheckpoint) -> tuple[torch.Tensor, ...]:
                    parts = [layout.load_expert_batch(checkpoint, e, e + 1)
                             for e in batch_ids]
                    return tuple(torch.cat([part[i] for part in parts], dim=0).to(
                        device=device, dtype=torch.float32) for i in range(3))
                base_weights, teacher_weights = load(base), load(teacher)
                scores = connection_channel_scores(base_weights, teacher_weights).cpu()
                for e, score in zip(batch_ids, scores):
                    selections[(layout.layer_index, e)][task] = select_connection_channels(
                        score, keep_fraction)
                del base_weights, teacher_weights, scores
    return selections


class ConnectionRepairCheckpoint(ConflictRepairCheckpoint):
    """Use the existing checkpoint interface with channel-masked shared updates."""

    def __init__(
        self,
        context: SafetensorCheckpoint,
        teachers: dict[str, SafetensorCheckpoint],
        layouts: list[ExpertLayerLayout],
        repair_sources: dict[Slot, str],
        *,
        base: SafetensorCheckpoint,
        shared_tasks: dict[Slot, tuple[str, ...]],
        shared_coefficients: dict[Slot, tuple[float, ...]],
        channel_selections: ChannelSelections,
    ) -> None:
        super().__init__(context, teachers, layouts, repair_sources, base=base,
                         shared_tasks=shared_tasks, shared_coefficients=shared_coefficients)
        if channel_selections.keys() != shared_tasks.keys():
            raise ValueError("Channel selections must exactly cover shared slots")
        layouts_by_layer = {layout.layer_index: layout for layout in layouts}
        for slot, tasks in shared_tasks.items():
            if channel_selections[slot].keys() != set(tasks):
                raise ValueError(f"Channel selections must cover selecting tasks at {slot}")
            size = layouts_by_layer[slot[0]].intermediate_size
            for selection in channel_selections[slot].values():
                mask = selection.mask
                if mask.dtype != torch.bool or mask.device.type != "cpu" or mask.shape != (size,):
                    raise ValueError(f"Expected a CPU boolean mask of length {size} at {slot}")
                if tuple(mask.nonzero().flatten().tolist()) != selection.retained_ids:
                    raise ValueError(f"Retained channel IDs disagree with mask at {slot}")
        self.channel_selections = channel_selections
        # Per-projection squared norms are replaced on repeated reads, not added.
        self._increment_squared_norms: dict[Slot, dict[str, tuple[float, float]]] = {}

    @torch.inference_mode()
    def _repair_packed_expert_tensor(self, key: str, layout: ExpertLayerLayout) -> torch.Tensor:
        output = self.context.get_tensor(key).clone()
        for (layer, e), task in sorted(self.repair_sources.items()):
            if layer == layout.layer_index:
                output[e].copy_(self.teachers[task].get_slice(key, slice(e, e + 1))[0])
        for slot, tasks in sorted(self.shared_tasks.items()):
            layer, e = slot
            if layer != layout.layer_index:
                continue
            norms = self._increment_squared_norms.setdefault(slot, {})
            coefficients = self.shared_coefficients[slot]
            if all(a == 0 for a in coefficients):
                norms[key] = (0.0, 0.0)
                continue
            base = self.base.get_slice(key, slice(e, e + 1))[0].float()
            repaired = output[e].float()
            full_increment = torch.zeros_like(base)
            masked_increment = torch.zeros_like(base)
            for task, a in zip(tasks, coefficients):
                if a == 0:
                    continue
                teacher = self.teachers[task].get_slice(key, slice(e, e + 1))[0].float()
                increment = a * (teacher - base)
                mask = self.channel_selections[slot][task].mask
                expanded = (torch.cat((mask, mask))[None, :] if key == layout.gate_up_key
                            else mask[:, None])
                masked = increment.masked_fill(~expanded, 0.0)
                full_increment.add_(increment)
                masked_increment.add_(masked)
                # Preserve exact context values for an empty mask; with all channels
                # kept this follows the original addition order, including FP32 rounding.
                if self.channel_selections[slot][task].retained_ids:
                    repaired = repaired + masked
            if not torch.isfinite(repaired).all():
                raise FloatingPointError(f"Non-finite repaired weights at {slot}, {key}")
            # The increments are FP32. FP64 reduction avoids overflow in diagnostics.
            before = full_increment.double().square().sum().item()
            after = masked_increment.double().square().sum().item()
            if not math.isfinite(before) or not math.isfinite(after):
                raise FloatingPointError(f"Non-finite repair increment norms at {slot}")
            norms[key] = (before, after)
            output[e].copy_(repaired.to(output.dtype))
        return output

    def increment_norm_report(self) -> list[dict[str, int | float]]:
        """Combine both projections after checkpoint saving has visited every key."""
        result = []
        for slot in sorted(self.shared_tasks):
            parts = self._increment_squared_norms.get(slot, {})
            if len(parts) != 2:
                raise RuntimeError(f"Repair norms require both expert tensors at {slot}")
            result.append({
                "layer_index": slot[0], "expert_id": slot[1],
                "unmasked_l2": math.sqrt(sum(pair[0] for pair in parts.values())),
                "masked_l2": math.sqrt(sum(pair[1] for pair in parts.values())),
            })
        return result


def _save_reports(
    output_dir: Path,
    *,
    layouts: list[ExpertLayerLayout],
    scores_by_task: dict[str, TaskScoreMap],
    selected_by_task: dict[str, frozenset[Slot]],
    repair_sources: dict[Slot, str],
    shared_tasks: dict[Slot, tuple[str, ...]],
    channel_selections: ChannelSelections,
    channel_keep_fraction: float,
    increment_norms: list[dict[str, int | float]],
    base_dir: Path,
    context_dir: Path,
    target_teacher_dirs: dict[str, Path],
    repair_fraction: float,
    selection_score: str,
    num_probes: int,
    seed: int,
    eps: float,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    total_slots = sum(layout.num_experts for layout in layouts)
    selected_tasks_by_slot: dict[Slot, list[str]] = {}
    for task, slots in selected_by_task.items():
        for slot in slots:
            selected_tasks_by_slot.setdefault(slot, []).append(task)

    report = {
        "method": "data_free_multi_task_post_merge_connection_conflict_repair",
        "base_model": str(base_dir.resolve()),
        "context_model": str(context_dir.resolve()),
        "target_tasks": list(target_teacher_dirs),
        "target_teachers": {
            task: str(path.resolve()) for task, path in target_teacher_dirs.items()
        },
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
            "shared_weight_update": (
                "theta_context + sum_{t in selecting_tasks} a_t * "
                "(M_t * (theta_t - theta_base))"
            ),
            "repair_coefficient": (
                "clip(sum_rh((f_task-f_context)*(f_task-f_base)) / "
                "(sum_rh((f_task-f_base)^2)+eps), 0, 1)"
            ),
        },
        "channel_keep_fraction": channel_keep_fraction,
        "channel_score_dtype": "float32",
        "connection_proxy": "sum_j g_j * u_j * d_j; teacher L1 connection strengths",
        "channel_score": "sum_{p in channel_j} abs((teacher_p-base_p) * dR/dtheta_p)",
        "channel_gradient_point": "teacher_weights; abs_subgradient_at_zero=0",
        "channel_selection_rule": (
            "top floor(intermediate_size * channel_keep_fraction) per selecting task "
            "per shared expert; ties prefer larger channel indices"
        ),
        "shared_increment_norms": increment_norms,
        "increment_norm_semantics": (
            "L2 of summed FP32 task repair increments, jointly over gate/up/down, "
            "before adding context or storage casting; FP64 norm reduction"
        ),
        "probe_distribution": "Rademacher({-1,+1})",
        "num_probes": num_probes,
        "seed": seed,
        "eps": eps,
        "device": str(device),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "repair_fraction": repair_fraction,
        "selection_score": selection_score,
        "selection_rule": (
            f"highest {selection_score} scores independently per task within "
            "each layer; floor(num_experts * repair_fraction) candidates"
        ),
        "conflict_rule": (
            "shared_experts_sum_connection_masked_directional_deltas_of_every_selecting_task; "
            "no_arbitration; no_backfill"
        ),
        "expert_slot_counts": {
            "selected_for_repair": len(selected_tasks_by_slot),
            "copied_from_target_teachers": len(repair_sources),
            "unselected_context": total_slots - len(selected_tasks_by_slot),
            "shared_connection_repairs": len(shared_tasks),
            "shared_all_zero_coefficients": sum(
                all(
                    scores_by_task[task][slot].repair_coefficient == 0.0
                    for task in tasks
                )
                for slot, tasks in shared_tasks.items()
            ),
        },
        "shared_slot_multiplicity": {
            str(count): sum(
                len(tasks) == count for tasks in shared_tasks.values()
            )
            for count in sorted({len(tasks) for tasks in shared_tasks.values()})
        },
        "shared_coefficient_sums": _summarize_values(
            [
                sum(scores_by_task[task][slot].repair_coefficient for task in tasks)
                for slot, tasks in shared_tasks.items()
            ]
        ),
        "per_task_slot_counts": {
            task: {
                "selected_for_repair": len(slots),
                "copied_from_teacher": sum(
                    repair_sources.get(slot) == task for slot in slots
                ),
                "shared_directional_contribution": len(slots & shared_tasks.keys()),
            }
            for task, slots in selected_by_task.items()
        },
        "semantics": {
            "uniquely_selected_expert": "copy_complete_selecting_task_teacher_expert",
            "multiply_selected_expert": (
                "context_plus_summed_connection_masked_directional_deltas_of_selecting_tasks"
            ),
            "unselected_expert": "copy_context",
            "non_expert_tensors": "copy_context",
            "scoring_context": "original_context_for_all_tasks",
            "update_context": "original_context; no_sequential_task_updates",
            "weight_update_dtype": "float32_accumulation_then_context_storage_dtype",
            "gate_up_and_down": "same_coefficient_and_coupled_channel_mask_per_task_per_shared_expert",
            "coefficient_aggregation": (
                "sum_not_mean; no_rescaling_after_channel_masking"
            ),
            "empty_channel_mask": "no_contribution_from_this_task",
            "channel_mask_scope": "teacher_minus_base_repair_increment_only",
            "all_zero_coefficient_shared_slot": "copy_context",
            "slot_counts": "planned_actions; not_counts_of_changed_stored_values",
            "csv_action": "final_expert_action_shared_by_all_task_rows",
            "csv_repair_coefficient": "per_task_coefficient; not_applied_at_unique_slots",
            "csv_applied_repair_coefficient": (
                "per_task_coefficient_at_shared_slots; blank_otherwise"
            ),
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
                "selected_for_repair",
                "selected_by_tasks",
                "repair_source_task",
                "repair_coefficient",
                "applied_repair_coefficient",
                "retained_channel_count",
                "retained_channel_ids",
                "channel_score_threshold",
            ]
        )
        for task, scores in scores_by_task.items():
            for layout in layouts:
                for expert_id in range(layout.num_experts):
                    slot = (layout.layer_index, expert_id)
                    score = scores[slot]
                    channel = channel_selections.get(slot, {}).get(task)
                    source_task = repair_sources.get(slot)
                    shared = shared_tasks.get(slot)
                    if shared is not None:
                        action = "repair_shared_connection_sum"
                    elif source_task is not None:
                        action = "repair"
                    else:
                        action = "keep_context"
                    writer.writerow(
                        [
                            task,
                            str(target_teacher_dirs[task].resolve()),
                            layout.layer_index,
                            layout.label,
                            expert_id,
                            f"{score.i_task:.17g}",
                            f"{score.i_lost:.17g}",
                            f"{score.geometric_mean:.17g}",
                            action,
                            slot in selected_by_task[task],
                            json.dumps(
                                selected_tasks_by_slot.get(slot, []),
                                ensure_ascii=False,
                            ),
                            source_task if source_task is not None else "",
                            f"{score.repair_coefficient:.17g}",
                            (f"{score.repair_coefficient:.17g}"
                             if shared is not None and task in shared else ""),
                            len(channel.retained_ids) if channel is not None else "",
                            json.dumps(channel.retained_ids) if channel is not None else "",
                            (f"{channel.threshold:.17g}"
                             if channel is not None and channel.threshold is not None else ""),
                        ]
                    )

    print(
        "Conflict-repair reports saved: conflict_repair_summary.json, "
        "conflict_repair_scores.csv"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use data-free post-merge functional loss to repair target tasks' "
            "Qwen3-VL packed MoE experts: copy the teacher at uniquely selected "
            "slots; at shared slots, add the sum of directionally weighted "
            "teacher-minus-base deltas from all selecting tasks to the context, "
            "retaining only their highest connection-saliency channels."
        )
    )
    parser.add_argument("--base", required=True, help="Local base checkpoint")
    parser.add_argument(
        "--context-model",
        "--context",
        dest="context_model",
        required=True,
        help=(
            "Previously merged checkpoint to score and repair. Unselected experts "
            "and non-expert tensors are copied from this model"
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
        "--target-tasks",
        "--target-task",
        dest="target_tasks",
        nargs="+",
        help="Tasks to repair from --task-names (default: all tasks)",
    )
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument(
        "--repair-fraction",
        type=float,
        default=0.125,
        help=(
            "Highest selected-score fraction nominated independently per task "
            "in every layer; shared slots sum directionally weighted deltas "
            "only from tasks selecting that slot, without backfilling "
            "(default: 0.125)"
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
    parser.add_argument(
        "--channel-keep-fraction",
        type=float,
        required=True,
        help=(
            "Connection-saliency channel fraction kept per selecting task at shared "
            "experts, in [0, 1]; keeps floor(intermediate_size * fraction) channels. "
            "0 retains context at shared slots; 1 reproduces directional repair. "
            "Independent of --repair-fraction; no rescaling after masking"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    if not math.isfinite(args.channel_keep_fraction) or not 0 <= args.channel_keep_fraction <= 1:
        raise ValueError("--channel-keep-fraction must be finite and in [0, 1]")
    if not 0.0 <= args.repair_fraction <= 1.0:
        raise ValueError("--repair-fraction must be in [0, 1]")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if args.expert_batch_size <= 0:
        raise ValueError("--expert-batch-size must be positive")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be positive")

    base_dir = Path(args.base)
    context_dir = Path(args.context_model)
    teacher_dirs = [Path(path) for path in args.teachers]
    task_names = _resolve_task_names(teacher_dirs, args.task_names)
    target_tasks = _resolve_target_tasks(task_names, args.target_tasks)
    teachers_by_name = dict(zip(task_names, teacher_dirs))
    target_teacher_dirs = {
        task: teachers_by_name[task] for task in target_tasks
    }
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
    # An overwritten run must not retain a previous successful completion marker.
    (output_dir / "repair_complete.json").unlink(missing_ok=True)

    device = _resolve_device(args.device)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype, device)
    print(f"Scoring device={device}, compute_dtype={compute_dtype}")
    print(f"Context model: {context_dir}")
    print(f"Target tasks: {', '.join(target_tasks)}")

    with ExitStack() as stack:
        base = stack.enter_context(SafetensorCheckpoint(base_dir))
        context = stack.enter_context(SafetensorCheckpoint(context_dir))
        teachers = {
            task: stack.enter_context(SafetensorCheckpoint(path))
            for task, path in target_teacher_dirs.items()
        }
        _validate_checkpoint_compatibility(base, [context, *teachers.values()])
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
        scores_by_task: dict[str, TaskScoreMap] = {}
        selected_by_task: dict[str, frozenset[Slot]] = {}
        for task, teacher in teachers.items():
            print(f"Scoring target task: {task}; teacher={target_teacher_dirs[task]}")
            scores = score_conflict_repairs(
                base,
                context,
                teacher,
                layouts,
                probes,
                target_task=task,
                device=device,
                compute_dtype=compute_dtype,
                expert_batch_size=args.expert_batch_size,
                eps=args.eps,
            )
            scores_by_task[task] = scores
            selected_by_task[task] = select_repairs(
                layouts,
                scores,
                target_task=task,
                repair_fraction=args.repair_fraction,
                selection_score=args.selection_score,
            )

        repair_sources, shared_tasks = resolve_repair_conflicts(selected_by_task)
        shared_coefficients = {
            slot: tuple(
                scores_by_task[task][slot].repair_coefficient for task in tasks
            )
            for slot, tasks in shared_tasks.items()
        }
        channel_selections = build_channel_selections(
            base, teachers, layouts, shared_tasks,
            keep_fraction=args.channel_keep_fraction,
            device=device, expert_batch_size=args.expert_batch_size,
        )
        total_slots = sum(layout.num_experts for layout in layouts)
        print(
            f"Repair slots: copied_teachers={len(repair_sources)}, "
            f"shared_connection={len(shared_tasks)}, "
            f"unselected_context={total_slots - len(repair_sources) - len(shared_tasks)}"
        )
        print(
            "Unique selections copy the complete teacher expert. Shared experts "
            "sum context + a_t * M_t * (teacher_t - base) over every selecting task. "
            "Unselected experts and non-expert tensors retain context."
        )

        repaired_state = ConnectionRepairCheckpoint(
            context,
            teachers,
            layouts,
            repair_sources,
            base=base,
            shared_tasks=shared_tasks,
            shared_coefficients=shared_coefficients,
            channel_selections=channel_selections,
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
        _save_reports(
            output_dir,
            layouts=layouts,
            scores_by_task=scores_by_task,
            selected_by_task=selected_by_task,
            repair_sources=repair_sources,
            shared_tasks=shared_tasks,
            channel_selections=channel_selections,
            channel_keep_fraction=args.channel_keep_fraction,
            increment_norms=repaired_state.increment_norm_report(),
            base_dir=base_dir,
            context_dir=context_dir,
            target_teacher_dirs=target_teacher_dirs,
            repair_fraction=args.repair_fraction,
            selection_score=args.selection_score,
            num_probes=args.num_probes,
            seed=args.seed,
            eps=args.eps,
            device=device,
            compute_dtype=compute_dtype,
        )

    with (output_dir / "repair_complete.json").open("w", encoding="utf-8") as f:
        json.dump({"status": "complete", "repair_mode": "copy_unique_sum_shared_connection"}, f)
        f.write("\n")

    print(
        f"Done. Conflict-repaired checkpoint for {', '.join(target_tasks)} "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
