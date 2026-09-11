#!/usr/bin/env python3
"""Ablate directional coefficients in connection-masked expert repair.

Expert scores, selections, and teacher-minus-base connection channel masks are
identical to merge_conflict_by_connection.py's directional branch. Only shared
expert updates change: every selecting task contributes its masked difference
from the ORIGINAL context, without a directional coefficient:

    theta'_e = theta_context,e + scale * sum_t M_t,e * (theta_t,e - theta_context,e)

Unique selections copy the complete teacher expert. Unselected experts and
non-expert tensors retain context. There is no task-count or channel-fraction
normalization. Differences and accumulation use FP32 with one final cast to
context storage precision. --shared-repair-scale defaults to 1 and only affects
shared experts. Cached directional coefficients are retained for comparison,
but never participate in weight updates.

Example:
    python src/merge/ablation_connection.py \
        --base /models/base --context-model /models/context \
        --teachers /models/med /models/math --task-names medvqa wemath2 \
        --selection-score i_task --repair-fraction 0.125 \
        --channel-keep-fraction 0.25 --shared-repair-scale 1.0 \
        --output /models/ablation-connection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.merge.identify_important_experts import (  # noqa: E402
    ExpertLayerLayout,
    SafetensorCheckpoint,
    _parse_size_bytes,
    _prepare_output_dir,
    _resolve_compute_dtype,
    _resolve_device,
    discover_expert_layers,
    make_rademacher_probes,
    save_selected_checkpoint,
)
from src.merge.merge_conflict import (  # noqa: E402
    Slot,
    TaskScoreMap,
    _resolve_target_tasks,
    _resolve_task_names,
    _validate_checkpoint_compatibility,
    resolve_repair_conflicts,
    select_repairs,
)
from src.merge.merge_conflict_by_connection import (  # noqa: E402
    ChannelSelections,
    ConnectionRepairCheckpoint,
    _score_cache_config,
    build_channel_selections,
    score_repairs_with_cache,
    validate_reference_selections,
)


REPAIR_MODE = "copy_unique_sum_shared_connection_context_no_direction"
SHARED_WEIGHT_UPDATE = (
    "theta_context + shared_repair_scale * sum_{t in selecting_tasks} "
    "M_t * (theta_t - theta_context)"
)


class AblationConnectionCheckpoint(ConnectionRepairCheckpoint):
    """Stream context-based shared updates and complete unique teacher copies."""

    def __init__(
        self,
        context: SafetensorCheckpoint,
        teachers: dict[str, SafetensorCheckpoint],
        layouts: list[ExpertLayerLayout],
        repair_sources: dict[Slot, str],
        *,
        shared_tasks: dict[Slot, tuple[str, ...]],
        channel_selections: ChannelSelections,
        shared_repair_scale: float = 1.0,
    ) -> None:
        # Reuse checkpoint access, mask validation, and norm reporting. The
        # inherited constructor requires coefficients and a base; these internal
        # placeholders are not used by the update implementation below.
        super().__init__(
            context, teachers, layouts, repair_sources,
            base=context, shared_tasks=shared_tasks,
            shared_coefficients={slot: (1.0,) * len(tasks)
                                 for slot, tasks in shared_tasks.items()},
            channel_selections=channel_selections,
            shared_repair_scale=shared_repair_scale,
        )

    @torch.inference_mode()
    def _repair_packed_expert_tensor(
        self, key: str, layout: ExpertLayerLayout,
    ) -> torch.Tensor:
        output = self.context.get_tensor(key).clone()
        for (layer, expert_id), task in sorted(self.repair_sources.items()):
            if layer != layout.layer_index:
                continue
            teacher = self.teachers[task].get_slice(key, slice(expert_id, expert_id + 1))[0]
            stored = teacher.to(output.dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Non-finite copied teacher weights after conversion to {output.dtype} "
                    f"at {(layer, expert_id)}, {key}, task={task}"
                )
            output[expert_id].copy_(stored)

        for slot, tasks in sorted(self.shared_tasks.items()):
            layer, expert_id = slot
            if layer != layout.layer_index:
                continue
            norms = self._increment_squared_norms.setdefault(slot, {})
            if self.shared_repair_scale == 0:
                norms[key] = (0.0, 0.0)
                continue
            # Never subtract a previously repaired value. This original context
            # stays fixed while all selecting tasks contribute to the sum.
            context = output[expert_id].float()
            full_increment = torch.zeros_like(context)
            masked_increment = torch.zeros_like(context)
            for task in tasks:
                teacher = self.teachers[task].get_slice(
                    key, slice(expert_id, expert_id + 1)
                )[0].float()
                increment = teacher - context
                mask = self.channel_selections[slot][task].mask
                expanded = (torch.cat((mask, mask))[None, :]
                            if key == layout.gate_up_key else mask[:, None])
                full_increment.add_(increment)
                masked_increment.add_(increment.masked_fill(~expanded, 0.0))

            full_increment.mul_(self.shared_repair_scale)
            masked_increment.mul_(self.shared_repair_scale)
            repaired = context + masked_increment
            if not torch.isfinite(repaired).all():
                raise FloatingPointError(f"Non-finite repaired weights at {slot}, {key}")
            before = full_increment.double().square().sum().item()
            after = masked_increment.double().square().sum().item()
            if not math.isfinite(before) or not math.isfinite(after):
                raise FloatingPointError(f"Non-finite repair increment norms at {slot}, {key}")
            stored = repaired.to(output.dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Non-finite repaired weights after conversion to {output.dtype} "
                    f"at {slot}, {key}; reduce --shared-repair-scale or use a wider storage dtype"
                )
            norms[key] = (before, after)
            # Leave unmasked channels bitwise intact (including signed zeros).
            active = torch.stack([
                self.channel_selections[slot][task].mask for task in tasks
            ]).any(dim=0)
            if key == layout.gate_up_key:
                output[expert_id, :, torch.cat((active, active))] = stored[:, torch.cat((active, active))]
            else:
                output[expert_id, active, :] = stored[active, :]
        return output


def _save_reports(
    output_dir: Path,
    *,
    layouts: list[ExpertLayerLayout],
    scores_by_task: dict[str, TaskScoreMap],
    selected_by_task: dict[str, frozenset[Slot]],
    repair_sources: dict[Slot, str],
    shared_tasks: dict[Slot, tuple[str, ...]],
    channel_selections: ChannelSelections,
    increment_norms: list[dict[str, int | float]],
    scoring_info: dict,
    base_dir: Path,
    context_dir: Path,
    target_teacher_dirs: dict[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    selecting: dict[Slot, list[str]] = {}
    for task, slots in selected_by_task.items():
        for slot in slots:
            selecting.setdefault(slot, []).append(task)
    total_slots = sum(layout.num_experts for layout in layouts)
    report = {
        "method": "data_free_multi_task_post_merge_connection_ablation",
        "repair_mode": REPAIR_MODE,
        "coefficient_mode": "none",
        "shared_repair_scale": args.shared_repair_scale,
        "expert_scoring": scoring_info,
        "base_model": str(base_dir.resolve()),
        "context_model": str(context_dir.resolve()),
        "target_tasks": list(target_teacher_dirs),
        "target_teachers": {task: str(path.resolve())
                            for task, path in target_teacher_dirs.items()},
        "formulas": {
            "I_task": "mean_r(||f_task(z_r)-f_base(z_r)||_2^2 / (||f_base(z_r)||_2^2+eps))",
            "I_lost": "mean_r(||f_task(z_r)-f_context(z_r)||_2^2 / (||f_task(z_r)||_2^2+eps))",
            "repair_score": "sqrt(I_task * I_lost)",
            "shared_weight_update": SHARED_WEIGHT_UPDATE,
            "original_repair_coefficient_unused": (
                "clip(sum_rh((f_task-f_context)*(f_task-f_base)) / "
                "(sum_rh((f_task-f_base)^2)+eps), 0, 1)"
            ),
        },
        "reference_repair_scores": (str(args.reference_repair_scores.resolve())
                                    if args.reference_repair_scores is not None else None),
        "channel_keep_fraction": args.channel_keep_fraction,
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
            "L2 of summed FP32 teacher-minus-context increments, jointly over gate/up/down, "
            "after shared_repair_scale, before adding context or storage casting; FP64 norm reduction"
        ),
        "probe_distribution": "Rademacher({-1,+1})",
        "num_probes": args.num_probes,
        "seed": args.seed,
        "eps": args.eps,
        "device": str(device),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "repair_fraction": args.repair_fraction,
        "selection_score": args.selection_score,
        "selection_rule": (
            f"highest {args.selection_score} scores independently per task within "
            "each layer; floor(num_experts * repair_fraction) candidates"
        ),
        "conflict_rule": "sum_masked_teacher_minus_context_for_selecting_tasks; no_arbitration; no_backfill",
        "expert_slot_counts": {
            "selected_for_repair": len(selecting),
            "copied_from_target_teachers": len(repair_sources),
            "unselected_context": total_slots - len(selecting),
            "shared_connection_repairs": len(shared_tasks),
        },
        "shared_slot_multiplicity": {
            str(count): sum(len(tasks) == count for tasks in shared_tasks.values())
            for count in sorted({len(tasks) for tasks in shared_tasks.values()})
        },
        "per_task_slot_counts": {
            task: {
                "selected_for_repair": len(slots),
                "copied_from_teacher": sum(repair_sources.get(slot) == task for slot in slots),
                "shared_connection_contribution": len(slots & shared_tasks.keys()),
            }
            for task, slots in selected_by_task.items()
        },
        "semantics": {
            "uniquely_selected_expert": "copy_complete_selecting_task_teacher_expert; unaffected_by_scale_or_masks",
            "multiply_selected_expert": "context_plus_scaled_sum_of_masked_teacher_minus_context",
            "unselected_expert": "copy_context",
            "non_expert_tensors": "copy_context",
            "scoring_context": "original_context_for_all_tasks",
            "update_context": "original_context; no_sequential_task_updates",
            "weight_update_dtype": "float32_accumulation_then_context_storage_dtype",
            "gate_up_and_down": "coupled_channel_mask_per_task_per_shared_expert",
            "aggregation": "sum_not_mean; scale_combined_increment; no_channel_fraction_normalization",
            "empty_channel_mask": "no_contribution_from_this_task",
            "channel_mask_scope": "teacher_minus_context_repair_increment_only",
            "zero_scale": "copy_context_at_shared_slots",
            "slot_counts": "planned_actions; not_counts_of_changed_stored_values",
            "csv_action": "final_expert_action_shared_by_all_task_rows",
            "csv_repair_coefficient": "original_directional_coefficient_for_comparison_only; never_applied",
            "csv_applied_repair_coefficient": "global_shared_repair_scale_for_selecting_tasks_at_shared_slots; blank_otherwise",
        },
        "limitation": "expert-local synthetic probes; routing frequency and real hidden-state distributions are not measured",
    }
    with (output_dir / "conflict_repair_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    # Preserve the baseline CSV columns so reference-selection validation and
    # existing analysis tools can compare the same expert/channel selections.
    with (output_dir / "conflict_repair_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "task_name", "teacher", "layer_index", "layer_path", "expert_id",
            "I_task", "I_lost", "geometric_mean", "action", "selected_for_repair",
            "selected_by_tasks", "repair_source_task", "repair_coefficient",
            "applied_repair_coefficient", "retained_channel_count",
            "retained_channel_ids", "channel_score_threshold",
        ])
        for task, scores in scores_by_task.items():
            for layout in layouts:
                for expert_id in range(layout.num_experts):
                    slot = (layout.layer_index, expert_id)
                    score = scores[slot]
                    channel = channel_selections.get(slot, {}).get(task)
                    source = repair_sources.get(slot)
                    shared = shared_tasks.get(slot)
                    action = ("repair_shared_connection_context_sum" if shared is not None
                              else "repair" if source is not None else "keep_context")
                    writer.writerow([
                        task, str(target_teacher_dirs[task].resolve()), layout.layer_index,
                        layout.label, expert_id, f"{score.i_task:.17g}", f"{score.i_lost:.17g}",
                        f"{score.geometric_mean:.17g}", action, slot in selected_by_task[task],
                        json.dumps(selecting.get(slot, []), ensure_ascii=False), source or "",
                        f"{score.repair_coefficient:.17g}",
                        (f"{args.shared_repair_scale:.17g}" if shared is not None and task in shared else ""),
                        len(channel.retained_ids) if channel is not None else "",
                        json.dumps(channel.retained_ids) if channel is not None else "",
                        (f"{channel.threshold:.17g}" if channel is not None and channel.threshold is not None else ""),
                    ])
    print("Ablation reports saved: conflict_repair_summary.json, conflict_repair_scores.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Connection repair ablation: copy unique teacher experts; at shared experts, "
        "add scale times the sum of masked teacher-minus-context deltas, without "
        "directional coefficients. Channel selection still uses teacher-minus-base."
    ))
    parser.add_argument("--base", required=True, help="Local base checkpoint for scores and channel masks")
    parser.add_argument("--context-model", "--context", dest="context_model", required=True,
                        help="Merged checkpoint; original context for scoring and shared weight differences")
    parser.add_argument("--teachers", nargs="+", required=True, help="Task checkpoint directories in task order")
    parser.add_argument("--task-names", nargs="+", required=True, help="Unique names in the same order as teachers")
    parser.add_argument("--target-tasks", "--target-task", dest="target_tasks", nargs="+",
                        help="Tasks to repair (default: all tasks)")
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument("--repair-fraction", type=float, default=0.125,
                        help="Top expert fraction selected independently per task per layer (default: 0.125)")
    parser.add_argument("--selection-score", choices=["geometric", "i_task", "i_lost"], default="geometric",
                        help="Expert ranking score (default: geometric); all scores are reported")
    parser.add_argument("--num-probes", "-R", type=int, default=16, help="Shared Rademacher probes (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Probe RNG seed (default: 42)")
    parser.add_argument("--expert-batch-size", type=int, default=8, help="Experts per scoring batch (default: 8)")
    parser.add_argument("--device", default="auto", help="Scoring device, e.g. auto, cuda:0, cpu (default: auto)")
    parser.add_argument("--compute-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto",
                        help="Synthetic forward dtype (default: auto); channel scoring and updates use FP32")
    parser.add_argument("--eps", type=float, default=1e-8, help="Positive score denominator stabilizer (default: 1e-8)")
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output shard size (default: 5GB)")
    parser.add_argument("--processor-source", help="Tokenizer/processor source (default: context model)")
    parser.add_argument("--save-processor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    parser.add_argument("--channel-keep-fraction", type=float, required=True, help=(
        "Channel fraction in [0, 1]; keep floor(intermediate_size * fraction) per selecting "
        "task at shared experts. 0 preserves context; 1 keeps all channels"
    ))
    parser.add_argument("--shared-repair-scale", type=float, default=1.0, help=(
        "Finite nonnegative multiplier of the combined shared increment (default: 1). "
        "0 preserves shared context; does not affect unique teacher copies or selection"
    ))
    parser.add_argument("--reference-repair-scores", type=Path, help=(
        "Optional baseline conflict_repair_scores.csv; require identical expert selections "
        "and retained channel IDs before saving"
    ))
    parser.add_argument("--score-cache-dir", type=Path, help=(
        "Reuse baseline full-expert scores and original coefficients (unused in updates). "
        "Matches resolved model paths, ordered tasks, probes, seed, dtype and eps. "
        "Clear the cache if weights change at the same path"
    ))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not math.isfinite(args.shared_repair_scale) or args.shared_repair_scale < 0:
        raise ValueError("--shared-repair-scale must be finite and nonnegative")
    if not math.isfinite(args.channel_keep_fraction) or not 0 <= args.channel_keep_fraction <= 1:
        raise ValueError("--channel-keep-fraction must be finite and in [0, 1]")
    if not 0 <= args.repair_fraction <= 1:
        raise ValueError("--repair-fraction must be in [0, 1]")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if args.expert_batch_size <= 0:
        raise ValueError("--expert-batch-size must be positive")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be finite and positive")
    _parse_size_bytes(args.max_shard_size)

    base_dir, context_dir = Path(args.base), Path(args.context_model)
    teacher_dirs = [Path(path) for path in args.teachers]
    task_names = _resolve_task_names(teacher_dirs, args.task_names)
    target_tasks = _resolve_target_tasks(task_names, args.target_tasks)
    teachers_by_name = dict(zip(task_names, teacher_dirs))
    target_teacher_dirs = {task: teachers_by_name[task] for task in target_tasks}
    output_dir = Path(args.output)
    processor_source = Path(args.processor_source) if args.processor_source else context_dir
    if output_dir.resolve() in {base_dir.resolve(), context_dir.resolve(),
                                *(path.resolve() for path in teacher_dirs)}:
        raise ValueError("Output directory must differ from base, context, and teacher directories")
    if args.save_processor and not processor_source.is_dir():
        raise FileNotFoundError(f"Processor source directory not found: {processor_source}")
    if args.reference_repair_scores is not None and not args.reference_repair_scores.is_file():
        raise FileNotFoundError(f"Reference report not found: {args.reference_repair_scores}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it.")

    # Match the baseline's probe forward precision and score-cache semantics.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = _resolve_device(args.device)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype, device)
    print(f"Scoring device={device}, compute_dtype={compute_dtype}")
    print(f"Context model: {context_dir}; target tasks: {', '.join(target_tasks)}")

    with ExitStack() as stack:
        base = stack.enter_context(SafetensorCheckpoint(base_dir))
        context = stack.enter_context(SafetensorCheckpoint(context_dir))
        teachers = {task: stack.enter_context(SafetensorCheckpoint(path))
                    for task, path in target_teacher_dirs.items()}
        _validate_checkpoint_compatibility(base, [context, *teachers.values()])
        layouts = discover_expert_layers(base)
        hidden_sizes = {layout.hidden_size for layout in layouts}
        if len(hidden_sizes) != 1:
            raise ValueError(f"All MoE layers must share one hidden size for shared probes, got {sorted(hidden_sizes)}")
        hidden_size = next(iter(hidden_sizes))
        probes = make_rademacher_probes(args.num_probes, hidden_size, args.seed)
        cache_config = _score_cache_config(
            base_dir, context_dir, target_teacher_dirs, num_probes=args.num_probes,
            seed=args.seed, compute_dtype=compute_dtype, eps=args.eps,
        )
        scores_by_task, scoring_info = score_repairs_with_cache(
            base, context, teachers, layouts, probes,
            cache_dir=args.score_cache_dir, cache_config=cache_config,
            device=device, compute_dtype=compute_dtype,
            expert_batch_size=args.expert_batch_size, eps=args.eps,
        )
        selected_by_task = {
            task: select_repairs(layouts, scores, target_task=task,
                                 repair_fraction=args.repair_fraction,
                                 selection_score=args.selection_score)
            for task, scores in scores_by_task.items()
        }
        repair_sources, shared_tasks = resolve_repair_conflicts(selected_by_task)
        # Keep teacher-minus-base saliency unchanged for direct baseline comparison.
        channel_selections = build_channel_selections(
            base, teachers, layouts, shared_tasks, keep_fraction=args.channel_keep_fraction,
            device=device, expert_batch_size=args.expert_batch_size,
        )
        if args.reference_repair_scores is not None:
            validate_reference_selections(
                args.reference_repair_scores, scores_by_task, selected_by_task, channel_selections,
            )
        total_slots = sum(layout.num_experts for layout in layouts)
        print(f"Repair slots: copied_teachers={len(repair_sources)}, shared_connection={len(shared_tasks)}, "
              f"unselected_context={total_slots - len(repair_sources) - len(shared_tasks)}")
        print(f"Shared update: {SHARED_WEIGHT_UPDATE}; scale={args.shared_repair_scale}")
        repaired_state = AblationConnectionCheckpoint(
            context, teachers, layouts, repair_sources, shared_tasks=shared_tasks,
            channel_selections=channel_selections, shared_repair_scale=args.shared_repair_scale,
        )

        # Finish validation, scoring, and reference checks before replacing outputs.
        (output_dir / "repair_complete.json").unlink(missing_ok=True)
        if args.overwrite:
            _prepare_output_dir(output_dir, overwrite=True)
        else:
            # A colocated score cache may have created the directory meanwhile.
            output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "connection_jvp_diagnostics.csv").unlink(missing_ok=True)
        save_selected_checkpoint(
            repaired_state, expert_dir=context_dir, processor_source=processor_source,
            output_dir=output_dir, max_shard_size=args.max_shard_size,
            save_processor=args.save_processor, trust_remote_code=args.trust_remote_code,
        )
        _save_reports(
            output_dir, layouts=layouts, scores_by_task=scores_by_task,
            selected_by_task=selected_by_task, repair_sources=repair_sources,
            shared_tasks=shared_tasks, channel_selections=channel_selections,
            increment_norms=repaired_state.increment_norm_report(), scoring_info=scoring_info,
            base_dir=base_dir, context_dir=context_dir, target_teacher_dirs=target_teacher_dirs,
            args=args, device=device, compute_dtype=compute_dtype,
        )

    with (output_dir / "repair_complete.json").open("w", encoding="utf-8") as stream:
        json.dump({"status": "complete", "repair_mode": REPAIR_MODE,
                   "coefficient_mode": "none", "shared_repair_scale": args.shared_repair_scale}, stream)
        stream.write("\n")
    print(f"Done. Connection ablation for {', '.join(target_tasks)} -> {output_dir}")


if __name__ == "__main__":
    main()
