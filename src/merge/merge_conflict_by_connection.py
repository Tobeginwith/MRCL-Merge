#!/usr/bin/env python3
"""Repair shared MoE experts with connection-salient channel updates.

Expert selection uses the original context and complete teacher outputs.
The default --coefficient-mode directional reuses merge_conflict.py coefficients.
With --coefficient-mode masked-jvp, coefficients are refitted in FP32 against
the actual masked task-vector response at the original context weights. Unique selections
copy the full teacher expert. For a shared expert e, only selecting tasks S_e
contribute:

    theta'_e = theta_context,e + scale * sum_t a_t,e * M_t,e * (theta_t,e - theta_base,e)

Each task independently retains floor(intermediate_size * channel_keep_fraction)
channels. A channel mask couples both gate/up columns and the corresponding
down row in the input-major packed checkpoint. There is no
channel-fraction normalization and no cross-task agreement modulation or iterative pruning.
--shared-repair-scale (default 1) scales the combined shared-expert increment
in either coefficient mode; unique teacher copies are unaffected.

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
        --channel-keep-fraction 0.25 --coefficient-mode masked-jvp \
        --output /models/connection-repair-jvp
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

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
    _swiglu_expert_forward,
    discover_expert_layers,
    make_rademacher_probes,
    save_selected_checkpoint,
)
from src.merge.merge_conflict import (  # noqa: E402
    ConflictRepairCheckpoint,
    RepairScore,
    Slot,
    TaskScoreMap,
    _resolve_target_tasks,
    _resolve_task_names,
    _summarize_values,
    _validate_checkpoint_compatibility,
    directional_repair_coefficient,
    resolve_repair_conflicts,
    select_repairs,
)


@torch.inference_mode()
def score_conflict_repairs_multi_task(
    base: SafetensorCheckpoint,
    context: SafetensorCheckpoint,
    teachers: dict[str, SafetensorCheckpoint],
    layouts: list[ExpertLayerLayout],
    probes_cpu: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    expert_batch_size: int,
    eps: float,
) -> dict[str, TaskScoreMap]:
    """Reuse base/context outputs within each batch; preserve original scoring math."""
    if expert_batch_size <= 0:
        raise ValueError("expert_batch_size must be positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    probes = probes_cpu.to(device=device, dtype=compute_dtype)
    scores_by_task: dict[str, TaskScoreMap] = {task: {} for task in teachers}
    for layer_number, layout in enumerate(layouts, start=1):
        print(f"Scoring layer {layer_number}/{len(layouts)}: {layout.label} "
              f"({layout.num_experts} packed experts, {len(teachers)} tasks)", flush=True)
        for start in range(0, layout.num_experts, expert_batch_size):
            end = min(start + expert_batch_size, layout.num_experts)
            base_output = _swiglu_expert_forward(
                probes, *layout.load_expert_batch(base, start, end),
                device=device, compute_dtype=compute_dtype,
            )
            context_output = _swiglu_expert_forward(
                probes, *layout.load_expert_batch(context, start, end),
                device=device, compute_dtype=compute_dtype,
            )
            base_denominator = base_output.square().sum(dim=-1).add_(eps)
            for task, teacher in teachers.items():
                task_output = _swiglu_expert_forward(
                    probes, *layout.load_expert_batch(teacher, start, end),
                    device=device, compute_dtype=compute_dtype,
                )
                task_numerator = (task_output - base_output).square().sum(dim=-1)
                lost_numerator = (task_output - context_output).square().sum(dim=-1)
                task_denominator = task_output.square().sum(dim=-1).add_(eps)
                i_task = task_numerator.div(base_denominator).mean(dim=-1)
                i_lost = lost_numerator.div(task_denominator).mean(dim=-1)
                geometric = torch.sqrt(i_task.clamp_min(0.0) * i_lost.clamp_min(0.0))
                if not (torch.isfinite(i_task).all() and torch.isfinite(i_lost).all()
                        and torch.isfinite(geometric).all()):
                    raise FloatingPointError(
                        f"Non-finite repair score for target task {task!r} "
                        f"at {layout.label}, experts [{start}:{end}]. "
                        "Try --compute-dtype float32."
                    )
                coefficients = directional_repair_coefficient(
                    base_output, context_output, task_output, eps
                ).cpu().tolist()
                values = zip(i_task.cpu().tolist(), i_lost.cpu().tolist(),
                             geometric.cpu().tolist(), coefficients)
                for expert_id, value in zip(range(start, end), values):
                    scores_by_task[task][(layout.layer_index, expert_id)] = RepairScore(*value)
                del (task_output, task_numerator, lost_numerator, task_denominator,
                     i_task, i_lost, geometric)
            del base_output, context_output, base_denominator
    return scores_by_task


_SCORE_CACHE_VERSION = 1


def _score_cache_config(
    base_dir: Path, context_dir: Path, target_teacher_dirs: dict[str, Path],
    *, num_probes: int, seed: int, compute_dtype: torch.dtype, eps: float,
) -> dict:
    # Deliberately lightweight: path identity, not a checkpoint content fingerprint.
    # Keep task order explicit; downstream conflict resolution also uses this order.
    return {
        "base": str(base_dir.resolve()), "context": str(context_dir.resolve()),
        "teachers": [[task, str(path.resolve())] for task, path in target_teacher_dirs.items()],
        "num_probes": num_probes, "seed": seed,
        "compute_dtype": str(compute_dtype).removeprefix("torch."), "eps": eps,
    }


def _score_cache_path(cache_dir: Path, config: dict) -> Path:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return cache_dir.resolve() / f"expert_scores_{hashlib.sha256(encoded).hexdigest()}.json"


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_score_cache(path: Path, config: dict,
                      layouts: list[ExpertLayerLayout]) -> dict[str, TaskScoreMap]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_unique_json_object)
    if not isinstance(payload, dict):
        raise ValueError("Cache must be a JSON object")
    if type(payload.get("version")) is not int or payload["version"] != _SCORE_CACHE_VERSION:
        raise ValueError("Unsupported score cache version")
    if payload.get("config") != config:
        raise ValueError("Score cache inputs do not match")
    tasks = [task for task, _ in config["teachers"]]
    records_by_task = payload.get("scores")
    if not isinstance(records_by_task, dict) or set(records_by_task) != set(tasks):
        raise ValueError("Score cache task coverage does not match")
    expected_slots = {(layout.layer_index, e) for layout in layouts
                      for e in range(layout.num_experts)}
    scores_by_task: dict[str, TaskScoreMap] = {}
    for task in tasks:
        records = records_by_task[task]
        if not isinstance(records, list):
            raise ValueError(f"Invalid score records for {task!r}")
        scores: TaskScoreMap = {}
        for row in records:
            # Each row: layer, expert, I_task, I_lost, geometric_mean, original a.
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError(f"Invalid score row for {task!r}")
            if any(type(x) is not int for x in row[:2]):
                raise ValueError(f"Invalid expert indices for {task!r}")
            slot = tuple(row[:2])
            if slot not in expected_slots or slot in scores:
                raise ValueError(f"Unexpected or duplicate expert slot {task!r}: {slot}")
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                   for x in row[2:]) or row[5] > 1:
                raise ValueError(f"Invalid score/coefficient for {task!r}: {slot}")
            scores[slot] = RepairScore(*(float(x) for x in row[2:]))
        if set(scores) != expected_slots:
            raise ValueError(f"Incomplete expert slot coverage for {task!r}")
        scores_by_task[task] = scores
    return scores_by_task


def _save_score_cache(path: Path, config: dict, scores_by_task: dict[str, TaskScoreMap]) -> None:
    payload = {
        "version": _SCORE_CACHE_VERSION, "config": config,
        "scores": {
            task: [[*slot, score.i_task, score.i_lost, score.geometric_mean,
                    score.repair_coefficient] for slot, score in sorted(scores.items())]
            for task, scores in scores_by_task.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def score_repairs_with_cache(
    base: SafetensorCheckpoint, context: SafetensorCheckpoint,
    teachers: dict[str, SafetensorCheckpoint], layouts: list[ExpertLayerLayout],
    probes_cpu: torch.Tensor, *, cache_dir: Path | None, cache_config: dict,
    device: torch.device, compute_dtype: torch.dtype, expert_batch_size: int, eps: float,
) -> tuple[dict[str, TaskScoreMap], dict]:
    """Cache only original full-expert scores, before selection, masking or JVP."""
    started = time.perf_counter()
    path = _score_cache_path(cache_dir, cache_config) if cache_dir is not None else None
    info = {"cache_status": "disabled" if path is None else "miss",
            "cache_path": str(path) if path is not None else None, "cache_saved": False}
    scores = None
    if path is not None:
        try:
            scores = _load_score_cache(path, cache_config, layouts)
            info["cache_status"] = "hit"
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
            info["cache_status"] = "invalid"
            info["cache_reason"] = str(exc)
            print(f"Ignoring invalid score cache {path}: {exc}", flush=True)
    if scores is None:
        print(f"Expert score cache: {info['cache_status']}; path={path}", flush=True)
        scores = score_conflict_repairs_multi_task(
            base, context, teachers, layouts, probes_cpu, device=device,
            compute_dtype=compute_dtype, expert_batch_size=expert_batch_size, eps=eps,
        )
        if path is not None:
            try:
                _save_score_cache(path, cache_config, scores)
                info["cache_saved"] = True
            except OSError as exc:
                info["cache_write_error"] = str(exc)
                print(f"Could not save score cache {path}: {exc}", flush=True)
    info["elapsed_seconds"] = time.perf_counter() - started
    print(f"Expert scoring: cache={info['cache_status']}, path={path}, "
          f"saved={info['cache_saved']}, elapsed={info['elapsed_seconds']:.3f}s", flush=True)
    return scores, info


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



JVP_DIAGNOSTIC_FIELDS = (
    "task_name", "layer_index", "expert_id", "original_coefficient",
    "original_fp32_coefficient", "original_fp32_difference", "raw_jvp_coefficient",
    "jvp_coefficient", "coefficient_difference", "jvp_l2", "context_residual_l2",
    "old_single_residual_l2", "new_single_residual_l2", "linear_residual_l2",
    "linearization_error_l2", "linearization_relative_error",
    "old_joint_residual_l2", "new_joint_residual_l2",
    "old_masked_increment_l2", "new_masked_increment_l2",
)


def _tensor_l2(value: torch.Tensor) -> float:
    result = value.double().square().sum().sqrt().item()
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite diagnostic norm")
    return result


def _weights_l2(weights: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(_tensor_l2(w) ** 2 for w in weights))


@torch.inference_mode()
def swiglu_masked_jvp(
    probes: torch.Tensor,
    context_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    direction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """FP32 analytic weight JVP at context; all weights use input-major layout.

    The direction is M * (teacher-base), without any old repair coefficient.
    Outputs have shape [experts, probes, hidden]. No Jacobian is materialized.
    """
    probes = probes.float()
    gate, up, down = (w.float() for w in context_weights)
    vg, vu, vd = (w.float() for w in direction)
    if any(not torch.isfinite(w).all() for w in (probes, gate, up, down, vg, vu, vd)):
        raise FloatingPointError("Non-finite JVP inputs")
    if any(w.shape != v.shape for w, v in zip((gate, up, down), (vg, vu, vd))):
        raise ValueError("JVP direction must match context weight shapes")
    g = torch.einsum("rh,ehi->eri", probes, gate)
    u = torch.einsum("rh,ehi->eri", probes, up)
    dg = torch.einsum("rh,ehi->eri", probes, vg)
    du = torch.einsum("rh,ehi->eri", probes, vu)
    sigmoid = torch.sigmoid(g)
    silu = F.silu(g)
    derivative = sigmoid + g * sigmoid * (1.0 - sigmoid)
    response = torch.einsum("eri,eih->erh", silu * u, vd)
    response = response + torch.einsum(
        "eri,eih->erh", derivative * dg * u + silu * du, down)
    if not torch.isfinite(response).all():
        raise FloatingPointError("Non-finite masked SwiGLU JVP")
    return response


@torch.inference_mode()
def project_jvp_coefficient(
    residual: torch.Tensor, jvp: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped and raw per-expert coefficients for the masked direction."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if residual.ndim != 3 or residual.shape != jvp.shape:
        raise ValueError("Residual and JVP must share [experts, probes, hidden] shape")
    residual, jvp = residual.float(), jvp.float()
    if not torch.isfinite(residual).all() or not torch.isfinite(jvp).all():
        raise FloatingPointError("Non-finite residual or JVP")
    numerator = (residual * jvp).sum(dim=(-2, -1))
    denominator = jvp.square().sum(dim=(-2, -1)) + eps
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise FloatingPointError("Non-finite JVP projection reduction")
    raw = numerator / denominator
    if not torch.isfinite(raw).all():
        raise FloatingPointError("Non-finite JVP repair coefficient")
    return raw.clamp(0.0, 1.0), raw


def validate_reference_selections(
    reference_csv: Path,
    scores_by_task: dict[str, TaskScoreMap],
    selected_by_task: dict[str, frozenset[Slot]],
    channel_selections: ChannelSelections,
) -> None:
    """Reject any changed expert/task selection or channel mask before refitting."""
    expected = {(task, *slot) for task, scores in scores_by_task.items() for slot in scores}
    seen = set()
    with reference_csv.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"task_name", "layer_index", "expert_id", "selected_for_repair",
                    "selected_by_tasks", "retained_channel_ids"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Reference CSV must contain expert selections and channel masks")
        for row in reader:
            task = row["task_name"]
            slot = (int(row["layer_index"]), int(row["expert_id"]))
            key = (task, *slot)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate reference row: {key}")
            seen.add(key)
            selected = slot in selected_by_task[task]
            selecting = [t for t, slots in selected_by_task.items() if slot in slots]
            channel = channel_selections.get(slot, {}).get(task)
            raw_ids = row["retained_channel_ids"]
            expected_ids = list(channel.retained_ids) if channel is not None else None
            actual_ids = json.loads(raw_ids) if raw_ids else None
            if (row["selected_for_repair"] != str(selected)
                    or json.loads(row["selected_by_tasks"]) != selecting
                    or actual_ids != expected_ids):
                raise ValueError(f"Reference expert selection or channel mask mismatch: {key}")
    if seen != expected:
        raise ValueError("Reference CSV does not cover all task/expert rows")
    print(f"Verified fixed expert selections and channel masks against {reference_csv}")


@torch.inference_mode()
def reestimate_masked_jvp_coefficients(
    base: SafetensorCheckpoint,
    context: SafetensorCheckpoint,
    teachers: dict[str, SafetensorCheckpoint],
    layouts: list[ExpertLayerLayout],
    shared_tasks: dict[Slot, tuple[str, ...]],
    channel_selections: ChannelSelections,
    original_coefficients: dict[Slot, tuple[float, ...]],
    probes_cpu: torch.Tensor,
    *,
    device: torch.device,
    eps: float,
    shared_repair_scale: float = 1.0,
) -> tuple[dict[Slot, tuple[float, ...]], list[dict]]:
    """Refit one shared expert at a time; diagnostics use pre-storage FP32 weights.

    Masks and original scores are never modified. Each task is fitted separately
    at the original context. Coefficient fitting is independent of the shared scale;
    residuals and increment norms measure updates after applying that scale.
    """
    probes = probes_cpu.to(device=device, dtype=torch.float32)
    layouts_by_layer = {layout.layer_index: layout for layout in layouts}
    coefficients, diagnostics = {}, []
    previous_layer = None
    for slot, tasks in sorted(shared_tasks.items()):
        layout = layouts_by_layer[slot[0]]
        if previous_layer != slot[0]:
            print(f"Refitting masked JVP coefficients: {layout.label}")
            previous_layer = slot[0]
        stage = "context/base outputs"
        try:
            def load(checkpoint):
                return tuple(w.to(device=device, dtype=torch.float32) for w in
                             layout.load_expert_batch(checkpoint, slot[1], slot[1] + 1))

            def forward(weights):
                return _swiglu_expert_forward(
                    probes, *weights, device=device, compute_dtype=torch.float32)

            bw, cw = load(base), load(context)
            base_output, context_output = forward(bw), forward(cw)
            old_joint, new_joint = tuple(w.clone() for w in cw), tuple(w.clone() for w in cw)
            old_increment = tuple(torch.zeros_like(w) for w in cw)
            new_increment = tuple(torch.zeros_like(w) for w in cw)
            slot_rows, teacher_outputs, fitted = [], {}, []
            for task, old_a in zip(tasks, original_coefficients[slot]):
                stage = f"task={task}"
                tw = load(teachers[task])
                teacher_output = forward(tw)
                teacher_outputs[task] = teacher_output
                mask = channel_selections[slot][task].mask.to(device)
                masks = (mask[None, None, :], mask[None, None, :], mask[None, :, None])
                direction = tuple((t - b).masked_fill(~m, 0.0)
                                  for t, b, m in zip(tw, bw, masks))
                jvp = swiglu_masked_jvp(probes, cw, direction)
                residual = teacher_output - context_output
                a_tensor, raw = project_jvp_coefficient(residual, jvp, eps)
                a = a_tensor.item()
                fitted.append(a)
                old_fp32 = directional_repair_coefficient(
                    base_output, context_output, teacher_output, eps).item()
                old_delta = tuple(old_a * v for v in direction)
                new_delta = tuple(a * v for v in direction)
                old_single = forward(tuple(c + shared_repair_scale * d for c, d in zip(cw, old_delta)))
                new_single = forward(tuple(c + shared_repair_scale * d for c, d in zip(cw, new_delta)))
                prediction = context_output + (shared_repair_scale * a) * jvp
                linear_error = _tensor_l2(new_single - prediction)
                for i in range(3):
                    old_joint[i].add_(old_delta[i])
                    new_joint[i].add_(new_delta[i])
                    old_increment[i].add_(old_delta[i])
                    new_increment[i].add_(new_delta[i])
                slot_rows.append({
                    "task_name": task, "layer_index": slot[0], "expert_id": slot[1],
                    "original_coefficient": old_a,
                    "original_fp32_coefficient": old_fp32,
                    "original_fp32_difference": old_fp32 - old_a,
                    "raw_jvp_coefficient": raw.item(), "jvp_coefficient": a,
                    "coefficient_difference": a - old_a, "jvp_l2": _tensor_l2(jvp),
                    "context_residual_l2": _tensor_l2(residual),
                    "old_single_residual_l2": _tensor_l2(teacher_output - old_single),
                    "new_single_residual_l2": _tensor_l2(teacher_output - new_single),
                    "linear_residual_l2": _tensor_l2(residual - (shared_repair_scale * a) * jvp),
                    "linearization_error_l2": linear_error,
                    "linearization_relative_error": linear_error / (_tensor_l2(new_single - context_output) + eps),
                })
                del tw, direction, jvp, old_delta, new_delta
            stage = "joint forward diagnostics"
            if shared_repair_scale != 1.0:
                old_joint = tuple(c + shared_repair_scale * d for c, d in zip(cw, old_increment))
                new_joint = tuple(c + shared_repair_scale * d for c, d in zip(cw, new_increment))
            old_output, new_output = forward(old_joint), forward(new_joint)
            old_norm = _weights_l2(tuple(shared_repair_scale * d for d in old_increment))
            new_norm = _weights_l2(tuple(shared_repair_scale * d for d in new_increment))
            for row in slot_rows:
                teacher_output = teacher_outputs[row["task_name"]]
                row.update({
                    "old_joint_residual_l2": _tensor_l2(teacher_output - old_output),
                    "new_joint_residual_l2": _tensor_l2(teacher_output - new_output),
                    "old_masked_increment_l2": old_norm,
                    "new_masked_increment_l2": new_norm,
                })
            coefficients[slot] = tuple(fitted)
            diagnostics.extend(slot_rows)
            del bw, cw, old_joint, new_joint, old_increment, new_increment, teacher_outputs
        except FloatingPointError as exc:
            raise FloatingPointError(f"Masked JVP failed at {slot}, {stage}: {exc}") from exc
    return coefficients, diagnostics


def _jvp_summary(rows: list[dict]) -> dict:
    def summarize(group):
        return {
            "count": len(group),
            "zero_coefficient_fraction": (sum(r["jvp_coefficient"] == 0 for r in group)
                                          / len(group) if group else 0.0),
            "one_coefficient_fraction": (sum(r["jvp_coefficient"] == 1 for r in group)
                                         / len(group) if group else 0.0),
            "zero_coefficients": sum(r["jvp_coefficient"] == 0 for r in group),
            "one_coefficients": sum(r["jvp_coefficient"] == 1 for r in group),
            "clipped_below_zero": sum(r["raw_jvp_coefficient"] < 0 for r in group),
            "clipped_above_one": sum(r["raw_jvp_coefficient"] > 1 for r in group),
            **{key: _summarize_values([r[key] for r in group]) for key in
               ("original_coefficient", "original_fp32_coefficient", "jvp_coefficient",
                "coefficient_difference", "original_fp32_difference", "jvp_l2",
                "linearization_relative_error")},
        }
    return {
        "arithmetic_dtype": "float32",
        "norm_reduction_dtype": "float64",
        "forward_diagnostics": "actual nonlinear SwiGLU at FP32 pre-storage weights",
        "linearization_relative_error": "||f(C+aV)-f(C)-aJ|| / (||f(C+aV)-f(C)|| + eps)",
        "increment_comparison": "old and refitted coefficients with the same channel masks",
        "overall": summarize(rows),
        "per_task": {task: summarize([r for r in rows if r["task_name"] == task])
                     for task in sorted({r["task_name"] for r in rows})},
    }


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
        shared_repair_scale: float = 1.0,
    ) -> None:
        if not math.isfinite(shared_repair_scale) or shared_repair_scale < 0:
            raise ValueError("shared_repair_scale must be finite and nonnegative")
        self.shared_repair_scale = shared_repair_scale
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
                teacher = self.teachers[task].get_slice(key, slice(e, e + 1))[0]
                stored = teacher.to(output.dtype)
                if not torch.isfinite(stored).all():
                    raise FloatingPointError(
                        f"Non-finite copied teacher weights after conversion to {output.dtype} "
                        f"at {(layer, e)}, {key}, task={task}"
                    )
                output[e].copy_(stored)
        for slot, tasks in sorted(self.shared_tasks.items()):
            layer, e = slot
            if layer != layout.layer_index:
                continue
            norms = self._increment_squared_norms.setdefault(slot, {})
            coefficients = self.shared_coefficients[slot]
            if self.shared_repair_scale == 0 or all(a == 0 for a in coefficients):
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
                if self.shared_repair_scale == 1.0 and self.channel_selections[slot][task].retained_ids:
                    repaired = repaired + masked
            if self.shared_repair_scale != 1.0:
                full_increment.mul_(self.shared_repair_scale)
                masked_increment.mul_(self.shared_repair_scale)
                repaired = repaired + masked_increment
            if not torch.isfinite(repaired).all():
                raise FloatingPointError(f"Non-finite repaired weights at {slot}, {key}")
            # The increments are FP32. FP64 reduction avoids overflow in diagnostics.
            before = full_increment.double().square().sum().item()
            after = masked_increment.double().square().sum().item()
            if not math.isfinite(before) or not math.isfinite(after):
                raise FloatingPointError(f"Non-finite repair increment norms at {slot}")
            stored = repaired.to(output.dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Non-finite repaired weights after conversion to {output.dtype} "
                    f"at {slot}, {key}; reduce --shared-repair-scale or use a wider storage dtype"
                )
            norms[key] = (before, after)
            output[e].copy_(stored)
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
    coefficient_mode: str,
    shared_coefficients: dict[Slot, tuple[float, ...]],
    jvp_diagnostics: list[dict],
    reference_repair_scores: Path | None,
    increment_norms: list[dict[str, int | float]],
    scoring_info: dict,
    shared_repair_scale: float,
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

    applied = {slot: {task: shared_repair_scale * a
                      for task, a in zip(tasks, shared_coefficients[slot])}
               for slot, tasks in shared_tasks.items()}
    report = {
        "method": "data_free_multi_task_post_merge_connection_conflict_repair",
        "expert_scoring": scoring_info,
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
                "theta_context + shared_repair_scale * sum_{t in selecting_tasks} a_t * "
                "(M_t * (theta_t - theta_base))"
            ),
            "repair_coefficient": (
                "clip(sum_rh((f_task-f_context)*(f_task-f_base)) / "
                "(sum_rh((f_task-f_base)^2)+eps), 0, 1)"
            ),
        },
        "coefficient_mode": coefficient_mode,
        "shared_repair_scale": shared_repair_scale,
        "reference_repair_scores": (str(reference_repair_scores.resolve())
                                    if reference_repair_scores is not None else None),
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
            "after shared_repair_scale, before adding context or storage casting; FP64 norm reduction"
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
                    applied[slot][task] == 0.0
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
                sum(applied[slot][task] for task in tasks)
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
                "sum_not_mean; apply_shared_repair_scale_to_combined_increment"
            ),
            "empty_channel_mask": "no_contribution_from_this_task",
            "channel_mask_scope": "teacher_minus_base_repair_increment_only",
            "all_zero_coefficient_shared_slot": "copy_context",
            "slot_counts": "planned_actions; not_counts_of_changed_stored_values",
            "csv_action": "final_expert_action_shared_by_all_task_rows",
            "csv_repair_coefficient": "original_directional_coefficient; applied value is in applied_repair_coefficient",
            "csv_applied_repair_coefficient": (
                "shared_repair_scale_times_active_mode_coefficient_at_shared_slots; blank_otherwise"
            ),
        },
        "limitation": (
            "expert-local synthetic probes; routing frequency, attention, vision, "
            "and real hidden-state distributions are not measured"
        ),
    }
    if coefficient_mode == "masked-jvp":
        report["formulas"]["original_repair_coefficient"] = report["formulas"]["repair_coefficient"]
        report["formulas"]["repair_coefficient"] = (
            "clip(<f_teacher(Z)-f_context(Z), J_context[M*(teacher-base)]> / "
            "(||J_context[M*(teacher-base)]||^2+eps), 0, 1)"
        )
        report["jvp_diagnostics"] = _jvp_summary(jvp_diagnostics)
        report["jvp_diagnostics"].update({
            "shared_repair_scale": shared_repair_scale,
            "coefficient_semantics": "fitted coefficients before shared scaling",
            "increment_comparison": "old and refitted coefficients with identical masks and shared_repair_scale",
            "linearization_relative_error": (
                "||f(C+scale*aV)-f(C)-scale*aJ|| / (||f(C+scale*aV)-f(C)|| + eps)"
            ),
        })
        with (output_dir / "connection_jvp_diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=JVP_DIAGNOSTIC_FIELDS)
            writer.writeheader()
            writer.writerows(jvp_diagnostics)
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
                            (f"{applied[slot][task]:.17g}"
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
            "0 retains context at shared slots; 1 keeps all channels. "
            "Independent of --repair-fraction; combined repair uses --shared-repair-scale"
        ),
    )
    parser.add_argument(
        "--coefficient-mode", choices=["directional", "masked-jvp"], default="directional",
        help=("Expert-level repair coefficient: original directional projection (default), "
              "or FP32 JVP refitting at context along the masked task vector"),
    )
    parser.add_argument(
        "--shared-repair-scale", type=float, default=1.0,
        help=("Common finite nonnegative multiplier of the combined masked repair "
              "at every shared expert, in either coefficient mode (default: 1). "
              "0 preserves context at shared experts; values above 1 amplify repairs. "
              "Does not change expert/channel selection, coefficient fitting, unique "
              "teacher copies or score-cache matching"),
    )
    parser.add_argument(
        "--reference-repair-scores", type=Path,
        help=("Optional baseline conflict_repair_scores.csv; require identical expert "
              "selections and retained channel IDs before computing repairs"),
    )
    parser.add_argument(
        "--score-cache-dir", type=Path,
        help=("Optional directory for reusable full-expert scores and original directional "
              "coefficients. Matches resolved model paths, ordered target tasks, R, seed, "
              "actual compute dtype and eps only. Batch size, device and repair settings "
              "do not invalidate it; cached scores remain authoritative. Clear this cache "
              "if weights change at the same path. Reference CSVs are validation-only"),
    )
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    if not math.isfinite(args.shared_repair_scale) or args.shared_repair_scale < 0:
        raise ValueError("--shared-repair-scale must be finite and nonnegative")
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

    # Validate the save setting before any existing output can be replaced.
    _parse_size_bytes(args.max_shard_size)

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
    if args.reference_repair_scores is not None and not args.reference_repair_scores.is_file():
        raise FileNotFoundError(f"Reference report not found: {args.reference_repair_scores}")
    # Reject invalid destinations early without modifying an existing checkpoint.
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it."
            )

    # TF32 only affects matmul, so it would silently reduce the two einsum paths
    # to a 10-bit mantissa: the FP32 probe forward, and the masked JVP, whose
    # FP32 arithmetic is independent of --compute-dtype. Channel saliency and the
    # score reductions use no matmul and are unaffected either way.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
        selected_by_task: dict[str, frozenset[Slot]] = {}
        for task, scores in scores_by_task.items():
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
        if args.reference_repair_scores is not None:
            validate_reference_selections(
                args.reference_repair_scores, scores_by_task,
                selected_by_task, channel_selections,
            )
        jvp_diagnostics = []
        if args.coefficient_mode == "masked-jvp":
            shared_coefficients, jvp_diagnostics = reestimate_masked_jvp_coefficients(
                base, context, teachers, layouts, shared_tasks, channel_selections,
                shared_coefficients, probes, device=device, eps=args.eps,
                shared_repair_scale=args.shared_repair_scale,
            )
        print(f"Repair coefficient mode: {args.coefficient_mode}; "
              f"shared repair scale={args.shared_repair_scale}")
        total_slots = sum(layout.num_experts for layout in layouts)
        print(
            f"Repair slots: copied_teachers={len(repair_sources)}, "
            f"shared_connection={len(shared_tasks)}, "
            f"unselected_context={total_slots - len(repair_sources) - len(shared_tasks)}"
        )
        print(
            "Unique selections copy the complete teacher expert. Shared experts "
            "add shared_repair_scale times the summed masked task repairs to context. "
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
            shared_repair_scale=args.shared_repair_scale,
        )
        # All input/reference checks, scoring, mask construction and coefficient
        # fitting have succeeded. Only now begin replacing the output checkpoint.
        # Invalidate completion before cleanup, which can itself fail partway.
        (output_dir / "repair_complete.json").unlink(missing_ok=True)
        if args.overwrite:
            _prepare_output_dir(output_dir, overwrite=True)
        else:
            # An explicitly colocated score cache may have created this directory
            # since the read-only destination check above.
            output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "connection_jvp_diagnostics.csv").unlink(missing_ok=True)
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
            coefficient_mode=args.coefficient_mode,
            shared_coefficients=shared_coefficients,
            jvp_diagnostics=jvp_diagnostics,
            reference_repair_scores=args.reference_repair_scores,
            increment_norms=repaired_state.increment_norm_report(),
            scoring_info=scoring_info,
            shared_repair_scale=args.shared_repair_scale,
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
        json.dump({
            "status": "complete",
            "repair_mode": ("copy_unique_sum_shared_connection_jvp"
                            if args.coefficient_mode == "masked-jvp"
                            else "copy_unique_sum_shared_connection"),
            "coefficient_mode": args.coefficient_mode,
            "shared_repair_scale": args.shared_repair_scale,
        }, f)
        f.write("\n")

    print(
        f"Done. Conflict-repaired checkpoint for {', '.join(target_tasks)} "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
