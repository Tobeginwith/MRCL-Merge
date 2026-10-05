#!/usr/bin/env python3
"""Calibrate all floating matrices of an already merged checkpoint with SVC.

Ordinary 2D tensors use their stored orientation. Qwen packed MoE experts are
split into individual gate/up/down matrices, transposed from [input, output]
to [output, input] for left-space SVC, then restored to their original layout.
Other tensor ranks and non-floating tensors are copied from the context model.
This script performs no expert selection or connection repair.

SVC (arXiv:2602.05536v2) uses context-minus-base and teacher-minus-base updates
in CUDA FP32, without additional merge scaling. SVD uses the default driver;
OOM and decomposition failures propagate without CPU fallback or noise retries.

Example:
    python src/merge/svc_ablation.py --base /models/base \
        --context-model /models/merged --teachers /models/task_a /models/task_b \
        --svc-alpha 1.0 --output /models/merged-svc
"""

from __future__ import annotations

import argparse
import json
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
    _resolve_device,
    discover_expert_layers,
    save_selected_checkpoint,
)
from src.merge.merge_conflict import (  # noqa: E402
    _resolve_target_tasks,
    _resolve_task_names,
    _validate_checkpoint_compatibility,
)
from src.merge.merge_conflict_by_connection import (  # noqa: E402
    _calibrate_svc_matrix,
    _validate_svc_options,
)


class SVCCalibratedCheckpoint:
    """Compute one tensor on demand through the existing shard saver interface."""

    def __init__(
        self,
        base: SafetensorCheckpoint,
        context: SafetensorCheckpoint,
        teachers: dict[str, SafetensorCheckpoint],
        *,
        device: torch.device,
        alpha: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        _validate_svc_options(alpha, eps)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("SVC requires a CUDA --device; no CPU fallback")
        if not teachers:
            raise ValueError("SVC requires at least one target teacher")
        _validate_checkpoint_compatibility(base, [context, *teachers.values()])
        self.base, self.context, self.teachers = base, context, teachers
        self.device, self.alpha, self.eps = device, alpha, eps
        packed_keys = {
            key for key in context.keys()
            if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
        }
        # Dense models are supported; malformed or unpaired packed experts fail early.
        layouts = discover_expert_layers(context) if packed_keys else []
        self.expert_key_layout = {
            key: layout for layout in layouts for key in layout.all_weight_keys()
        }
        if packed_keys != self.expert_key_layout.keys():
            raise ValueError(f"Unrecognized packed expert keys: {sorted(packed_keys - self.expert_key_layout.keys())}")
        for layout in layouts:
            if min(layout.num_experts, layout.hidden_size, layout.intermediate_size) <= 0:
                raise ValueError(f"Empty packed expert dimensions at {layout.prefix}")
        # Small per-matrix statuses only; repeated reads replace entries.
        self._results: dict[str, tuple[str, str]] = {}
        self._copied_keys: set[str] = set()

    def keys(self) -> list[str]:
        return self.context.keys()

    def get_shape(self, key: str) -> tuple[int, ...]:
        return self.context.get_shape(key)

    def get_dtype(self, key: str) -> str:
        return self.context.get_dtype(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        context = self.context.get_tensor(key)
        layout = self.expert_key_layout.get(key)
        if not context.is_floating_point() or (layout is None and context.ndim != 2):
            self._copied_keys.add(key)
            return context
        if layout is not None:
            return self._calibrate_experts(key, context, layout)
        output, status = _calibrate_svc_matrix(
            key, context, lambda: self.base.get_tensor(key),
            {task: (lambda teacher=teacher: teacher.get_tensor(key))
             for task, teacher in self.teachers.items()},
            device=self.device, alpha=self.alpha, eps=self.eps,
        )
        self._results[key] = ("ordinary", status)
        return output

    def _calibrate_experts(
        self, key: str, context: torch.Tensor, layout: ExpertLayerLayout,
    ) -> torch.Tensor:
        output = torch.empty_like(context, device="cpu")
        projections = (("gate", slice(0, layout.intermediate_size)),
                       ("up", slice(layout.intermediate_size, 2 * layout.intermediate_size)))
        if key == layout.down_key:
            projections = (("down", slice(None)),)
        for expert in range(layout.num_experts):
            for projection, columns in projections:
                identity = f"{key}/expert={expert}/projection={projection}"

                def load_part(checkpoint: SafetensorCheckpoint) -> torch.Tensor:
                    # Only this expert is materialized from each base/teacher shard.
                    return checkpoint.get_slice(key, expert)[:, columns].T

                result, status = _calibrate_svc_matrix(
                    identity, context[expert, :, columns].T,
                    lambda: load_part(self.base),
                    {task: (lambda teacher=teacher: load_part(teacher))
                     for task, teacher in self.teachers.items()},
                    device=self.device, alpha=self.alpha, eps=self.eps,
                )
                output[expert, :, columns].copy_(result.T)
                self._results[identity] = (projection, status)
        return output

    def summary(self) -> dict:
        def counts(kind: str | None = None) -> dict:
            statuses = [status for category, status in self._results.values()
                        if kind is None or category == kind]
            return {
                "processed": len(statuses),
                "calibrated": statuses.count("calibrated"),
                "zero_delta": statuses.count("zero_delta"),
            }

        return {
            "method": "svc_ablation",
            "base_model": str(self.base.model_dir.resolve()),
            "context_model": str(self.context.model_dir.resolve()),
            "target_tasks": list(self.teachers),
            "target_teachers": {task: str(teacher.model_dir.resolve())
                                for task, teacher in self.teachers.items()},
            "alpha": self.alpha, "eps": self.eps, "device": str(self.device),
            "compute_dtype": "float32", "svd_driver": "default", "tf32": False,
            "parameter_scope": "all_floating_2d_tensors_and_individual_packed_expert_gate_up_down",
            "other_parameters": "copy_context",
            "packed_orientation": "transpose_to_output_input_then_restore",
            "merged_delta": "context_minus_base; no_additional_scaling",
            "matrix_counts": counts(),
            "ordinary_matrix_counts": counts("ordinary"),
            "expert_projection_counts": {kind: counts(kind) for kind in ("gate", "up", "down")},
            "copied_tensors": len(self._copied_keys),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Local pretrained base checkpoint")
    parser.add_argument("--context-model", "--context", dest="context_model", required=True,
                        help="Already merged checkpoint to calibrate")
    parser.add_argument("--teachers", nargs="+", required=True, help="Task checkpoint directories")
    parser.add_argument("--task-names", nargs="+", help="Unique teacher names; defaults to directory names")
    parser.add_argument("--target-tasks", "--target-task", dest="target_tasks", nargs="+",
                        help="Teacher subset used for SVC; defaults to all tasks")
    parser.add_argument("--output", required=True, help="Output HF checkpoint")
    parser.add_argument("--svc-alpha", type=float, default=1.0,
                        help="Projection coefficient floor in (0, 1]; 1 suppresses only (default: 1)")
    parser.add_argument("--svc-eps", type=float, default=1e-12,
                        help="Positive projection-energy denominator floor (default: 1e-12)")
    parser.add_argument("--device", default="auto", help="CUDA device; auto selects cuda:0 (no CPU fallback)")
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output shard size (default: 5GB)")
    parser.add_argument("--processor-source", help="Tokenizer/processor source; defaults to context model")
    parser.add_argument("--save-processor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_svc_options(args.svc_alpha, args.svc_eps)
    _parse_size_bytes(args.max_shard_size)
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("SVC requires a CUDA --device; no CPU fallback")
    torch.cuda.get_device_properties(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    base_dir, context_dir = Path(args.base), Path(args.context_model)
    teacher_dirs = [Path(path) for path in args.teachers]
    task_names = _resolve_task_names(teacher_dirs, args.task_names)
    target_tasks = _resolve_target_tasks(task_names, args.target_tasks)
    teachers_by_name = dict(zip(task_names, teacher_dirs))
    output_dir = Path(args.output)
    processor_source = Path(args.processor_source) if args.processor_source else context_dir
    if output_dir.resolve() in {base_dir.resolve(), context_dir.resolve(),
                                *(path.resolve() for path in teacher_dirs), processor_source.resolve()}:
        raise ValueError("Output directory must differ from all input model/processor directories")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it.")
    if args.save_processor and not processor_source.is_dir():
        raise FileNotFoundError(f"Processor source directory not found: {processor_source}")

    print(f"SVC device={device}, dtype=float32, alpha={args.svc_alpha}, target_tasks={target_tasks}", flush=True)
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorCheckpoint(base_dir))
        context = stack.enter_context(SafetensorCheckpoint(context_dir))
        teachers = {task: stack.enter_context(SafetensorCheckpoint(teachers_by_name[task]))
                    for task in target_tasks}
        state = SVCCalibratedCheckpoint(base, context, teachers, device=device,
                                        alpha=args.svc_alpha, eps=args.svc_eps)
        # All input metadata and output options are checked before replacement begins.
        (output_dir / "svc_complete.json").unlink(missing_ok=True)
        _prepare_output_dir(output_dir, args.overwrite)
        (output_dir / "svc_summary.json").unlink(missing_ok=True)
        save_selected_checkpoint(
            state, expert_dir=context_dir, processor_source=processor_source,
            output_dir=output_dir, max_shard_size=args.max_shard_size,
            save_processor=args.save_processor, trust_remote_code=args.trust_remote_code,
        )
        report = state.summary()
        with (output_dir / "svc_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    with (output_dir / "svc_complete.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "method": "svc_ablation",
                   "matrix_counts": report["matrix_counts"]}, handle, indent=2)
        handle.write("\n")
    print(f"Done. SVC calibrated checkpoint -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
