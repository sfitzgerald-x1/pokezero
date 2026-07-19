#!/usr/bin/env python3
"""Model-eval throughput grid for real checkpoints (engine-swap stream).

Prices batched leaf evaluation for the native search crate across
device x dtype x batch on ONE machine, in a form that reruns on a cluster
GPU as a config change (--devices cuda --dtypes fp32,fp16). Everything is
TorchScript (the crate's runtime, bit-exact vs eager per
docs/model_export_findings.md): each (device, dtype) cell traces the
checkpoint on that device in-process — identical to the shipped
scripts/export_model.py artifact, without needing one per dtype.

Measured per cell (steady state, warmup + trace excluded):
  evals/s      batch rows per wall second
  ms/batch     per-forward latency at that batch shape
  ms/eval      ms/batch / batch

Also measured on request (--partial-study): the MPS partial-final-batch
regression documented in docs/crate_model_integration.md — a search round
whose leftover (< batch) row count changes the input shape forces MPS to
compile a new shape-specialized graph. The study times never-seen partial
shapes cold, the same shapes warm, and the same rows run through
``pad_obs_batch`` (the documented-but-unbuilt fix: pad to the fixed batch
shape, slice the padded rows off the outputs), and checks the padded
outputs match the unpadded forward.

fp16 is a throughput probe only: fp16 outputs have no parity story
(model_export_findings.md corrections) and must not feed collection.
On CUDA, fp16 is a REAL half-precision weight copy (model.half(), half
inputs) traced to TorchScript. On MPS, a real half copy CANNOT run in this
torch build — both the traced and the eager half forward hard-abort in
Metal ("Destination NDArray and Accumulator NDArray cannot have different
datatype in MPSNDArrayMatrixMultiplication", torch 2.12.1 / macOS 25.5) —
so --fp16-runtime auto falls back to EAGER AUTOCAST on MPS (fp32 weights,
per-op fp16 cast: the same probe docs/model_export_findings.md measured).

Usage:
    python scripts/bench_model_eval.py \
        --checkpoint checkpoints/curated/<ckpt>.pt --label emeta-final \
        --devices cpu,mps --batch-sizes 1,16,64,128,256,512 \
        --partial-study 256 --out-json bench.json --out-md bench.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from export_model import TRACE_BATCH, build_exportable_module, make_random_inputs

DEFAULT_BATCH_SIZES = (1, 16, 64, 128, 256, 512)


def pad_obs_batch(inputs: tuple[Any, ...], target_batch: int) -> tuple[tuple[Any, ...], int]:
    """Pad a (batch, ...) observation-input tuple up to ``target_batch`` rows.

    The fix for the MPS partial-final-batch recompile regression
    (docs/crate_model_integration.md): keep every forward at ONE batch shape
    by repeating the final row into the padding slots, then slice outputs
    back to the real row count (``outputs[:real_rows]``). Row-independent
    forwards (this model attends within a row only) make the padding inert;
    repeating a real row (rather than zeros) keeps every embedding lookup
    in-vocabulary so no clamp path activates.

    Returns (padded_inputs, real_rows). No-op when already at target size.
    """

    import torch

    real_rows = int(inputs[0].shape[0])
    if real_rows > target_batch:
        raise ValueError(f"batch {real_rows} exceeds pad target {target_batch}")
    if real_rows == target_batch:
        return inputs, real_rows
    pad_rows = target_batch - real_rows
    padded = []
    for tensor in inputs:
        filler = tensor[-1:].expand(pad_rows, *tensor.shape[1:])
        padded.append(torch.cat([tensor, filler], dim=0).contiguous())
    return tuple(padded), real_rows


@dataclass
class Cell:
    label: str
    device: str
    dtype: str
    batch: int
    evals_per_s: float
    ms_per_batch: float
    ms_per_eval: float
    runtime: str = "ts"


@dataclass
class PartialStudy:
    label: str
    device: str
    dtype: str
    batch: int
    steady_ms_per_batch: float
    partial_sizes: list[int]
    cold_ms: list[float]
    warm_ms: list[float]
    padded_first_ms: list[float]
    padded_second_ms: list[float]
    pad_output_max_abs_diff: float
    notes: list[str] = field(default_factory=list)


class _AutocastForward:
    """Eager forward under torch.autocast — the only fp16 path MPS can run."""

    def __init__(self, module: Any, device: str) -> None:
        self.module = module
        self.device = device

    def __call__(self, *inputs: Any) -> Any:
        import torch

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            return self.module(*inputs)


def _sync_fn(device: str) -> Callable[[], None] | None:
    import torch

    if device == "mps":
        return torch.mps.synchronize
    if device == "cuda":
        return torch.cuda.synchronize
    return None


def _steady_seconds(
    run_once: Callable[[], None],
    *,
    warmup: int,
    min_time_s: float,
    min_iters: int,
    synchronize: Callable[[], None] | None,
) -> float:
    for _ in range(warmup):
        run_once()
    if synchronize is not None:
        synchronize()
    iterations = 0
    start = time.perf_counter()
    while True:
        run_once()
        iterations += 1
        if synchronize is not None:
            synchronize()
        elapsed = time.perf_counter() - start
        if iterations >= min_iters and elapsed >= min_time_s:
            return elapsed / iterations


def _trace_on(shim: Any, config: Any, device: str, dtype: str, seed: int) -> Any:
    import torch

    example = make_random_inputs(config, TRACE_BATCH, seed=seed, device=device)
    if dtype == "fp16":
        example = _cast_fp16(example)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.jit.trace(shim, example)


def _cast_fp16(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    import torch

    return tuple(
        tensor.half() if tensor.dtype in (torch.float32, torch.float64) else tensor
        for tensor in inputs
    )


def _forward_ms(module: Any, inputs: tuple[Any, ...], synchronize: Callable[[], None] | None) -> float:
    import torch

    if synchronize is not None:
        synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        module(*inputs)
    if synchronize is not None:
        synchronize()
    return (time.perf_counter() - start) * 1e3


def run_partial_study(
    module: Any,
    config: Any,
    *,
    label: str,
    device: str,
    dtype: str,
    batch: int,
    seed: int,
    steady_ms: float,
    cast_half: bool = False,
) -> PartialStudy:
    """Cold/warm partial-shape latency vs pad-to-batch at fixed shape."""

    import torch

    synchronize = _sync_fn(device)
    # Distinct leftover sizes a search loop would produce (sims % batch),
    # chosen off the benched-batch grid so every shape is genuinely new.
    fractions = (0.97, 0.73, 0.51, 0.37, 0.23, 0.11, 0.05)
    sizes = sorted({max(1, int(batch * f)) for f in fractions} - {batch}, reverse=True)

    per_size_inputs = {}
    for size in sizes:
        inputs = make_random_inputs(config, size, seed=seed + size, device=device)
        per_size_inputs[size] = _cast_fp16(inputs) if cast_half else inputs

    cold = [_forward_ms(module, per_size_inputs[size], synchronize) for size in sizes]
    warm = [_forward_ms(module, per_size_inputs[size], synchronize) for size in sizes]

    padded_first: list[float] = []
    padded_second: list[float] = []
    max_diff = 0.0
    for size in sizes:
        inputs = per_size_inputs[size]
        padded, real_rows = pad_obs_batch(inputs, batch)
        padded_first.append(_forward_ms(module, padded, synchronize))
        padded_second.append(_forward_ms(module, padded, synchronize))
        with torch.no_grad():
            direct = module(*inputs)
            via_pad = module(*padded)
        for reference, candidate in zip(direct[:2], via_pad[:2]):  # policy_logits, value
            diff = (reference - candidate[:real_rows]).abs().max().item()
            max_diff = max(max_diff, diff)

    return PartialStudy(
        label=label,
        device=device,
        dtype=dtype,
        batch=batch,
        steady_ms_per_batch=steady_ms,
        partial_sizes=sizes,
        cold_ms=[round(v, 3) for v in cold],
        warm_ms=[round(v, 3) for v in warm],
        padded_first_ms=[round(v, 3) for v in padded_first],
        padded_second_ms=[round(v, 3) for v in padded_second],
        pad_output_max_abs_diff=max_diff,
    )


def render_markdown(cells: list[Cell], batch_sizes: list[int]) -> str:
    keys: list[tuple[str, str, str]] = []
    for cell in cells:
        key = (cell.label, cell.device, cell.dtype)
        if key not in keys:
            keys.append(key)
    by_cell = {(cell.label, cell.device, cell.dtype, cell.batch): cell for cell in cells}
    header = "| model | device/dtype | " + " | ".join(f"batch {size}" for size in batch_sizes) + " |"
    divider = "|---" * (len(batch_sizes) + 2) + "|"
    lines = [header, divider]
    for label, device, dtype in keys:
        row = [label, f"{device}/{dtype}"]
        for size in batch_sizes:
            cell = by_cell.get((label, device, dtype, size))
            row.append("-" if cell is None else f"{cell.evals_per_s:,.0f}/s ({cell.ms_per_batch:.1f}ms)")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="Row label, e.g. emeta-final or m50-latest.")
    parser.add_argument("--devices", default="cpu,mps", help="Comma subset of {cpu,mps,cuda}; unavailable devices are skipped with a note.")
    parser.add_argument("--dtypes", default="fp32,fp16", help="Comma subset of {fp32,fp16}; fp16 runs on mps/cuda only.")
    parser.add_argument("--batch-sizes", default=",".join(str(size) for size in DEFAULT_BATCH_SIZES))
    parser.add_argument("--partial-study", type=int, default=0, metavar="BATCH", help="Run the partial-final-batch study at this batch size on non-cpu devices (0 = off).")
    parser.add_argument(
        "--fp16-runtime",
        choices=("auto", "ts", "eager", "autocast"),
        default="auto",
        help="ts/eager = real half weight copy (traced/eager); autocast = eager autocast over fp32 "
        "weights. auto = autocast on MPS (real half copies hard-abort in Metal), ts elsewhere.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=1.5)
    parser.add_argument("--min-iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)

    import torch

    from pokezero.neural_policy import load_transformer_checkpoint

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]

    model, result = load_transformer_checkpoint(args.checkpoint, map_location="cpu")
    config = result.model_config
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{args.label}: {parameters / 1e6:.1f}M params, d={config.embedding_dim}, layers={config.transformer_layers}", file=sys.stderr)

    cells: list[Cell] = []
    studies: list[PartialStudy] = []
    skipped: list[str] = []
    timing = dict(warmup=args.warmup, min_time_s=args.min_time, min_iters=args.min_iters)

    for device in devices:
        if device == "mps" and not torch.backends.mps.is_available():
            skipped.append("mps unavailable")
            continue
        if device == "cuda" and not torch.cuda.is_available():
            skipped.append("cuda unavailable")
            continue
        synchronize = _sync_fn(device)
        for dtype in dtypes:
            if dtype == "fp16" and device == "cpu":
                continue
            fp16_runtime = args.fp16_runtime
            if fp16_runtime == "auto":
                fp16_runtime = "autocast" if device == "mps" else "ts"
            uses_shared_model = dtype == "fp32" or fp16_runtime == "autocast"
            cast_half = dtype == "fp16" and fp16_runtime != "autocast"
            if dtype == "fp16" and fp16_runtime != "autocast":
                # A REAL half-precision copy; deepcopy keeps the base fp32
                # weights pristine for every other cell (half() is lossy).
                import copy

                device_model = copy.deepcopy(model).to(device).half()
            else:
                device_model = model.to(device)
            shim = build_exportable_module(device_model)
            if dtype == "fp16" and fp16_runtime == "autocast":
                runner: Any = _AutocastForward(shim, device)
                runtime_label = "eager-autocast"
            elif dtype == "fp16" and fp16_runtime == "eager":
                runner = shim
                runtime_label = "eager-half"
            else:
                runner = _trace_on(shim, config, device, dtype if cast_half else "fp32", args.seed)
                runtime_label = "ts-half" if cast_half else "ts"
            for batch in batch_sizes:
                inputs = make_random_inputs(config, batch, seed=args.seed + batch, device=device)
                if cast_half:
                    inputs = _cast_fp16(inputs)

                def run_once(runner: Any = runner, inputs: tuple[Any, ...] = inputs) -> None:
                    with torch.no_grad():
                        runner(*inputs)

                seconds = _steady_seconds(run_once, synchronize=synchronize, **timing)
                cell = Cell(
                    label=args.label,
                    device=device,
                    dtype=dtype,
                    batch=batch,
                    evals_per_s=batch / seconds,
                    ms_per_batch=seconds * 1e3,
                    ms_per_eval=seconds * 1e3 / batch,
                    runtime=runtime_label,
                )
                cells.append(cell)
                print(f"  {device}/{dtype} batch={batch}: {cell.evals_per_s:,.0f} evals/s ({cell.ms_per_batch:.1f}ms/batch)", file=sys.stderr)

            if args.partial_study and device != "cpu":
                steady = next(
                    (cell.ms_per_batch for cell in cells
                     if (cell.label, cell.device, cell.dtype, cell.batch) == (args.label, device, dtype, args.partial_study)),
                    float("nan"),
                )
                study = run_partial_study(
                    runner, config, label=args.label, device=device, dtype=dtype,
                    batch=args.partial_study, seed=args.seed, steady_ms=steady,
                    cast_half=cast_half,
                )
                studies.append(study)
                print(
                    f"  {device}/{dtype} partial-study@{args.partial_study}: steady {steady:.1f}ms | "
                    f"cold {min(study.cold_ms):.1f}-{max(study.cold_ms):.1f}ms | "
                    f"warm {min(study.warm_ms):.1f}-{max(study.warm_ms):.1f}ms | "
                    f"padded1st {min(study.padded_first_ms):.1f}-{max(study.padded_first_ms):.1f}ms | "
                    f"padded2nd {min(study.padded_second_ms):.1f}-{max(study.padded_second_ms):.1f}ms | "
                    f"pad diff {study.pad_output_max_abs_diff:.2e}",
                    file=sys.stderr,
                )
            if uses_shared_model:
                # fp32 and autocast moved the shared module in place; return
                # it to CPU so the next device starts clean. (Real-half cells
                # used a deepcopy.)
                model = model.to("cpu")
            del device_model

    table = render_markdown(cells, batch_sizes)
    print()
    print(table)
    for item in skipped:
        print(f"skipped: {item}")

    if args.out_md:
        Path(args.out_md).write_text(table + "\n")
    if args.out_json:
        payload = {
            "label": args.label,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "parameters": parameters,
            "config": {"d": config.embedding_dim, "layers": config.transformer_layers, "window": config.window_size, "tokens": config.token_count},
            "timing": {"warmup": args.warmup, "min_time_s": args.min_time, "min_iters": args.min_iters, "seed": args.seed},
            "cells": [asdict(cell) for cell in cells],
            "partial_studies": [asdict(study) for study in studies],
            "skipped": skipped,
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"json written: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
