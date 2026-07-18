#!/usr/bin/env python3
"""Benchmark checkpoint inference across runtimes for the native search crate.

Measures forward-pass throughput (evals/s = batch rows per second) for:
  eager-cpu     eager PyTorch on CPU (the model-cost-ladder baseline)
  ts-cpu        TorchScript trace on CPU
  ort-cpu       ONNX Runtime CPU EP (intra_op thread count tuned per batch)
  eager-mps     eager PyTorch on Apple MPS
  ts-mps        TorchScript re-traced on MPS (traces bake device constants)
  ort-coreml    ONNX Runtime CoreML EP (reported gracefully when unavailable)
  eager-mps-fp16  eager on MPS under torch.autocast(float16) — cheap fp16 probe

Inputs are dummy-filled at checkpoint shapes (timing is encoding-independent,
same methodology as docs/engine_search_poc.md's model-cost ladder).

Usage:
    python scripts/export_model.py --checkpoint ckpt.pt --out-dir exports/
    python scripts/bench_inference.py --checkpoint ckpt.pt --export-dir exports/ \
        --batch-sizes 1,16,64,256 --out bench.md
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from export_model import INPUT_NAMES, build_exportable_module, make_random_inputs

DEFAULT_BATCH_SIZES = (1, 16, 64, 256)
DEFAULT_INTRA_OP_CANDIDATES = (1, 2, 4, 8)


@dataclass
class BenchResult:
    runtime: str
    batch_size: int
    evals_per_s: float
    ms_per_batch: float
    note: str = ""


def _time_forward(
    run_once: Callable[[], None],
    *,
    warmup: int,
    min_time_s: float,
    min_iters: int,
    synchronize: Callable[[], None] | None = None,
) -> float:
    """Seconds per call, median-free steady-state estimate (total/iters)."""

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


def _torch_runner(module: Any, inputs: tuple[Any, ...]) -> Callable[[], None]:
    import torch

    def run_once() -> None:
        with torch.no_grad():
            module(*inputs)

    return run_once


def bench_torch(
    module: Any,
    runtime: str,
    config: Any,
    batch_sizes: list[int],
    *,
    device: str,
    seed: int,
    warmup: int,
    min_time_s: float,
    min_iters: int,
    autocast_fp16: bool = False,
) -> list[BenchResult]:
    import torch

    synchronize = None
    if device == "mps":
        synchronize = torch.mps.synchronize
    results = []
    for batch_size in batch_sizes:
        inputs = make_random_inputs(config, batch_size, seed=seed, device=device)
        if autocast_fp16:
            base_runner = _torch_runner(module, inputs)

            def run_once(base_runner: Callable[[], None] = base_runner) -> None:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    base_runner()

        else:
            run_once = _torch_runner(module, inputs)
        seconds = _time_forward(
            run_once, warmup=warmup, min_time_s=min_time_s, min_iters=min_iters, synchronize=synchronize
        )
        results.append(
            BenchResult(runtime, batch_size, batch_size / seconds, seconds * 1e3)
        )
        print(f"  {runtime} batch={batch_size}: {batch_size / seconds:,.0f} evals/s", file=sys.stderr)
    return results


def bench_ort(
    onnx_path: Path,
    runtime: str,
    config: Any,
    batch_sizes: list[int],
    *,
    providers: list[Any],
    intra_op_candidates: tuple[int, ...],
    seed: int,
    warmup: int,
    min_time_s: float,
    min_iters: int,
) -> list[BenchResult]:
    import onnxruntime as ort

    sessions: dict[int, Any] = {}
    for threads in intra_op_candidates:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        sessions[threads] = ort.InferenceSession(str(onnx_path), options, providers=providers)

    results = []
    for batch_size in batch_sizes:
        inputs = make_random_inputs(config, batch_size, seed=seed)
        feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, inputs)}
        best: tuple[float, int] | None = None
        for threads, session in sessions.items():
            def run_once(session: Any = session, feeds: dict[str, Any] = feeds) -> None:
                session.run(None, feeds)

            seconds = _time_forward(run_once, warmup=warmup, min_time_s=min_time_s, min_iters=min_iters)
            if best is None or seconds < best[0]:
                best = (seconds, threads)
        assert best is not None
        seconds, threads = best
        results.append(
            BenchResult(
                runtime,
                batch_size,
                batch_size / seconds,
                seconds * 1e3,
                note=f"intra_op={threads}",
            )
        )
        print(
            f"  {runtime} batch={batch_size}: {batch_size / seconds:,.0f} evals/s (intra_op={threads})",
            file=sys.stderr,
        )
    return results


def render_markdown(results: list[BenchResult], batch_sizes: list[int]) -> str:
    runtimes: list[str] = []
    for result in results:
        if result.runtime not in runtimes:
            runtimes.append(result.runtime)
    by_key = {(result.runtime, result.batch_size): result for result in results}
    header = "| Runtime | " + " | ".join(f"batch {size}" for size in batch_sizes) + " | Note |"
    divider = "|---" * (len(batch_sizes) + 2) + "|"
    lines = [header, divider]
    for runtime in runtimes:
        cells = []
        notes = []
        for size in batch_sizes:
            result = by_key.get((runtime, size))
            if result is None:
                cells.append("-")
                continue
            cells.append(f"{result.evals_per_s:,.0f}/s ({result.ms_per_batch:.1f}ms)")
            if result.note and result.note not in notes:
                notes.append(result.note)
        lines.append(f"| {runtime} | " + " | ".join(cells) + " | " + ("; ".join(notes) or "") + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", default=None, help="Directory holding model_ts.pt / model.onnx from export_model.py; TS/ORT rows are skipped when absent.")
    parser.add_argument("--batch-sizes", default=",".join(str(size) for size in DEFAULT_BATCH_SIZES))
    parser.add_argument("--intra-op-threads", default=",".join(str(count) for count in DEFAULT_INTRA_OP_CANDIDATES), help="Candidate ORT intra_op thread counts; the best per batch is reported.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=1.5, help="Minimum measured seconds per (runtime, batch) cell.")
    parser.add_argument("--min-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--skip-mps", action="store_true")
    parser.add_argument("--out", default=None, help="Also write the markdown table to this path.")
    args = parser.parse_args(argv)

    import torch

    from pokezero.neural_policy import load_transformer_checkpoint

    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    intra_op_candidates = tuple(int(item) for item in args.intra_op_threads.split(",") if item.strip())

    model, result = load_transformer_checkpoint(args.checkpoint, map_location="cpu")
    config = result.model_config
    model.eval()
    shim = build_exportable_module(model)

    export_dir = Path(args.export_dir) if args.export_dir else None
    ts_path = export_dir / "model_ts.pt" if export_dir else None
    onnx_path = export_dir / "model.onnx" if export_dir else None

    timing = dict(seed=args.seed, warmup=args.warmup, min_time_s=args.min_time, min_iters=args.min_iters)
    results: list[BenchResult] = []
    skipped: list[str] = []

    print("eager-cpu", file=sys.stderr)
    results += bench_torch(shim, "eager-cpu", config, batch_sizes, device="cpu", **timing)

    if ts_path is not None and ts_path.exists():
        traced_cpu = torch.jit.load(str(ts_path), map_location="cpu")
        print("ts-cpu", file=sys.stderr)
        results += bench_torch(traced_cpu, "ts-cpu", config, batch_sizes, device="cpu", **timing)
    else:
        skipped.append("ts-cpu: no model_ts.pt in --export-dir")

    if onnx_path is not None and onnx_path.exists():
        print("ort-cpu", file=sys.stderr)
        results += bench_ort(
            onnx_path,
            "ort-cpu",
            config,
            batch_sizes,
            providers=["CPUExecutionProvider"],
            intra_op_candidates=intra_op_candidates,
            **timing,
        )
    else:
        skipped.append("ort-cpu: no model.onnx in --export-dir")

    mps_available = (not args.skip_mps) and torch.backends.mps.is_available()
    if mps_available:
        model_mps = model.to("mps")
        shim_mps = build_exportable_module(model_mps)
        print("eager-mps", file=sys.stderr)
        results += bench_torch(shim_mps, "eager-mps", config, batch_sizes, device="mps", **timing)
        try:
            print("eager-mps-fp16", file=sys.stderr)
            results += bench_torch(
                shim_mps, "eager-mps-fp16", config, batch_sizes, device="mps", autocast_fp16=True, **timing
            )
        except Exception as error:  # noqa: BLE001 - fp16 probe is best-effort.
            skipped.append(f"eager-mps-fp16: {error}")
        # torch.jit.trace bakes device constants (e.g. the history-position
        # arange) into the graph, so the CPU artifact cannot be relocated to
        # MPS via map_location — a TorchScript-on-MPS deployment needs a
        # trace taken on MPS. Re-trace here to price that path.
        try:
            import warnings

            example_mps = make_random_inputs(config, 4, seed=args.seed, device="mps")
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traced_mps = torch.jit.trace(shim_mps, example_mps)
            print("ts-mps", file=sys.stderr)
            results += bench_torch(traced_mps, "ts-mps", config, batch_sizes, device="mps", **timing)
        except Exception as error:  # noqa: BLE001 - report, don't abort the table.
            skipped.append(f"ts-mps: {error}")
        model.to("cpu")
    else:
        skipped.append("eager-mps/ts-mps: MPS unavailable or --skip-mps")

    if onnx_path is not None and onnx_path.exists():
        try:
            import onnxruntime as ort

            if "CoreMLExecutionProvider" in ort.get_available_providers():
                print("ort-coreml", file=sys.stderr)
                results += bench_ort(
                    onnx_path,
                    "ort-coreml",
                    config,
                    batch_sizes,
                    providers=[
                        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                        "CPUExecutionProvider",
                    ],
                    intra_op_candidates=intra_op_candidates[:1],
                    **timing,
                )
            else:
                skipped.append("ort-coreml: CoreMLExecutionProvider not in this onnxruntime build")
        except Exception as error:  # noqa: BLE001 - CoreML EP is best-effort.
            skipped.append(f"ort-coreml: {error}")

    table = render_markdown(results, batch_sizes)
    print()
    print(table)
    if skipped:
        print()
        for item in skipped:
            print(f"skipped {item}")
    if args.out:
        out_path = Path(args.out)
        body = table + ("\n\n" + "\n".join(f"skipped {item}" for item in skipped) if skipped else "") + "\n"
        out_path.write_text(body)
        print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
