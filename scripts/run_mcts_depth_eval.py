#!/usr/bin/env python3
"""Checkpoint-parameterized runner for the MCTS depth/throughput/strength study.

One invocation drives the resumable nine-stage pipeline of
``docs/mcts_depth_strength_eval_plan.md``. Its public interface is exactly what
section 7 specifies — checkpoint, worker count, shard id, seed range, output
path, inference endpoint — so the private deployment repo only has to wrap it in
a Job.

    python scripts/run_mcts_depth_eval.py \
        --checkpoint /shared/.../transformer-policy.pt \
        --showdown-root /opt/pokemon-showdown \
        --out-root /shared/mcts-depth-eval/run-1 \
        --depths 2,4 --sims 512,1024 --corpus-decisions 64

Stages reuse completed work on restart; nothing is held in memory across a
restart. ``--stage-through`` stops after a named stage so a probe-scale
rehearsal can publish a report without running the full matrix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from pokezero.mcts_eval import (  # noqa: E402
    MatrixManifest,
    ResourceProfile,
    Stage,
    TerminalFailure,
    default_lattice,
    resolve_checkpoint_contract,
)
from pokezero.mcts_eval.controller import (  # noqa: E402
    run_pipeline,
    stage_dir,
    write_status,
)
from pokezero.mcts_eval.manifest import SearchConfig  # noqa: E402
from pokezero.mcts_eval.report import TimingRow, render_report, select_candidates  # noqa: E402


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)
    return str(path)


def _int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", default=None, help="Terminal mismatch guard.")
    parser.add_argument("--showdown-root", default=os.environ.get("POKEZERO_SHOWDOWN_ROOT"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--depths", type=_int_list, default=(2, 4, 6, 8, 10))
    parser.add_argument("--sims", type=_int_list, default=(512, 1024, 2048, 4096, 8192))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--corpus-decisions", type=int, default=256)
    parser.add_argument("--seed-band", default="default")
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--inference-endpoint", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Lattice worker PROCESSES. Cells are dealt round-robin across workers, so N "
            "workers share M GPUs instead of reserving one GPU per cell."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help=(
            "Physical GPUs available to this run. Worker i is pinned to GPU i%%gpus via "
            "CUDA_VISIBLE_DEVICES (the crate always asks for cuda:0, so the mapping is what "
            "spreads and shares devices). 0 keeps the run on CPU."
        ),
    )
    parser.add_argument(
        "--stage-through",
        default=None,
        choices=[stage.value for stage in Stage],
        help="Stop after this stage (probe-scale rehearsal).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- identity: derived from the checkpoint itself, never assumed --------
    contract = resolve_checkpoint_contract(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
        model_device=args.model_device,
        showdown_root=args.showdown_root,
    )
    configs = tuple(
        SearchConfig(depth=d, sims=s, batch=args.batch, worlds=args.worlds)
        for d, s in ((d, s) for d in sorted(set(args.depths)) for s in sorted(set(args.sims)))
    )
    manifest = MatrixManifest(
        checkpoint_manifest=contract.to_manifest(),
        configs=configs,
        resource_profile=ResourceProfile(
            concurrency=args.concurrency,
            torch_threads=args.torch_threads,
            inference_endpoint=args.inference_endpoint,
        ),
        worlds=args.worlds,
        seed_band=args.seed_band,
        corpus_decisions=args.corpus_decisions,
    )
    experiment_id = manifest.experiment_id
    execution_id = manifest.execution_id
    print(f"experiment_id={experiment_id[:16]}… execution_id={execution_id[:16]}…", flush=True)
    print(f"matrix: {len(configs)} cells, corpus {args.corpus_decisions} decisions", flush=True)

    stop_after = Stage(args.stage_through) if args.stage_through else None
    handlers: dict[Stage, Any] = {}

    def stage_handler(stage: Stage):
        def register(fn):
            def wrapped(directory: Path):
                started = time.time()
                artifacts = fn(directory)
                print(f"[{stage.value}] {time.time() - started:.1f}s", flush=True)
                return artifacts

            handlers[stage] = wrapped
            return fn

        return register

    @stage_handler(Stage.MATERIALIZE_CHECKPOINT)
    def _materialize(directory: Path) -> list[str]:
        """Export the TorchScript artifact + encoder tables keyed by reuse key."""
        from pokezero.mcts_eval.resolver import export_reuse_key

        payload = {
            "contract": contract.to_manifest(),
            "export_reuse_key": export_reuse_key(contract),
            "manifest": manifest.to_payload(),
        }
        return [_write_json(directory / "materialization.json", payload)]

    @stage_handler(Stage.VALIDATE_CONTRACT)
    def _validate(directory: Path) -> list[str]:
        """Re-derive the contract and fail terminally on any drift."""
        again = resolve_checkpoint_contract(
            args.checkpoint, expected_sha256=contract.checkpoint_sha256,
            model_device=args.model_device, showdown_root=args.showdown_root,
        )
        if again.observation_contract_sha256 != contract.observation_contract_sha256:
            raise TerminalFailure("observation contract changed between stages")
        return [
            _write_json(
                directory / "validation.json",
                {
                    "observation_contract_sha256": again.observation_contract_sha256,
                    "checkpoint_sha256": again.checkpoint_sha256,
                    "policy_id": again.policy_id,
                    "supported": True,
                },
            )
        ]

    @stage_handler(Stage.MECHANICS_SMOKE)
    def _smoke(directory: Path) -> list[str]:
        """Crate import + model load + finite priors/values on one encode."""
        result: dict[str, Any] = {"crate_import": False, "native_leaf_model": False}
        try:
            import pokezero_search  # noqa: PLC0415

            result["crate_import"] = True
            result["native_leaf_model"] = hasattr(pokezero_search, "NativeLeafModel")
        except ImportError as error:
            raise TerminalFailure(
                f"native search crate unavailable ({error}); the study image must carry the "
                "model-enabled pokezero-search build."
            ) from error
        if not result["native_leaf_model"]:
            raise TerminalFailure("pokezero_search lacks NativeLeafModel (built without `model`).")
        return [_write_json(directory / "smoke.json", result)]

    @stage_handler(Stage.BUILD_TIMING_CORPUS)
    def _corpus(directory: Path) -> list[str]:
        from pokezero.mcts_eval.timing_corpus import read_corpus

        target = directory / "timing-corpus.jsonl"
        if target.is_file():
            corpus_manifest, records = read_corpus(target)  # fails closed on drift
            return [str(target)]
        raise TerminalFailure(
            f"timing corpus absent at {target}. Build it from held-out FoulPlay games with "
            "pokezero.mcts_eval.timing_corpus.build_corpus (plan A2) and re-run; the runner "
            "will not silently substitute a different decision set."
        )

    @stage_handler(Stage.RUN_TIMING_LATTICE)
    def _lattice(directory: Path) -> list[str]:
        from pokezero.mcts_eval.timing_corpus import read_corpus

        corpus_path = stage_dir(out_root, Stage.BUILD_TIMING_CORPUS) / "timing-corpus.jsonl"
        _, records = read_corpus(corpus_path)

        if args.workers > 1:
            return _run_lattice_workers(directory)

        from pokezero.mcts_eval.lattice import time_lattice_cell

        rows = []
        for index, config in enumerate(configs):
            if index % args.shards != args.shard_id:
                continue
            target = directory / f"timing-{config.config_id}.json"
            if target.is_file():  # a resumed worker never re-times a finished cell
                rows.append(json.loads(target.read_text()))
                continue
            row = time_lattice_cell(
                config,
                records=records,
                contract=contract,
                showdown_root=args.showdown_root,
            )
            rows.append(row.to_payload())
            _write_json(target, row.to_payload())
        return [_write_json(directory / f"timing-shard-{args.shard_id}.json", rows)]

    def _run_lattice_workers(directory: Path) -> list[str]:
        """Fan the lattice across worker processes packed onto the available GPUs.

        The crate always requests ``cuda:0``, so device spreading is done with
        CUDA_VISIBLE_DEVICES: worker i sees GPU ``i % gpus`` as its only device.
        Several workers may share one GPU — search is not GPU-saturating at these
        batch sizes, so packing raises utilization instead of reserving a device
        per cell. Each worker writes its own per-cell artifacts, so a worker that
        dies is resumed without re-timing completed cells.
        """
        import subprocess

        procs = []
        for worker in range(args.workers):
            env = dict(os.environ)
            if args.gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(worker % args.gpus)
            # Keep per-process CPU use bounded so packed workers do not thrash.
            env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
            env.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--checkpoint", args.checkpoint,
                "--out-root", str(out_root),
                "--model-device", args.model_device,
                "--depths", ",".join(str(d) for d in args.depths),
                "--sims", ",".join(str(s) for s in args.sims),
                "--batch", str(args.batch), "--worlds", str(args.worlds),
                "--corpus-decisions", str(args.corpus_decisions),
                "--concurrency", str(args.concurrency),
                "--torch-threads", str(args.torch_threads),
                "--shards", str(args.workers), "--shard-id", str(worker),
                "--stage-through", Stage.RUN_TIMING_LATTICE.value,
                "--workers", "1",
            ]
            if args.showdown_root:
                command += ["--showdown-root", args.showdown_root]
            procs.append((worker, subprocess.Popen(command, env=env)))
        failures = [worker for worker, proc in procs if proc.wait() != 0]
        if failures:
            raise TerminalFailure(f"lattice worker(s) {failures} failed; see their stage artifacts")
        shard_files = sorted(str(p) for p in directory.glob("timing-shard-*.json"))
        return shard_files or [_write_json(directory / "timing-shard-merged.json", [])]

    @stage_handler(Stage.MERGE_AND_SELECT)
    def _select(directory: Path) -> list[str]:
        lattice_dir = stage_dir(out_root, Stage.RUN_TIMING_LATTICE)
        rows = [
            TimingRow(**{k: v for k, v in json.loads(path.read_text()).items() if k != "eligible"})
            for path in sorted(lattice_dir.glob("timing-d*.json"))
        ]
        candidates = select_candidates(rows)
        return [
            _write_json(
                directory / "candidates.json",
                {
                    "timing_rows": [row.to_payload() for row in rows],
                    "selected": [row.config_id for row in candidates],
                },
            )
        ]

    @stage_handler(Stage.PUBLISH_REPORT)
    def _publish(directory: Path) -> list[str]:
        lattice_dir = stage_dir(out_root, Stage.RUN_TIMING_LATTICE)
        rows = [
            TimingRow(**{k: v for k, v in json.loads(path.read_text()).items() if k != "eligible"})
            for path in sorted(lattice_dir.glob("timing-d*.json"))
        ]
        payload = render_report([], timing_rows=rows, manifest_payload=manifest.to_payload())
        return [_write_json(directory / "report.json", payload)]

    if stop_after is not None:
        allowed = []
        for stage in Stage:
            allowed.append(stage)
            if stage is stop_after:
                break
        handlers = {stage: fn for stage, fn in handlers.items() if stage in allowed}

    try:
        status = run_pipeline(
            out_root,
            experiment_id=experiment_id,
            execution_id=execution_id,
            handlers=handlers,
            max_retries=args.max_retries,
        )
    except TerminalFailure as error:
        print(f"TERMINAL: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: status[k] for k in ("state", "stage", "experiment_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
