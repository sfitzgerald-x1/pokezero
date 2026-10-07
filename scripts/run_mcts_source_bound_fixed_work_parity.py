#!/usr/bin/env python3
"""Prove fixed-work root-action parity for serial and parallel model MCTS.

This is deliberately not a strength evaluation or deadline qualification.  It
replays the frozen public timing corpus twice per decision with identical
search work: once serially and once with independent native model workers.
The only accepted terminal result is complete, source-bound, fallback-free
root-action agreement across every corpus decision.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import run_mcts_deadline_qualification as legacy


SCHEMA_VERSION = "pokezero.mcts-source-bound-fixed-work-parity.v1"
RUNNER_PATH = "scripts/run_mcts_source_bound_fixed_work_parity.py"
EXPECTED_DECISIONS = 16


class FixedWorkParityError(RuntimeError):
    """A fixed-work parity witness is incomplete, malformed, or divergent."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-corpus-file-sha256", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--expected-showdown-source-sha256", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-device", choices=("cpu",), default="cpu")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument(
        "--model-priors",
        action="store_true",
        help="Use the candidate's own policy priors in both fixed-work arms.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("depth", "sims", "batch", "worlds", "parallel_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.parallel_workers <= 1:
        parser.error("--parallel-workers must exceed the serial worker count")
    if args.parallel_workers > args.worlds:
        parser.error("--parallel-workers must not exceed --worlds")
    return args


def _source_provenance(receipt_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind the running image, current source tree, and installed native engine."""

    receipt, active = legacy._verify_source_receipt(receipt_path)
    declared = receipt["source_files_sha256"]
    if RUNNER_PATH not in declared:
        raise FixedWorkParityError("source receipt omits the fixed-work parity runner")
    try:
        legacy.assert_fresh()
        fingerprint = legacy.compute_fingerprint()
    except BaseException as error:  # assert_fresh exits with SystemExit on stale native code.
        raise FixedWorkParityError("installed native engine failed its freshness check") from error
    if fingerprint.get("fingerprint") != receipt["engine_fingerprint"]:
        raise FixedWorkParityError("native engine fingerprint differs from immutable source receipt")
    return receipt, active, dict(fingerprint)


def _manifest(
    *,
    args: argparse.Namespace,
    receipt: Mapping[str, Any],
    active_source: Mapping[str, Any],
    native_fingerprint: Mapping[str, Any],
    corpus_manifest: Any,
    corpus_file_sha256: str,
    checkpoint_contract: Any,
    showdown_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "fixed_work_root_action_parity",
        "source_receipt": dict(receipt),
        "active_source": dict(active_source),
        "native_engine": dict(native_fingerprint),
        "corpus_path": str(Path(args.corpus).resolve()),
        "corpus_sha256": corpus_manifest.corpus_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "checkpoint": checkpoint_contract.to_manifest(),
        "showdown_source": dict(showdown_source),
        "search_config": {
            "depth": args.depth,
            "sims": args.sims,
            "batch": args.batch,
            "worlds": args.worlds,
            "early_stop": False,
            "model_priors": args.model_priors,
            "use_opponent_priors": False,
            "model_decision_time_ms": None,
            "model_native_batch_guard_ms": 0,
            "serial_workers": 1,
            "parallel_workers": args.parallel_workers,
        },
        "requirements": {
            "expected_decisions": EXPECTED_DECISIONS,
            "root_action_match": True,
            "fallbacks": 0,
            "prior_fallbacks": 0,
            "invalid_actions": 0,
            "completed_worlds": args.worlds,
        },
    }


def _terminal(*, state: str, result: Mapping[str, Any] | None = None, error: BaseException | None = None) -> dict[str, Any]:
    if state not in {"PASS", "NONPASS"}:
        raise ValueError(f"unsupported terminal state: {state}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "marker": f"FIXED_WORK_ROOT_ACTION_PARITY_{state}",
    }
    if result is not None:
        payload.update(result)
    if error is not None:
        payload.update({"error_type": type(error).__name__, "error": str(error)})
    return payload


def _arm_payload(*, record: Any, arm: str, telemetry: Mapping[str, Any], outer_wall_ms: float) -> dict[str, Any]:
    engine = telemetry.get("engine_mcts")
    return {
        "arm": arm,
        "decision_id": record.decision_id,
        "corpus_record_sha256": legacy.canonical_json_sha256(record.to_payload()),
        "root_action": telemetry.get("root_action"),
        "outer_wall_ms": round(outer_wall_ms, 3),
        "fallbacks": telemetry.get("fallbacks"),
        "prior_fallbacks": telemetry.get("prior_fallbacks"),
        "invalid_actions": telemetry.get("invalid_actions"),
        "total_iterations": telemetry.get("total_iterations"),
        "model_evals": telemetry.get("model_evals"),
        "engine_mcts": dict(engine) if isinstance(engine, Mapping) else engine,
    }


def _validate_arm(payload: Mapping[str, Any], *, record: Any, arm: str, worlds: int, workers: int) -> None:
    if payload.get("arm") != arm:
        raise FixedWorkParityError(f"{record.decision_id}: durable arm label differs")
    if payload.get("decision_id") != record.decision_id:
        raise FixedWorkParityError(f"{record.decision_id}: durable decision identity differs")
    if payload.get("corpus_record_sha256") != legacy.canonical_json_sha256(record.to_payload()):
        raise FixedWorkParityError(f"{record.decision_id}: durable corpus identity differs")
    if not isinstance(payload.get("root_action"), str) or not payload["root_action"]:
        raise FixedWorkParityError(f"{record.decision_id}: {arm} has no root action")
    for field in ("fallbacks", "prior_fallbacks", "invalid_actions"):
        if payload.get(field) != 0:
            raise FixedWorkParityError(f"{record.decision_id}: {arm} reported {field}")
    engine = payload.get("engine_mcts")
    if not isinstance(engine, Mapping):
        raise FixedWorkParityError(f"{record.decision_id}: {arm} has no engine telemetry")
    if engine.get("leaf_eval") != "model":
        raise FixedWorkParityError(f"{record.decision_id}: {arm} did not use model leaf evaluation")
    if engine.get("worlds_constructed") != worlds or engine.get("worlds_searched") != worlds:
        raise FixedWorkParityError(f"{record.decision_id}: {arm} did not complete every fixed-work world")
    parallelism = engine.get("world_parallelism")
    if arm == "serial":
        if parallelism is not None:
            raise FixedWorkParityError(f"{record.decision_id}: serial arm unexpectedly reports parallel dispatch")
    elif not isinstance(parallelism, Mapping) or parallelism.get("workers") != workers or parallelism.get("mode") != "fixed_work":
        raise FixedWorkParityError(f"{record.decision_id}: parallel arm lacks fixed-work dispatch receipt")
    if not isinstance(payload.get("total_iterations"), int) or payload["total_iterations"] <= 0:
        raise FixedWorkParityError(f"{record.decision_id}: {arm} has no completed fixed-work iterations")


def _validate_row(payload: Mapping[str, Any], *, record: Any, args: argparse.Namespace) -> None:
    serial = payload.get("serial")
    parallel = payload.get("parallel")
    if not isinstance(serial, Mapping) or not isinstance(parallel, Mapping):
        raise FixedWorkParityError(f"{record.decision_id}: durable row has no complete arm pair")
    _validate_arm(serial, record=record, arm="serial", worlds=args.worlds, workers=1)
    _validate_arm(parallel, record=record, arm="parallel", worlds=args.worlds, workers=args.parallel_workers)
    if serial["root_action"] != parallel["root_action"]:
        raise FixedWorkParityError(
            f"{record.decision_id}: fixed-work root actions diverged ({serial['root_action']} != {parallel['root_action']})"
        )
    for field in ("total_iterations", "model_evals"):
        if serial.get(field) != parallel.get(field):
            raise FixedWorkParityError(f"{record.decision_id}: fixed-work {field} differs by dispatch mode")
    if _semantic_engine_metadata(serial["engine_mcts"]) != _semantic_engine_metadata(parallel["engine_mcts"]):
        raise FixedWorkParityError(
            f"{record.decision_id}: fixed-work semantic engine metadata differs by dispatch mode"
        )


def _semantic_engine_metadata(value: object) -> dict[str, Any]:
    """Keep every stable public tree witness while removing dispatch mechanics.

    The two arms intentionally differ in how native model requests are
    scheduled.  That receipt is useful evidence that the parallel arm ran, but
    it cannot be part of semantic parity.  Everything else emitted under the
    decision's public ``engine_mcts`` metadata is compared exactly: in
    particular the folded root-choice surface, completed-world count and
    early-stop witness.  This makes an unchanged selected action insufficient
    to pass if parallel dispatch changed the observed root state.
    """

    if not isinstance(value, Mapping):
        raise FixedWorkParityError("fixed-work arm has malformed engine metadata")
    return {
        str(key): child
        for key, child in value.items()
        if str(key) != "world_parallelism"
    }


def _new_decider(contract: Any, args: argparse.Namespace, *, workers: int) -> Any:
    return legacy._LiveEngineTimingDecider(
        contract,
        args.showdown_root,
        model_decision_time_ms=None,
        model_native_batch_guard_ms=0,
        model_world_workers=workers,
        model_priors=args.model_priors,
        use_opponent_priors=False,
        override_telemetry=False,
    )


def _execute_row(record: Any, *, contract: Any, args: argparse.Namespace, config: Any) -> dict[str, Any]:
    rows: dict[str, Mapping[str, Any]] = {}
    deciders: dict[str, Any] = {}
    try:
        for arm, workers in (("serial", 1), ("parallel", args.parallel_workers)):
            decider = deciders.setdefault(arm, _new_decider(contract, args, workers=workers))
            callback = decider.prepare(record, config)
            started = time.perf_counter()
            rows[arm] = _arm_payload(
                record=record,
                arm=arm,
                telemetry=callback(),
                outer_wall_ms=(time.perf_counter() - started) * 1000.0,
            )
    finally:
        for decider in deciders.values():
            decider.close()
    return {"decision_id": record.decision_id, "serial": rows.get("serial"), "parallel": rows.get("parallel")}


def _summary(rows: Sequence[Mapping[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    if len(rows) != EXPECTED_DECISIONS:
        raise FixedWorkParityError("fixed-work parity did not cover every corpus decision")
    walls: dict[str, list[float]] = {"serial": [], "parallel": []}
    for row in rows:
        for arm in walls:
            payload = row[arm]
            assert isinstance(payload, Mapping)
            wall = payload.get("outer_wall_ms")
            if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                raise FixedWorkParityError(f"{payload.get('decision_id')}: {arm} has invalid outer wall")
            walls[arm].append(float(wall))
    def percentile(values: Sequence[float]) -> float:
        return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]

    return {
        "complete": True,
        "decision_count": len(rows),
        "root_action_matches": len(rows),
        "root_action_mismatches": 0,
        "parallel_workers": args.parallel_workers,
        "completed_worlds_per_arm": args.worlds,
        "outer_wall_ms": {
            arm: {"p50": statistics.median(values), "p95": percentile(values), "max": max(values)}
            for arm, values in walls.items()
        },
    }


def _run(args: argparse.Namespace, *, ownership: dict[str, bool]) -> dict[str, Any]:
    receipt, active_source, native_fingerprint = _source_provenance(args.source_receipt)
    showdown_source = legacy._verify_showdown_source(args.showdown_root, args.expected_showdown_source_sha256)
    corpus_file_sha256 = legacy.sha256_file(args.corpus)
    if corpus_file_sha256 != args.expected_corpus_file_sha256:
        raise FixedWorkParityError("raw corpus SHA-256 differs from frozen corpus file")
    try:
        corpus_manifest, records = legacy.read_corpus(args.corpus)
        coverage = legacy.validate_representative_timing_panel(records)
    except legacy.CorpusError as error:
        raise FixedWorkParityError(f"timing corpus rejected: {error}") from error
    if corpus_manifest.corpus_sha256 != args.expected_corpus_sha256:
        raise FixedWorkParityError("canonical corpus SHA-256 differs from frozen corpus")
    if len(records) != EXPECTED_DECISIONS:
        raise FixedWorkParityError(f"corpus has {len(records)} decisions; parity requires {EXPECTED_DECISIONS}")
    contract = legacy.resolve_checkpoint_contract(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        model_device=args.model_device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source["content_sha256"],
        expected_showdown_source_sha256=args.expected_showdown_source_sha256,
    )
    manifest = _manifest(
        args=args,
        receipt=receipt,
        active_source=active_source,
        native_fingerprint=native_fingerprint,
        corpus_manifest=corpus_manifest,
        corpus_file_sha256=corpus_file_sha256,
        checkpoint_contract=contract,
        showdown_source=showdown_source,
    )
    out_root = Path(args.out_root)
    legacy._prepare_root(out_root, manifest, resume=args.resume)
    ownership["verified"] = True
    legacy._atomic_json(out_root / "CORPUS_COVERAGE.json", coverage)
    config = legacy.SearchConfig(depth=args.depth, sims=args.sims, batch=args.batch, worlds=args.worlds)
    rows: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        target = out_root / "decisions" / legacy._safe_decision_name(record.decision_id, index)
        if target.exists():
            payload = legacy._read_json(target)
            _validate_row(payload, record=record, args=args)
            rows.append(payload)
            print(f"reused {record.decision_id}: fixed-work root action parity", flush=True)
            continue
        payload: Mapping[str, Any] | None = None
        try:
            payload = _execute_row(record, contract=contract, args=args, config=config)
            _validate_row(payload, record=record, args=args)
        except Exception as error:  # preserve the partial live witness before terminalizing.
            legacy._atomic_json(
                out_root / "failures" / legacy._safe_decision_name(record.decision_id, index),
                {"decision_id": record.decision_id, "error_type": type(error).__name__, "error": str(error), "payload": payload},
            )
            raise
        legacy._atomic_json(target, payload)
        rows.append(payload)
        print(f"completed {record.decision_id}: fixed-work root action parity", flush=True)
    return {"manifest": manifest, "summary": _summary(rows, args=args)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out_root)
    new_or_empty = not out_root.exists() or not any(out_root.iterdir())
    ownership = {"verified": new_or_empty}
    if not new_or_empty and not args.resume:
        print(f"NONPASS: {out_root}: exists; pass --resume to revalidate durable units", file=sys.stderr)
        return 2
    try:
        result = _run(args, ownership=ownership)
    except Exception as error:  # terminal diagnostics must survive every failure after ownership is proven.
        if ownership["verified"]:
            out_root.mkdir(parents=True, exist_ok=True)
            legacy._create_terminal_json(out_root / "NONPASS.json", _terminal(state="NONPASS", error=error))
            (out_root / "RUNNING.json").unlink(missing_ok=True)
        print(f"NONPASS: {error}", file=sys.stderr)
        return 2
    legacy._create_terminal_json(out_root / "PASS.json", _terminal(state="PASS", result=result))
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    print("FIXED-WORK ROOT-ACTION PARITY PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
