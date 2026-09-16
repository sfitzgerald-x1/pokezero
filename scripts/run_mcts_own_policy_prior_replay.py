#!/usr/bin/env python3
"""Durably replay the frozen own-policy-prior MCTS contrast.

This is the unscored, matched-state part of the investigation.  It runs both
prior settings in both execution orders against exactly the same public replay
records.  It is deliberately a measurement gate, never a strength evaluator.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mcts_deadline_qualification as common  # noqa: E402
from pokezero.mcts_eval.deadline_qualification import (  # noqa: E402
    DeadlineQualificationError,
    DeadlineQualificationRequirements,
    validate_deadline_decision,
)
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider  # noqa: E402
from pokezero.mcts_eval.manifest import SearchConfig  # noqa: E402
from pokezero.mcts_eval.resolver import resolve_checkpoint_contract, sha256_file  # noqa: E402
from pokezero.mcts_eval.timing_corpus import (  # noqa: E402
    CorpusError,
    canonical_json_sha256,
    read_corpus,
    validate_representative_timing_panel,
)


SCHEMA_VERSION = "pokezero.mcts-own-policy-prior-replay.v1"
EXPECTED_DECISIONS = 16
EXPECTED_SEAT_COUNT = 8
ORDERS = (("guided", "uniform"), ("uniform", "guided"))
FROZEN_CONTRAST = {
    "model_device": "cpu",
    "deadline_ms": 1_000,
    "native_batch_guard_ms": 64,
    "depth": 2,
    "sims": 256,
    "batch": 16,
    "worlds": 4,
    "model_world_workers": 1,
}


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
    parser.add_argument("--model-device", default="cpu", choices=("cpu",))
    parser.add_argument("--deadline-ms", type=int, default=1_000)
    parser.add_argument("--native-batch-guard-ms", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--model-world-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name, expected in FROZEN_CONTRAST.items():
        actual = getattr(args, name)
        if actual != expected:
            parser.error(f"this contrast freezes --{name.replace('_', '-')}={expected!r}")
    return args


def _receipt_and_source(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the fresh B2 image, active source tree, and installed native engine.

    The deadline qualification's receipt deliberately pins an older reviewed
    deadline mechanism.  This study changes the engine telemetry contract, so
    borrowing that receipt would either reject valid new source or, worse,
    imply that it reviewed source it did not.  The B2 image receipt is the
    correct source of image identity and native-build provenance here.
    """

    receipt = dict(common._read_json(Path(path).expanduser().resolve()))
    if receipt.get("schema_version") != "pokezero.b2-source-image-receipt.v7":
        raise DeadlineQualificationError("source receipt schema is not the supported B2 image receipt")
    if receipt.get("complete") is not True:
        raise DeadlineQualificationError("source receipt is not complete")
    image = receipt.get("immutable_image")
    digest = receipt.get("image_digest")
    if (
        not isinstance(image, str)
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or image.rsplit("@", 1)[-1] != digest
        or not common._is_lower_hex(digest.removeprefix("sha256:"), 64)
    ):
        raise DeadlineQualificationError("source receipt image is not an immutable digest binding")
    commit = receipt.get("source_commit")
    if not common._is_lower_hex(commit, 40):
        raise DeadlineQualificationError("source receipt source_commit is invalid")
    runtime = receipt.get("model_runtime")
    if not isinstance(runtime, Mapping):
        raise DeadlineQualificationError("source receipt omits model runtime provenance")
    if runtime.get("source") != {"commit": commit, "tree_status": "clean_tracked_checkout"}:
        raise DeadlineQualificationError("source receipt model runtime source differs from image commit")
    fingerprint = runtime.get("engine_fingerprint")
    if not common._is_lower_hex(fingerprint, 64):
        raise DeadlineQualificationError("source receipt model runtime fingerprint is invalid")
    active = common._active_source_provenance()
    if active["commit"] != commit:
        raise DeadlineQualificationError("active source commit differs from immutable image receipt")
    try:
        common.assert_fresh()
        installed = common.compute_fingerprint()
    except BaseException as error:  # assert_fresh can raise SystemExit.
        raise DeadlineQualificationError("installed native engine failed its freshness check") from error
    if installed.get("fingerprint") != fingerprint:
        raise DeadlineQualificationError("installed native engine differs from immutable image receipt")
    return receipt, active


def _balanced_coverage(records: Sequence[Any]) -> dict[str, Any]:
    coverage = validate_representative_timing_panel(records)
    if len(records) != EXPECTED_DECISIONS:
        raise DeadlineQualificationError(
            f"corpus has {len(records)} decisions; own-prior replay requires {EXPECTED_DECISIONS}"
        )
    seat_counts = {seat: sum(record.seat == seat for record in records) for seat in ("p1", "p2")}
    if seat_counts != {"p1": EXPECTED_SEAT_COUNT, "p2": EXPECTED_SEAT_COUNT}:
        raise DeadlineQualificationError(
            "own-prior replay corpus must contain exactly eight p1 and eight p2 decisions"
        )
    coverage["exact_seat_counts"] = seat_counts
    return coverage


def _finite(value: Any, field: str, decision_id: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DeadlineQualificationError(f"{decision_id}: {field} must be a finite number")
    return float(value)


def _allocation(payload: Mapping[str, Any], *, arm: str, worlds: int) -> dict[str, Any]:
    """Validate the complete root witness without treating uniform as a policy."""

    decision_id = str(payload.get("decision_id", "?"))
    engine = payload.get("engine_mcts")
    if not isinstance(engine, Mapping):
        raise DeadlineQualificationError(f"{decision_id}: engine telemetry is absent")
    override = engine.get("override")
    if not isinstance(override, Mapping):
        raise DeadlineQualificationError(f"{decision_id}: root override telemetry is absent")
    raw = override.get("root_allocation")
    if not isinstance(raw, Mapping):
        raise DeadlineQualificationError(
            f"{decision_id}: complete root-allocation telemetry is absent (stale source image)"
        )
    if raw.get("worlds") != worlds:
        raise DeadlineQualificationError(f"{decision_id}: allocation world count differs from frozen contrast")
    authority = raw.get("prior_authority")
    if type(authority) is not bool:
        raise DeadlineQualificationError(f"{decision_id}: allocation prior_authority must be boolean")
    cause = raw.get("prior_cause")
    arms = raw.get("arms")
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence) or len(arms) < 2:
        raise DeadlineQualificationError(f"{decision_id}: allocation needs at least two root arms")
    normalized: list[dict[str, Any]] = []
    moves: set[str] = set()
    for index, raw_arm in enumerate(arms):
        if not isinstance(raw_arm, Mapping):
            raise DeadlineQualificationError(f"{decision_id}: allocation arm {index} is not an object")
        move = raw_arm.get("move")
        if not isinstance(move, str) or not move or move in moves:
            raise DeadlineQualificationError(f"{decision_id}: allocation arms need unique serialized moves")
        moves.add(move)
        visit_share = _finite(raw_arm.get("visit_share"), "allocation visit_share", decision_id)
        if visit_share < 0:
            raise DeadlineQualificationError(f"{decision_id}: allocation visit_share is negative")
        reported = _finite(raw_arm.get("reported_prior"), "allocation reported_prior", decision_id)
        if reported < 0:
            raise DeadlineQualificationError(f"{decision_id}: allocation reported_prior is negative")
        model = raw_arm.get("model_prior")
        if arm == "guided":
            if type(model) not in (int, float) or not math.isfinite(float(model)) or float(model) < 0:
                raise DeadlineQualificationError(f"{decision_id}: guided allocation lacks model prior")
            normalized_model: float | None = float(model)
        else:
            if model is not None:
                raise DeadlineQualificationError(
                    f"{decision_id}: uniform allocation manufactured a model prior"
                )
            normalized_model = None
        normalized.append(
            {
                "move": move,
                "visit_share": visit_share,
                "q": raw_arm.get("q"),
                "reported_prior": reported,
                "model_prior": normalized_model,
            }
        )
    if abs(sum(item["visit_share"] for item in normalized) - 1.0) > 2e-5:
        raise DeadlineQualificationError(f"{decision_id}: allocation visit shares do not sum to one")
    if abs(sum(item["reported_prior"] for item in normalized) - 1.0) > 2e-5:
        raise DeadlineQualificationError(f"{decision_id}: allocation reported priors do not sum to one")
    if arm == "guided":
        if authority is not True or cause is not None:
            raise DeadlineQualificationError(f"{decision_id}: guided root priors are not authoritative")
        if abs(sum(item["model_prior"] or 0.0 for item in normalized) - 1.0) > 2e-5:
            raise DeadlineQualificationError(f"{decision_id}: guided model priors do not sum to one")
    else:
        if authority is not False or cause != "no_root_priors":
            raise DeadlineQualificationError(f"{decision_id}: uniform arm has the wrong root-prior authority")
        uniform = 1.0 / len(normalized)
        if any(abs(item["reported_prior"] - uniform) > 2e-5 for item in normalized):
            raise DeadlineQualificationError(f"{decision_id}: uniform arm did not restore uniform root priors")
    return {"prior_authority": authority, "prior_cause": cause, "arms": normalized}


def _decision_payload(record: Any, telemetry: Mapping[str, Any], *, wall_ms: float, arm: str, worlds: int) -> dict[str, Any]:
    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        engine = {}
    payload = {
        "decision_id": record.decision_id,
        "corpus_record_sha256": canonical_json_sha256(record.to_payload()),
        "battle_id": record.battle_id,
        "seat": record.seat,
        "turn_index": record.turn_index,
        "arm": arm,
        "root_action": telemetry.get("root_action"),
        "outer_wall_ms": round(wall_ms, 3),
        "invalid_actions": telemetry.get("invalid_actions"),
        "engine_mcts": dict(engine),
    }
    payload["root_allocation"] = _allocation(payload, arm=arm, worlds=worlds)
    return payload


def _validate_arm(payload: Mapping[str, Any], *, record: Any, arm: str, requirements: DeadlineQualificationRequirements) -> dict[str, Any]:
    if payload.get("decision_id") != record.decision_id:
        raise DeadlineQualificationError("durable decision belongs to a different corpus record")
    if payload.get("corpus_record_sha256") != canonical_json_sha256(record.to_payload()):
        raise DeadlineQualificationError("durable decision corpus identity differs from the frozen record")
    if payload.get("arm") != arm:
        raise DeadlineQualificationError("durable arm label differs from execution order")
    validate_deadline_decision(payload, requirements=requirements)
    allocation = _allocation(payload, arm=arm, worlds=requirements.worlds)
    return allocation


def _new_decider(contract: Any, args: argparse.Namespace, *, guided: bool) -> _LiveEngineTimingDecider:
    return _LiveEngineTimingDecider(
        contract,
        args.showdown_root,
        model_decision_time_ms=args.deadline_ms,
        model_native_batch_guard_ms=args.native_batch_guard_ms,
        model_world_workers=args.model_world_workers,
        model_priors=guided,
        use_opponent_priors=False,
        override_telemetry=True,
    )


def _run_order(record: Any, *, order: tuple[str, str], contract: Any, args: argparse.Namespace, config: SearchConfig, requirements: DeadlineQualificationRequirements) -> dict[str, Any]:
    result: dict[str, Any] = {}
    deciders: dict[str, _LiveEngineTimingDecider] = {}
    try:
        for arm in order:
            decider = deciders.setdefault(arm, _new_decider(contract, args, guided=arm == "guided"))
            timed = decider.prepare(record, config)
            started = time.perf_counter()
            telemetry = timed()
            payload = _decision_payload(
                record, telemetry, wall_ms=(time.perf_counter() - started) * 1000.0,
                arm=arm, worlds=requirements.worlds,
            )
            _validate_arm(payload, record=record, arm=arm, requirements=requirements)
            result[arm] = payload
    finally:
        for decider in deciders.values():
            decider.close()
    return result


def _allocation_changed(guided: Mapping[str, Any], uniform: Mapping[str, Any]) -> bool:
    guided_arms = {arm["move"]: arm["visit_share"] for arm in guided["root_allocation"]["arms"]}
    uniform_arms = {arm["move"]: arm["visit_share"] for arm in uniform["root_allocation"]["arms"]}
    return set(guided_arms) == set(uniform_arms) and any(
        abs(guided_arms[move] - uniform_arms[move]) > 2e-5 for move in guided_arms
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    walls = {arm: [] for arm in ("guided", "uniform")}
    nonflat = 0
    allocation_changes = 0
    for row in rows:
        for sequence in row["orders"].values():
            for arm in ("guided", "uniform"):
                walls[arm].append(float(sequence[arm]["outer_wall_ms"]) / 1000.0)
            priors = [item["model_prior"] for item in sequence["guided"]["root_allocation"]["arms"]]
            if max(priors) - min(priors) > 2e-5:
                nonflat += 1
            if _allocation_changed(sequence["guided"], sequence["uniform"]):
                allocation_changes += 1
    if not nonflat:
        raise DeadlineQualificationError("matched replay found no non-flat guided multi-action root")
    if not allocation_changes:
        raise DeadlineQualificationError("matched replay found no guided/uniform root-allocation difference")
    if not all(walls.values()):
        raise DeadlineQualificationError("matched replay has no arm wall-time evidence")
    mean_ratio = statistics.fmean(walls["guided"]) / statistics.fmean(walls["uniform"])
    return {
        "decision_count": len(rows),
        "arm_samples": {arm: len(values) for arm, values in walls.items()},
        "wall_seconds": {
            arm: {"mean": statistics.fmean(values), "p95": sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)]}
            for arm, values in walls.items()
        },
        "guided_to_uniform_mean_wall_ratio": mean_ratio,
        "nonflat_guided_multi_action_observations": nonflat,
        "guided_uniform_allocation_changes": allocation_changes,
    }


def _manifest(*, args: argparse.Namespace, receipt: Mapping[str, Any], active: Mapping[str, Any], corpus: Any, corpus_file_sha256: str, checkpoint: Any, showdown: Mapping[str, Any]) -> dict[str, Any]:
    fixed = {
        "depth": args.depth, "sims": args.sims, "batch": args.batch, "worlds": args.worlds,
        "early_stop": False, "use_opponent_priors": False,
        "model_decision_time_ms": args.deadline_ms,
        "model_native_batch_guard_ms": args.native_batch_guard_ms,
        "model_world_workers": args.model_world_workers,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_receipt": dict(receipt), "active_source": dict(active),
        "checkpoint": checkpoint.to_manifest(), "showdown_source": dict(showdown),
        "corpus_path": str(Path(args.corpus).resolve()), "corpus_sha256": corpus.corpus_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "orders": [list(order) for order in ORDERS],
        "guided": {**fixed, "model_priors": True},
        "uniform": {**fixed, "model_priors": False},
        "admission": {"max_p95_wall_seconds": 1.20, "max_guided_uniform_mean_ratio": 1.05},
    }


def _run(args: argparse.Namespace, *, ownership: dict[str, bool]) -> dict[str, Any]:
    receipt, active = _receipt_and_source(args.source_receipt)
    showdown = common._verify_showdown_source(args.showdown_root, args.expected_showdown_source_sha256)
    corpus_file_sha256 = sha256_file(args.corpus)
    if corpus_file_sha256 != args.expected_corpus_file_sha256:
        raise DeadlineQualificationError("raw corpus SHA-256 differs from the frozen corpus file")
    try:
        corpus, records = read_corpus(args.corpus)
        coverage = _balanced_coverage(records)
    except CorpusError as error:
        raise DeadlineQualificationError(f"timing corpus rejected: {error}") from error
    if corpus.corpus_sha256 != args.expected_corpus_sha256:
        raise DeadlineQualificationError("canonical corpus SHA-256 differs from frozen corpus")
    checkpoint = resolve_checkpoint_contract(
        args.checkpoint, expected_sha256=args.expected_checkpoint_sha256, model_device=args.model_device,
        showdown_root=args.showdown_root, showdown_source_sha256=showdown["content_sha256"],
        expected_showdown_source_sha256=args.expected_showdown_source_sha256,
    )
    manifest = _manifest(args=args, receipt=receipt, active=active, corpus=corpus,
                         corpus_file_sha256=corpus_file_sha256, checkpoint=checkpoint, showdown=showdown)
    out_root = Path(args.out_root)
    common._prepare_root(out_root, manifest, resume=args.resume)
    ownership["verified"] = True
    common._atomic_json(out_root / "CORPUS_COVERAGE.json", coverage)
    requirements = DeadlineQualificationRequirements(
        requested_ms=args.deadline_ms, native_batch_guard_ms=args.native_batch_guard_ms,
        sims_per_world=args.sims, worlds=args.worlds, model_world_workers=args.model_world_workers,
        expected_decisions=EXPECTED_DECISIONS,
    )
    config = SearchConfig(depth=args.depth, sims=args.sims, batch=args.batch, worlds=args.worlds)
    rows: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        target = out_root / "decisions" / common._safe_decision_name(record.decision_id, index)
        if target.exists():
            row = common._read_json(target)
            if row.get("decision_id") != record.decision_id or row.get("corpus_record_sha256") != canonical_json_sha256(record.to_payload()):
                raise DeadlineQualificationError(f"{target}: durable decision differs from frozen corpus")
            for name, order in (("guided_then_uniform", ORDERS[0]), ("uniform_then_guided", ORDERS[1])):
                sequence = row.get("orders", {}).get(name)
                if not isinstance(sequence, Mapping):
                    raise DeadlineQualificationError(f"{target}: missing durable execution order {name}")
                for arm in order:
                    _validate_arm(sequence.get(arm, {}), record=record, arm=arm, requirements=requirements)
            rows.append(row)
            continue
        try:
            row = {"decision_id": record.decision_id, "corpus_record_sha256": canonical_json_sha256(record.to_payload()),
                   "battle_id": record.battle_id, "seat": record.seat, "turn_index": record.turn_index, "orders": {}}
            for name, order in (("guided_then_uniform", ORDERS[0]), ("uniform_then_guided", ORDERS[1])):
                row["orders"][name] = _run_order(record, order=order, contract=checkpoint, args=args,
                                                  config=config, requirements=requirements)
        except Exception as error:  # noqa: BLE001 - retain live failure diagnostics without admitting a partial row
            common._atomic_json(out_root / "failures" / common._safe_decision_name(record.decision_id, index),
                                {"decision_id": record.decision_id, "error_type": type(error).__name__, "error": str(error)})
            raise
        common._atomic_json(target, row)
        rows.append(row)
        print(f"completed {record.decision_id}: both execution orders", flush=True)
    summary = _summary(rows)
    timing = summary["wall_seconds"]
    summary["timing_eligible"] = (
        timing["guided"]["p95"] <= 1.20 and timing["uniform"]["p95"] <= 1.20
        and summary["guided_to_uniform_mean_wall_ratio"] <= 1.05
    )
    if not summary["timing_eligible"]:
        raise DeadlineQualificationError("matched replay missed the frozen timing admission limits")
    return {"manifest": manifest, "coverage": coverage, "summary": summary}


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
    except Exception as error:  # noqa: BLE001
        if ownership["verified"]:
            out_root.mkdir(parents=True, exist_ok=True)
            common._create_terminal_json(
                out_root / "NONPASS.json",
                {"schema_version": SCHEMA_VERSION, "state": "NONPASS", "marker": "OWN_POLICY_PRIOR_REPLAY_NONPASS",
                 "error_type": type(error).__name__, "error": str(error)},
            )
            (out_root / "RUNNING.json").unlink(missing_ok=True)
        print(f"NONPASS: {error}", file=sys.stderr)
        return 2
    common._create_terminal_json(
        out_root / "PASS.json",
        {"schema_version": SCHEMA_VERSION, "state": "PASS", "marker": "OWN_POLICY_PRIOR_REPLAY_PASS", **result},
    )
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    print("OWN POLICY PRIOR REPLAY PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
