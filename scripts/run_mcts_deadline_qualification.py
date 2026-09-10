#!/usr/bin/env python3
"""Run the bounded, source-bound model-MCTS deadline qualification.

This is a measurement gate, not a game trial.  It replays the pre-frozen timing
corpus with one fixed model decision budget, writes atomic per-decision evidence,
and emits PASS only when the durable evidence proves a nonzero native deadline
prefix without fallbacks, invalid choices, refused native calls, or zero-work
decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

# The bounded runner is necessarily newer than the deadline mechanism it
# qualifies.  These are the complete implementation surfaces whose source must
# remain exactly equal to the reviewed deadline commit; any drift is a
# different study, not a replay of that mechanism.
DEADLINE_MECHANICS_PATHS = (
    "src/pokezero/engine_search.py",
    "rust/pokezero-search",
)

from pokezero.mcts_eval.deadline_qualification import (  # noqa: E402
    DEADLINE_QUALIFICATION_SCHEMA_VERSION,
    DeadlineQualificationError,
    DeadlineQualificationRequirements,
    validate_deadline_decision,
    validate_deadline_qualification,
)
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider  # noqa: E402
from pokezero.mcts_eval.manifest import SearchConfig  # noqa: E402
from pokezero.mcts_eval.resolver import resolve_checkpoint_contract, sha256_file  # noqa: E402
from pokezero.mcts_eval.timing_corpus import (  # noqa: E402
    CorpusError,
    read_corpus,
    validate_representative_timing_panel,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DeadlineQualificationError(f"{path}: expected a JSON object")
    return value


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        raise DeadlineQualificationError("cannot resolve the source commit") from error


def _require_clean_source() -> None:
    try:
        changes = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        )
    except subprocess.CalledProcessError as error:
        raise DeadlineQualificationError("cannot inspect source cleanliness") from error
    if changes:
        raise DeadlineQualificationError("source checkout is dirty; refusing an unbound replay")


def _safe_decision_name(decision_id: str, index: int) -> str:
    digest = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:16]
    return f"{index:02d}-{digest}.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-corpus-file-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--expected-deadline-source-commit",
        required=True,
        help="Reviewed commit whose engine-search and native tree must match this source exactly.",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--deadline-ms", type=int, default=1_000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only independently revalidated, durable decision units for this exact manifest.",
    )
    return parser.parse_args(argv)


def _frozen_manifest(
    *,
    args: argparse.Namespace,
    source_commit: str,
    corpus_manifest: Any,
    corpus_file_sha256: str,
    checkpoint_contract: Any,
    deadline_mechanics_source: Mapping[str, Any],
    requirements: DeadlineQualificationRequirements,
) -> dict[str, Any]:
    return {
        "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
        "source_commit": source_commit,
        "corpus_path": str(Path(args.corpus).resolve()),
        "corpus_sha256": corpus_manifest.corpus_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "checkpoint": checkpoint_contract.to_manifest(),
        "deadline_mechanics_source": dict(deadline_mechanics_source),
        "requirements": requirements.to_payload(),
        "search_config": {
            "depth": args.depth,
            "sims": args.sims,
            "batch": args.batch,
            "worlds": args.worlds,
            "early_stop": False,
            # This is a deadline-mechanics qualification, not a prior-policy
            # study.  Freeze both selection-prior toggles off exactly as the
            # predeclared contract requires.
            "model_priors": False,
            "use_opponent_priors": False,
            "model_decision_time_ms": args.deadline_ms,
        },
    }


def _deadline_mechanics_source(expected_commit: str) -> dict[str, Any]:
    """Prove this newer runner is exercising the reviewed deadline mechanism."""

    try:
        canonical_commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{expected_commit}^{{commit}}"], text=True
        ).strip()
        objects = {
            path: subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", f"{canonical_commit}:{path}"], text=True
            ).strip()
            for path in DEADLINE_MECHANICS_PATHS
        }
    except subprocess.CalledProcessError as error:
        raise DeadlineQualificationError(
            f"cannot resolve reviewed deadline source {expected_commit}"
        ) from error
    compared = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", canonical_commit, "--", *DEADLINE_MECHANICS_PATHS],
        check=False,
    )
    if compared.returncode == 1:
        raise DeadlineQualificationError(
            "engine-search or native search source differs from the reviewed deadline source"
        )
    if compared.returncode != 0:
        raise DeadlineQualificationError("cannot compare source against reviewed deadline source")
    return {
        "commit": canonical_commit,
        "paths": dict(objects),
    }


def _prepare_root(out_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    manifest_path = out_root / "MANIFEST.json"
    terminals = [out_root / name for name in ("PASS.json", "NONPASS.json") if (out_root / name).exists()]
    if terminals:
        raise DeadlineQualificationError(
            f"{out_root}: already has terminal artifact(s): {', '.join(str(path.name) for path in terminals)}"
        )
    if manifest_path.exists():
        if not resume:
            raise DeadlineQualificationError(f"{out_root}: exists; pass --resume to revalidate durable units")
        if _read_json(manifest_path) != manifest:
            raise DeadlineQualificationError(f"{out_root}: existing manifest differs from this frozen contract")
    else:
        if out_root.exists() and any(out_root.iterdir()):
            raise DeadlineQualificationError(
                f"{out_root}: nonempty root has no manifest and cannot be adopted"
            )
        _atomic_json(manifest_path, manifest)
    _atomic_json(out_root / "RUNNING.json", {"manifest": manifest, "state": "RUNNING"})


def _decision_payload(
    *,
    record: Any,
    telemetry: Mapping[str, Any],
    outer_wall_ms: float,
) -> dict[str, Any]:
    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        engine = {}
    return {
        "decision_id": record.decision_id,
        "battle_id": record.battle_id,
        "seat": record.seat,
        "turn_index": record.turn_index,
        "root_action": telemetry.get("root_action"),
        "outer_wall_ms": round(outer_wall_ms, 3),
        "invalid_actions": telemetry.get("invalid_actions"),
        "engine_mcts": dict(engine),
    }


def _run(args: argparse.Namespace, *, ownership: dict[str, bool]) -> dict[str, Any]:
    source_commit = _source_commit()
    if source_commit != args.expected_source_commit:
        raise DeadlineQualificationError(
            f"source commit {source_commit} != expected {args.expected_source_commit}"
        )
    _require_clean_source()
    requirements = DeadlineQualificationRequirements(
        requested_ms=args.deadline_ms,
        sims_per_world=args.sims,
        worlds=args.worlds,
        expected_decisions=16,
    )
    corpus_file_sha256 = sha256_file(args.corpus)
    if corpus_file_sha256 != args.expected_corpus_file_sha256:
        raise DeadlineQualificationError(
            "raw corpus SHA-256 differs from the frozen corpus file"
        )
    try:
        corpus_manifest, records = read_corpus(args.corpus)
        coverage = validate_representative_timing_panel(records)
    except CorpusError as error:
        raise DeadlineQualificationError(f"timing corpus rejected: {error}") from error
    if corpus_manifest.corpus_sha256 != args.expected_corpus_sha256:
        raise DeadlineQualificationError("canonical corpus SHA-256 differs from the frozen corpus")
    if len(records) != requirements.expected_decisions:
        raise DeadlineQualificationError(
            f"corpus has {len(records)} decisions; qualification requires {requirements.expected_decisions}"
        )
    checkpoint_contract = resolve_checkpoint_contract(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        model_device=args.model_device,
        showdown_root=args.showdown_root,
    )
    deadline_mechanics_source = _deadline_mechanics_source(args.expected_deadline_source_commit)
    manifest = _frozen_manifest(
        args=args,
        source_commit=source_commit,
        corpus_manifest=corpus_manifest,
        corpus_file_sha256=corpus_file_sha256,
        checkpoint_contract=checkpoint_contract,
        deadline_mechanics_source=deadline_mechanics_source,
        requirements=requirements,
    )
    out_root = Path(args.out_root)
    _prepare_root(out_root, manifest, resume=args.resume)
    ownership["verified"] = True
    _atomic_json(out_root / "CORPUS_COVERAGE.json", coverage)

    config = SearchConfig(depth=args.depth, sims=args.sims, batch=args.batch, worlds=args.worlds)
    rows: list[Mapping[str, Any]] = []
    decider = _LiveEngineTimingDecider(
        checkpoint_contract,
        args.showdown_root,
        model_decision_time_ms=args.deadline_ms,
        model_priors=False,
        use_opponent_priors=False,
    )
    try:
        for index, record in enumerate(records):
            target = out_root / "decisions" / _safe_decision_name(record.decision_id, index)
            if target.exists():
                payload = _read_json(target)
                if payload.get("decision_id") != record.decision_id:
                    raise DeadlineQualificationError(
                        f"{target}: durable unit belongs to a different corpus decision"
                    )
                validated = validate_deadline_decision(payload, requirements=requirements)
                rows.append(payload)
                print(f"reused {record.decision_id}: {validated['native_prefixes']} native prefixes", flush=True)
                continue
            payload: dict[str, Any] | None = None
            try:
                timed = decider.prepare(record, config)
                started = time.perf_counter()
                telemetry = timed()
                payload = _decision_payload(
                    record=record,
                    telemetry=telemetry,
                    outer_wall_ms=(time.perf_counter() - started) * 1000.0,
                )
                validated = validate_deadline_decision(payload, requirements=requirements)
            except Exception as error:  # noqa: BLE001 - preserve the failed live witness too
                _atomic_json(
                    out_root / "failures" / _safe_decision_name(record.decision_id, index),
                    {
                        "decision_id": record.decision_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "payload": payload,
                    },
                )
                raise
            assert payload is not None
            _atomic_json(target, payload)
            rows.append(payload)
            print(f"completed {record.decision_id}: {validated['native_prefixes']} native prefixes", flush=True)
    finally:
        decider.close()
    summary = validate_deadline_qualification(rows, requirements=requirements)
    return {"manifest": manifest, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out_root)
    # A new/empty root is ours to terminalize even when preflight itself fails.
    # A resume root becomes ours only *after* its immutable manifest has matched;
    # this prevents a mistaken invocation from appending a NONPASS to somebody
    # else's in-flight or terminal study.
    new_or_empty_root = not out_root.exists() or not any(out_root.iterdir())
    ownership = {"verified": new_or_empty_root}
    if not new_or_empty_root and not args.resume:
        print(f"NONPASS: {out_root}: exists; pass --resume to revalidate durable units", file=sys.stderr)
        return 2
    try:
        result = _run(args, ownership=ownership)
    except Exception as error:  # noqa: BLE001 - terminal diagnostics must survive every failure
        if ownership["verified"]:
            out_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                out_root / "NONPASS.json",
                {
                    "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
                    "state": "NONPASS",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            (out_root / "RUNNING.json").unlink(missing_ok=True)
        print(f"NONPASS: {error}", file=sys.stderr)
        return 2
    _atomic_json(
        out_root / "PASS.json",
        {
            "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
            "state": "PASS",
            **result,
        },
    )
    _atomic_json(
        out_root / "DEADLINE_QUALIFICATION_PASS.json",
        {
            "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
            "state": "PASS",
            "marker": "DEADLINE_QUALIFICATION_PASS",
        },
    )
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    print("DEADLINE QUALIFICATION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
