#!/usr/bin/env python3
"""Fail-closed fan-in for an observer-only MCTS cohort repair.

This is deliberately an *aggregator*, not a copier.  It keeps a terminal
record at its original path and makes a second, explicit entry for each replay
made with a source image that differs only in the post-action evidence writer.
The result is therefore a transparent recovered cohort rather than a relabelled
fresh run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from pokezero.mcts_eval.head_to_head import (
    HeadToHeadError,
    MctsPolicySpec,
    complete_pair,
    load_pair,
)


SCHEMA_VERSION = "pokezero.mcts-observer-only-recovery.v1"
DEFAULT_SEEDS = tuple(range(2026093000, 2026093200))
DEFAULT_REPAIR_SEEDS = (2026093055, 2026093092, 2026093125, 2026093150, 2026093180)


class RecoveryError(RuntimeError):
    """The mixed-source recovery cannot be accepted."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} {path} must be a JSON object.")
    return value


def _require_string(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"{label} has no non-empty {key!r}.")
    return value


def _seed_dir(root: Path, seed: int, *, original: bool) -> Path:
    if original:
        shard = "s0" if seed < 2026093100 else "s1"
        return root / "shards" / shard / "seeds" / f"seed-{seed}"
    return root / "seeds" / f"seed-{seed}"


def _policy_pair(manifest: Mapping[str, Any], *, label: str) -> tuple[MctsPolicySpec, MctsPolicySpec]:
    candidate_raw = manifest.get("candidate")
    raw_raw = manifest.get("raw")
    if not isinstance(candidate_raw, Mapping) or not isinstance(raw_raw, Mapping):
        raise RecoveryError(f"{label} has no candidate/raw policy specifications.")
    try:
        return MctsPolicySpec.from_payload(candidate_raw), MctsPolicySpec.from_payload(raw_raw)
    except (TypeError, ValueError, HeadToHeadError) as error:
        raise RecoveryError(f"{label} has invalid policy specifications: {error}") from error


def _source_identity(manifest: Mapping[str, Any], *, label: str) -> dict[str, str]:
    active = manifest.get("active_source")
    if not isinstance(active, Mapping):
        raise RecoveryError(f"{label} has no active source receipt.")
    return {
        "commit": _require_string(active, "commit", label=label),
        "tree_sha256": _require_string(active, "tree_sha256", label=label),
        "engine_fingerprint": _require_string(manifest, "active_engine_fingerprint", label=label),
        "showdown_source_sha256": _require_string(
            manifest.get("active_showdown_source")
            if isinstance(manifest.get("active_showdown_source"), Mapping)
            else {},
            "content_sha256",
            label=label,
        ),
    }


def _execution_identity(manifest: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    declared = manifest.get("declared_manifest")
    if not isinstance(declared, Mapping):
        raise RecoveryError(f"{label} has no declared manifest.")
    candidate = declared.get("candidate")
    raw = declared.get("raw")
    if not isinstance(candidate, Mapping) or not isinstance(raw, Mapping):
        raise RecoveryError(f"{label} has no declared candidate/raw identity.")
    # Source receipt fields are intentionally excluded here: they are recorded
    # separately and must differ for the observer fix. Everything that can
    # affect a battle/search remains byte-for-byte equal.
    def without_source(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): item
            for key, item in value.items()
            if key not in {"source_commit", "source_tree_sha256"}
        }

    return {
        "candidate": without_source(candidate),
        "raw": without_source(raw),
        "checkpoint_sha256": declared.get("checkpoint_sha256"),
        "showdown_source_sha256": declared.get("showdown_source_sha256"),
        "max_decision_rounds": declared.get("max_decision_rounds"),
    }


def _validate_terminal_seed(
    directory: Path,
    *,
    seed: int,
    expected_commit: str,
    label: str,
) -> dict[str, Any]:
    manifest = _read_object(directory / "manifest.json", label=f"{label} manifest")
    declared = manifest.get("declared_manifest")
    if not isinstance(declared, Mapping) or declared.get("seeds") != [seed]:
        raise RecoveryError(f"{label} does not declare exactly seed {seed}.")
    source = _source_identity(manifest, label=label)
    if source["commit"] != expected_commit:
        raise RecoveryError(
            f"{label} source commit {source['commit']} is not expected {expected_commit}."
        )
    candidate, incumbent = _policy_pair(manifest, label=label)
    if candidate.source_commit != expected_commit or incumbent.source_commit != expected_commit:
        raise RecoveryError(f"{label} policy source commits do not bind the expected receipt.")
    if candidate.engine_fingerprint != source["engine_fingerprint"]:
        raise RecoveryError(f"{label} candidate engine fingerprint disagrees with its receipt.")
    if candidate.showdown_source_sha256 != source["showdown_source_sha256"]:
        raise RecoveryError(f"{label} candidate Showdown receipt disagrees with active receipt.")

    terminal = _read_object(directory / "runner-terminal.json", label=f"{label} runner terminal")
    if terminal.get("status") != "COMPLETE" or terminal.get("exit_code") != 0:
        raise RecoveryError(f"{label} runner is not a clean terminal completion.")
    complete = _read_object(directory / "COMPLETE.json", label=f"{label} complete receipt")
    if complete.get("status") != "COMPLETE" or complete.get("pairs") != 1 or complete.get("games") != 2:
        raise RecoveryError(f"{label} is not exactly one complete mirrored pair.")
    summary = directory / "summary.json"
    if not summary.is_file() or complete.get("summary_sha256") != _sha256(summary):
        raise RecoveryError(f"{label} summary is missing or disagrees with COMPLETE.json.")
    if complete.get("candidate_provenance_sha256") != candidate.provenance_sha256:
        raise RecoveryError(f"{label} candidate provenance is not bound by COMPLETE.json.")
    if complete.get("raw_provenance_sha256") != incumbent.provenance_sha256:
        raise RecoveryError(f"{label} raw provenance is not bound by COMPLETE.json.")
    try:
        games = load_pair(directory, seed=seed, candidate=candidate, incumbent=incumbent)
        complete_pair(list(games.values()), seed=seed, candidate=candidate, incumbent=incumbent)
    except HeadToHeadError as error:
        raise RecoveryError(f"{label} has no complete valid mirrored game pair: {error}") from error
    return {
        "seed": seed,
        "directory": str(directory),
        "complete_sha256": _sha256(directory / "COMPLETE.json"),
        "summary_sha256": _sha256(summary),
        "source": source,
        "execution": _execution_identity(manifest, label=label),
    }


def build_recovery_manifest(
    *,
    original_root: Path,
    replay_root: Path,
    original_commit: str,
    replay_commit: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    repair_seeds: Sequence[int] = DEFAULT_REPAIR_SEEDS,
) -> dict[str, Any]:
    roster = tuple(seeds)
    repairs = tuple(repair_seeds)
    if len(roster) != 200 or len(set(roster)) != len(roster):
        raise RecoveryError("the registered roster must contain exactly 200 distinct seeds.")
    if len(repairs) != 5 or len(set(repairs)) != len(repairs) or not set(repairs) <= set(roster):
        raise RecoveryError("the repair roster must contain exactly five registered seeds.")

    retained: list[dict[str, Any]] = []
    for seed in roster:
        directory = _seed_dir(original_root, seed, original=True)
        complete = directory / "COMPLETE.json"
        if seed in repairs:
            if complete.exists():
                raise RecoveryError(f"repair seed {seed} unexpectedly has an original COMPLETE.json.")
            continue
        if not complete.is_file():
            raise RecoveryError(f"retained seed {seed} has no original COMPLETE.json.")
        retained.append(
            _validate_terminal_seed(
                directory, seed=seed, expected_commit=original_commit, label=f"original seed {seed}"
            )
        )
    if len(retained) != 195:
        raise RecoveryError("the original root did not provide exactly 195 immutable complete pairs.")

    repaired: list[dict[str, Any]] = []
    for seed in repairs:
        directory = _seed_dir(replay_root, seed, original=False)
        repaired.append(
            _validate_terminal_seed(
                directory, seed=seed, expected_commit=replay_commit, label=f"replay seed {seed}"
            )
        )

    original_execution = retained[0]["execution"]
    for entry in retained[1:] + repaired:
        if entry["execution"] != original_execution:
            raise RecoveryError(
                f"seed {entry['seed']} changes a battle/search setting; observer-only recovery refused."
            )
    original_source = retained[0]["source"]
    replay_source = repaired[0]["source"]
    for entry in retained:
        if entry["source"] != original_source:
            raise RecoveryError("original retained pairs have mixed source provenance.")
    for entry in repaired:
        if entry["source"] != replay_source:
            raise RecoveryError("repaired pairs have mixed source provenance.")
    if original_source["engine_fingerprint"] != replay_source["engine_fingerprint"]:
        raise RecoveryError("replay changes the native engine fingerprint; observer-only recovery refused.")
    if original_source["showdown_source_sha256"] != replay_source["showdown_source_sha256"]:
        raise RecoveryError("replay changes the Showdown source; observer-only recovery refused.")

    roster_sha256 = hashlib.sha256(",".join(str(seed) for seed in roster).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RECOVERED_COMPLETE",
        "registered_roster": list(roster),
        "registered_roster_sha256": roster_sha256,
        "repair_seeds": list(repairs),
        "retained_pairs": retained,
        "replayed_pairs": repaired,
        "pair_count": len(retained) + len(repaired),
        "game_count": 2 * (len(retained) + len(repaired)),
        "original_source": original_source,
        "observer_replay_source": replay_source,
        "execution_identity": original_execution,
        "recovery_scope": {
            "original_pairs_reused": 195,
            "replayed_pairs": 5,
            "replayed_games": 10,
            "rule": "no original durable game is copied, moved, or relabelled",
        },
    }


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = _canonical_bytes(payload)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != expected:
                    raise RecoveryError(
                        f"refusing to replace different existing recovery manifest {path}."
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)
        return
    if existing != expected:
        raise RecoveryError(f"refusing to replace different existing recovery manifest {path}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-commit", required=True)
    parser.add_argument("--replay-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_recovery_manifest(
        original_root=args.original_root,
        replay_root=args.replay_root,
        original_commit=args.original_commit,
        replay_commit=args.replay_commit,
    )
    write_immutable_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
