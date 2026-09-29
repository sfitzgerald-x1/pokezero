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
from pokezero.mcts_eval.scoring import bootstrap_indices, bootstrap_mean


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
    matches = sorted(root.glob(f"lanes/s*/seeds/seed-{seed}"))
    if len(matches) != 1:
        raise RecoveryError(
            f"replay seed {seed} must occur in exactly one durable lane; found {len(matches)}."
        )
    return matches[0]


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
        "bootstrap": declared.get("bootstrap"),
    }


def _summary_evidence(
    directory: Path,
    *,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], float]:
    """Load one terminal summary and reject fallback-tainted scoring evidence."""
    summary = _read_object(directory / "summary.json", label=f"{label} summary")
    if summary.get("seeds") != [seed]:
        raise RecoveryError(f"{label} summary does not contain exactly seed {seed}.")
    pair_scores = summary.get("pair_scores")
    if (
        not isinstance(pair_scores, list)
        or len(pair_scores) != 1
        or isinstance(pair_scores[0], bool)
        or not isinstance(pair_scores[0], (int, float))
        or not 0.0 <= float(pair_scores[0]) <= 1.0
    ):
        raise RecoveryError(f"{label} summary lacks one valid mirrored pair score.")

    for prefix in ("candidate", "incumbent"):
        for key in ("prior_fallbacks", "root_prior_fallbacks", "branch_prior_fallbacks"):
            field = f"{prefix}_{key}"
            if summary.get(field) != 0:
                raise RecoveryError(f"{label} has nonzero {field}; recovery refused.")
        for key in (
            "opponent_request_order_root_fallback_statuses",
            "opponent_request_order_root_omission_statuses",
        ):
            field = f"{prefix}_{key}"
            if summary.get(field) != {}:
                raise RecoveryError(f"{label} has nonempty {field}; recovery refused.")
    return summary, float(pair_scores[0])


def _bootstrap_contract(execution: Mapping[str, Any]) -> dict[str, float | int]:
    bootstrap = execution.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise RecoveryError("execution identity has no bootstrap contract.")
    resamples = bootstrap.get("resamples")
    seed = bootstrap.get("seed")
    confidence_level = bootstrap.get("confidence_level")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise RecoveryError("execution identity has an invalid bootstrap contract.")
    return {
        "resamples": resamples,
        "seed": seed,
        "confidence_level": float(confidence_level),
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
    summary_path = directory / "summary.json"
    if not summary_path.is_file() or complete.get("summary_sha256") != _sha256(summary_path):
        raise RecoveryError(f"{label} summary is missing or disagrees with COMPLETE.json.")
    if complete.get("candidate_provenance_sha256") != candidate.provenance_sha256:
        raise RecoveryError(f"{label} candidate provenance is not bound by COMPLETE.json.")
    if complete.get("raw_provenance_sha256") != incumbent.provenance_sha256:
        raise RecoveryError(f"{label} raw provenance is not bound by COMPLETE.json.")
    try:
        games = load_pair(directory, seed=seed, candidate=candidate, incumbent=incumbent)
        completed_games = complete_pair(
            list(games.values()), seed=seed, candidate=candidate, incumbent=incumbent
        )
    except HeadToHeadError as error:
        raise RecoveryError(f"{label} has no complete valid mirrored game pair: {error}") from error
    seat_scores: dict[str, float] = {}
    seat_outcomes: dict[str, str] = {}
    capped_games = 0
    for game in completed_games:
        seat = getattr(game, "candidate_seat", None)
        result = getattr(game, "result", None)
        score = getattr(result, "score", None)
        outcome = getattr(result, "outcome", None)
        capped = getattr(game, "terminal_capped", None)
        if (
            seat not in {"p1", "p2"}
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
            or outcome not in {"win", "tie", "cap", "loss"}
            or not isinstance(capped, bool)
            or seat in seat_scores
        ):
            raise RecoveryError(f"{label} game terminal evidence is malformed.")
        seat_scores[seat] = float(score)
        seat_outcomes[seat] = outcome
        capped_games += int(capped)
    if set(seat_scores) != {"p1", "p2"}:
        raise RecoveryError(f"{label} does not provide both candidate seats.")
    _summary, pair_score = _summary_evidence(directory, seed=seed, label=label)
    if pair_score != (seat_scores["p1"] + seat_scores["p2"]) / 2.0:
        raise RecoveryError(f"{label} summary pair score disagrees with its games.")
    return {
        "seed": seed,
        "directory": str(directory),
        "complete_sha256": _sha256(directory / "COMPLETE.json"),
        "summary_sha256": _sha256(summary_path),
        "pair_score": pair_score,
        "seat_scores": seat_scores,
        "seat_outcomes": seat_outcomes,
        "capped_games": capped_games,
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

    bootstrap = _bootstrap_contract(original_execution)
    scores = [entry["pair_score"] for entry in retained + repaired]
    indices = bootstrap_indices(
        sample_size=len(scores),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    score_interval = bootstrap_mean(
        scores,
        indices,
        confidence_level=float(bootstrap["confidence_level"]),
    )
    margin_over_neutral = {
        key: value - 0.5 for key, value in score_interval.to_payload().items()
    }
    seat_scores = {
        seat: [entry["seat_scores"][seat] for entry in retained + repaired]
        for seat in ("p1", "p2")
    }
    seat_intervals = {
        seat: bootstrap_mean(
            values,
            indices,
            confidence_level=float(bootstrap["confidence_level"]),
        ).to_payload()
        for seat, values in seat_scores.items()
    }
    seat_gap = bootstrap_mean(
        [p1 - p2 for p1, p2 in zip(seat_scores["p1"], seat_scores["p2"], strict=True)],
        indices,
        confidence_level=float(bootstrap["confidence_level"]),
    )
    outcomes = {outcome: 0 for outcome in ("win", "tie", "cap", "loss")}
    capped_games = 0
    for entry in retained + repaired:
        capped_games += int(entry["capped_games"])
        for outcome in entry["seat_outcomes"].values():
            outcomes[outcome] += 1

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
        "candidate_score_95ci": score_interval.to_payload(),
        "candidate_margin_over_neutral_95ci": margin_over_neutral,
        "meets_registered_strength_lower_bound": margin_over_neutral["low"] >= 0.05,
        "seat_sensitivity": {
            "candidate_score_95ci_by_seat": seat_intervals,
            "p1_minus_p2_score_95ci": seat_gap.to_payload(),
        },
        "cap_sensitivity": {
            "capped_games": capped_games,
            "scored_games": 2 * len(scores),
            "capped_game_fraction": capped_games / (2 * len(scores)),
            "candidate_outcomes": outcomes,
        },
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
