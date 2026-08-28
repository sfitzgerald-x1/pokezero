#!/usr/bin/env python3
"""Validate a replay-backed oracle-label corpus before a value-only Stage-1 run.

The root-policy-continuation bank is a *label ledger*: it records source seeds,
decision indexes, fixed opponent actions, and policy-continuation values.  It
does not contain successor observations.  A seed alone is not a training
example, so this verifier makes the missing join explicit and refuses a corpus
unless every non-terminal candidate label is bound to a replayed successor
observation in an immutable training cache.

The producer is intentionally separate.  It must replay the pinned source
runtime, materialise each fixed-opponent joint successor, and write ordinary
``pokezero.training_cache.v2`` caches.  This tool is the reader-side fence:
it checks the bank, source contract, registered seed split, manifest, cache
tree digests, cache row targets, and the one-to-one candidate coverage.  Thus a
bank-only, seed-only, partial, or re-labelled cache cannot be passed to Stage 1
as an oracle corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CORPUS_SCHEMA = "pokezero.expert-iteration.oracle-label-corpus.v1"
ROOT_BANK_SCHEMA = "pokezero.root-policy-continuation-oracle-bank.v1"
SPLIT_SCHEMA = "pokezero.expert-iteration.p0-label-splits.v1"
TRAINING_CACHE_SCHEMA = "pokezero.training_cache.v2"
BRANCH_RULE = "bank-selected-action-fixed-opponent-joint-successor.v1"


class CorpusError(ValueError):
    """Raised when bytes cannot support an expert-iteration oracle corpus."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise CorpusError(f"{label} must be an array")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusError(f"{label} must be an integer")
    return value


def _finite_probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusError(f"{label} must be a finite probability")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise CorpusError(f"{label} must be a finite probability in [0, 1]")
    return parsed


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CorpusError(f"{label} must be a regular file: {path}")
    return path.read_bytes()


def _json(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _regular_bytes(path, label=label)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{label} is not JSON: {exc}") from exc
    return _mapping(value, label=label), raw


def training_cache_tree_sha256(path: Path) -> str:
    """Digest the complete cache directory, with names included and links refused."""

    if not path.is_dir() or path.is_symlink():
        raise CorpusError(f"training cache must be a non-symlink directory: {path}")
    digest = hashlib.sha256()
    files = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    for item in files:
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise CorpusError(f"training cache contains symlink: {relative}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise CorpusError(f"training cache contains non-regular entry: {relative}")
        raw = item.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CorpusError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CorpusError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def _git_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise CorpusError(f"{label} must be a 40-character git commit")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CorpusError(f"{label} must be a 40-character git commit") from exc
    return value


def _cache_arrays(path: Path, *, label: str) -> tuple[Any, Any, Any, int]:
    """Read only the target/address arrays needed to prove the label join."""

    metadata, _ = _json(path / "metadata.json", label=f"{label}.metadata")
    if metadata.get("schema_version") != TRAINING_CACHE_SCHEMA:
        raise CorpusError(f"{label} has unsupported training-cache schema")
    example_count = _integer(metadata.get("example_count"), label=f"{label}.metadata.example_count")
    if example_count < 0:
        raise CorpusError(f"{label}.metadata.example_count must be non-negative")
    try:
        import numpy
    except ModuleNotFoundError as exc:  # pragma: no cover - installation error
        raise CorpusError("NumPy is required to validate a training cache") from exc
    arrays = []
    for name in ("returns", "seeds", "turn_indices"):
        array_path = path / f"{name}.npy"
        if not array_path.is_file() or array_path.is_symlink():
            raise CorpusError(f"{label} lacks regular {name}.npy")
        array = numpy.load(array_path, mmap_mode="r", allow_pickle=False)
        if len(array) != example_count:
            raise CorpusError(f"{label}.{name}.npy length disagrees with metadata.example_count")
        arrays.append(array)
    return arrays[0], arrays[1], arrays[2], example_count


def _validated_contract(contract: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    experiment_id = contract.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CorpusError("contract.experiment_id must be a non-empty string")
    runtime = _mapping(contract.get("runtime"), label="contract.runtime")
    estimator = _mapping(contract.get("estimator"), label="contract.estimator")
    sample = _mapping(contract.get("sample"), label="contract.sample")
    if contract.get("status") != "REGISTERED-NOT-RUN":
        raise CorpusError("contract is not the registered pre-run source contract")
    if sample.get("analysis_unit") != "source_game_seed":
        raise CorpusError("contract.sample.analysis_unit must be source_game_seed")
    if _integer(sample.get("seed_pairs"), label="contract.sample.seed_pairs") <= 0:
        raise CorpusError("contract.sample.seed_pairs must be positive")
    _hex(runtime.get("checkpoint_sha256"), label="contract.runtime.checkpoint_sha256")
    for field in ("public_source_commit", "showdown_commit"):
        _git_commit(runtime.get(field), label=f"contract.runtime.{field}")
    expected = {
        "subject": "p1",
        "opponent": "raw-transformer-policy",
        "top_k_legal_priors": 3,
        "rollouts_per_action": 16,
        "paired_continuation_policy_rng_across_actions": True,
        "leaf_estimator": "vhprobe-policy-continuation-v1",
        "uniform_leaves_excluded": True,
        "capped_source_or_continuation_policy": "fail-shard-never-label",
    }
    for field, expected_value in expected.items():
        if estimator.get(field) != expected_value:
            raise CorpusError(f"contract.estimator.{field} differs from the registered oracle")
    for field in ("source_max_decision_rounds", "continuation_max_decision_rounds"):
        if _integer(estimator.get(field), label=f"contract.estimator.{field}") <= 0:
            raise CorpusError(f"contract.estimator.{field} must be positive")
    return experiment_id, runtime, estimator


def _bank_candidates(
    bank: Mapping[str, Any], *, experiment_id: str, runtime: Mapping[str, Any], estimator: Mapping[str, Any]
) -> dict[tuple[int, int, int], Mapping[str, Any]]:
    if bank.get("schema") != ROOT_BANK_SCHEMA:
        raise CorpusError("bank has an unsupported root-oracle schema")
    if bank.get("experiment_id") != experiment_id:
        raise CorpusError("bank experiment_id differs from contract")
    provenance = _mapping(bank.get("provenance"), label="bank.provenance")
    for contract_field, bank_field in (
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("public_source_commit", "public_source_commit"),
        ("showdown_commit", "showdown_commit"),
    ):
        if provenance.get(bank_field) != runtime[contract_field]:
            raise CorpusError(f"bank.provenance.{bank_field} differs from contract runtime")
    expected_provenance = {
        "analysis_unit": "source_game_seed",
        "top_k": estimator["top_k_legal_priors"],
        "rollouts_per_action": estimator["rollouts_per_action"],
        "source_max_decision_rounds": estimator["source_max_decision_rounds"],
        "leaf_estimator": estimator["leaf_estimator"],
        "uniform_leaves_excluded": True,
    }
    for field, expected_value in expected_provenance.items():
        if provenance.get(field) != expected_value:
            raise CorpusError(f"bank.provenance.{field} differs from contract")
    records: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    seen_source_decisions: set[tuple[int, int]] = set()
    for pair_number, pair_value in enumerate(_sequence(bank.get("pairs"), label="bank.pairs")):
        pair = _mapping(pair_value, label=f"bank.pairs[{pair_number}]")
        seed = _integer(pair.get("seed"), label=f"bank.pairs[{pair_number}].seed")
        if seed < 0:
            raise CorpusError(f"bank.pairs[{pair_number}].seed must be non-negative")
        for decision_number, decision_value in enumerate(
            _sequence(pair.get("oracle_decisions"), label=f"bank.pairs[{pair_number}].oracle_decisions")
        ):
            decision = _mapping(decision_value, label=f"bank decision {seed}/{decision_number}")
            decision_index = _integer(decision.get("decision_index"), label=f"bank decision {seed}/{decision_number}.decision_index")
            source_key = (seed, decision_index)
            if source_key in seen_source_decisions:
                raise CorpusError(f"bank repeats source decision {source_key}")
            seen_source_decisions.add(source_key)
            selected = _integer(decision.get("selected_action"), label=f"bank decision {source_key}.selected_action")
            fixed = _mapping(decision.get("opponent_actions_fixed_before_selection"), label=f"bank decision {source_key}.opponent_actions_fixed_before_selection")
            if set(fixed) != {"p2"} or _integer(fixed.get("p2"), label=f"bank decision {source_key}.opponent_actions_fixed_before_selection.p2") < 0:
                raise CorpusError(f"bank decision {source_key} must bind exactly one non-negative p2 action")
            candidates = _sequence(decision.get("candidate_scores"), label=f"bank decision {source_key}.candidate_scores")
            if decision.get("candidate_count") != len(candidates) or not 1 <= len(candidates) <= int(estimator["top_k_legal_priors"]):
                raise CorpusError(f"bank decision {source_key} has invalid candidate geometry")
            actions: set[int] = set()
            selected_value: float | None = None
            for candidate_number, candidate_value in enumerate(candidates):
                candidate = _mapping(candidate_value, label=f"bank candidate {source_key}/{candidate_number}")
                action = _integer(candidate.get("action"), label=f"bank candidate {source_key}/{candidate_number}.action")
                if action < 0 or action in actions:
                    raise CorpusError(f"bank decision {source_key} has duplicate/negative candidate action")
                actions.add(action)
                shortcut = candidate.get("terminal_shortcut")
                if shortcut not in (True, False):
                    raise CorpusError(f"bank candidate {source_key}/{action} lacks terminal-shortcut provenance")
                completed = candidate.get("rollouts_completed")
                if shortcut is True and completed != 0:
                    raise CorpusError(f"bank terminal candidate {source_key}/{action} has rollout trials")
                if shortcut is False and completed != estimator["rollouts_per_action"]:
                    raise CorpusError(f"bank non-terminal candidate {source_key}/{action} is not R={estimator['rollouts_per_action']}")
                target = _finite_probability(candidate.get("policy_continuation_value"), label=f"bank candidate {source_key}/{action}.policy_continuation_value")
                key = (seed, decision_index, action)
                records[key] = {
                    "target": target,
                    "terminal_shortcut": shortcut,
                    "selected_action": selected,
                    "fixed_opponent_action": fixed["p2"],
                }
                if action == selected:
                    selected_value = target
            if selected not in actions:
                raise CorpusError(f"bank decision {source_key} selected an unscored action")
            if not math.isclose(
                _finite_probability(decision.get("selected_policy_continuation_value"), label=f"bank decision {source_key}.selected_policy_continuation_value"),
                float(selected_value), abs_tol=1e-12, rel_tol=0.0,
            ):
                raise CorpusError(f"bank decision {source_key} selected label differs from candidate label")
    if not records:
        raise CorpusError("bank contains no oracle candidates")
    return records


def _validate_split(splits: Mapping[str, Any], *, bank_sha256: str, source_seeds: set[int]) -> None:
    if splits.get("schema") != SPLIT_SCHEMA:
        raise CorpusError("split record has an unsupported schema")
    bank = _mapping(splits.get("bank"), label="splits.bank")
    if bank.get("sha256") != bank_sha256:
        raise CorpusError("split record binds a different bank")
    rule = _mapping(splits.get("registered_rule"), label="splits.registered_rule")
    if rule != {"heldout": "source_game_seed % 4 == 0", "train": "source_game_seed % 4 != 0"}:
        raise CorpusError("split record does not carry the registered source-game seed rule")
    heldout = {_integer(value, label="splits.heldout_source_game_seeds entry") for value in _sequence(splits.get("heldout_source_game_seeds"), label="splits.heldout_source_game_seeds")}
    train = {_integer(value, label="splits.train_source_game_seeds entry") for value in _sequence(splits.get("train_source_game_seeds"), label="splits.train_source_game_seeds")}
    if train & heldout or train | heldout != source_seeds:
        raise CorpusError("split record is not a partition of the bank source-game seeds")
    if any(seed % 4 == 0 for seed in train) or any(seed % 4 != 0 for seed in heldout):
        raise CorpusError("split record violates its registered seed-modulo rule")


def _validate_corpus_runtime(corpus: Mapping[str, Any], runtime: Mapping[str, Any], estimator: Mapping[str, Any]) -> None:
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise CorpusError("corpus has an unsupported schema")
    corpus_runtime = _mapping(corpus.get("runtime"), label="corpus.runtime")
    for field in ("checkpoint_sha256", "public_source_commit", "showdown_commit"):
        if corpus_runtime.get(field) != runtime[field]:
            raise CorpusError(f"corpus.runtime.{field} differs from contract runtime")
    replay = _mapping(corpus.get("replay"), label="corpus.replay")
    expected = {
        "subject": estimator["subject"],
        "opponent_policy": estimator["opponent"],
        "sampling_temperature": 1.0,
        "deterministic": False,
        "source_max_decision_rounds": estimator["source_max_decision_rounds"],
        "branch_rule": BRANCH_RULE,
    }
    for field, expected_value in expected.items():
        if replay.get(field) != expected_value:
            raise CorpusError(f"corpus.replay.{field} differs from registered source replay")


def validate(
    *, bank_path: Path, contract_path: Path, split_path: Path, corpus_path: Path,
    train_cache_path: Path, heldout_cache_path: Path,
) -> dict[str, int | str]:
    contract, _ = _json(contract_path, label="contract")
    experiment_id, runtime, estimator = _validated_contract(contract)
    bank, bank_raw = _json(bank_path, label="bank")
    bank_sha256 = _sha256(bank_raw)
    candidates = _bank_candidates(bank, experiment_id=experiment_id, runtime=runtime, estimator=estimator)
    _validate_split(
        _json(split_path, label="splits")[0],
        bank_sha256=bank_sha256,
        source_seeds={key[0] for key in candidates},
    )
    corpus, _ = _json(corpus_path, label="corpus")
    _validate_corpus_runtime(corpus, runtime, estimator)
    corpus_bank = _mapping(corpus.get("bank"), label="corpus.bank")
    if corpus_bank.get("sha256") != bank_sha256:
        raise CorpusError("corpus binds a different bank")
    declared_caches = _mapping(corpus.get("training_caches"), label="corpus.training_caches")
    actual_caches = {"train": train_cache_path, "heldout": heldout_cache_path}
    cache_arrays: dict[str, tuple[Any, Any, Any, int]] = {}
    for split, path in actual_caches.items():
        declared = _mapping(declared_caches.get(split), label=f"corpus.training_caches.{split}")
        if _hex(declared.get("tree_sha256"), label=f"corpus.training_caches.{split}.tree_sha256") != training_cache_tree_sha256(path):
            raise CorpusError(f"{split} training-cache tree digest differs from corpus manifest")
        cache_arrays[split] = _cache_arrays(path, label=f"{split} training_cache")
        if declared.get("example_count") != cache_arrays[split][3]:
            raise CorpusError(f"{split} training-cache example count differs from corpus manifest")
    seen: set[tuple[int, int, int]] = set()
    cache_rows: dict[str, set[int]] = {"train": set(), "heldout": set()}
    terminal_shortcuts = 0
    for index, record_value in enumerate(_sequence(corpus.get("records"), label="corpus.records")):
        record = _mapping(record_value, label=f"corpus.records[{index}]")
        split = record.get("split")
        if split not in actual_caches:
            raise CorpusError(f"corpus.records[{index}].split must be train or heldout")
        seed = _integer(record.get("source_game_seed"), label=f"corpus.records[{index}].source_game_seed")
        decision = _integer(record.get("decision_index"), label=f"corpus.records[{index}].decision_index")
        action = _integer(record.get("candidate_action"), label=f"corpus.records[{index}].candidate_action")
        key = (seed, decision, action)
        expected = candidates.get(key)
        if expected is None:
            raise CorpusError(f"corpus record {key} is not an oracle-bank candidate")
        if key in seen:
            raise CorpusError(f"corpus repeats oracle-bank candidate {key}")
        seen.add(key)
        if (split == "heldout") != (seed % 4 == 0):
            raise CorpusError(f"corpus record {key} is assigned to the wrong registered split")
        if record.get("selected_action") != expected["selected_action"]:
            raise CorpusError(f"corpus record {key} differs from bank selected action")
        fixed = _mapping(record.get("fixed_opponent_actions"), label=f"corpus record {key}.fixed_opponent_actions")
        if fixed != {"p2": expected["fixed_opponent_action"]}:
            raise CorpusError(f"corpus record {key} differs from bank fixed opponent action")
        target = _finite_probability(record.get("target"), label=f"corpus record {key}.target")
        if not math.isclose(target, float(expected["target"]), abs_tol=1e-12, rel_tol=0.0):
            raise CorpusError(f"corpus record {key} target differs from bank label")
        shortcut = record.get("terminal_shortcut")
        if shortcut is not expected["terminal_shortcut"]:
            raise CorpusError(f"corpus record {key} terminal-shortcut status differs from bank")
        if shortcut:
            terminal_shortcuts += 1
            if record.get("cache_index") is not None or record.get("source_state_sha256") is not None or record.get("successor_observation_sha256") is not None:
                raise CorpusError(f"terminal corpus record {key} must not claim a trainable successor observation")
            continue
        _hex(record.get("source_state_sha256"), label=f"corpus record {key}.source_state_sha256")
        _hex(record.get("successor_observation_sha256"), label=f"corpus record {key}.successor_observation_sha256")
        row = _integer(record.get("cache_index"), label=f"corpus record {key}.cache_index")
        returns, seeds, turns, count = cache_arrays[str(split)]
        if not 0 <= row < count or row in cache_rows[str(split)]:
            raise CorpusError(f"corpus record {key} has duplicate/out-of-range cache_index")
        cache_rows[str(split)].add(row)
        if int(seeds[row]) != seed or int(turns[row]) != decision:
            raise CorpusError(f"corpus record {key} does not address its declared cache row")
        if not math.isclose(float(returns[row]), target, abs_tol=2e-7, rel_tol=0.0):
            raise CorpusError(f"corpus record {key} cache return differs from oracle label")
    if seen != set(candidates):
        missing = len(set(candidates) - seen)
        extra = len(seen - set(candidates))
        raise CorpusError(f"corpus does not cover the exact bank candidate set (missing={missing}, extra={extra})")
    for split, arrays in cache_arrays.items():
        if cache_rows[split] != set(range(arrays[3])):
            raise CorpusError(f"{split} cache contains an unbound or missing manifest row")
    return {
        "experiment_id": experiment_id,
        "bank_sha256": bank_sha256,
        "bank_candidates": len(candidates),
        "terminal_shortcuts": terminal_shortcuts,
        "train_examples": cache_arrays["train"][3],
        "heldout_examples": cache_arrays["heldout"][3],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--heldout-cache", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate(
        bank_path=args.bank,
        contract_path=args.contract,
        split_path=args.splits,
        corpus_path=args.corpus,
        train_cache_path=args.train_cache,
        heldout_cache_path=args.heldout_cache,
    )
    print(
        "VALID EXPERT-ITERATION ORACLE LABEL CORPUS: "
        f"experiment={summary['experiment_id']} candidates={summary['bank_candidates']} "
        f"terminal_shortcuts={summary['terminal_shortcuts']} train={summary['train_examples']} "
        f"heldout={summary['heldout_examples']} bank_sha256={summary['bank_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorpusError as exc:
        print(f"EXPERT-ITERATION ORACLE LABEL CORPUS REFUSED: {exc}")
        raise SystemExit(2)
