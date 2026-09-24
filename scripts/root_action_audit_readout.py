#!/usr/bin/env python3
"""Validate and summarize a completed sealed MCTS root-action audit.

Trials within one source root share the same fixed source boundary. Roots from
the same game and seed are correlated, so this tool reports roots as measurement
units and never promotes within-root continuation trials into game-strength data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


WRAPPER_SCHEMA = "pokezero.mcts-guided-vs-raw-sealed-root-action-audit.v2"
READOUT_SCHEMA = "pokezero.sealed-root-action-audit.v2"
GRID_SCHEMA = "pokezero.sealed-root-action-grid.v2"
OUTPUT_SCHEMA = "pokezero.root-action-audit-readout.v1"
TARGETS = ("policy_consistent", "uniform_own")
TARGET_POLICIES = {
    "policy_consistent": {
        "subject": "sampled_raw_transformer",
        "opponent": "sampled_raw_transformer",
    },
    "uniform_own": {
        "subject": "uniform_legal",
        "opponent": "sampled_raw_transformer",
    },
}
ROOT_COMPLETE_SCHEMA = "pokezero.root-action-audit-recovery-complete.v1"


class ReadoutError(ValueError):
    """The sealed evidence cannot support a root-level readout."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadoutError(f"{label} must be an object")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReadoutError(f"{label} must be finite")
    return float(value)


def _subject_value(winner: object, subject: str) -> float:
    if winner is None:
        return 0.5
    if winner not in {"p1", "p2"}:
        raise ReadoutError("terminal winner must be p1, p2, or null")
    return 1.0 if winner == subject else 0.0


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReadoutError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ReadoutError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def _mean(values: list[float]) -> float:
    if not values:
        raise ReadoutError("cannot summarize an empty value list")
    return sum(values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_bar, y_bar = _mean(xs), _mean(ys)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    x_sum = sum((x - x_bar) ** 2 for x in xs)
    y_sum = sum((y - y_bar) ** 2 for y in ys)
    if x_sum == 0.0 or y_sum == 0.0:
        return None
    return numerator / math.sqrt(x_sum * y_sum)


def _cluster_effects(
    roots: list[dict[str, Any]], target: str, *, key: str
) -> list[dict[str, Any]]:
    """Report correlated root groups without manufacturing a confidence interval."""

    grouped: dict[object, list[float]] = {}
    for row in roots:
        source = row["source"]
        cluster: object
        if key == "source_seed":
            cluster = source["seed"]
        elif key == "source_game":
            cluster = (source["seed"], source["candidate_seat"])
        else:  # pragma: no cover - fixed internal call sites
            raise AssertionError(f"unsupported cluster key {key}")
        grouped.setdefault(cluster, []).append(row["targets"][target]["mcts_minus_raw_trial_mean"])
    result: list[dict[str, Any]] = []
    for cluster, effects in sorted(grouped.items()):
        label = (
            {"seed": cluster}
            if key == "source_seed"
            else {"seed": cluster[0], "candidate_seat": cluster[1]}
        )
        result.append({**label, "root_count": len(effects), "mcts_minus_raw_root_mean": _mean(effects)})
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadoutError(f"cannot read {label}: {exc}") from exc


def _read_root(path: Path) -> dict[str, Any]:
    wrapper = _load_json(path, str(path))
    if set(wrapper) != {
        "schema_version", "candidate_provenance_sha256", "raw_provenance_sha256",
        "candidate_seat", "readout",
    } or wrapper.get("schema_version") != WRAPPER_SCHEMA:
        raise ReadoutError(f"{path}: unsupported wrapper")
    _sha256(wrapper.get("candidate_provenance_sha256"), f"{path}: candidate provenance")
    _sha256(wrapper.get("raw_provenance_sha256"), f"{path}: raw provenance")
    readout = _mapping(wrapper["readout"], f"{path}: readout")
    if set(readout) != {
        "schema_version", "seed", "battle_id", "candidate_seat", "decision_round_index",
        "audit_status", "audit",
    } or readout.get("schema_version") != READOUT_SCHEMA or readout.get("audit_status") != "PAIRED":
        raise ReadoutError(f"{path}: root is not a paired audit")
    subject = readout.get("candidate_seat")
    if subject not in {"p1", "p2"} or wrapper.get("candidate_seat") != subject:
        raise ReadoutError(f"{path}: invalid subject seat")
    seed, round_index = readout.get("seed"), readout.get("decision_round_index")
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(round_index, bool) or not isinstance(round_index, int):
        raise ReadoutError(f"{path}: invalid source address")
    audit = _mapping(readout["audit"], f"{path}: audit")
    required = {
        "schema_version", "source_battle_id", "source_seed", "source_decision_round",
        "subject_player", "opponent_player", "opponent_action_held_fixed", "actions",
        "search_evidence", "continuation_targets",
    }
    if set(audit) != required or audit.get("schema_version") != GRID_SCHEMA:
        raise ReadoutError(f"{path}: unsupported grid")
    if (
        audit.get("source_seed") != seed
        or audit.get("source_decision_round") != round_index
        or audit.get("source_battle_id") != readout.get("battle_id")
        or audit.get("subject_player") != subject
        or audit.get("opponent_player") not in {"p1", "p2"}
        or audit["opponent_player"] == subject
        or audit.get("opponent_action_held_fixed") is not True
    ):
        raise ReadoutError(f"{path}: source binding drift")
    actions = audit.get("actions")
    if not isinstance(actions, list) or len(actions) not in {2, 3}:
        raise ReadoutError(f"{path}: expected two or three actions")
    labels: list[str] = []
    action_indices: list[int] = []
    for row in actions:
        row = _mapping(row, f"{path}: action")
        if set(row) != {"action_label", "action_index"} or row.get("action_label") not in {
            "raw_policy", "mcts_selected", "visit_alternative",
        }:
            raise ReadoutError(f"{path}: malformed action")
        index = row.get("action_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ReadoutError(f"{path}: malformed action index")
        labels.append(row["action_label"])
        action_indices.append(index)
    expected_labels = ["raw_policy", "mcts_selected"] + (
        ["visit_alternative"] if len(actions) == 3 else []
    )
    if labels != expected_labels or len(set(action_indices)) != len(action_indices):
        raise ReadoutError(f"{path}: action order or uniqueness drift")
    evidence = _mapping(audit["search_evidence"], f"{path}: search evidence")
    expected_evidence = {
        "model_argmax", "search_argmax", "model_override", "root_q_gap",
        "root_visit_gap", "root_gap_action_indices", "root_allocation",
    }
    if set(evidence) != expected_evidence or evidence.get("model_override") is not True:
        raise ReadoutError(f"{path}: unsupported search evidence")
    if (
        evidence.get("model_argmax") != action_indices[0]
        or evidence.get("search_argmax") != action_indices[1]
    ):
        raise ReadoutError(f"{path}: actions disagree with search evidence")
    allocation = _mapping(evidence.get("root_allocation"), f"{path}: root allocation")
    if set(allocation) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise ReadoutError(f"{path}: unsupported root allocation")
    worlds = allocation.get("worlds")
    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0 or not isinstance(allocation.get("prior_authority"), bool):
        raise ReadoutError(f"{path}: malformed root allocation")
    if allocation.get("prior_cause") is not None and not isinstance(allocation.get("prior_cause"), str):
        raise ReadoutError(f"{path}: malformed root allocation cause")
    arms = allocation.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ReadoutError(f"{path}: root allocation has no arms")
    arm_by_index: dict[int, Mapping[str, Any]] = {}
    for arm in arms:
        arm = _mapping(arm, f"{path}: root allocation arm")
        if set(arm) != {"action_index", "visit_share", "q", "reported_prior", "model_prior"}:
            raise ReadoutError(f"{path}: unsupported root allocation arm")
        action_index = arm.get("action_index")
        if isinstance(action_index, bool) or not isinstance(action_index, int) or action_index in arm_by_index:
            raise ReadoutError(f"{path}: invalid root allocation action")
        share = _number(arm.get("visit_share"), f"{path}: root allocation visit_share")
        if not 0.0 <= share <= 1.0:
            raise ReadoutError(f"{path}: root allocation visit_share is out of range")
        for name in ("q", "reported_prior", "model_prior"):
            if arm.get(name) is not None:
                _number(arm.get(name), f"{path}: root allocation {name}")
        arm_by_index[action_index] = arm
    if any(index not in arm_by_index for index in action_indices[:2]):
        raise ReadoutError(f"{path}: selected action missing from allocation")
    if len(actions) == 3:
        ranked_other = next(
            (
                arm["action_index"] for arm in sorted(
                    arms, key=lambda arm: (-arm["visit_share"], arm["action_index"])
                )
                if arm["visit_share"] > 0.0 and arm["action_index"] not in action_indices[:2]
            ),
            None,
        )
        if ranked_other != action_indices[2]:
            raise ReadoutError(f"{path}: visit alternative disagrees with allocation")
    gap_actions = evidence.get("root_gap_action_indices")
    if (
        not isinstance(gap_actions, list)
        or len(gap_actions) != 2
        or any(isinstance(action, bool) or not isinstance(action, int) for action in gap_actions)
        or len(set(gap_actions)) != 2
        or any(action not in arm_by_index for action in gap_actions)
    ):
        raise ReadoutError(f"{path}: malformed leading-arm gap witness")
    gap_arms = [arm_by_index[action] for action in gap_actions]
    if any(arm["visit_share"] <= 0.0 for arm in gap_arms):
        raise ReadoutError(f"{path}: leading-arm gap witness includes a zero-visit action")
    positive_visit_shares = sorted(
        (arm["visit_share"] for arm in arms if arm["visit_share"] > 0.0), reverse=True
    )
    if (
        len(positive_visit_shares) < 2
        or sorted((arm["visit_share"] for arm in gap_arms), reverse=True)
        != positive_visit_shares[:2]
        or gap_arms[0]["visit_share"] < gap_arms[1]["visit_share"]
    ):
        raise ReadoutError(f"{path}: leading-arm gap witness disagrees with allocation")
    q_gap = _number(evidence.get("root_q_gap"), f"{path}: root_q_gap")
    visit_gap = _number(evidence.get("root_visit_gap"), f"{path}: root_visit_gap")
    expected_visit_gap = gap_arms[0]["visit_share"] - gap_arms[1]["visit_share"]
    expected_q_gap = (
        abs(gap_arms[0]["q"] - gap_arms[1]["q"])
        if gap_arms[0]["q"] is not None and gap_arms[1]["q"] is not None
        else None
    )
    if (
        not math.isclose(visit_gap, expected_visit_gap, abs_tol=1e-5)
        or expected_q_gap is None
        or not math.isclose(q_gap, expected_q_gap, abs_tol=1e-5)
    ):
        raise ReadoutError(f"{path}: root gaps disagree with leading-arm allocation witness")
    target_rows = audit.get("continuation_targets")
    if not isinstance(target_rows, list) or [row.get("target") for row in target_rows if isinstance(row, Mapping)] != list(TARGETS):
        raise ReadoutError(f"{path}: target order drift")
    target_summary: dict[str, dict[str, Any]] = {}
    for target_row in target_rows:
        target_row = _mapping(target_row, f"{path}: target")
        trials = target_row.get("trials")
        if set(target_row) != {"target", "trials"} or not isinstance(trials, list) or len(trials) != 16:
            raise ReadoutError(f"{path}: target trials are incomplete")
        trial_seeds: set[int] = set()
        values = {label: [] for label in labels}
        paired_deltas: list[float] = []
        continuation_envelopes: set[tuple[int, int, bool]] = set()
        for trial in trials:
            trial = _mapping(trial, f"{path}: trial")
            trial_seed = trial.get("continuation_rng_seed")
            outcomes = trial.get("outcomes")
            if (
                set(trial) != {"continuation_rng_seed", "outcomes"}
                or isinstance(trial_seed, bool) or not isinstance(trial_seed, int) or trial_seed < 0
                or trial_seed in trial_seeds or not isinstance(outcomes, list) or len(outcomes) != len(actions)
            ):
                raise ReadoutError(f"{path}: malformed trial")
            trial_seeds.add(trial_seed)
            trial_values: dict[str, float] = {}
            for outcome, action in zip(outcomes, actions):
                outcome = _mapping(outcome, f"{path}: outcome")
                if outcome.get("action_label") != action["action_label"] or outcome.get("action_index") != action["action_index"]:
                    raise ReadoutError(f"{path}: outcome/action mismatch")
                continuation = _mapping(outcome.get("continuation"), f"{path}: continuation")
                terminal = _mapping(continuation.get("terminal"), f"{path}: terminal")
                if set(outcome) != {"action_label", "action_index", "continuation"} or set(continuation) != {
                    "decision_round_count", "terminal_after_fixed_joint_step", "terminal",
                    "initial_max_continuation_decision_rounds", "effective_max_continuation_decision_rounds", "cap_retry",
                } or set(terminal) != {"winner", "turn_count", "capped"}:
                    raise ReadoutError(f"{path}: unsupported continuation")
                count = continuation["decision_round_count"]
                initial, effective = continuation["initial_max_continuation_decision_rounds"], continuation["effective_max_continuation_decision_rounds"]
                if (
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                    or isinstance(initial, bool) or not isinstance(initial, int) or initial <= 0
                    or isinstance(effective, bool) or not isinstance(effective, int) or effective < initial
                    or continuation["terminal_after_fixed_joint_step"] is not (count == 0)
                    or not isinstance(continuation["cap_retry"], bool)
                    or continuation["cap_retry"] is not (effective > initial)
                    or (continuation["cap_retry"] and count <= initial)
                    or count > effective or terminal.get("capped") is not False
                    or isinstance(terminal.get("turn_count"), bool) or not isinstance(terminal.get("turn_count"), int) or terminal["turn_count"] < 0
                ):
                    raise ReadoutError(f"{path}: incomplete or capped continuation")
                value = _subject_value(terminal.get("winner"), subject)
                continuation_envelopes.add((initial, effective, continuation["cap_retry"]))
                values[action["action_label"]].append(value)
                trial_values[action["action_label"]] = value
            paired_deltas.append(trial_values["mcts_selected"] - trial_values["raw_policy"])
        target_summary[target_row["target"]] = {
            "trials": len(trials),
            "continuation_rng_seeds": sorted(trial_seeds),
            "continuation_envelopes": sorted(continuation_envelopes),
            "action_mean_subject_win": {label: _mean(values[label]) for label in labels},
            "mcts_minus_raw_trial_mean": _mean(paired_deltas),
        }
    return {
        "provenance": {
            "candidate": wrapper["candidate_provenance_sha256"],
            "raw": wrapper["raw_provenance_sha256"],
        },
        "source": {
            "seed": seed, "battle_id": readout["battle_id"], "candidate_seat": subject,
            "decision_round_index": round_index,
        },
        "actions": actions,
        # These diagnostics quantify the native top-two positive-visit arms,
        # not necessarily the raw-policy and selected-MCTS pair above. Keep
        # the witness with every row so downstream correlation analysis cannot
        # silently relabel it as a raw-vs-MCTS margin.
        "root_gap_semantics": "top_two_positive_visit_arms",
        "root_gap_action_indices": list(gap_actions),
        "root_q_gap": q_gap,
        "root_visit_gap": visit_gap,
        "targets": target_summary,
    }


def _manifest_contract(root: Path, seed: int) -> dict[str, Any]:
    """Read the immutable per-seed registration that names its audit roots."""

    manifest_path = root / "seeds" / f"seed-{seed}" / "input" / "MANIFEST.json"
    manifest = _load_json(manifest_path, f"seed {seed} manifest")
    if manifest.get("seeds") != [seed]:
        raise ReadoutError(f"seed {seed}: manifest seed binding drift")
    audit = _mapping(manifest.get("sealed_root_action_audit"), f"seed {seed}: audit registration")
    if audit.get("schema_version") != "pokezero.mcts-guided-vs-raw-sealed-root-action-audit.v2":
        raise ReadoutError(f"seed {seed}: unsupported audit registration")
    rows = audit.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ReadoutError(f"seed {seed}: missing registered roots")
    addresses: set[tuple[int, str, int]] = set()
    for row in rows:
        row = _mapping(row, f"seed {seed}: registered root")
        if set(row) != {"seed", "candidate_seat", "decision_round_index"}:
            raise ReadoutError(f"seed {seed}: malformed registered root")
        row_seed, seat, round_index = row.get("seed"), row.get("candidate_seat"), row.get("decision_round_index")
        if (
            isinstance(row_seed, bool) or not isinstance(row_seed, int) or row_seed != seed
            or seat not in {"p1", "p2"}
            or isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0
        ):
            raise ReadoutError(f"seed {seed}: invalid registered root")
        address = (row_seed, seat, round_index)
        if address in addresses:
            raise ReadoutError(f"seed {seed}: duplicate registered root")
        addresses.add(address)
    rng_seeds = audit.get("continuation_rng_seeds")
    if (
        not isinstance(rng_seeds, list) or len(rng_seeds) != 16
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in rng_seeds)
        or len(set(rng_seeds)) != len(rng_seeds)
    ):
        raise ReadoutError(f"seed {seed}: invalid registered continuation RNG schedule")
    targets = _mapping(audit.get("continuation_targets"), f"seed {seed}: continuation targets")
    if set(targets) != set(TARGETS) or any(
        not isinstance(targets[name], Mapping) or dict(targets[name]) != TARGET_POLICIES[name]
        for name in TARGETS
    ):
        raise ReadoutError(f"seed {seed}: continuation target registration drift")
    initial, expanded = audit.get("max_continuation_decision_rounds"), audit.get(
        "expanded_max_continuation_decision_rounds"
    )
    if (
        isinstance(initial, bool) or not isinstance(initial, int) or initial <= 0
        or isinstance(expanded, bool) or not isinstance(expanded, int) or expanded <= initial
    ):
        raise ReadoutError(f"seed {seed}: invalid continuation ceilings")
    return {
        "addresses": addresses,
        "rng_seeds": tuple(sorted(rng_seeds)),
        "initial_ceiling": initial,
        "expanded_ceiling": expanded,
    }


def summarize(root: Path, *, expected_roots: int) -> dict[str, Any]:
    paths = sorted(root.glob("seeds/seed-*/sealed-root-action-audits/seed-*/*.json"))
    if len(paths) != expected_roots:
        raise ReadoutError(f"expected exactly {expected_roots} root sidecars, found {len(paths)}")
    roots = [_read_root(path) for path in paths]
    addresses = {(row["source"]["seed"], row["source"]["candidate_seat"], row["source"]["decision_round_index"]) for row in roots}
    if len(addresses) != len(roots):
        raise ReadoutError("duplicate source root")
    source_seeds = sorted({row["source"]["seed"] for row in roots})
    contracts = {seed: _manifest_contract(root, seed) for seed in source_seeds}
    expected_addresses = set().union(*(contract["addresses"] for contract in contracts.values()))
    if len(expected_addresses) != expected_roots or addresses != expected_addresses:
        raise ReadoutError("sidecars do not match the registered root roster")
    for path, row in zip(paths, roots):
        source = row["source"]
        expected_path = (
            root / "seeds" / f"seed-{source['seed']}" / "sealed-root-action-audits"
            / f"seed-{source['seed']}-{source['candidate_seat']}"
            / f"round-{source['decision_round_index']:04d}.json"
        )
        if path != expected_path:
            raise ReadoutError(f"{path}: sidecar path/source address drift")
        contract = contracts[source["seed"]]
        for target in TARGETS:
            if tuple(row["targets"][target]["continuation_rng_seeds"]) != contract["rng_seeds"]:
                raise ReadoutError(f"{path}: {target} continuation RNG schedule disagrees with manifest")
            permitted_envelopes = {
                (contract["initial_ceiling"], contract["initial_ceiling"], False),
                (contract["initial_ceiling"], contract["expanded_ceiling"], True),
            }
            if not set(row["targets"][target]["continuation_envelopes"]).issubset(permitted_envelopes):
                raise ReadoutError(f"{path}: {target} continuation ceilings disagree with manifest")
    expected_complete = {
        "schema_version": ROOT_COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "seeds": source_seeds,
        "pairs": len(source_seeds),
        "games": 2 * len(source_seeds),
    }
    complete_path = root / "COMPLETE.json"
    complete_raw = complete_path.read_bytes() if complete_path.is_file() else b""
    complete = _load_json(complete_path, "root COMPLETE receipt")
    if dict(complete) != expected_complete:
        raise ReadoutError("root COMPLETE receipt is not the declared complete audit")
    all_provenance = {(row["provenance"]["candidate"], row["provenance"]["raw"]) for row in roots}
    if len(all_provenance) != 1:
        raise ReadoutError("candidate/raw provenance drifts across root sidecars")
    for source_seed in source_seeds:
        seed_complete = _load_json(
            root / "seeds" / f"seed-{source_seed}" / "COMPLETE.json",
            f"seed {source_seed} COMPLETE receipt",
        )
        if seed_complete.get("status") != "COMPLETE" or seed_complete.get("pairs") != 1 or seed_complete.get("games") != 2:
            raise ReadoutError(f"seed {source_seed} is not a complete paired game")
        expected_provenance = next(iter(all_provenance))
        actual_provenance = (
            _sha256(seed_complete.get("candidate_provenance_sha256"), f"seed {source_seed}: candidate provenance"),
            _sha256(seed_complete.get("raw_provenance_sha256"), f"seed {source_seed}: raw provenance"),
        )
        if actual_provenance != expected_provenance:
            raise ReadoutError(f"seed {source_seed}: completion provenance disagrees with sidecars")
    aggregate: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        schedules = {tuple(row["targets"][target]["continuation_rng_seeds"]) for row in roots}
        if len(schedules) != 1:
            raise ReadoutError(f"{target}: continuation RNG schedules drift across roots")
        effects = [row["targets"][target]["mcts_minus_raw_trial_mean"] for row in roots]
        q_gaps = [row["root_q_gap"] for row in roots]
        visit_gaps = [row["root_visit_gap"] for row in roots]
        aggregate[target] = {
            "measurement_unit": "source_root",
            "correlation_clusters": {"source_seed_count": len({row["source"]["seed"] for row in roots}), "source_game_count": len({(row["source"]["seed"], row["source"]["candidate_seat"]) for row in roots})},
            "root_count": len(roots),
            "continuation_rng_seeds": list(next(iter(schedules))),
            "mcts_minus_raw_root_mean": _mean(effects),
            "mcts_better_roots": sum(effect > 0.0 for effect in effects),
            "raw_better_roots": sum(effect < 0.0 for effect in effects),
            "tied_roots": sum(effect == 0.0 for effect in effects),
            "pearson_root_q_gap_vs_effect": _pearson(q_gaps, effects),
            "pearson_root_visit_gap_vs_effect": _pearson(visit_gaps, effects),
            "by_source_seed": _cluster_effects(roots, target, key="source_seed"),
            "by_source_game": _cluster_effects(roots, target, key="source_game"),
        }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "source_root": str(root),
        "source_complete_sha256": hashlib.sha256(complete_raw).hexdigest(),
        "expected_root_count": expected_roots,
        "measurement_unit": "source_root",
        "correlation_note": "Roots from the same source game and source seed are correlated; this readout reports no independent-root confidence interval.",
        "targets": aggregate,
        "roots": roots,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    value = {**payload, "sha256": digest}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ReadoutError(f"refusing to replace existing output {path}") from exc
    finally:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-roots", type=int, default=16)
    args = parser.parse_args()
    if args.expected_roots <= 0:
        raise SystemExit("--expected-roots must be positive")
    try:
        result = summarize(args.root, expected_roots=args.expected_roots)
        _write_create_only(args.out, result)
    except ReadoutError as exc:
        raise SystemExit(f"ROOT ACTION AUDIT READOUT REFUSED: {exc}") from exc
    print(f"WROTE ROOT ACTION AUDIT READOUT {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
