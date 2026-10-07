"""Frozen source identities for the bounded policy-opponent root screen.

This validates selection and replay-input bytes, not replay eligibility or
strength. Refused roots remain in the original denominator. No run may adopt a
different root, prefix witness, or outcome-conditioned replacement silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "pokezero.paper-policy-opponent-roster.v1"
STATUS = "FROZEN_SELECTION_PENDING_REPLAY_AND_COVERAGE_QUALIFICATION"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RosterError(ValueError):
    """A registered selection or its source bytes no longer match."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RosterError(message)


def _reference(row: Any, shard: str) -> tuple[int, str, int]:
    _require(isinstance(row, Mapping), "source reference must be an object")
    seed, seat, turn = row.get("seed"), row.get("seat"), row.get("turn_index")
    _require(type(seed) is int and seed >= 0, "invalid source seed")
    _require(seat in ("p1", "p2"), "invalid source seat")
    _require(type(turn) is int and turn >= 0, "invalid source turn")
    decision = row.get("decision_id")
    _require(isinstance(decision, str) and SHA256.fullmatch(decision) is not None,
             "invalid decision identity")
    digest = row.get("source_file_sha256")
    _require(isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
             "invalid source file hash")
    _require(row.get("battle_id") == f"mcts-h2h-{seed}-{seat}", "source battle identity drift")
    expected_path = (
        f"shards/{shard}/public-decision-records/seed-{seed}-{seat}/"
        f"turn-{turn:03d}-{decision}.json"
    )
    _require(row.get("source_relative_path") == expected_path, "source relative path drift")
    return seed, seat, turn


def validate_roster(payload: Any) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), "roster must be an object")
    _require(payload.get("schema_version") == SCHEMA, "unsupported roster schema")
    _require(payload.get("status") == STATUS, "selection receipt is not an execution receipt")
    _require(payload.get("source_cohort") == "mcts-guided-raw-df5-20260930-pilot-r3",
             "registered source cohort drift")
    _require(payload.get("source_shard") == "s0", "registered source shard drift")
    terminal = payload.get("source_terminal_sha256")
    _require(isinstance(terminal, str) and SHA256.fullmatch(terminal) is not None,
             "missing source terminal hash")
    selection = payload.get("selection_rule")
    _require(isinstance(selection, Mapping), "missing selection rule")
    _require(selection.get("paired_seeds") == list(range(2026100100, 2026100108))
             and all(type(seed) is int for seed in selection["paired_seeds"]),
             "registered paired seed selection drift")
    _require(selection.get("seats") == ["p1", "p2"]
             and selection.get("decision_indices") == [0, 9]
             and all(type(turn) is int for turn in selection["decision_indices"]),
             "registered root indices drift")
    _require(selection.get("selected_without_candidate_actions_or_outcomes") is True,
             "selection must be outcome independent")
    _require(payload.get("planned_comparison_arms") ==
             ["raw_policy", "incumbent_mcts", "own_policy_opponent_mcts"],
             "comparison arms drift")
    roots = payload.get("profile_roots")
    _require(isinstance(roots, list) and len(roots) == 32, "profile denominator must be 32")
    expected = [(seed, seat, turn) for seed in selection["paired_seeds"]
                for seat in selection["seats"] for turn in selection["decision_indices"]]
    actual = [_reference(row, "s0") for row in roots]
    _require(actual == expected, "root selection/order drift")
    _require(len({row["decision_id"] for row in roots}) == 32, "duplicate root identity")
    continuation = [row["decision_id"] for row in roots if row["turn_index"] == 9]
    _require(payload.get("proposed_continuation_root_ids") == continuation,
             "continuation selection drift")
    prefixes = payload.get("prefix_witnesses")
    _require(isinstance(prefixes, list), "missing prefix witness inventory")
    references = [_reference(row, "s0") for row in prefixes]
    _require(len(references) == len(set(references)), "duplicate prefix reference")
    _require(all(seed in selection["paired_seeds"] and seat in ("p1", "p2") and turn < 9
                 for seed, seat, turn in references), "prefix is outside registered source history")
    by_identity = dict(zip(references, prefixes))
    for row, identity in zip(roots, actual):
        if row["turn_index"] == 0:
            _require(by_identity.get(identity) == row, "opening prefix witness differs from root")
    gaps = [{"seed": seed, "seat": seat,
             "uncaptured_prior_indices": [turn for turn in range(9)
                                           if (seed, seat, turn) not in by_identity]}
            for seed in selection["paired_seeds"] for seat in ("p1", "p2")]
    gaps = [row for row in gaps if row["uncaptured_prior_indices"]]
    _require(payload.get("uncaptured_prior_records") == gaps, "uncaptured prefix inventory drift")
    rules = payload.get("qualification_rules")
    _require(isinstance(rules, Mapping) and rules.get("exact_source_file_hashes_required") is True
             and rules.get("full_live_public_observation_equality_required") is True
             and rules.get("source_owned_prefix_repairs_only") is True
             and rules.get("unknown_other_player_actions_refuse") is True
             and rules.get("missing_or_unsupported_root_disposition") ==
             "retain_refusal_in_the_original_roster_no_redraw", "qualification boundary weakened")
    _require(payload.get("execution_status") == "NOT_LAUNCHED", "selection is not execution evidence")
    return payload


def load_frozen_roster(path: str | Path, *, expected_sha256: str) -> Mapping[str, Any]:
    _require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256) is not None,
             "explicit frozen roster hash required")
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "frozen roster byte hash drift")
    return validate_roster(json.loads(raw))


def verify_source_files(source_root: str | Path, payload: Mapping[str, Any]) -> None:
    """Verify every retained input before replay; never overwrite or repair it."""
    validate_roster(payload)
    base = Path(source_root).resolve(strict=True)
    refs = payload["profile_roots"] + payload["prefix_witnesses"]
    for row in refs:
        path = (base / row["source_relative_path"]).resolve(strict=True)
        _require(path.is_relative_to(base), "source path escapes cohort")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == row["source_file_sha256"],
                 f"source bytes drift: {row['decision_id']}")
        wrapper = json.loads(raw)
        _require(wrapper.get("schema_version") == "pokezero.mcts-guided-vs-raw-public-decision.v1",
                 "source wrapper schema drift")
        record = wrapper.get("record", {})
        _require(isinstance(record, Mapping) and all(record.get(key) == row[reference]
                 for key, reference in (("seed", "seed"), ("acting_player", "seat"),
                    ("turn_index", "turn_index"), ("battle_id", "battle_id"),
                    ("decision_id", "decision_id"))), "source record identity drift")
    terminal_path = (base / f"shards/{payload['source_shard']}/COMPLETE.json").resolve(strict=True)
    _require(terminal_path.is_relative_to(base), "source terminal path escapes cohort")
    raw = terminal_path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == payload["source_terminal_sha256"],
             "source terminal bytes drift")
    terminal = json.loads(raw)
    _require(terminal.get("status") == "COMPLETE" and terminal.get("games") == 50
             and terminal.get("pairs") == 25, "source shard is not complete")
