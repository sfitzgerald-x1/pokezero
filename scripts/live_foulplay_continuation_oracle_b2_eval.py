#!/usr/bin/env python3
"""Create-only B2 paired evaluation: raw checkpoint vs live continuation oracle.

Each invocation evaluates exactly one registered seat/seed unit twice: ``raw``
and ``oracle``, against an external FoulPlay process.  The deployment runner
owns ordered progress: it groups 25 atomic units into one shard and 24 shards
into one 600-unit orientation.  This program deliberately never groups or
rewrites units, so a crash cannot hide a completed earlier cell behind an
unpublished 25- or 600-unit aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.actions import ACTION_COUNT  # noqa: E402


SCHEMA_VERSION = "pokezero.root-oracle-b2-transfer-foulplay-unit.v1"
EXPERIMENT_ID = "root-oracle-b2-transfer-foulplay-20260831"
WRITE_PROTOCOL = "create-only-atomic-unit-v1"
B2_SEED_START = 108_000_000
REGISTERED_UNITS_PER_ORIENTATION = 600
SHARDS_PER_WORKER = 24
UNITS_PER_SHARD = 25
MAX_CONTINUATION_DECISION_ROUNDS = 128
# A capped candidate is retried once from the same boundary at the source
# workload's hard 1024-decision ceiling. A second cap remains a fail-closed
# source failure, so no partial candidate can influence B2.
EXPANDED_CONTINUATION_DECISION_ROUNDS = 1024
_ORIENTATION_REGISTRATION = {
    "p1": {"foulplay_player": "p2", "seed_start": B2_SEED_START, "worker_index": 0},
    "p2": {
        "foulplay_player": "p1",
        "seed_start": B2_SEED_START + REGISTERED_UNITS_PER_ORIENTATION,
        "worker_index": 1,
    },
}
SOURCE_SCHEMA_VERSION = "pokezero.controlled-foulplay-benchmark.v1"
ORACLE_RECEIPT_SCHEMA_VERSION = "pokezero.live-foulplay-continuation-oracle.v2"
SUCCESS_MARKER = "WROTE B2 LIVE FOULPLAY CONTINUATION ORACLE PAIRED UNIT"
_EXPERIMENT_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_SOURCE_FILE_PATHS = (
    "scripts/live_foulplay_continuation_oracle_b2_eval.py",
    "src/pokezero/foulplay_bridge.py",
    "src/pokezero/live_foulplay_continuation.py",
)


class B2EvaluationError(RuntimeError):
    """The live B2 evaluator cannot safely produce a paired result."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files_sha256() -> dict[str, str]:
    """Hashes for exactly the source files that define this B2 unit protocol."""

    return {relative: _sha256_file(REPO_ROOT / relative) for relative in _SOURCE_FILE_PATHS}


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write once with a hard-link commit; never replace an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing B2 evaluation: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing B2 evaluation: {path}") from error
        # A completed evaluator unit is consumed by the R25 parent/finalizer as
        # a durable child-owned handoff source.  Flushing its parent directory
        # makes the create-only link durable as well as the JSON bytes, so a
        # pod/node interruption cannot leave a successful file commit dependent
        # on directory-entry writeback.
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B2EvaluationError(f"{field} must be an object")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise B2EvaluationError(f"{field} must be a non-negative integer")
    return value


def _require_zero(value: object, *, field: str) -> None:
    if _require_nonnegative_int(value, field=field) != 0:
        raise B2EvaluationError(f"{field} must be zero for B2")


def _require_exact_keys(value: Mapping[str, Any], *, field: str, keys: set[str]) -> None:
    actual = set(value)
    if actual != keys:
        raise B2EvaluationError(
            f"{field} has an unexpected receipt shape: missing={sorted(keys - actual)!r} "
            f"extra={sorted(actual - keys)!r}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise B2EvaluationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_experiment_id(value: object, *, field: str) -> str:
    """Require a stable, explicit identity for the immutable unit series.

    The evaluator is shared by the registered B2 study and non-bankable
    reliability gates.  Those gates may use a disjoint seed reservation, but
    must never silently stamp their receipts with B2's experiment identity.
    The caller supplies the identity and the validator receives it again as an
    explicit expected value; this check is deliberately stricter than merely
    accepting any JSON string in a receipt.
    """

    if not isinstance(value, str) or _EXPERIMENT_ID_RE.fullmatch(value) is None:
        raise B2EvaluationError(f"{field} must be a lowercase hyphenated experiment identity")
    return value


_STATE_AUDIT_ALLOWED_FIELDS = frozenset(
    {
        # These are digests of live requests, not generic snapshots.  The
        # oracle-decision schema below validates their exact p1/p2 shape.
        "source_request_sha256",
        "snapshot_request_sha256",
        # These are assertions about confinement, not serialized state.
        "controller_only_full_state",
        "full_state_snapshot_scope",
    }
)


def _arm_evidence_key(key: object) -> str:
    """Normalize JSON keys, including camel-case aliases, for the state audit."""

    if not isinstance(key, str):
        return ""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return snake.replace("-", "_").lower()


def _audit_arm_evidence_no_full_state(value: object, *, field: str) -> None:
    """Fail closed on generic/full-state evidence anywhere in an arm document.

    The bridge result has useful nested blocks (including ``game_results``), so
    a shallow envelope check is insufficient.  This walk refuses snapshot and
    state aliases at every depth--``raw_snapshot``, ``snapshot_data``, and
    ``state`` included--while permitting only the request-digest fields and
    controller-confinement assertions needed by the B2 receipt.  Provenance
    identities such as ``showdown_sim_sha256`` do not contain those state
    tokens and remain admissible.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _arm_evidence_key(key)
            if not normalized:
                raise B2EvaluationError(f"{field} has a non-string evidence key")
            nested_field = f"{field}.{key}"
            if normalized in _STATE_AUDIT_ALLOWED_FIELDS:
                if normalized == "controller_only_full_state":
                    if nested is not True:
                        raise B2EvaluationError(f"{nested_field} must assert controller-only confinement")
                    continue
                if normalized == "full_state_snapshot_scope":
                    if nested != "controller-only":
                        raise B2EvaluationError(f"{nested_field} must be controller-only")
                    continue
                if not isinstance(nested, Mapping):
                    raise B2EvaluationError(f"{nested_field} must be a request-digest mapping")
                if set(nested) != {"p1", "p2"}:
                    raise B2EvaluationError(f"{nested_field} must bind exactly the p1/p2 requests")
                for player in ("p1", "p2"):
                    _require_sha256(nested[player], field=f"{nested_field}.{player}")
                continue
            if "snapshot" in normalized or "state" in normalized:
                raise B2EvaluationError(
                    f"{nested_field} contains serialized generic/full-state evidence"
                )
            _audit_arm_evidence_no_full_state(nested, field=nested_field)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _audit_arm_evidence_no_full_state(nested, field=f"{field}[{index}]")


def _registered_unit(*, seat: str, seed: int, registration_seed_start: int | None = None) -> dict[str, int | str]:
    """Bind one unit to the caller-declared seat band.

    The evaluator verifies the internal shape of a declared 600-unit orientation band.  The
    immutable deployment contract owns which distinct band is admissible for a particular B2
    run; keeping that allocation outside this reusable source validator lets a probe use its
    own disjoint diagnostic band without ever reusing production evidence.
    """

    if seat not in _ORIENTATION_REGISTRATION:
        raise B2EvaluationError("B2 unit has an unknown PokeZero orientation")
    registration = _ORIENTATION_REGISTRATION[seat]
    if registration_seed_start is None:
        seed_start = int(registration["seed_start"])
    else:
        seed_start = _require_nonnegative_int(registration_seed_start, field="registration seed start")
    offset = seed - seed_start
    if not 0 <= offset < REGISTERED_UNITS_PER_ORIENTATION:
        raise B2EvaluationError("B2 unit seed is outside its registered orientation band")
    shard_index, unit_index_in_shard = divmod(offset, UNITS_PER_SHARD)
    return {
        "pokezero_player": seat,
        "foulplay_player": str(registration["foulplay_player"]),
        "worker_index": int(registration["worker_index"]),
        "seed_start": seed_start,
        "seed": seed,
        "seed_offset": offset,
        "shard_index": shard_index,
        "unit_index_in_shard": unit_index_in_shard,
        "shard_id": f"b2-{seat}-worker-{registration['worker_index']}-shard-{shard_index:02d}",
    }


def _require_terminal_score(game: Mapping[str, Any]) -> None:
    """Require a completed, uncapped result with coherent winner/score fields."""

    if game.get("capped") is not False:
        raise B2EvaluationError("B2 game result is capped rather than terminal")
    if not isinstance(game.get("tied"), bool) or not isinstance(game.get("pokezero_won"), bool):
        raise B2EvaluationError("B2 game lacks boolean tied/pokezero_won terminal evidence")
    score = game.get("pokezero_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise B2EvaluationError("B2 game has an invalid PokeZero score")
    winner = game.get("winner")
    tied = bool(game["tied"])
    pokezero_won = bool(game["pokezero_won"])
    if winner is None:
        if not tied or pokezero_won or float(score) != 0.5:
            raise B2EvaluationError("B2 tie winner/score fields are inconsistent")
    elif winner == "PokeZeroBot":
        if tied or not pokezero_won or float(score) != 1.0:
            raise B2EvaluationError("B2 PokeZero win winner/score fields are inconsistent")
    elif winner == "FoulPlayBot":
        if tied or pokezero_won or float(score) != 0.0:
            raise B2EvaluationError("B2 FoulPlay win winner/score fields are inconsistent")
    else:
        raise B2EvaluationError("B2 game has an unknown terminal winner")


def _require_clean_integrity(summary: Mapping[str, Any], game: Mapping[str, Any]) -> None:
    integrity = _require_mapping(summary.get("execution_integrity"), field="execution_integrity")
    _require_exact_keys(
        integrity,
        field="execution_integrity",
        keys={
            "retries",
            "errors",
            "refusal_records",
            "refusal_recorder_instrument_errors",
            "refusal_records_unrowed",
            "forced_boundary_raw_decisions",
        },
    )
    for field in (
        "retries",
        "errors",
        "refusal_records",
        "refusal_recorder_instrument_errors",
        "refusal_records_unrowed",
    ):
        _require_zero(integrity.get(field), field=f"execution_integrity.{field}")
    _require_nonnegative_int(
        integrity.get("forced_boundary_raw_decisions"),
        field="execution_integrity.forced_boundary_raw_decisions",
    )
    # These per-game fields are omitted by the bridge when clean.  If they are
    # present, they must explicitly remain zero; non-empty captured refusals are
    # never admissible in a B2 arm.
    for field in (
        "opponent_moves_record_failures",
        "opponent_think_record_failures",
    ):
        if field in game:
            _require_zero(game[field], field=f"game.{field}")
    refusals = game.get("refusals")
    if refusals is not None and (not isinstance(refusals, list) or refusals):
        raise B2EvaluationError("B2 game contains captured refusal records")


def _expected_oracle_action(*, scored: Sequence[Mapping[str, Any]], raw_action: int,
                            foulplay_player: str) -> int:
    """Recompute the source controller's raw-preserving safe tie rule."""
    best_score = max(float(candidate["score"]) for candidate in scored)
    tied = [candidate for candidate in scored if float(candidate["score"]) == best_score]

    def immediate_subject_loss(candidate: Mapping[str, Any]) -> bool:
        terminal = candidate["terminal"]
        return (candidate["terminal_after_fixed_joint_step"] is True and
                terminal["winner"] == foulplay_player)

    eligible = [candidate for candidate in tied if not immediate_subject_loss(candidate)] or tied
    raw_candidate = next((candidate for candidate in eligible if candidate["action_index"] == raw_action), None)
    return int(raw_candidate["action_index"] if raw_candidate is not None else
               min(eligible, key=lambda candidate: int(candidate["action_index"]))["action_index"])


def _require_oracle_candidate_receipt(
    item: Mapping[str, Any], *, pokezero_player: str, foulplay_player: str
) -> None:
    if _require_nonnegative_int(item.get("candidate_cap"), field="candidate_cap") != ACTION_COUNT:
        raise B2EvaluationError("oracle receipt does not declare the full PokeZero action-space cap")
    if (
        _require_nonnegative_int(
            item.get("max_continuation_decision_rounds"),
            field="max_continuation_decision_rounds",
        )
        != MAX_CONTINUATION_DECISION_ROUNDS
    ):
        raise B2EvaluationError("oracle receipt does not declare the registered continuation eligibility bound")
    if (
        _require_nonnegative_int(
            item.get("expanded_continuation_decision_rounds"),
            field="expanded_continuation_decision_rounds",
        )
        != EXPANDED_CONTINUATION_DECISION_ROUNDS
    ):
        raise B2EvaluationError("oracle receipt does not declare the registered expansion eligibility bound")
    candidates = item.get("candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise B2EvaluationError("oracle receipt omits scored legal candidates")
    if _require_nonnegative_int(item.get("candidate_count"), field="candidate_count") != len(candidates):
        raise B2EvaluationError("oracle receipt candidate count does not match its candidates")
    legal_actions = item.get("legal_action_indices")
    if not isinstance(legal_actions, (list, tuple)) or not legal_actions:
        raise B2EvaluationError("oracle receipt omits the complete legal action set")
    legal = tuple(
        _require_nonnegative_int(action, field="legal_action_index")
        for action in legal_actions
    )
    if any(action >= ACTION_COUNT for action in legal) or len(set(legal)) != len(legal):
        raise B2EvaluationError("oracle receipt has malformed legal action indices")
    scored: list[Mapping[str, Any]] = []
    for candidate in candidates:
        record = _require_mapping(candidate, field="oracle candidate")
        _require_exact_keys(
            record,
            field="oracle candidate",
            keys={
                "action_index",
                "score",
                "continuation_decision_round_count",
                "max_continuation_decision_rounds",
                "terminal",
                "terminal_after_fixed_joint_step",
            },
        )
        action = _require_nonnegative_int(record.get("action_index"), field="candidate.action_index")
        if action >= ACTION_COUNT:
            raise B2EvaluationError("oracle receipt candidate action is outside the PokeZero action space")
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise B2EvaluationError("oracle receipt candidate has an invalid score")
        if float(score) not in {0.0, 0.5, 1.0}:
            raise B2EvaluationError("oracle receipt candidate score is not a terminal PokeZero score")
        terminal = _require_mapping(record.get("terminal"), field="oracle candidate terminal")
        _require_exact_keys(terminal, field="oracle candidate terminal", keys={"winner", "turn_count", "capped"})
        if terminal.get("capped") is not False:
            raise B2EvaluationError("oracle receipt candidate is capped")
        if _require_nonnegative_int(terminal.get("turn_count"), field="candidate.terminal.turn_count") < 1:
            raise B2EvaluationError("oracle receipt candidate terminal has no turn count")
        continuation_decision_round_count = _require_nonnegative_int(
            record.get("continuation_decision_round_count"),
            field="candidate.continuation_decision_round_count",
        )
        candidate_bound = _require_nonnegative_int(
            record.get("max_continuation_decision_rounds"),
            field="candidate.max_continuation_decision_rounds",
        )
        if candidate_bound not in {MAX_CONTINUATION_DECISION_ROUNDS, EXPANDED_CONTINUATION_DECISION_ROUNDS}:
            raise B2EvaluationError("oracle receipt candidate has an unregistered continuation eligibility bound")
        if continuation_decision_round_count > candidate_bound:
            raise B2EvaluationError(
                "oracle receipt candidate exceeds the registered continuation "
                "eligibility bound"
            )
        terminal_after_fixed_joint_step = record.get("terminal_after_fixed_joint_step")
        if not isinstance(terminal_after_fixed_joint_step, bool):
            raise B2EvaluationError(
                "oracle candidate must declare terminal_after_fixed_joint_step "
                "as a boolean"
            )
        if terminal_after_fixed_joint_step and continuation_decision_round_count != 0:
            raise B2EvaluationError(
                "terminal fixed-step candidate must have zero continuation "
                "decision rounds"
            )
        if (
            not terminal_after_fixed_joint_step
            and continuation_decision_round_count < 1
        ):
            raise B2EvaluationError("B2 candidate did not continue from its mid-game joint step")
        winner = terminal.get("winner")
        expected_score = (
            1.0 if winner == pokezero_player else 0.5 if winner is None else 0.0 if winner == foulplay_player else None
        )
        if expected_score is None or float(score) != expected_score:
            raise B2EvaluationError("oracle candidate terminal winner and score are inconsistent")
        scored.append(record)
    actions = [int(candidate["action_index"]) for candidate in scored]
    if len(set(actions)) != len(actions):
        raise B2EvaluationError("oracle receipt repeats a legal candidate action")
    if tuple(actions) != legal:
        raise B2EvaluationError("oracle receipt candidates do not exhaust the recorded legal action set")
    selected = _require_nonnegative_int(item.get("selected_action_index"), field="selected_action_index")
    if selected not in actions:
        raise B2EvaluationError("oracle receipt selected an action outside its candidates")
    raw_action = _require_nonnegative_int(item.get("raw_action_index"), field="raw_action_index")
    if raw_action not in legal:
        raise B2EvaluationError("oracle receipt raw action is outside the recorded legal action set")
    expected_selected = _expected_oracle_action(
        scored=scored, raw_action=raw_action, foulplay_player=foulplay_player,
    )
    if selected != expected_selected:
        raise B2EvaluationError("oracle receipt does not use the safe raw-preserving tie rule")
    if not isinstance(item.get("selected_changed_raw_action"), bool) or (
        bool(item["selected_changed_raw_action"]) != (selected != raw_action)
    ):
        raise B2EvaluationError("oracle receipt selected_changed_raw_action disagrees with the selected action")
    if _require_nonnegative_int(item.get("action_index"), field="action_index") != selected:
        raise B2EvaluationError("oracle receipt action_index does not match its selected action")
    fixed_joint_step = _require_mapping(item.get("first_restored_joint_step"), field="first_restored_joint_step")
    if fixed_joint_step != {
        pokezero_player: selected,
        foulplay_player: item.get("decoded_actual_foulplay_action"),
    }:
        raise B2EvaluationError("oracle receipt fixed joint step does not bind selected and FoulPlay actions")


def _require_controller_receipt(
    summary: Mapping[str, Any], *, oracle: bool, seat: str, expected_seed: int | None = None
) -> Mapping[str, Any]:
    _audit_arm_evidence_no_full_state(
        summary, field="oracle continuation arm" if oracle else "raw arm"
    )
    if summary.get("schema_version") != SOURCE_SCHEMA_VERSION or summary.get("status") != "complete":
        raise B2EvaluationError("B2 arm did not emit the complete controlled-FoulPlay source schema")
    if summary.get("games") != 1 or summary.get("complete") is not True or summary.get("completed_games") != 1:
        raise B2EvaluationError("bridge arm did not complete exactly one configured game")
    if summary.get("capped_games") != 0:
        raise B2EvaluationError("B2 arm reports capped games")
    if summary.get("policy_mode") != "raw" or summary.get("opponent_policy_id") != "foul-play":
        raise B2EvaluationError("B2 arm was not a raw-policy game against external FoulPlay")
    if summary.get("capture_driver") != "checkpoint":
        raise B2EvaluationError("B2 arm was not driven directly by the registered checkpoint")
    if summary.get("pokezero_player") != seat:
        raise B2EvaluationError("bridge arm returned the wrong PokeZero orientation")
    expected_foulplay = "p2" if seat == "p1" else "p1"
    if summary.get("foulplay_player") != expected_foulplay:
        raise B2EvaluationError("bridge arm returned the wrong external FoulPlay orientation")
    controller = _require_mapping(summary.get("live_continuation_oracle"), field="live_continuation_oracle")
    _require_exact_keys(
        controller,
        field="live_continuation_oracle",
        keys={
            "enabled",
            "schema_version",
            "candidate_cap",
            "max_continuation_decision_rounds",
            "expanded_continuation_decision_rounds",
            "expanded_continuation_decision_rounds",
            "controller_only_full_state",
            "oracle_decisions",
            "games_with_oracle_decision",
            "forced_boundary_raw_decisions",
        },
    )
    if bool(controller.get("enabled")) is not oracle:
        raise B2EvaluationError("bridge arm's continuation-oracle witness disagrees with its requested arm")
    if controller.get("schema_version") != ORACLE_RECEIPT_SCHEMA_VERSION:
        raise B2EvaluationError("B2 arm has an unexpected continuation-oracle schema")
    if controller.get("controller_only_full_state") is not True:
        raise B2EvaluationError("B2 arm does not assert controller-only full-state confinement")
    if _require_nonnegative_int(controller.get("candidate_cap"), field="controller.candidate_cap") != ACTION_COUNT:
        raise B2EvaluationError("B2 arm is not using the registered full PokeZero action-space cap")
    if (
        _require_nonnegative_int(
            controller.get("max_continuation_decision_rounds"),
            field="controller.max_continuation_decision_rounds",
        )
        != MAX_CONTINUATION_DECISION_ROUNDS
    ):
        raise B2EvaluationError("B2 arm is not using the registered continuation eligibility bound")
    if controller.get("expanded_continuation_decision_rounds") != EXPANDED_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError("B2 arm is not using the registered continuation expansion bound")
    _require_nonnegative_int(
        controller.get("forced_boundary_raw_decisions"), field="controller.forced_boundary_raw_decisions"
    )
    games = summary.get("game_results")
    if not isinstance(games, list) or len(games) != 1:
        raise B2EvaluationError("bridge arm has no single game result")
    game = _require_mapping(games[0], field="game_results[0]")
    _require_terminal_score(game)
    _require_clean_integrity(summary, game)
    summary_seed = _require_nonnegative_int(summary.get("seed_start"), field="summary.seed_start")
    game_seed = _require_nonnegative_int(game.get("seed"), field="game.seed")
    foulplay_seed = _require_nonnegative_int(
        summary.get("foulplay_random_seed"), field="summary.foulplay_random_seed"
    )
    if summary_seed != game_seed or foulplay_seed != game_seed:
        raise B2EvaluationError("B2 arm does not bind BattleStream and FoulPlay seeds to its game seed")
    if expected_seed is not None and game_seed != expected_seed:
        raise B2EvaluationError("B2 arm returned the wrong registered seed")
    if oracle:
        if _require_nonnegative_int(
            controller.get("games_with_oracle_decision"), field="controller.games_with_oracle_decision"
        ) != 1:
            raise B2EvaluationError("oracle game completed without a successful continuation-oracle decision")
        receipt = _require_mapping(game.get("live_continuation_oracle"), field="game continuation receipt")
        if _require_nonnegative_int(receipt.get("oracle_decision_count"), field="oracle_decision_count") < 1:
            raise B2EvaluationError("oracle game receipt has no successful controller decision")
        decisions = receipt.get("oracle_decisions")
        if not isinstance(decisions, list) or not decisions:
            raise B2EvaluationError("oracle game receipt omits its bounded decisions")
        _require_exact_keys(
            receipt,
            field="game continuation receipt",
            keys={
                "schema_version",
                "controller_only_full_state",
                "oracle_decisions",
                "oracle_decision_count",
                "forced_boundary_raw_decisions",
            },
        )
        if receipt.get("schema_version") != ORACLE_RECEIPT_SCHEMA_VERSION:
            raise B2EvaluationError("oracle game receipt has an unexpected schema")
        if receipt.get("controller_only_full_state") is not True:
            raise B2EvaluationError("oracle game receipt does not confine full state to the controller")
        if _require_nonnegative_int(
            receipt.get("forced_boundary_raw_decisions"), field="receipt.forced_boundary_raw_decisions"
        ) != _require_nonnegative_int(
            controller.get("forced_boundary_raw_decisions"), field="controller.forced_boundary_raw_decisions"
        ):
            raise B2EvaluationError("oracle game receipt forced-boundary count disagrees with its arm header")
        if receipt["oracle_decision_count"] != len(decisions):
            raise B2EvaluationError("oracle game receipt count does not match its decisions")
        if _require_nonnegative_int(
            controller.get("oracle_decisions"), field="controller.oracle_decisions"
        ) != len(decisions):
            raise B2EvaluationError("oracle arm count does not match its game receipt")
        for decision in decisions:
            item = _require_mapping(decision, field="oracle decision")
            _require_exact_keys(
                item,
                field="oracle decision",
                keys={
                    "schema_version",
                    "controller",
                    "controller_status",
                    "full_state_snapshot_scope",
                    "source_decision_round",
                    "pokezero_player",
                    "foulplay_player",
                    "source_request_sha256",
                    "snapshot_request_sha256",
                    "actual_foulplay_choice",
                    "decoded_actual_foulplay_action",
                    "raw_action_index",
                    "selected_action_index",
                    "selected_changed_raw_action",
                    "first_restored_joint_step",
                    "candidate_count",
                    "candidate_cap",
                    "max_continuation_decision_rounds",
                    "expanded_continuation_decision_rounds",
                    "legal_action_indices",
                    "candidates",
                    "action_index",
                },
            )
            if (
                item.get("schema_version") != ORACLE_RECEIPT_SCHEMA_VERSION
                or item.get("controller") != "live-foulplay-continuation-oracle"
                or item.get("controller_status") != "oracle-selected"
            ):
                raise B2EvaluationError("oracle receipt contains a non-oracle controller result")
            if item.get("pokezero_player") != seat or item.get("foulplay_player") != expected_foulplay:
                raise B2EvaluationError("oracle receipt seats do not match the registered orientation")
            if item.get("expanded_continuation_decision_rounds") != EXPANDED_CONTINUATION_DECISION_ROUNDS:
                raise B2EvaluationError("oracle receipt has the wrong continuation expansion bound")
            if _require_nonnegative_int(item.get("source_decision_round"), field="source_decision_round") < 1:
                raise B2EvaluationError("B2 oracle receipt is not a mid-game continuation")
            source_hashes = _require_mapping(item.get("source_request_sha256"), field="source request hashes")
            snapshot_hashes = _require_mapping(item.get("snapshot_request_sha256"), field="snapshot request hashes")
            if set(source_hashes) != {"p1", "p2"}:
                raise B2EvaluationError("oracle receipt does not bind both source requests")
            if set(snapshot_hashes) != {"p1", "p2"}:
                raise B2EvaluationError("oracle receipt does not bind both snapshot requests")
            for player in ("p1", "p2"):
                _require_sha256(source_hashes[player], field=f"source_request_sha256.{player}")
                _require_sha256(snapshot_hashes[player], field=f"snapshot_request_sha256.{player}")
            if not isinstance(item.get("actual_foulplay_choice"), str) or not item["actual_foulplay_choice"].strip():
                raise B2EvaluationError("oracle receipt lacks the actual external FoulPlay choice")
            foulplay_action = item.get("decoded_actual_foulplay_action")
            if isinstance(foulplay_action, bool) or not isinstance(foulplay_action, int) or not 0 <= foulplay_action < ACTION_COUNT:
                raise B2EvaluationError("oracle receipt lacks the decoded external FoulPlay action")
            # The only admitted full-state evidence is an explicit assertion of
            # controller confinement.  A serialized generic snapshot is forbidden.
            if item.get("full_state_snapshot_scope") != "controller-only" or "snapshot" in item:
                raise B2EvaluationError("oracle receipt leaks or mis-scopes the generic full-state snapshot")
            _require_oracle_candidate_receipt(
                item, pokezero_player=seat, foulplay_player=expected_foulplay
            )
    else:
        _require_zero(controller.get("oracle_decisions"), field="raw controller.oracle_decisions")
        _require_zero(
            controller.get("games_with_oracle_decision"), field="raw controller.games_with_oracle_decision"
        )
    return game


_ARM_PROVENANCE_KEYS = {
    "bridge_schema_version",
    "bridge_source_sha256",
    "unit_evaluator_source_sha256",
    "live_continuation_source_sha256",
    "format_id",
    "capture_driver",
    "belief_set_source",
    "max_decision_rounds",
    "max_continuation_decision_rounds",
    "expanded_continuation_decision_rounds",
    "foulplay_search_time_ms",
    "checkpoint_path",
    "checkpoint_sha256",
    "showdown_root",
    "showdown_sim_sha256",
    "foulplay_root",
    "foulplay_entrypoint_sha256",
    "foulplay_python",
    "node_binary",
}


def _require_arm_provenance(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = _require_mapping(summary.get("b2_provenance"), field="b2_provenance")
    _require_exact_keys(provenance, field="b2_provenance", keys=_ARM_PROVENANCE_KEYS)
    for field in (
        "bridge_source_sha256",
        "unit_evaluator_source_sha256",
        "live_continuation_source_sha256",
        "checkpoint_sha256",
        "showdown_sim_sha256",
        "foulplay_entrypoint_sha256",
    ):
        _require_sha256(provenance.get(field), field=f"b2_provenance.{field}")
    if provenance.get("bridge_schema_version") != SOURCE_SCHEMA_VERSION:
        raise B2EvaluationError("B2 provenance has an unexpected controlled-FoulPlay source schema")
    if provenance.get("bridge_schema_version") != summary.get("schema_version"):
        raise B2EvaluationError("B2 provenance source schema does not match its arm")
    if provenance.get("format_id") != summary.get("format_id"):
        raise B2EvaluationError("B2 provenance format does not match its arm")
    if provenance.get("capture_driver") != summary.get("capture_driver"):
        raise B2EvaluationError("B2 provenance capture driver does not match its arm")
    if provenance.get("belief_set_source") != summary.get("belief_set_source"):
        raise B2EvaluationError("B2 provenance belief setting does not match its arm")
    if provenance.get("max_decision_rounds") != summary.get("max_decision_rounds"):
        raise B2EvaluationError("B2 provenance decision cap does not match its arm")
    controller = _require_mapping(summary.get("live_continuation_oracle"), field="live_continuation_oracle")
    if provenance.get("max_continuation_decision_rounds") != controller.get(
        "max_continuation_decision_rounds"
    ):
        raise B2EvaluationError("B2 provenance continuation eligibility bound does not match its arm")
    if provenance.get("expanded_continuation_decision_rounds") != controller.get(
        "expanded_continuation_decision_rounds"
    ):
        raise B2EvaluationError("B2 provenance continuation expansion bound does not match its arm")
    if provenance.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise B2EvaluationError("B2 provenance checkpoint does not match its arm")
    foulplay_think = _require_mapping(summary.get("foulplay_think"), field="foulplay_think")
    if provenance.get("foulplay_search_time_ms") != foulplay_think.get("budget_ms_configured"):
        raise B2EvaluationError("B2 provenance FoulPlay search budget does not match its arm")
    for field in (
        "format_id",
        "capture_driver",
        "checkpoint_path",
        "showdown_root",
        "foulplay_root",
        "foulplay_python",
        "node_binary",
    ):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise B2EvaluationError(f"b2_provenance.{field} must be a non-empty identity string")
    if not isinstance(provenance.get("belief_set_source"), bool):
        raise B2EvaluationError("b2_provenance.belief_set_source must be boolean")
    if _require_nonnegative_int(
        provenance.get("max_decision_rounds"), field="b2_provenance.max_decision_rounds"
    ) <= 2:
        raise B2EvaluationError("B2 provenance has no room for a mid-game continuation")
    if (
        _require_nonnegative_int(
            provenance.get("max_continuation_decision_rounds"),
            field="b2_provenance.max_continuation_decision_rounds",
        )
        != MAX_CONTINUATION_DECISION_ROUNDS
    ):
        raise B2EvaluationError("B2 provenance has the wrong continuation eligibility bound")
    if provenance.get("expanded_continuation_decision_rounds") != EXPANDED_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError("B2 provenance has the wrong continuation expansion bound")
    if _require_nonnegative_int(
        provenance.get("foulplay_search_time_ms"), field="b2_provenance.foulplay_search_time_ms"
    ) <= 0:
        raise B2EvaluationError("B2 provenance has an invalid FoulPlay search budget")
    return provenance


def validate_experiment_document(
    payload: Mapping[str, Any], *, expected_experiment_id: str,
) -> None:
    """Validate one completed paired unit for one explicitly named experiment.

    ``expected_experiment_id`` is deployment-owned evidence, not an inference
    from the receipt.  Keeping it explicit lets a diagnostic use an isolated
    seed band while preserving ``validate_b2_document`` as the strict default
    used by the registered study.
    """

    expected_experiment_id = _require_experiment_id(
        expected_experiment_id, field="expected experiment id"
    )

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise B2EvaluationError("unexpected B2 document schema")
    if payload.get("experiment_id") != expected_experiment_id:
        raise B2EvaluationError("unexpected paired-unit experiment id")
    if payload.get("complete") is not True or payload.get("write_protocol") != WRITE_PROTOCOL:
        raise B2EvaluationError("B2 unit is not a completed create-only atomic unit")
    if payload.get("status") != "PASS":
        raise B2EvaluationError("B2 document status is not PASS")
    seat = payload.get("pokezero_player")
    seed = _require_nonnegative_int(payload.get("seed"), field="seed")
    if seat not in _ORIENTATION_REGISTRATION:
        raise B2EvaluationError("B2 unit has an unknown PokeZero orientation")
    source_files = _require_mapping(payload.get("source_files_sha256"), field="source_files_sha256")
    _require_exact_keys(
        source_files,
        field="source_files_sha256",
        keys=set(_SOURCE_FILE_PATHS),
    )
    for relative_path in _SOURCE_FILE_PATHS:
        digest = source_files.get(relative_path)
        _require_sha256(digest, field=f"source_files_sha256.{relative_path}")
    registration = _require_mapping(payload.get("registration"), field="registration")
    if registration.get("pokezero_player") != seat or registration.get("seed") != seed:
        raise B2EvaluationError("B2 unit registration does not match top-level seat and seed")
    expected = _registered_unit(
        seat=seat,
        seed=seed,
        registration_seed_start=_require_nonnegative_int(
            registration.get("seed_start"), field="registration.seed_start"
        ),
    )
    _require_exact_keys(
        registration,
        field="registration",
        keys={
            "registered_units_per_orientation",
            "shards_per_worker",
            "units_per_shard",
            "worker_index",
            "pokezero_player",
            "foulplay_player",
            "seed_start",
            "seed",
            "seed_offset",
            "shard_index",
            "unit_index_in_shard",
            "shard_id",
            "candidate_cap",
            "max_continuation_decision_rounds",
            "expanded_continuation_decision_rounds",
            "arms",
            "external_opponent",
            "checkpoint",
            "checkpoint_sha256",
        },
    )
    for field in (
        "foulplay_player",
        "worker_index",
        "seed_start",
        "seed_offset",
        "shard_index",
        "unit_index_in_shard",
        "shard_id",
    ):
        if registration.get(field) != expected[field]:
            raise B2EvaluationError(f"B2 unit registration field {field} does not match its registered band")
    if registration.get("registered_units_per_orientation") != REGISTERED_UNITS_PER_ORIENTATION:
        raise B2EvaluationError("B2 unit has the wrong registered orientation size")
    if registration.get("shards_per_worker") != SHARDS_PER_WORKER:
        raise B2EvaluationError("B2 unit has the wrong registered shard count")
    if registration.get("units_per_shard") != UNITS_PER_SHARD:
        raise B2EvaluationError("B2 unit has the wrong registered shard size")
    if registration.get("candidate_cap") != ACTION_COUNT:
        raise B2EvaluationError("B2 unit does not record the registered full action-space cap")
    if registration.get("max_continuation_decision_rounds") != MAX_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError("B2 unit does not record the registered continuation eligibility bound")
    if registration.get("expanded_continuation_decision_rounds") != EXPANDED_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError("B2 unit does not record the registered continuation expansion bound")
    if registration.get("arms") != ["raw", "live-continuation-oracle"]:
        raise B2EvaluationError("B2 unit has the wrong paired treatment arms")
    if registration.get("external_opponent") != "FoulPlay":
        raise B2EvaluationError("B2 unit does not name external FoulPlay")
    if not isinstance(registration.get("checkpoint"), str) or not registration["checkpoint"]:
        raise B2EvaluationError("B2 unit lacks a checkpoint identity")
    _require_sha256(registration.get("checkpoint_sha256"), field="registration.checkpoint_sha256")
    raw = _require_mapping(payload.get("raw"), field="raw arm")
    oracle = _require_mapping(payload.get("oracle_continuation"), field="oracle continuation arm")
    if raw.get("treatment_policy_mode") != "raw-transformer-policy":
        raise B2EvaluationError("raw arm has the wrong treatment identity")
    if oracle.get("treatment_policy_mode") != "oracle-continuation":
        raise B2EvaluationError("oracle arm has the wrong treatment identity")
    raw_game = _require_controller_receipt(raw, oracle=False, seat=seat, expected_seed=seed)
    oracle_game = _require_controller_receipt(oracle, oracle=True, seat=seat, expected_seed=seed)
    for field in ("seed_start", "foulplay_random_seed"):
        if raw.get(field) != oracle.get(field):
            raise B2EvaluationError(f"paired arms disagree on common-random-number field {field}")
    expected_schedule = {
        "count": 1,
        "first_seed": seed,
        "last_seed": seed,
        "mode": "constant",
        "seeds": [seed],
    }
    if raw.get("foulplay_random_seed_schedule") != expected_schedule:
        raise B2EvaluationError("raw arm does not record the exact one-seed FoulPlay schedule")
    if oracle.get("foulplay_random_seed_schedule") != expected_schedule:
        raise B2EvaluationError("oracle arm does not record the exact one-seed FoulPlay schedule")
    raw_provenance = _require_arm_provenance(raw)
    oracle_provenance = _require_arm_provenance(oracle)
    if raw_provenance != oracle_provenance:
        raise B2EvaluationError("paired arms disagree on source, checkpoint, Showdown, or FoulPlay provenance")
    if raw_provenance["checkpoint_sha256"] != registration["checkpoint_sha256"]:
        raise B2EvaluationError("B2 unit checkpoint registration does not match paired arm provenance")
    if (
        raw_provenance["checkpoint_path"] != registration["checkpoint"]
        or oracle_provenance["checkpoint_path"] != registration["checkpoint"]
    ):
        raise B2EvaluationError("B2 unit checkpoint registration path does not match paired arm provenance")
    source_provenance_fields = {
        "scripts/live_foulplay_continuation_oracle_b2_eval.py": "unit_evaluator_source_sha256",
        "src/pokezero/foulplay_bridge.py": "bridge_source_sha256",
        "src/pokezero/live_foulplay_continuation.py": "live_continuation_source_sha256",
    }
    for source_path, provenance_field in source_provenance_fields.items():
        if raw_provenance[provenance_field] != source_files[source_path]:
            raise B2EvaluationError(
                f"B2 {provenance_field} does not match the top-level source file identity"
            )
    if float(payload.get("oracle_minus_raw_score")) != (
        float(oracle_game.get("pokezero_score")) - float(raw_game.get("pokezero_score"))
    ):
        raise B2EvaluationError("paired delta does not equal oracle score minus raw score")


def validate_b2_document(payload: Mapping[str, Any]) -> None:
    """Validate a completed receipt for the registered B2 study only."""

    validate_experiment_document(payload, expected_experiment_id=EXPERIMENT_ID)


def _arm_provenance(args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, object]:
    """Bind each locally captured bridge arm to its runnable source dependencies."""

    showdown_simulator = args.showdown_root / "dist" / "sim" / "dex.js"
    foulplay_entrypoint = args.foulplay_root / "run.py"
    foulplay_think = _require_mapping(summary.get("foulplay_think"), field="foulplay_think")
    return {
        "bridge_schema_version": summary.get("schema_version"),
        "bridge_source_sha256": _sha256_file(REPO_ROOT / "src" / "pokezero" / "foulplay_bridge.py"),
        "unit_evaluator_source_sha256": _sha256_file(
            REPO_ROOT / "scripts" / "live_foulplay_continuation_oracle_b2_eval.py"
        ),
        "live_continuation_source_sha256": _sha256_file(
            REPO_ROOT / "src" / "pokezero" / "live_foulplay_continuation.py"
        ),
        "format_id": summary.get("format_id"),
        "capture_driver": summary.get("capture_driver"),
        "belief_set_source": summary.get("belief_set_source"),
        "max_decision_rounds": summary.get("max_decision_rounds"),
        "max_continuation_decision_rounds": _require_mapping(
            summary.get("live_continuation_oracle"), field="live_continuation_oracle"
        ).get("max_continuation_decision_rounds"),
        "expanded_continuation_decision_rounds": _require_mapping(
            summary.get("live_continuation_oracle"), field="live_continuation_oracle"
        ).get("expanded_continuation_decision_rounds"),
        "foulplay_search_time_ms": foulplay_think.get("budget_ms_configured"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "showdown_root": str(args.showdown_root.resolve()),
        "showdown_sim_sha256": _sha256_file(showdown_simulator),
        "foulplay_root": str(args.foulplay_root.resolve()),
        "foulplay_entrypoint_sha256": _sha256_file(foulplay_entrypoint),
        "foulplay_python": str(args.foulplay_python.resolve()),
        "node_binary": args.node_binary,
    }


def _run_arm(args: argparse.Namespace, *, seed: int, seat: str, oracle: bool) -> Mapping[str, Any]:
    command = [
        args.python,
        "-m",
        "pokezero.foulplay_bridge",
        "--checkpoint",
        str(args.checkpoint),
        "--showdown-root",
        str(args.showdown_root),
        "--foulplay-root",
        str(args.foulplay_root),
        "--foulplay-python",
        str(args.foulplay_python),
        "--games",
        "1",
        "--seed-start",
        str(seed),
        "--foulplay-random-seed",
        str(seed),
        "--search-time-ms",
        str(args.foulplay_search_time_ms),
        "--max-decision-rounds",
        str(args.max_decision_rounds),
        "--policy-mode",
        "raw",
        "--pokezero-player",
        seat,
        "--device",
        args.device,
        "--node-binary",
        args.node_binary,
        "--json",
    ]
    if oracle:
        command.extend(
            [
                "--live-continuation-oracle",
                "--live-continuation-oracle-candidate-cap",
                str(args.candidate_cap),
                "--live-continuation-oracle-max-continuation-decision-rounds",
                str(args.max_continuation_decision_rounds),
                "--live-continuation-oracle-expanded-continuation-decision-rounds",
                str(args.expanded_continuation_decision_rounds),
            ]
        )
        if args.oracle_progress_dir is not None:
            command.extend(
                [
                    "--live-continuation-oracle-progress-dir",
                    str((args.oracle_progress_dir / f"{seat}-{seed}").resolve()),
                ]
            )
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    if completed.returncode != 0:
        raise B2EvaluationError(
            f"{'oracle' if oracle else 'raw'} arm failed for seed={seed} seat={seat}: "
            f"exit={completed.returncode}; stderr={completed.stderr[-1000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise B2EvaluationError(
            f"bridge did not emit JSON for seed={seed} seat={seat}: {completed.stdout[-1000:]}"
        ) from error
    summary = _require_mapping(payload, field="bridge JSON")
    return {**dict(summary), "b2_provenance": _arm_provenance(args, summary)}


def _paired_unit(args: argparse.Namespace) -> dict[str, object]:
    raw = _run_arm(args, seed=args.seed, seat=args.pokezero_player, oracle=False)
    oracle = _run_arm(args, seed=args.seed, seat=args.pokezero_player, oracle=True)
    raw_game = _require_controller_receipt(
        raw, oracle=False, seat=args.pokezero_player, expected_seed=args.seed
    )
    oracle_game = _require_controller_receipt(
        oracle, oracle=True, seat=args.pokezero_player, expected_seed=args.seed
    )
    return {
        "seed": args.seed,
        "seat": args.pokezero_player,
        "raw": {"treatment_policy_mode": "raw-transformer-policy", **dict(raw)},
        "oracle": {"treatment_policy_mode": "oracle-continuation", **dict(oracle)},
        "oracle_minus_raw_score": float(oracle_game["pokezero_score"]) - float(raw_game["pokezero_score"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--foulplay-root", type=Path, required=True)
    parser.add_argument("--foulplay-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pokezero-player", choices=("p1", "p2"), required=True)
    parser.add_argument("--foulplay-player", choices=("p1", "p2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--experiment-id",
        default=EXPERIMENT_ID,
        help=(
            "immutable identity for this paired unit series; defaults to the "
            "registered B2 study and must be overridden by a diagnostic gate"
        ),
    )
    parser.add_argument(
        "--registration-seed-start",
        type=int,
        required=True,
        help="first seed in this declared 600-unit PokeZero orientation band",
    )
    parser.add_argument(
        "--expanded-continuation-decision-rounds",
        type=int,
        default=EXPANDED_CONTINUATION_DECISION_ROUNDS,
    )
    parser.add_argument("--foulplay-search-time-ms", type=int, default=1000)
    parser.add_argument("--max-decision-rounds", type=int, default=64)
    parser.add_argument("--candidate-cap", type=int, default=ACTION_COUNT)
    parser.add_argument(
        "--max-continuation-decision-rounds",
        type=int,
        default=MAX_CONTINUATION_DECISION_ROUNDS,
    )
    parser.add_argument(
        "--oracle-progress-dir",
        type=Path,
        default=None,
        help="Optional absolute root for immutable action-only oracle progress records.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    experiment_id = _require_experiment_id(args.experiment_id, field="--experiment-id")
    expected = _ORIENTATION_REGISTRATION[args.pokezero_player]
    if args.foulplay_player != expected["foulplay_player"]:
        raise B2EvaluationError("--foulplay-player must be the external FoulPlay complement of --pokezero-player")
    registered_unit = _registered_unit(
        seat=args.pokezero_player,
        seed=args.seed,
        registration_seed_start=args.registration_seed_start,
    )
    if args.max_decision_rounds <= 2:
        raise B2EvaluationError("B2 requires room for opening and continued live-oracle decisions")
    if args.candidate_cap != ACTION_COUNT:
        raise B2EvaluationError(
            f"B2 candidate cap must be {ACTION_COUNT}: the complete PokeZero action space"
        )
    if args.max_continuation_decision_rounds != MAX_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError(
            "B2 continuation eligibility bound must be "
            f"{MAX_CONTINUATION_DECISION_ROUNDS} decisions per candidate"
        )
    if args.expanded_continuation_decision_rounds != EXPANDED_CONTINUATION_DECISION_ROUNDS:
        raise B2EvaluationError(
            "B2 continuation expansion bound must be "
            f"{EXPANDED_CONTINUATION_DECISION_ROUNDS} decisions per capped candidate"
        )
    if args.oracle_progress_dir is not None and not args.oracle_progress_dir.is_absolute():
        raise B2EvaluationError("--oracle-progress-dir must be absolute when set")
    unit = _paired_unit(args)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "complete": True,
        "write_protocol": WRITE_PROTOCOL,
        "status": "PASS",
        "success_marker": SUCCESS_MARKER,
        "pokezero_player": args.pokezero_player,
        "seed": args.seed,
        "source_files_sha256": _source_files_sha256(),
        "registration": {
            "registered_units_per_orientation": REGISTERED_UNITS_PER_ORIENTATION,
            "shards_per_worker": SHARDS_PER_WORKER,
            "units_per_shard": UNITS_PER_SHARD,
            "worker_index": registered_unit["worker_index"],
            "pokezero_player": args.pokezero_player,
            "foulplay_player": args.foulplay_player,
            "seed_start": registered_unit["seed_start"],
            "seed": args.seed,
            "seed_offset": registered_unit["seed_offset"],
            "shard_index": registered_unit["shard_index"],
            "unit_index_in_shard": registered_unit["unit_index_in_shard"],
            "shard_id": registered_unit["shard_id"],
            "candidate_cap": args.candidate_cap,
            "max_continuation_decision_rounds": args.max_continuation_decision_rounds,
            "expanded_continuation_decision_rounds": args.expanded_continuation_decision_rounds,
            "arms": ["raw", "live-continuation-oracle"],
            "external_opponent": "FoulPlay",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
        },
        "raw": unit["raw"],
        "oracle_continuation": unit["oracle"],
        "oracle_minus_raw_score": unit["oracle_minus_raw_score"],
        "summary": {
            "seat_game_count": 1,
            "oracle_successful_games": 1,
            "oracle_minus_raw_score": unit["oracle_minus_raw_score"],
        },
    }
    validate_experiment_document(payload, expected_experiment_id=experiment_id)
    _write_new_json(args.out, payload)
    print(SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
