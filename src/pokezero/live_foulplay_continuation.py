"""Fail-closed reconstruction and continuation of a live FoulPlay boundary.

This module is deliberately narrow.  It accepts the generic BattleStream
snapshot produced at a *real* controlled-FoulPlay decision boundary, binds that
snapshot to the raw requests delivered by the source bridge, and continues one
fixed joint action in a new local Showdown shell.  It is a harness-capability
primitive, not a policy-input or evaluation result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .actions import ACTION_COUNT
from .belief import PublicBattleBeliefEngine
from .env import PlayerId, PokeZeroEnv
from .local_showdown import LocalShowdownSnapshot
from .policy import Policy
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .showdown import parse_showdown_replay


_PLAYER_IDS: tuple[PlayerId, PlayerId] = ("p1", "p2")


class LiveFoulPlayContinuationError(RuntimeError):
    """The live boundary cannot safely support a continuation proof."""


LIVE_FOULPLAY_CONTINUATION_ORACLE_SCHEMA_VERSION = (
    "pokezero.live-foulplay-continuation-oracle.v1"
)


@dataclass(frozen=True)
class LiveFoulPlayContinuationOracleDecision:
    """A bounded, receipt-safe decision from the live continuation oracle.

    The controller deliberately returns only action indices and terminal summaries.
    It never returns the generic snapshot or simulator state that it used while
    scoring candidates.  That keeps the full state confined to the controller
    path rather than allowing it to become a policy observation or a durable
    evaluation artifact.
    """

    action_index: int
    metadata: Mapping[str, Any]


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _clone_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
    if not isinstance(cloned, dict):  # defensive: JSON object input must remain one.
        raise LiveFoulPlayContinuationError("expected a JSON object while cloning live boundary state")
    return cloned


def _request_payload(raw_line: str, *, player: PlayerId) -> dict[str, Any]:
    prefix = "|request|"
    if not raw_line.startswith(prefix):
        raise LiveFoulPlayContinuationError(f"live {player} request is missing the |request| prefix")
    try:
        payload = json.loads(raw_line[len(prefix) :])
    except json.JSONDecodeError as error:
        raise LiveFoulPlayContinuationError(f"live {player} request is not valid JSON") from error
    if not isinstance(payload, dict):
        raise LiveFoulPlayContinuationError(f"live {player} request JSON must be an object")
    return payload


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiveFoulPlayBoundary:
    """Full-state scorer input bound to the source bridge's exact request boundary."""

    snapshot: LocalShowdownSnapshot
    source_request_sha256: Mapping[PlayerId, str]
    snapshot_request_sha256: Mapping[PlayerId, str]


def reconstruct_live_foulplay_boundary(
    *,
    source_battle_id: str,
    format_id: str,
    bridge_snapshot: Mapping[str, Any],
    public_protocol_lines: Sequence[str],
    current_request_lines: Mapping[PlayerId, str],
    request_history_lines: Mapping[PlayerId, Sequence[str]],
    belief_set_source: object | None,
) -> LiveFoulPlayBoundary:
    """Build a restorable local snapshot only when both boundary requests match.

    BattleStream emits raw request *lines*, whereas its generic snapshot carries
    the same requests as JSON objects.  Comparing their canonical JSON forms is
    the stable byte-level contract across that representation boundary.  The raw
    line digests are retained separately for the durable receipt.
    """

    if not source_battle_id:
        raise LiveFoulPlayContinuationError("live source battle id must be non-empty")
    if not format_id:
        raise LiveFoulPlayContinuationError("live source format id must be non-empty")
    raw_boundary_requests = bridge_snapshot.get("boundaryRequests")
    if not isinstance(raw_boundary_requests, Mapping):
        raise LiveFoulPlayContinuationError("generic live snapshot has no boundaryRequests object")

    latest_requests: dict[PlayerId, dict[str, Any]] = {}
    first_requests: dict[PlayerId, dict[str, Any]] = {}
    request_history: dict[PlayerId, tuple[Mapping[str, Any], ...]] = {}
    source_request_sha256: dict[PlayerId, str] = {}
    snapshot_request_sha256: dict[PlayerId, str] = {}

    for player in _PLAYER_IDS:
        raw_line = current_request_lines.get(player)
        snapshot_request = raw_boundary_requests.get(player)
        if not isinstance(raw_line, str) or not raw_line:
            raise LiveFoulPlayContinuationError(f"live boundary is missing the current {player} request")
        if not isinstance(snapshot_request, Mapping):
            raise LiveFoulPlayContinuationError(
                f"generic live snapshot is missing the {player} boundary request"
            )
        live_request = _request_payload(raw_line, player=player)
        snapshot_request_clone = _clone_mapping(snapshot_request)
        if _canonical_json(live_request) != _canonical_json(snapshot_request_clone):
            raise LiveFoulPlayContinuationError(
                f"generic live snapshot {player} boundary request does not match the live request"
            )
        history_payloads = tuple(
            _request_payload(line, player=player)
            for line in request_history_lines.get(player, ())
        )
        if not history_payloads:
            history_payloads = (live_request,)
        latest_requests[player] = _clone_mapping(live_request)
        first_requests[player] = _clone_mapping(history_payloads[0])
        request_history[player] = tuple(_clone_mapping(item) for item in history_payloads)
        source_request_sha256[player] = _sha256_text(raw_line)
        snapshot_request_sha256[player] = _sha256_text(_canonical_json(snapshot_request_clone))

    protocol_lines = tuple(str(line) for line in public_protocol_lines)
    replay = parse_showdown_replay(
        protocol_lines,
        battle_id=source_battle_id,
        complete_prefix=True,
        hp_visibility={"p1": "exact", "p2": "exact"},
    )
    if replay.winner is not None:
        raise LiveFoulPlayContinuationError("cannot continue a terminal live FoulPlay state")
    belief_engine = PublicBattleBeliefEngine.from_events(
        replay.public_events,
        format_id=format_id,
        set_source=belief_set_source,
    )
    snapshot = LocalShowdownSnapshot(
        battle_token=f"live-foulplay-{source_battle_id}",
        battle_id=source_battle_id,
        format_id=format_id,
        observation_format_id=format_id,
        bridge_snapshot=_clone_mapping(bridge_snapshot),
        protocol_lines=protocol_lines,
        latest_requests=latest_requests,
        first_requests=first_requests,
        request_history=request_history,
        replay=replay,
        belief_engine=belief_engine,
        latest_turn=replay.turn_number,
        terminal=None,
    )
    return LiveFoulPlayBoundary(
        snapshot=snapshot,
        source_request_sha256=source_request_sha256,
        snapshot_request_sha256=snapshot_request_sha256,
    )


def run_live_foulplay_continuation(
    *,
    boundary: LiveFoulPlayBoundary,
    source_seed: int,
    source_decision_round: int,
    pokezero_action: int,
    foulplay_action: int,
    foulplay_choice: str,
    pokezero_player: PlayerId = "p1",
    foulplay_player: PlayerId = "p2",
    allow_opening_boundary: bool = False,
    allow_terminal_fixed_step: bool = False,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]],
    rollout_config: RolloutConfig,
) -> dict[str, Any]:
    """Execute one fixed live joint action then a fresh local continuation.

    The caller intentionally supplies *factories*: no source-game policy or
    source battle shell is reused for the continuation.  Any rejected restore,
    capped terminal, capped continuation, or missing terminal result fails the
    source battle before its choices can be submitted.  Callers that explicitly
    opt in may record an uncapped terminal reached by the fixed joint step as a
    zero-decision direct outcome.
    """

    if source_decision_round < 0:
        raise LiveFoulPlayContinuationError("live continuation source decision round must be non-negative")
    if source_decision_round == 0 and not allow_opening_boundary:
        raise LiveFoulPlayContinuationError("live continuation smoke refuses opening-round states")
    if not foulplay_choice.strip():
        raise LiveFoulPlayContinuationError("live continuation smoke requires a decoded FoulPlay choice")
    if {pokezero_player, foulplay_player} != set(_PLAYER_IDS) or pokezero_player == foulplay_player:
        raise LiveFoulPlayContinuationError(
            "live continuation requires distinct p1/p2 PokeZero and FoulPlay seats"
        )
    if rollout_config.max_decision_rounds <= source_decision_round + 1 and not allow_terminal_fixed_step:
        raise LiveFoulPlayContinuationError("continuation rollout leaves no decision round after the fixed joint step")
    env = env_factory()
    try:
        env.reset(seed=source_seed, format_id=boundary.snapshot.format_id)
        restore = getattr(env, "restore", None)
        if not callable(restore):
            raise LiveFoulPlayContinuationError("continuation environment does not support generic snapshot restore")
        restore(boundary.snapshot)
        if tuple(env.requested_players()) != _PLAYER_IDS:
            raise LiveFoulPlayContinuationError("restored live snapshot is not a simultaneous p1/p2 boundary")
        first_restored_joint_step = {
            pokezero_player: pokezero_action,
            foulplay_player: foulplay_action,
        }
        first_step = env.step(first_restored_joint_step)
        if first_step.terminal is not None:
            if not allow_terminal_fixed_step:
                raise LiveFoulPlayContinuationError("fixed live joint step reached terminal before continuation")
            if first_step.terminal.capped:
                raise LiveFoulPlayContinuationError("fixed live joint step ended in a capped terminal state")
            return {
                "source_battle_id": boundary.snapshot.battle_id,
                "source_seed": source_seed,
                "source_decision_round": source_decision_round,
                "source_request_sha256": dict(boundary.source_request_sha256),
                "snapshot_request_sha256": dict(boundary.snapshot_request_sha256),
                "actual_foulplay_choice": foulplay_choice,
                "decoded_actual_foulplay_action": foulplay_action,
                "first_restored_joint_step": first_restored_joint_step,
                "continuation": {
                    "decision_round_count": 0,
                    "terminal_after_fixed_joint_step": True,
                    "terminal": {
                        "winner": first_step.terminal.winner,
                        "turn_count": first_step.terminal.turn_count,
                        "capped": False,
                    },
                },
            }
        policies = dict(continuation_policy_factory())
        if set(policies) != set(_PLAYER_IDS):
            raise LiveFoulPlayContinuationError("fresh continuation policies must cover exactly p1 and p2")
        continuation = continue_rollout_from_current_state(
            env=env,
            policies=policies,
            config=rollout_config,
            seed=source_seed,
            battle_id=f"{boundary.snapshot.battle_id}-live-continuation",
            starting_decision_round_index=source_decision_round + 1,
            available_observations=first_step.observations,
            reset_policies=True,
        )
        if continuation.terminal.capped:
            raise LiveFoulPlayContinuationError("live continuation smoke capped before a terminal result")
        return {
            "source_battle_id": boundary.snapshot.battle_id,
            "source_seed": source_seed,
            "source_decision_round": source_decision_round,
            "source_request_sha256": dict(boundary.source_request_sha256),
            "snapshot_request_sha256": dict(boundary.snapshot_request_sha256),
            "actual_foulplay_choice": foulplay_choice,
            "decoded_actual_foulplay_action": foulplay_action,
            "first_restored_joint_step": first_restored_joint_step,
            "continuation": {
                "decision_round_count": continuation.decision_round_count,
                "terminal_after_fixed_joint_step": False,
                "terminal": {
                    "winner": continuation.terminal.winner,
                    "turn_count": continuation.terminal.turn_count,
                    "capped": continuation.terminal.capped,
                },
            },
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def select_live_foulplay_continuation_oracle_action(
    *,
    boundary: LiveFoulPlayBoundary,
    source_seed: int,
    source_decision_round: int,
    raw_action: int,
    legal_actions: Sequence[int],
    foulplay_action: int,
    foulplay_choice: str,
    pokezero_player: PlayerId,
    foulplay_player: PlayerId,
    candidate_cap: int,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]],
    rollout_config: RolloutConfig,
) -> LiveFoulPlayContinuationOracleDecision:
    """Choose a live action by evaluating every legal candidate to terminal.

    This is intentionally a controller, not a regular policy: it receives the
    generic live snapshot only after the external FoulPlay choice is decoded,
    and it never gives that snapshot to the raw checkpoint policy.  The action
    set is *all* legal actions at the boundary.  A candidate cap is therefore a
    safety contract, never a request to truncate the set.  Every malformed
    candidate list, restore/binding failure, capped terminal/continuation, or
    failed candidate raises -- a purported oracle arm must not silently submit
    the raw action because its controller could not run.  An uncapped terminal
    after the fixed PokeZero/FoulPlay step is an exact direct outcome, not a
    continuation failure: it is retained with zero continuation decisions.
    """

    if candidate_cap <= 0:
        raise LiveFoulPlayContinuationError("live continuation oracle candidate cap must be positive")
    if source_decision_round < 1:
        raise LiveFoulPlayContinuationError(
            "B2 live continuation oracle requires a mid-game source decision round"
        )
    candidates = tuple(int(action) for action in legal_actions)
    if not candidates:
        raise LiveFoulPlayContinuationError("live continuation oracle received no legal candidates")
    if len(set(candidates)) != len(candidates):
        raise LiveFoulPlayContinuationError("live continuation oracle received duplicate legal candidates")
    if any(action < 0 or action >= ACTION_COUNT for action in candidates):
        raise LiveFoulPlayContinuationError(
            "live continuation oracle received an action outside the PokeZero action space"
        )
    if raw_action not in candidates:
        raise LiveFoulPlayContinuationError(
            "live continuation oracle raw action is not among the live legal candidates"
        )
    if len(candidates) > candidate_cap:
        raise LiveFoulPlayContinuationError(
            "live continuation oracle legal candidate count "
            f"{len(candidates)} exceeds cap {candidate_cap}; refusing to truncate"
        )

    scored: list[dict[str, Any]] = []
    for action in candidates:
        proof = run_live_foulplay_continuation(
            boundary=boundary,
            source_seed=source_seed,
            source_decision_round=source_decision_round,
            pokezero_action=action,
            foulplay_action=foulplay_action,
            foulplay_choice=foulplay_choice,
            pokezero_player=pokezero_player,
            foulplay_player=foulplay_player,
            allow_opening_boundary=False,
            allow_terminal_fixed_step=True,
            env_factory=env_factory,
            continuation_policy_factory=continuation_policy_factory,
            rollout_config=rollout_config,
        )
        continuation = proof.get("continuation")
        if not isinstance(continuation, Mapping):
            raise LiveFoulPlayContinuationError("live continuation oracle candidate lacks continuation readout")
        terminal = continuation.get("terminal")
        if not isinstance(terminal, Mapping) or terminal.get("capped") is not False:
            raise LiveFoulPlayContinuationError("live continuation oracle candidate capped or lacks terminal")
        continuation_decision_round_count = continuation.get("decision_round_count")
        if (
            isinstance(continuation_decision_round_count, bool)
            or not isinstance(continuation_decision_round_count, int)
            or continuation_decision_round_count < 0
        ):
            raise LiveFoulPlayContinuationError(
                "live continuation oracle candidate has an invalid "
                "continuation decision count"
            )
        terminal_after_fixed_joint_step = continuation.get(
            "terminal_after_fixed_joint_step"
        )
        if not isinstance(terminal_after_fixed_joint_step, bool):
            raise LiveFoulPlayContinuationError(
                "live continuation oracle candidate must declare whether "
                "the fixed joint step "
                "terminated"
            )
        if terminal_after_fixed_joint_step and continuation_decision_round_count != 0:
            raise LiveFoulPlayContinuationError(
                "terminal fixed-step candidate must have zero continuation decisions"
            )
        if (
            not terminal_after_fixed_joint_step
            and continuation_decision_round_count < 1
        ):
            raise LiveFoulPlayContinuationError(
                "B2 live continuation oracle candidate had no decision after the fixed joint step"
            )
        winner = terminal.get("winner")
        if winner == pokezero_player:
            score = 1.0
        elif winner is None:
            score = 0.5
        elif winner == foulplay_player:
            score = 0.0
        else:
            raise LiveFoulPlayContinuationError(
                f"live continuation oracle candidate has unknown winner {winner!r}"
            )
        scored.append(
            {
                "action_index": action,
                "score": score,
                "continuation_decision_round_count": continuation_decision_round_count,
                "terminal": {
                    "winner": winner,
                    "turn_count": terminal.get("turn_count"),
                    "capped": False,
                },
                "terminal_after_fixed_joint_step": terminal_after_fixed_joint_step,
            }
        )

    # Stable action-index tie-break.  In particular, the raw action receives no
    # privileged tie break: choosing it is an oracle result only when it wins the
    # same comparison as every other candidate.
    selected = max(scored, key=lambda candidate: (float(candidate["score"]), -int(candidate["action_index"])))
    selected_action = int(selected["action_index"])
    return LiveFoulPlayContinuationOracleDecision(
        action_index=selected_action,
        metadata={
            "schema_version": LIVE_FOULPLAY_CONTINUATION_ORACLE_SCHEMA_VERSION,
            "controller": "live-foulplay-continuation-oracle",
            "controller_status": "oracle-selected",
            "full_state_snapshot_scope": "controller-only",
            "source_decision_round": source_decision_round,
            "pokezero_player": pokezero_player,
            "foulplay_player": foulplay_player,
            "source_request_sha256": dict(boundary.source_request_sha256),
            "snapshot_request_sha256": dict(boundary.snapshot_request_sha256),
            "actual_foulplay_choice": foulplay_choice,
            "decoded_actual_foulplay_action": foulplay_action,
            "raw_action_index": raw_action,
            "selected_action_index": selected_action,
            "selected_changed_raw_action": selected_action != raw_action,
            "first_restored_joint_step": {
                pokezero_player: selected_action,
                foulplay_player: foulplay_action,
            },
            "candidate_count": len(candidates),
            "candidate_cap": candidate_cap,
            "legal_action_indices": candidates,
            "candidates": tuple(scored),
        },
    )
