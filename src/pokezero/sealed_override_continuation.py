"""Independent, source-bound continuations for MCTS override audits.

The public evaluation ledger intentionally excludes private simulator state and
the opponent's raw action.  That makes it safe to publish but insufficient to
ask whether a searched override was actually better.  This controller accepts
one *ephemeral* actionable snapshot at the source boundary, replays a fixed
joint action in a fresh environment, and retains only bindings and terminal
outcomes.  The snapshot is never returned or serialized.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from .actions import ACTION_COUNT
from .env import PlayerId, PokeZeroEnv
from .local_showdown import LocalShowdownSnapshot
from .policy import Policy
from .rollout import RolloutConfig, continue_rollout_from_current_state


SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION = (
    "pokezero.sealed-override-continuation.v1"
)
_PLAYERS: tuple[PlayerId, PlayerId] = ("p1", "p2")


class SealedOverrideContinuationError(RuntimeError):
    """The source boundary cannot support a valid independent continuation."""


def _validate_action(action: int, *, label: str) -> None:
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        raise SealedOverrideContinuationError(
            f"{label} must be an action index in [0, {ACTION_COUNT})"
        )


def run_sealed_override_continuation(
    *,
    snapshot: LocalShowdownSnapshot,
    source_seed: int,
    source_decision_round: int,
    subject_player: PlayerId,
    subject_action: int,
    opponent_player: PlayerId,
    opponent_action: int,
    action_label: str,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]],
    rollout_config: RolloutConfig,
    max_continuation_decision_rounds: int | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed source action to an uncapped terminal outcome.

    Both the fixed action and the continuation policies are evaluated in a
    newly created environment.  An immediate, uncapped terminal after the
    fixed joint action is evidence, not an error: a forced win or loss is the
    strongest possible continuation outcome.  Any capped continuation,
    missing restore capability, mismatched simultaneous boundary, or malformed
    policy factory fails closed.
    """

    if not isinstance(snapshot, LocalShowdownSnapshot):
        raise SealedOverrideContinuationError("sealed override audit requires LocalShowdownSnapshot")
    if source_decision_round < 0:
        raise SealedOverrideContinuationError("source decision round must be non-negative")
    if not action_label or not action_label.strip():
        raise SealedOverrideContinuationError("action label must be non-empty")
    if {subject_player, opponent_player} != set(_PLAYERS) or subject_player == opponent_player:
        raise SealedOverrideContinuationError("subject and opponent must be distinct p1/p2 seats")
    _validate_action(subject_action, label="subject action")
    _validate_action(opponent_action, label="opponent action")
    if (
        max_continuation_decision_rounds is not None
        and (
            isinstance(max_continuation_decision_rounds, bool)
            or not isinstance(max_continuation_decision_rounds, int)
            or max_continuation_decision_rounds <= 0
        )
    ):
        raise SealedOverrideContinuationError(
            "maximum continuation decision rounds must be a positive integer when set"
        )

    env = env_factory()
    try:
        env.reset(seed=source_seed, format_id=snapshot.format_id)
        restore = getattr(env, "restore", None)
        if not callable(restore):
            raise SealedOverrideContinuationError(
                "independent continuation environment does not support generic snapshot restore"
            )
        restore(snapshot)
        if env.terminal() is not None:
            raise SealedOverrideContinuationError("restored source boundary is already terminal")
        if tuple(env.requested_players()) != _PLAYERS:
            raise SealedOverrideContinuationError(
                "restored source boundary is not a simultaneous p1/p2 action boundary"
            )
        fixed_joint_action = {
            subject_player: subject_action,
            opponent_player: opponent_action,
        }
        first_step = env.step(fixed_joint_action)
        common = {
            "schema_version": SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION,
            "source_battle_id": snapshot.battle_id,
            "source_seed": source_seed,
            "source_decision_round": source_decision_round,
            "subject_player": subject_player,
            "opponent_player": opponent_player,
            "action_label": action_label,
            "fixed_joint_action": dict(fixed_joint_action),
        }
        if first_step.terminal is not None:
            if first_step.terminal.capped:
                raise SealedOverrideContinuationError(
                    "fixed source joint action reached a capped terminal state"
                )
            return {
                **common,
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
        if set(policies) != set(_PLAYERS):
            raise SealedOverrideContinuationError(
                "fresh continuation policies must cover exactly p1 and p2"
            )
        config = rollout_config
        if max_continuation_decision_rounds is not None:
            config = replace(
                rollout_config,
                max_decision_rounds=(
                    source_decision_round + 1 + max_continuation_decision_rounds
                ),
            )
        continuation = continue_rollout_from_current_state(
            env=env,
            policies=policies,
            config=config,
            seed=source_seed,
            battle_id=f"{snapshot.battle_id}-sealed-override-continuation",
            starting_decision_round_index=source_decision_round + 1,
            available_observations=first_step.observations,
            reset_policies=True,
        )
        if continuation.terminal.capped:
            raise SealedOverrideContinuationError(
                "independent continuation capped before a terminal result"
            )
        return {
            **common,
            "continuation": {
                "decision_round_count": continuation.decision_round_count,
                "terminal_after_fixed_joint_step": False,
                "terminal": {
                    "winner": continuation.terminal.winner,
                    "turn_count": continuation.terminal.turn_count,
                    "capped": False,
                },
            },
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
