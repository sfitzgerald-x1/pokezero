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
from typing import Any, Callable, Mapping, Sequence

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
    source_battle_id: str,
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
    continuation_rng_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed source action to an uncapped terminal outcome.

    Both the fixed action and the continuation policies are evaluated in a
    newly created environment.  An immediate, uncapped terminal after the
    fixed joint action is evidence, not an error: a forced win or loss is the
    strongest possible continuation outcome.  Any capped continuation,
    missing restore capability, mismatched simultaneous boundary, or malformed
    policy factory fails closed. ``continuation_rng_seed`` controls only the
    post-branch policy RNG.  The source snapshot is *always* restored from
    ``source_seed``.  Separating those two identities is necessary for paired
    action studies: every candidate action in a trial must receive the same
    continuation randomness without pretending that it came from a different
    source battle.
    """

    if not isinstance(snapshot, LocalShowdownSnapshot):
        raise SealedOverrideContinuationError("sealed override audit requires LocalShowdownSnapshot")
    if not isinstance(source_battle_id, str) or not source_battle_id:
        raise SealedOverrideContinuationError("source battle id must be non-empty")
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
    if continuation_rng_seed is None:
        continuation_rng_seed = source_seed
    if (
        isinstance(continuation_rng_seed, bool)
        or not isinstance(continuation_rng_seed, int)
        or continuation_rng_seed < 0
    ):
        raise SealedOverrideContinuationError(
            "continuation RNG seed must be a non-negative integer when set"
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
        # Do not retain the joint action.  The opponent's committed action is
        # used exactly once to make the two subject-action arms comparable,
        # then remains inside this trusted controller.  The durable readout
        # may say that it was held fixed, but must not disclose it.
        common = {
            "schema_version": SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION,
            # `snapshot.battle_id` is the environment-local identifier
            # (for example `local-gen3randombattle-...`), whereas the audit
            # ledger must bind to the caller's rollout battle identity.  The
            # snapshot is ephemeral and never serialized, so use the latter.
            "source_battle_id": source_battle_id,
            "source_seed": source_seed,
            "source_decision_round": source_decision_round,
            "continuation_rng_seed": continuation_rng_seed,
            "subject_player": subject_player,
            "opponent_player": opponent_player,
            "action_label": action_label,
            "subject_action": subject_action,
            "opponent_action_held_fixed": True,
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
        # A source rollout may have any combination of public/progress/audit
        # hooks installed.  A fresh continuation must not re-enter the audit
        # controller or append source-run progress/evidence.  Its only output
        # is the terminal summary returned below.
        config = replace(
            rollout_config,
            decision_sink=None,
            public_decision_sink=None,
            sealed_pre_step_sink=None,
        )
        if max_continuation_decision_rounds is not None:
            config = replace(
                config,
                max_decision_rounds=(
                    source_decision_round + 1 + max_continuation_decision_rounds
                ),
            )
        continuation = continue_rollout_from_current_state(
            env=env,
            policies=policies,
            config=config,
            seed=continuation_rng_seed,
            battle_id=f"{snapshot.battle_id}-sealed-override-continuation",
            starting_decision_round_index=source_decision_round + 1,
            available_observations=first_step.observations,
            # Factories supply fresh policies, but a source-bound controller
            # may deliberately seed their private history to the exact prefix
            # before this branch. Resetting here would silently turn that
            # suffix into a cold-start policy evaluation.
            reset_policies=False,
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


def evaluate_sealed_override_pair(
    *,
    snapshot: LocalShowdownSnapshot,
    source_battle_id: str,
    source_seed: int,
    source_decision_round: int,
    subject_player: PlayerId,
    mcts_action: int,
    raw_action: int,
    opponent_player: PlayerId,
    opponent_action: int,
    search_evidence: Mapping[str, Any],
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]],
    rollout_config: RolloutConfig,
    max_continuation_decision_rounds: int | None = None,
) -> dict[str, Any]:
    """Return matched independent outcomes for an actual MCTS override.

    Both arms begin from the same ephemeral snapshot and hold the opponent's
    already-selected source action fixed.  Each arm allocates a new simulator
    and a new continuation policy set, so neither source search state nor one
    candidate's continuation state can leak into the other.  ``search_evidence``
    is caller-supplied, sanitized root Q/visit data; this controller binds it
    to the action pair but never receives or returns a serializable snapshot.
    """

    _validate_action(mcts_action, label="MCTS action")
    _validate_action(raw_action, label="raw action")
    if mcts_action == raw_action:
        raise SealedOverrideContinuationError(
            "sealed override pair requires distinct MCTS and raw actions"
        )
    if not isinstance(search_evidence, Mapping):
        raise SealedOverrideContinuationError("search evidence must be a mapping")
    mcts = run_sealed_override_continuation(
        snapshot=snapshot,
        source_battle_id=source_battle_id,
        source_seed=source_seed,
        source_decision_round=source_decision_round,
        subject_player=subject_player,
        subject_action=mcts_action,
        opponent_player=opponent_player,
        opponent_action=opponent_action,
        action_label="mcts-override",
        env_factory=env_factory,
        continuation_policy_factory=continuation_policy_factory,
        rollout_config=rollout_config,
        max_continuation_decision_rounds=max_continuation_decision_rounds,
    )
    raw = run_sealed_override_continuation(
        snapshot=snapshot,
        source_battle_id=source_battle_id,
        source_seed=source_seed,
        source_decision_round=source_decision_round,
        subject_player=subject_player,
        subject_action=raw_action,
        opponent_player=opponent_player,
        opponent_action=opponent_action,
        action_label="raw-policy",
        env_factory=env_factory,
        continuation_policy_factory=continuation_policy_factory,
        rollout_config=rollout_config,
        max_continuation_decision_rounds=max_continuation_decision_rounds,
    )
    return {
        "schema_version": "pokezero.sealed-override-pair.v1",
        "source_battle_id": mcts["source_battle_id"],
        "source_seed": source_seed,
        "source_decision_round": source_decision_round,
        "subject_player": subject_player,
        "opponent_player": opponent_player,
        "mcts_action": mcts_action,
        "raw_action": raw_action,
        "opponent_action_held_fixed": True,
        "search_evidence": dict(search_evidence),
        "mcts": mcts["continuation"],
        "raw": raw["continuation"],
    }


def evaluate_sealed_root_action_grid(
    *,
    snapshot: LocalShowdownSnapshot,
    source_battle_id: str,
    source_seed: int,
    source_decision_round: int,
    subject_player: PlayerId,
    actions: Mapping[str, int],
    opponent_player: PlayerId,
    opponent_action: int,
    continuation_policy_factories: Mapping[str, Callable[[], Mapping[PlayerId, Policy]]],
    continuation_rng_seeds: Mapping[str, Sequence[int]] | Sequence[int],
    search_evidence: Mapping[str, Any],
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    max_continuation_decision_rounds: int | None = None,
) -> dict[str, Any]:
    """Evaluate selected root actions under matched continuation targets.

    This is intentionally a controller-only primitive for the root-action
    target-quality audit.  Every row restores the exact same sealed source
    snapshot, holds the already-selected opponent action fixed, and changes
    only the subject's root action plus the declared continuation target.  A
    continuation target (for example ``policy_consistent`` or
    ``uniform_own``) owns a *fresh* policy factory for every action/trial.

    A trial's RNG seed is shared across every action in a target.  This is the
    common-random-numbers contract; callers must not assign a separate seed to
    each action and then report the result as paired.  The returned durable
    object never includes the snapshot or opponent action.
    """

    if not isinstance(search_evidence, Mapping):
        raise SealedOverrideContinuationError("search evidence must be a mapping")
    if not isinstance(actions, Mapping) or not 2 <= len(actions) <= 3:
        raise SealedOverrideContinuationError(
            "root action grid requires two or three labelled candidate actions"
        )
    normalized_actions: list[tuple[str, int]] = []
    for label, action in actions.items():
        if not isinstance(label, str) or not label.strip():
            raise SealedOverrideContinuationError("root action label must be a non-empty string")
        _validate_action(action, label=f"root action {label!r}")
        normalized_actions.append((label, action))
    if len({label for label, _ in normalized_actions}) != len(normalized_actions):
        raise SealedOverrideContinuationError("root action grid repeats a label")
    if len({action for _, action in normalized_actions}) != len(normalized_actions):
        raise SealedOverrideContinuationError("root action grid repeats an action")
    if not isinstance(continuation_policy_factories, Mapping) or not continuation_policy_factories:
        raise SealedOverrideContinuationError("root action grid requires continuation target factories")
    normalized_modes: list[tuple[str, Callable[[], Mapping[PlayerId, Policy]]]] = []
    for mode, factory in continuation_policy_factories.items():
        if not isinstance(mode, str) or not mode.strip():
            raise SealedOverrideContinuationError("continuation target name must be a non-empty string")
        if not callable(factory):
            raise SealedOverrideContinuationError(
                f"continuation target {mode!r} factory must be callable"
            )
        normalized_modes.append((mode, factory))
    if len({mode for mode, _ in normalized_modes}) != len(normalized_modes):
        raise SealedOverrideContinuationError("root action grid repeats a continuation target")
    if isinstance(continuation_rng_seeds, Mapping):
        if set(continuation_rng_seeds) != {target for target, _ in normalized_modes}:
            raise SealedOverrideContinuationError(
                "root action grid target RNG schedules do not match continuation targets"
            )
        normalized_seeds_by_target = {
            target: list(continuation_rng_seeds[target])
            for target, _ in normalized_modes
        }
    else:
        shared_seeds = list(continuation_rng_seeds)
        normalized_seeds_by_target = {
            target: list(shared_seeds)
            for target, _ in normalized_modes
        }
    for target, normalized_seeds in normalized_seeds_by_target.items():
        if not normalized_seeds:
            raise SealedOverrideContinuationError(
                f"root action grid target {target!r} requires at least one continuation RNG seed"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in normalized_seeds
        ) or len(set(normalized_seeds)) != len(normalized_seeds):
            raise SealedOverrideContinuationError(
                f"root action grid target {target!r} RNG seeds must be unique non-negative integers"
            )

    target_rows: list[dict[str, Any]] = []
    for target, policy_factory in normalized_modes:
        trials: list[dict[str, Any]] = []
        for rng_seed in normalized_seeds_by_target[target]:
            outcomes: list[dict[str, Any]] = []
            for action_label, action_index in normalized_actions:
                outcome = run_sealed_override_continuation(
                    snapshot=snapshot,
                    source_battle_id=source_battle_id,
                    source_seed=source_seed,
                    source_decision_round=source_decision_round,
                    subject_player=subject_player,
                    subject_action=action_index,
                    opponent_player=opponent_player,
                    opponent_action=opponent_action,
                    action_label=action_label,
                    env_factory=env_factory,
                    continuation_policy_factory=policy_factory,
                    rollout_config=rollout_config,
                    max_continuation_decision_rounds=max_continuation_decision_rounds,
                    continuation_rng_seed=rng_seed,
                )
                outcomes.append(
                    {
                        "action_label": action_label,
                        "action_index": action_index,
                        "continuation": outcome["continuation"],
                    }
                )
            trials.append({"continuation_rng_seed": rng_seed, "outcomes": outcomes})
        target_rows.append({"target": target, "trials": trials})
    return {
        "schema_version": "pokezero.sealed-root-action-grid.v1",
        "source_battle_id": source_battle_id,
        "source_seed": source_seed,
        "source_decision_round": source_decision_round,
        "subject_player": subject_player,
        "opponent_player": opponent_player,
        "opponent_action_held_fixed": True,
        "actions": [
            {"action_label": label, "action_index": action}
            for label, action in normalized_actions
        ],
        "search_evidence": dict(search_evidence),
        "continuation_targets": target_rows,
    }
