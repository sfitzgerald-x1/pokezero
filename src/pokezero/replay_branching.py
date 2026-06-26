"""Replay-from-root helpers for future search/forking code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .actions import ACTION_COUNT
from .env import BattleFormat, PlayerId, PokeZeroEnv, StepResult, TerminalState
from .trajectory import BattleTrajectory


@dataclass(frozen=True)
class ReplayActionRound:
    """One environment decision boundary worth of recorded player actions."""

    turn_index: int
    actions: Mapping[PlayerId, int]

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not self.actions:
            raise ValueError("actions must be non-empty.")
        normalized = {
            str(player): int(action_index)
            for player, action_index in sorted(self.actions.items(), key=lambda item: str(item[0]))
        }
        invalid_actions = [
            action_index
            for action_index in normalized.values()
            if action_index < 0 or action_index >= ACTION_COUNT
        ]
        if invalid_actions:
            raise ValueError(f"action indices must be between 0 and {ACTION_COUNT - 1}.")
        object.__setattr__(self, "actions", normalized)


@dataclass(frozen=True)
class ReplayPrefixResult:
    """State summary after replaying a prefix into an environment."""

    replayed_round_count: int
    requested_players: tuple[PlayerId, ...]
    terminal: TerminalState | None


@dataclass(frozen=True)
class ReplayBranchResult:
    """State summary after replaying a prefix and submitting one branch action round."""

    prefix: ReplayPrefixResult
    branch_round: ReplayActionRound
    step_result: StepResult


def action_rounds_from_trajectory(
    trajectory: BattleTrajectory,
    *,
    decision_round_count: int | None = None,
) -> tuple[ReplayActionRound, ...]:
    """Group trajectory steps into replayable action rounds.

    ``TrajectoryStep.turn_index`` is the rollout driver's decision-round index. Search can replay
    the first ``decision_round_count`` rounds from the original seed, then submit a different next
    action to explore a branch.
    """

    if decision_round_count is not None and decision_round_count < 0:
        raise ValueError("decision_round_count must be non-negative when set.")

    grouped: dict[int, dict[PlayerId, int]] = {}
    for step in trajectory.steps:
        if decision_round_count is not None and step.turn_index >= decision_round_count:
            continue
        actions = grouped.setdefault(step.turn_index, {})
        if step.player_id in actions:
            raise ValueError(
                f"trajectory has duplicate action for player {step.player_id!r} "
                f"at decision round {step.turn_index}."
            )
        actions[step.player_id] = step.action_index

    expected_turn = 0
    rounds: list[ReplayActionRound] = []
    for turn_index in sorted(grouped):
        if turn_index != expected_turn:
            raise ValueError(
                f"trajectory action rounds must be contiguous from 0; "
                f"missing decision round {expected_turn}."
            )
        rounds.append(ReplayActionRound(turn_index=turn_index, actions=grouped[turn_index]))
        expected_turn += 1
    if decision_round_count is not None and len(rounds) != decision_round_count:
        raise ValueError(
            f"trajectory contains {len(rounds)} replayable decision rounds, "
            f"but {decision_round_count} were requested."
        )
    return tuple(rounds)


def replay_action_rounds(
    env: PokeZeroEnv,
    *,
    seed: int,
    format_id: BattleFormat = "gen3randombattle",
    action_rounds: tuple[ReplayActionRound, ...],
) -> ReplayPrefixResult:
    """Reset ``env`` and replay a recorded action prefix from the battle root."""

    env.reset(seed=seed, format_id=format_id)
    for expected_index, action_round in enumerate(action_rounds):
        if action_round.turn_index != expected_index:
            raise ValueError(
                f"action_rounds must be contiguous from 0; expected decision round "
                f"{expected_index}, got {action_round.turn_index}."
            )
        terminal = env.terminal()
        if terminal is not None:
            raise ValueError(
                f"cannot replay decision round {action_round.turn_index}; "
                "environment reached terminal early."
            )
        _require_requested_players(
            action_round,
            requested_players=env.requested_players(),
        )
        env.step(action_round.actions)

    return ReplayPrefixResult(
        replayed_round_count=len(action_rounds),
        requested_players=env.requested_players(),
        terminal=env.terminal(),
    )


def replay_trajectory_prefix(
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    *,
    decision_round_count: int,
) -> ReplayPrefixResult:
    """Replay the first N decision rounds from a trajectory into ``env``."""

    return replay_action_rounds(
        env,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        action_rounds=action_rounds_from_trajectory(
            trajectory,
            decision_round_count=decision_round_count,
        ),
    )


def replay_trajectory_branch(
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    *,
    prefix_decision_round_count: int,
    branch_actions: Mapping[PlayerId, int],
) -> ReplayBranchResult:
    """Replay a trajectory prefix, submit one explicit branch action, and leave ``env`` there."""

    prefix = replay_trajectory_prefix(
        env,
        trajectory,
        decision_round_count=prefix_decision_round_count,
    )
    if prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal replay prefix.")
    branch_round = ReplayActionRound(
        turn_index=prefix_decision_round_count,
        actions=branch_actions,
    )
    _require_requested_players(
        branch_round,
        requested_players=prefix.requested_players,
    )
    step_result = env.step(branch_round.actions)
    return ReplayBranchResult(
        prefix=prefix,
        branch_round=branch_round,
        step_result=step_result,
    )


def _require_requested_players(
    action_round: ReplayActionRound,
    *,
    requested_players: tuple[PlayerId, ...],
) -> None:
    requested_set = set(requested_players)
    action_players = set(action_round.actions)
    if action_players == requested_set:
        return
    missing = sorted(requested_set - action_players)
    extra = sorted(action_players - requested_set)
    details: list[str] = []
    if missing:
        details.append(f"missing requested players: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected players: {', '.join(extra)}")
    raise ValueError(
        f"replay actions for decision round {action_round.turn_index} "
        f"do not match environment request ({'; '.join(details)})."
    )
