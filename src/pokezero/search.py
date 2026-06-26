"""Search helpers built on replay-from-root branch rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from .actions import ACTION_COUNT
from .env import PlayerId, PokeZeroEnv, TerminalState
from .observation import PokeZeroObservationV0
from .policy import Policy
from .replay_branching import (
    ReplayBranchResult,
    ReplayBranchRolloutResult,
    replay_trajectory_branch,
    replay_trajectory_branch_rollout,
)
from .rollout import RolloutConfig
from .trajectory import BattleTrajectory

ObservationValueFunction = Callable[[tuple[PokeZeroObservationV0, ...]], float]


@dataclass(frozen=True)
class BranchSearchCandidate:
    action_index: int
    value: float
    terminal: TerminalState
    rollout: ReplayBranchRolloutResult

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "value": self.value,
            "terminal": {
                "winner": self.terminal.winner,
                "turn_count": self.terminal.turn_count,
                "capped": self.terminal.capped,
            },
            "continuation_decision_round_count": self.rollout.continuation.decision_round_count,
        }


@dataclass(frozen=True)
class ValueBranchSearchCandidate:
    action_index: int
    value: float
    terminal: TerminalState | None
    branch: ReplayBranchResult
    evaluated_history_length: int

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "value": self.value,
            "terminal": None
            if self.terminal is None
            else {
                "winner": self.terminal.winner,
                "turn_count": self.terminal.turn_count,
                "capped": self.terminal.capped,
            },
            "post_branch_requested_players": list(self.branch.step_result.requested_players),
            "evaluated_history_length": self.evaluated_history_length,
        }


@dataclass(frozen=True)
class ValueBranchSearchResult:
    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    candidates: tuple[ValueBranchSearchCandidate, ...]

    @property
    def best_candidate(self) -> ValueBranchSearchCandidate:
        if not self.candidates:
            raise ValueError("value branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.value, -candidate.action_index))

    @property
    def action_index(self) -> int:
        return self.best_candidate.action_index

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "selected_action_index": best.action_index,
            "selected_value": best.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class FlatBranchSearchResult:
    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    candidates: tuple[BranchSearchCandidate, ...]

    @property
    def best_candidate(self) -> BranchSearchCandidate:
        if not self.candidates:
            raise ValueError("flat branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.value, -candidate.action_index))

    @property
    def action_index(self) -> int:
        return self.best_candidate.action_index

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "selected_action_index": best.action_index,
            "selected_value": best.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def value_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
) -> ValueBranchSearchResult:
    """Enumerate legal root actions and score post-branch states with a value function."""

    if len(legal_action_mask) != ACTION_COUNT:
        raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
    candidate_indices = tuple(index for index, legal in enumerate(legal_action_mask) if legal)
    if not candidate_indices:
        raise ValueError("value branch search requires at least one legal action.")
    if player_id in opponent_actions:
        raise ValueError("opponent_actions must not include the searched player.")

    prefix_history = _player_observation_history(
        trajectory,
        player_id=player_id,
        through_decision_round=prefix_decision_round_count,
    )
    candidates: list[ValueBranchSearchCandidate] = []
    for action_index in candidate_indices:
        branch_actions = {
            **dict(opponent_actions),
            player_id: action_index,
        }
        branch = replay_trajectory_branch(
            env,
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
            branch_actions=branch_actions,
        )
        terminal = branch.step_result.terminal
        if terminal is not None:
            value = terminal_value_for_player(terminal, player_id=player_id)
            evaluated_history = prefix_history
        else:
            post_branch_observation = branch.step_result.observations.get(player_id)
            if post_branch_observation is None:
                post_branch_observation = env.observe(player_id)
            evaluated_history = (*prefix_history, post_branch_observation)
            value = float(value_fn(evaluated_history))
            if not math.isfinite(value):
                raise ValueError("value_fn returned a non-finite branch value.")
        candidates.append(
            ValueBranchSearchCandidate(
                action_index=action_index,
                value=value,
                terminal=terminal,
                branch=branch,
                evaluated_history_length=len(evaluated_history),
            )
        )

    return ValueBranchSearchResult(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        candidates=tuple(candidates),
    )


def flat_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    rollout_policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
) -> FlatBranchSearchResult:
    """Enumerate legal root actions, roll each branch out, and score terminal outcomes."""

    if len(legal_action_mask) != ACTION_COUNT:
        raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
    candidate_indices = tuple(index for index, legal in enumerate(legal_action_mask) if legal)
    if not candidate_indices:
        raise ValueError("flat branch search requires at least one legal action.")
    if player_id in opponent_actions:
        raise ValueError("opponent_actions must not include the searched player.")

    candidates: list[BranchSearchCandidate] = []
    for action_index in candidate_indices:
        branch_actions = {
            **dict(opponent_actions),
            player_id: action_index,
        }
        rollout = replay_trajectory_branch_rollout(
            env,
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
            branch_actions=branch_actions,
            policies=rollout_policies,
            rollout_config=rollout_config,
            battle_id=f"flat-branch-search-{player_id}-{prefix_decision_round_count}-{action_index}",
        )
        terminal = rollout.continuation.terminal
        candidates.append(
            BranchSearchCandidate(
                action_index=action_index,
                value=terminal_value_for_player(terminal, player_id=player_id),
                terminal=terminal,
                rollout=rollout,
            )
        )

    return FlatBranchSearchResult(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        candidates=tuple(candidates),
    )


def terminal_value_for_player(terminal: TerminalState, *, player_id: PlayerId) -> float:
    if terminal.winner == player_id:
        return 1.0
    if terminal.winner is None:
        return 0.0
    return -1.0


def _player_observation_history(
    trajectory: BattleTrajectory,
    *,
    player_id: PlayerId,
    through_decision_round: int,
) -> tuple[PokeZeroObservationV0, ...]:
    return tuple(
        step.observation
        for step in trajectory.steps
        if step.player_id == player_id and step.turn_index <= through_decision_round
    )
