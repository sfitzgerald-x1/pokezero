"""Search helpers built on replay-from-root branch rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .actions import ACTION_COUNT
from .env import PlayerId, PokeZeroEnv, TerminalState
from .policy import Policy
from .replay_branching import ReplayBranchRolloutResult, replay_trajectory_branch_rollout
from .rollout import RolloutConfig
from .trajectory import BattleTrajectory


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
