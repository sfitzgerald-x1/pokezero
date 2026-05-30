"""Rollout driver for wiring policies to PokeZero environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Mapping

from .env import BattleFormat, PlayerId, PokeZeroEnv, TerminalState
from .policy import Policy, PolicyDecision
from .trajectory import BattleTrajectory, TrajectoryStep


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 250
    format_id: BattleFormat = "gen3randombattle"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class RolloutResult:
    trajectory: BattleTrajectory
    terminal: TerminalState
    step_count: int


@dataclass
class RolloutDriver:
    env: PokeZeroEnv
    policies: Mapping[PlayerId, Policy]
    config: RolloutConfig = field(default_factory=RolloutConfig)

    def run(self, *, seed: int, battle_id: str = "rollout") -> RolloutResult:
        rng = random.Random(seed)
        self.env.reset(seed=seed, format_id=self.config.format_id)
        trajectory = BattleTrajectory(
            battle_id=battle_id,
            format_id=self.config.format_id,
            seed=seed,
            metadata={"max_steps": self.config.max_steps},
        )

        for turn_index in range(self.config.max_steps):
            terminal = self.env.terminal()
            if terminal is not None:
                trajectory.record_terminal(terminal)
                return RolloutResult(trajectory=trajectory, terminal=terminal, step_count=turn_index)

            requested_players = self.env.requested_players()
            if not requested_players:
                terminal = self.env.terminal()
                if terminal is not None:
                    trajectory.record_terminal(terminal)
                    return RolloutResult(trajectory=trajectory, terminal=terminal, step_count=turn_index)
                raise ValueError("environment requested no players before reaching terminal state.")

            decisions: dict[PlayerId, PolicyDecision] = {}
            observations = {}
            for player_id in requested_players:
                policy = self._policy_for_player(player_id)
                observation = self.env.observe(player_id)
                decision = policy.select_action(observation, rng=rng)
                decisions[player_id] = decision
                observations[player_id] = observation

            step_result = self.env.step(
                {player_id: decision.action_index for player_id, decision in decisions.items()}
            )

            for player_id in requested_players:
                decision = decisions[player_id]
                opponent_action_index = _opponent_action_index(player_id, decisions)
                trajectory.append(
                    TrajectoryStep(
                        player_id=player_id,
                        turn_index=turn_index,
                        observation=observations[player_id],
                        legal_action_mask=tuple(observations[player_id].legal_action_mask),
                        action_index=decision.action_index,
                        reward=float(step_result.rewards.get(player_id, 0.0)),
                        opponent_action_index=opponent_action_index,
                        action_probability=decision.action_probability,
                        metadata={
                            "policy_id": decision.policy_id,
                            **dict(decision.metadata),
                        },
                    )
                )

            if step_result.terminal is not None:
                trajectory.record_terminal(step_result.terminal)
                return RolloutResult(
                    trajectory=trajectory,
                    terminal=step_result.terminal,
                    step_count=turn_index + 1,
                )

        terminal = TerminalState(winner=None, turn_count=self.config.max_steps, capped=True)
        trajectory.record_terminal(terminal)
        return RolloutResult(
            trajectory=trajectory,
            terminal=terminal,
            step_count=self.config.max_steps,
        )

    def _policy_for_player(self, player_id: PlayerId) -> Policy:
        try:
            return self.policies[player_id]
        except KeyError as exc:
            raise ValueError(f"no policy configured for requested player {player_id!r}.") from exc


def _opponent_action_index(
    player_id: PlayerId,
    decisions: Mapping[PlayerId, PolicyDecision],
) -> int | None:
    if len(decisions) != 2:
        return None
    for other_player, decision in decisions.items():
        if other_player != player_id:
            return decision.action_index
    return None
