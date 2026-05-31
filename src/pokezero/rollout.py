"""Rollout driver for wiring policies to PokeZero environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Mapping

from .env import BattleFormat, PlayerId, PokeZeroEnv, TerminalState
from .policy import Policy, PolicyDecision
from .trajectory import BattleTrajectory, TrajectoryStep


@dataclass(frozen=True)
class RolloutConfig:
    max_decision_rounds: int = 250
    format_id: BattleFormat = "gen3randombattle"

    def __post_init__(self) -> None:
        if self.max_decision_rounds <= 0:
            raise ValueError("max_decision_rounds must be positive.")


@dataclass(frozen=True)
class RolloutResult:
    trajectory: BattleTrajectory
    terminal: TerminalState
    decision_round_count: int


@dataclass
class RolloutDriver:
    env: PokeZeroEnv
    policies: Mapping[PlayerId, Policy]
    config: RolloutConfig = field(default_factory=RolloutConfig)

    def run(self, *, seed: int, battle_id: str = "rollout") -> RolloutResult:
        self.env.reset(seed=seed, format_id=self.config.format_id)
        player_rngs: dict[PlayerId, random.Random] = {}
        trajectory = BattleTrajectory(
            battle_id=battle_id,
            format_id=self.config.format_id,
            seed=seed,
            metadata={"max_decision_rounds": self.config.max_decision_rounds},
        )
        requested_players = self.env.requested_players()
        available_observations = {}

        for decision_round_index in range(self.config.max_decision_rounds):
            terminal = self.env.terminal()
            if terminal is not None:
                trajectory.record_terminal(terminal)
                return RolloutResult(
                    trajectory=trajectory,
                    terminal=terminal,
                    decision_round_count=decision_round_index,
                )

            if not requested_players:
                terminal = self.env.terminal()
                if terminal is not None:
                    trajectory.record_terminal(terminal)
                    return RolloutResult(
                        trajectory=trajectory,
                        terminal=terminal,
                        decision_round_count=decision_round_index,
                    )
                raise ValueError("environment requested no players before reaching terminal state.")

            decisions: dict[PlayerId, PolicyDecision] = {}
            observations = {}
            for player_id in requested_players:
                policy = self._policy_for_player(player_id)
                observation = available_observations.get(player_id) or self.env.observe(player_id)
                decision = policy.select_action(
                    observation,
                    rng=_rng_for_player(seed, player_id, player_rngs),
                )
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
                        turn_index=decision_round_index,
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
                    decision_round_count=decision_round_index + 1,
                )
            requested_players = step_result.requested_players
            available_observations = dict(step_result.observations)

        terminal = TerminalState(winner=None, turn_count=self.config.max_decision_rounds, capped=True)
        trajectory.record_terminal(terminal)
        return RolloutResult(
            trajectory=trajectory,
            terminal=terminal,
            decision_round_count=self.config.max_decision_rounds,
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


def _rng_for_player(
    seed: int,
    player_id: PlayerId,
    player_rngs: dict[PlayerId, random.Random],
) -> random.Random:
    rng = player_rngs.get(player_id)
    if rng is not None:
        return rng
    digest = hashlib.sha256(f"{seed}:{player_id}".encode("utf-8")).digest()
    player_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(player_seed)
    player_rngs[player_id] = rng
    return rng
