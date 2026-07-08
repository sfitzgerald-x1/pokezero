"""Rollout driver for wiring policies to PokeZero environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from time import perf_counter
from typing import Mapping

from .env import BattleFormat, PlayerId, PokeZeroEnv, TerminalState
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision
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
    # Per-phase collection timing, carried on the in-memory result as a sidecar so it can reach the
    # metrics accumulator WITHOUT being stamped into the trajectory/record — record payloads must
    # stay deterministic (twin-replay byte-exactness; the parallel-vs-serial equality invariant).
    # None when timing was not collected.
    timing: "RolloutTiming | None" = None


@dataclass
class RolloutTiming:
    env_reset_elapsed_seconds: float = 0.0
    env_reset_calls: int = 0
    env_observe_elapsed_seconds: float = 0.0
    env_observe_calls: int = 0
    policy_select_elapsed_seconds: float = 0.0
    policy_select_calls: int = 0
    env_step_elapsed_seconds: float = 0.0
    env_step_calls: int = 0

    def add_env_reset(self, elapsed_seconds: float) -> None:
        self.env_reset_elapsed_seconds += elapsed_seconds
        self.env_reset_calls += 1

    def add_env_observe(self, elapsed_seconds: float) -> None:
        self.env_observe_elapsed_seconds += elapsed_seconds
        self.env_observe_calls += 1

    def add_policy_select(self, elapsed_seconds: float) -> None:
        self.policy_select_elapsed_seconds += elapsed_seconds
        self.policy_select_calls += 1

    def add_env_step(self, elapsed_seconds: float) -> None:
        self.env_step_elapsed_seconds += elapsed_seconds
        self.env_step_calls += 1

    @property
    def total_profiled_elapsed_seconds(self) -> float:
        return (
            self.env_reset_elapsed_seconds
            + self.env_observe_elapsed_seconds
            + self.policy_select_elapsed_seconds
            + self.env_step_elapsed_seconds
        )

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "env_reset_elapsed_seconds": self.env_reset_elapsed_seconds,
            "env_reset_calls": self.env_reset_calls,
            "env_observe_elapsed_seconds": self.env_observe_elapsed_seconds,
            "env_observe_calls": self.env_observe_calls,
            "policy_select_elapsed_seconds": self.policy_select_elapsed_seconds,
            "policy_select_calls": self.policy_select_calls,
            "env_step_elapsed_seconds": self.env_step_elapsed_seconds,
            "env_step_calls": self.env_step_calls,
            "total_profiled_elapsed_seconds": self.total_profiled_elapsed_seconds,
        }
        if self.env_observe_calls:
            payload["average_env_observe_elapsed_seconds"] = (
                self.env_observe_elapsed_seconds / self.env_observe_calls
            )
        if self.policy_select_calls:
            payload["average_policy_select_elapsed_seconds"] = (
                self.policy_select_elapsed_seconds / self.policy_select_calls
            )
        if self.env_step_calls:
            payload["average_env_step_elapsed_seconds"] = (
                self.env_step_elapsed_seconds / self.env_step_calls
            )
        return payload


@dataclass
class RolloutDriver:
    env: PokeZeroEnv
    policies: Mapping[PlayerId, Policy]
    config: RolloutConfig = field(default_factory=RolloutConfig)

    def run(self, *, seed: int, battle_id: str = "rollout") -> RolloutResult:
        self._reset_policies()
        timing = RolloutTiming()
        reset_started = perf_counter()
        self.env.reset(seed=seed, format_id=self.config.format_id)
        timing.add_env_reset(perf_counter() - reset_started)
        return continue_rollout_from_current_state(
            env=self.env,
            policies=self.policies,
            config=self.config,
            seed=seed,
            battle_id=battle_id,
            initial_timing=timing,
        )

    def _policy_for_player(self, player_id: PlayerId) -> Policy:
        try:
            return self.policies[player_id]
        except KeyError as exc:
            raise ValueError(f"no policy configured for requested player {player_id!r}.") from exc

    def _reset_policies(self) -> None:
        _reset_unique_policies(self.policies)


def continue_rollout_from_current_state(
    *,
    env: PokeZeroEnv,
    policies: Mapping[PlayerId, Policy],
    config: RolloutConfig,
    seed: int,
    battle_id: str = "rollout-continuation",
    starting_decision_round_index: int = 0,
    available_observations: Mapping[PlayerId, PokeZeroObservationV0] | None = None,
    reset_policies: bool = False,
    initial_timing: RolloutTiming | None = None,
) -> RolloutResult:
    """Continue a rollout from the env's current request boundary without resetting it."""

    if starting_decision_round_index < 0:
        raise ValueError("starting_decision_round_index must be non-negative.")
    if starting_decision_round_index > config.max_decision_rounds:
        raise ValueError("starting_decision_round_index cannot exceed max_decision_rounds.")
    if reset_policies:
        _reset_unique_policies(policies)

    player_rngs: dict[PlayerId, random.Random] = {}
    timing = initial_timing or RolloutTiming()
    trajectory = BattleTrajectory(
        battle_id=battle_id,
        format_id=config.format_id,
        seed=seed,
        metadata={
            "max_decision_rounds": config.max_decision_rounds,
            "starting_decision_round_index": starting_decision_round_index,
        },
    )
    requested_players = env.requested_players()
    cached_observations = dict(available_observations or {})

    for decision_round_index in range(starting_decision_round_index, config.max_decision_rounds):
        terminal = env.terminal()
        if terminal is not None:
            trajectory.record_terminal(terminal)
            return _rollout_result(
                trajectory=trajectory,
                terminal=terminal,
                decision_round_count=decision_round_index - starting_decision_round_index,
                timing=timing,
            )

        if not requested_players:
            terminal = env.terminal()
            if terminal is not None:
                trajectory.record_terminal(terminal)
                return _rollout_result(
                    trajectory=trajectory,
                    terminal=terminal,
                    decision_round_count=decision_round_index - starting_decision_round_index,
                    timing=timing,
                )
            raise ValueError("environment requested no players before reaching terminal state.")

        requested_policies = {
            player_id: _policy_for_player(policies, player_id)
            for player_id in requested_players
        }
        decisions: dict[PlayerId, PolicyDecision] = {}
        observations = {}
        for player_id in requested_players:
            observation = cached_observations.get(player_id)
            if observation is None:
                observe_started = perf_counter()
                observation = env.observe(player_id)
                timing.add_env_observe(perf_counter() - observe_started)
            observations[player_id] = observation
        for player_id in requested_players:
            policy = requested_policies[player_id]
            observation = observations[player_id]
            context = PolicyContext(
                player_id=player_id,
                decision_round_index=decision_round_index,
                battle_id=battle_id,
                format_id=config.format_id,
                seed=seed,
                observation=observation,
                requested_players=tuple(requested_players),
                trajectory=_trajectory_snapshot(trajectory),
                requested_legal_action_masks={
                    requested_player: tuple(requested_observation.legal_action_mask)
                    for requested_player, requested_observation in observations.items()
                },
                requested_observations=dict(observations),
            )
            policy_started = perf_counter()
            decision = _select_policy_decision(
                policy,
                observation=observation,
                context=context,
                rng=_rng_for_player(seed, player_id, player_rngs),
            )
            timing.add_policy_select(perf_counter() - policy_started)
            decisions[player_id] = decision

        step_started = perf_counter()
        step_result = env.step(
            {player_id: decision.action_index for player_id, decision in decisions.items()}
        )
        timing.add_env_step(perf_counter() - step_started)

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
                    value_estimate=decision.value_estimate,
                    metadata={
                        "policy_id": decision.policy_id,
                        **dict(decision.metadata),
                    },
                )
            )

        if step_result.terminal is not None:
            trajectory.record_terminal(step_result.terminal)
            return _rollout_result(
                trajectory=trajectory,
                terminal=step_result.terminal,
                decision_round_count=decision_round_index - starting_decision_round_index + 1,
                timing=timing,
            )
        requested_players = step_result.requested_players
        cached_observations = dict(step_result.observations)

    terminal = TerminalState(winner=None, turn_count=config.max_decision_rounds, capped=True)
    trajectory.record_terminal(terminal)
    return _rollout_result(
        trajectory=trajectory,
        terminal=terminal,
        decision_round_count=config.max_decision_rounds - starting_decision_round_index,
        timing=timing,
    )


def _rollout_result(
    *,
    trajectory: BattleTrajectory,
    terminal: TerminalState,
    decision_round_count: int,
    timing: RolloutTiming,
) -> RolloutResult:
    return RolloutResult(
        trajectory=trajectory,
        terminal=terminal,
        decision_round_count=decision_round_count,
        timing=timing,
    )


def _policy_for_player(policies: Mapping[PlayerId, Policy], player_id: PlayerId) -> Policy:
    try:
        return policies[player_id]
    except KeyError as exc:
        raise ValueError(f"no policy configured for requested player {player_id!r}.") from exc


def _select_policy_decision(
    policy: Policy,
    *,
    observation: PokeZeroObservationV0,
    context: PolicyContext,
    rng: random.Random,
) -> PolicyDecision:
    contextual_selector = getattr(policy, "select_action_with_context", None)
    if callable(contextual_selector):
        return contextual_selector(context, rng=rng)
    return policy.select_action(observation, rng=rng)


def _trajectory_snapshot(trajectory: BattleTrajectory) -> BattleTrajectory:
    return BattleTrajectory(
        battle_id=trajectory.battle_id,
        format_id=trajectory.format_id,
        seed=trajectory.seed,
        steps=list(trajectory.steps),
        terminal=trajectory.terminal,
        metadata=dict(trajectory.metadata),
    )


def _reset_unique_policies(policies: Mapping[PlayerId, Policy]) -> None:
    seen: set[int] = set()
    for policy in policies.values():
        policy_id = id(policy)
        if policy_id in seen:
            continue
        seen.add(policy_id)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()


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
