"""Policy adapters backed by replay-from-root search."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import math
import random
from typing import Callable, Mapping

from .actions import ACTION_COUNT
from .env import PlayerId, PokeZeroEnv
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision, RandomLegalPolicy, legal_action_indices
from .rollout import RolloutConfig
from .search import ActionPriorVector, ObservationValueFunction, player_observation_history, puct_branch_search
from .trajectory import BattleTrajectory, TrajectoryStep

OpponentActionPlanner = Callable[[PolicyContext, random.Random], Mapping[PlayerId, int]]
ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
OpponentActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]


def no_opponent_action_planner(context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
    del context, rng
    return {}


def greedy_opponent_action_planner(
    prior_fn: OpponentActionPriorFunction,
) -> OpponentActionPlanner:
    """Create a planner that predicts each requested opponent action from player-local history.

    The prior function sees only the acting player's observation history. It should model the
    opponent-action auxiliary head, not the acting player's legal-action policy head. The selected
    action is not legality-masked because opponent legal choices are not player-private data in the
    current observation; future belief/request plumbing can replace this planner with a constrained
    one.
    """

    def planner(context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
        del rng
        requested_opponents = tuple(player for player in context.requested_players if player != context.player_id)
        if not requested_opponents:
            return {}
        trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        priors = tuple(float(value) for value in prior_fn(history))
        _validate_action_prior_vector(priors, name="opponent action priors")
        action_index = max(range(ACTION_COUNT), key=lambda index: (priors[index], -index))
        return {player: action_index for player in requested_opponents}

    return planner


@dataclass
class RootPUCTSearchPolicy:
    """Context-aware policy adapter that selects actions with root-level PUCT.

    The policy intentionally receives only the acting player's observation through
    ``PolicyContext``. Simultaneous-turn opponent actions must come from the explicit
    ``opponent_action_planner`` hook, which keeps hidden-information assumptions auditable.
    Branch search runs in a separate env from ``env_factory`` so it cannot mutate the live rollout.
    """

    env_factory: Callable[[], PokeZeroEnv]
    rollout_config: RolloutConfig
    value_fn: ObservationValueFunction
    prior_fn: ActionPriorFunction
    policy_id: str = "root-puct-search"
    cpuct: float = 1.25
    opponent_action_planner: OpponentActionPlanner = no_opponent_action_planner
    fallback_policy: Policy = field(default_factory=RandomLegalPolicy)
    allow_fallback: bool = False

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        decision = self.fallback_policy.select_action(observation, rng=rng)
        return PolicyDecision(
            action_index=decision.action_index,
            policy_id=self.policy_id,
            action_probability=decision.action_probability,
            metadata={
                **dict(decision.metadata),
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_reason": "missing policy context",
                "fallback_policy_id": decision.policy_id,
            },
        )

    def select_action_with_context(
        self,
        context: PolicyContext,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        if context.player_id not in context.requested_players:
            return self._fallback(context, rng=rng, reason="player is not requested")
        opponent_actions = dict(self.opponent_action_planner(context, rng))
        planner_error = _opponent_action_planner_error(
            player_id=context.player_id,
            requested_players=context.requested_players,
            opponent_actions=opponent_actions,
        )
        if planner_error is not None:
            return self._fallback(context, rng=rng, reason=planner_error)

        search_trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            search_trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        priors = self.prior_fn(history)
        env = self.env_factory()
        try:
            start = perf_counter()
            search = puct_branch_search(
                env=env,
                trajectory=search_trajectory,
                player_id=context.player_id,
                prefix_decision_round_count=context.decision_round_index,
                legal_action_mask=context.observation.legal_action_mask,
                opponent_actions=opponent_actions,
                value_fn=self.value_fn,
                action_priors=priors,
                cpuct=self.cpuct,
            )
            elapsed_seconds = perf_counter() - start
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        best = search.best_candidate
        return PolicyDecision(
            action_index=search.action_index,
            policy_id=self.policy_id,
            action_probability=None,
            metadata={
                "policy_family": "root-puct-search",
                "root_puct_fallback": False,
                "root_puct_cpuct": self.cpuct,
                "root_puct_selected_value": best.value,
                "root_puct_selected_score": best.score,
                "root_puct_candidate_count": len(search.candidates),
                "root_puct_elapsed_seconds": elapsed_seconds,
                "root_puct_opponent_actions": dict(opponent_actions),
            },
        )

    def _fallback(self, context: PolicyContext, *, rng: random.Random, reason: str) -> PolicyDecision:
        if not self.allow_fallback:
            raise ValueError(f"root PUCT search cannot select an action: {reason}")
        decision = self.fallback_policy.select_action(context.observation, rng=rng)
        return PolicyDecision(
            action_index=decision.action_index,
            policy_id=self.policy_id,
            action_probability=decision.action_probability,
            metadata={
                **dict(decision.metadata),
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_reason": reason,
                "fallback_policy_id": decision.policy_id,
            },
        )


def _opponent_action_planner_error(
    *,
    player_id: PlayerId,
    requested_players: tuple[PlayerId, ...],
    opponent_actions: Mapping[PlayerId, int],
) -> str | None:
    if player_id in opponent_actions:
        return "opponent_action_planner returned the acting player's action"
    requested_opponents = set(requested_players) - {player_id}
    action_players = set(opponent_actions)
    missing = sorted(requested_opponents - action_players)
    extra = sorted(action_players - requested_opponents)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing opponent actions for {', '.join(missing)}")
        if extra:
            details.append(f"unexpected opponent actions for {', '.join(extra)}")
        return "; ".join(details)
    return None


def _validate_action_prior_vector(priors: tuple[float, ...], *, name: str) -> None:
    if len(priors) != ACTION_COUNT:
        raise ValueError(f"{name} must contain {ACTION_COUNT} values.")
    if any(value < 0.0 or not math.isfinite(value) for value in priors):
        raise ValueError(f"{name} must contain finite non-negative values.")


def _trajectory_with_current_observation(context: PolicyContext) -> BattleTrajectory:
    trajectory = BattleTrajectory(
        battle_id=context.trajectory.battle_id,
        format_id=context.trajectory.format_id,
        seed=context.trajectory.seed,
        steps=list(context.trajectory.steps),
        terminal=context.trajectory.terminal,
        metadata=dict(context.trajectory.metadata),
    )
    if any(
        step.player_id == context.player_id and step.turn_index == context.decision_round_index
        for step in trajectory.steps
    ):
        return trajectory
    legal = legal_action_indices(context.observation.legal_action_mask)
    trajectory.append(
        TrajectoryStep(
            player_id=context.player_id,
            turn_index=context.decision_round_index,
            observation=context.observation,
            legal_action_mask=tuple(context.observation.legal_action_mask),
            action_index=legal[0],
            metadata={"synthetic_current_observation": True},
        )
    )
    return trajectory
