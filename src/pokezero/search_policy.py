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
from .search import (
    ActionPriorVector,
    ObservationValueFunction,
    PUCTBranchSearchCandidate,
    player_observation_history,
    puct_branch_search,
)
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
    opponent-action auxiliary head, not the acting player's legal-action policy head. When the
    rollout harness has already observed requested-player legal masks, this planner uses those
    masks only to keep the planned opponent action submit-valid for replay search.
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
        opponent_actions = {}
        for player in requested_opponents:
            legal = _requested_legal_action_indices_for_player(context, player)
            if legal:
                opponent_actions[player] = max(legal, key=lambda index: (priors[index], -index))
            else:
                opponent_actions[player] = max(range(ACTION_COUNT), key=lambda index: (priors[index], -index))
        return opponent_actions

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
    minimum_value_improvement: float | None = None
    selection_mode: str = "puct"

    def __post_init__(self) -> None:
        if self.selection_mode not in {"puct", "value"}:
            raise ValueError("selection_mode must be 'puct' or 'value'.")
        if self.minimum_value_improvement is None:
            return
        if self.minimum_value_improvement < 0.0 or not math.isfinite(self.minimum_value_improvement):
            raise ValueError("minimum_value_improvement must be a finite non-negative value when set.")

    def reset(self) -> None:
        reset = getattr(self.fallback_policy, "reset", None)
        if callable(reset):
            reset()

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
        legality_report = _opponent_action_legality_report(context, opponent_actions)
        if legality_report.error is not None:
            return self._fallback(context, rng=rng, reason=legality_report.error)

        search_trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            search_trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        priors = self.prior_fn(history)
        start = perf_counter()
        env = self.env_factory()
        try:
            try:
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
            except Exception as exc:
                return self._fallback(context, rng=rng, reason=f"search failed: {exc}")
            elapsed_seconds = perf_counter() - start
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        search_best = _selected_candidate(search.candidates, mode=self.selection_mode)
        best = search_best
        gate_metadata = {}
        if self.minimum_value_improvement is not None:
            value_gate_used = False
            prior_best = _best_prior_candidate(search.candidates)
            if (
                search_best.action_index != prior_best.action_index
                and search_best.value < prior_best.value + self.minimum_value_improvement
            ):
                best = prior_best
                value_gate_used = True
            gate_metadata = {
                "root_puct_minimum_value_improvement": self.minimum_value_improvement,
                "root_puct_value_gate_used": value_gate_used,
                "root_puct_pre_gate_action": search_best.action_index,
                "root_puct_prior_action": prior_best.action_index,
                "root_puct_prior_value": prior_best.value,
                "root_puct_prior_score": prior_best.score,
            }
        return PolicyDecision(
            action_index=best.action_index,
            policy_id=self.policy_id,
            action_probability=None,
            metadata={
                "policy_family": "root-puct-search",
                "root_puct_fallback": False,
                "root_puct_cpuct": self.cpuct,
                "root_puct_selection_mode": self.selection_mode,
                "root_puct_selected_value": best.value,
                "root_puct_selected_score": best.score,
                "root_puct_candidate_count": len(search.candidates),
                "root_puct_elapsed_seconds": elapsed_seconds,
                "root_puct_opponent_actions": dict(opponent_actions),
                "root_puct_opponent_actions_legality_checked": legality_report.checked,
                **gate_metadata,
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


def _best_prior_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
) -> PUCTBranchSearchCandidate:
    if not candidates:
        raise ValueError("root PUCT search produced no candidates.")
    return max(candidates, key=lambda candidate: (candidate.prior, -candidate.action_index))


def _best_value_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
) -> PUCTBranchSearchCandidate:
    if not candidates:
        raise ValueError("root PUCT search produced no candidates.")
    return max(candidates, key=lambda candidate: (candidate.value, -candidate.action_index))


def _selected_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
    *,
    mode: str,
) -> PUCTBranchSearchCandidate:
    if mode == "puct":
        if not candidates:
            raise ValueError("root PUCT search produced no candidates.")
        return max(candidates, key=lambda candidate: (candidate.score, candidate.value, -candidate.action_index))
    if mode == "value":
        return _best_value_candidate(candidates)
    raise ValueError("selection mode must be 'puct' or 'value'.")


@dataclass(frozen=True)
class _OpponentActionLegalityReport:
    checked: bool
    error: str | None = None


def _opponent_action_legality_report(
    context: PolicyContext,
    opponent_actions: Mapping[PlayerId, int],
) -> _OpponentActionLegalityReport:
    checked = False
    for player, action_index in opponent_actions.items():
        legal = _requested_legal_action_indices_for_player(context, player)
        if not legal:
            continue
        checked = True
        if action_index not in legal:
            return _OpponentActionLegalityReport(
                checked=True,
                error=(
                    "opponent_action_planner returned an illegal action "
                    f"for {player}: {action_index}"
                ),
            )
    return _OpponentActionLegalityReport(checked=checked)


def _requested_legal_action_indices_for_player(
    context: PolicyContext,
    player: PlayerId,
) -> tuple[int, ...]:
    legal_action_mask = context.requested_legal_action_masks.get(player)
    if legal_action_mask is None:
        return ()
    if len(legal_action_mask) != ACTION_COUNT:
        return ()
    return tuple(index for index, legal in enumerate(legal_action_mask) if legal)


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
