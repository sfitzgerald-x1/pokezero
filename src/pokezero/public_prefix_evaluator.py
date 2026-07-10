"""Hidden-mode initial candidate sweeps over public replay prefixes.

This adapter is intentionally separate from :mod:`prior_belief_profile`: the
profiler is pure, while this adapter materializes public-belief worlds in a
local simulator. It never accepts an opponent observation, request payload, or
opponent legal-action mask.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Callable, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .determinization import gen3_randbat_belief_start_override
from .prior_belief_profile import (
    CandidateValueEvaluation,
    WorldScenarioEvaluation,
    public_belief_sampling_profile,
    public_policy_context,
)
from .public_decision_corpus import PublicDecisionRecord, PublicResolvedActionRound
from .public_replay_materializer import PublicReplayError, replay_public_action_rounds


ObservationValueEvaluator = Callable[[tuple[Any, ...]], float]
OpponentPriorEvaluator = Callable[[tuple[Any, ...]], Sequence[float]]


class PublicPrefixCandidateValueEvaluator:
    """Produce mandatory-sweep values from public prefixes and hidden worlds.

    The opponent's current action is predicted solely from the acting player's
    history. The predicted action may be replay-illegal in a sampled world; in
    that case the world/scenario is excluded instead of consulting a private
    legal mask to repair it.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[], Any],
        value_evaluator: ObservationValueEvaluator,
        opponent_prior_evaluator: OpponentPriorEvaluator,
        set_source: Any,
        world_sample_cap: int,
        scenario_count: int = 1,
    ) -> None:
        if world_sample_cap <= 0:
            raise ValueError("world_sample_cap must be positive.")
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive.")
        self._env_factory = env_factory
        self._value_evaluator = value_evaluator
        self._opponent_prior_evaluator = opponent_prior_evaluator
        self._set_source = set_source
        self._world_sample_cap = world_sample_cap
        self._scenario_count = scenario_count

    def __call__(self, record: PublicDecisionRecord) -> CandidateValueEvaluation:
        context = public_policy_context(record)
        belief = public_belief_sampling_profile(
            record,
            sample_cap=self._world_sample_cap,
            set_source=self._set_source,
        )
        scenarios = _hidden_opponent_scenarios(
            self._opponent_prior_evaluator(record.observations()),
            acting_player=record.acting_player,
            scenario_count=self._scenario_count,
        )
        rows: list[WorldScenarioEvaluation] = []
        failure_reasons: Counter[str] = Counter()
        for world_index in range(belief.sample_count):
            rng = random.Random(f"{record.decision_id}:public-belief-world:{world_index}")
            override = gen3_randbat_belief_start_override(
                context=context,
                set_source=self._set_source,
                rng=rng,
            )
            if override is None:
                failure_reasons["missing_public_belief_world"] += 1
                continue
            for scenario_index, (opponent_action, scenario_weight, scenario_label) in enumerate(scenarios):
                candidate_values, canonicalizations, failure_reason = self._candidate_values(
                    record,
                    public_action_rounds=record.public_resolved_action_rounds,
                    start_override=override,
                    opponent_action=opponent_action,
                )
                if candidate_values:
                    rows.append(
                        WorldScenarioEvaluation(
                            world_index=world_index,
                            scenario_index=scenario_index,
                            scenario_label=scenario_label,
                            scenario_weight=scenario_weight,
                            candidate_values=candidate_values,
                            public_event_canonicalizations=tuple(
                                canonicalization.to_dict() for canonicalization in canonicalizations
                            ),
                        )
                    )
                elif failure_reason is not None:
                    failure_reasons[failure_reason] += 1
        if rows:
            return CandidateValueEvaluation(contexts=tuple(rows))
        skip_reason = _primary_failure_reason(failure_reasons)
        return CandidateValueEvaluation(
            contexts=(),
            skip_reason=skip_reason,
            failure_reasons=dict(failure_reasons),
        )

    def _candidate_values(
        self,
        record: PublicDecisionRecord,
        *,
        public_action_rounds: tuple[PublicResolvedActionRound, ...],
        start_override: Any,
        opponent_action: int,
    ) -> tuple[dict[int, float], tuple[Any, ...], str | None]:
        opponent_player = "p2" if record.acting_player == "p1" else "p1"
        values: dict[int, float] = {}
        canonicalizations: tuple[Any, ...] = ()
        for action_index, legal in enumerate(record.current_legal_action_mask):
            if not legal:
                continue
            env = self._env_factory()
            try:
                prefix = replay_public_action_rounds(
                    env,
                    seed=record.seed,
                    format_id=record.format_id,
                    public_action_rounds=public_action_rounds,
                    start_override=start_override,
                )
                canonicalizations = prefix.event_canonicalizations
                if prefix.terminal is not None:
                    return {}, canonicalizations, "public_prefix_terminal"
                # This map is deliberately constructed without inspecting what the
                # sampled opponent currently considers legal. env.step validates the
                # prediction; failed worlds are simply not selection contexts.
                branch_actions = {record.acting_player: action_index, opponent_player: opponent_action}
                if set(prefix.requested_players) != set(branch_actions):
                    return {}, canonicalizations, "sampled_world_branch_request_shape_mismatch"
                result = env.step(branch_actions)
                if result.terminal is not None:
                    value = 1.0 if result.terminal.winner == record.acting_player else -1.0
                    if result.terminal.winner is None:
                        value = 0.0
                else:
                    observation = result.observations.get(record.acting_player) or env.observe(record.acting_player)
                    value = float(self._value_evaluator((*record.observations(), observation)))
                if math.isfinite(value):
                    values[action_index] = value
            except PublicReplayError as exc:
                return {}, canonicalizations, exc.reason
            except (RuntimeError, ValueError):
                # A public sampled world can fail to reproduce a replay prefix or
                # make a hidden predicted action illegal. Neither case authorizes
                # a private-mask fallback.
                continue
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        return values, canonicalizations, None if values else "no_candidate_values"


def _hidden_opponent_scenarios(
    raw_priors: Sequence[float],
    *,
    acting_player: str,
    scenario_count: int,
) -> tuple[tuple[int, float, str], ...]:
    if len(raw_priors) != ACTION_COUNT:
        raise ValueError(f"opponent prior evaluator must return {ACTION_COUNT} values.")
    values = tuple(float(value) for value in raw_priors)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("opponent priors must be finite non-negative values.")
    # With no opponent mask, move slots are distinct while switch slots form one
    # exchangeable information-set bucket. Pick a deterministic representative
    # for replay but preserve the summed bucket mass for ranking/weighting.
    switch_indices = tuple(range(MOVE_ACTION_COUNT, ACTION_COUNT))
    representative_switch = max(switch_indices, key=lambda index: (values[index], -index))
    candidates = [(index, values[index]) for index in range(MOVE_ACTION_COUNT)]
    candidates.append((representative_switch, sum(values[index] for index in switch_indices)))
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))[:scenario_count]
    total = sum(weight for _action, weight in ranked)
    if total <= 0.0:
        total = float(len(ranked))
        ranked = [(action, 1.0) for action, _weight in ranked]
    opponent_player = "p2" if acting_player == "p1" else "p1"
    return tuple(
        # The index is in-memory only. Persisted context labels identify the
        # model-ranked scenario without exposing a request-local slot.
        (action, weight / total, f"hidden-prior:{opponent_player}:rank-{rank}")
        for rank, (action, weight) in enumerate(ranked)
    )



def _primary_failure_reason(failure_reasons: Mapping[str, int]) -> str:
    if not failure_reasons:
        return "no_public_replay_contexts"
    return min(failure_reasons, key=lambda reason: (-failure_reasons[reason], reason))
