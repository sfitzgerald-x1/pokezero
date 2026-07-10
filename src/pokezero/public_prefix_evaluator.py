"""Hidden-mode initial candidate sweeps over public replay prefixes.

This adapter is intentionally separate from :mod:`prior_belief_profile`: the
profiler is pure, while this adapter materializes public-belief worlds in a
local simulator. It never accepts an opponent observation, request payload, or
opponent legal-action mask.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .determinization import gen3_randbat_belief_start_override
from .replay_branching import replay_action_rounds
from .prior_belief_profile import (
    WorldScenarioEvaluation,
    public_belief_sampling_profile,
    public_policy_context,
)
from .public_decision_corpus import (
    PublicActionIdentifier,
    PublicDecisionRecord,
    PublicResolvedActionRound,
)


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

    def __call__(self, record: PublicDecisionRecord) -> tuple[WorldScenarioEvaluation, ...]:
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
        for world_index in range(belief.sample_count):
            rng = random.Random(f"{record.decision_id}:public-belief-world:{world_index}")
            override = gen3_randbat_belief_start_override(
                context=context,
                set_source=self._set_source,
                rng=rng,
            )
            if override is None:
                continue
            for scenario_index, (opponent_action, scenario_weight, scenario_label) in enumerate(scenarios):
                candidate_values = self._candidate_values(
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
                        )
                    )
        return tuple(rows)

    def _candidate_values(
        self,
        record: PublicDecisionRecord,
        *,
        public_action_rounds: tuple[PublicResolvedActionRound, ...],
        start_override: Any,
        opponent_action: int,
    ) -> dict[int, float]:
        opponent_player = "p2" if record.acting_player == "p1" else "p1"
        values: dict[int, float] = {}
        for action_index, legal in enumerate(record.current_legal_action_mask):
            if not legal:
                continue
            env = self._env_factory()
            try:
                prefix = _replay_public_action_rounds(
                    env,
                    seed=record.seed,
                    format_id=record.format_id,
                    public_action_rounds=public_action_rounds,
                    start_override=start_override,
                )
                if prefix.terminal is not None:
                    continue
                # This map is deliberately constructed without inspecting what the
                # sampled opponent currently considers legal. env.step validates the
                # prediction; failed worlds are simply not selection contexts.
                branch_actions = {record.acting_player: action_index, opponent_player: opponent_action}
                if set(prefix.requested_players) != set(branch_actions):
                    continue
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
            except (RuntimeError, ValueError):
                # A public sampled world can fail to reproduce a replay prefix or
                # make a hidden predicted action illegal. Neither case authorizes
                # a private-mask fallback.
                continue
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        return values


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


def _replay_public_action_rounds(
    env: Any,
    *,
    seed: int,
    format_id: str,
    public_action_rounds: tuple[PublicResolvedActionRound, ...],
    start_override: Any,
) -> Any:
    """Replay a public prefix by resolving identifiers inside this sampled world."""

    replay_action_rounds(
        env,
        seed=seed,
        format_id=format_id,
        action_rounds=(),
        start_override=start_override,
        check_prefix_observations=False,
    )
    for expected_turn, action_round in enumerate(public_action_rounds):
        if action_round.turn_index != expected_turn:
            raise ValueError("public action rounds must be contiguous from turn zero.")
        requested_players = tuple(env.requested_players())
        if set(requested_players) != set(action_round.actions):
            raise ValueError("sampled world request shape does not match the public action round.")
        actions = {
            player: _resolve_public_action_identifier(
                env.observe(player),
                identifier,
            )
            for player, identifier in action_round.actions.items()
        }
        env.step(actions)
    return type(
        "PublicReplayPrefix",
        (),
        {"terminal": env.terminal(), "requested_players": tuple(env.requested_players())},
    )()


def _resolve_public_action_identifier(
    observation: Any,
    identifier: PublicActionIdentifier,
) -> int:
    """Resolve a public move/species ID against the sampled world's legal request."""

    candidates = observation.metadata.get("action_candidates")
    if not isinstance(candidates, Sequence):
        raise ValueError("sampled world observation has no action candidates.")
    matches: list[int] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        action_index = candidate.get("action_index")
        if not isinstance(action_index, int) or not observation.legal_action_mask[action_index]:
            continue
        if identifier.kind == "move" and candidate.get("kind") == "move":
            if str(candidate.get("move_id") or "").lower() == str(identifier.move_id).lower():
                matches.append(action_index)
        elif identifier.kind == "switch" and candidate.get("kind") == "switch":
            species = candidate.get("switched_species")
            if species is None and isinstance(candidate.get("pokemon"), dict):
                species = candidate["pokemon"].get("species")
            if str(species or "").lower() == str(identifier.switched_species).lower():
                matches.append(action_index)
    if identifier.kind == "event":
        raise ValueError("public event identifier cannot be replayed as a sampled-world action.")
    if not matches:
        raise ValueError("public action identifier is unavailable in the sampled world.")
    return min(matches)
