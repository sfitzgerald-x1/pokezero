from dataclasses import replace
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.determinization import BeliefWorldSamplingProfile
from pokezero.env import BattleStartOverride, StepResult
from pokezero.observation import PokeZeroObservationV0
from pokezero.public_decision_corpus import (
    PublicActionIdentifier,
    PublicActorObservation,
    PublicDecisionRecord,
    PublicObservation,
    PublicResolvedActionRound,
    public_decision_id,
)
from pokezero.public_prefix_evaluator import PublicPrefixCandidateValueEvaluator


def _mask(*legal: int) -> tuple[bool, ...]:
    return tuple(index in legal for index in range(ACTION_COUNT))


def _observation(*legal: int, candidates: list[dict]) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(*legal),
        metadata={"action_candidates": candidates},
    )


class SampledPrefixEnv:
    def __init__(self, actor_history: PokeZeroObservationV0, actor_current: PokeZeroObservationV0) -> None:
        self.actor_history = actor_history
        self.actor_current = actor_current
        self.phase = -1
        self.prefix_actions: dict[str, int] | None = None
        self.closed = False

    def reset_with_start_override(self, *, seed: int, format_id: str, start_override) -> None:
        del seed, format_id
        self.assert_sampled_override(start_override)
        self.phase = 0

    def assert_sampled_override(self, start_override) -> None:
        if not isinstance(start_override, BattleStartOverride):
            raise AssertionError("public prefix did not use the sampled world")

    def requested_players(self) -> tuple[str, ...]:
        return ("p1", "p2")

    def terminal(self):
        return None

    def observe(self, player: str) -> PokeZeroObservationV0:
        if self.phase == 0:
            if player == "p1":
                return self.actor_history
            return _observation(
                2,
                candidates=[{"action_index": 2, "kind": "move", "move_id": "tackle", "legal": True}],
            )
        if player == "p1":
            return self.actor_current
        return _observation(
            0,
            candidates=[{"action_index": 0, "kind": "move", "move_id": "growl", "legal": True}],
        )

    def step(self, actions: dict[str, int]) -> StepResult:
        if self.phase == 0:
            if actions != {"p1": 1, "p2": 2}:
                raise AssertionError(f"public identifiers were not resolved in sampled world: {actions!r}")
            self.prefix_actions = dict(actions)
            self.phase = 1
            return StepResult(observations={"p1": self.actor_current}, rewards={"p1": 0.0, "p2": 0.0}, terminal=None, requested_players=("p1", "p2"))
        if actions["p2"] != 0:
            raise AssertionError("hidden scenario action should remain in-memory only")
        return StepResult(
            observations={"p1": _observation(0, candidates=[{"action_index": 0, "kind": "move", "move_id": "tackle", "legal": True}])},
            rewards={"p1": 0.0, "p2": 0.0},
            terminal=None,
            requested_players=("p1", "p2"),
        )

    def close(self) -> None:
        self.closed = True


class PublicPrefixCandidateValueEvaluatorTest(unittest.TestCase):
    def test_replays_public_identifiers_only_after_sampling_a_world(self) -> None:
        historical = _observation(
            1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
        )
        current = _observation(
            0,
            1,
            candidates=[
                {"action_index": 0, "kind": "move", "move_id": "tackle", "legal": True},
                {"action_index": 1, "kind": "move", "move_id": "spikes", "legal": True},
            ],
        )
        prototype = PublicDecisionRecord(
            decision_id="pending",
            battle_id="sampled-prefix",
            seed=7,
            format_id="gen3randombattle",
            acting_player="p1",
            turn_index=1,
            recorded_action_index=0,
            observation=PublicObservation.from_observation(current),
            history=(PublicActorObservation(turn_index=0, observation=PublicObservation.from_observation(historical)),),
            current_legal_action_mask=_mask(0, 1),
            public_resolved_action_rounds=(
                PublicResolvedActionRound(
                    turn_index=0,
                    actions={
                        "p1": PublicActionIdentifier(kind="move", move_id="tackle"),
                        "p2": PublicActionIdentifier(kind="move", move_id="tackle"),
                    },
                ),
            ),
            public_belief_view={"self_slot": "p1", "opponent_slot": "p2", "self_pokemon": [], "opponent_pokemon": []},
        )
        record = replace(prototype, decision_id=public_decision_id(prototype))
        env = SampledPrefixEnv(historical, current)
        profile = BeliefWorldSamplingProfile(
            sample_cap=1,
            sample_count=1,
            combination_count=1,
            uncertainty_bits=0.0,
            uncertain_slot_count=0,
            public_checksum="sampled-prefix",
        )
        evaluator = PublicPrefixCandidateValueEvaluator(
            env_factory=lambda: env,
            value_evaluator=lambda _history: 0.25,
            opponent_prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            set_source=object(),
            world_sample_cap=1,
        )

        with (
            patch("pokezero.public_prefix_evaluator.public_belief_sampling_profile", return_value=profile),
            patch(
                "pokezero.public_prefix_evaluator.gen3_randbat_belief_start_override",
                return_value=BattleStartOverride(player_teams={"p1": "sampled-p1", "p2": "sampled-p2"}),
            ),
        ):
            contexts = evaluator(record)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].candidate_values, {0: 0.25, 1: 0.25})
        self.assertEqual(env.prefix_actions, {"p1": 1, "p2": 2})
        self.assertTrue(env.closed)
        self.assertNotIn(":0", contexts[0].scenario_label)


if __name__ == "__main__":
    unittest.main()
