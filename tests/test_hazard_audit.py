import json
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.determinization import BeliefWorldSamplingProfile
from pokezero.env import BattleStartOverride, StepResult
from pokezero.hazard_audit import (
    AuditConfig,
    AuditWorld,
    HazardAuditDecision,
    PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
    PublicActorObservation,
    PublicActionRound,
    PublicBeliefWorldProvider,
    aggregate_hazard_audit_records,
    hazard_audit_decisions_from_trajectory,
    hazard_audit_decisions_from_public_corpus,
    run_hazard_blind_spot_audit,
)
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _mask(*legal_indices: int) -> tuple[bool, ...]:
    return tuple(index in legal_indices for index in range(ACTION_COUNT))


def _observation(*legal_indices: int, metadata=None) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(*legal_indices),
        metadata=metadata or {},
    )


class AuditEnv:
    def __init__(self) -> None:
        self._terminal = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        del seed, format_id
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0, 1) if player == "p1" else _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        if set(actions) != {"p1", "p2"}:
            raise ValueError("both players must act")
        action = int(actions["p1"])
        return StepResult(
            observations={"p1": _observation(0, 1, metadata={"branch_value": float(action)})},
            rewards={"p1": 0.0, "p2": 0.0},
            terminal=None,
            requested_players=("p1", "p2"),
        )

    def terminal(self):
        return self._terminal

    def close(self) -> None:
        self.closed = True


class MaskDriftAuditEnv(AuditEnv):
    """Advertises two actor actions but rejects one during replay branching."""

    def step(self, actions: dict[str, int]) -> StepResult:
        if int(actions["p1"]) == 1:
            raise ValueError("p1: action_index 1 is not legal for the current request.")
        return super().step(actions)


class SampledWorldPolicy:
    policy_id = "sampled-world-fixed"

    def __init__(self) -> None:
        self.observations: list[PokeZeroObservationV0] = []

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        del rng
        self.observations.append(observation)
        return PolicyDecision(action_index=3, policy_id=self.policy_id, action_probability=1.0)


class SampledWorldReplayEnv:
    def __init__(self, actor_observation: PokeZeroObservationV0) -> None:
        self.actor_observation = actor_observation
        self.sampled_opponent_observation = _observation(3, metadata={"sampled_world": True})
        self.start_override: BattleStartOverride | None = None
        self.p2_observe_calls = 0
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        raise AssertionError("provider must replay with a sampled start override")

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: str,
        start_override: BattleStartOverride,
    ) -> None:
        del seed, format_id
        if start_override.player_teams["p2"] != "sampled-p2-team":
            raise AssertionError("provider did not use the determinized sampled world")
        self.start_override = start_override

    def observe(self, player: str) -> PokeZeroObservationV0:
        if self.start_override is None:
            raise AssertionError("sampled world must be materialized before observation")
        if player == "p1":
            return self.actor_observation
        if player == "p2":
            self.p2_observe_calls += 1
            return self.sampled_opponent_observation
        raise AssertionError(f"unexpected player {player}")

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        raise AssertionError("provider must use the sampled observation legal mask, not a request mask")

    def requested_players(self) -> tuple[str, ...]:
        return ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        raise AssertionError("provider only replays the prefix before planning opponent actions")

    def terminal(self):
        return None

    @property
    def true_opponent_request(self):
        raise AssertionError("true opponent request must never be accessed")

    def close(self) -> None:
        self.closed = True


def _decision() -> HazardAuditDecision:
    observation = _observation(
        0,
        1,
        metadata={
            "action_candidates": [
                {"action_index": 0, "move_id": "tackle", "legal": True},
                {"action_index": 1, "move_id": "spikes", "legal": True},
            ]
        },
    )
    return HazardAuditDecision(
        battle_id="audit-battle",
        format_id="gen3randombattle",
        seed=7,
        driver_id="fixed-driver",
        player_id="p1",
        decision_round=0,
        target_action_index=1,
        target_move_id="spikes",
        observation_history=(PublicActorObservation(turn_index=0, observation=observation),),
        public_action_rounds=(),
    )


class HazardAuditTest(unittest.TestCase):
    def test_corpus_state_is_public_only_and_replay_uses_placeholder_opponent_steps(self) -> None:
        p1_first = _observation(0, 1)
        p2_private = _observation(0, metadata={"request": {"private": "do-not-retain"}})
        p1_target = _observation(
            0,
            1,
            metadata={"action_candidates": [{"action_index": 1, "move_id": "Rapid Spin", "legal": True}]},
        )
        trajectory = BattleTrajectory(battle_id="source", format_id="gen3randombattle", seed=3)
        trajectory.append(TrajectoryStep("p1", 0, p1_first, p1_first.legal_action_mask, 0))
        trajectory.append(TrajectoryStep("p2", 0, p2_private, p2_private.legal_action_mask, 0))
        trajectory.append(TrajectoryStep("p1", 1, p1_target, p1_target.legal_action_mask, 0))

        decisions = hazard_audit_decisions_from_trajectory(trajectory, driver_id="fixed-driver")

        self.assertEqual(len(decisions), 1)
        payload = decisions[0].to_dict()
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("do-not-retain", serialized)
        self.assertNotIn('"request"', serialized)
        self.assertNotIn("requested_legal_action_masks", serialized)
        opponent_step = next(step for step in payload["replay_trajectory"]["steps"] if step["player_id"] == "p2")
        self.assertEqual(opponent_step["legal_action_mask"], [True] * ACTION_COUNT)
        self.assertTrue(opponent_step["metadata"]["hazard_audit_public_replay_placeholder"])
        with self.assertRaisesRegex(ValueError, "forbidden private key"):
            AuditWorld("bad", {}, metadata={"requested_legal_action_masks": {"p2": [True]}})
        with self.assertRaisesRegex(ValueError, "forbidden private key"):
            AuditWorld("bad-actions", {}, metadata={"opponent_actions": {"p2": 0}})
        with self.assertRaisesRegex(ValueError, "forbidden private key"):
            AuditWorld("bad-override", {}, metadata={"start_override": "private"})
        with self.assertRaisesRegex(ValueError, "forbidden private key"):
            AuditWorld("bad-observation", {}, metadata={"true_opponent_observation": "private"})

    def test_aggregate_defines_entrenchment_as_no_revisits_not_never_visited(self) -> None:
        records = []
        for state_id, low_prior, revisits in (("low-entrenched", True, 0), ("low-rescued", True, 2), ("high", False, 3)):
            for budget in (0, 24, 120):
                records.extend(
                    (
                        {
                            "state_id": state_id,
                            "world_id": "w0",
                            "arm": "deterministic",
                            "extra_visits": budget,
                            "status": "searched",
                            "low_prior": low_prior,
                            "target_visits": 1 + revisits,
                            "target_revisits": revisits,
                            "target_selected": state_id == "low-rescued",
                        },
                        {
                            "state_id": state_id,
                            "world_id": "w0",
                            "arm": "dirichlet_audit_only",
                            "extra_visits": budget,
                            "status": "searched",
                            "low_prior": low_prior,
                            "target_visits": 1 + revisits,
                            "target_revisits": revisits,
                            "target_selected": True,
                        },
                    )
                )

        aggregate = aggregate_hazard_audit_records(records)

        self.assertEqual(aggregate["E"]["low_prior_lines"], 2)
        self.assertEqual(aggregate["E"]["legal_target_lines"], 3)
        self.assertEqual(aggregate["R_off"]["24"]["rescued_low_prior_lines"], 1)
        self.assertEqual(aggregate["R_off"]["24"]["eligible_low_prior_lines"], 2)
        self.assertEqual(aggregate["R_off"]["24"]["rate"], 0.5)
        self.assertIn("target_revisits == 0", aggregate["definitions"]["entrenchment"])
        delta = aggregate["DeltaChoice_on"]["0"]
        self.assertEqual(delta["paired_low_prior_target_lines"], 2)
        self.assertEqual(delta["toward_low_prior_target_lines"], 1)
        self.assertEqual(delta["away_from_low_prior_target_lines"], 0)
        self.assertEqual(delta["interpretation"], "noise_only_choice_sensitivity")
        self.assertEqual(delta["delta_choice_on"], 0.5)

    def test_audit_consumes_the_generic_public_decision_corpus_schema(self) -> None:
        source = _decision()
        corpus = {
            "schema_version": PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
            "records": [source.to_public_decision_record()],
        }

        decisions = hazard_audit_decisions_from_public_corpus(corpus)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].target_move_id, "spikes")
        self.assertEqual(decisions[0].target_action_index, 1)
        self.assertEqual(decisions[0].observation.legal_action_mask, _mask(0, 1))
        self.assertEqual(decisions[0].public_action_rounds, ())
        with self.assertRaisesRegex(ValueError, "requires 'pokezero.public-decision-corpus.v1'"):
            hazard_audit_decisions_from_public_corpus({"schema_version": "wrong", "records": []})

    def test_public_belief_provider_replays_sampled_world_and_uses_its_own_opponent_observation(self) -> None:
        decision = _decision()
        env = SampledWorldReplayEnv(decision.observation)
        policy = SampledWorldPolicy()
        start_override = BattleStartOverride(
            player_teams={"p1": "sampled-p1-team", "p2": "sampled-p2-team"},
        )
        profile = BeliefWorldSamplingProfile(
            sample_cap=1,
            sample_count=1,
            combination_count=1,
            uncertainty_bits=0.0,
            uncertain_slot_count=0,
            public_checksum="public-world",
        )

        def planner_factory(*args, **kwargs):
            del args, kwargs

            def planner(context, scenario, scenario_index, rng):
                del scenario, scenario_index, rng
                self.assertEqual(context.player_id, "p1")
                self.assertEqual(context.requested_observations, {})
                self.assertEqual(context.requested_legal_action_masks, {})
                return lambda: start_override

            return planner

        with (
            patch("pokezero.hazard_audit.belief_world_sampling_profile", return_value=profile),
            patch("pokezero.hazard_audit.gen3_randbat_belief_start_override_planner", side_effect=planner_factory),
        ):
            worlds = PublicBeliefWorldProvider(
                env_factory=lambda: env,
                set_source=object(),
                sampled_world_opponent_policy=policy,
            )(decision)

        self.assertEqual(len(worlds), 1)
        world = worlds[0]
        self.assertTrue(world.available)
        self.assertIs(world.start_override, start_override)
        self.assertEqual(world.opponent_actions, {"p2": 3})
        self.assertEqual(policy.observations, [env.sampled_opponent_observation])
        self.assertEqual(env.p2_observe_calls, 1)
        self.assertIs(env.start_override, start_override)
        self.assertTrue(env.closed)
        self.assertEqual(world.metadata["sampled_world_opponent_policy"], policy.policy_id)
        serialized = json.dumps(world.to_dict(), sort_keys=True)
        self.assertNotIn("opponent_actions", serialized)
        self.assertNotIn("start_override", serialized)

    def test_mandatory_sweep_visit_mismatch_invalidates_records(self) -> None:
        decision = _decision()

        payload = run_hazard_blind_spot_audit(
            decisions=(decision,),
            env_factory=MaskDriftAuditEnv,
            action_priors=lambda history: (0.9, 0.1) + (0.0,) * 7,
            value_fn=lambda history: float(history[-1].metadata["branch_value"]),
            world_provider=lambda state: (AuditWorld(f"{state.state_id}-w0", {"p2": 0}),),
            provenance={"fixture": "mandatory-sweep-mismatch"},
        )

        invalid = [record for record in payload["records"] if record["status"] == "search_invalid"]
        self.assertEqual(len(invalid), 6)
        self.assertTrue(all(record["mandatory_sweep_candidate_count"] == 1 for record in invalid))
        self.assertTrue(
            all(record["total_visits"] != record["expected_total_visits"] for record in invalid)
        )

    def test_records_are_deterministic_for_fixed_public_state_world_and_puct(self) -> None:
        decision = _decision()
        config = AuditConfig(low_prior_threshold=0.02, dirichlet_seed=123)

        def priors(history):
            del history
            return (0.99, 0.01) + (0.0,) * 7

        def value(history):
            return float(history[-1].metadata["branch_value"])

        provider = lambda state: (AuditWorld(f"{state.state_id}-w0", {"p2": 0}),)
        kwargs = {
            "decisions": (decision,),
            "env_factory": AuditEnv,
            "action_priors": priors,
            "value_fn": value,
            "world_provider": provider,
            "config": config,
            "provenance": {"fixture": "deterministic"},
        }

        first = run_hazard_blind_spot_audit(**kwargs)
        second = run_hazard_blind_spot_audit(**kwargs)

        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["hashes"], second["hashes"])
        target_rows = [row for row in first["records"] if row["target_action_index"] == 1]
        self.assertTrue(all(row["target_visits"] >= 1 for row in target_rows if row["status"] == "searched"))
        self.assertTrue(all(row["dirichlet_audit_only"] == (row["arm"] == "dirichlet_audit_only") for row in target_rows))
        searched = [row for row in target_rows if row["status"] == "searched"]
        self.assertTrue(all(row["mandatory_sweep_candidate_count"] == 2 for row in searched))
        self.assertTrue(
            all(row["total_visits"] == row["mandatory_sweep_candidate_count"] + row["extra_visits"] for row in searched)
        )
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn('"opponent_actions"', serialized)
        self.assertNotIn('"start_override"', serialized)
        self.assertNotIn('"true_opponent_request"', serialized)


if __name__ == "__main__":
    unittest.main()
