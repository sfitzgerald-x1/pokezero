import json
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult
from pokezero.hazard_audit import (
    AuditConfig,
    AuditWorld,
    HazardAuditDecision,
    PublicActorObservation,
    PublicActionRound,
    aggregate_hazard_audit_records,
    hazard_audit_decisions_from_trajectory,
    run_hazard_blind_spot_audit,
)
from pokezero.observation import PokeZeroObservationV0
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
        self.assertAlmostEqual(aggregate["DeltaChoice_on"]["0"]["delta_choice_on"], 2 / 3)

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


if __name__ == "__main__":
    unittest.main()
