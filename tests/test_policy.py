import random
import unittest

from pokezero.policy import (
    Policy,
    PolicyDecision,
    RandomLegalPolicy,
    ScriptedTeacherPolicy,
    SimpleLegalPolicy,
    legal_action_indices,
    legal_move_action_indices,
    legal_switch_action_indices,
)
from pokezero.dex import showdown_dex_from_payload
from pokezero.observation import ObservationSpec, PokeZeroObservationV0


def observation(mask: tuple[bool, ...], *, metadata=None) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
        metadata=metadata or {},
    )


class PolicyBaselineTest(unittest.TestCase):
    def test_legal_action_helpers_partition_mask(self) -> None:
        mask = (True, False, True, False, True, False, False, True, False)

        self.assertEqual(legal_action_indices(mask), (0, 2, 4, 7))
        self.assertEqual(legal_move_action_indices(mask), (0, 2))
        self.assertEqual(legal_switch_action_indices(mask), (4, 7))

    def test_legal_action_helpers_reject_empty_or_bad_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            legal_action_indices((False,) * 9)
        with self.assertRaisesRegex(ValueError, "9 values"):
            legal_action_indices((True,))

    def test_random_legal_policy_selects_only_legal_actions(self) -> None:
        policy = RandomLegalPolicy()
        obs = observation((False, True, False, False, True, False, False, False, False))

        decisions = {
            policy.select_action(obs, rng=random.Random(seed)).action_index
            for seed in range(20)
        }

        self.assertTrue(decisions.issubset({1, 4}))
        self.assertIsInstance(policy, Policy)

    def test_policies_require_explicit_rng_for_reproducibility(self) -> None:
        obs = observation((True, False, False, False, False, False, False, False, False))

        with self.assertRaises(TypeError):
            RandomLegalPolicy().select_action(obs)
        with self.assertRaises(TypeError):
            SimpleLegalPolicy().select_action(obs)

    def test_simple_legal_policy_can_force_switch_participation(self) -> None:
        policy = SimpleLegalPolicy(switch_probability=1.0)
        obs = observation((True, False, False, False, False, True, False, False, False))

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 5)
        self.assertEqual(decision.metadata["action_family"], "switch")
        self.assertEqual(decision.action_probability, 1.0)

    def test_simple_legal_policy_falls_back_to_moves_when_no_switch_is_legal(self) -> None:
        policy = SimpleLegalPolicy(switch_probability=1.0)
        obs = observation((False, False, True, False, False, False, False, False, False))

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 2)
        self.assertEqual(decision.metadata["action_family"], "move")
        self.assertEqual(decision.action_probability, 1.0)

    def test_simple_legal_policy_records_marginal_action_probability(self) -> None:
        policy = SimpleLegalPolicy(switch_probability=0.25)
        obs = observation((True, False, True, False, True, True, False, True, False))
        probabilities_by_action = {}

        for seed in range(2000):
            decision = policy.select_action(obs, rng=random.Random(seed))
            probabilities_by_action[decision.action_index] = decision.action_probability

        self.assertEqual(set(probabilities_by_action), {0, 2, 4, 5, 7})
        self.assertAlmostEqual(probabilities_by_action[0], 0.375)
        self.assertAlmostEqual(probabilities_by_action[2], 0.375)
        self.assertAlmostEqual(probabilities_by_action[4], 0.25 / 3)
        self.assertAlmostEqual(sum(probabilities_by_action.values()), 1.0)

    def test_policy_decision_validates_action_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            PolicyDecision(action_index=0, policy_id="bad", action_probability=1.5)

    def test_scripted_teacher_prefers_highest_scoring_gen3_attack(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "flamethrower", "move_name": "Flamethrower"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "shadowball", "move_name": "Shadow Ball"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["policy_family"], "scripted-teacher")
        self.assertIn("eff=2", decision.metadata["teacher_reason"])

    def test_scripted_teacher_uses_switch_when_no_move_is_legal(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (False, False, False, False, True, True, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 0.5, "status": "none"},
                "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    },
                    {
                        "action_index": 5,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Snorlax", "hp_fraction": 0.2, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertEqual(decision.metadata["action_family"], "switch")

    def test_scripted_teacher_fails_loudly_without_dex_by_default(self) -> None:
        policy = ScriptedTeacherPolicy()
        obs = observation((True, False, False, False, False, False, False, False, False))

        with self.assertRaisesRegex(ValueError, "dex unavailable"):
            policy.select_action(obs, rng=random.Random(1))

    def test_scripted_teacher_fails_loudly_on_unknown_legal_move_by_default(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "mysterymove", "move_name": "Mystery Move"},
                ],
            },
        )

        with self.assertRaisesRegex(ValueError, "Mystery Move"):
            policy.select_action(obs, rng=random.Random(1))

    def test_scripted_teacher_keeps_good_attack_over_neutral_switch(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, True, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "bodyslam", "move_name": "Body Slam"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)

    def test_scripted_teacher_switches_when_switch_score_clears_margin(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, True, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)


def teacher_dex():
    return showdown_dex_from_payload(
        {
            "moves": {
                "flamethrower": {
                    "id": "flamethrower",
                    "name": "Flamethrower",
                    "type": "Fire",
                    "category": "Special",
                    "basePower": 90,
                    "accuracy": 100,
                    "priority": 0,
                },
                "shadowball": {
                    "id": "shadowball",
                    "name": "Shadow Ball",
                    "type": "Ghost",
                    "category": "Special",
                    "basePower": 80,
                    "accuracy": 100,
                    "priority": 0,
                },
                "bodyslam": {
                    "id": "bodyslam",
                    "name": "Body Slam",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 85,
                    "accuracy": 100,
                    "priority": 0,
                },
                "tackle": {
                    "id": "tackle",
                    "name": "Tackle",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 30,
                    "accuracy": 100,
                    "priority": 0,
                },
            },
            "species": {
                "charizard": {"id": "charizard", "name": "Charizard", "types": ["Fire", "Flying"], "baseStats": {}},
                "xatu": {"id": "xatu", "name": "Xatu", "types": ["Psychic", "Flying"], "baseStats": {}},
                "golem": {"id": "golem", "name": "Golem", "types": ["Rock", "Ground"], "baseStats": {}},
                "starmie": {"id": "starmie", "name": "Starmie", "types": ["Water", "Psychic"], "baseStats": {}},
                "snorlax": {"id": "snorlax", "name": "Snorlax", "types": ["Normal"], "baseStats": {}},
            },
            "typeChart": {
                "flying": {"Rock": 1, "Ground": 3},
                "fire": {"Rock": 1, "Ground": 1, "Fire": 2, "Water": 1},
                "psychic": {"Ghost": 1},
                "rock": {"Fire": 2, "Water": 1},
                "ground": {"Water": 1},
                "water": {"Fire": 2, "Rock": 2, "Ground": 2},
                "normal": {"Ghost": 3},
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
