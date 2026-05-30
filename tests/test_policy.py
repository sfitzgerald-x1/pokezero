import random
import unittest

from pokezero.policy import (
    Policy,
    PolicyDecision,
    RandomLegalPolicy,
    SimpleLegalPolicy,
    legal_action_indices,
    legal_move_action_indices,
    legal_switch_action_indices,
)
from pokezero.observation import ObservationSpec, PokeZeroObservationV0


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
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

    def test_simple_legal_policy_can_force_switch_participation(self) -> None:
        policy = SimpleLegalPolicy(switch_probability=1.0)
        obs = observation((True, False, False, False, False, True, False, False, False))

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 5)
        self.assertEqual(decision.metadata["action_family"], "switch")

    def test_simple_legal_policy_falls_back_to_moves_when_no_switch_is_legal(self) -> None:
        policy = SimpleLegalPolicy(switch_probability=1.0)
        obs = observation((False, False, True, False, False, False, False, False, False))

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 2)
        self.assertEqual(decision.metadata["action_family"], "move")

    def test_policy_decision_validates_action_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            PolicyDecision(action_index=0, policy_id="bad", action_probability=1.5)


if __name__ == "__main__":
    unittest.main()
