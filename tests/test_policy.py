import random
import unittest

from pokezero.policy import (
    MaxDamagePolicy,
    Policy,
    PolicyDecision,
    RandomLegalPolicy,
    ScriptedTeacherPolicy,
    SimpleLegalPolicy,
    legal_action_indices,
    legal_move_action_indices,
    legal_switch_action_indices,
)
from pokezero.collection import policy_factory_from_spec, policy_spec_with_showdown_root, reject_eval_only_specs
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

    def test_policy_decision_validates_value_estimate(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_estimate"):
            PolicyDecision(action_index=0, policy_id="bad", value_estimate=float("nan"))

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

    def test_scripted_teacher_can_break_score_ties_deterministically(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex(), tie_breaker="first")
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "bodyslam", "move_name": "Body Slam"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "bodyslam", "move_name": "Body Slam"},
                ],
            },
        )

        decisions = {policy.select_action(obs, rng=random.Random(seed)).action_index for seed in range(20)}

        self.assertEqual(decisions, {0})
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertEqual(decision.metadata["teacher_tie_count"], 2)
        self.assertEqual(decision.metadata["teacher_tie_breaker"], "first")

    def test_scripted_teacher_rejects_unknown_tie_breaker(self) -> None:
        with self.assertRaisesRegex(ValueError, "tie_breaker"):
            ScriptedTeacherPolicy(dex=teacher_dex(), tie_breaker="middle")

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

    def test_max_damage_prefers_damaging_move_over_status(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "toxic", "move_name": "Toxic"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "flamethrower", "move_name": "Flamethrower"},
                ],
            },
        )
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertEqual(decision.action_index, 1)  # damaging move, not the status move
        self.assertEqual(decision.metadata["policy_family"], "max-damage")
        self.assertEqual(decision.metadata["branch"], "max_damage_move")

    def test_max_damage_prefers_higher_damage_move(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
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
        # Shadow Ball is super-effective vs Xatu (Psychic) -> higher estimated damage.
        self.assertEqual(decision.action_index, 1)
        self.assertGreater(decision.metadata["damage_estimate"], 0.0)

    def test_max_damage_switches_when_no_move_is_legal(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
        obs = observation(
            (False, False, False, False, True, True, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 0.5, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 4, "kind": "switch", "legal": True, "pokemon": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"}},
                    {"action_index": 5, "kind": "switch", "legal": True, "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"}},
                ],
            },
        )
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertIn(decision.action_index, (4, 5))
        self.assertEqual(decision.metadata["branch"], "forced_switch")

    def test_max_damage_requires_dex(self) -> None:
        policy = MaxDamagePolicy()
        obs = observation((True, False, False, False, False, False, False, False, False))
        with self.assertRaisesRegex(ValueError, "requires a Showdown dex"):
            policy.select_action(obs, rng=random.Random(1))

    def test_max_damage_spec_resolves_and_injects_showdown_root(self) -> None:
        self.assertIsInstance(policy_factory_from_spec("max-damage")(), MaxDamagePolicy)
        with self.assertRaisesRegex(ValueError, "Unsupported max-damage option"):
            policy_factory_from_spec("max-damage?bogus=1")
        rooted = policy_spec_with_showdown_root("max-damage", "/tmp/showdown")
        self.assertIn("showdown_root", rooted)
        policy = policy_factory_from_spec(rooted)()
        self.assertEqual(str(policy.showdown_root), "/tmp/showdown")

    def test_aggressive_damage_spec_is_training_allowed_max_damage_family(self) -> None:
        policy = policy_factory_from_spec("aggressive-damage")()
        self.assertIsInstance(policy, MaxDamagePolicy)
        self.assertEqual(policy.policy_id, "aggressive-damage")

        rooted = policy_spec_with_showdown_root("aggressive-damage", "/tmp/showdown")
        self.assertIn("showdown_root", rooted)
        rooted_policy = policy_factory_from_spec(rooted)()
        self.assertEqual(rooted_policy.policy_id, "aggressive-damage")
        self.assertEqual(str(rooted_policy.showdown_root), "/tmp/showdown")

    def test_scripted_teacher_status_pressure_option(self) -> None:
        policy = policy_factory_from_spec("scripted-teacher?status_pressure_score=75")()
        self.assertIsInstance(policy, ScriptedTeacherPolicy)
        self.assertEqual(policy.status_pressure_score, 75.0)

    def test_max_damage_unknown_move_scores_zero(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "notarealmove", "move_name": "Not Real"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "flamethrower", "move_name": "Flamethrower"},
                ],
            },
        )
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertEqual(decision.action_index, 1)  # unknown move scores 0, damaging move wins

    def test_max_damage_all_status_moves_pick_a_legal_move(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "toxic", "move_name": "Toxic"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "thunderwave", "move_name": "Thunder Wave"},
                ],
            },
        )
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertIn(decision.action_index, (0, 1))
        self.assertEqual(decision.metadata["branch"], "max_damage_move")
        self.assertEqual(decision.metadata["damage_estimate"], 0.0)

    def test_max_damage_falls_back_without_candidate_metadata(self) -> None:
        policy = MaxDamagePolicy(dex=teacher_dex())
        obs = observation((True, False, True, False, False, False, False, False, False))
        decision = policy.select_action(obs, rng=random.Random(1))
        self.assertIn(decision.action_index, (0, 2))
        self.assertEqual(decision.metadata["branch"], "fallback")

    def test_reject_eval_only_specs_blocks_max_damage_for_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            reject_eval_only_specs(["max-damage"], role="self-play training opponent")
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            reject_eval_only_specs(["max-damage?showdown_root=/x"], role="self-play initial policy")
        # Ordinary training opponents/specs pass through untouched (including None).
        reject_eval_only_specs(
            ["random-legal", "simple-legal", "aggressive-damage", "linear:foo.json", None],
            role="opponent",
        )

    def test_scripted_teacher_root_injection_preserves_existing_options(self) -> None:
        rooted = policy_spec_with_showdown_root("scripted-teacher?allow_fallback=true", "/tmp/showdown")
        self.assertIn("showdown_root", rooted)
        self.assertIn("allow_fallback", rooted)

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

    def test_scripted_teacher_accepts_forced_recharge_pseudo_move(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "recharge", "move_name": "Recharge"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["action_family"], "move")
        self.assertIn("forced pseudo-move", decision.metadata["teacher_reason"])

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

    def test_scripted_teacher_scores_switches_against_possible_opponent_moves(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (False, False, False, False, True, True, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 0.2, "status": "none"},
                "opponent_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active_possible_moves": ["Flamethrower"],
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
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertIn("source=opponent_moves", decision.metadata["teacher_reason"])

    def test_scripted_teacher_scores_switches_against_incoming_move_power(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (False, False, False, False, True, True, False, False, False),
            metadata={
                "self_active": {"species": "Skarmory", "hp_fraction": 0.5, "status": "none"},
                "opponent_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active_possible_moves": ["Tackle", "Flamethrower"],
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
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertIn("incoming=67.5", decision.metadata["teacher_reason"])

    def test_scripted_teacher_pivots_from_super_effective_pressure(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, True, False, False, False, False),
            metadata={
                "self_active": {"species": "Skarmory", "hp_fraction": 0.5, "status": "none"},
                "opponent_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "opponent_active_possible_moves": ["Flamethrower"],
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "bodyslam", "move_name": "Body Slam"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertIn("danger=", decision.metadata["teacher_reason"])

    def test_scripted_teacher_values_team_status_cure_when_teammate_is_statused(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "self_team": [
                    {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    {"species": "Starmie", "hp_fraction": 1.0, "status": "par"},
                ],
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "healbell", "move_name": "Heal Bell"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("team status cure", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_score"], 50.0)

    def test_scripted_teacher_marks_type_immune_status_move_as_no_effect(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "thunderwave", "move_name": "Thunder Wave"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["teacher_branch"], "status_no_effect")
        self.assertIn("no effect", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_score"], 4.0)

    def test_scripted_teacher_prefers_damage_over_type_immune_status_pressure(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "thunderwave", "move_name": "Thunder Wave"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["teacher_branch"], "damaging_move")

    def test_scripted_teacher_does_not_treat_glare_into_ghost_as_type_immune(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "glare", "move_name": "Glare"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["teacher_branch"], "status_pressure")
        self.assertEqual(decision.metadata["teacher_score"], 55.0)

    def test_scripted_teacher_marks_toxic_into_steel_as_no_effect(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Skarmory", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "toxic", "move_name": "Toxic"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["teacher_branch"], "status_no_effect")
        self.assertIn("no effect", decision.metadata["teacher_reason"])

    def test_scripted_teacher_marks_burn_into_fire_as_no_effect(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "willowisp", "move_name": "Will-O-Wisp"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["teacher_branch"], "status_no_effect")
        self.assertIn("no effect", decision.metadata["teacher_reason"])

    def test_scripted_teacher_values_rapid_spin_when_own_side_has_hazards(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                "self_side_conditions": ["spikes"],
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "rapidspin", "move_name": "Rapid Spin"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("clears hazards=1", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_branch"], "rapid_spin_clear_hazards")

    def test_scripted_teacher_does_not_value_rapid_spin_without_hazards(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                "self_side_conditions": [],
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "rapidspin", "move_name": "Rapid Spin"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)

    def test_scripted_teacher_uses_rapid_spin_chip_without_hazards(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                "self_side_conditions": [],
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "self_team": [],
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "healbell", "move_name": "Heal Bell"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "rapidspin", "move_name": "Rapid Spin"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("no side hazards", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_score"], 20.0)

    def test_scripted_teacher_avoids_rapid_spin_when_current_opponent_blocks_it(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                "self_side_conditions": ["spikes"],
                "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "flamethrower", "move_name": "Flamethrower"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "rapidspin", "move_name": "Rapid Spin"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)

    def test_scripted_teacher_values_spikes_when_no_layers_are_known(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_side_conditions": [],
                "opponent_side_condition_counts": {},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "spikes", "move_name": "Spikes"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("layers=0/3", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_branch"], "spikes_available")
        self.assertEqual(decision.metadata["teacher_score"], 62.0)

    def test_scripted_teacher_values_spikes_when_layers_remain_available(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_side_conditions": ["spikes"],
                "opponent_side_condition_counts": {"spikes": 2},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "spikes", "move_name": "Spikes"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("layers=2/3", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_score"], 62.0)

    def test_scripted_teacher_preserves_legacy_spikes_metadata_behavior(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_side_conditions": ["spikes"],
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "spikes", "move_name": "Spikes"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIn("layers=1/3", decision.metadata["teacher_reason"])
        self.assertEqual(decision.metadata["teacher_score"], 62.0)

    def test_scripted_teacher_does_not_value_spikes_when_maxed(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, True, False, False, False, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "opponent_side_conditions": ["spikes"],
                "opponent_side_condition_counts": {"spikes": 3},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "spikes", "move_name": "Spikes"},
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertNotEqual(decision.metadata["teacher_branch"], "spikes_available")

    def test_scripted_teacher_penalizes_statused_switch_targets(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (False, False, False, False, True, True, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 0.8, "status": "none"},
                "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    },
                    {
                        "action_index": 5,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Snorlax", "hp_fraction": 1.0, "status": "brn"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertIn("status_penalty=0.0", decision.metadata["teacher_reason"])

    def test_scripted_teacher_boosts_safe_switches_when_active_is_critically_low(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, True, False, False, False, False),
            metadata={
                "self_active": {"species": "Charizard", "hp_fraction": 0.1, "status": "none"},
                "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "flamethrower", "move_name": "Flamethrower"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 4)
        self.assertIn("preserve=", decision.metadata["teacher_reason"])

    def test_scripted_teacher_does_not_panic_switch_low_hp_active_into_bad_matchup(self) -> None:
        policy = ScriptedTeacherPolicy(dex=teacher_dex())
        obs = observation(
            (True, False, False, False, True, False, False, False, False),
            metadata={
                "self_active": {"species": "Snorlax", "hp_fraction": 0.1, "status": "none"},
                "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "tackle", "move_name": "Tackle"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                    },
                ],
            },
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)


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
                "rapidspin": {
                    "id": "rapidspin",
                    "name": "Rapid Spin",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 20,
                    "accuracy": 100,
                    "priority": 0,
                },
                "healbell": {
                    "id": "healbell",
                    "name": "Heal Bell",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": True,
                    "priority": 0,
                },
                "spikes": {
                    "id": "spikes",
                    "name": "Spikes",
                    "type": "Ground",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": True,
                    "priority": 0,
                },
                "thunderwave": {
                    "id": "thunderwave",
                    "name": "Thunder Wave",
                    "type": "Electric",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "priority": 0,
                    "status": "par",
                },
                "glare": {
                    "id": "glare",
                    "name": "Glare",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "priority": 0,
                    "status": "par",
                },
                "toxic": {
                    "id": "toxic",
                    "name": "Toxic",
                    "type": "Poison",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 85,
                    "priority": 0,
                    "status": "tox",
                },
                "willowisp": {
                    "id": "willowisp",
                    "name": "Will-O-Wisp",
                    "type": "Fire",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 75,
                    "priority": 0,
                    "status": "brn",
                },
            },
            "species": {
                "charizard": {"id": "charizard", "name": "Charizard", "types": ["Fire", "Flying"], "baseStats": {}},
                "xatu": {"id": "xatu", "name": "Xatu", "types": ["Psychic", "Flying"], "baseStats": {}},
                "golem": {"id": "golem", "name": "Golem", "types": ["Rock", "Ground"], "baseStats": {}},
                "starmie": {"id": "starmie", "name": "Starmie", "types": ["Water", "Psychic"], "baseStats": {}},
                "snorlax": {"id": "snorlax", "name": "Snorlax", "types": ["Normal"], "baseStats": {}},
                "dusclops": {"id": "dusclops", "name": "Dusclops", "types": ["Ghost"], "baseStats": {}},
                "skarmory": {"id": "skarmory", "name": "Skarmory", "types": ["Steel", "Flying"], "baseStats": {}},
            },
            "typeChart": {
                "flying": {"Rock": 1, "Ground": 3},
                "fire": {"Rock": 1, "Ground": 1, "Fire": 2, "Water": 1},
                "psychic": {"Ghost": 1},
                "ghost": {"Normal": 3, "Ghost": 2},
                "rock": {"Fire": 2, "Water": 1},
                "ground": {"Electric": 3, "Water": 1},
                "steel": {"Poison": 3, "Fire": 1},
                "water": {"Fire": 2, "Rock": 2, "Ground": 2},
                "normal": {"Ghost": 3},
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
