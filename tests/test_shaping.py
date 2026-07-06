import json
import math
from pathlib import Path
import tempfile
import unittest

from pokezero.collection import RolloutRecord, rollout_record_from_dict, rollout_record_to_dict
from pokezero.env import TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.shaping import (
    NON_VOLATILE_STATUSES,
    SHAPING_PRESETS,
    ShapingConfig,
    SideSnapshot,
    annotate_record_with_shaping,
    action_class_components_by_step_index,
    action_class_names,
    action_class_shaping_rewards_by_step_index,
    component_names,
    components_from_sides,
    ground_truth_components_by_step_index,
    parse_shaping_spec,
    potential_from_sides,
    potential_shaping_rewards_by_step_index,
    potentials_by_step_index,
    resolve_shaping_config,
    shaping_rewards_by_step_index,
    shaping_terms,
    side_snapshot_from_observation_metadata,
)
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

MASK = (True, False, False, False, False, False, False, False, False)

WSE = SHAPING_PRESETS["wse-arm1"]


def observation(metadata: dict | None = None, legal_action_mask: tuple[bool, ...] = MASK) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=legal_action_mask,
        metadata=metadata or {},
    )


def team_metadata(*mons, spikes: int = 0) -> dict:
    return {
        "self_team": [dict(mon) for mon in mons],
        "self_side_condition_counts": {"spikes": spikes} if spikes else {},
    }


def mon(hp: float = 1.0, status: str | None = None, fainted: bool = False) -> dict:
    return {"hp_fraction": hp, "status": status, "fainted": fainted}


def step(
    player_id: str,
    turn_index: int,
    metadata: dict,
    shaping_reward: float | None = None,
    action_index: int = 0,
    legal_action_mask: tuple[bool, ...] = MASK,
) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn_index,
        observation=observation(metadata, legal_action_mask=legal_action_mask),
        legal_action_mask=legal_action_mask,
        action_index=action_index,
        reward=0.0,
        shaping_reward=shaping_reward,
    )


def record_from_steps(steps, winner: str | None = "p1", turn_count: int | None = None) -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="shaping-test", format_id="gen3randombattle", seed=7)
    for item in steps:
        trajectory.append(item)
    terminal = TerminalState(winner=winner, turn_count=turn_count or (steps[-1].turn_index + 1))
    trajectory.record_terminal(terminal)
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "test", "p2": "test"},
        decision_round_count=len(steps),
        elapsed_seconds=0.1,
        terminal=terminal,
        trajectory=trajectory,
    )


class ShapingConfigTest(unittest.TestCase):
    def test_preset_matches_wse_arm1_weight_structure(self) -> None:
        self.assertEqual(WSE.hp_weight, 0.5)
        self.assertEqual(WSE.faint_weight, 0.5)
        self.assertEqual(WSE.hazard_weight, 0.0)
        self.assertEqual({status for status, _ in WSE.status_weights}, set(NON_VOLATILE_STATUSES))
        self.assertTrue(all(weight == 0.25 for _, weight in WSE.status_weights))
        self.assertEqual(WSE.terminal_mode, "zero")

    def test_action_class_basis_rewards_behaviors_not_tools(self) -> None:
        self.assertEqual(
            action_class_names(),
            ("damage_dealt", "damage_taken", "switch_made", "boost_used", "heal_used", "ko"),
        )
        forbidden_tools = {"hazard", "status", "lay_hazard", "use_status", *component_names()}
        self.assertTrue(forbidden_tools.isdisjoint(action_class_names()))

    def test_parse_spec_supports_preset_json_file_and_off(self) -> None:
        self.assertEqual(parse_shaping_spec("wse-arm1"), WSE)
        inline = parse_shaping_spec('{"hp_weight": 1.0, "status_weight": 0.5}')
        self.assertEqual(inline.hp_weight, 1.0)
        self.assertEqual(inline.status_weight("par"), 0.5)
        self.assertIsNone(parse_shaping_spec("none"))
        self.assertIsNone(parse_shaping_spec("off"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.json"
            path.write_text(WSE.canonical_json(), encoding="utf-8")
            self.assertEqual(parse_shaping_spec(f"@{path}"), WSE)
        with self.assertRaisesRegex(ValueError, "unsupported shaping weights spec"):
            parse_shaping_spec("not-a-preset")

    def test_config_round_trip_and_canonical_json(self) -> None:
        config = ShapingConfig(
            hp_weight=0.3,
            faint_weight=0.7,
            status_weights=(("par", 0.1), ("brn", 0.2)),
            damage_dealt_weight=0.4,
            switch_made_weight=-0.2,
        )
        self.assertEqual(ShapingConfig.from_dict(config.to_dict()), config)
        self.assertEqual(ShapingConfig.from_json(config.canonical_json()), config)
        # Canonical: sorted status keys regardless of construction order.
        self.assertEqual(config.status_weights, (("brn", 0.2), ("par", 0.1)))
        self.assertEqual(config.action_class_weights()["damage_dealt"], 0.4)
        self.assertEqual(config.action_class_weights()["switch_made"], -0.2)
        self.assertEqual(resolve_shaping_config(config.to_dict()), config)
        self.assertEqual(resolve_shaping_config(config), config)
        self.assertIsNone(resolve_shaping_config(None))

    def test_config_rejects_unknown_statuses_keys_and_terminal_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown status"):
            ShapingConfig(status_weights=(("frozen", 1.0),))
        with self.assertRaisesRegex(ValueError, "duplicate status"):
            ShapingConfig(status_weights=(("par", 1.0), ("PAR", 2.0)))
        with self.assertRaisesRegex(ValueError, "terminal_mode"):
            ShapingConfig(terminal_mode="carry-on")
        with self.assertRaisesRegex(ValueError, "unknown shaping config key"):
            ShapingConfig.from_dict({"hp": 0.5})
        with self.assertRaisesRegex(ValueError, "both"):
            ShapingConfig.from_dict({"status_weights": {"par": 1.0}, "status_weight": 1.0})

    def test_is_zero(self) -> None:
        self.assertTrue(ShapingConfig().is_zero())
        self.assertTrue(ShapingConfig(status_weights=(("par", 0.0),)).is_zero())
        self.assertFalse(ShapingConfig(hazard_weight=0.1).is_zero())
        self.assertFalse(ShapingConfig(boost_used_weight=0.1).is_zero())

    def test_action_class_component_names(self) -> None:
        self.assertEqual(
            action_class_names(),
            ("damage_dealt", "damage_taken", "switch_made", "boost_used", "heal_used", "ko"),
        )


class PotentialTest(unittest.TestCase):
    def test_symmetric_states_have_zero_potential(self) -> None:
        side = SideSnapshot(hp_total=4.5, alive=5, status_counts=(("par", 1),), spikes_layers=2)
        self.assertEqual(potential_from_sides(side, side, WSE), 0.0)

    def test_own_ko_lowers_potential_and_foe_status_raises_it(self) -> None:
        healthy = SideSnapshot(hp_total=6.0, alive=6)
        after_own_ko = SideSnapshot(hp_total=5.0, alive=5)
        self.assertLess(potential_from_sides(after_own_ko, healthy, WSE), 0.0)
        foe_statused = SideSnapshot(hp_total=6.0, alive=6, status_counts=(("brn", 1),))
        self.assertGreater(potential_from_sides(healthy, foe_statused, WSE), 0.0)

    def test_components_normalization_and_ordering(self) -> None:
        own = SideSnapshot(hp_total=6.0, alive=6)
        foe = SideSnapshot(hp_total=3.0, alive=4, status_counts=(("tox", 2),), spikes_layers=3)
        components = components_from_sides(own, foe)
        self.assertEqual(set(components), set(component_names()))
        self.assertAlmostEqual(components["hp"], 0.5)
        self.assertAlmostEqual(components["faint"], 2 / 6)
        self.assertAlmostEqual(components["status:tox"], 2 / 6)
        self.assertAlmostEqual(components["hazard"], 1.0)

    def test_fainted_mons_drop_out_of_hp_alive_and_status_counts(self) -> None:
        snapshot = side_snapshot_from_observation_metadata(
            team_metadata(mon(0.0, status="par", fainted=True), mon(0.5, status="brn"), mon(1.0), spikes=2)
        )
        self.assertEqual(snapshot.alive, 2)
        self.assertAlmostEqual(snapshot.hp_total, 1.5)
        # The fainted mon's paralysis must NOT count (a KO is not "status cleared").
        self.assertEqual(snapshot.status_count("par"), 0)
        self.assertEqual(snapshot.status_count("brn"), 1)
        self.assertEqual(snapshot.spikes_layers, 2)

    def test_missing_metadata_degrades_to_empty_side(self) -> None:
        self.assertEqual(side_snapshot_from_observation_metadata(None), SideSnapshot())
        self.assertEqual(side_snapshot_from_observation_metadata({}), SideSnapshot())


class ShapingTermsTest(unittest.TestCase):
    def test_telescoping_to_terminal_and_initial_potential(self) -> None:
        gamma = 0.97
        potentials = [0.0, 0.25, -0.4, 0.6, 0.55]
        for terminal_potential in (0.0, 0.55, -1.0):
            terms = shaping_terms(potentials, gamma=gamma, terminal_potential=terminal_potential)
            self.assertEqual(len(terms), len(potentials))
            discounted_total = sum((gamma**k) * term for k, term in enumerate(terms))
            expected = (gamma ** len(potentials)) * terminal_potential - potentials[0]
            self.assertAlmostEqual(discounted_total, expected, places=12)

    def test_zero_weights_produce_zero_terms(self) -> None:
        record = record_from_steps(
            [
                step("p1", 0, team_metadata(mon(1.0))),
                step("p2", 0, team_metadata(mon(1.0))),
                step("p1", 1, team_metadata(mon(0.2))),
            ]
        )
        rewards = potential_shaping_rewards_by_step_index(record, config=ShapingConfig(), gamma=0.99)
        self.assertEqual(set(rewards.values()), {0.0})
        self.assertEqual(len(rewards), 3)


class RecordShapingTest(unittest.TestCase):
    def two_player_record(self) -> RolloutRecord:
        # p1 stays healthy; p2's side degrades: hp loss at turn 1, a faint + burn at turn 2.
        return record_from_steps(
            [
                step("p1", 0, team_metadata(mon(1.0), mon(1.0))),
                step("p2", 0, team_metadata(mon(1.0), mon(1.0))),
                step("p1", 1, team_metadata(mon(1.0), mon(1.0))),
                step("p2", 1, team_metadata(mon(0.5), mon(1.0))),
                step("p1", 2, team_metadata(mon(1.0), mon(1.0))),
                step("p2", 2, team_metadata(mon(0.0, fainted=True), mon(1.0, status="brn"))),
            ],
            winner="p1",
        )

    def test_ground_truth_combines_both_self_views(self) -> None:
        record = self.two_player_record()
        components = ground_truth_components_by_step_index(record)
        # p1's turn-1 step sees p2's turn-1 self view (0.5 + 1.0 hp).
        self.assertAlmostEqual(components[2]["hp"], (2.0 - 1.5) / 6)
        # p2's turn-1 step is the mirror.
        self.assertAlmostEqual(components[3]["hp"], (1.5 - 2.0) / 6)
        # p1's turn-2 step: foe has one fainted mon and one burned mon.
        self.assertAlmostEqual(components[4]["faint"], (2 - 1) / 6)
        self.assertAlmostEqual(components[4]["status:brn"], 1 / 6)
        # Mirrored perspectives are exact negations.
        for name in component_names():
            self.assertAlmostEqual(components[4][name], -components[5][name])

    def test_sign_conventions_over_a_record(self) -> None:
        record = self.two_player_record()
        rewards = potential_shaping_rewards_by_step_index(record, config=WSE, gamma=1.0)
        # p1's first decision precedes the foe hp drop: positive shaping. p2's mirrors negative.
        self.assertGreater(rewards[0], 0.0)
        self.assertLess(rewards[1], 0.0)
        # p1's second decision precedes the foe faint + burn: positive again.
        self.assertGreater(rewards[2], 0.0)
        self.assertLess(rewards[3], 0.0)
        # Terminal-mode zero: the last decision gives back the accumulated potential.
        self.assertLess(rewards[4], 0.0)
        self.assertGreater(rewards[5], 0.0)

    def test_per_player_terms_telescope_through_the_record(self) -> None:
        record = self.two_player_record()
        gamma = 0.9999
        rewards = potential_shaping_rewards_by_step_index(record, config=WSE, gamma=gamma)
        potentials = potentials_by_step_index(record, config=WSE)
        for player_id in ("p1", "p2"):
            indices = [
                index
                for index, trajectory_step in enumerate(record.trajectory.steps)
                if trajectory_step.player_id == player_id
            ]
            total = sum((gamma**k) * rewards[index] for k, index in enumerate(indices))
            expected = (gamma ** len(indices)) * 0.0 - potentials[indices[0]]
            self.assertAlmostEqual(total, expected, places=12)

    def test_carry_terminal_mode_keeps_final_potential(self) -> None:
        record = self.two_player_record()
        carry = ShapingConfig.from_dict({**WSE.to_dict(), "terminal_mode": "carry"})
        rewards = potential_shaping_rewards_by_step_index(record, config=carry, gamma=1.0)
        potentials = potentials_by_step_index(record, config=carry)
        indices = [0, 2, 4]
        total = sum(rewards[index] for index in indices)
        self.assertAlmostEqual(total, potentials[4] - potentials[0], places=12)

    def test_hazard_component_counts_spikes_layers(self) -> None:
        config = ShapingConfig(hazard_weight=0.3)
        record = record_from_steps(
            [
                step("p1", 0, team_metadata(mon(1.0))),
                step("p2", 0, team_metadata(mon(1.0))),
                step("p1", 1, team_metadata(mon(1.0))),
                step("p2", 1, team_metadata(mon(1.0), spikes=2)),
            ],
            winner="p1",
        )
        rewards = potential_shaping_rewards_by_step_index(record, config=config, gamma=1.0)
        # Foe (p2) side gained 2 spikes layers between p1's decisions: +0.3 * 2/3.
        self.assertAlmostEqual(rewards[0], 0.3 * 2 / 3)

    def test_action_class_components_and_rewards(self) -> None:
        switch_mask = (False, False, False, False, True, False, False, False, False)
        move_metadata = {
            "action_candidates": [
                {"action_index": 0, "kind": "move", "move_id": "swordsdance", "move_name": "Swords Dance"},
            ],
        }
        heal_metadata = {
            "action_candidates": [
                {"action_index": 0, "kind": "move", "move_id": "recover", "move_name": "Recover"},
            ],
        }
        switch_metadata = {
            "action_candidates": [
                {"action_index": 4, "kind": "switch", "pokemon": {"species": "Starmie"}},
            ],
        }
        record = record_from_steps(
            [
                step("p1", 0, {**team_metadata(mon(1.0), mon(1.0)), **move_metadata}),
                step("p2", 0, {**team_metadata(mon(1.0), mon(1.0)), **heal_metadata}),
                step(
                    "p1",
                    1,
                    {**team_metadata(mon(1.0), mon(0.5)), **switch_metadata},
                    action_index=4,
                    legal_action_mask=switch_mask,
                ),
                step("p2", 1, team_metadata(mon(0.0, fainted=True), mon(1.0))),
            ],
            winner="p1",
        )

        components = action_class_components_by_step_index(record)
        self.assertEqual(components[0]["boost_used"], 1.0)
        self.assertAlmostEqual(components[0]["damage_dealt"], 1 / 6)
        self.assertAlmostEqual(components[0]["damage_taken"], 0.5 / 6)
        self.assertAlmostEqual(components[0]["ko"], 1 / 6)
        self.assertEqual(components[1]["heal_used"], 1.0)
        self.assertEqual(components[2]["switch_made"], 1.0)

        config = ShapingConfig(
            damage_dealt_weight=6.0,
            damage_taken_weight=-6.0,
            ko_weight=6.0,
            boost_used_weight=0.25,
            heal_used_weight=0.5,
            switch_made_weight=-0.75,
        )
        rewards = action_class_shaping_rewards_by_step_index(record, config=config)
        self.assertAlmostEqual(rewards[0], 1.0 - 0.5 + 1.0 + 0.25)
        self.assertAlmostEqual(rewards[1], 0.5 - 1.0 + 0.5)
        self.assertAlmostEqual(rewards[2], -0.75)

    def test_full_shaping_combines_potential_and_action_class_terms(self) -> None:
        config = ShapingConfig(hazard_weight=0.3, switch_made_weight=0.4)
        switch_mask = (False, False, False, False, True, False, False, False, False)
        record = record_from_steps(
            [
                step(
                    "p1",
                    0,
                    {
                        **team_metadata(mon(1.0)),
                        "action_candidates": [{"action_index": 4, "kind": "switch"}],
                    },
                    action_index=4,
                    legal_action_mask=switch_mask,
                ),
                step("p2", 0, team_metadata(mon(1.0))),
                step("p1", 1, team_metadata(mon(1.0))),
                step("p2", 1, team_metadata(mon(1.0), spikes=2)),
            ],
            winner="p1",
        )

        potential_rewards = potential_shaping_rewards_by_step_index(record, config=config, gamma=1.0)
        full_rewards = shaping_rewards_by_step_index(record, config=config, gamma=1.0)
        self.assertAlmostEqual(potential_rewards[0], 0.3 * 2 / 3)
        self.assertAlmostEqual(full_rewards[0], potential_rewards[0] + 0.4)

    def test_annotate_record_round_trips_and_preserves_raw_reward(self) -> None:
        record = self.two_player_record()
        annotated = annotate_record_with_shaping(record, config=WSE, gamma=0.9999)
        expected = potential_shaping_rewards_by_step_index(record, config=WSE, gamma=0.9999)
        for index, trajectory_step in enumerate(annotated.trajectory.steps):
            self.assertAlmostEqual(trajectory_step.shaping_reward, expected[index])
            self.assertEqual(trajectory_step.reward, record.trajectory.steps[index].reward)
        payload = rollout_record_to_dict(annotated)
        restored = rollout_record_from_dict(json.loads(json.dumps(payload)))
        self.assertAlmostEqual(restored.trajectory.steps[0].shaping_reward, expected[0])

    def test_annotate_record_includes_action_class_terms(self) -> None:
        switch_mask = (False, False, False, False, True, False, False, False, False)
        record = record_from_steps(
            [
                step(
                    "p1",
                    0,
                    {
                        **team_metadata(mon(1.0)),
                        "action_candidates": [{"action_index": 4, "kind": "switch"}],
                    },
                    action_index=4,
                    legal_action_mask=switch_mask,
                )
            ],
            winner="p1",
        )

        annotated = annotate_record_with_shaping(
            record,
            config=ShapingConfig(switch_made_weight=0.2),
            gamma=1.0,
        )
        self.assertAlmostEqual(annotated.trajectory.steps[0].shaping_reward, 0.2)

    def test_unshaped_records_serialize_without_the_field(self) -> None:
        record = self.two_player_record()
        payload = rollout_record_to_dict(record)
        for step_payload in payload["trajectory"]["steps"]:
            self.assertNotIn("shaping_reward", step_payload)
        self.assertIsNone(rollout_record_from_dict(payload).trajectory.steps[0].shaping_reward)

    def test_step_shaping_reward_must_be_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "shaping_reward"):
            step("p1", 0, team_metadata(mon(1.0)), shaping_reward=math.nan)


if __name__ == "__main__":
    unittest.main()
