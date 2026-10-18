"""Authored callback fixtures only; no real sources, models or native search."""
from dataclasses import replace
import hashlib
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, EngineSearchFallbackWarning
from pokezero.engine_world import EngineWorld
from pokezero.mcts_eval.search_over_raw_information_worlds import OneWorldTeamCapture
import tests.test_engine_search as engine_fixtures


class InitialTeamCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config = EngineMctsConfig(worlds=1, leaf_eval="model", model_path="model",
            checkpoint_path="checkpoint", tables_path="tables", search_sims=1024,
            search_batch=16, search_depth=6, early_stop=False, strict_fallbacks=True)
        self.context = SimpleNamespace(seed=7, battle_id="authored", decision_round_index=1, player_id="p1")
        self.override = SimpleNamespace(player_teams={"p1": "authored-own", "p2": "authored-opponent"})
        self.world = SimpleNamespace(slot_sides={"p1": "side_one", "p2": "side_two"},
                                     party_species={"p1": ["pikachu"], "p2": ["snorlax"]})
        self.state = SimpleNamespace(to_string=lambda: "authored-state")
        self.capture = OneWorldTeamCapture(self.context, self.config, boundary_key=lambda c: "a" * 64)

    def record(self):
        self.capture.attempt(self.context, 0, self.override)
        self.capture.world(self.context, self.world, self.state)

    def finish(self, **kwargs):
        return self.capture.finish(self.context, self.config,
            **(dict(attempts=1, constructed=1, searched=1, iterations=1024) | kwargs))

    def test_copies_draw_and_world_at_the_original_callbacks(self):
        self.record()
        self.override.player_teams["p2"] = "changed"
        self.world.party_species["p2"].append("changed")
        row = self.finish()
        self.assertEqual(row["attempts"][0]["initial_packed_teams"]["p2"], "authored-opponent")
        self.assertEqual(row["constructed"][0]["party_species"]["p2"], ["snorlax"])
        self.assertEqual(row["constructed"][0]["constructed_state_sha256"],
                         hashlib.sha256(b"authored-state").hexdigest())
        row["attempts"].clear()
        self.assertEqual(len(self.finish()["attempts"]), 1)
        for flag in ("explicit_native_input_hash_comparison", "source_truth_bound",
                     "current_state_accuracy_established", "scientific_strength_claim"):
            self.assertIs(row[flag], False)

    def test_refused_draw_is_retained_not_mistaken_for_consumed_world(self):
        self.capture.attempt(self.context, 0, None)
        self.capture.attempt(self.context, 1, self.override)
        self.capture.world(self.context, self.world, self.state)
        row = self.finish(attempts=2)
        self.assertIsNone(row["attempts"][0]["initial_packed_teams"])
        self.assertEqual(row["constructed"][0]["attempt_ordinal"], 1)

    def test_missing_swallowed_callbacks_cannot_be_accepted(self):
        with self.assertRaises(ValueError):
            self.finish()
        self.capture.attempt(self.context, 0, self.override)
        with self.assertRaises(ValueError):
            self.finish()
        self.capture.world(self.context, self.world, self.state)
        with self.assertRaises(ValueError):
            self.finish(attempts=2)

    def test_construction_is_not_successful_search(self):
        self.record()
        for change in (dict(searched=0), dict(iterations=1023), dict(constructed=2), dict(attempts=True)):
            with self.assertRaises(ValueError):
                self.finish(**change)

    def test_callback_identity_order_and_multiple_worlds_refuse(self):
        changed = SimpleNamespace(**(vars(self.context) | {"decision_round_index": 2}))
        with self.assertRaises(ValueError):
            self.capture.attempt(changed, 0, self.override)
        with self.assertRaises(ValueError):
            self.capture.attempt(self.context, 1, self.override)
        self.record()
        with self.assertRaises(ValueError):
            self.capture.world(self.context, self.world, self.state)
        with self.assertRaises(ValueError):
            self.capture.attempt(self.context, 1, self.override)

    def test_only_the_reviewed_cardinality_regime_can_be_associated(self):
        for change in (dict(worlds=2), dict(early_stop=True), dict(depth_min=3),
                       dict(model_decision_time_ms=1000), dict(search_sims=512)):
            with self.assertRaises(ValueError):
                OneWorldTeamCapture(self.context, replace(self.config, **change), boundary_key=lambda c: "a" * 64)
        overridden = SimpleNamespace(**(vars(self.config) | {"model_leaf_override": "hp_fraction"}))
        with self.assertRaises(ValueError):
            OneWorldTeamCapture(self.context, overridden, boundary_key=lambda c: "a" * 64)


class AttemptObserverIntegrationTests(unittest.TestCase):
    def run_source_path(self, observer):
        module = Mock()
        module.monte_carlo_tree_search.return_value = engine_fixtures.OwnSideSelectionTests._Result()
        policy = EngineMctsPolicy(dex=None, set_source=None, module=module,
                                 config=EngineMctsConfig(worlds=1, sample_retry_factor=3))
        policy._world_attempt_observer = observer
        candidates = [{"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
                      {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"}]
        context = engine_fixtures._FakeContext(engine_fixtures._FakeObservation(
            (True, True, False, False, False, False, False, False, False), candidates))
        world = EngineWorld(spec=None, slot_sides={"p1": "side_one", "p2": "side_two"},
                            party_species={"p1": (), "p2": ()})
        accepted = object()
        with patch("pokezero.engine_search._gen3_randbat_belief_start_override_result",
                   side_effect=[(None, "authored-refusal"), (accepted, None)]), \
             patch("pokezero.engine_search.world_battle_spec", return_value=world) as build, \
             patch("pokezero.engine_search.build_poke_engine_state", return_value=object()):
            decision = policy.select_action_with_context(context, rng=random.Random(0))
        self.assertIs(build.call_args.args[1], accepted)
        return decision, policy.stats

    def test_hook_observes_real_draws_before_construction_without_redrawing(self):
        observed = []
        decision, stats = self.run_source_path(lambda context, ordinal, override: observed.append((ordinal, override)))
        self.assertEqual([r[0] for r in observed], [0, 1])
        self.assertIsNone(observed[0][1])
        self.assertIsNotNone(observed[1][1])
        self.assertEqual(stats.worlds_attempted, 2)
        self.assertEqual(stats.worlds_constructed, 1)
        self.assertEqual(decision.action_index, self.run_source_path(None)[0].action_index)

    def test_raising_observer_does_not_change_search_acceptance(self):
        def fail(*args):
            raise ValueError("authored missing telemetry")
        with self.assertWarnsRegex(EngineSearchFallbackWarning, "world_attempt_observer raised"):
            decision, stats = self.run_source_path(fail)
        self.assertEqual(stats.worlds_attempted, 2)
        self.assertEqual(stats.worlds_searched, 1)
        self.assertEqual(decision.action_index, self.run_source_path(None)[0].action_index)


if __name__ == "__main__":
    unittest.main()
