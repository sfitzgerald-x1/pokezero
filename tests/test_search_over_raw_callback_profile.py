"""Component profiling contracts; never launch a search or continuation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_search_over_raw_callback as driver
from pokezero.mcts_eval.search_over_raw import digest, ENGINEERING_EXCLUDED_SEEDS
from pokezero.neural_policy import TransformerInferenceTimingAccumulator
from pokezero.policy_opponent import make_policy_opponent_callback
from test_policy_opponent import config
from test_policy_opponent_request import bundle
from test_policy_opponent_view import LINES, VOCAB
from test_engine_world import _dex
from test_showdown import FakeSetSource
from _showdown_root import requires_showdown, showdown_root


def registration(output):
    return dict(schema=driver.SCHEMA, namespace=driver.NAMESPACE,
        fixture_seed=driver.FIXTURE_SEED, seeds=[driver.FIXTURE_SEED], repetitions=driver.REPETITIONS,
        seats=["p1", "p2"], configurations=[], attempt_directory=str(output.resolve()),
        fixture={}, fixture_sha256=digest({}), source_root=str(driver.ROOT),
        input_sha256={str(Path(driver.__file__).resolve()): driver.sha256_file(driver.__file__)},
        retry_authorized=False, phase_a_admission=False, phase_b_authorized=False,
        scientific_strength_evidence=False, representative_runtime_evidence=False, deployable=False)


class CallbackProfileTests(unittest.TestCase):
    def test_new_fixture_is_excluded_and_not_a_closed_search_seed(self):
        self.assertIn(driver.FIXTURE_SEED, ENGINEERING_EXCLUDED_SEEDS)
        self.assertNotIn(driver.FIXTURE_SEED, (2026101009, 2026101013))

    def test_contract_rejects_changed_scope_seed_repetitions_or_admission(self):
        output = Path("fixture").resolve()
        driver.validate_contract(registration(output), output)
        for field, changed in (("namespace", "old"), ("repetitions", 1), ("seats", ["p1"]),
                ("fixture_seed", 2026101013), ("configurations", [{}]), ("fixture", {"changed": True}),
                ("retry_authorized", True), ("phase_a_admission", True), ("phase_b_authorized", True),
                ("scientific_strength_evidence", True), ("representative_runtime_evidence", True),
                ("deployable", True), ("seeds", [2026101013])):
            r = registration(output)
            r[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_contract(r, output)

    def test_source_hash_and_attempt_directory_are_bound(self):
        output = Path("fixture").resolve()
        for field, value in (("source_root", "other"), ("input_sha256", {}), ("attempt_directory", "other")):
            r = registration(output)
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_contract(r, output)

    def test_repeated_calls_refuse_changed_output(self):
        counter = iter(((1., 0.), (0., 1.)))
        with self.assertRaisesRegex(ValueError, "output changed"):
            driver.repeated(lambda payload: next(counter), "authored", 2)

    def test_component_profile_calls_complete_unchanged_canonical_provider(self):
        cfg = config()
        args = dict(public_lines=LINES, hp_visibility={"p1": "exact", "p2": "exact"},
            opponent_slot="p2", battle_id="authored", battle_seed=20, format_id="gen3randombattle",
            set_source=FakeSetSource(), model=SimpleNamespace(config=cfg),
            result=SimpleNamespace(model_config=cfg, belief_set_source_hash=None), category_vocab=VOCAB,
            dex=_dex(), raw_argmax=True)
        payload = json.dumps(dict(native_request_bundle=bundle(), public_branch_lines=[], opponent_slot="p2"))
        callback = make_policy_opponent_callback(**args)
        timing = TransformerInferenceTimingAccumulator()
        timed = make_policy_opponent_callback(**args, timing=timing)
        calls = []
        def forward(**kwargs):
            calls.append(kwargs)
            if kwargs["timing"] is not None:
                kwargs["timing"].add_observation_encoding(0.)
                kwargs["timing"].add_neural_forward(0., role="action_prior")
            return (.8, .1, 0, 0, .1, 0, 0, 0, 0)
        with patch("pokezero.neural_policy.evaluate_transformer_action_priors", new=forward):
            measured = driver.component_profile(callback, timed, payload, 3, timing)
        self.assertEqual(len(calls), 7)  # warmup + two measured passes
        self.assertEqual(measured["canonical_output"], [1., 0., 0.])
        self.assertEqual(measured["functions"]["view_reconstruction"]["calls"], 3)
        self.assertEqual(measured["functions"]["canonical_observation"]["calls"], 3)
        self.assertEqual(measured["functions"]["payload_json_loads_direct"]["calls"], 3)
        self.assertTrue(measured["cumulative_metrics_overlap_do_not_sum"])
        self.assertIn("combined_instrumentation_overhead_ratio", measured)
        self.assertNotIn("profiler_overhead_ratio", measured)
        self.assertEqual(measured["neural_timing_profiled"]["neural_forward_count"], 3)
        self.assertEqual(measured["payload_sha256"], digest(json.loads(payload)))

    def test_failure_claim_precedes_runtime_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            driver.save_new(output / "registration.json", registration(output))
            def fail(r, target, progress):
                self.assertTrue((target / "attempt.json").exists())
                progress["stage"] = "p2:profile"
                raise ValueError("authored failure")
            with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=fail) as calls:
                self.assertEqual(driver.run(output), 1)
                before = (output / "terminal.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    driver.run(output)
                calls.assert_called_once()
                self.assertEqual(before, (output / "terminal.json").read_bytes())
            terminal = json.loads(before)
            self.assertEqual(terminal["status"], "FAILED_NO_RETRY")
            self.assertEqual(terminal["stage"], "p2:profile")
            self.assertEqual(terminal["exit_code"], 1)

    def test_drift_refuses_before_runtime_and_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            driver.save_new(output / "registration.json", registration(output))
            with patch.object(driver, "verify_inputs", side_effect=ValueError("drift")), \
                    patch.object(driver, "measure") as measured:
                with self.assertRaisesRegex(ValueError, "drift"):
                    driver.run(output)
                measured.assert_not_called()
            self.assertFalse((output / "attempt.json").exists())

    def test_success_is_component_only_not_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            driver.save_new(output / "registration.json", registration(output))
            with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", return_value=[{"seat": "p1"}, {"seat": "p2"}]):
                self.assertEqual(driver.run(output), 0)
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "COMPLETE_COMPONENT_PROFILE_ONLY")
            self.assertEqual(terminal["searched_decisions"], 0)
            self.assertEqual(terminal["terminal_games"], 0)
            self.assertFalse(terminal["phase_a_admission"])


@requires_showdown()
class ComponentAuthoredFixtureTests(unittest.TestCase):
    def test_full_teams_roundtrip_without_search_or_game_steps(self):
        import poke_engine
        import pokezero_search
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import build_engine_world
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.mcts_eval.search_over_raw_oracle import opening_team
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.showdown_fixture import pack_team
        source = load_gen3_randbat_source_cached(showdown_root())
        dex = load_showdown_dex_cached(showdown_root())
        authored = driver.fixture(source)
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True))
        self.addCleanup(env.close)
        override = BattleStartOverride(player_teams=authored["player_teams"], observation_format_id="gen3randombattle")
        env.reset_with_start_override(seed=driver.FIXTURE_SEED, start_override=override)
        snapshot = env.snapshot_actionable_boundary()
        for slot in ("p1", "p2"):
            self.assertEqual(pack_team(opening_team(snapshot.first_requests[slot], slot, source)),
                authored["player_teams"][slot])
            self.assertEqual(len(snapshot.first_requests[slot]["side"]["pokemon"]), 6)
            materialization = env.public_materialization_state(slot)
            world, native = build_engine_world(materialization, override, dex=dex, module=poke_engine)
            names = [dex.species_info(name).name for name in world.party_species[slot]]
            native_request = json.loads(pokezero_search.sampled_policy_request(native.to_string(), slot, names, names,
                {name: info.max_pp for name, info in dex.moves.items()}, base_pp={name: info.pp for name, info in dex.moves.items()}))
            self.assertEqual(native_request["native_action_indices"], list(range(9)))
            self.assertEqual([(m["id"], m["pp"], m["maxpp"]) for m in native_request["request"]["active"][0]["moves"]],
                [(m["id"], m["pp"], m["maxpp"]) for m in snapshot.first_requests[slot]["active"][0]["moves"]])
        self.assertIsNone(snapshot.terminal)


if __name__ == "__main__":
    unittest.main()
