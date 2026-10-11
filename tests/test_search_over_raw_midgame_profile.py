"""Authored histories and create-only diagnostic lifecycle; no scientific search."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_search_over_raw_midgame as driver
from pokezero.mcts_eval.search_over_raw import ENGINEERING_EXCLUDED_SEEDS, digest
from _showdown_root import requires_showdown, showdown_root


def registration(output, exposure):
    excluded, bindings = driver.exposure_inventory([exposure])
    return dict(driver.core(), fixture={}, fixture_sha256=digest({}),
        attempt_directory=str(output.resolve()), source_root=str(driver.ROOT),
        exposure_registrations=bindings, excluded_seeds=excluded,
        excluded_seed_inventory_sha256=digest(excluded),
        input_sha256={**bindings, **{str(Path(p).resolve()): driver.sha256_file(p)
            for p in (driver.__file__, driver.opening.__file__)}})


def own_request(seat="p1", turn=2):
    moves = driver.opening.SETS[0 if seat == "p1" else 4][1]
    team = [s[0] for s in driver.opening.SETS]
    team = team if seat == "p1" else team[4:] + team[:4]
    return dict(active=[dict(moves=[dict(id=move, pp=23 if move == "surf" else
        33-turn if move == "recover" else 16, maxpp=24 if move == "surf" else 32 if move == "recover" else 16)
        for move in moves])], side=dict(pokemon=[dict(details=species,
            condition="300/300", active=i == 0) for i, species in enumerate(team)]))


def rebuild_mock(r, inputs, seat):
    return inputs["payload"]["native_request_bundle"]


def synthetic_completion(output, progress, r):
    for cell in progress["roster"]:
        driver.save_new(output / (cell["cell_id"] + "-attempt.json"),
            dict(cell_id=cell["cell_id"], registration_sha256=digest(r)))
        request = own_request(cell["seat"], cell["completed_turns"])
        inputs = dict(payload=dict(native_request_bundle=dict(request=request,
            native_action_indices=list(range(9))), opponent_slot=cell["seat"], public_branch_lines=[]),
            own_request=request, public_lines=["|start", *[f"|turn|{t}" for t in
                range(1, cell["completed_turns"] + 2)]], native_state="synthetic-state",
            native_species=[p["details"] for p in request["side"]["pokemon"]],
            script_trace=[dict(turn=t, actions={"p1": 0, "p2": 0} if t == 1 else {"p1": 3, "p2": 2},
                move="surf" if t == 1 else "recover") for t in range(1, cell["completed_turns"] + 1)])
        row = dict(payload_sha256=digest(inputs["payload"]), public_prefix_sha256=digest(inputs["public_lines"]),
            native_state_sha256=digest(inputs["native_state"]), repetitions_per_pass=driver.REPETITIONS,
            native_request_calls=driver.REPETITIONS, callback_total_calls=51, native_helper_profile_calls=26,
            warmup_calls=1, measured_passes=2, warmup_seconds=.01, unprofiled_seconds=.2, profiled_seconds=.4,
            native_request_first_call_seconds=.001, native_request_seconds=.02,
            combined_instrumentation_overhead_ratio=2., cumulative_metrics_overlap_do_not_sum=True,
            canonical_output=[1., *([0.] * 8)],
            functions={name: dict(calls=driver.REPETITIONS, exclusive_seconds=.01, cumulative_seconds=.02)
                for name in driver.profile_functions()},
            profiler_rows=[dict(file=key[0], line=key[1], function=key[2], primitive_calls=driver.REPETITIONS,
                calls=driver.REPETITIONS, exclusive_seconds=.01, cumulative_seconds=.02, callers=[])
                for key in map(driver.opening.code_key, driver.profile_functions().values())],
            neural_timing_profiled=dict(neural_forward_count=driver.REPETITIONS,
                observation_encoding_count=driver.REPETITIONS, action_prior_neural_forward_count=driver.REPETITIONS,
                policy_neural_forward_count=0, value_neural_forward_count=0,
                opponent_action_prior_neural_forward_count=0, neural_forward_seconds=.1,
                observation_encoding_seconds=.1, action_prior_neural_forward_seconds=.1,
                policy_neural_forward_seconds=0., value_neural_forward_seconds=0.,
                opponent_action_prior_neural_forward_seconds=0.))
        driver.record_profile(output, progress, cell, row, inputs)


class MidgameContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.exposure = self.output / "exposure.json"
        self.exposure.write_text(json.dumps(dict(seeds=[2026101015, 2026101016])))
        self.r = registration(self.output, self.exposure)

    def prepare(self):
        driver.save_new(self.output / "registration.json", self.r)
        driver.save_new(self.output / "registration-binding.json", dict(registration_sha256=digest(self.r)))

    def test_seed_reserved_not_closed_attempt(self):
        self.assertIn(driver.FIXTURE_SEED, ENGINEERING_EXCLUDED_SEEDS)
        self.assertNotIn(driver.FIXTURE_SEED, (2026101013, 2026101014, 2026101015, 2026101016))
        self.assertEqual(len(driver.cells()), 6)
        self.assertEqual(len({c["cell_id"] for c in driver.cells()}), 6)

    def test_typed_core_and_fixed_roster_cannot_change(self):
        driver.validate_contract(self.r, self.output)
        for field, value in (("repetitions", 25.), ("fixture_seed", float(driver.FIXTURE_SEED)),
                ("strata", [2, 8]), ("seats", ["p1"]), ("namespace", "old"),
                ("retry_authorized", 0), ("deployable", True), ("configurations", [{}]),
                ("seeds", [2026101013]), ("scientific_strength_evidence", True)):
            r = copy.deepcopy(self.r)
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_contract(r, self.output)

    def test_fixture_directory_script_and_exposure_drift(self):
        for field, value in (("fixture", {"bad": True}), ("source_root", "/other"),
                ("attempt_directory", str(driver.ROOT)), ("input_sha256", {}), ("excluded_seeds", [])):
            r = copy.deepcopy(self.r)
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_contract(r, self.output)

    def test_overlap_or_noninteger_exposure_refuses(self):
        for roster in ([driver.FIXTURE_SEED], [True], [], [1.0]):
            self.exposure.write_text(json.dumps(dict(seeds=roster)))
            with self.subTest(roster=roster), self.assertRaises(ValueError):
                driver.exposure_inventory([self.exposure])

    def test_whole_registration_binding_refuses_before_claim(self):
        self.prepare()
        self.r["checkpoint"] = "changed"
        (self.output / "registration.json").write_text(json.dumps(self.r))
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as work:
            with self.assertRaisesRegex(ValueError, "whole-registration"):
                driver.run(self.output)
            work.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_existing_terminal_refuses_even_if_claim_missing(self):
        self.prepare()
        driver.save_new(self.output / "terminal.json", dict(status="FAILED_NO_RETRY"))
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as work:
            with self.assertRaisesRegex(ValueError, "no repeat"):
                driver.run(self.output)
            work.assert_not_called()

    def test_failure_has_claim_fixed_uncertain_suffix_and_no_repeat(self):
        self.prepare()
        def fail(r, output, progress):
            self.assertTrue((output / "attempt.json").exists())
            progress["stage"] = "turn2-p1"
            raise ValueError("fixture refusal")
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=fail) as work:
            self.assertEqual(driver.run(self.output), 1)
            before = (self.output / "terminal.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "no repeat"):
                driver.run(self.output)
            work.assert_called_once()
        terminal = json.loads(before)
        self.assertEqual(terminal["profiles_completed"], 0)
        self.assertEqual(terminal["roster"], driver.cells())
        self.assertEqual(terminal["error"], "fixture refusal")

    def test_full_durable_completion_not_admission(self):
        self.prepare()
        def complete(r, output, progress):
            synthetic_completion(output, progress, r)
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=complete), \
                patch.object(driver, "rebuild_bundle", side_effect=rebuild_mock):
            self.assertEqual(driver.run(self.output), 0)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["profiles_completed"], 6)
        self.assertEqual(terminal["status"], "COMPLETE_AUTHORED_MIDGAME_COMPONENT_ONLY")
        self.assertTrue(all(terminal[key] is False for key in driver.FLAGS))

    def test_mutable_counters_cannot_fake_completion(self):
        self.prepare()
        def lie(r, output, progress):
            progress["profiles_completed"] = 6
            for c in progress["roster"]:
                c["status"] = "COMPLETE_COMPONENT_ONLY"
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=lie):
            self.assertEqual(driver.run(self.output), 1)
        self.assertEqual(json.loads((self.output / "terminal.json").read_text())["error_type"], "FileNotFoundError")

    def test_completion_rejects_receipt_tampering(self):
        progress = dict(roster=driver.cells(), profiles_completed=0)
        synthetic_completion(self.output, progress, self.r)
        with patch.object(driver, "rebuild_bundle", side_effect=rebuild_mock):
            driver.verify_completion(self.output, progress, self.r)
        path = self.output / "turn24-p2-profile.json"
        original = json.loads(path.read_text())
        for key, value in (("payload_sha256", "changed"), ("native_state_sha256", "changed"),
                ("public_prefix_sha256", "changed"), ("repetitions_per_pass", 24),
                ("seat", "p1"), ("inputs_sha256", "changed")):
            row = {**original, key: value}
            path.write_text(json.dumps(row))
            with self.subTest(key=key), patch.object(driver, "rebuild_bundle", side_effect=rebuild_mock), self.assertRaises(ValueError):
                driver.verify_completion(self.output, progress, self.r)
        path.write_text(json.dumps(original))
        with patch.object(driver, "rebuild_bundle", side_effect=rebuild_mock):
            driver.verify_completion(self.output, progress, self.r)

    def test_partial_receipt_refuses_before_runtime_even_without_top_claim(self):
        self.prepare()
        driver.save_new(self.output / "turn2-p1-inputs.json", {})
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as work:
            with self.assertRaisesRegex(ValueError, "partial cell"):
                driver.run(self.output)
            work.assert_not_called()

    def test_saved_boundary_rejects_old_reproducer(self):
        request = own_request("p2", 24)
        lines = ["|start", *[f"|turn|{t}" for t in range(1, 26)]]
        driver.certify_saved_boundary(request, lines, "p2", 24)
        with self.assertRaisesRegex(ValueError, "boundary"):
            driver.certify_saved_boundary(request, ["|start"], "p2", 24)
        with self.assertRaisesRegex(ValueError, "identity"):
            driver.certify_saved_boundary(own_request("p1", 24), lines, "p2", 24)
        bad = copy.deepcopy(request)
        bad["active"][0]["moves"][2]["pp"] = 10
        with self.assertRaisesRegex(ValueError, "PP"):
            driver.certify_saved_boundary(bad, lines, "p2", 24)

    def test_raw_timings_output_rows_and_summaries_cannot_be_pruned_or_forged(self):
        progress = dict(roster=driver.cells(), profiles_completed=0)
        synthetic_completion(self.output, progress, self.r)
        original = json.loads((self.output / "turn2-p1-profile.json").read_text())
        driver.validate_profile(original)
        for key, value in (("profiler_rows", []), ("unprofiled_seconds", -1), ("profiled_seconds", "bad"),
                ("canonical_output", [1, 1]), ("warmup_calls", 1.), ("combined_instrumentation_overhead_ratio", 3)):
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.validate_profile(bad)
        for key in ("canonical_output", "profiler_rows", "profiled_seconds"):
            bad = copy.deepcopy(original)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(KeyError):
                driver.validate_profile(bad)
        bad = copy.deepcopy(original)
        bad["functions"]["player_normalization"]["exclusive_seconds"] = .012
        with self.assertRaisesRegex(ValueError, "raw profile"):
            driver.validate_profile(bad)

    def test_native_pp_hp_and_action_order_are_certified(self):
        request = own_request()
        bundle = dict(request=copy.deepcopy(request), native_action_indices=list(range(9)))
        driver.certify_request(bundle, request)
        for mutation in ("pp", "hp", "actions"):
            bad = copy.deepcopy(bundle)
            if mutation == "pp":
                bad["request"]["active"][0]["moves"][0]["pp"] = 9
            elif mutation == "hp":
                bad["request"]["side"]["pokemon"][0]["condition"] = "299/300"
            else:
                bad["native_action_indices"] = [0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                driver.certify_request(bad, request)

    def test_cleanup_failure_prevents_success_and_preserves_primary(self):
        class Resource:
            def close(self):
                raise ValueError("cleanup")
        progress = dict(stage="fixture")
        with self.assertRaisesRegex(ValueError, "cleanup"):
            driver.close_resources([("fixture", Resource())], progress)
        self.assertEqual(progress["cleanup_failures"][0]["resource"], "fixture")
        try:
            raise RuntimeError("primary")
        except RuntimeError:
            driver.close_resources([("fixture", Resource())], progress)


@requires_showdown()
class MidgameAuthoredFixtureTests(unittest.TestCase):
    def test_valid_nested_histories_both_seats_current_gen3_pp_and_hp(self):
        import poke_engine
        import pokezero_search
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import build_engine_world
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached
        source = load_gen3_randbat_source_cached(showdown_root())
        dex = load_showdown_dex_cached(showdown_root())
        authored = driver.fixture(source)
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True))
        self.addCleanup(env.close)
        override = BattleStartOverride(player_teams=authored["player_teams"], observation_format_id="gen3randombattle")
        env.reset_with_start_override(seed=driver.FIXTURE_SEED, start_override=override)
        completed, trace, counts = 0, [], []
        self.assertEqual(dex.moves["recover"].pp, 20)
        self.assertEqual(dex.moves["recover"].max_pp, 32)
        for target in driver.STRATA:
            driver.advance(env, completed, target, trace)
            completed = target
            for slot in driver.SEATS:
                public = env.public_materialization_state(slot)
                driver.certify_boundary(public, slot, target)
                self.assertEqual(env.observe(slot).legal_action_mask, (True,) * 9)
                world, native = build_engine_world(public, override, dex=dex, module=poke_engine)
                names = [dex.species_info(name).name for name in world.party_species[slot]]
                bundle = json.loads(pokezero_search.sampled_policy_request(native.to_string(), slot, names, names,
                    {name: info.max_pp for name, info in dex.moves.items()},
                    base_pp={name: info.pp for name, info in dex.moves.items()}))
                driver.certify_request(bundle, public.self_request)
                self.assertEqual(driver.rebuild_bundle(dict(showdown_root=str(showdown_root())),
                    dict(native_state=native.to_string(), native_species=names, own_request=public.self_request), slot), bundle)
                recover = next(m for m in public.self_request["active"][0]["moves"] if m["id"] == "recover")
                self.assertEqual(recover["pp"], 32 - (target - 1))
                counts.append(len(public.replay.public_lines))
        self.assertEqual(len(trace), 24)
        self.assertLess(counts[0], counts[2])
        self.assertLess(counts[2], counts[4])
        self.assertIsNone(env.terminal())


if __name__ == "__main__":
    unittest.main()
