"""Create-only timing plumbing; no champion selections or scientific outcomes."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_search_over_raw_synthetic as driver
from _showdown_root import requires_showdown, showdown_root
from pokezero.engine_search import EngineSearchFallbackError
from pokezero.mcts_eval.search_over_raw import ENGINEERING_EXCLUDED_SEEDS, digest


def registration(output):
    return dict(schema=driver.SCHEMA, namespace="7636a4d9-8214-4a31-b806-4338170ddde8",
        configurations=[asdict(driver.CONFIGURATION)], allocation=driver.ALLOCATION.copy(),
        native_configuration=driver.NATIVE_CONFIGURATION.copy(),
        seeds=[driver.FIXTURE_SEED], fixture_seed=driver.FIXTURE_SEED, candidate_seat="p1",
        retry_authorized=False, phase_a_admission=False, phase_b_authorized=False,
        scientific_strength_evidence=False, representative_runtime_evidence=False, deployable=False,
        fixture={}, fixture_sha256=digest({}), attempt_directory=str(output.resolve()),
        source_root=str(driver.ROOT), input_sha256={str(Path(driver.__file__).resolve()):
            driver.digest_file(Path(driver.__file__))})


class SyntheticTimingTests(unittest.TestCase):
    def test_fresh_fixture_seed_is_excluded_from_all_future_panels(self):
        self.assertIn(driver.FIXTURE_SEED, ENGINEERING_EXCLUDED_SEEDS)
        self.assertNotEqual(driver.FIXTURE_SEED, 2026101009)

    def test_full_allocation_and_raw_terminal_semantics_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            r = registration(Path(directory))
            driver.validate_contract(r)
            for field in ("worlds", "search_sims", "search_batch", "search_depth",
                          "model_decision_time_ms", "rollout_max_plies", "rollout_count",
                          "model_world_workers", "strict_fallbacks", "model_leaf_override"):
                changed = registration(Path(directory))
                changed["native_configuration"][field] = None
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    driver.validate_contract(changed)

    def test_oracle_budget_or_evaluator_cannot_change(self):
        for key, value in (("belief", "public"), ("seconds", 3.), ("leaf", "hp_fraction"), ("workers", 2)):
            r = registration(Path("fixture"))
            r["configurations"][0][key] = value
            with self.assertRaisesRegex(ValueError, "contract changed"):
                driver.validate_contract(r)

    def test_no_scientific_admission_retry_or_representative_cost_claim(self):
        for field in ("retry_authorized", "phase_a_admission", "phase_b_authorized",
                      "scientific_strength_evidence", "representative_runtime_evidence", "deployable"):
            r = registration(Path("fixture"))
            r[field] = True
            with self.assertRaisesRegex(ValueError, "contract changed"):
                driver.validate_contract(r)

    def test_fixture_seed_and_hash_drift_refuse(self):
        for field, value in (("fixture_seed", 2026101009), ("seeds", [2026101009]),
                             ("fixture", {"changed": True}), ("schema", "opening-benchmark"),
                             ("namespace", "d84ec07c-0a49-4d57-89b4-ad42d19cbe18")):
            r = registration(Path("fixture"))
            r[field] = value
            with self.assertRaisesRegex(ValueError, "contract changed"):
                driver.validate_contract(r)

    def test_executing_checkout_and_driver_hash_are_mandatory(self):
        for key, value in (("source_root", "other"), ("input_sha256", {})):
            r = registration(Path("fixture"))
            r[key] = value
            with self.assertRaisesRegex(ValueError, "executing source"):
                driver.validate_contract(r)

    def setup_run(self, output):
        driver.save_new(output / "registration.json", registration(output))

    def test_success_is_timing_only_and_claim_precedes_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.setup_run(output)
            def measured(r, target, progress):
                self.assertTrue((target / "attempt.json").exists())
                return dict(elapsed_seconds=10.2, exceeded_nominal_seconds=True)
            with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=measured):
                self.assertEqual(driver.run(output), 0)
            receipt = json.loads((output / "terminal.json").read_text())
            self.assertEqual(receipt["status"], "COMPLETE_SYNTHETIC_TIMING_ONLY")
            self.assertEqual(receipt["terminal_games"], 0)
            self.assertFalse(receipt["phase_a_admission"])
            self.assertTrue(receipt["exceeded_nominal_seconds"])

    def test_refusal_retains_diagnostic_and_attempt_cannot_repeat(self):
        diagnostic = dict(schema="engine-search-fallback-refusal-v1",
            diagnostic_only_not_policy_input=True, reason="model_time_budget_exhausted",
            engine_mcts={"raw_leaf_invocations": [{"completed_rollouts": 0}]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.setup_run(output)
            def refused(r, target, progress):
                progress.update(stage="select_once", adapter_failure={"status": "UNCERTAIN_REFUSED"})
                raise EngineSearchFallbackError("fixture deadline", diagnostic=diagnostic)
            with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=refused) as measured:
                self.assertEqual(driver.run(output), 1)
                before = (output / "terminal.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    driver.run(output)
                measured.assert_called_once()
                self.assertEqual(before, (output / "terminal.json").read_bytes())
            receipt = json.loads(before)
            self.assertEqual(receipt["engine_fallback_diagnostic"], diagnostic)
            self.assertEqual(receipt["stage"], "select_once")
            self.assertEqual(receipt["exit_code"], 1)
            self.assertFalse(receipt["retry_authorized"])

    def test_drift_refuses_before_attempt_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.setup_run(output)
            with patch.object(driver, "verify_inputs", side_effect=ValueError("input drift")), \
                    patch.object(driver, "measure") as measured:
                with self.assertRaisesRegex(ValueError, "input drift"):
                    driver.run(output)
                measured.assert_not_called()
                self.assertFalse((output / "attempt.json").exists())

    def test_copied_registration_cannot_create_a_second_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            r = registration(output / "different")
            driver.save_new(output / "registration.json", r)
            with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as measured:
                with self.assertRaisesRegex(ValueError, "directory changed"):
                    driver.run(output)
                measured.assert_not_called()
                self.assertFalse((output / "attempt.json").exists())


@requires_showdown()
class AuthoredFixtureTests(unittest.TestCase):
    def test_full_teams_roundtrip_through_actual_requests_without_search(self):
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.mcts_eval.search_over_raw_oracle import opening_team
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.showdown_fixture import pack_team
        source = load_gen3_randbat_source_cached(showdown_root())
        authored = driver.fixture(source)
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True))
        self.addCleanup(env.close)
        env.reset_with_start_override(seed=driver.FIXTURE_SEED, start_override=BattleStartOverride(
            player_teams=authored["player_teams"], observation_format_id="gen3randombattle"))
        snapshot = env.snapshot_actionable_boundary()
        self.assertIsNone(snapshot.terminal)
        for seat in ("p1", "p2"):
            self.assertEqual(pack_team(opening_team(snapshot.first_requests[seat], seat, source)),
                authored["player_teams"][seat])
            rows = snapshot.first_requests[seat]["side"]["pokemon"]
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(len(row["moves"]), 4)
                hp, maximum = row["condition"].split()[0].split("/")
                self.assertEqual(hp, maximum)
                self.assertGreater(int(hp), 0)


if __name__ == "__main__":
    unittest.main()
