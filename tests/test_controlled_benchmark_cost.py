from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pokezero.controlled_benchmark_cost as cost


class ControlledBenchmarkCostTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        showdown = self.root / "showdown/dist/sim"
        showdown.mkdir(parents=True)
        (showdown / "battle-stream.js").write_bytes(b"authored unused simulator")
        foulplay = self.root / "foulplay"
        foulplay.mkdir()
        (foulplay / "run.py").write_bytes(b"authored unused opponent")
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"authored not a model")
        python = str(Path(sys.executable).resolve())
        files = [Path(python), checkpoint, showdown / "battle-stream.js", foulplay / "run.py"]
        self.recipe = {"kind": cost.RECIPE, "python": python, "checkpoint": str(checkpoint),
                       "showdown_root": str(self.root / "showdown"), "foulplay_root": str(foulplay),
                       "foulplay_python": python, "node": python, "policy_mode": "raw", "seat": "p1",
                       "games": 2, "seed_start": 71, "max_decision_rounds": 250,
                       "timeout_seconds": 20, "native": None,
                       "input_sha256": {str(p.resolve()): cost._hash(p) for p in files}}

    def attempt(self, mode="success", directory=None):
        directory = directory or self.root / "attempt"
        fixture = Path(__file__).with_name("_controlled_benchmark_cost_fixture.py")
        with patch.object(cost, "_command", side_effect=lambda recipe, output:
                          [recipe["python"], str(fixture), str(output), mode]):
            return directory, cost.run_attempt(self.recipe, directory)

    def test_success_executes_fixed_child_and_reads_original_cells(self):
        directory, result = self.attempt()
        self.assertEqual(result["status"], "MEASURED_CHILD")
        self.assertTrue(result["child"]["wait4_reaped"])
        self.assertGreater(result["child"]["wall_seconds"], 0)
        self.assertGreater(result["child"]["cpu_seconds"], 0)
        self.assertEqual([r["seed"] for r in result["summary"]["cells"]], [71, 72])
        self.assertEqual(result, cost.read_attempt(directory))
        for name in ("complete_workflow_cost", "descendant_cleanup_complete", "scientific_qualified",
                     "loaded_native_extension_qualified", "phase_b_authorized"):
            self.assertIs(result[name], False)
        self.assertEqual(result["scientific_outcomes_accepted"], 0)
        self.assertIsNone(result["worker"]["cleanup_cpu_seconds"])

    def test_exact_child_cpu_includes_reaped_cpu_child_not_double_summed(self):
        _, result = self.attempt("cpu_child")
        self.assertEqual(result["status"], "MEASURED_CHILD")
        self.assertGreater(result["child"]["cpu_seconds"], .09)
        self.assertAlmostEqual(result["child"]["cpu_seconds"],
                               result["child"]["user_seconds"] + result["child"]["system_seconds"])

    def test_failure_after_one_game_retains_full_uncertain_roster(self):
        _, result = self.attempt("partial_failure")
        self.assertEqual(result["status"], "REFUSED")
        self.assertEqual(result["child"]["returncode"], 1)
        self.assertEqual(result["summary"]["reported_games"], 1)
        self.assertEqual([r["reported_score"] for r in result["summary"]["cells"]], [1., None])
        self.assertTrue(all(r["scientific_score_interval"] == [0., 1.]
                            for r in result["summary"]["cells"]))

    def test_cleanup_failure_cannot_bank_complete_progress_summary(self):
        _, result = self.attempt("cleanup_failure")
        self.assertTrue(result["summary"]["uncapped_complete_summary"])
        self.assertEqual(result["status"], "REFUSED")
        self.assertEqual(result["child"]["returncode"], 1)

    def test_startup_failure_records_reaped_cost_without_summary(self):
        _, result = self.attempt("startup_failure")
        self.assertEqual(result["status"], "REFUSED")
        self.assertEqual(result["worker"]["phase"], "preflight")
        self.assertIsNone(result["summary"])
        self.assertIsNone(result["worker"]["benchmark_call_seconds"])

    def test_missing_summary_cannot_promote_exit_zero(self):
        _, result = self.attempt("missing_summary")
        self.assertEqual(result["child"]["returncode"], 0)
        self.assertEqual(result["status"], "REFUSED")

    def test_capped_game_keeps_score_unknown_despite_bridge_half_score_law(self):
        _, result = self.attempt("capped")
        self.assertEqual(result["status"], "REFUSED")
        self.assertIsNone(result["summary"]["cells"][0]["reported_score"])

    def test_malformed_original_seed_prefix_refuses_reading(self):
        _, result = self.attempt("malformed_summary")
        self.assertEqual(result["status"], "REFUSED")
        self.assertEqual(result["summary_error"], "CostRefusal")

    def test_live_separate_session_is_cleaned_before_owner_reap(self):
        directory, result = self.attempt("escaped_child")
        self.assertEqual(result["status"], "REFUSED")
        self.assertEqual(result["worker"]["error_type"], "OwnedChildStillLive")
        self.assertTrue(result["child"]["forced_cleanup"])
        self.assertEqual(result["child"]["returncode"], -9)
        self.assertEqual(result["worker"]["returncode"], 1)
        row = json.loads((directory / "children.jsonl").read_text().splitlines()[0])
        self.assertIsNone(cost._parent_pid(row["pid"]))

    def test_timeout_is_terminal_no_retry_and_owned_group_cleanup(self):
        self.recipe["timeout_seconds"] = 1.5
        directory, result = self.attempt("timeout")
        self.assertEqual(result["status"], "REFUSED")
        self.assertTrue(result["child"]["timed_out"])
        self.assertTrue(result["child"]["wait4_reaped"])
        self.assertEqual(result["retries"], 0)
        self.assertLess(result["child"]["wall_seconds"], 6)
        rows = (directory / "children.jsonl").read_text().splitlines()
        self.assertTrue(rows, "fixture must actually create the isolated child before timeout")
        self.assertTrue(all(cost._parent_pid(json.loads(row)["pid"]) is None for row in rows))

    def test_launch_failure_has_durable_terminal_receipt(self):
        with patch.object(cost, "_command", return_value=["/nonexistent-authored-benchmark-python"]):
            result = cost.run_attempt(self.recipe, self.root / "attempt")
        self.assertEqual(result["status"], "REFUSED")
        self.assertFalse(result["child"]["wait4_reaped"])
        self.assertIsNone(result["child"]["returncode"])
        self.assertIsNone(result["child"]["cpu_seconds"])
        self.assertTrue((self.root / "attempt/terminal.json").is_file())

    def test_input_drift_after_execution_retains_files_and_refuses_readback(self):
        with self.assertRaisesRegex(cost.CostRefusal, "binding"):
            self.attempt("input_drift")
        self.assertTrue((self.root / "attempt/summary.json").is_file())
        self.assertTrue((self.root / "attempt/terminal.json").is_file())

    def test_bound_summary_cannot_be_substituted(self):
        directory, _ = self.attempt()
        (directory / "summary.json").write_text("{}")
        with self.assertRaisesRegex(cost.CostRefusal, "changed"):
            cost.read_attempt(directory)

    def test_new_source_file_is_detected_in_frozen_inventory(self):
        directory, _ = self.attempt()
        original = cost._source_bindings()
        with patch.object(cost, "_source_bindings", return_value={**original, "/authored/new.py": "0" * 64}):
            with self.assertRaisesRegex(cost.CostRefusal, "source drift"):
                cost.read_attempt(directory)

    def test_existing_output_never_reused_or_overwritten(self):
        directory, _ = self.attempt()
        before = cost._hash(directory / "terminal.json")
        with self.assertRaises(FileExistsError):
            cost.run_attempt(self.recipe, directory)
        self.assertEqual(before, cost._hash(directory / "terminal.json"))

    def test_closed_recipe_and_boolean_numbers_refuse_before_allocation(self):
        for key, value in (("games", True), ("timeout_seconds", True), ("timeout_seconds", float("nan")),
                           ("seed_start", -1), ("policy_mode", "trajectory-reference"), ("seat", "p3")):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.recipe)
                bad[key] = value
                with self.assertRaises(cost.CostRefusal):
                    cost.run_attempt(bad, self.root / "attempt")
                self.assertFalse((self.root / "attempt").exists())
        self.recipe["argv"] = ["arbitrary command forbidden"]
        with self.assertRaises(cost.CostRefusal):
            cost.validate_recipe(self.recipe)

    def test_prospective_input_hash_mismatch_refuses_before_allocation(self):
        key = self.recipe["checkpoint"]
        self.recipe["input_sha256"][key] = "0" * 64
        with self.assertRaises(cost.CostRefusal):
            cost.run_attempt(self.recipe, self.root / "attempt")
        self.assertFalse((self.root / "attempt").exists())

    def test_valid_recipe_field_cannot_drift_after_prospective_launch(self):
        fixture = Path(__file__).with_name("_controlled_benchmark_cost_fixture.py")
        def command(recipe, output):
            changed = cost._json(output / "recipe.json")
            changed["seed_start"] += 1
            (output / "recipe.json").write_text(json.dumps(changed))
            return [recipe["python"], str(fixture), str(output), "success"]
        with patch.object(cost, "_command", side_effect=command):
            with self.assertRaisesRegex(cost.CostRefusal, "prospective launch"):
                cost.run_attempt(self.recipe, self.root / "attempt")
        terminal = cost._json(self.root / "attempt/worker-terminal.json")
        self.assertEqual(terminal["phase"], "preflight")
        self.assertIsNone(terminal["benchmark_call_seconds"])
        self.assertFalse((self.root / "attempt/summary.json").exists())

    def test_launch_argv_cannot_drift_independently_of_valid_recipe(self):
        fixture = Path(__file__).with_name("_controlled_benchmark_cost_fixture.py")
        def command(recipe, output):
            launch = cost._json(output / "launch.json")
            launch["argv"].append("--allow-unregistered-change")
            (output / "launch.json").write_text(json.dumps(launch))
            return [recipe["python"], str(fixture), str(output), "success"]
        with patch.object(cost, "_command", side_effect=command):
            with self.assertRaisesRegex(cost.CostRefusal, "prospective launch"):
                cost.run_attempt(self.recipe, self.root / "attempt")
        self.assertFalse((self.root / "attempt/summary.json").exists())

    def test_stopped_event_is_not_terminal_reap_evidence(self):
        self.assertFalse(cost._terminal_event(SimpleNamespace(si_code=os.CLD_STOPPED)))
        self.assertFalse(cost._terminal_event(None))
        self.assertTrue(cost._terminal_event(SimpleNamespace(si_code=os.CLD_EXITED)))
        self.assertTrue(cost._terminal_event(SimpleNamespace(si_code=os.CLD_KILLED)))

    def test_group_identity_checks_follow_confirmed_owner_freeze(self):
        import signal
        order = []
        def stop(owner):
            order.append("stop-confirmed")
            return True
        def clean(directory, owner, sig):
            self.assertEqual(order, ["stop-confirmed"])
            order.append("owned-groups")
            return [92]
        with patch.object(cost, "_stop_owner", side_effect=stop), \
                patch.object(cost, "_clean_registered_groups", side_effect=clean), \
                patch.object(os, "waitid", return_value=None), \
                patch.object(os, "killpg", side_effect=lambda *_: order.append("owner-signal")), \
                patch.object(os, "kill", side_effect=lambda *_: order.append("owner-resume")):
            self.assertEqual(cost._terminate_owned(self.root, 91, signal.SIGTERM), [92])
        self.assertEqual(order, ["stop-confirmed", "owned-groups", "owner-signal", "owner-resume"])

    def test_owner_exit_before_stop_confirmation_never_signals_registered_groups(self):
        import signal
        with patch.object(cost, "_stop_owner", return_value=False), \
                patch.object(cost, "_clean_registered_groups") as clean, \
                patch.object(os, "waitid", return_value=SimpleNamespace(si_code=os.CLD_EXITED)), \
                patch.object(os, "killpg") as kill:
            self.assertEqual(cost._terminate_owned(self.root, 91, signal.SIGKILL), [])
        clean.assert_not_called()
        kill.assert_not_called()

    def test_reparented_or_unrelated_registered_pid_never_authorizes_group_signal(self):
        import signal
        (self.root / "children.jsonl").write_text(json.dumps({"owner": 91, "pid": 92,
                                                             "new_session": True}) + "\n")
        with patch.object(os, "getpgid", return_value=92), \
                patch.object(cost, "_parent_pid", return_value=1000), patch.object(os, "killpg") as kill:
            self.assertEqual(cost._clean_registered_groups(self.root, 91, signal.SIGKILL), [])
        kill.assert_not_called()

    def test_duplicate_json_and_nonfinite_tokens_refuse(self):
        path = self.root / "bad.json"
        for value in ('{"games":1,"games":2}', '{"seconds":NaN}'):
            path.write_text(value)
            with self.assertRaises(cost.CostRefusal):
                cost._json(path)

    def test_exact_arguments_freeze_resources_no_fallback_or_retry(self):
        args = cost.bridge_arguments(self.recipe, self.root / "summary.json")
        self.assertEqual(args[args.index("--search-time-ms") + 1], "1000")
        self.assertIn("--no-search-fallback", args)
        self.assertNotIn("--opponent-crash-retries", args)
        self.assertEqual(args[args.index("--device") + 1], "cpu")
        self.assertEqual(cost._command(self.recipe, self.root)[1:3],
                         ["-m", "pokezero.controlled_benchmark_cost"])

    def test_native_fixed_work_is_consumed_and_not_claimed_as_wall_ceiling(self):
        for name in ("model.pt", "tables.json"):
            path = self.root / name
            path.write_bytes(b"authored unused native input")
            self.recipe["input_sha256"][str(path)] = cost._hash(path)
        self.recipe["policy_mode"] = "engine-mcts"
        self.recipe["native"] = {"model": str(self.root / "model.pt"), "tables": str(self.root / "tables.json"),
                                 "depth": 6, "sims": 1024, "batch": 16, "worlds": 4,
                                 "workers": 1, "cpuct": 1.4, "opponent_priors": True}
        _, result = self.attempt()
        self.assertEqual(result["status"], "MEASURED_CHILD")
        self.assertEqual(result["summary"]["search_statistics"]["sims"], 1024)
        self.assertFalse(result["loaded_native_extension_qualified"])
        self.recipe["native"]["workers"] = 2
        with self.assertRaises(cost.CostRefusal):
            cost.validate_recipe(self.recipe)

    def test_cli_read_returns_nonzero_for_failed_attempt(self):
        directory, _ = self.attempt("cleanup_failure")
        with patch("builtins.print"):
            self.assertEqual(cost.main(["read", "--output", str(directory)]), 1)


if __name__ == "__main__":
    unittest.main()
