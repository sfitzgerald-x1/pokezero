"""Fixed exploration registration/lifecycle; synthetic outcomes only."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import Mock, patch
import unittest

import collect_search_over_raw_a1_exploration as driver
import qualify_search_over_raw_a1 as a1
from pokezero.mcts_eval.search_over_raw import digest, panel_summary
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger
from tests import test_search_over_raw_a1 as fixtures
from pokezero.mcts_eval import paper_reference_parallel as parallel
from tests import test_paper_reference_parallel as pool_fixtures


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()
        self.exposure = self.output / "old.json"
        self.plan = self.output / "plan.txt"
        a1.save_new(self.exposure, dict(seeds=[123]))
        a1.save_new(self.plan, dict(synthetic_plan=True))

    def registration(self):
        excluded, bindings = driver.exposure_inventory([self.exposure])
        c = driver.cohort(excluded)
        c["input_hashes"] = {**bindings, str(self.plan): driver.sha256_file(self.plan)}
        return dict(schema=driver.SCHEMA, namespace=driver.NAMESPACE,
            phase_a_cohort=c, phase_a_cohort_sha256=digest(c), source_contract=driver.source_contract(c),
            configurations=[asdict(cfg) for cfg in a1.configurations()],
            seeds=c["panels"]["exploration"]["seeds"], fixture_seed=c["panels"]["exploration"]["seeds"][0],
            candidate_seat="p1", initial_dispatch_workers=6, required_independent_review=True,
            owned_receipts_protocol="active-generations-v1",
            retry_authorized=False, phase_a_admission=False, phase_b_authorized=False,
            scientific_strength_evidence=False, holdout_opened=False,
            qualifies_uninstrumented_runtime=False, representative_runtime_evidence=False,
            historical_attempt_reused=False, source_root=str(driver.ROOT), source_commit="synthetic",
            attempt_directory=str(self.output), excluded_seeds=excluded,
            excluded_seed_inventory_sha256=digest(excluded), exposure_registrations=bindings, plan=str(self.plan),
            input_sha256={**c["input_hashes"], **{str(p): driver.sha256_file(p) for p in driver.bound_files()}})

    def prepare_run(self, *, worker=False):
        r = self.registration()
        a1.save_new(self.output / "registration.json", r)
        a1.save_new(self.output / "registration-binding.json", dict(registration_sha256=digest(r)))
        PhaseALedger.create(self.output / "phase-a-ledger", r["phase_a_cohort"])
        review = self.output / "review.json"
        a1.save_new(review, dict(disposition=driver.REVIEW, registration_sha256=digest(r),
            source_commit=r["source_commit"], independent_reviewer="synthetic-other-agent",
            scientific_admission=False, holdout_authorized=False))
        if worker:
            a1.save_new(self.output / "supervision.json", dict(supervisor_pid=os.getppid(),
                registration_sha256=digest(r), review_sha256=driver.sha256_file(review)))
        return r, review

    def test_full_four_hundred_slots_and_disjoint_thirty_two_seed_panels_are_fixed(self):
        r = self.registration()
        c = r["phase_a_cohort"]
        self.assertEqual(c, driver.cohort(r["excluded_seeds"]) | {"input_hashes": c["input_hashes"]})
        exploration, validation = c["panels"]["exploration"], c["panels"]["validation"]
        self.assertEqual([len(p["seeds"]) for p in (exploration, validation)], [32, 32])
        self.assertEqual([len(p["root_slots"]) for p in (exploration, validation)], [200, 200])
        self.assertFalse(set(exploration["seeds"]) & set(validation["seeds"]))
        self.assertFalse((set(exploration["seeds"]) | set(validation["seeds"])) & set(r["excluded_seeds"]))
        self.assertTrue(set(a1.SEEDS) <= set(r["excluded_seeds"]))
        self.assertEqual(len(c["configurations"]), 37)
        self.assertFalse(c["execution_ready"])

    def test_collector_has_no_holdout_execution_path_and_full_one_thousand_cells(self):
        r = self.registration()
        self.assertEqual(set(r["source_contract"]["panels"]), {"exploration"})
        p = driver.new_progress(r)
        self.assertEqual((len(p["fixed_roster"]), len(p["continuation_roster"])), (1000, 8000))
        self.assertEqual(len(p["source_roster"]), 32)
        self.assertTrue(all(c["root_id"].startswith("exploration:") for c in p["fixed_roster"]))
        self.assertTrue(all(c["signed_outcome"] is None for c in p["continuation_roster"]))
        self.assertEqual([(c.arm, c.belief, c.seconds, c.workers) for c in a1.configurations()], [
            ("raw", "public", 1., 1), ("incumbent", "public", 1., 1),
            ("incumbent", "oracle", 1., 1), ("reference", "public", 1., 20),
            ("reference", "oracle", 1., 20)])

    def test_core_cohort_quota_resource_admission_or_input_drift_fails(self):
        r = self.registration()
        driver.validate_contract(r, self.output)
        for change in (dict(phase_a_admission=True), dict(holdout_opened=True),
                       dict(initial_dispatch_workers=20), dict(seeds=r["seeds"][:-1]),
                       dict(required_independent_review=False), dict(retry_authorized=True)):
            bad = deepcopy(r)
            bad.update(change)
            with self.assertRaisesRegex(ValueError, "core/cohort drift"):
                driver.validate_contract(bad, self.output)
        for name in ("phase_a_cohort", "source_contract"):
            bad = deepcopy(r)
            bad[name]["panels"]["exploration"]["root_slots"].pop()
            with self.assertRaisesRegex(ValueError, "core/cohort drift"):
                driver.validate_contract(bad, self.output)
        bad = deepcopy(r)
        bad["input_sha256"][str(Path(driver.__file__).resolve())] = "0"*64
        with self.assertRaisesRegex(ValueError, "driver/output drift"):
            driver.validate_contract(bad, self.output)

    def small_contract(self):
        contract = driver.source_contract(self.registration()["phase_a_cohort"])
        seeds = contract["panels"]["exploration"]["seeds"][:2]
        # One source has three slots, another has one; synthetic fixture only.
        contract["panels"]["exploration"] = dict(seeds=seeds, root_slots=[
            dict(root_id=f"exploration:{seed}:{slot}", source_seed=seed, root_slot=slot)
            for seed, count in zip(seeds, (3, 1)) for slot in range(count)])
        return contract

    def source(self, contract, seed, count):
        slots = [r for r in contract["panels"]["exploration"]["root_slots"] if r["source_seed"] == seed]
        indices = list(range(1, count+1))
        selected = a1.select_source_requests(contract["namespace"], seed, indices, len(slots))
        records = {i: dict(seed=seed, recorded_action_index=0, source_request_index=i) for i in indices}
        eligible = [dict(source_request_index=i, public_record_sha256=digest(records[i])) for i in indices]
        return dict(contract_sha256=digest(contract), source_seed=seed, panel="exploration", status="COMPLETE",
            source_terminal_complete=True, source_policy="raw_argmax_both_seats", eligible_requests=count,
            eligible_public_records=eligible, eligible_catalog_sha256=digest(eligible),
            requested_root_slots=len(slots), missing_root_ids=[s["root_id"] for s in slots[len(selected):]],
            roots=[dict(root_id=s["root_id"], source_request_index=i, public_record=records[i],
                public_record_sha256=digest(records[i])) for s, i in zip(slots, selected)])

    def test_short_completed_sources_leave_missing_slots_and_cannot_redraw(self):
        contract = self.small_contract()
        seed = contract["panels"]["exploration"]["seeds"][0]
        source = self.source(contract, seed, 2)
        a1.validate_source(source, seed, contract=contract, panel="exploration")
        self.assertEqual(len(source["missing_root_ids"]), 1)
        source["missing_root_ids"] = []
        with self.assertRaisesRegex(ValueError, "no replacement"):
            a1.validate_source(source, seed, contract=contract, panel="exploration")

    def test_zero_eligible_source_slots_stay_missing_not_replaced(self):
        contract = self.small_contract()
        seed = contract["panels"]["exploration"]["seeds"][0]
        source = self.source(contract, seed, 0)
        a1.validate_source(source, seed, contract=contract, panel="exploration")
        self.assertEqual((source["roots"], len(source["missing_root_ids"])), ([], 3))
        source["status"] = "UNCERTAIN_SOURCE"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            a1.validate_source(source, seed, contract=contract, panel="exploration")

    def test_generic_collection_reconciles_variable_quotas_and_all_missing_aliases(self):
        helper = fixtures.StagedA1Tests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        contract = self.small_contract()
        p = driver.new_progress(dict(source_contract=contract, seeds=contract["panels"]["exploration"]["seeds"]))
        for seed in contract["panels"]["exploration"]["seeds"]:
            d = self.output / f"source-{seed}"
            d.mkdir()
            source = self.source(contract, seed, 2)
            a1.save_new(d / "source.json", source)
            p["source_roster"][str(seed)]["status"] = "COMPLETE"
            p["source_games_completed"] += 1
            for root in source["roots"]:
                kwargs, _, _ = helper.root_fixture()
                root_dir = d / f"root-{root['root_id'].rsplit(':', 1)[1]}"
                root_dir.mkdir()
                kwargs.update(root=root, progress=p, output=root_dir, namespace=contract["namespace"],
                    completion_status="COMPLETE_EXPLORATION")
                with patch("builtins.print"):
                    a1.collect_root(**kwargs)
        a1.verify_completion(self.output, p, contract=contract, panel="exploration",
            completion_status="COMPLETE_EXPLORATION")
        self.assertEqual((p["roots_completed"], p["selections_completed"], p["continuations_completed"]), (3, 15, 24))
        self.assertEqual(len(p["fixed_roster"]), 20)
        missing = next(c for c in p["fixed_roster"] if c["status"] == "UNSTARTED_UNCERTAIN")
        missing["contrast_interval"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "missing root was scored"):
            a1.verify_completion(self.output, p, contract=contract, panel="exploration",
                completion_status="COMPLETE_EXPLORATION")

    def test_equal_seed_summary_keeps_all_two_hundred_slots_uncertain(self):
        r = self.registration()
        cfg = a1.configurations()[1]
        summary = panel_summary(r["phase_a_cohort"], panel="exploration", configuration=cfg.identity,
            root_intervals={}, bootstrap_reps=1000)
        self.assertEqual(summary["identification_interval"], [-1., 1.])
        self.assertEqual(summary["uncertain_roots"], 200)

    def test_worker_failure_preserves_entire_roster_no_ledger_results_or_retry(self):
        r, review = self.prepare_run(worker=True)
        with patch.object(driver, "verify_inputs"), patch.object(a1, "measure", side_effect=ValueError("synthetic failure")) as runtime:
            self.assertEqual(driver.run(self.output, review), 1)
            terminal = json.loads((self.output / "terminal.json").read_text())
            self.assertEqual((terminal["exit_code"], len(terminal["fixed_roster"]), len(terminal["continuation_roster"])),
                (1, 1000, 8000))
            self.assertFalse(terminal["holdout_opened"])
            self.assertFalse(list((self.output / "phase-a-ledger/exploration").glob("*.json")))
            with self.assertRaisesRegex(ValueError, "no retry"):
                driver.run(self.output, review)
            self.assertEqual(runtime.call_count, 1)

    def test_deadline_cleanup_failure_is_still_exit124_and_never_reopens(self):
        _, review = self.prepare_run()
        child = Mock(pid=987654)
        child.wait.side_effect = subprocess.TimeoutExpired("synthetic", 1)
        with patch.object(driver, "verify_inputs"), patch.object(driver.subprocess, "Popen", return_value=child) as spawn, \
             patch.object(a1, "stop_owned", side_effect=OSError("synthetic cleanup failure")):
            self.assertEqual(driver.supervise(self.output, review), 124)
            receipt = json.loads((self.output / "supervision-terminal.json").read_text())
            self.assertTrue(receipt["wall_cap_reached"])
            self.assertEqual(receipt["cleanup_error_type"], "OSError")
            terminal = json.loads((self.output / "terminal.json").read_text())
            self.assertTrue(all(c["signed_outcome"] is None for c in terminal["continuation_roster"]))
            with self.assertRaisesRegex(ValueError, "no retry"):
                driver.supervise(self.output, review)
            self.assertEqual(spawn.call_count, 1)

    def test_engineering_clearance_cannot_authorize_exploration_or_open_holdout(self):
        r, review = self.prepare_run()
        old = driver.json.loads
        def wrong_review(payload):
            value = old(payload)
            if value.get("disposition") == driver.REVIEW:
                value["disposition"] = "CLEAR_ONE_STAGED_A1_ENGINEERING_ATTEMPT"
            return value
        with patch.object(driver, "verify_inputs"), patch.object(driver.json, "loads", side_effect=wrong_review):
            with self.assertRaisesRegex(ValueError, "independent exploration review"):
                driver.admit(self.output, review)
        self.assertFalse((self.output / "attempt.json").exists())
        ledger = PhaseALedger(self.output / "phase-a-ledger")
        with self.assertRaisesRegex(ValueError, "admission is incomplete"):
            ledger.validation_once(Mock())

    def test_same_pid_multiple_generations_never_overwrite_historical_receipts(self):
        ownership = dict(directory=str(self.output), controller_pid=os.getpid(),
            registration_sha256="synthetic", protocol="active-generations-v1")
        with patch.object(parallel, "owned_process_identity", return_value="synthetic birth"):
            first = parallel.record_owned_process(12345, ownership)
            second = parallel.record_owned_process(12345, ownership)
        self.assertNotEqual(first, second)
        self.assertEqual(len(list((self.output / "active").glob("group-*.json"))), 2)
        self.assertTrue(first.exists() and second.exists())

    def test_actual_two_worker_generations_are_reaped_and_only_active_links_retired(self):
        ownership = dict(directory=str(self.output), controller_pid=os.getpid(),
            registration_sha256="synthetic", protocol="active-generations-v1")
        with parallel.ParallelTrajectorySearch(pool_fixtures.ReferenceConfig(.5, 1),
                pool_fixtures.ToyRuntime, workers=2, owned_process_receipts=ownership) as pool:
            pool.search(pool_fixtures.Request(), pool_fixtures.ROOT, battle_id="synthetic", seed=1,
                trajectories_per_worker=1)
            self.assertEqual(len(list((self.output / "active").glob("group-*.json"))), 2)
        self.assertEqual(len(list((self.output / "active").glob("group-*.json"))), 0)
        launches = list(self.output.glob("group-*.json"))
        closes = list(self.output.glob("closed-group-*.json"))
        self.assertEqual((len(launches), len(closes)), (2, 2))
        self.assertTrue(all(json.loads(path.read_text())["worker_exit_code"] == 0 for path in closes))

    def test_cleanup_ignores_eight_thousand_historical_generations_queries_active_once(self):
        ownership = dict(directory=str(self.output), controller_pid=os.getpid(),
            registration_sha256="synthetic", protocol="active-generations-v1")
        active = self.output / "active"
        active.mkdir()
        # The glob itself is adversarial: history must never be scanned.
        row = dict(pid=987654, group=987654, birth_identity="new birth", generation="synthetic",
            controller_pid=987653, registration_sha256="synthetic", protocol="active-generations-v1")
        a1.save_new(active / "group-987654-synthetic.json", row)
        child = Mock(pid=987653)
        child.wait.return_value = -15
        with patch.object(a1.os, "killpg"), patch.object(parallel, "signal_owned_group") as signals, \
             patch.object(parallel, "owned_process_identity", side_effect=AssertionError("historical PID query")), \
             patch.object(parallel, "owned_process_identities", return_value={987654: "new birth"}) as identities:
            self.assertEqual(a1.stop_owned(child, 1, ownership), -15)
        identities.assert_called_once_with([987654])
        self.assertIn(987654, [call.args[0] for call in signals.call_args_list])

    def test_reused_active_pid_is_never_signalled(self):
        ownership = dict(directory=str(self.output), registration_sha256="synthetic", protocol="active-generations-v1")
        active = self.output / "active"
        active.mkdir()
        a1.save_new(active / "group-987654-synthetic.json", dict(pid=987654, group=987654,
            birth_identity="old birth", generation="synthetic", controller_pid=987653,
            registration_sha256="synthetic", protocol="active-generations-v1"))
        child = Mock(pid=987653)
        child.wait.return_value = -15
        with patch.object(a1.os, "killpg"), patch.object(parallel, "signal_owned_group") as signals, \
             patch.object(parallel, "owned_process_identities", return_value={987654: "new unrelated birth"}):
            with self.assertRaisesRegex(ValueError, "reused"):
                a1.stop_owned(child, 1, ownership)
        self.assertNotIn(987654, [call.args[0] for call in signals.call_args_list])

    def test_live_generation_cannot_be_marked_closed_or_removed(self):
        ownership = dict(directory=str(self.output), controller_pid=os.getpid(),
            registration_sha256="synthetic", protocol="active-generations-v1")
        with patch.object(parallel, "owned_process_identity", return_value="synthetic birth"):
            path = parallel.record_owned_process(12345, ownership)
        with patch.object(parallel, "live_owned_groups", return_value={12345}):
            with self.assertRaisesRegex(parallel.ReferenceRefusal, "live descendants"):
                parallel.close_owned_generations({0: path}, [Mock(pid=12345, exitcode=0)])
        self.assertTrue((self.output / "active" / path.name).exists())
        self.assertFalse(list(self.output.glob("closed-group-*.json")))

    def test_real_permission_failure_propagates_but_disappeared_group_race_is_safe(self):
        with patch.object(parallel.os, "killpg", side_effect=PermissionError("synthetic denial")), \
             patch.object(parallel, "live_owned_groups", return_value={12345}):
            with self.assertRaises(PermissionError):
                parallel.signal_owned_group(12345, 15)
        with patch.object(parallel.os, "killpg", side_effect=PermissionError("synthetic exited-group race")), \
             patch.object(parallel, "live_owned_groups", return_value=set()):
            parallel.signal_owned_group(12345, 15)

    def test_deadline_stops_real_detached_active_generation_not_historical_processes(self):
        ownership = dict(directory=str(self.output), registration_sha256="synthetic", protocol="active-generations-v1")
        child_code = "import os,sys,time; sys.stdin.readline(); os.setsid(); print(os.getpid(),flush=True); time.sleep(60)"
        parent_code = "\n".join([
            "import json,os,subprocess,sys,time",
            "from pokezero.mcts_eval.paper_reference_parallel import record_owned_process",
            "ownership=json.loads(sys.argv[1]); ownership['controller_pid']=os.getpid()",
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)",
            "record_owned_process(child.pid,ownership)",
            "child.stdin.write('owned launch\\n'); child.stdin.flush()",
            "print(child.stdout.readline().strip(),flush=True)",
            "time.sleep(60)"])
        parent = subprocess.Popen([sys.executable, "-B", "-c", parent_code, json.dumps(ownership), child_code],
            start_new_session=True, stdout=subprocess.PIPE, text=True)
        self.addCleanup(parent.stdout.close)
        self.addCleanup(lambda: parent.poll() is None and a1.stop_owned(parent, 1, ownership))
        nested_pid = int(parent.stdout.readline().strip())
        code, expired = a1.wait_bounded(parent, .05, 1, ownership)
        self.assertTrue(expired)
        self.assertNotEqual(code, 0)
        state = subprocess.run(["ps", "-p", str(nested_pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
        self.assertTrue(not state or state.startswith("Z"), state)


if __name__ == "__main__":
    unittest.main()
