"""Real disposable owners and fixed prospective specs; never scientific work."""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import pickle
import signal
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_attempt_cost import AttemptCostSpec,reconcile_attempt_cost_roster
from pokezero.mcts_eval.search_over_raw_resources import resource_source_hash
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
import pokezero.mcts_eval.search_over_raw_attempt_cost as costs
from test_search_over_raw_supervision import SyntheticJob,BINDING
from test_search_over_raw_nested_ownership import NestedJob,BINDING as NESTED_BINDING


def save(path,value):Path(path).write_text(json.dumps(value))


class AttemptCostTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()

    def spec(self,job,directory,binding=BINDING,parent=None,watchdog=1800):
        return AttemptCostSpec(str(directory),hashlib.sha256(pickle.dumps(job)).hexdigest(),binding,
            "fixed-work","selection",FixedWorkDiagnosticContract().receipt(),resource_source_hash(),watchdog,
            None if parent is None else str(parent))

    def read(self,specs):
        return reconcile_attempt_cost_roster(tuple(specs),attempt_root=self.root,
            roster_sha256=digest([asdict(spec) for spec in specs]))

    def standalone(self,mode="complete",limit=None,label="attempt"):
        job=SyntheticJob(mode);directory=self.root/label;spec=self.spec(job,directory,watchdog=limit or 1800)
        terminal=supervise_fixed_work(job,directory=directory,binding_sha256=BINDING,
            verify_admission=lambda:True,_synthetic_watchdog_seconds=limit)
        return spec,terminal

    def nested(self,mode="cpu",limit=None):
        outer=self.root/"outer";child=self.root/"child"
        job=NestedJob(str(self.root),mode)
        child_mode={"cpu":"cpu_child","hang_nested":"hang","swallow_failure":"fail"}.get(mode,"complete")
        specs=[self.spec(job,outer,NESTED_BINDING,watchdog=limit or 1800),self.spec(SyntheticJob(child_mode),
            child,NESTED_BINDING,outer,watchdog=5. if mode=="swallow_failure" else 1800)]
        terminal=supervise_fixed_work(job,directory=outer,binding_sha256=NESTED_BINDING,
            verify_admission=lambda:True,_synthetic_watchdog_seconds=limit)
        return specs,terminal

    def test_actual_normal_failure_and_unstarted_fixed_denominator(self):
        complete,first=self.standalone(label="complete")
        failed,second=self.standalone("fail",label="failed")
        unstarted=self.spec(SyntheticJob(),self.root/"unstarted")
        before=costs._inventory([self.root])
        with patch("subprocess.Popen") as spawn,patch("os.wait4") as wait:
            result=self.read([complete,failed,unstarted])
        spawn.assert_not_called();wait.assert_not_called()
        self.assertEqual(costs._inventory([self.root]),before)
        self.assertEqual(result["required_attempt_slots"],3)
        self.assertEqual([r["state"] for r in result["attempts"]],
            ["REAPED_SUCCESS_COST_OBSERVED","REAPED_FAILURE_COST_OBSERVED","UNSTARTED_UNOBSERVED"])
        self.assertEqual(result["attempts"][1]["actual_worker_exit_code"],1)
        self.assertEqual(result["attempts"][1]["original_supervisor_exit_code"],1)
        self.assertIsNone(result["attempts"][1]["phase_costs"])
        self.assertIsNone(result["attempts"][2]["job_cpu_seconds_including_reaped_children"])
        self.assertAlmostEqual(result["known_nonoverlapping_job_cpu_seconds"],
            first["cpu_seconds_including_reaped_children"]+second["cpu_seconds_including_reaped_children"])
        self.assertFalse(result["complete_workflow_cost"])
        self.assertIsNone(result["total_workflow_cpu_seconds"])

    def test_actor_annotated_kind_retains_exact_success_failure_and_status_binding(self):
        from pokezero.mcts_eval.search_over_raw_actor_timed_run import KIND
        from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedRunContract
        from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
        specs = []; terminals = []
        for mode in ("complete", "fail"):
            job = SyntheticJob(mode); directory = self.root / mode
            spec = replace(self.spec(job, directory), kind=KIND, contract=NativeTimedRunContract().receipt())
            terminal = _supervise_owned_job(job, directory=directory, binding_sha256=BINDING,
                verify_admission=lambda:True, kind=KIND, contract=NativeTimedRunContract(), work_phase="selection",
                validate_result=lambda result:None)
            specs.append(spec); terminals.append(terminal)
        result = self.read(specs)
        self.assertEqual(result["evidence_errors"], [])
        self.assertEqual([row["state"] for row in result["attempts"]],
            ["REAPED_SUCCESS_COST_OBSERVED", "REAPED_FAILURE_COST_OBSERVED"])
        self.assertEqual(terminals[0]["status"], "ACTOR_ANNOTATED_NATIVE_TIMED_RUN_COMPLETE")
        self.assertFalse(result["work_validated"]); self.assertFalse(result["complete_workflow_cost"])
        path = self.root / "complete" / "terminal.json"; original = path.read_bytes()
        changed = json.loads(original); changed["status"] = "NATIVE_TIMED_RUN_COMPLETE"
        save(path, changed)
        self.assertEqual(self.read(specs)["attempts"][0]["state"], "MALFORMED_EVIDENCE")
        path.write_bytes(original)
        self.assertEqual(self.read(specs)["attempts"][0]["state"], "REAPED_SUCCESS_COST_OBSERVED")

    def test_unknown_actor_kind_is_not_an_open_family_admission(self):
        spec = self.spec(SyntheticJob(), self.root / "unstarted")
        for kind in ("actor-annotated-native-timed-run-extra", "actor-annotated-native-run", "arbitrary-run"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "typed cost-attempt"):
                replace(spec, kind=kind)
        self.assertFalse((self.root / "unstarted").exists())

    def test_actual_nested_parent_cpu_not_added_to_child_or_nested_observers_again(self):
        specs,terminal=self.nested();result=self.read(specs)
        self.assertEqual(result["evidence_errors"],[],result)
        self.assertEqual(result["delegation_errors"],[],result)
        self.assertEqual(result["measured_cpu_frontier"],[specs[0].directory])
        self.assertEqual(result["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        self.assertGreater(result["attempts"][1]["job_cpu_seconds_including_reaped_children"],.1)
        self.assertFalse(result["nested_supervisor_or_sampler_cpu_added_again"])
        self.assertEqual(len(result["outer_observer_components"]),1)
        self.assertFalse(result["job_wall_exposure_is_campaign_elapsed"])

    def test_swallowed_nested_failure_cost_survives_without_success_promotion(self):
        specs,terminal=self.nested("swallow_failure");result=self.read(specs)
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(result["attempts"][0]["state"],"REAPED_FAILURE_COST_OBSERVED",result)
        self.assertEqual(result["attempts"][1]["state"],"REAPED_FAILURE_COST_OBSERVED",result)
        self.assertEqual(result["delegation_errors"],[],result)
        self.assertEqual(result["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        self.assertFalse(result["attempts"][0]["normal_descendant_closure_validated"])
        outer=result["attempts"][0]
        self.assertEqual(outer["actual_worker_exit_code"],-signal.SIGKILL)
        self.assertEqual(outer["worker_reported_exit_code"],1)
        self.assertIs(outer["worker_declared_cleanup_pending"],True)
        self.assertFalse(outer["measurement_window_cpu_wall_complete"])
        self.assertFalse(result["work_validated"])
        child = json.loads((Path(specs[1].directory)/"terminal.json").read_bytes())
        child_worker = json.loads((Path(specs[1].directory)/"worker-result.json").read_bytes())
        self.assertIs(child["watchdog_reached"], False)
        self.assertEqual(child_worker["error_type"], "RuntimeError")

    def test_forced_pending_cleanup_keeps_kernel_exit_and_rejects_false_worker_intent(self):
        specs,terminal=self.nested("swallow_failure")
        path=Path(specs[0].directory)/"worker-result.json"
        original=path.read_bytes();saved=json.loads(original)
        self.assertEqual(saved["exit_code"],1)
        self.assertIs(saved["cleanup_pending"],True)
        self.assertEqual(terminal["worker_exit_code"],-signal.SIGKILL)
        for key,value in (("cleanup_pending",False),("cleanup_pending",1),("exit_code",True),
                ("exit_code",0),("exit_code",7),("exit_code",-signal.SIGKILL),
                ("error_type",None),("error_type",""),("result",{})):
            with self.subTest(key=key,value=value):
                changed=dict(saved);changed[key]=value;save(path,changed)
                result=self.read(specs)
                self.assertEqual(result["attempts"][0]["state"],"MALFORMED_EVIDENCE")
                self.assertFalse(result["complete_workflow_cost"])
                self.assertIsNone(result["total_workflow_cpu_seconds"])
        changed=dict(saved);changed.pop("result");save(path,changed)
        self.assertEqual(self.read(specs)["attempts"][0]["state"],"MALFORMED_EVIDENCE")
        path.write_bytes(original)
        result=self.read(specs)
        self.assertEqual(result["evidence_errors"],[],result)
        self.assertEqual(result["delegation_errors"],[],result)
        self.assertEqual(result["attempts"][0]["actual_worker_exit_code"],-signal.SIGKILL)
        self.assertEqual(result["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])

    def test_actual_watchdog_keeps_reaped_nonzero_usage_and_failure(self):
        spec,terminal=self.standalone("hang",limit=.4)
        result=self.read([spec]);row=result["attempts"][0]
        self.assertEqual(terminal["supervisor_exit_code"],124)
        self.assertEqual(row["original_supervisor_exit_code"],124,result)
        self.assertEqual(row["actual_worker_exit_code"],terminal["worker_exit_code"])
        self.assertGreaterEqual(row["job_cpu_seconds_including_reaped_children"],0)
        self.assertFalse(row["measurement_window_cpu_wall_complete"])
        self.assertFalse(result["retries_permitted"])

    def test_missing_terminal_keeps_original_slot_unknown_never_polls_or_infers_exit(self):
        spec,terminal=self.standalone();(Path(spec.directory)/"terminal.json").unlink()
        with patch("os.wait4") as wait,patch("pokezero.mcts_eval.paper_reference_parallel.owned_process_identity") as identity:
            value=self.read([spec])
        wait.assert_not_called();identity.assert_not_called()
        self.assertEqual(value["attempts"][0]["state"],"LAUNCHED_TERMINAL_UNOBSERVED")
        self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])

    def test_actual_prelaunch_failure_retains_supervisor_cost_without_invented_worker(self):
        job=SyntheticJob();directory=self.root/"prelaunch";spec=self.spec(job,directory)
        with patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
            terminal=supervise_fixed_work(job,directory=directory,binding_sha256=BINDING,verify_admission=lambda:True)
        result=self.read([spec]);row=result["attempts"][0]
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(row["state"],"PRELAUNCH_FAILURE_COST_OBSERVED",result)
        self.assertEqual(row["supervisor_thread_cpu_seconds"],terminal["supervisor_thread_cpu_seconds"])
        self.assertEqual(row["elapsed_job_wall_seconds"],terminal["elapsed_wall_seconds"])
        self.assertIsNone(row["job_cpu_seconds_including_reaped_children"])
        self.assertIsNone(result["known_nonoverlapping_job_cpu_seconds"])

    def test_nested_watchdog_keeps_known_parent_and_unobserved_child_usage_separate(self):
        specs,terminal=self.nested("hang_nested",limit=1.5);result=self.read(specs)
        self.assertEqual(terminal["supervisor_exit_code"],124)
        self.assertEqual(result["delegation_errors"],[],result)
        self.assertEqual(result["attempts"][1]["state"],"LAUNCHED_TERMINAL_UNOBSERVED",result)
        self.assertIsNone(result["attempts"][1]["job_cpu_seconds_including_reaped_children"])
        self.assertEqual(result["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        self.assertFalse(result["complete_workflow_cost"])

    def test_effective_watchdog_and_conflicting_memory_error_cannot_drift(self):
        spec,terminal=self.standalone()
        for field,value in (("watchdog_seconds",False),("watchdog_seconds",1801),
                ("maxrss_units","seconds"),("resource_measurement_error_type","ValueError")):
            with self.subTest(field=field,value=value):
                changed=dict(terminal);changed[field]=value;save(Path(spec.directory)/"terminal.json",changed)
                self.assertEqual(self.read([spec])["attempts"][0]["state"],"MALFORMED_EVIDENCE")
        save(Path(spec.directory)/"terminal.json",terminal)
        self.assertEqual(self.read([spec])["evidence_errors"],[])

    def test_original_spec_drift_malformed_receipt_and_missing_claim_are_visible(self):
        spec,terminal=self.standalone();changed=replace(spec,job_sha256="b"*64)
        value=self.read([changed]);self.assertEqual(value["attempts"][0]["state"],"MALFORMED_EVIDENCE")
        self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])
        path=Path(spec.directory)/"terminal.json";path.write_text('{"binding_sha256":false,"binding_sha256":true}')
        value=self.read([spec]);self.assertEqual(len(value["evidence_errors"]),1)
        (Path(spec.directory)/"claim.json").unlink()
        value=self.read([spec]);self.assertEqual(value["required_attempt_slots"],1)
        self.assertEqual(value["attempts"][0]["state"],"MALFORMED_EVIDENCE")

    def test_omitted_real_delegation_and_malformed_parent_join_never_summed(self):
        specs,terminal=self.nested()
        value=self.read(specs[:1]);self.assertTrue(value["delegation_errors"])
        self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])
        registry=Path(specs[0].directory)/"delegated-groups"
        path=next(registry.glob("child-*.json"));row=json.loads(path.read_text());row["parent_binding"]="b"*64;save(path,row)
        value=self.read(specs);self.assertTrue(value["delegation_errors"])
        self.assertEqual(value["required_attempt_slots"],2)

    def test_declared_parent_cycle_outside_namespace_or_duplicate_slot_refuse(self):
        first=self.spec(SyntheticJob(),self.root/"first")
        second=self.spec(SyntheticJob(),self.root/"second",parent=self.root/"first")
        with self.assertRaisesRegex(ValueError,"distinct original attempts"):self.read([first,first])
        with self.assertRaisesRegex(ValueError,"parent omitted"):self.read([second])
        first=replace(first,parent_directory=second.directory)
        with self.assertRaisesRegex(ValueError,"cycle/depth"):self.read([first,second])
        outside=self.spec(SyntheticJob(),self.root.parent/"never-created-cost-fixture")
        with self.assertRaisesRegex(ValueError,"fixed canonical namespace"):self.read([outside])

    def test_late_memory_reader_mutation_is_rejected_against_original_inventory(self):
        spec,terminal=self.standalone();actual=costs.validate_memory_receipt
        def validate(*args,**kwargs):
            result=actual(*args,**kwargs);path=Path(spec.directory)/"terminal.json"
            path.write_bytes(path.read_bytes()+b" ");return result
        with patch.object(costs,"validate_memory_receipt",side_effect=validate):
            with self.assertRaisesRegex(ValueError,"changed during reconciliation"):self.read([spec])


if __name__=="__main__":unittest.main()
