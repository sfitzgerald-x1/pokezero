"""Real disposable supervised jobs only; no source/model or scientific run."""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import pickle
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_attempt_cost import AttemptCostSpec
from pokezero.mcts_eval.search_over_raw_attempt_journal import (
    create_attempt_journal,outer_attempt_journal_scope,read_attempt_journal_cost,
    declare_supervised_attempt,current_attempt_journal,SCHEMA)
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_resources import resource_source_hash
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from test_search_over_raw_supervision import SyntheticJob,BINDING


@dataclass(frozen=True)
class CloseBeforeKillJob:
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        import pokezero.mcts_eval.search_over_raw_attempt_journal as journal
        original=journal._write_new
        def barrier(path,value):
            original(path,value)
            if Path(path).parent.name=="writer-closes":signal.pause()
        journal._write_new=barrier  # Private disposable child; persists through scope exit.
        return SyntheticJob()(ownership,phase)


@dataclass(frozen=True)
class JournalJob:
    directory: str
    mode: str="normal"
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        import pokezero.mcts_eval.search_over_raw_supervision as supervision
        phase("startup");phase("selection")
        child=Path(self.directory)/"child"
        def run():
            job=(JournalJob(str(Path(self.directory)/"nested")) if self.mode=="nested_subtree"
                else CloseBeforeKillJob() if self.mode=="close_before_kill"
                else SyntheticJob("hang" if self.mode in {"hang","local_incomplete"} else "complete"))
            from pokezero.mcts_eval.search_over_raw_nested_ownership import DiagnosticIncompletePolicy
            policy=(DiagnosticIncompletePolicy("1"*64,"2"*64,hashlib.sha256(pickle.dumps(job)).hexdigest(),BINDING)
                if self.mode in {"local_incomplete","close_before_kill"} else None)
            if self.mode == "actor":
                from pokezero.mcts_eval.search_over_raw_actor_timed_run import KIND
                from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedRunContract
                return supervision._supervise_owned_job(job, directory=child, binding_sha256=BINDING,
                    verify_admission=lambda:True, contract=NativeTimedRunContract(), kind=KIND, work_phase="selection",
                    validate_result=lambda result:None)
            return supervise_fixed_work(job,directory=child,binding_sha256=BINDING,verify_admission=lambda:True,
                diagnostic_incomplete_policy=policy,_synthetic_watchdog_seconds=1.5 if policy is not None else None)
        if self.mode=="prelaunch_fail":
            with patch.object(supervision.subprocess,"Popen",side_effect=OSError("authored failed spawn")):value=run()
        elif self.mode=="claim_fail":
            real=supervision._save_new
            def fail(path,value):
                if Path(path)==child/"claim.json":
                    from pokezero.mcts_eval.search_over_raw_attempt_journal import _name
                    writer=current_attempt_journal()
                    assert (Path(writer.handle.directory)/"declarations"/_name(child)).is_file()
                    raise OSError("authored failed claim write")
                return real(path,value)
            with patch.object(supervision,"_save_new",side_effect=fail):
                try:run()
                except OSError:value=dict(supervisor_exit_code=1)
                else:raise AssertionError("claim failure did not propagate")
        elif self.mode=="threaded":
            errors=[]
            def target():
                try:run()
                except ValueError:errors.append(True)
            thread=threading.Thread(target=target);thread.start();thread.join()
            assert errors==[True] and not child.exists()
            value=dict(supervisor_exit_code=1)
        elif self.mode=="strip_header":
            real=supervision.subprocess.Popen
            def bootstrap(command,**kwargs):
                if "-m" in command and supervision.__name__ in command:
                    code="\n".join((
                        "import sys,json,io,types,pickle", "from pathlib import Path",
                        "import pokezero.mcts_eval.search_over_raw_supervision as supervision",
                        "header=json.loads(sys.stdin.buffer.read())", "header.pop('attempt_declaration')",
                        "sys.stdin=types.SimpleNamespace(buffer=io.BytesIO(json.dumps(header).encode()))",
                        "original=pickle.loads", "def reconstruct(data):",
                        "    Path(sys.argv[1]).joinpath('job-loaded').write_text('unexpected')",
                        "    return original(data)", "pickle.loads=reconstruct",
                        "raise SystemExit(supervision._worker(sys.argv[1],sys.argv[2]))"))
                    import sys
                    command=[sys.executable,"-B","-c",code,command[-2],command[-1]]
                return real(command,**kwargs)
            with patch.object(supervision.subprocess,"Popen",side_effect=bootstrap):value=run()
        else:value=run()
        phase("cleanup")
        return dict(authored=True,child_supervisor_exit_code=value["supervisor_exit_code"],scientific_measurements=0)


class AttemptJournalTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
    def spec(self,job,directory,watchdog=1800):
        return AttemptCostSpec(str(directory),hashlib.sha256(pickle.dumps(job)).hexdigest(),BINDING,
            "fixed-work","selection",FixedWorkDiagnosticContract().receipt(),resource_source_hash(),watchdog)
    def launch(self,mode="normal",watchdog=None):
        job=JournalJob(str(self.root),mode);outer=self.spec(job,self.root/"outer",watchdog or 1800)
        handle=create_attempt_journal(self.root/"journal",outer_attempt=outer,attempt_root=self.root)
        with outer_attempt_journal_scope(handle):
            terminal=supervise_fixed_work(job,directory=outer.directory,binding_sha256=BINDING,
                verify_admission=lambda:True,_synthetic_watchdog_seconds=watchdog)
        return handle,terminal

    def test_actual_dynamic_child_census_wait_and_nonoverlapping_cpu(self):
        handle,terminal=self.launch()
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        before={str(p):p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("subprocess.Popen") as spawn,patch("os.wait4") as wait:
            value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        spawn.assert_not_called();wait.assert_not_called()
        self.assertEqual(value["required_attempt_slots"],2,value)
        self.assertTrue(value["declared_supervised_attempt_census_complete"],value)
        self.assertEqual(value["measured_cpu_frontier"],[str(self.root/"outer")])
        self.assertEqual(value["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        self.assertFalse(value["attempt_population_exhaustiveness_validated"])
        self.assertFalse(value["resource_cost_complete"])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        parent=json.loads((self.root/"outer/claim.json").read_text())
        worker=json.loads((self.root/"outer/worker-result.json").read_text())
        self.assertEqual(parent["attempt_declaration"]["declaration_sha256"],terminal["attempt_declaration_sha256"])
        self.assertEqual(worker["attempt_journal_writer_close_sha256"],terminal["attempt_journal_writer_close_sha256"])

    def test_actual_actor_child_declares_and_closes_original_journal_without_cpu_double_count(self):
        handle, terminal = self.launch("actor")
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        result = read_attempt_journal_cost(handle, expected_outer_terminal=terminal)
        self.assertTrue(result["declared_supervised_attempt_census_complete"], result)
        self.assertTrue(result["all_journal_writers_normal_closed"])
        self.assertEqual(result["required_attempt_slots"], 2)
        child = json.loads((self.root / "child" / "terminal.json").read_bytes())
        self.assertEqual(child["status"], "ACTOR_ANNOTATED_NATIVE_TIMED_RUN_COMPLETE")
        self.assertEqual(result["measured_cpu_frontier"], [str(self.root / "outer")])
        self.assertEqual(result["known_nonoverlapping_job_cpu_seconds"], terminal["cpu_seconds_including_reaped_children"])
        self.assertFalse(result["work_validated"]); self.assertFalse(result["complete_workflow_cost"])

    def test_dynamic_prelaunch_failure_is_declared_and_costs_retained_without_promotion(self):
        handle,terminal=self.launch("prelaunch_fail");value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(value["required_attempt_slots"],2,value)
        child=next(row for row in value["attempts"] if row["directory"]==str(self.root/"child"))
        self.assertEqual(child["state"],"PRELAUNCH_FAILURE_COST_OBSERVED",value)
        self.assertGreaterEqual(child["supervisor_thread_cpu_seconds"],0)
        self.assertIsNone(child["job_cpu_seconds_including_reaped_children"])
        self.assertFalse(child["work_validated"])
        self.assertTrue(value["declared_supervised_attempt_census_complete"],value)

    def test_clean_local_timeout_has_complete_control_census_not_normal_writers(self):
        handle,terminal=self.launch("local_incomplete")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        with patch("subprocess.Popen") as spawn,patch("os.wait4") as wait:
            value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        spawn.assert_not_called();wait.assert_not_called()
        self.assertTrue(value["declared_supervised_attempt_census_complete"],value)
        self.assertFalse(value["all_journal_writers_normal_closed"])
        self.assertEqual(value["cleanly_terminated_diagnostic_writer_directories"],[str(self.root/"child")])
        child=next(r for r in value["attempts"] if r["directory"]==str(self.root/"child"))
        self.assertEqual(child["state"],"REAPED_FAILURE_COST_OBSERVED")
        self.assertFalse(child["work_validated"])
        self.assertFalse(child["measurement_window_cpu_wall_complete"])

    def test_writer_close_before_kill_is_bound_abnormal_not_orphaned_or_normal(self):
        handle,terminal=self.launch("close_before_kill")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertTrue(value["declared_supervised_attempt_census_complete"],value)
        self.assertFalse(value["all_journal_writers_normal_closed"])
        self.assertFalse((self.root/"child/worker-result.json").exists())
        self.assertEqual(value["declaration_errors"],[])

    def test_claim_write_failure_keeps_consumed_declaration_in_denominator(self):
        handle,terminal=self.launch("claim_fail");value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(value["required_attempt_slots"],2,value)
        self.assertEqual(next(r for r in value["attempts"] if r["directory"]==str(self.root/"child"))["state"],"MALFORMED_EVIDENCE")
        self.assertFalse(value["declared_supervised_attempt_census_complete"])

    def test_outer_watchdog_never_seals_missing_child_writer_or_terminal(self):
        handle,terminal=self.launch("hang",watchdog=1.5)
        self.assertEqual(terminal["supervisor_exit_code"],124,terminal)
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],2,value)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        self.assertTrue(value["writer_close_errors"],value)
        self.assertEqual(next(r for r in value["attempts"] if r["directory"]==str(self.root/"child"))["state"],"LAUNCHED_TERMINAL_UNOBSERVED")

    def test_retry_cannot_change_job_at_same_consumed_directory(self):
        job=SyntheticJob();outer=self.spec(job,self.root/"outer")
        handle=create_attempt_journal(self.root/"journal",outer_attempt=outer,attempt_root=self.root)
        with outer_attempt_journal_scope(handle):
            declare_supervised_attempt(outer)
            changed=self.spec(SyntheticJob("fail"),self.root/"outer")
            with self.assertRaisesRegex(ValueError,"original and one-shot"):declare_supervised_attempt(changed)
            with self.assertRaisesRegex(ValueError,"original and one-shot"):declare_supervised_attempt(outer)
        value=read_attempt_journal_cost(handle)
        self.assertEqual(value["required_attempt_slots"],1)
        self.assertEqual(value["attempts"][0]["state"],"UNSTARTED_UNOBSERVED")
        self.assertFalse(value["declared_supervised_attempt_census_complete"])

    def test_threaded_nested_call_refuses_before_declaration_and_claim(self):
        handle,terminal=self.launch("threaded")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],1,value)
        self.assertFalse((self.root/"child").exists())
        self.assertTrue(value["declared_supervised_attempt_census_complete"],value)

    def test_stripped_nested_bootstrap_reference_cannot_opt_out_before_job_load(self):
        handle,terminal=self.launch("strip_header")
        self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
        child=json.loads((self.root/"child/worker-result.json").read_text())
        self.assertEqual(child["exit_code"],1)
        self.assertEqual(child["error_type"],"ValueError")
        self.assertFalse((self.root/"child/job-loaded").exists())
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],2,value)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])

    def test_malformed_root_or_child_declaration_keeps_original_slot_count(self):
        handle,terminal=self.launch();paths=list((self.root/"journal/declarations").glob("*.json"))
        for path in paths:
            with self.subTest(path=path.name):
                original=path.read_bytes();path.write_text("{}")
                value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
                self.assertEqual(value["required_attempt_slots"],2,value)
                self.assertFalse(value["declared_supervised_attempt_census_complete"])
                self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])
                path.write_bytes(original)

    def test_lost_middle_declaration_retains_three_slots_without_fabricated_forest_cost(self):
        handle,terminal=self.launch("nested_subtree")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        original=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(original["required_attempt_slots"],3,original)
        self.assertTrue(original["declared_supervised_attempt_census_complete"],original)
        paths=list((self.root/"journal/declarations").glob("*.json"))
        middle=next(path for path in paths if json.loads(path.read_text())["spec"]["directory"]==str(self.root/"child"))
        middle.unlink()
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],3,value)
        self.assertIn("UnjoinableDeclaredForest",value["declaration_errors"])
        self.assertTrue(any(row["state"]=="MISSING_DECLARATION" for row in value["attempts"]),value)
        self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])
        self.assertFalse(value["declared_supervised_attempt_census_complete"])

    def test_original_writer_close_detects_removed_or_replaced_declaration(self):
        handle,terminal=self.launch();paths=list((self.root/"journal/declarations").glob("*.json"))
        child=next(p for p in paths if json.loads(p.read_text())["spec"]["parent_directory"] is not None)
        original=child.read_bytes();child.write_bytes(original+b" ")
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        self.assertTrue(value["writer_close_errors"],value)
        child.unlink()
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        self.assertTrue(value["delegation_errors"],value)
        self.assertIsNone(value["known_nonoverlapping_job_cpu_seconds"])
        self.assertEqual(value["required_attempt_slots"],2,value)
        self.assertTrue(any(row["state"]=="MISSING_DECLARATION" for row in value["attempts"]),value)

    def test_claim_permission_and_controller_scope_or_parent_pid_cannot_drift(self):
        handle,terminal=self.launch()
        claim_path=self.root/"child/claim.json";claim=json.loads(claim_path.read_text())
        changed=dict(claim);changed.pop("attempt_declaration");claim_path.write_text(json.dumps(changed))
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        self.assertTrue(value["writer_close_errors"],value)
        claim_path.write_text(json.dumps(claim))
        path=self.root/"journal/writer-closes/outer-controller.json";original=json.loads(path.read_text())
        for key,change in (("writer_pid",original["writer_pid"]+1),("complete_workflow_cost",True),("work_validated",True),("extra",1)):
            with self.subTest(key=key):
                changed=dict(original);changed[key]=change;path.write_text(json.dumps(changed))
                value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
                self.assertFalse(value["declared_supervised_attempt_census_complete"])

    def test_late_writer_read_cannot_mutate_already_costed_attempt_receipt(self):
        handle,terminal=self.launch()
        import pokezero.mcts_eval.search_over_raw_attempt_journal as journal
        actual=journal.read_job_writer_close
        def mutate(*args,**kwargs):
            result=actual(*args,**kwargs)
            path=self.root/"outer/terminal.json";path.write_bytes(path.read_bytes()+b" ")
            return result
        with patch.object(journal,"read_job_writer_close",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"namespace changed during census"):
                read_attempt_journal_cost(handle,expected_outer_terminal=terminal)

    def test_writer_cleanup_failure_does_not_replace_original_error(self):
        job=SyntheticJob();outer=self.spec(job,self.root/"outer")
        handle=create_attempt_journal(self.root/"journal",outer_attempt=outer,attempt_root=self.root)
        import pokezero.mcts_eval.search_over_raw_attempt_journal as journal
        with self.assertRaisesRegex(OSError,"authored primary") as caught:
            with patch.object(journal._Writer,"close",side_effect=ValueError("authored cleanup")):
                with outer_attempt_journal_scope(handle):raise OSError("authored primary")
        self.assertEqual(caught.exception.__notes__,["attempt journal writer close failed: ValueError"])
        self.assertIsNone(current_attempt_journal())

    def test_late_census_callback_cannot_change_original_registry(self):
        handle,terminal=self.launch()
        import pokezero.mcts_eval.search_over_raw_attempt_journal as journal
        actual=journal.reconcile_attempt_cost_roster
        def mutate(*args,**kwargs):
            result=actual(*args,**kwargs)
            path=self.root/"journal/declarations/late.json";path.write_text("{}")
            return result
        with patch.object(journal,"reconcile_attempt_cost_roster",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"namespace changed during census"):
                read_attempt_journal_cost(handle,expected_outer_terminal=terminal)

    def test_legacy_nonopted_run_has_no_declaration_fields(self):
        terminal=supervise_fixed_work(SyntheticJob(),directory=self.root/"legacy",binding_sha256=BINDING,verify_admission=lambda:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertNotIn("attempt_declaration_sha256",terminal)
        self.assertNotIn("attempt_declaration",json.loads((self.root/"legacy/claim.json").read_text()))
        self.assertFalse((self.root/"journal").exists())


if __name__=="__main__":unittest.main()
