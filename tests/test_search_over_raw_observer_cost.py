"""Actual own ps/disposable workers plus authored wait failures; no science."""
from copy import deepcopy
from dataclasses import asdict,replace
import itertools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
import pokezero.mcts_eval.search_over_raw_observer_cost as observer
from pokezero.mcts_eval.search_over_raw_resources import _process_table
from pokezero.mcts_eval.paper_reference_parallel import owned_process_identity,owned_process_identities,live_owned_groups
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from test_search_over_raw_supervision import SyntheticJob,BINDING
import pokezero.mcts_eval.search_over_raw_exploration_entry as entry
import test_search_over_raw_exploration_entry as entry_tests
import pokezero.mcts_eval.search_over_raw_attempt_journal as journal
import test_search_over_raw_attempt_journal as journal_tests


class ObserverCostTests(unittest.TestCase):
    def fixture(self,admitted=True):
        fixture=entry_tests.ExplorationEntryTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.make(admitted=admitted)
        return fixture
    def failed_entry(self,spawn=False):
        fixture=self.fixture();job,_=fixture.preflight()
        if spawn:
            with patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
                with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                    entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        else:
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        return fixture,job,caught.exception
    def fake_pending(self):
        fake=SimpleNamespace(pid=987654,returncode=None)
        window=observer.ObserverWindow("authored_disposable_test")
        clock=itertools.count(start=1.,step=.01)
        with patch.object(observer.subprocess,"Popen",return_value=fake),\
                patch.object(observer.os,"wait4",return_value=(0,0,None)),\
                patch.object(observer.os,"kill") as kill,\
                patch.object(observer.time,"perf_counter",side_effect=lambda:next(clock)):
            with window:
                with self.assertRaises(subprocess.TimeoutExpired):
                    observer.run_owned_ps(["ps","-p","1","-o","lstart="],timeout=.01,purpose="rss")
        self.assertEqual(kill.call_count,1)
        query_id=window.snapshot()["own_process_queries"][0]["query_id"]
        def clean():
            # Only this authored fake retained handle; never remove a real one.
            for registry in (observer._PENDING,observer._LOST):
                if query_id in registry:
                    self.assertIs(registry[query_id].process,fake);registry.pop(query_id)
            observer._LATE.pop(query_id,None)
        self.addCleanup(clean)
        return window,fake,query_id

    def test_actual_identity_group_rss_and_freeze_queries_record_disjoint_ps_usage_not_tables(self):
        window=observer.ObserverWindow("authored_disposable_test")
        with window:
            self.assertTrue(owned_process_identity(os.getpid()))
            self.assertIn(os.getpid(),owned_process_identities([os.getpid()]))
            self.assertIn(os.getpid(),live_owned_groups({os.getpgrp()}))
            rows,cpu=_process_table(.2)
            self.assertTrue(rows);self.assertGreaterEqual(cpu,0)
            with observer.QueryPurpose("ownership_freeze"):_process_table(.2)
        trace=window.snapshot();summary=observer.read_observer_window(trace)
        self.assertEqual({r["purpose"] for r in trace["own_process_queries"]},observer.PURPOSES)
        self.assertGreater(summary["known_own_query_cpu_seconds"],0)
        self.assertEqual(summary["own_query_usage_unknown"],0)
        self.assertFalse(trace["process_table_output_retained"])
        self.assertFalse(summary["legacy_sampler_cpu_added_again"])
        self.assertNotIn("stdout",json.dumps(trace));self.assertNotIn("lstart",json.dumps(trace))

    def test_actual_emergency_cleanup_captures_query_cpu_without_nested_counter_addition(self):
        with tempfile.TemporaryDirectory() as name:
            window=observer.ObserverWindow("authored_disposable_test")
            with window:
                terminal=supervise_fixed_work(SyntheticJob("hang"),directory=Path(name)/"job",binding_sha256=BINDING,
                    verify_admission=lambda:True,_synthetic_watchdog_seconds=.4)
            self.assertEqual(terminal["supervisor_exit_code"],124,terminal)
            trace=window.snapshot();summary=observer.read_observer_window(trace)
            self.assertTrue(any(r["purpose"]=="ownership_freeze" for r in trace["own_process_queries"]),trace)
            self.assertGreater(summary["known_own_query_cpu_seconds"],0)
            self.assertFalse(trace["nested_counters_added_again"])
            self.assertTrue(all(r["owner_pid"]==os.getpid() for r in trace["own_process_queries"]))

    def test_timeout_late_reap_keeps_original_frozen_window_and_never_adopts_cpu(self):
        first,fake,query_id=self.fake_pending();before=json.dumps(first.snapshot(),sort_keys=True)
        self.assertIsNone(first.snapshot()["own_process_queries"][0]["actual_cpu_seconds"])
        usage=SimpleNamespace(ru_utime=.2,ru_stime=.1)
        # Actual observation occurs outside any active window; it must survive.
        with patch.object(observer.os,"wait4",return_value=(fake.pid,0,usage)):
            observations=observer.drain_pending_queries()
        self.assertEqual(observations[0]["actual_cpu_seconds"],.2+.1)
        second=observer.ObserverWindow("authored_disposable_test")
        with second:observer.drain_pending_queries()
        trace=second.snapshot();summary=observer.read_observer_window(trace)
        self.assertEqual(summary["known_own_query_cpu_seconds"],0)
        self.assertEqual(trace["foreign_late_reap_observations"][0]["origin_window_id"],first.window_id)
        self.assertEqual(trace["foreign_late_reap_observations"][0]["query_id"],query_id)
        self.assertFalse(trace["foreign_late_reap_observations"][0]["adopted_into_current_window_cpu"])
        self.assertEqual(before,json.dumps(first.snapshot(),sort_keys=True))

    def test_echild_is_permanently_unknown_without_signalling_or_repolling(self):
        first,fake,query_id=self.fake_pending()
        with patch.object(observer.os,"wait4",side_effect=ChildProcessError) as wait,patch.object(observer.os,"kill") as kill:
            second=observer.ObserverWindow("authored_disposable_test")
            with second:observer.drain_pending_queries();observer.drain_pending_queries()
            self.assertEqual(wait.call_count,1);kill.assert_not_called()
        self.assertIn(query_id,observer._LOST)
        observation=second.snapshot()["foreign_late_reap_observations"][0]
        self.assertEqual(observation["status"],"WAIT_OWNERSHIP_LOST")
        self.assertIsNone(observation["actual_cpu_seconds"])
        observer.read_observer_window(second.snapshot())

    def test_unknown_spawn_usage_and_original_exception_are_preserved(self):
        window=observer.ObserverWindow("authored_disposable_test")
        with patch.object(observer.subprocess,"Popen",side_effect=OSError("authored spawn error")):
            with self.assertRaisesRegex(OSError,"authored spawn error"):
                with window:observer.run_owned_ps(["ps","-p","1","-o","lstart="],timeout=2,purpose="ownership_identity")
        trace=window.snapshot();summary=observer.read_observer_window(trace)
        self.assertEqual(trace["own_process_queries"][0]["state"],"SPAWN_FAILED_USAGE_UNKNOWN")
        self.assertEqual(summary["own_query_usage_unknown"],1)
        self.assertEqual(trace["error_type"],"OSError")

    def test_failed_admission_measures_preflight_without_any_new_output_namespace(self):
        fixture=self.fixture(admitted=False);before=entry._evidence_inventory(fixture.directory)
        with patch("subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError,"admission is incomplete") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        spawn.assert_not_called();self.assertEqual(before,entry._evidence_inventory(fixture.directory))
        trace=caught.exception.observer_cost_measurement;summary=observer.read_observer_window(trace)
        self.assertEqual(trace["bindings"],{})
        self.assertGreater(summary["calling_thread_cpu_seconds"],0)
        self.assertFalse((fixture.directory/"attempt-cost-journal").exists())

    def test_real_entry_worker_failure_and_spawn_failure_keep_bound_external_costs(self):
        for spawn in (False,True):
            with self.subTest(spawn=spawn):
                fixture,job,primary=self.failed_entry(spawn)
                path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
                trace=json.loads(path.read_text());self.assertEqual(trace,primary.observer_cost_measurement)
                terminal=json.loads((fixture.directory/"exploration-entry-run/terminal.json").read_text())
                with patch("subprocess.Popen") as launch,patch("os.wait4") as wait:
                    cost=entry.read_entry_observer_cost(job,terminal)
                launch.assert_not_called();wait.assert_not_called()
                self.assertGreater(cost["observer_window"]["calling_thread_cpu_seconds"],0)
                self.assertEqual(cost["actual_outer_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
                self.assertFalse(cost["narrower_supervisor_and_sampler_counters_added_again"])
                self.assertFalse(cost["complete_workflow_cost"])
                self.assertIsNone(cost["total_workflow_cpu_seconds"])
                publication=json.loads((fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json").read_text())
                self.assertEqual(publication,primary.observer_publication_measurement)
                self.assertEqual(publication["started_monotonic_seconds"],trace["ended_monotonic_seconds"])
                self.assertEqual(publication["calling_thread_cpu_started_seconds"],trace["calling_thread_cpu_ended_seconds"])
                self.assertEqual(publication["trace_bytes_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(cost["known_disjoint_cpu_subtotal_seconds"],
                    cost["observer_window"]["calling_thread_cpu_seconds"]+cost["observer_window"]["known_own_query_cpu_seconds"]+
                    publication["calling_thread_cpu_seconds"]+(cost["actual_outer_job_cpu_seconds"] or 0))
                self.assertEqual(cost["campaign_elapsed_wall_seconds"],publication["ended_monotonic_seconds"]-trace["started_monotonic_seconds"])
                self.assertFalse(cost["publication_usage_unknown"])
                self.assertFalse(publication["supplemental_receipt_validation_serialization_write_included"])

    def test_final_census_callback_cannot_change_external_spec_review_or_returned_terminal(self):
        for target in ("spec","review","returned"):
            with self.subTest(target=target):
                fixture,job,_=self.failed_entry(spawn=True)
                terminal=json.loads((fixture.directory/"exploration-entry-run/terminal.json").read_text())
                real=journal.read_attempt_journal_cost
                def change(*args,**kwargs):
                    value=real(*args,**kwargs)
                    if target=="returned":terminal["authored_late_mutation"]=True
                    else:
                        path=fixture.spec_path if target=="spec" else Path(fixture.spec["review_path"])
                        self.assertFalse(path.is_relative_to(fixture.directory))
                        path.write_bytes(path.read_bytes()+b" ")
                    return value
                with patch.object(journal,"read_attempt_journal_cost",side_effect=change):
                    with self.assertRaisesRegex(ValueError,"changed during final readback"):
                        entry.read_entry_observer_cost(job,terminal)

    def test_outer_descriptor_is_bound_beyond_advertised_binding_hash(self):
        fixture,job,_=self.failed_entry(spawn=True)
        real=journal.journal_metadata
        original_contract=entry.ExplorationAuditorContract(fixture.spec["emergency_watchdog_seconds"]).receipt()
        alterations=(dict(directory=str(fixture.directory/"other")),dict(parent_directory=str(fixture.directory)),
            dict(job_sha256="a"*64),dict(kind="fixed-work"),dict(work_phase="boundaries"),
            dict(contract=dict(original_contract,authored_different_contract=True)),dict(effective_watchdog_seconds=19.))
        for alteration in alterations:
            with self.subTest(alteration=alteration):
                def metadata(*args,**kwargs):
                    original,outer=real(*args,**kwargs)
                    return original,replace(outer,**alteration)
                with patch.object(journal,"journal_metadata",side_effect=metadata),\
                        patch.object(journal,"read_attempt_journal_cost") as census:
                    with self.assertRaisesRegex(ValueError,"lost original outer job/watchdog"):
                        entry.read_entry_observer_cost(job)
                census.assert_not_called()

    def test_query_outer_numeric_pid_collision_refuses_cost_join_only(self):
        fixture,job,_=self.failed_entry()
        path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
        trace=json.loads(path.read_text())
        launch=json.loads((fixture.directory/"exploration-entry-run/launch.json").read_text())
        self.assertTrue(trace["own_process_queries"])
        trace["own_process_queries"][0]["query_pid"]=launch["pid"]
        path.write_text(json.dumps(trace))
        publication_path=fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json"
        publication=json.loads(publication_path.read_text())
        publication.update(trace_sha256=digest(trace),trace_bytes_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        publication_path.write_text(json.dumps(publication))
        before=entry._evidence_inventory(fixture.directory)
        with patch("subprocess.Popen") as spawn,patch("os.wait4") as wait:
            with self.assertRaisesRegex(ValueError,"attribution ambiguous"):
                entry.read_entry_observer_cost(job)
        spawn.assert_not_called();wait.assert_not_called()
        self.assertEqual(before,entry._evidence_inventory(fixture.directory))

    def test_actual_nested_job_cost_is_already_in_outer_wait_and_context_is_not_propagated(self):
        fixture=journal_tests.AttemptJournalTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        window=observer.ObserverWindow("authored_disposable_test")
        with window:handle,terminal=fixture.launch("nested_subtree")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        costs=journal.read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(costs["required_attempt_slots"],3)
        self.assertEqual(costs["measured_cpu_frontier"],[str(fixture.root/"outer")])
        self.assertEqual(costs["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        trace=window.snapshot();observer.read_observer_window(trace)
        self.assertTrue(trace["own_process_queries"])
        self.assertTrue(all(row["owner_pid"]==os.getpid() for row in trace["own_process_queries"]))
        self.assertFalse(trace["nested_counters_added_again"])

    def test_receipt_write_failure_keeps_primary_error_type_and_no_success_claim(self):
        fixture=self.fixture()
        real=entry._write_new
        def write(path,value):
            if Path(path).name=="entry-observer-window.json":raise OSError("authored measurement write failure")
            return real(path,value)
        with patch.object(entry,"_write_new",side_effect=write),\
                patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        self.assertIn("entry observer evidence failed: OSError",caught.exception.__notes__)
        self.assertEqual(caught.exception.observer_cost_measurement["error_type"],"ValueError")
        self.assertFalse((fixture.directory/"attempt-cost-journal/entry-observer-window.json").exists())
        publication=caught.exception.observer_publication_measurement
        self.assertEqual(publication["status"],"PUBLICATION_FAILED")
        self.assertFalse(publication["first_trace_write_closed"])
        self.assertIsNone(publication["trace_bytes_sha256"])
        self.assertGreaterEqual(publication["calling_thread_cpu_seconds"],0)
        self.assertFalse((fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json").exists())

    def test_supplement_write_failure_keeps_primary_and_original_trace_but_tail_unknown_durably(self):
        fixture=self.fixture();job,_=fixture.preflight();real=entry._write_new
        def write(path,value):
            if Path(path).name=="entry-observer-publication-cost.json":raise OSError("authored supplement failure")
            return real(path,value)
        with patch.object(entry,"_write_new",side_effect=write),\
                patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        self.assertEqual(caught.exception.observer_publication_measurement["status"],"TRACE_PUBLISHED")
        self.assertIn("entry observer evidence failed: OSError",caught.exception.__notes__)
        path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
        self.assertEqual(json.loads(path.read_text()),caught.exception.observer_cost_measurement)
        cost=entry.read_entry_observer_cost(job)
        self.assertIsNone(cost["publication_measurement"]);self.assertTrue(cost["publication_usage_unknown"])
        self.assertIn("original observer trace publication usage unknown",cost["incomplete_components"])
        self.assertFalse(cost["complete_workflow_cost"])

    def test_absent_historical_supplement_is_not_fabricated_or_required_for_known_cost(self):
        fixture,job,_=self.failed_entry(spawn=True)
        path=fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json"
        path.unlink()  # Only this disposable fixture's supplemental measurement.
        before=entry._evidence_inventory(fixture.directory)
        cost=entry.read_entry_observer_cost(job)
        self.assertEqual(before,entry._evidence_inventory(fixture.directory))
        self.assertTrue(cost["publication_usage_unknown"])
        self.assertIsNone(cost["publication_measurement"])

    def test_first_trace_writer_cannot_establish_self_consistent_replacement_baseline(self):
        fixture=self.fixture();real=entry._write_new
        def write(path,value):
            if Path(path).name=="entry-observer-window.json":
                value["calling_thread_cpu_started_seconds"]=value["calling_thread_cpu_ended_seconds"]
                value["calling_thread_cpu_seconds"]=0.
                observer.read_observer_window(value)  # Locally valid, but not the original window.
            return real(path,value)
        with patch.object(entry,"_write_new",side_effect=write),\
                patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        self.assertIn("entry observer evidence failed: ValueError",caught.exception.__notes__)
        original=caught.exception.observer_cost_measurement
        self.assertGreater(original["calling_thread_cpu_seconds"],0)
        path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
        self.assertNotEqual(digest(json.loads(path.read_text())),digest(original))
        self.assertFalse((fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json").exists())

    def test_supplement_writer_mutation_cannot_be_returned_as_original_measurement(self):
        fixture=self.fixture();real=entry._write_new
        def write(path,value):
            if Path(path).name=="entry-observer-publication-cost.json":value["trace_bytes_sha256"]="a"*64
            return real(path,value)
        with patch.object(entry,"_write_new",side_effect=write),\
                patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn failure")):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry") as caught:
                entry.run_exploration_entry(fixture.directory,fixture.spec_path)
        self.assertIn("entry observer evidence failed: ValueError",caught.exception.__notes__)
        job=entry.ExplorationAuditorJob(**caught.exception.observer_cost_measurement["bindings"]["entry_job"])
        original=caught.exception.observer_publication_measurement
        path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
        self.assertEqual(original["trace_bytes_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError,"original trace/path/bytes/owner"):
            entry.read_entry_observer_cost(job)

    def test_publication_reader_rejects_overlap_alias_bytes_and_completeness(self):
        fixture,job,_=self.failed_entry(spawn=True)
        path=fixture.directory/"attempt-cost-journal/entry-observer-window.json"
        trace=json.loads(path.read_text());raw_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        original=json.loads((fixture.directory/"attempt-cost-journal/entry-observer-publication-cost.json").read_text())
        for key,value in (("observer_pid",True),("calling_thread_cpu_started_seconds",trace["calling_thread_cpu_started_seconds"]),
                ("started_monotonic_seconds",trace["started_monotonic_seconds"]),("trace_bytes_sha256","a"*64),
                ("complete_workflow_cost",True),("supplemental_receipt_validation_serialization_write_included",True)):
            with self.subTest(key=key):
                changed=deepcopy(original);changed[key]=value
                with self.assertRaises(ValueError):observer.read_publication_cost(changed,trace,trace_path=str(path),trace_bytes_sha256=raw_sha)
        window=observer.ObserverWindow("authored_disposable_test")
        with window:pass
        changed=deepcopy(original)
        with self.assertRaisesRegex(ValueError,"original trace/path/bytes/owner"):
            observer.read_publication_cost(changed,window.snapshot(),trace_path=str(path),trace_bytes_sha256=raw_sha)

    def test_pure_reader_rejects_numeric_alias_and_false_complete_or_query_adoption(self):
        window=observer.ObserverWindow("authored_disposable_test")
        with window:owned_process_identity(os.getpid())
        trace=window.snapshot()
        for key,value in (("observer_pid",True),("calling_thread_cpu_seconds",False),("resource_cost_complete",True)):
            with self.subTest(key=key):
                changed=deepcopy(trace);changed[key]=value
                with self.assertRaises(ValueError):observer.read_observer_window(changed)
        changed=deepcopy(trace);changed["own_process_queries"][0]["origin_window_id"]="a"*64
        with self.assertRaises(ValueError):observer.read_observer_window(changed)
        for state in ("REAPED","PENDING_UNREAPED","WAIT_OWNERSHIP_LOST"):
            changed=deepcopy(trace);row=changed["own_process_queries"][0]
            row.update(state=state,query_pid=None)
            if state!="REAPED":row.update(actual_exit_code=None,actual_cpu_seconds=None)
            with self.subTest(state=state),self.assertRaises(ValueError):observer.read_observer_window(changed)
        changed=deepcopy(trace)
        changed["calling_thread_cpu_seconds"]=changed["elapsed_wall_seconds"]+1.
        changed["calling_thread_cpu_ended_seconds"]=changed["calling_thread_cpu_started_seconds"]+changed["calling_thread_cpu_seconds"]
        # Keep the arithmetic internally consistent; reject its physical bound.
        changed["calling_thread_cpu_seconds"]=changed["calling_thread_cpu_ended_seconds"]-changed["calling_thread_cpu_started_seconds"]
        with self.assertRaises(ValueError):observer.read_observer_window(changed)


if __name__=="__main__":unittest.main()
