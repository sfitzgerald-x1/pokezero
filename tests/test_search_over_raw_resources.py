"""Pure cost readers and disposable process fixtures; no games or models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_resources import (
    SCHEMA, OwnedRSSObserver, SamplingFailure, _select, measurement_protocol,
    summarize_memory_trace, validate_memory_receipt, derive_fixed_work_accounting,
    _process_table,
    _PENDING_SAMPLERS, _pending_samplers,
    read_sampling_cpu_components,
)
from test_search_over_raw_supervision import BINDING, INPUT_BINDING, selection_inputs


def process(pid=10, parent=1, group=10, rss=100, birth="birth-a", state="S"):
    return dict(pid=pid,parent_pid=parent,group=group,rss_bytes=rss,birth=birth,state=state)


def main_launch():
    return dict(pid=10,group=10,birth_identity="birth-a",supervisor_pid=1,binding_sha256=BINDING)


def generation(main=None):
    main = main or main_launch()
    return dict(pid=main["pid"],group=main["group"],birth=main["birth_identity"],launch_sha256=digest(main))


def trace():
    generations = {"job":generation()}
    return dict(schema=SCHEMA,protocol=measurement_protocol(),binding_sha256=BINDING,
        main_launch=main_launch(),began=0.,ended=2.,generations=generations,inventory_error=None,
        pending_sampler_cleanup_pids=[],
        samples=[dict(ordinal=0,started=.1,ended=.2,generations=deepcopy(generations),
            rows=[process(),process(11,10,11,200,"birth-b"),process(12,11,11,300,"birth-c")],
            error=None,sampling_child_cpu_seconds=.01),
        dict(ordinal=1,started=1.,ended=1.1,generations=deepcopy(generations),
            rows=[process(rss=150),process(11,10,11,150,"birth-b")],error=None,sampling_child_cpu_seconds=.02)])


class ResourceReaderTests(unittest.TestCase):
    def test_partial_sampler_cpu_keeps_known_subtotal_without_rewriting_old_summary(self):
        value=trace();value["samples"][1].update(rows=None,error="query_unobserved",sampling_child_cpu_seconds=None)
        old=summarize_memory_trace(value)
        self.assertIsNone(old["sampling_child_cpu_seconds"])
        result=read_sampling_cpu_components(value)
        self.assertEqual(result["known_reaped_sampling_cpu_seconds"],.01)
        self.assertEqual(result["unknown_sample_cpu_records"],1)
        self.assertIsNone(result["sampling_cpu_seconds"])
        self.assertIsNone(result["emergency_ownership_query_cpu_seconds"])
        self.assertFalse(result["rss_query_cost_window_complete"])
        self.assertEqual(old,summarize_memory_trace(value))

    def test_postquery_selection_failure_does_not_discard_reaped_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/"owned-groups").mkdir()
            observer=OwnedRSSObserver(root,main_launch(),BINDING,0.)
            with patch("pokezero.mcts_eval.search_over_raw_resources._process_table",return_value=([process()],.02)),\
                    patch("pokezero.mcts_eval.search_over_raw_resources._select",side_effect=ValueError("authored failure")):
                observer.sample()
            self.assertEqual(observer.samples[0]["sampling_child_cpu_seconds"],.02)
            self.assertEqual(observer.samples[0]["error"],"ValueError")
            self.assertIsNone(observer.samples[0]["rows"])

    def test_uninterruptible_sampler_never_uses_blocking_wait_or_launches_more_queries(self):
        import os,time
        from types import SimpleNamespace
        child=SimpleNamespace(pid=7777777,returncode=None)
        start=time.perf_counter()
        try:
            with patch("pokezero.mcts_eval.search_over_raw_resources.subprocess.Popen",return_value=child) as launch, \
                 patch("pokezero.mcts_eval.search_over_raw_resources.os.kill") as kill, \
                 patch("pokezero.mcts_eval.search_over_raw_resources.os.wait4",return_value=(0,0,None)) as wait:
                with self.assertRaises(SamplingFailure) as caught: _process_table(.001)
                self.assertEqual(caught.exception.reason,"query_timeout_cleanup_pending")
                self.assertIsNone(caught.exception.cpu_seconds)
                self.assertLess(time.perf_counter()-start,.2)
                self.assertEqual(_pending_samplers(),[child.pid])
                with self.assertRaises(SamplingFailure) as caught: _process_table(.001)
                self.assertEqual(caught.exception.reason,"pending_sampler_cleanup")
                launch.assert_called_once(); kill.assert_called_once()
                self.assertTrue(all(call.args[1] == os.WNOHANG for call in wait.call_args_list))
            with patch("pokezero.mcts_eval.search_over_raw_resources.os.wait4",side_effect=ChildProcessError):
                self.assertEqual(_pending_samplers(),[child.pid])
        finally: _PENDING_SAMPLERS.pop(child.pid,None)
    def test_actual_query_keeps_exact_wait4_and_timeout_reaps_only_sampler(self):
        import os, subprocess, sys, time
        actual=subprocess.Popen; children=[]
        def launch(*args,**kwargs):
            child=actual(*args,**kwargs); children.append(child)
            child.poll=lambda: self.fail("sampler poll must not consume wait4")
            return child
        with patch("pokezero.mcts_eval.search_over_raw_resources.subprocess.Popen",side_effect=launch):
            rows,cpu=_process_table(.2)
        self.assertTrue(rows); self.assertGreaterEqual(cpu,0)
        self.assertEqual(children[0].returncode,0)
        def slow(*args,**kwargs):
            child=actual([sys.executable,"-c","import time; time.sleep(60)"],**kwargs)
            children.append(child); return child
        start=time.perf_counter()
        with patch("pokezero.mcts_eval.search_over_raw_resources.subprocess.Popen",side_effect=slow):
            with self.assertRaises(SamplingFailure) as error: _process_table(.01)
        self.assertEqual(error.exception.reason,"query_timeout")
        self.assertGreaterEqual(error.exception.cpu_seconds,0)
        self.assertLess(time.perf_counter()-start,1.)
        self.assertLess(children[-1].returncode,0)
        with self.assertRaises(ChildProcessError): os.wait4(children[-1].pid,os.WNOHANG)

    def test_aggregate_overlap_and_shared_page_limits_are_explicit(self):
        before=trace(); result=summarize_memory_trace(before)
        self.assertEqual(result["maximum_observed_rss_sum_bytes"],600)
        self.assertAlmostEqual(result["sampling_child_cpu_seconds"],.03)
        self.assertAlmostEqual(result["maximum_unsampled_gap_seconds"],.9)
        self.assertTrue(result["all_declared_launch_generations_observed_without_query_errors"])
        for field in ("exact_peak_measured","physical_memory_measured","true_peak_lower_bound_claim","resource_cost_complete"):
            self.assertFalse(result["protocol"][field])
        self.assertFalse(result["independently_validated"])
        self.assertEqual(before,trace())

    def test_missing_short_lived_generation_is_not_zero_peak_or_complete_coverage(self):
        value=trace()
        value["generations"]["group-20-short.json"]=dict(pid=20,group=20,birth="short",launch_sha256="c"*64)
        result=summarize_memory_trace(value)
        self.assertEqual(result["missed_launch_generations"],["group-20-short.json"])
        self.assertFalse(result["all_declared_launch_generations_observed_without_query_errors"])
        self.assertEqual(result["maximum_observed_rss_sum_bytes"],600)

    def test_sampler_failure_and_no_observations_remain_unknown_not_zero(self):
        value=trace()
        for sample in value["samples"]:
            sample.update(rows=None,error="query_timeout",sampling_child_cpu_seconds=None)
        result=summarize_memory_trace(value)
        self.assertIsNone(result["maximum_observed_rss_sum_bytes"])
        self.assertIsNone(result["sampling_child_cpu_seconds"])
        self.assertFalse(result["all_declared_launch_generations_observed_without_query_errors"])
        self.assertEqual(result["successful_sample_count"],0)
        self.assertEqual(result["missed_launch_generations"],["job"])

    def test_birth_reuse_unknown_processes_and_zombies_are_excluded(self):
        rows=[process(),process(11,10,11,200,"child"),process(12,11,11,300,"grandchild"),
            process(99,1,99,900,"unrelated"),process(13,10,10,1000,"dead","Z")]
        chosen=_select(rows,{"job":generation()}, {})
        self.assertEqual([r["pid"] for r in chosen],[10,11,12])
        rows[0]["birth"]="reused"
        self.assertEqual(_select(rows,{"job":generation()},{}),[])
        # A previously observed child remains attributed after reparenting;
        # a different birth with its recycled PID cannot acquire attribution.
        self.assertEqual([r["pid"] for r in _select([process(11,1,11,200,"child")],
            {"job":generation()},{11:"child"})],[11])
        self.assertEqual(_select([process(11,1,11,200,"new")],{"job":generation()},{11:"child"}),[])

    def test_mutated_metrics_identity_units_clocks_and_schema_rejected(self):
        for mode in ("duplicate","unowned","birth","rss","nan","order","binding","scope","failed_zero","inventory"):
            value=trace(); sample=value["samples"][0]
            if mode=="duplicate": sample["rows"].append(deepcopy(sample["rows"][0]))
            if mode=="unowned": sample["rows"].append(process(99,1,99))
            if mode=="birth": sample["rows"][0]["birth"]="new"
            if mode=="rss": sample["rows"][0]["rss_bytes"]=True
            if mode=="nan": sample["ended"]=float("nan")
            if mode=="order": value["samples"][1]["started"]=0.
            if mode=="binding": value["binding_sha256"]="b"*64
            if mode=="scope": value["protocol"]["exact_peak_measured"]=True
            if mode=="failed_zero": sample.update(error="error",rows=[])
            if mode=="inventory": value["generations"]["job"]["launch_sha256"]="b"*64
            with self.subTest(mode=mode),self.assertRaises(ValueError): summarize_memory_trace(value)

    def test_observer_exports_only_owned_rows_and_preserves_sampling_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/"owned-groups").mkdir()
            observer=OwnedRSSObserver(root,main_launch(),BINDING,0.)
            with patch("pokezero.mcts_eval.search_over_raw_resources._process_table",return_value=(
                    [process(),process(99,1,99,123456,"private-unrelated")],.02)):
                observer.sample()
            with patch("pokezero.mcts_eval.search_over_raw_resources._process_table",
                    side_effect=SamplingFailure("query_timeout",.03)):
                observer.sample(timeout=.001)
            import time
            value,result=observer.finish(time.perf_counter())
            self.assertNotIn("private-unrelated",json.dumps(value))
            self.assertEqual(result["maximum_observed_rss_sum_bytes"],100)
            self.assertEqual(result["sampling_child_cpu_seconds"],.05)
            self.assertFalse(result["all_declared_launch_generations_observed_without_query_errors"])
            # Runtime output of the implementation under test, not a source edit.
            (root/"memory-trace.json").write_text(json.dumps(value))
            self.assertEqual(validate_memory_receipt(root,binding=BINDING,main_launch=main_launch(),
                expected_summary=result),digest(value))
            changed=deepcopy(result); changed["maximum_observed_rss_sum_bytes"]=0
            with self.assertRaises(ValueError): validate_memory_receipt(root,binding=BINDING,
                main_launch=main_launch(),expected_summary=changed)

    def test_final_inventory_error_retains_observed_memory_but_cannot_clear(self):
        value=trace(); value.update(generations=None,inventory_error="ValueError")
        result=summarize_memory_trace(value)
        self.assertEqual(result["maximum_observed_rss_sum_bytes"],600)
        self.assertIsNone(result["missed_launch_generations"])
        self.assertFalse(result["all_declared_launch_generations_observed_without_query_errors"])


class WorkCostReaderTests(unittest.TestCase):
    def test_actual_native_calls_not_multiplied_again_by_world_multiplicity(self):
        evidence=selection_inputs("incumbent")
        value=derive_fixed_work_accounting(evidence,arm="incumbent",binding=INPUT_BINDING,
            statistics_root_id="same-root",leaf="model")
        self.assertEqual((value["attempted_units"],value["accepted_units"],value["cancelled_units"]),(64,64,0))
        self.assertEqual(value["committed_leaf_prices"],2)
        self.assertEqual(value["observed_path_transitions"],64)
        self.assertIsNone(value["reported_terminal_rollout_leaves"])
        self.assertFalse(value["equal_cpu_or_work_claim"])

    def test_reference_all_workers_atomic_attempts_and_transitions(self):
        value=derive_fixed_work_accounting(selection_inputs("reference"),arm="reference",binding=INPUT_BINDING,
            statistics_root_id="same-root",leaf="raw_rollout")
        self.assertEqual((value["attempted_units"],value["accepted_units"],value["cancelled_units"]),(80,80,0))
        self.assertIsNone(value["reported_terminal_rollout_leaves"])
        self.assertIsNone(value["reported_refused_rollout_leaves"])
        self.assertFalse(value["resource_cost_complete"])

    def test_missing_or_partial_atomic_work_cannot_be_accepted_cost(self):
        for arm in ("incumbent","reference"):
            evidence=selection_inputs(arm)
            if arm=="incumbent": evidence["consumed_world_inputs"]["calls"][0]["native_report"]["atomic_work"]["rounds"][0]["commits"].pop()
            else: evidence["worker_receipts"][0]["evidence"]["atomic_work"]["rows"][0]["status"]="REFUSED"
            with self.subTest(arm=arm),self.assertRaises(ValueError): derive_fixed_work_accounting(evidence,
                arm=arm,binding=INPUT_BINDING,statistics_root_id="same-root",leaf="model")


if __name__ == "__main__": unittest.main()
