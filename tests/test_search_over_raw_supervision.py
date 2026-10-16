"""Real disposable synthetic processes only; no adapters, engines or games."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw_supervision import (
    FixedWorkAdapterJob, supervise_fixed_work, _groups, _phase_costs, _stop_unadmitted_child,
)
from pokezero.mcts_eval.paper_reference_parallel import record_owned_process, close_owned_generations, live_owned_groups

BINDING = "a" * 64
INPUT_BINDING = dict(root_binding=BINDING, subject="p1", set_source_hash=BINDING, information_key=b"root-public".hex())


def selection_inputs(arm):
    from pokezero.mcts_eval.search_over_raw_world_inputs import _manifest
    if arm == "reference":
        from test_search_over_raw_atomic_work import selection_fixture
        result = selection_fixture()
        for receipt in result["worker_receipts"]:
            receipt["evidence"]["atomic_work"]["binding"] = dict(INPUT_BINDING)
            receipt["own_statistics"]["battle_id"] = "search-over-raw-root:same-root"
        return dict(**result, consumed_world_inputs=_manifest("reference", INPUT_BINDING,
            [dict(worker=w, ordinal=i, consumed=True, status="ROOT_VALIDATED", input_sha256="c"*64)
             for w in range(20) for i in range(4)]))
    # The real dispatch witness machinery, with synthetic constructed states.
    from test_search_over_raw_world_inputs import arguments
    from test_search_over_raw_construction import construct_actual_python
    from pokezero.mcts_eval.search_over_raw_world_inputs import NativeWorldInputs, NativeInputHandle
    from test_search_over_raw_native_atomic_work import native_report
    recorder = NativeWorldInputs(INPUT_BINDING, 16, deadline_limited=False, record_atomic_work=True,record_construction=True)
    for i in range(4): world,state=construct_actual_python(recorder)
    args=arguments(iterations=64)
    args[5]=json.dumps(dict(world.party_species,turn=3))
    NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a,**k:json.dumps(native_report(64))),
        recorder).search_batched_multi_encoded(*args)
    return dict(total_iterations=64, consumed_world_inputs=recorder.finish(dict(total_iterations=64),
        attempted=4, constructed=4, searched=4))


@dataclass(frozen=True)
class SyntheticContext:
    player_id: str
    observation: object
    requested_observations: dict
    public_materialization_state: object


@dataclass(frozen=True)
class SyntheticPublicState:
    belief_engine: object


class SyntheticBelief:
    def __init__(self): self.set_source = None
    def clone(self): return SyntheticBelief()


class FixedWorkJobWiringTests(unittest.TestCase):
    """Execute the actual job method, with no real model/catalog or environment."""
    def job(self, arm="incumbent", leaf="model", statistics_root="same-root", root=None):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        cfg = SearchConfiguration(arm, leaf=leaf, seconds=1., workers=20 if arm == "reference" else 1)
        state = SyntheticPublicState(SyntheticBelief())
        observed = object()
        context = SyntheticContext("p1", observed, {"p1":observed, "p2":"never-transport"}, state)
        request = SimpleNamespace(state=state, observation=observed, set_source_hash=BINDING)
        with patch("pokezero.mcts_eval.head_to_head.public_only_context", side_effect=lambda c:c), \
             patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=request), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=INPUT_BINDING):
            return FixedWorkAdapterJob(cfg, SimpleNamespace(showdown_source_sha256=BINDING),
                "mock-showdown", context, root or statistics_root + ":" + cfg.identity, 7,
                statistics_root, BINDING, reference_factory="mock-worker-factory")

    def test_actual_job_dispatches_reviewed_witness_adapter_for_all_diagnostic_leaves(self):
        ownership = dict(protocol="synthetic-owned-groups")
        for arm in ("incumbent", "reference"):
            for leaf in ("model", "hp_fraction", "raw_rollout"):
                with self.subTest(arm=arm, leaf=leaf):
                    job = self.job(arm, leaf)
                    phases = []
                    source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
                    selected = dict(synthetic=True, runtime_sha256=BINDING, evidence=selection_inputs(arm))
                    adapter = Mock(runtime_sha256=BINDING); adapter.select.return_value = selected
                    with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
                         patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter",
                               return_value=adapter) as factory:
                        result = job(ownership, phases.append)
                    self.assertEqual(result, selected)
                    self.assertEqual(phases, ["startup", "selection", "cleanup"])
                    self.assertEqual(factory.call_args.args, (job.configuration,))
                    self.assertEqual(factory.call_args.kwargs.get("owned_process_receipts"),
                        ownership if arm == "reference" else None)
                    kwargs = adapter.select.call_args.kwargs
                    self.assertEqual((kwargs["statistics_root_id"], kwargs["root_id"],
                        kwargs["selection_seed"]), ("same-root", job.root_id, 7))
                    actor = adapter.select.call_args.args[0]
                    self.assertEqual(set(actor.requested_observations), {"p1"})
                    self.assertIs(actor.public_materialization_state.belief_engine.set_source, source)
                    self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
                    adapter.close.assert_called_once()

    def test_parent_acceptance_derives_observed_work_from_bound_actual_call_roster(self):
        from copy import deepcopy
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        contract=FixedWorkDiagnosticContract()
        for arm in ("incumbent","reference"):
            job=self.job(arm)
            selected=dict(root_id=job.root_id,configuration_sha256=job.configuration.identity,
                statistics_root_id=job.statistics_root_id,selection_seed=job.selection_seed,
                runtime_sha256=job.expected_runtime_sha256,evidence=selection_inputs(arm),
                fixed_work_diagnostic=dict(contract=contract.receipt(),accepted_actual=contract.accepted_boundary(arm)))
            def call(worker,**kwargs):
                self.assertIs(worker,job)
                return kwargs["validate_result"](selected)
            with self.subTest(arm=arm),patch("pokezero.mcts_eval.search_over_raw_supervision._supervise_owned_job",side_effect=call):
                value=supervise_fixed_work(job,directory="not-created",binding_sha256=BINDING,
                    verify_admission=lambda:None)
                self.assertEqual(value["accepted_units"],contract.accepted_boundary(arm))
                self.assertEqual(value["attempted_units"],contract.accepted_boundary(arm))
                self.assertFalse(value["resource_cost_complete"])
                selected["fixed_work_diagnostic"]["accepted_actual"] *= 2
                with self.assertRaises(ValueError): supervise_fixed_work(job,directory="not-created",
                    binding_sha256=BINDING,verify_admission=lambda:None)

    def test_job_failure_still_closes_owned_adapter_without_retry(self):
        job = self.job()
        adapter = Mock(runtime_sha256=BINDING); adapter.select.side_effect = RuntimeError("synthetic refusal")
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
        phases = []
        with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter",
                   return_value=adapter) as factory:
            with self.assertRaises(RuntimeError): job({}, phases.append)
        factory.assert_called_once(); adapter.select.assert_called_once(); adapter.close.assert_called_once()
        self.assertEqual(phases, ["startup", "selection", "cleanup"])

    def test_missing_or_aliased_statistics_binding_rejected_before_adapter(self):
        with self.assertRaises(ValueError): self.job(statistics_root="")
        with self.assertRaises(ValueError): self.job(root="receipt-without-shared-root-binding")

    def test_worker_catalog_drift_prevents_adapter_start(self):
        job = self.job()
        phases = []
        with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=SimpleNamespace(
                metadata=SimpleNamespace(source_hash="b"*64))), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter") as factory:
            with self.assertRaises(ValueError): job({}, phases.append)
        factory.assert_not_called()
        self.assertEqual(phases, ["startup", "cleanup"])

    def test_unreviewed_runtime_rejected_before_search_and_adapter_closed(self):
        job = self.job()
        adapter = Mock(runtime_sha256="b"*64)
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter",
                   return_value=adapter):
            with self.assertRaises(ValueError): job({}, lambda phase:None)
        adapter.select.assert_not_called(); adapter.close.assert_called_once()

    def test_missing_or_conflicting_input_witness_cannot_return_success(self):
        from copy import deepcopy
        for mode in ("missing", "root", "partial", "count", "call", "construction-flag", "construction-receipt"):
            job = self.job()
            evidence = deepcopy(selection_inputs("incumbent"))
            if mode == "missing": evidence.clear()
            if mode == "root": evidence["consumed_world_inputs"]["root_binding"] = "b"*64
            if mode == "partial": evidence["consumed_world_inputs"]["rows"][0]["consumed"] = False
            if mode == "count": evidence["total_iterations"] = 63
            if mode == "call": evidence["consumed_world_inputs"]["calls"][0]["construction_ordinals"] = [0, 0, 1, 2]
            if mode == "construction-flag": evidence["consumed_world_inputs"].pop("construction_observed")
            if mode == "construction-receipt": evidence["consumed_world_inputs"]["rows"][0].pop("construction_receipt")
            adapter = Mock(runtime_sha256=BINDING)
            adapter.select.return_value = dict(runtime_sha256=BINDING, evidence=evidence)
            source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
            with self.subTest(mode=mode), \
                 patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
                 patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter",
                       return_value=adapter):
                with self.assertRaises(ValueError): job({}, lambda phase:None)
            adapter.close.assert_called_once()

    def test_reference_allocation_checker_retains_full_symmetric_roster(self):
        from copy import deepcopy
        from pokezero.mcts_eval.search_over_raw_world_inputs import validate_fixed_work_inputs
        evidence = selection_inputs("reference")
        original = validate_fixed_work_inputs(evidence, arm="reference", binding=INPUT_BINDING,
            statistics_root_id="same-root")
        for mode in ("missing", "worker", "duplicate", "count", "scope"):
            changed = deepcopy(evidence)
            rows = changed["consumed_world_inputs"]["rows"]
            if mode == "missing": rows.pop()
            if mode == "worker": rows[-1]["worker"] = 20
            if mode == "duplicate": rows[-1] = deepcopy(rows[0])
            if mode == "count": changed["result"]["trajectories"] = 79
            if mode == "scope": changed["consumed_world_inputs"]["qualifies_uninstrumented_runtime"] = True
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError): validate_fixed_work_inputs(changed,
                    arm="reference", binding=INPUT_BINDING, statistics_root_id="same-root")
        self.assertEqual(validate_fixed_work_inputs(evidence, arm="reference", binding=INPUT_BINDING,
            statistics_root_id="same-root"), original)

    def test_reference_atomic_draw_manifest_and_external_root_must_agree(self):
        from copy import deepcopy
        from pokezero.mcts_eval.search_over_raw_world_inputs import validate_fixed_work_inputs
        original = selection_inputs("reference")
        for mode in ("hash", "battle", "atomic", "owner", "root"):
            evidence=deepcopy(original)
            if mode == "hash": evidence["consumed_world_inputs"]["rows"][0]["input_sha256"]="d"*64
            if mode == "battle":
                for receipt in evidence["worker_receipts"]:
                    receipt["own_statistics"]["battle_id"]="search-over-raw-root:substituted-root"
            if mode == "atomic": evidence["worker_receipts"][0]["evidence"].pop("atomic_work")
            if mode == "owner":
                evidence["worker_receipts"][0]["evidence"]["draws"][0]["consumed_world_input"]["owned_hypothetical_only"]=False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_fixed_work_inputs(evidence,arm="reference",binding=INPUT_BINDING,
                    statistics_root_id="different-root" if mode == "root" else "same-root")


@dataclass(frozen=True)
class SyntheticJob:
    mode: str = "complete"
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup")
        phase("selection")
        child = None
        if self.mode == "memory_children":
            children, receipts = {}, {}
            for i in range(2):
                children[i] = subprocess.Popen([sys.executable,"-c",
                    "import sys,time; sys.stdin.buffer.read(); payload=bytearray(b'x'*(12*1024*1024)); time.sleep(.7)"],
                    stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                receipts[i] = record_owned_process(children[i].pid,ownership)
                children[i].stdin.close()
            exits = {i:child.wait() for i,child in children.items()}
            phase("cleanup")
            close_owned_generations(receipts,{i:SimpleNamespace(pid=child.pid,exitcode=exits[i])
                for i,child in children.items()})
            return dict(synthetic=True,child_pids=[child.pid for child in children.values()])
        if self.mode in {"cpu_child", "hang_child", "leak_child", "frame_fail_child"}:
            code = ("import sys,time,resource,json; sys.stdin.buffer.read(); "
                "began=time.process_time(); "
                "exec('while time.process_time()-began<.12: pass'); "
                "print(json.dumps({'cpu_seconds':time.process_time()-began}),flush=True)"
                if self.mode == "cpu_child" else "import sys,time; sys.stdin.buffer.read(); time.sleep(60)")
            child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
            path = record_owned_process(child.pid, ownership)
            child.stdin.close()
            if self.mode in {"leak_child", "frame_fail_child"}:
                phase("cleanup")
                if self.mode == "frame_fail_child":
                    def broken_capture(_failure):
                        raise OSError("PRIVATE_CAPTURE_ERROR")
                    phase.__globals__["_safe_error_frames"] = broken_capture
                    raise RuntimeError("PRIVATE_ORIGINAL_ERROR")
                return dict(synthetic=True, child_pid=child.pid)
            output = child.stdout.read()
            code = child.wait()
            child.stdout.close()
            phase("cleanup")
            close_owned_generations({0: path}, {0: SimpleNamespace(pid=child.pid, exitcode=code)})
            return dict(synthetic=True, child_cpu_seconds=json.loads(output)["cpu_seconds"])
        if self.mode == "hang":
            time.sleep(60)
        if self.mode == "exit":
            os._exit(7)
        if self.mode == "exit_zero":
            os._exit(0)
        phase("cleanup")
        if self.mode == "fail":
            raise RuntimeError("PRIVATE_DETAIL_MUST_NOT_BE_REPORTED")
        if self.mode in {"contract_fail", "contract_fail_during_cleanup"}:
            from pokezero.mcts_eval.policy_opponent_profile import validate_selection
            try:
                validate_selection({}, arm="incumbent_mcts", mode="matched_deadline", config=None,
                    mask=(), opponent_seed=0, deadline_ms=1000, native_batch_guard_ms=64)
            finally:
                if self.mode == "contract_fail_during_cleanup":
                    raise RuntimeError("PRIVATE_CLEANUP_DETAIL")
        return dict(synthetic=True)


class FixedWorkSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_job(self, mode="complete", *, limit=None, directory=None):
        return supervise_fixed_work(SyntheticJob(mode), directory=directory or self.root / "attempt",
            binding_sha256=BINDING, verify_admission=lambda: None, _synthetic_watchdog_seconds=limit)

    def test_complete_actual_exit_phase_costs_and_create_only_attempt(self):
        value = self.run_job()
        self.assertEqual(value["status"], "DIAGNOSTIC_COMPLETE")
        self.assertEqual((value["supervisor_exit_code"], value["worker_exit_code"]), (0, 0))
        self.assertTrue(value["cpu_wall_cost_complete"])
        self.assertFalse(value["resource_cost_complete"])
        self.assertFalse(value["peak_tree_memory_measured"])
        self.assertFalse(value["runtime_qualified"])
        self.assertFalse(value["deployment_eligible"])
        self.assertTrue(all(v >= 0 for v in value["phase_costs"].values()))
        self.assertEqual(value["watchdog_seconds"], 1800)
        self.assertGreater(value["memory_measurement"]["maximum_observed_rss_sum_bytes"],0)
        self.assertFalse(value["memory_measurement"]["protocol"]["exact_peak_measured"])
        with self.assertRaises(FileExistsError):
            self.run_job()

    def test_memory_sampler_observes_overlapping_owned_generations_and_recomputes_receipt(self):
        from pokezero.mcts_eval.search_over_raw_resources import validate_memory_receipt
        value=self.run_job("memory_children")
        self.assertEqual(value["status"],"DIAGNOSTIC_COMPLETE")
        directory=self.root/"attempt"
        rows=json.loads((directory/"memory-trace.json").read_bytes())["samples"]
        children=set(value["result"]["child_pids"])
        self.assertTrue(any(children <= {r["pid"] for r in sample["rows"] or []} for sample in rows))
        self.assertGreater(value["memory_measurement"]["maximum_observed_rss_sum_bytes"],24*1024*1024)
        self.assertEqual(value["memory_measurement"]["missed_launch_generations"],[])
        self.assertGreaterEqual(value["memory_measurement"]["sampling_child_cpu_seconds"],0)
        self.assertGreaterEqual(value["supervisor_thread_cpu_seconds"],0)
        self.assertFalse(value["resource_cost_complete"])
        validate_memory_receipt(directory,binding=BINDING,
            main_launch=json.loads((directory/"launch.json").read_bytes()),expected_summary=value["memory_measurement"])

    def test_sampling_timeout_does_not_obscure_job_or_watchdog_cleanup(self):
        from pokezero.mcts_eval.search_over_raw_resources import SamplingFailure, _process_table
        from pokezero.mcts_eval.search_over_raw_observer_cost import _PURPOSE
        def rss_only_failure(timeout):
            if _PURPOSE.get() == "rss":
                raise SamplingFailure("query_timeout", .01)
            return _process_table(timeout)
        with patch("pokezero.mcts_eval.search_over_raw_resources._process_table",
                side_effect=rss_only_failure):
            value=self.run_job("hang_child",limit=.75)
        self.assertEqual(value["status"],"DIAGNOSTIC_INCOMPLETE")
        self.assertEqual(value["supervisor_exit_code"],124)
        self.assertIsNone(value["cleanup_error_type"])
        self.assertIsNone(value["memory_measurement"]["maximum_observed_rss_sum_bytes"])
        self.assertFalse(value["memory_measurement"]["all_declared_launch_generations_observed_without_query_errors"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_auxiliary_summary_or_write_failure_never_replaces_primary_or_erases_terminal(self):
        from pokezero.mcts_eval.search_over_raw_supervision import _save_new
        for mode in ("summary","write"):
            directory=self.root/mode
            def save(path,value):
                if path.name=="memory-trace.json": raise OSError("PRIVATE_FAIL_DETAIL")
                return _save_new(path,value)
            target=("pokezero.mcts_eval.search_over_raw_supervision._save_new" if mode=="write"
                else "pokezero.mcts_eval.search_over_raw_resources.summarize_memory_trace")
            kwargs=dict(side_effect=save) if mode=="write" else dict(side_effect=ValueError("PRIVATE_FAIL_DETAIL"))
            with self.subTest(mode=mode),patch(target,**kwargs):
                value=self.run_job("fail",directory=directory)
            self.assertEqual(value["status"],"OPERATIONAL_STOP")
            self.assertEqual(value["worker_exit_code"],1)
            self.assertEqual(value["error_type"],"ValueError")
            self.assertEqual(value["resource_measurement_error_type"],"OSError" if mode=="write" else "ValueError")
            self.assertIsNone(value["memory_measurement"])
            self.assertEqual(json.loads((directory/"terminal.json").read_bytes()),value)
            self.assertNotIn("PRIVATE_FAIL_DETAIL",json.dumps(value))

    def test_cpu_receipt_includes_normal_reaped_child(self):
        value = self.run_job("cpu_child")
        self.assertEqual(value["status"], "DIAGNOSTIC_COMPLETE")
        self.assertGreaterEqual(value["cpu_seconds_including_reaped_children"],
            value["result"]["child_cpu_seconds"])
        self.assertGreater(value["cpu_seconds_including_reaped_children"], .1)
        self.assertFalse(list((self.root / "attempt" / "owned-groups" / "active").iterdir()))

    def test_timeout_incomplete_and_actual_signal_exit_no_result_no_retry(self):
        value = self.run_job("hang", limit=.5)
        self.assertEqual(value["status"], "DIAGNOSTIC_INCOMPLETE")
        self.assertEqual(value["supervisor_exit_code"], 124)
        self.assertIsNotNone(value["worker_exit_code"])
        self.assertLess(value["worker_exit_code"], 0)
        self.assertIsNone(value["result"])
        self.assertFalse(value["cpu_wall_cost_complete"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))
        with self.assertRaises(FileExistsError):
            self.run_job()

    def test_timeout_cleans_recorded_detached_child_group(self):
        value = self.run_job("hang_child", limit=.75)
        self.assertEqual(value["status"], "DIAGNOSTIC_INCOMPLETE")
        self.assertGreaterEqual(len(value["owned_groups"]), 2)
        self.assertIsNone(value["cleanup_error_type"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_live_descendant_after_worker_exit_zero_is_operational_failure(self):
        value = self.run_job("leak_child")
        self.assertEqual(value["worker_exit_code"], -signal.SIGKILL)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertEqual(value["supervisor_exit_code"], 1)
        self.assertIsNone(value["result"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_nonzero_and_missing_terminal_receipt_cannot_be_inferred_success(self):
        value = self.run_job("exit")
        self.assertEqual(value["worker_exit_code"], 7)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertFalse(value["cpu_wall_cost_complete"])
        self.assertIsNone(value["result"])

    def test_failure_details_sanitized_but_actual_exit_retained(self):
        value = self.run_job("fail")
        self.assertEqual(value["worker_exit_code"], 1)
        self.assertNotIn("PRIVATE_DETAIL", (self.root / "attempt" / "terminal.json").read_text())
        raw = (self.root / "attempt" / "worker-result.json").read_text()
        self.assertNotIn("PRIVATE_DETAIL", raw)
        self.assertNotIn(str(self.root), raw)
        self.assertEqual(json.loads(raw)["error_type"], "RuntimeError")

    def test_failure_keeps_bounded_repository_code_sites_without_payloads(self):
        value = self.run_job("contract_fail")
        self.assertEqual(value["worker_exit_code"], 1)
        raw = (self.root / "attempt" / "worker-result.json").read_text()
        worker = json.loads(raw)
        self.assertEqual(worker["error_type"], "ContractError")
        frames = worker["error_frames"]
        self.assertTrue(0 < len(frames) <= 16)
        self.assertTrue(any(row["module"] == "pokezero.mcts_eval.policy_opponent_profile"
            and row["function"] == "validate_selection" for row in frames))
        self.assertTrue(all(set(row) == {"module", "function", "line"}
            and row["module"].startswith("pokezero.") and type(row["line"]) is int
            and row["line"] > 0 for row in frames))
        self.assertNotIn("profile has no serialized selected action", raw)
        self.assertNotIn(str(self.root), raw)
        self.assertFalse(value["cpu_wall_cost_complete"])

    def test_frame_capture_failure_preserves_original_receipt_and_owned_cleanup(self):
        value = self.run_job("frame_fail_child")
        raw = (self.root / "attempt" / "worker-result.json").read_text()
        worker = json.loads(raw)
        self.assertEqual(worker["error_type"], "RuntimeError")
        self.assertEqual(worker["error_frames"], [])
        self.assertTrue(worker["cleanup_pending"])
        self.assertNotIn("PRIVATE_", raw)
        self.assertNotEqual(value["worker_exit_code"], 0)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertFalse(value["cpu_wall_cost_complete"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_cleanup_error_does_not_hide_primary_repository_failure_site(self):
        value = self.run_job("contract_fail_during_cleanup")
        raw = (self.root / "attempt" / "worker-result.json").read_text()
        worker = json.loads(raw)
        self.assertEqual(worker["error_type"],"RuntimeError")
        self.assertTrue(any(r["module"] == "pokezero.mcts_eval.policy_opponent_profile"
            and r["function"] == "validate_selection" for r in worker["error_frames"]))
        self.assertLessEqual(len(worker["error_frames"]),16)
        self.assertNotIn("PRIVATE_CLEANUP_DETAIL",raw)

    def test_exit_zero_without_terminal_evidence_is_not_success(self):
        value = self.run_job("exit_zero")
        self.assertEqual(value["worker_exit_code"], 0)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertIsNone(value["result"])

    def test_false_admission_prevents_launch_and_claim(self):
        with patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                supervise_fixed_work(SyntheticJob(), directory=self.root / "absent",
                    binding_sha256=BINDING, verify_admission=lambda: False)
            launch.assert_not_called()
        self.assertFalse((self.root / "absent").exists())

    def test_revoked_admission_rejects_successful_worker_result(self):
        replies = iter((True, False))
        value = supervise_fixed_work(SyntheticJob(), directory=self.root / "attempt",
            binding_sha256=BINDING, verify_admission=lambda: next(replies))
        self.assertEqual(value["worker_exit_code"], 0)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertIsNone(value["result"])

    def test_unbound_active_index_never_queries_or_signals_its_pid(self):
        directory = self.root / "unbound"
        active = directory / "owned-groups" / "active"
        active.mkdir(parents=True)
        (active / "group-123-fixture.json").write_text("unbound bytes")
        with patch("pokezero.mcts_eval.search_over_raw_supervision.owned_process_identities") as query:
            with self.assertRaises(ValueError):
                _groups(directory, 456, BINDING)
            query.assert_not_called()

    def test_identity_failure_before_handshake_still_reaps_exact_child(self):
        with patch("pokezero.mcts_eval.search_over_raw_supervision.owned_process_identity", return_value=None):
            value = self.run_job()
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertIsNotNone(value["worker_exit_code"])
        self.assertIsNone(value["cleanup_error_type"])
        with self.assertRaises(ProcessLookupError):
            os.kill(value["worker_pid"], 0)

    def test_exited_unadmitted_child_keeps_wait4_exit_and_usage(self):
        child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(7)"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(.15)  # Deliberately leave an exited child unreaped.
        with patch.object(child, "poll", side_effect=AssertionError("waitpid must not own receipt")):
            receipt = _stop_unadmitted_child(child)
        child.returncode = receipt[0]
        self.assertEqual(receipt[0], 7)
        self.assertGreaterEqual(receipt[1].ru_utime + receipt[1].ru_stime, 0)

    def test_admission_failure_precedes_claim_and_process_launch(self):
        def refuse():
            raise ValueError("not admitted")
        with patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                supervise_fixed_work(SyntheticJob(), directory=self.root / "absent",
                    binding_sha256=BINDING, verify_admission=refuse)
            launch.assert_not_called()
        self.assertFalse((self.root / "absent").exists())

    def test_closed_historical_launch_not_queried_as_active_pid(self):
        directory = self.root / "closed"
        (directory / "owned-groups" / "active").mkdir(parents=True)
        (directory / "owned-groups" / "group-123-fixture.json").write_text("unrelated historical bytes")
        with patch("pokezero.mcts_eval.search_over_raw_supervision.owned_process_identities") as query:
            self.assertEqual(_groups(directory, 456, BINDING), {456})
            query.assert_not_called()

    def test_phase_clocks_use_worker_terminal_not_parent_post_verification(self):
        values = _phase_costs([dict(phase=p, monotonic_seconds=t) for p, t in
            (("startup", 10.), ("selection", 12.), ("cleanup", 17.))], 18.)
        self.assertEqual(values, dict(startup_wall_seconds=2., selection_wall_seconds=5., cleanup_wall_seconds=1.))
        with self.assertRaises(ValueError):
            _phase_costs([], 1.)


if __name__ == "__main__":
    unittest.main()
