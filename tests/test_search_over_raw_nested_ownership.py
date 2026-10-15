"""Disposable real processes; nested authority never implies collection."""
from dataclasses import dataclass
import json
import os
import pickle
import hashlib
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from pokezero.mcts_eval.search_over_raw_nested_ownership import (
    MAX_DEPTH, OwnedScope, current_scope, worker_scope, check_capacity, verify_delegated_closure,
    DiagnosticIncompletePolicy,
)
from pokezero.mcts_eval.paper_reference_parallel import live_owned_groups
from test_search_over_raw_supervision import SyntheticJob

BINDING = "e" * 64


def restore_scoped_job(directory):
    from pokezero.mcts_eval.search_over_raw_ledger import _write_new
    scope = current_scope()
    assert scope is not None
    _write_new(Path(directory) / "reconstruction.json", dict(pid=scope.pid, depth=scope.depth))
    return SyntheticJob()


@dataclass(frozen=True)
class ScopedReconstructionJob:
    directory: str
    synthetic_process_fixture = True
    def __reduce__(self):
        return restore_scoped_job, (self.directory,)


@dataclass(frozen=True)
class NestedJob:
    directory: str
    mode: str = "complete"
    level: int = 0
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        import pokezero.mcts_eval.search_over_raw_supervision as supervision
        phase("startup"); phase("selection")
        root = Path(self.directory)
        scope = current_scope()
        assert scope.pid == os.getpid() and scope.directory == Path(ownership["directory"]).parent
        def nested(job, label="child", limit=None, local_incomplete=False):
            policy = (DiagnosticIncompletePolicy("1"*64, "2"*64,
                hashlib.sha256(pickle.dumps(job)).hexdigest(), BINDING) if local_incomplete else None)
            return supervise_fixed_work(job, directory=root / label, binding_sha256=BINDING,
                verify_admission=lambda: True, _synthetic_watchdog_seconds=limit,
                diagnostic_incomplete_policy=policy)
        if self.mode in ("bad_ordinary", "bad_delegated"):
            from pokezero.mcts_eval.search_over_raw_ledger import _write_new
            active = scope.directory / ("owned-groups" if self.mode == "bad_ordinary" else "delegated-groups") / "active"
            active.mkdir(parents=True, exist_ok=True)
            _write_new(active / ("group-000-invalid.json" if self.mode == "bad_ordinary" else "child-000-invalid.json"), {})
        if self.mode in ("pending_unpublished", "pending_published"):
            original = supervision.publish_delegation
            def barrier(scope, directory, launch):
                permission = original(scope, directory, launch) if self.mode == "pending_published" else None
                supervision._save_new(root / "barrier.json", dict(child_pid=launch["pid"],
                    parent_pid=os.getpid(), published=permission is not None))
                signal.pause()  # Exact launch-before-permission barrier; no race sleeps.
                raise AssertionError("barrier cannot resume as success")
            with patch.object(supervision, "publish_delegation", side_effect=barrier):
                nested(SyntheticJob())
        elif self.mode == "partial_ack":
            original = supervision.subprocess.Popen
            def bootstrap(command, **kwargs):
                if "-m" in command and supervision.__name__ in command:
                    code = "\n".join((
                        "import sys,os,signal", "from pathlib import Path",
                        "import pokezero.mcts_eval.search_over_raw_nested_ownership as owned",
                        "import pokezero.mcts_eval.search_over_raw_supervision as supervision",
                        "real = owned._write_new",
                        "def barrier(path,value):",
                        "    if path.name == 'nested-detach.json':",
                        "        with path.open('x') as stream:",
                        "            stream.write('{'); stream.flush(); os.fsync(stream.fileno())",
                        "        signal.pause()",
                        "        raise AssertionError('barrier cannot resume')",
                        "    return real(path,value)",
                        "owned._write_new = barrier",
                        "raise SystemExit(supervision._worker(sys.argv[1],sys.argv[2]))"))
                    command = [sys.executable, "-B", "-c", code, command[-2], command[-1]]
                return original(command, **kwargs)
            with patch.object(supervision.subprocess, "Popen", side_effect=bootstrap):
                nested(SyntheticJob())
        elif self.mode == "depth":
            nested(NestedJob(str(root / f"level-{self.level}"), "depth", self.level + 1))
        elif self.mode == "missing_close":
            def barrier(permission, receipt):
                supervision._save_new(root / "barrier.json", dict(child_pid=receipt["worker_pid"],
                    child_exit=receipt["worker_exit_code"]))
                os._exit(7)  # Child was reaped; direct parent dies before close publication.
            with patch.object(supervision, "close_delegation", side_effect=barrier):
                nested(SyntheticJob())
        else:
            mode = {"hang_nested":"hang", "hang_grandchild":"hang_child", "swallow_failure":"fail",
                "clean_incomplete":"hang", "clean_pool_incomplete":"hang_child", "policy_failure":"fail",
                "bad_ordinary":"hang", "bad_delegated":"hang"}.get(
                self.mode, "cpu_child" if self.mode == "cpu" else "complete")
            job = ScopedReconstructionJob(str(root)) if self.mode == "unpickle" else SyntheticJob(mode)
            local = self.mode in {"clean_incomplete", "clean_pool_incomplete", "policy_failure", "subtree_incomplete"}
            if self.mode == "subtree_incomplete":
                job = NestedJob(str(root / "subtree"), "hang_nested")
            # This case tests an exception, not a startup-time watchdog race.
            result = nested(job, limit=1.5 if local else 5. if self.mode == "swallow_failure" else None,
                local_incomplete=local)
            # Deliberately swallow child failure. Ancestor's closure audit must reject this.
            phase("cleanup")
            return dict(synthetic=True, child_status=result["status"],
                child_cpu=result["cpu_seconds_including_reaped_children"],
                child_pid=result["worker_pid"], depth=scope.depth)


class NestedOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_job(self, mode="complete", limit=None):
        return supervise_fixed_work(NestedJob(str(self.root), mode), directory=self.root / "outer",
            binding_sha256=BINDING, verify_admission=lambda: True, _synthetic_watchdog_seconds=limit)

    def test_actual_nested_success_bound_detach_wait_and_closure(self):
        value = self.run_job()
        self.assertEqual(value["status"], "DIAGNOSTIC_COMPLETE")
        self.assertEqual(value["result"]["child_status"], "DIAGNOSTIC_COMPLETE")
        child = json.loads((self.root / "child" / "launch.json").read_bytes())
        self.assertEqual(json.loads((self.root / "child" / "nested-detach.json").read_bytes())["depth"], 1)
        self.assertEqual(child["supervisor_pid"], value["worker_pid"])
        registry = self.root / "outer" / "delegated-groups"
        self.assertEqual(list((registry / "active").iterdir()), [])
        self.assertEqual(len(list(registry.glob("closed-child-*.json"))), 1)
        self.assertFalse(value["resource_cost_complete"])
        self.assertFalse(value["runtime_qualified"])

    def test_scope_installed_before_trusted_job_reconstruction(self):
        value = self.run_job("unpickle")
        self.assertEqual(value["supervisor_exit_code"], 0)
        reconstructed = json.loads((self.root / "reconstruction.json").read_bytes())
        self.assertEqual(reconstructed, dict(pid=value["result"]["child_pid"], depth=1))

    def test_explicit_clean_timeout_closes_control_not_normal_work(self):
        from pokezero.mcts_eval.search_over_raw_attempt_cost import AttemptCostSpec, reconcile_attempt_cost_roster
        from pokezero.mcts_eval.search_over_raw import digest
        value = self.run_job("clean_incomplete")
        self.assertEqual(value["supervisor_exit_code"], 0)
        terminal = json.loads((self.root / "child" / "terminal.json").read_bytes())
        self.assertEqual(terminal["supervisor_exit_code"], 124)
        self.assertIsInstance(terminal["worker_exit_code"], int)
        self.assertIsNone(terminal["result"])
        self.assertIsNone(terminal["observed_work_accounting"])
        registry = self.root / "outer" / "delegated-groups"
        closed = json.loads(next(registry.glob("closed-child-*.json")).read_bytes())
        self.assertFalse(closed["normal"])
        self.assertTrue(closed["clean_diagnostic_incomplete"])
        self.assertEqual(list((registry / "active").iterdir()), [])
        specs=[]
        for label,parent in (("outer",None),("child",str(self.root/"outer"))):
            directory=self.root/label
            claim=json.loads((directory/"claim.json").read_bytes())
            term=json.loads((directory/"terminal.json").read_bytes())
            specs.append(AttemptCostSpec(str(directory),claim["job_sha256"],BINDING,"fixed-work","selection",
                claim["contract"],claim["resource_measurement_source_sha256"],term["watchdog_seconds"],parent))
        specs=tuple(specs)
        from dataclasses import asdict
        costs=reconcile_attempt_cost_roster(specs,attempt_root=self.root,roster_sha256=digest([asdict(s) for s in specs]))
        self.assertEqual(costs["delegation_errors"], [])
        self.assertTrue(costs["attempts"][0]["control_descendant_closure_validated"])
        self.assertFalse(costs["attempts"][0]["normal_descendant_closure_validated"])
        self.assertEqual(costs["attempts"][1]["state"],"REAPED_FAILURE_COST_OBSERVED")
        self.assertFalse(costs["attempts"][1]["work_validated"])

    def test_clean_timeout_with_forced_ordinary_pool_keeps_abnormal_receipts(self):
        value=self.run_job("clean_pool_incomplete")
        self.assertEqual(value["supervisor_exit_code"],0)
        child=self.root/"child"
        self.assertTrue(list((child/"owned-groups"/"active").glob("group-*.json")))
        terminal=json.loads((child/"terminal.json").read_bytes())
        self.assertFalse(live_owned_groups(set(terminal["owned_groups"])))

    def test_policy_does_not_swallow_integrity_failure_or_supervised_subtree(self):
        for mode in ("policy_failure", "subtree_incomplete"):
            with self.subTest(mode=mode):
                root=self.root/mode;root.mkdir()
                value=supervise_fixed_work(NestedJob(str(root),mode),directory=root/"outer",
                    binding_sha256=BINDING,verify_admission=lambda:True)
                self.assertEqual(value["supervisor_exit_code"],1)
                self.assertIsNone(value["result"])

    def test_clean_exception_requires_original_policy_cleanup_and_wait(self):
        from pokezero.mcts_eval.search_over_raw_nested_ownership import _clean_diagnostic_incomplete, _read
        value=self.run_job("clean_incomplete")
        registry=self.root/"outer"/"delegated-groups"
        path=next(registry.glob("child-*.json"))
        row,child=_read(path,self.root/"outer",value["worker_pid"],BINDING,0)
        terminal=json.loads((child/"terminal.json").read_bytes())
        for field,bad in (("worker_exit_code",None),("cpu_seconds_including_reaped_children",None),
                ("cleanup_error_type","ValueError"),("observed_work_accounting",{}),("watchdog_reached",False)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                _clean_diagnostic_incomplete(row,child,dict(terminal,**{field:bad}))
        claim_path=child/"claim.json"
        original=claim_path.read_bytes()
        altered=json.loads(original);altered["diagnostic_incomplete_policy"]["slot_sha256"]="3"*64
        claim_path.write_text(json.dumps(altered))
        with self.assertRaises(ValueError):_clean_diagnostic_incomplete(row,child,terminal)
        claim_path.write_bytes(original)
        policy_row=dict(row,diagnostic_incomplete_policy=None)
        self.assertFalse(_clean_diagnostic_incomplete(policy_row,child,terminal))

    def test_policy_binds_job_before_any_claim_or_spawn(self):
        job=SyntheticJob("hang")
        policy=DiagnosticIncompletePolicy("1"*64,"2"*64,"3"*64,BINDING)
        with self.assertRaises(ValueError):
            supervise_fixed_work(job,directory=self.root/"never",binding_sha256=BINDING,
                verify_admission=lambda:True,diagnostic_incomplete_policy=policy,_synthetic_watchdog_seconds=.6)
        self.assertFalse((self.root/"never").exists())

    def test_partial_detach_ack_preserves_cleanup_but_never_complete_evidence(self):
        value = self.run_job("partial_ack", limit=1.5)
        self.assertEqual((self.root / "child" / "nested-detach.json").read_bytes(), b"{")
        child = json.loads((self.root / "child" / "launch.json").read_bytes())["pid"]
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertEqual(value["supervisor_exit_code"], 1)
        self.assertIsNotNone(value["cleanup_error_type"])
        self.assertIsNotNone(value["worker_exit_code"])
        self.assertIn(child, value["owned_groups"])
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_bad_inventory_does_not_strand_independent_valid_nested_worker(self):
        for mode in ("bad_ordinary", "bad_delegated"):
            with self.subTest(mode=mode):
                root = self.root / mode; root.mkdir()
                value = supervise_fixed_work(NestedJob(str(root), mode), directory=root / "outer",
                    binding_sha256=BINDING, verify_admission=lambda: True, _synthetic_watchdog_seconds=1.5)
                child = json.loads((root / "child" / "launch.json").read_bytes())["pid"]
                self.assertEqual(value["status"], "OPERATIONAL_STOP")
                self.assertIn(child, value["owned_groups"])
                self.assertIsNotNone(value["worker_exit_code"])
                self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_late_result_callback_cannot_rewrite_original_closure_bytes(self):
        from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        def mutate(result):
            path = next((self.root / "outer" / "delegated-groups").glob("closed-child-*.json"))
            path.write_text(path.read_text() + "\n")  # Same JSON, different original bytes.
            return None
        value = _supervise_owned_job(NestedJob(str(self.root)), directory=self.root / "outer",
            binding_sha256=BINDING, verify_admission=lambda: True, contract=FixedWorkDiagnosticContract(),
            kind="fixed-work", work_phase="selection", validate_result=mutate)
        self.assertEqual(value["worker_exit_code"], 0)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertIsNone(value["result"])

    def test_wait4_cpu_contains_nested_child_without_double_sum(self):
        value = self.run_job("cpu")
        self.assertEqual(value["status"], "DIAGNOSTIC_COMPLETE")
        self.assertGreaterEqual(value["cpu_seconds_including_reaped_children"], value["result"]["child_cpu"])
        self.assertGreater(value["result"]["child_cpu"], .1)

    def test_outer_watchdog_cleans_detached_nested_worker_and_ordinary_grandchild(self):
        for mode in ("hang_nested", "hang_grandchild"):
            with self.subTest(mode=mode):
                value = supervise_fixed_work(NestedJob(str(self.root / mode), mode),
                    directory=self.root / (mode + "-outer"), binding_sha256=BINDING,
                    verify_admission=lambda: True, _synthetic_watchdog_seconds=2.)
                self.assertEqual(value["supervisor_exit_code"], 124)
                self.assertIsNone(value["cleanup_error_type"])
                self.assertGreaterEqual(len(value["owned_groups"]), 3 if mode == "hang_grandchild" else 2)
                self.assertFalse(live_owned_groups(set(value["owned_groups"])))
                self.assertFalse(value["cpu_wall_cost_complete"])

    def test_startup_barriers_leave_unpublished_and_pending_children_in_owned_parent_group(self):
        for mode in ("pending_unpublished", "pending_published"):
            with self.subTest(mode=mode):
                root = self.root / mode
                root.mkdir()
                value = supervise_fixed_work(NestedJob(str(root), mode), directory=root / "outer",
                    binding_sha256=BINDING, verify_admission=lambda: True, _synthetic_watchdog_seconds=1.5)
                barrier = json.loads((root / "barrier.json").read_bytes())
                self.assertEqual(barrier["published"], mode == "pending_published")
                self.assertFalse((root / "child" / "nested-detach.json").exists())
                self.assertEqual(value["supervisor_exit_code"], 124)
                self.assertIsNone(value["cleanup_error_type"])
                self.assertFalse(live_owned_groups({barrier["parent_pid"], barrier["child_pid"]}))

    def test_ignored_nested_failure_prevents_outer_success_and_retry(self):
        value = self.run_job("swallow_failure")
        self.assertEqual(value["worker_exit_code"], -signal.SIGKILL)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertIsNone(value["result"])
        self.assertFalse(value["retries_permitted"])
        child = json.loads((self.root / "child" / "terminal.json").read_bytes())
        self.assertEqual(child["worker_exit_code"], 1)
        self.assertIs(child["watchdog_reached"], False)
        worker = json.loads((self.root / "child" / "worker-result.json").read_bytes())
        self.assertEqual(worker["error_type"], "RuntimeError")
        self.assertFalse(live_owned_groups(set(value["owned_groups"])))

    def test_parent_death_after_actual_child_exit_cannot_infer_closure(self):
        value = self.run_job("missing_close")
        self.assertEqual(value["worker_exit_code"], 7)
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        self.assertEqual(json.loads((self.root / "barrier.json").read_bytes())["child_exit"], 0)
        registry = self.root / "outer" / "delegated-groups"
        self.assertEqual(len(list((registry / "active").glob("child-*.json"))), 1)
        self.assertFalse(list(registry.glob("closed-child-*.json")))

    def test_depth_exhaustion_stops_before_extra_launch_even_if_all_parents_swallow(self):
        value = self.run_job("depth")
        self.assertEqual(value["status"], "OPERATIONAL_STOP")
        launches = list(self.root.rglob("launch.json"))
        self.assertEqual(len(launches), MAX_DEPTH + 1)
        self.assertFalse(live_owned_groups({json.loads(p.read_bytes())["pid"] for p in launches}))

    def test_scope_is_cleared_and_inherited_pid_and_depth_rejected(self):
        import pokezero.mcts_eval.search_over_raw_nested_ownership as owned
        with patch.object(owned, "_WORKER_PID", None), patch.object(owned, "_SCOPE", None):
            with worker_scope(self.root, BINDING, 0):
                self.assertEqual(current_scope().pid, os.getpid())
                with self.assertRaises(ValueError):
                    with worker_scope(self.root, BINDING, 0): pass
            self.assertIsNone(owned._SCOPE)
            with self.assertRaises(ValueError): current_scope()
        with patch.object(owned, "_WORKER_PID", None), patch.object(owned, "_SCOPE", None):
            with worker_scope(self.root, BINDING, MAX_DEPTH):
                with self.assertRaises(ValueError): check_capacity(current_scope())
        with patch.object(owned, "_SCOPE", OwnedScope(self.root, -1, BINDING, 0, threading.get_ident())):
            with self.assertRaises(ValueError): current_scope()

    def test_threaded_nested_launch_rejected_before_claim_or_process(self):
        import pokezero.mcts_eval.search_over_raw_nested_ownership as owned
        errors = []
        def threaded():
            try: supervise_fixed_work(SyntheticJob(), directory=self.root / "absent", binding_sha256=BINDING,
                verify_admission=lambda: True)
            except BaseException as failure: errors.append(failure)
        with patch.object(owned, "_WORKER_PID", None), patch.object(owned, "_SCOPE", None):
            with worker_scope(self.root, BINDING, 0), patch("subprocess.Popen") as launch:
                thread = threading.Thread(target=threaded); thread.start(); thread.join()
                launch.assert_not_called()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertFalse((self.root / "absent").exists())

    def test_symlink_or_missing_closure_cannot_be_normal_completion(self):
        value = self.run_job()
        registry = self.root / "outer" / "delegated-groups"
        path = next(registry.glob("closed-child-*.json"))
        path.rename(path.with_suffix(".saved"))
        path.symlink_to(path.with_suffix(".saved"))
        with self.assertRaises(ValueError):
            verify_delegated_closure(self.root / "outer", value["worker_pid"], BINDING, lambda *a: None)

    def test_unexpected_active_entry_and_orphan_closure_cannot_pass(self):
        value = self.run_job()
        registry = self.root / "outer" / "delegated-groups"
        stray = registry / "active" / "unbound.json"
        stray.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected active"):
            verify_delegated_closure(self.root / "outer", value["worker_pid"], BINDING, lambda *a: None)
        stray.unlink()
        (registry / "closed-child-orphan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "orphan"):
            verify_delegated_closure(self.root / "outer", value["worker_pid"], BINDING, lambda *a: None)

    def test_live_birth_or_pgid_mismatch_never_signalled(self):
        import pokezero.mcts_eval.search_over_raw_nested_ownership as owned
        for group, birth in ((9876, "other birth"), (9999, "expected birth")):
            with patch("pokezero.mcts_eval.search_over_raw_resources._process_table", return_value=(
                    [dict(pid=1234, group=group, birth=birth, state="S")], 0.)), patch.object(owned.os, "kill") as stop:
                with self.assertRaises(ValueError): owned.freeze_launcher(1234, "expected birth", {9876})
                stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
