"""Live authority in the actual collector, never historical PID permission."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import signal
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import pokezero.mcts_eval.paper_reference_parallel as parallel
from pokezero.mcts_eval.paper_reference import ReferenceConfig
import pokezero.mcts_eval.search_over_raw_supervision as supervision
import pokezero.mcts_eval.search_over_raw_nested_ownership as nested
from test_paper_reference_parallel import ToyRuntime, Request, ROOT
from test_search_over_raw_supervision import SyntheticJob, BINDING

ORIGINAL_WORKER = parallel._worker


class DiagnosticRuntime(ToyRuntime):
    def prepare(self, request):
        result = super().prepare(request)
        result.finish_diagnostics = lambda progress=None: {"authored": True}
        return result


class PartialDiagnosticPipe:
    def __init__(self, connection):
        self.connection = connection

    def recv(self):
        return self.connection.recv()

    def send(self, message):
        if message[0] == "diagnostics_done":
            os.write(self.connection.fileno(), struct.pack("!i", 100) + b"incomplete")
            os.kill(os.getpid(), signal.SIGSTOP)
        else:
            self.connection.send(message)

    def close(self):
        self.connection.close()


def partial_diagnostic_worker(connection, *args):
    ORIGINAL_WORKER(PartialDiagnosticPipe(connection), *args)


@dataclass(frozen=True)
class LaunchSentinelJob(SyntheticJob):
    def __call__(self, ownership, phase):
        Path(ownership["directory"]).parent.joinpath("authored-work-started").write_text("started")
        result=super().__call__(ownership, phase)
        return dict(result, original_scope_depth=nested.current_scope().depth)


class CollectorLiveCleanupTests(unittest.TestCase):
    def test_original_launch_registration_is_durable_before_any_execution_header(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()/"job"
            job=LaunchSentinelJob();observed=[]
            def record(descriptor):
                self.assertEqual(set(descriptor),{"directory","kind","job_sha256","launch"})
                self.assertEqual(descriptor["directory"],str(directory))
                self.assertEqual(descriptor["kind"],"fixed-work")
                self.assertEqual(descriptor["job_sha256"],hashlib.sha256(pickle.dumps(job)).hexdigest())
                self.assertEqual(descriptor["launch"]["supervisor_pid"],os.getpid())
                self.assertFalse((directory/"authored-work-started").exists())
                self.assertFalse((directory/"worker-result.json").exists())
                self.assertIsNone(os.waitid(os.P_PID,descriptor["launch"]["pid"],os.WEXITED|os.WNOHANG|os.WNOWAIT))
                supervision._save_new(directory/"authored-external-registration.json",descriptor)
                observed.append(descriptor)
                # Editing a delivered copy cannot alter the execution header.
                descriptor["launch"]["binding_sha256"]="d"*64
            with supervision.observe_original_supervisor_launch(record):
                terminal=supervision.supervise_fixed_work(job,directory=directory,binding_sha256=BINDING,
                    verify_admission=lambda:True)
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            self.assertEqual(len(observed),1)
            self.assertEqual(terminal["result"]["original_scope_depth"],0)
            self.assertEqual(terminal["binding_sha256"],BINDING)
            registered=json.loads((directory/"authored-external-registration.json").read_bytes())
            self.assertEqual(registered["launch"],json.loads((directory/"launch.json").read_bytes()))
            self.assertTrue((directory/"authored-work-started").exists())
            self.assertIsNone(supervision._ORIGINAL_LAUNCH_OBSERVER.get())

    def test_failed_external_registration_never_dispatches_and_preserves_failure(self):
        class RegistrationRefusal(RuntimeError):pass
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()/"job";seen=[]
            def refuse(descriptor):
                seen.append(descriptor)
                raise RegistrationRefusal("authored unavailable registration")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",ResourceWarning)
                with supervision.observe_original_supervisor_launch(refuse):
                    terminal=supervision.supervise_fixed_work(LaunchSentinelJob(),directory=directory,
                        binding_sha256=BINDING,verify_admission=lambda:True)
            self.assertFalse([w for w in caught if issubclass(w.category,ResourceWarning)])
            self.assertEqual(len(seen),1)
            self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
            self.assertEqual(terminal["error_type"],"RegistrationRefusal")
            self.assertEqual(terminal["worker_exit_code"],-signal.SIGKILL)
            self.assertFalse(terminal["cpu_wall_cost_complete"])
            self.assertIsNone(terminal["result"])
            self.assertFalse((directory/"authored-work-started").exists())
            self.assertFalse((directory/"worker-result.json").exists())
            self.assertIsNone(supervision._ORIGINAL_LAUNCH_OBSERVER.get())

    def test_registration_callback_cannot_mutate_original_launch_claim_or_job(self):
        for filename in ("launch.json","claim.json","job.pickle"):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as folder:
                directory=Path(folder).resolve()/"job"
                def mutate(descriptor):
                    path=directory/filename
                    path.write_bytes(path.read_bytes()+b" ")
                with supervision.observe_original_supervisor_launch(mutate):
                    terminal=supervision.supervise_fixed_work(LaunchSentinelJob(),directory=directory,
                        binding_sha256=BINDING,verify_admission=lambda:True)
                self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
                self.assertEqual(terminal["error_type"],"ValueError")
                self.assertIsNone(terminal["result"])
                self.assertFalse((directory/"authored-work-started").exists())
                self.assertFalse((directory/"worker-result.json").exists())

    def test_recursive_roster_failure_still_signals_descendants_before_owners(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            child, grandchild = root / "child", root / "grandchild"
            binding = "b" * 64
            rows = {}
            for owner, descendant, pid, depth in ((root, child, 2001, 1), (child, grandchild, 3001, 2)):
                active = owner / "delegated-groups/active"
                active.mkdir(parents=True)
                original = active.parent / "child-original.json"
                original.write_text("{}")
                os.link(original, active / original.name)
                rows[owner] = (dict(pid=pid, birth_identity="birth", binding_sha256=binding, depth=depth), descendant)
            active = grandchild / "delegated-groups/active"
            active.mkdir(parents=True)
            for label in ("one", "two"):
                (active / f"child-{label}.json").write_text("{}")
            calls = []
            receipt = (-signal.SIGKILL, object())
            def ordinary(directory, pid, binding, **kwargs):
                return {pid, pid + 1}
            with patch.object(nested, "MAX_ACTIVE", 1), \
                    patch.object(nested, "_read", side_effect=lambda path, owner, *args:rows[owner]), \
                    patch.object(nested, "_detached", return_value=True), \
                    patch.object(nested, "freeze_launcher", side_effect=lambda pid, *args, **kwargs:pid), \
                    patch.object(supervision, "freeze_launcher", return_value=1001), \
                    patch.object(supervision, "_groups", side_effect=ordinary), \
                    patch.object(supervision.os, "waitid", return_value=None), \
                    patch.object(supervision, "owned_process_identity", return_value="birth"), \
                    patch.object(supervision, "signal_owned_group", side_effect=lambda group, sig:calls.append((group, sig))), \
                    patch.object(supervision, "_reap", return_value=receipt), \
                    patch.object(supervision, "live_owned_groups", return_value=set()):
                with self.assertRaisesRegex(RuntimeError, "evidence incomplete") as caught:
                    supervision._stop_and_reap(root, SimpleNamespace(pid=1001), "birth", binding)
            self.assertEqual(calls, [(p,signal.SIGKILL) for p in (3002,3001,2002,2001,1002,1001)])
            self.assertEqual(caught.exception.verified_groups, {1001,1002,2001,2002,3001,3002})
            self.assertIs(caught.exception.reaped_receipt, receipt)

    def test_failure_before_recursive_setup_retains_admitted_child_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            active = root / "delegated-groups/active"
            active.mkdir(parents=True)
            original = active.parent / "child-original.json"
            original.write_text("{}")
            os.link(original, active / original.name)
            row = dict(pid=2001, birth_identity="birth", binding_sha256="b" * 64, depth=1)
            order, known, issues = [], {1001}, []
            denied = RuntimeError("authored ack inspection failure")
            with patch.object(nested, "_read", return_value=(row, root / "child")), \
                    patch.object(nested, "freeze_launcher", return_value=2001), \
                    patch.object(nested, "_detached", side_effect=denied):
                groups = nested.freeze_delegated_groups(root, 1001, "b" * 64,
                    lambda *args:{1001}, issues=issues, known=known, kill_order=order)
            self.assertEqual(groups, {1001})
            self.assertEqual(known, {1001,2001})
            self.assertEqual(order, [2001,1001])
            self.assertEqual(issues, ["RuntimeError"])

    def test_direct_roster_error_preserves_ordinary_before_owner_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            active = root / "delegated-groups/active"
            active.mkdir(parents=True)
            (active / "child-over-limit.json").write_text("{}")
            order, known = [], {1001}
            with patch.object(nested, "MAX_ACTIVE", 0), self.assertRaisesRegex(ValueError, "roster drift"):
                nested.freeze_delegated_groups(root, 1001, "b" * 64,
                    lambda *args:{1001,2001}, known=known, kill_order=order)
            self.assertEqual(order, [2001,1001])
            self.assertEqual(known, {1001,2001})

    def test_current_parentage_required_before_stopping_any_launcher(self):
        row = dict(pid=1234, group=1234, birth="original birth", state="S", parent_pid=9999)
        with patch("pokezero.mcts_eval.search_over_raw_resources._process_table", return_value=([row], 0.)), \
                patch.object(nested.os, "kill") as kill:
            with self.assertRaisesRegex(ValueError, "unrelated"):
                nested.freeze_launcher(1234, "original birth", {1234}, expected_parent=1001)
        kill.assert_not_called()

    def test_no_redundant_group_stop_can_abort_nested_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            active = root / "delegated-groups/active"
            active.mkdir(parents=True)
            (active / "child-1.json").write_text("{}")
            order = []
            with patch.object(nested, "_freeze_child", return_value={3001}) as child, \
                    patch.object(nested, "signal_owned_group", side_effect=PermissionError("redundant group STOP forbidden")) as signal_group:
                result = nested.freeze_delegated_groups(root, 1001, "b" * 64,
                    lambda *args:{1001,2001}, kill_order=order)
            child.assert_called_once()
            signal_group.assert_not_called()
            self.assertEqual(result,{1001,2001,3001})

    def test_combined_signal_and_inspection_failure_preserves_first_error(self):
        receipt = (-signal.SIGKILL, object())
        denied = PermissionError("authored original signal denial")
        with patch.object(supervision.os, "waitid", return_value=None), \
                patch.object(supervision, "owned_process_identity", return_value="birth"), \
                patch.object(supervision, "freeze_launcher", return_value=1001), \
                patch.object(supervision, "freeze_delegated_groups", return_value={1001}), \
                patch.object(supervision, "signal_owned_group", side_effect=denied), \
                patch.object(supervision, "_reap", return_value=receipt), \
                patch.object(supervision, "live_owned_groups", side_effect=OSError("authored later query failure")):
            with self.assertRaises(PermissionError) as caught:
                supervision._stop_and_reap(Path("/authored"), SimpleNamespace(pid=1001), "birth", "b" * 64)
        self.assertIs(caught.exception,denied)
        self.assertIs(caught.exception.reaped_receipt,receipt)

    def test_late_ordinary_freeze_error_retains_verified_sibling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            owned = root / "owned-groups"
            active = owned / "active"
            active.mkdir(parents=True)
            for pid in (2001, 2002):
                path = owned / f"group-{pid}-original.json"
                path.write_text(json.dumps(dict(pid=pid, group=pid, birth_identity="original birth",
                    controller_pid=1001, registration_sha256="b" * 64, generation="original",
                    protocol="active-generations-v1")))
                os.link(path, active / path.name)
            issues = []
            with patch.object(supervision, "owned_process_identities", return_value={2001:"original birth",2002:"original birth"}), \
                    patch.object(supervision, "freeze_launcher", side_effect=[2001, PermissionError("authored inspection denial")]):
                result = supervision._groups(root, 1001, "b" * 64, _issues=issues, _live_authority=True)
            self.assertEqual(result, {1001, 2001})
            self.assertIn("PermissionError", issues)

    def test_one_signal_error_does_not_skip_sibling_owner_or_reap(self):
        receipt = (-signal.SIGKILL, object())
        calls = []
        def signal_group(group, sig):
            calls.append((group, sig))
            if group == 2001:
                raise PermissionError("authored group denial")
        def freeze(*args, **kwargs):
            kwargs["kill_order"].extend([2001, 2002, 1001])
            return {1001, 2001, 2002}
        with patch.object(supervision.os, "waitid", return_value=None), \
                patch.object(supervision, "owned_process_identity", return_value="birth"), \
                patch.object(supervision, "freeze_launcher", return_value=1001), \
                patch.object(supervision, "freeze_delegated_groups", side_effect=freeze), \
                patch.object(supervision, "signal_owned_group", side_effect=signal_group), \
                patch.object(supervision, "_reap", return_value=receipt) as reap, \
                patch.object(supervision, "live_owned_groups", return_value=set()):
            with self.assertRaises(PermissionError) as caught:
                supervision._stop_and_reap(Path("/authored"), SimpleNamespace(pid=1001), "birth", "b" * 64)
        self.assertEqual(calls, [(2001,signal.SIGKILL),(2002,signal.SIGKILL),(1001,signal.SIGKILL)])
        reap.assert_called_once_with(1001)
        self.assertIs(caught.exception.reaped_receipt, receipt)
        self.assertEqual(caught.exception.verified_groups, {1001,2001,2002})

    def test_post_reap_inspection_failure_retains_actual_wait_receipt(self):
        receipt = (-signal.SIGKILL, object())
        with patch.object(supervision.os, "waitid", return_value=None), \
                patch.object(supervision, "owned_process_identity", return_value="birth"), \
                patch.object(supervision, "freeze_launcher", return_value=1001), \
                patch.object(supervision, "freeze_delegated_groups", return_value={1001}), \
                patch.object(supervision, "signal_owned_group"), \
                patch.object(supervision, "_reap", return_value=receipt) as reap, \
                patch.object(supervision, "live_owned_groups", side_effect=OSError("authored query failure")):
            with self.assertRaises(OSError) as caught:
                supervision._stop_and_reap(Path("/authored"), SimpleNamespace(pid=1001), "birth", "b" * 64)
        reap.assert_called_once_with(1001)
        self.assertIs(caught.exception.reaped_receipt, receipt)
        self.assertEqual(caught.exception.verified_groups, {1001})

    def test_closed_directory_entry_is_durable_before_active_retirement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            owned = root / "owned-groups"
            owned.mkdir()
            ownership = dict(directory=str(owned), controller_pid=os.getpid(),
                registration_sha256="b" * 64, protocol="active-generations-v1")
            pool = parallel.ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime,
                workers=1, owned_process_receipts=ownership)
            real_sync, real_unlink = os.fsync, Path.unlink
            events = []
            def sync(fd):
                events.append(("sync", os.fstat(fd).st_ino))
                return real_sync(fd)
            def unlink(path, *args, **kwargs):
                if path.parent == owned / "active":
                    self.assertIn(("sync",owned.stat().st_ino), events)
                    self.assertTrue((owned / ("closed-" + path.name)).is_file())
                    events.append(("unlink", path.name))
                return real_unlink(path, *args, **kwargs)
            with patch.object(parallel.os, "fsync", side_effect=sync), \
                    patch.object(Path, "unlink", autospec=True, side_effect=unlink):
                pool.close()
            self.assertTrue(any(kind == "unlink" for kind, _ in events))

    def test_terminal_directory_sync_failure_keeps_active_index_and_refuses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            owned = root / "owned-groups"
            owned.mkdir()
            ownership = dict(directory=str(owned), controller_pid=os.getpid(),
                registration_sha256="b" * 64, protocol="active-generations-v1")
            pool = parallel.ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime,
                workers=1, owned_process_receipts=ownership)
            real_sync = os.fsync
            def sync(fd):
                if os.fstat(fd).st_ino == owned.stat().st_ino:
                    raise OSError("authored publication failure")
                return real_sync(fd)
            with patch.object(parallel.os, "fsync", side_effect=sync), self.assertRaises(parallel.ParallelRefusal):
                pool.close()
            self.assertEqual(pool.cleanup_evidence["status"], "REFUSED")
            self.assertEqual(len(list((owned / "active").iterdir())), 1)
            self.assertEqual(len(list(owned.glob("closed-*.json"))), 1)

    def test_already_reaped_owner_never_uses_saved_groups(self):
        receipt = (0, object())
        with patch.object(supervision, "signal_owned_group") as signal_group, \
                patch.object(supervision, "owned_process_identity") as identity:
            actual = supervision._stop_and_reap(Path("/authored"), SimpleNamespace(pid=1234),
                "historic birth", "b" * 64, already_reaped=receipt)
        self.assertEqual(actual, (receipt, set()))
        signal_group.assert_not_called()
        identity.assert_not_called()

    def test_terminal_unreaped_owner_never_signals_its_group(self):
        event = SimpleNamespace(si_code=os.CLD_EXITED)
        receipt = (1, object())
        with patch.object(supervision.os, "waitid", return_value=event), \
                patch.object(supervision, "_reap", return_value=receipt), \
                patch.object(supervision, "signal_owned_group") as signal_group:
            result = supervision._stop_and_reap(Path("/authored"), SimpleNamespace(pid=1234),
                "historic birth", "b" * 64)
        self.assertEqual(result, (receipt, set()))
        signal_group.assert_not_called()

    def test_zombie_launcher_is_not_stopped_live_authority(self):
        row = dict(pid=1234, group=1234, birth="original birth", state="Z")
        with patch("pokezero.mcts_eval.search_over_raw_resources._process_table", return_value=([row], 0.)), \
                patch.object(nested.os, "kill") as kill:
            self.assertIsNone(nested.freeze_launcher(1234, "original birth", {1234}))
        kill.assert_not_called()

    def test_detach_history_cannot_reauthorize_missing_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / "active/child-test.json"
            path.parent.mkdir()
            path.write_text("{}")
            original = root / "child-test.json"
            os.link(path, original)
            row = dict(pid=4321, birth_identity="old", binding_sha256="b" * 64, depth=1)
            issues, known = [], set()
            with patch.object(nested, "_read", return_value=(row, root / "child")), \
                    patch.object(nested, "freeze_launcher", return_value=None), \
                    patch.object(nested, "_detached", return_value=True), \
                    patch.object(nested, "signal_owned_group") as group_signal:
                result = nested._freeze_child(path, root, 1234, "b" * 64,
                    lambda *args: set(), 0, set(), issues, known)
            self.assertEqual(result, set())
            self.assertEqual(known, set())
            self.assertIn("delegated_live_authority_lost", issues)
            group_signal.assert_not_called()

    def test_actual_clean_pool_generations_require_cleanup_not_exit_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            owned = root / "owned-groups"
            owned.mkdir()
            binding = "b" * 64
            ownership = dict(directory=str(owned), controller_pid=os.getpid(),
                registration_sha256=binding, protocol="active-generations-v1")
            with parallel.ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime,
                    workers=2, owned_process_receipts=ownership) as pool:
                pool.search(Request(), ROOT, battle_id="authored", seed=7, trajectories_per_worker=1)
            supervision._verify_closed_generations(root, os.getpid(), binding)
            launches = list(owned.glob("group-*.json"))
            self.assertEqual(len(launches), 2)
            self.assertEqual(list((owned / "active").iterdir()), [])
            closed = launches[0].with_name("closed-" + launches[0].name)
            raw = closed.read_bytes()
            for key, value in (("clean_acknowledgment", False), ("forced_cleanup", True),
                    ("runtime_close_failed", True), ("cleanup_exception_count", True),
                    ("cleanup_protocol", "old-exit-only")):
                row = json.loads(raw)
                self.assertEqual(row["worker_exit_code"], 0)
                row[key] = value
                closed.write_text(json.dumps(row))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    supervision._verify_closed_generations(root, os.getpid(), binding)
                closed.write_bytes(raw)

    def test_actual_partial_diagnostic_frame_refuses_and_reaps_owned_worker(self):
        began = time.perf_counter()
        with patch.object(parallel, "_worker", partial_diagnostic_worker):
            pool = parallel.ParallelTrajectorySearch(ReferenceConfig(.5, 1), DiagnosticRuntime, workers=1)
        selected = pool.search(Request(), ROOT, battle_id="authored", seed=7, trajectories_per_worker=1)
        pool.transport_timeout = .1
        with self.assertRaises(parallel.ParallelRefusal), pool:
            pool.finish_diagnostics(selected)
        self.assertLess(time.perf_counter() - began, 10)
        self.assertEqual(pool.cleanup_evidence["status"], "REFUSED")
        self.assertTrue(pool.cleanup_evidence["workers"][0]["group_signalled_before_reap"])
        self.assertEqual(pool.cleanup_evidence["workers"][0]["actual_exitcode"], -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
