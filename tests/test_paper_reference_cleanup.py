"""Actual owner/worker teardown fixtures; no game or model execution."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pokezero.mcts_eval.paper_reference_parallel as parallel
from pokezero.mcts_eval.paper_reference import ReferenceConfig
from pokezero.mcts_eval.paper_reference_parallel import ParallelRefusal, ParallelTrajectorySearch
from test_paper_reference_parallel import ToyRuntime, Request
from test_paper_reference import ROOT


class CleanChildRuntime(ToyRuntime):
    def __init__(self, index):
        super().__init__(index)
        self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def close(self):
        self.child.terminate()
        self.child.wait(timeout=2.)


class CloseErrorRuntime(ToyRuntime):
    def close(self):
        raise RuntimeError("deliberate teardown error")


class CloseHangRuntime(ToyRuntime):
    def close(self):
        time.sleep(60)


@dataclass
class ConstructorFixture:
    output: str
    hang: bool = False

    def __call__(self, index):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        os.kill(child.pid, signal.SIGSTOP)
        Path(self.output).write_text(json.dumps({"owner": os.getpid(), "child": child.pid}))
        if self.hang:
            time.sleep(60)
        raise RuntimeError("deliberate constructor error")


def assert_dead(test, pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    status = result.stdout.strip()
    test.assertTrue(not status or status.startswith("Z"), f"fixture still live: {pid} {status}")


def partial_frame_worker(connection, index, runtime_factory, config):
    os.setsid()
    connection.send(("isolation", index, os.getpid(), os.getpgrp()))
    connection.send(("boot", index, os.getpid(), 0., os.getpgrp()))
    os.write(connection.fileno(), struct.pack("!i", 1000000) + b"partial")
    os.kill(os.getpid(), signal.SIGSTOP)
    time.sleep(60)


class ReferenceCleanupTests(unittest.TestCase):
    def test_empty_or_wrong_cleanup_envelope_still_cleans_all_workers(self):
        for malformed in ((), [], None):
            with self.subTest(message=malformed):
                pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2)
                receive = parallel._bounded_message
                def mutated(connection, deadline):
                    message = receive(connection, deadline)
                    return malformed if connection is pool._connections[0] and message[0] == "cleanup_ready" else message
                with patch.object(parallel, "_bounded_message", side_effect=mutated):
                    with self.assertRaises(ParallelRefusal):
                        pool.close()
                self.assertEqual(pool.cleanup_evidence["status"], "REFUSED")
                self.assertEqual(len(pool.cleanup_evidence["workers"]), 2)
                self.assertTrue(all(row["terminal_reaped"] for row in pool.cleanup_evidence["workers"]))
                self.assertTrue(all(c.closed for c in pool._connections))
                evidence = pool.cleanup_evidence
                pool.close()
                self.assertIs(pool.cleanup_evidence, evidence)

    def test_partial_frame_stopped_writer_cannot_block_bounded_cleanup(self):
        with patch.object(parallel, "_worker", partial_frame_worker):
            pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1)
        started = time.perf_counter()
        with self.assertRaises(ParallelRefusal):
            pool.close()
        self.assertLess(time.perf_counter() - started, 10.)
        row = pool.cleanup_evidence["workers"][0]
        self.assertEqual(row["actual_exitcode"], -signal.SIGKILL)
        self.assertTrue(row["group_signalled_before_reap"])
        self.assertTrue(row["terminal_reaped"])
        self.assertIn("frame completion timed out", str(pool.cleanup_evidence["errors"]))

    def test_partial_frame_failure_diagnostics_preserve_owner_for_cleanup(self):
        with patch.object(parallel, "_worker", partial_frame_worker):
            pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1)
        pool.transport_timeout = .05
        with self.assertRaises(ParallelRefusal):
            list(pool._receive({0}))
        self.assertIsNone(pool._processes[0]._popen.returncode)
        with self.assertRaises(ParallelRefusal):
            pool.close()
        self.assertTrue(pool.cleanup_evidence["workers"][0]["terminal_reaped"])

    def test_one_group_signal_error_does_not_strand_later_worker(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), CloseErrorRuntime, workers=2)
        signal_group = os.killpg
        calls = []
        def one_error(pid, sig):
            calls.append(pid)
            if len(calls) == 1:
                raise PermissionError("injected fixture permission failure")
            return signal_group(pid, sig)
        with patch.object(parallel.os, "killpg", side_effect=one_error):
            with self.assertRaises(ParallelRefusal):
                pool.close()
        records = pool.cleanup_evidence["workers"]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["terminal_reaped"] for row in records))
        self.assertTrue(all(c.closed for c in pool._connections))
        self.assertIn("PermissionError", str(records[0]["cleanup_exceptions"]))
        self.assertEqual(set(calls), set(pool.worker_pids))

    def test_join_exception_finalizes_error_and_attempts_remaining_worker(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2)
        process = pool._processes[0]
        join = process.join
        calls = []
        def one_error(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("injected join failure")
            return join(*args, **kwargs)
        with patch.object(process, "join", side_effect=one_error):
            with self.assertRaises(ParallelRefusal):
                pool.close()
        self.assertTrue(all(row["terminal_reaped"] for row in pool.cleanup_evidence["workers"]))
        self.assertTrue(all(c.closed for c in pool._connections))
        self.assertIn("injected join failure", str(pool.cleanup_evidence))

    def test_starting_next_worker_revokes_reaped_prior_group_authority(self):
        process_type = parallel.mp.get_context("spawn").Process
        start = process_type.start
        created = []
        def start_and_expose_cleanup(process):
            if created:
                os.kill(created[0].pid, signal.SIGKILL)
                until = time.perf_counter() + 2.
                while parallel._nonreaping_state(created[0])["state"] == "LIVE" and time.perf_counter() < until:
                    time.sleep(.01)
            start(process)  # CPython's real _cleanup reaps the previous child.
            created.append(process)
        with patch.object(process_type, "start", start_and_expose_cleanup):
            with patch.object(parallel.os, "killpg", wraps=os.killpg) as group_signal:
                with self.assertRaisesRegex(ParallelRefusal, "ownership lost") as caught:
                    ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2)
        old = caught.exception.evidence["cleanup"]["workers"][0]
        self.assertEqual(old["before_reap"]["state"], "ALREADY_REAPED")
        self.assertFalse(old["group_signalled_before_reap"])
        self.assertTrue(all(call.args[0] != created[0].pid for call in group_signal.call_args_list))
        self.assertTrue(all(not p.is_alive() for p in created))

    def test_clean_actual_subprocess_reaped_and_release_no_group_signal(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), CleanChildRuntime, workers=2)
        with patch.object(parallel.os, "killpg", wraps=os.killpg) as group_signal:
            pool.close()
            group_signal.assert_not_called()
        self.assertEqual(pool.cleanup_evidence["status"], "CLEAN_OBSERVED_SCOPE")
        self.assertTrue(all(row["actual_exitcode"] == 0 and row["terminal_reaped"]
            and row["live_children_reported"] == [] for row in pool.cleanup_evidence["workers"]))
        self.assertFalse(pool.cleanup_evidence["complete_descendant_accounting"])
        self.assertFalse(pool.cleanup_evidence["cpu_accounting_complete"])
        old = pool.cleanup_evidence
        pool.close()
        self.assertIs(pool.cleanup_evidence, old)

    def test_close_error_propagates_without_successful_exit_claim(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), CloseErrorRuntime, workers=1)
        with self.assertRaisesRegex(ParallelRefusal, "cleanup refused") as caught:
            pool.close()
        row = caught.exception.evidence["cleanup"]["workers"][0]
        self.assertIn("deliberate teardown error", row["runtime_close_error"])
        self.assertTrue(row["forced"])
        self.assertEqual(row["actual_exitcode"], -signal.SIGKILL)
        self.assertTrue(row["group_signalled_before_reap"])
        self.assertTrue(row["terminal_reaped"])

    def test_existing_search_error_is_primary_cleanup_failure_retained(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), CloseErrorRuntime, workers=1)
        with self.assertRaisesRegex(ParallelRefusal, "nonbankable") as caught, pool:
            pool.search(Request(failure=True), ROOT, battle_id="b", seed=1, trajectories_per_worker=1)
        self.assertEqual(caught.exception.evidence["cleanup"]["status"], "REFUSED")
        self.assertIn("deliberate public-root refusal", str(caught.exception.evidence["errors"]))

    def test_constructor_spawns_then_raises_owned_group_closed_before_reaping(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "actual.json"
            with self.assertRaisesRegex(ParallelRefusal, "startup") as caught:
                ParallelTrajectorySearch(ReferenceConfig(.5, 1), ConstructorFixture(str(out)), workers=1)
            handles = json.loads(out.read_text())
            row = caught.exception.evidence["cleanup"]["workers"][0]
            self.assertEqual(row["pid"], handles["owner"])
            self.assertEqual(row["live_children_reported"], [handles["child"]])
            self.assertTrue(row["group_signalled_before_reap"])
            self.assertEqual(row["actual_exitcode"], -signal.SIGKILL)
            assert_dead(self, handles["child"])

    def test_preboot_constructor_hang_uses_original_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "actual.json"
            started = time.perf_counter()
            with self.assertRaisesRegex(ParallelRefusal, "stalled") as caught:
                ParallelTrajectorySearch(ReferenceConfig(.5, 1), ConstructorFixture(str(out), True),
                    workers=1, transport_timeout=.5)
            self.assertLess(time.perf_counter() - started, 10.)
            handles = json.loads(out.read_text())
            row = caught.exception.evidence["cleanup"]["workers"][0]
            self.assertIsNone(row["live_children_reported"])
            self.assertTrue(row["group_signalled_before_reap"])
            assert_dead(self, handles["child"])

    def test_shared_grace_not_workers_times_grace_for_hung_close(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), CloseHangRuntime, workers=2)
        started = time.perf_counter()
        with self.assertRaises(ParallelRefusal):
            pool.close()
        self.assertLess(time.perf_counter() - started, 10.)
        self.assertEqual(len(pool.cleanup_evidence["workers"]), 2)
        self.assertTrue(all(row["terminal_reaped"] and row["actual_exitcode"] == -signal.SIGKILL
            for row in pool.cleanup_evidence["workers"]))

    def test_already_reaped_original_leader_has_no_group_authority(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1)
        process = pool._processes[0]
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=2.)  # Simulate multiprocessing's startup cleanup.
        with patch.object(parallel.os, "killpg", wraps=os.killpg) as group_signal:
            with self.assertRaises(ParallelRefusal):
                pool.close()
            group_signal.assert_not_called()
        row = pool.cleanup_evidence["workers"][0]
        self.assertEqual(row["before_reap"]["state"], "ALREADY_REAPED")
        self.assertFalse(row["group_signalled_before_reap"])

    def test_stopped_worker_is_not_terminal_or_reaped_by_diagnostics(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1)
        pool.transport_timeout = .05  # Startup is not the stopped-transport test.
        process = pool._processes[0]
        os.kill(process.pid, signal.SIGSTOP)
        try:
            with self.assertRaisesRegex(ParallelRefusal, "stalled"):
                list(pool._receive({0}))
            self.assertEqual(parallel._nonreaping_state(process)["state"], "LIVE")
            self.assertIsNone(process._popen.returncode)
        finally:
            os.kill(process.pid, signal.SIGCONT)
            pool.close()

    def test_mismatched_isolation_does_not_register_unowned_group(self):
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1)
        try:
            pool._owned_groups.clear()
            with self.assertRaises(ParallelRefusal):
                pool._record_isolation(0, ("isolation", 0, os.getpid(), os.getpid()))
            self.assertEqual(pool._owned_groups, {})
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
