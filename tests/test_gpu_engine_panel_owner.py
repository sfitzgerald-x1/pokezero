"""Terminal artifact guards; no fresh root or scientific retry."""
import gzip
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

from gpu_engine import panel_owner as owner
from gpu_engine.panel_owner import own, reconstruct, validate_slot


ENTRY = dict(slots=2, mass_tolerance=1e-5, probability_tolerance=1e-5)


class OwnerTests(unittest.TestCase):
    def inputs(self, root):
        payload = b'{"slot":0}\n{"slot":1}\n'
        with gzip.open(root/"inputs.jsonl.gz", "wb") as stream:
            stream.write(payload)
        frozen = dict(slots=2, input_jsonl_sha256=hashlib.sha256(payload).hexdigest(),
            compressed_sha256=hashlib.sha256((root/"inputs.jsonl.gz").read_bytes()).hexdigest())
        (root/"inputs-frozen.json").write_text(json.dumps(frozen))
        return frozen

    def test_missing_inputs_remain_all_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reconstruct(Path(directory), ENTRY)
        self.assertEqual(result["counts"], {"UNKNOWN":2})
        self.assertFalse(result["input_freeze_valid"])

    def test_valid_unknown_is_completed_but_not_a_passing_law(self):
        record = dict(slot=0,input_jsonl_sha256="bound",status="UNKNOWN",category="CASE_CAP",
                      identified_mass="0",unknown_mass="1")
        self.assertEqual(validate_slot(record, ENTRY, "bound", 0), "UNKNOWN")
        record["identified_mass"] = "1"
        with self.assertRaises(ValueError):
            validate_slot(record, ENTRY, "bound", 0)

    def test_complete_file_counts_even_if_progress_receipt_was_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = self.inputs(root)
            record = dict(slot=0,input_jsonl_sha256=frozen["input_jsonl_sha256"],status="AGREE",
                identified_mass="1",unknown_mass="0",outcomes=[dict(key=[["hp",1]],
                    jax_probability="1",native_probability=1.)])
            with gzip.open(root/"slot-0000.json.gz", "wb") as stream:
                stream.write(json.dumps(record).encode())
            (root/"progress.json").write_text('{"truncated":')
            (root/"current-slot.json").write_text(json.dumps(dict(slot=1,status="STARTED",inputs=frozen)))
            result = reconstruct(root, ENTRY)
        self.assertEqual(result["counts"], {"AGREE":1,"UNKNOWN":1})
        self.assertEqual(result["valid_completed_slots"], 1)
        self.assertEqual(result["attempted_slots"], 2)
        self.assertEqual(result["incomplete_slots"]["1"], "INTERRUPTED_WITHOUT_COMPLETE_LAW")

    def test_corrupt_slot_cannot_be_completed_by_optimistic_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.inputs(root)
            (root/"slot-0000.json.gz").write_bytes(b'not gzip')
            (root/"progress.json").write_text('{"completed_slots":2}')
            result = reconstruct(root, ENTRY)
        self.assertEqual(result["valid_completed_slots"], 0)
        self.assertEqual(result["counts"], {"UNKNOWN":2})
        self.assertEqual(result["unattempted_slots"], 1)

    def test_stored_claim_is_checked_against_full_unnormalized_law(self):
        record = dict(slot=0,input_jsonl_sha256="bound",status="AGREE",identified_mass="1",unknown_mass="0",
            outcomes=[dict(key=[["hp",1]],jax_probability="1",native_probability=.9)])
        with self.assertRaises(ValueError):
            validate_slot(record, ENTRY, "bound", 0)

    def test_owned_deadline_interrupts_in_progress_child_and_records_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/"attempt"
            code = own([sys.executable,"-c","import time;time.sleep(30)"],
                out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=1)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(code,124)
        self.assertTrue(terminal["timed_out"])
        self.assertTrue(terminal["worker_reaped"])
        self.assertNotEqual(terminal["actual_worker_exit"],0)
        self.assertEqual(terminal["counts"],{"UNKNOWN":2})

    def test_exit_zero_without_complete_artifacts_is_not_experiment_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/"attempt"
            code = own([sys.executable,"-c","pass"], out=out,panel=out/"panel",
                entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(terminal["actual_worker_exit"],0)
        self.assertEqual(code,1)
        self.assertEqual(terminal["status"],"FAILED_OR_INTERRUPTED")

    def test_launch_publication_failure_still_drains_actual_child(self):
        actual_publish = owner.atomic_json
        def publish(path, value):
            if path.name == "launch.json":
                raise OSError("authored publication failure")
            return actual_publish(path,value)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(owner,"atomic_json",side_effect=publish):
            out = Path(directory)/"attempt"
            code = own([sys.executable,"-c","import time;time.sleep(30)"],
                out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(code,1)
        self.assertTrue(terminal["worker_reaped"])
        self.assertTrue(terminal["owned_group_absent"])
        self.assertIn("OSError",terminal["owner_error"])

    def test_owner_signal_is_deferred_until_actual_worker_cleanup(self):
        code = "import os,signal,time;os.kill(os.getppid(),signal.SIGTERM);time.sleep(30)"
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/"attempt"
            result = own([sys.executable,"-c",code],out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(result,1)
        self.assertTrue(terminal["owner_signals"])
        self.assertTrue(terminal["worker_reaped"])
        self.assertTrue(terminal["owned_group_absent"])

    def test_unexpected_wait_exception_still_drains(self):
        actual_spawn = owner.subprocess.Popen
        def spawn(*args,**kwargs):
            worker = actual_spawn(*args,**kwargs)
            actual_wait = worker.wait
            calls = 0
            def wait(*a,**kw):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("authored wait failure")
                return actual_wait(*a,**kw)
            worker.wait = wait
            return worker
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(owner.subprocess,"Popen",side_effect=spawn):
            out = Path(directory)/"attempt"
            code = own([sys.executable,"-c","import time;time.sleep(30)"],
                out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(code,1)
        self.assertTrue(terminal["worker_reaped"])
        self.assertTrue(terminal["owned_group_absent"])
        self.assertIn("RuntimeError",terminal["owner_error"])

    def test_unverified_descendants_exclude_success_even_with_complete_files(self):
        actual_drain = owner.drain_owned
        def unverified(worker):
            result = actual_drain(worker)
            result["owned_group_absent"] = False
            return result
        tally = dict(valid_completed_slots=2, counts={"AGREE":2})
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(owner,"drain_owned",side_effect=unverified), \
                mock.patch.object(owner,"reconstruct",return_value=tally):
            out = Path(directory)/"attempt"
            code = own([sys.executable,"-c","pass"],out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(code,1)
        self.assertEqual(terminal["actual_worker_exit"],0)
        self.assertEqual(terminal["status"],"FAILED_OR_INTERRUPTED")

    def test_drain_wait_error_still_attempts_final_kill_and_reap(self):
        worker = owner.subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True)
        actual_wait = worker.wait
        calls = 0
        def wait(*args,**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("authored drain failure")
            return actual_wait(*args,**kwargs)
        with mock.patch.object(worker,"wait",side_effect=wait):
            result = owner.drain_owned(worker,grace_seconds=1,kill_seconds=1)
        self.assertGreaterEqual(calls,2)
        self.assertTrue(result["reaped"])
        self.assertTrue(result["owned_group_absent"])
        self.assertNotEqual(result["actual_exit"],0)
        self.assertTrue(result["cleanup_errors"])

    def test_late_observed_exit_zero_remains_capped(self):
        actual_spawn, actual_clock = owner.subprocess.Popen, owner.time.monotonic
        finished = False
        def spawn(*args,**kwargs):
            worker = actual_spawn(*args,**kwargs)
            actual_wait = worker.wait
            def wait(*a,**kw):
                nonlocal finished
                result = actual_wait(*a,**kw)
                finished = True
                return result
            worker.wait = wait
            return worker
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(owner.subprocess,"Popen",side_effect=spawn), \
                mock.patch.object(owner.time,"monotonic",side_effect=lambda:actual_clock()+(10 if finished else 0)):
            out = Path(directory)/"attempt"
            result = own([sys.executable,"-c","import time;time.sleep(.05)"],
                out=out,panel=out/"panel",entry=ENTRY,calendar_seconds=5)
            terminal = json.loads((out/"owner-terminal.json").read_bytes())
        self.assertEqual(terminal["actual_worker_exit"],0)
        self.assertTrue(terminal["timed_out"])
        self.assertEqual(result,124)


if __name__ == "__main__":
    unittest.main()
