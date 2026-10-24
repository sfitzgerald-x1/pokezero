"""Owned tiny child processes only; no engine/model dependencies or games."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

DRIVER = Path(__file__).resolve().parents[1] / "scripts/bench_shallow_exact_native.py"
spec = importlib.util.spec_from_file_location("shallow_exact_bench_test", DRIVER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_success_output_and_real_exit_are_retained(self):
        result = runner.run_owned([sys.executable, "-c", "print('done')"], env=os.environ, hard_seconds=5)
        self.assertEqual(result["actual_exit_code"], 0)
        self.assertEqual(result["stdout"], "done\n")
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["owned_group_drain_attempted"])
        self.assertTrue(result["owned_child_reaped"])

    def test_hard_timeout_reaps_owned_child(self):
        result = runner.run_owned([sys.executable, "-c", "import time;time.sleep(30)"],
                                  env=os.environ, hard_seconds=0.05)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["actual_exit_code"], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["owned_child_pid"], 0)

    def test_signal_during_popen_is_deferred_until_handle_is_owned(self):
        original = subprocess.Popen
        def signal_while_returning(*args, **kwargs):
            child = original(*args, **kwargs)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child
        with patch.object(runner.subprocess, "Popen", side_effect=signal_while_returning):
            result = runner.run_owned([sys.executable, "-c", "import time;time.sleep(30)"],
                                      env=os.environ, hard_seconds=5)
        self.assertEqual(result["owner_signal"], signal.SIGTERM)
        self.assertTrue(result["owned_child_reaped"])
        self.assertEqual(result["actual_exit_code"], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["owned_child_pid"], 0)

    def test_post_kill_reap_timeout_is_unknown_exit_not_success(self):
        child = SimpleNamespace(pid=123456789, returncode=None,
            stdout=io.StringIO(), stderr=io.StringIO(),
            communicate=Mock(side_effect=[subprocess.TimeoutExpired("test", 1),
                                          subprocess.TimeoutExpired("test", 1, output=b"partial")]))
        with patch.object(runner.subprocess, "Popen", return_value=child), patch.object(runner.os, "killpg"):
            result = runner.run_owned(["unused"], env=os.environ, hard_seconds=1, reap_seconds=0.1)
        self.assertIsNone(result["actual_exit_code"])
        self.assertFalse(result["owned_child_reaped"])
        self.assertEqual(result["stdout"], "partial")
        self.assertEqual(child.communicate.call_args.kwargs, {"timeout": 0.1})

    def test_timeout_stays_failure_even_if_child_naturally_returns_zero_during_reap(self):
        result = {"actual_exit_code": 0, "timed_out": True,
                  "owned_child_reaped": True, "owner_signal": None}
        self.assertEqual(runner.owned_exit_status(result), 1)

    def test_owner_term_drains_worker_before_returning(self):
        code = (
            "import importlib.util,os,signal,sys,threading,json;"
            f"s=importlib.util.spec_from_file_location('b',{str(DRIVER)!r});"
            "b=importlib.util.module_from_spec(s);s.loader.exec_module(b);"
            "threading.Timer(.2,lambda:os.kill(os.getpid(),signal.SIGTERM)).start();"
            "r=b.run_owned([sys.executable,'-c','import time;time.sleep(30)'],env=os.environ,hard_seconds=5);"
            "print(json.dumps(r))"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["owner_signal"], signal.SIGTERM)
        self.assertEqual(result["actual_exit_code"], -signal.SIGKILL)
        self.assertTrue(result["owned_group_drain_attempted"])
        with self.assertRaises(ProcessLookupError):
            os.kill(result["owned_child_pid"], 0)


if __name__ == "__main__":
    unittest.main()
