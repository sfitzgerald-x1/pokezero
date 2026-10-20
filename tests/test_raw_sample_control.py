"""Authored control fixtures; no checkpoint, source game or timed search."""
import importlib.util
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

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("control", SCRIPTS / "run_raw_sample_control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class ControlTests(unittest.TestCase):
    def entry(self, directory):
        entry = dict(seeds=list(range(32)), calendar_seconds=86400,
                     no_completion_seconds=1800, first_completion_seconds=900,
                     max_decision_rounds=250, cpu=0,
                     host=dict(hostname=control.socket.gethostname(),
                               boot_id="fixture-boot"),
                     checkpoint="/weights", checkpoint_sha256="fixture-sha",
                     node_binary="/pinned/node", python="/python", runtime="/runtime",
                     run_directory=str(Path(directory) / "run"),
                     environment=dict(PATH="/pinned:/usr/bin", **{key: "1" for key in (
                         "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "POKEZERO_TORCH_NUM_THREADS", "POKEZERO_TORCH_NUM_INTEROP_THREADS")}),
                     input_hashes={str(Path(control.__file__).resolve()): "runner",
                                   str(Path(control.panel.__file__).resolve()): "helper",
                                   "/weights": "fixture-sha", "/pinned/node": "node",
                                   "/python": "python"})
        entry["slots"] = control.slots(entry)
        path, review = Path(directory) / "entry.json", Path(directory) / "review.json"
        control.panel.write(path, entry)
        control.panel.write(review, dict(status="CLEAR", entry_sha256=control.panel.digest(path)))
        return entry, path, review

    def test_exact_64_slot_order_and_fresh_distinct_seed_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, _, _ = self.entry(directory)
            self.assertEqual(len(control.slots(entry)), 64)
            self.assertEqual(control.slots(entry)[:2], [dict(seed=0, raw_seat="p1"), dict(seed=0, raw_seat="p2")])
            for seeds in ([1] * 32, list(range(31)), [True] + list(range(1, 32)), list(range(31)) + [2**32]):
                with self.assertRaises(ValueError):
                    control.slots(dict(seeds=seeds))

    def test_review_must_bind_exact_entry_before_any_runtime_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, review = self.entry(directory)
            control.panel.write(review, dict(status="CLEAR", entry_sha256="other"))
            with patch.object(control.panel, "verify") as verify:
                with self.assertRaisesRegex(ValueError, "exact entry"):
                    control.check(path, review)
                verify.assert_not_called()

    def test_torch_thread_initialization_has_one_owner_and_checks_readback(self):
        torch = Mock()
        torch.get_num_threads.return_value = 1
        torch.get_num_interop_threads.return_value = 1
        require = Mock(return_value=torch)
        with patch.dict(sys.modules, {"pokezero.neural_policy": SimpleNamespace(require_torch=require)}):
            self.assertIs(control.configure_torch(), torch)
            require.assert_called_once_with()
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()
            torch.get_num_interop_threads.return_value = 2
            with self.assertRaisesRegex(ValueError, "resolved torch"):
                control.configure_torch()

    def test_real_terminal_scores_and_caps_unknown(self):
        slot = dict(seed=1, raw_seat="p2")
        for winner, capped, score in (("p2", False, 1), ("p1", False, 0),
                                      (None, False, .5), (None, True, None)):
            row = control.terminal_row(slot, SimpleNamespace(winner=winner, capped=capped))
            self.assertEqual(row["raw_score"], score)
        with self.assertRaises(ValueError):
            control.terminal_row(slot, SimpleNamespace(winner="other", capped=False))

    def test_progress_rejects_redraw_reordering_and_scored_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, _, _ = self.entry(directory)
            output = Path(entry["run_directory"])
            output.mkdir()
            good = dict(seed=0, raw_seat="p1", status="COMPLETE", winner="p1", raw_score=1)
            control.panel.write(output / "progress.json", dict(games=[good]))
            self.assertEqual(control.prefix(output, entry), [good])
            for bad in (dict(good, seed=2), dict(good, raw_seat="p2"),
                        dict(good, status="CAPPED", raw_score=.5), dict(good, status="REFUSED")):
                control.panel.write(output / "progress.json", dict(games=[bad]))
                with self.assertRaises(ValueError):
                    control.prefix(output, entry)

    def test_no_resume_or_overwrite_of_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, path, review = self.entry(directory)
            Path(entry["run_directory"]).mkdir()
            with patch.object(control, "check", return_value=entry), patch.object(control.subprocess, "Popen") as spawn:
                with self.assertRaises(FileExistsError):
                    control.run(path, review)
                spawn.assert_not_called()

    def run_fixture(self, *, code, count, times=(100, 101, 102), cleanup=True):
        with tempfile.TemporaryDirectory() as directory:
            entry, path, review = self.entry(directory)
            process = Mock(pid=123456)
            process.poll.return_value = code
            with patch.object(control, "check", return_value=entry), \
                    patch.object(control.subprocess, "Popen", return_value=process) as spawn, \
                    patch.object(control.panel, "stop_owned", return_value=(code, cleanup)), \
                    patch.object(control, "prefix", return_value=[{}] * count), \
                    patch.object(control, "process_identity", return_value=dict(start_ticks="fixture")), \
                    patch.object(control.time, "time", side_effect=times):
                returned = control.run(path, review)
            self.assertEqual(spawn.call_count, 1)
            return returned, json.loads((Path(entry["run_directory"]) / "terminal.json").read_text())

    def test_only_complete_exit_zero_with_cleanup_is_collection_finished(self):
        returned, terminal = self.run_fixture(code=0, count=64)
        self.assertEqual(returned, 0)
        self.assertEqual(terminal["status"], "COLLECTION_FINISHED")
        self.assertEqual(terminal["scientific_qualification"], "UNREVIEWED")
        for code, count, cleanup in ((1, 12, True), (0, 63, True), (0, 64, False)):
            returned, terminal = self.run_fixture(code=code, count=count, cleanup=cleanup)
            self.assertEqual(returned, 1)
            self.assertEqual(terminal["status"], "PROCESS_FAILURE")

    def test_already_exited_worker_cannot_bypass_absolute_cap(self):
        returned, terminal = self.run_fixture(code=0, count=64, times=(100, 86500, 86501))
        self.assertEqual(returned, 1)
        self.assertEqual(terminal["status"], "CALENDAR_CAP")

    def test_policy_modes_fresh_histories_and_hidden_opponent_context(self):
        created, configs = [], []
        def policy(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(history={})
        env = Mock()
        env.__enter__ = Mock(return_value=env)
        env.__exit__ = Mock(return_value=False)
        env.protocol_lines = ["|win|p1"]
        result = SimpleNamespace(trajectory="trajectory", terminal=SimpleNamespace(winner="p1", capped=False), decision_round_count=3)
        driver = Mock()
        driver.return_value.run.return_value = result
        modules = {
            "pokezero.local_showdown": SimpleNamespace(LocalShowdownEnv=Mock(return_value=env)),
            "pokezero.neural_policy": SimpleNamespace(TransformerSoftmaxPolicy=policy),
            "pokezero.rollout": SimpleNamespace(RolloutDriver=driver, RolloutConfig=lambda **kw: configs.append(kw)),
            "pokezero.trajectory": SimpleNamespace(trajectory_to_dict=lambda obj: {"fixture": obj}),
        }
        with patch.dict(sys.modules, modules):
            for _ in range(2):
                control.play_game(dict(checkpoint="weights", checkpoint_sha256="sha"),
                                  dict(seed=1, raw_seat="p1"), "model", "result", "config")
        self.assertEqual([p["deterministic"] for p in created], [True, False, True, False])
        self.assertTrue(all(p["sampling_temperature"] == 1 and p["exploration_epsilon"] == 0
                            and not p["family_gated_selection"] and p["history_mask_k"] is None for p in created))
        self.assertTrue(all(c["hide_opponent_legal_action_masks"] for c in configs))
        self.assertTrue(all(c["max_decision_rounds"] == 250 for c in configs))
        self.assertIsNot(driver.call_args_list[0].kwargs["policies"]["p1"],
                         driver.call_args_list[1].kwargs["policies"]["p1"])

    def test_failed_bridge_close_does_not_admit_a_terminal(self):
        # Reuse the policy fixture while making context exit fail: no row returns.
        env = Mock()
        env.__enter__ = Mock(return_value=env)
        env.__exit__ = Mock(side_effect=RuntimeError("close failure"))
        env.protocol_lines = []
        result = SimpleNamespace(trajectory=None, terminal=SimpleNamespace(winner="p1", capped=False), decision_round_count=1)
        modules = {
            "pokezero.local_showdown": SimpleNamespace(LocalShowdownEnv=Mock(return_value=env)),
            "pokezero.neural_policy": SimpleNamespace(TransformerSoftmaxPolicy=Mock()),
            "pokezero.rollout": SimpleNamespace(RolloutDriver=Mock(return_value=SimpleNamespace(run=lambda **kw: result)), RolloutConfig=Mock()),
            "pokezero.trajectory": SimpleNamespace(trajectory_to_dict=Mock(return_value={})),
        }
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "close failure"):
            control.play_game(dict(checkpoint="weights", checkpoint_sha256="sha"),
                              dict(seed=1, raw_seat="p1"), None, None, None)

    def test_guard_rejects_parent_loss_pid_reuse_zombie_and_expired_clock(self):
        launch = dict(supervisor_pid=123, supervisor_start_ticks="10", deadline_unix=100)
        for identity, now, reason in ((None, 1, "SUPERVISOR_LOST"),
                                     (dict(start_ticks="11", state="S"), 1, "SUPERVISOR_LOST"),
                                     (dict(start_ticks="10", state="Z"), 1, "SUPERVISOR_LOST"),
                                     (dict(start_ticks="10", state="S"), 100, "CALENDAR_CAP"),
                                     (dict(start_ticks="10", state="S"), 99, None)):
            with patch.object(control, "process_identity", return_value=identity):
                self.assertEqual(control.parent_guard_reason(launch, now), reason)

    @unittest.skipUnless(sys.platform == "linux", "actual /proc lifetime fixture runs on Linux")
    def test_actual_supervisor_term_and_kill_stop_fake_worker_and_descendant(self):
        # No game/model imports: an ordinary sleeping child stands in for Node.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as directory:
                entry, path, review = self.entry(directory)
                entry.update(runtime=directory, python=sys.executable)
                entry["environment"]["PYTHONPATH"] = str(SCRIPTS)
                control.panel.write(path, entry)
                fake = """
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
import run_raw_sample_control as control
output = Path(sys.argv[1])
while True:
    launch = json.loads((output / 'launch.json').read_text())
    if launch.get('worker_pid') == os.getpid(): break
    time.sleep(.01)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'])
(output / 'fake-child.json').write_text(json.dumps(dict(pid=child.pid)))
def term(*_):
    child.wait(timeout=5)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, term)
threading.Thread(target=control.guard_parent, args=(launch, output), daemon=True).start()
while True: time.sleep(.1)
"""
                parent = f"""
import json, sys
import run_raw_sample_control as control
entry = json.loads(open({str(path)!r}).read())
control.check = lambda *_: entry
real_spawn = control.subprocess.Popen
def spawn(*_, **kwargs):
    return real_spawn([sys.executable, '-c', {fake!r}, entry['run_directory']], **kwargs)
control.subprocess.Popen = spawn
sys.argv = ['control', '--entry', {str(path)!r}, '--review', {str(review)!r}]
raise SystemExit(control.main())
"""
                env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
                process = subprocess.Popen([sys.executable, "-c", parent], env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = Path(entry["run_directory"])
                try:
                    limit = time.monotonic() + 10
                    while not (output / "fake-child.json").exists() and time.monotonic() < limit:
                        time.sleep(.05)
                    self.assertTrue((output / "fake-child.json").exists())
                    launch = json.loads((output / "launch.json").read_text())
                    child = json.loads((output / "fake-child.json").read_text())["pid"]
                    os.kill(process.pid, signum)
                    process.communicate(timeout=20)
                    limit = time.monotonic() + 5
                    while time.monotonic() < limit:
                        identities = [control.process_identity(pid) for pid in (launch["worker_pid"], child)]
                        if all(row is None or row["state"] == "Z" for row in identities): break
                        time.sleep(.05)
                    self.assertTrue(all(row is None or row["state"] == "Z" for row in identities))
                    if signum == signal.SIGTERM:
                        self.assertEqual(process.returncode, 1)
                        terminal = json.loads((output / "terminal.json").read_text())
                        self.assertEqual(terminal["status"], "SUPERVISOR_FAILURE")
                        self.assertTrue(terminal["cleanup_ok"])
                    else:
                        self.assertEqual(process.returncode, -signal.SIGKILL)
                        self.assertEqual(json.loads((output / "worker-stop.json").read_text())["reason"], "SUPERVISOR_LOST")
                        self.assertFalse((output / "terminal.json").exists())
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
