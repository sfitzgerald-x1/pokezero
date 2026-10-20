"""Authored supervisor fixtures. No scientific game, model or search execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "panel", Path(__file__).resolve().parents[1] / "scripts" / "run_fullgame_panel.py")
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


class PanelTests(unittest.TestCase):
    def coordinated_fixture(self, directory):
        entry, review, output = self.fixture_entry(directory)
        doc = json.loads(entry.read_text())
        doc["coordination"] = dict(lease_path=str(Path(directory) / "lease.json"),
                                   abort_path=str(Path(directory) / "abort.json"),
                                   lease_timeout_seconds=60)
        panel.write(entry, doc)
        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
        return entry, review, output, doc

    def test_shared_deadline_cannot_extend_the_entry_cap_or_be_nonfinite(self):
        for deadline in (86401, float("inf"), float("nan"), True):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as directory:
                entry, review, output = self.fixture_entry(directory)
                with patch.object(panel, "verify"), patch.object(panel.time, "time", return_value=0), \
                        patch.object(panel.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(ValueError, "shared deadline"):
                        panel.run(entry, review, deadline_unix=deadline)
                self.assertFalse(output.exists())
                spawn.assert_not_called()

    def test_coordinated_entry_cannot_start_a_replacement_relative_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.coordinated_fixture(directory)
            with patch.object(panel, "verify"), patch.object(panel.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "common absolute deadline"):
                    panel.run(entry, review)
            self.assertFalse(output.exists())
            spawn.assert_not_called()

    def test_already_expired_common_deadline_prevents_every_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.coordinated_fixture(directory)
            with patch.object(panel.sys, "platform", "linux"), patch.object(panel, "verify"), \
                    patch.object(panel.time, "time", return_value=10), \
                    patch.object(panel.subprocess, "Popen") as spawn:
                self.assertEqual(panel.run(entry, review, deadline_unix=5), 1)
            spawn.assert_not_called()
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "CALENDAR_CAP")
            self.assertEqual(terminal["deadline_unix"], 5)
            self.assertEqual(terminal["deadline_source"], "shared_absolute")

    def test_lease_binds_entry_and_deadline_without_accepting_a_future_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, _review, _output, doc = self.coordinated_fixture(directory)
            entry_hash = panel.digest(entry)
            lease = Path(doc["coordination"]["lease_path"])
            for payload, mtime, expected in (
                (dict(entry_sha256=entry_hash, deadline_unix=2000), 1000, None),
                (dict(entry_sha256="another-entry", deadline_unix=2000), 1000, "COORDINATION_INVALID"),
                (dict(entry_sha256=entry_hash, deadline_unix=3000), 1000, "COORDINATION_INVALID"),
                (dict(entry_sha256=entry_hash, deadline_unix=True), 1000, "COORDINATION_INVALID"),
                (dict(entry_sha256=entry_hash, deadline_unix=2000), 1002, "COORDINATION_INVALID"),
                (dict(entry_sha256=entry_hash, deadline_unix=2000), 939, "COORDINATOR_LOST"),
            ):
                with self.subTest(payload=payload, mtime=mtime):
                    panel.write(lease, payload)
                    os.utime(lease, (mtime, mtime))
                    self.assertEqual(panel.coordination_reason(doc, entry_hash, 2000, 1000), expected)

    def test_missing_lease_stops_before_any_worker_even_with_time_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.coordinated_fixture(directory)
            with patch.object(panel.sys, "platform", "linux"), patch.object(panel, "verify"), \
                    patch.object(panel.time, "time", return_value=10), \
                    patch.object(panel.subprocess, "Popen") as spawn:
                self.assertEqual(panel.run(entry, review, deadline_unix=1000), 1)
            spawn.assert_not_called()
            self.assertEqual(json.loads((output / "terminal.json").read_text())["status"],
                             "COORDINATOR_LOST")

    def test_coordination_failure_stops_owned_worker_and_cannot_launch_next_shard(self):
        for reason in ("COORDINATOR_ABORT", "COORDINATOR_LOST", "COORDINATION_INVALID"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                entry, review, output, doc = self.coordinated_fixture(directory)
                doc["shards"].append(dict(name="raw-p1", seat="p1", arguments=[]))
                panel.write(entry, doc)
                panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
                worker = Mock(pid=123456)
                worker.poll.return_value = None
                with patch.object(panel.sys, "platform", "linux"), patch.object(panel, "verify"), \
                        patch.object(panel.time, "time", return_value=10), \
                        patch.object(panel, "coordination_reason", side_effect=[None, reason]), \
                        patch.object(panel, "stop_owned", return_value=(-15, True)) as stop, \
                        patch.object(panel.subprocess, "Popen", return_value=worker) as spawn:
                    self.assertEqual(panel.run(entry, review, deadline_unix=1000), 1)
                spawn.assert_called_once()
                stop.assert_called_once_with(worker)
                terminal = json.loads((output / "terminal.json").read_text())
                self.assertEqual(terminal["status"], reason)
                self.assertEqual(terminal["shards"][0]["exit_code"], -15)
                self.assertEqual(terminal["deadline_unix"], 1000)
                self.assertFalse((output / "raw-p1-launch.json").exists())

    def test_abort_marker_stops_a_fresh_valid_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, _review, _output, doc = self.coordinated_fixture(directory)
            entry_hash = panel.digest(entry)
            lease = Path(doc["coordination"]["lease_path"])
            panel.write(lease, dict(entry_sha256=entry_hash, deadline_unix=2000))
            os.utime(lease, (1000, 1000))
            panel.write(doc["coordination"]["abort_path"], dict(reason="PEER_FAILURE"))
            self.assertEqual(panel.coordination_reason(doc, entry_hash, 2000, 1000),
                             "COORDINATOR_ABORT")

    def test_zero_exit_complete_prefix_still_checks_guards_at_acceptance(self):
        for coordinated, reason in (
            (False, "CALENDAR_CAP"), (True, "CALENDAR_CAP"),
            (True, "COORDINATOR_ABORT"), (True, "COORDINATOR_LOST"),
            (True, "COORDINATION_INVALID"), (False, None), (True, None),
        ):
            for later_shard in (False, True):
                with self.subTest(coordinated=coordinated, reason=reason,
                                  later_shard=later_shard), \
                        tempfile.TemporaryDirectory() as directory:
                    if coordinated:
                        entry, review, output, doc = self.coordinated_fixture(directory)
                    else:
                        entry, review, output = self.fixture_entry(directory)
                        doc = json.loads(entry.read_text())
                    if later_shard:
                        doc["shards"].append(dict(name="raw-p1", seat="p1", arguments=[]))
                        panel.write(entry, doc)
                        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
                    worker = Mock(pid=123456)
                    worker.poll.return_value = 0  # Never enters the polling loop.
                    # Entry start, prelaunch, shard start, then post-verification.
                    deadline = 1000 if coordinated else 86400
                    clock = [0, 1, 2, deadline if reason == "CALENDAR_CAP" else 3]
                    with patch.object(panel.sys, "platform", "linux"), \
                            patch.object(panel, "verify"), \
                            patch.object(panel.time, "time", side_effect=clock + [3] * 10), \
                            patch.object(panel, "observed_count", return_value=128), \
                            patch.object(panel, "coordination_reason", side_effect=[None, reason] + [None] * 4), \
                            patch.object(panel, "stop_owned", return_value=(0, True)), \
                            patch.object(panel.subprocess, "Popen", return_value=worker) as spawn:
                        code = panel.run(entry, review,
                                         deadline_unix=deadline if coordinated else None)
                    terminal = json.loads((output / "terminal.json").read_text())
                    shard = json.loads((output / "candidate-p1-terminal.json").read_text())
                    self.assertEqual(shard["exit_code"], 0)
                    self.assertEqual(shard["observed_games"], 128)
                    self.assertTrue(shard["cleanup_ok"])
                    if reason is not None:
                        self.assertEqual(code, 1)
                        self.assertEqual(terminal["status"], reason)
                        self.assertEqual(shard["stop_reason"], reason)
                        spawn.assert_called_once()
                        self.assertFalse((output / "raw-p1-launch.json").exists())
                    else:
                        self.assertEqual(code, 0)
                        self.assertEqual(terminal["status"], "COLLECTION_ATTEMPT_COMPLETE")
                        self.assertIsNone(shard["stop_reason"])
                        self.assertEqual(spawn.call_count, 2 if later_shard else 1)

    def test_missing_summary_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(panel.observed_count(Path(directory) / "none.json", 10, 128), 0)

    def test_prefix_must_match_frozen_seed_order_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            panel.write(path, {"completed_games": 2, "game_results": [{"seed": 10}, {"seed": 11}]})
            self.assertEqual(panel.observed_count(path, 10, 128), 2)
            for doc in ({"completed_games": 1, "game_results": [{"seed": 11}]},
                        {"completed_games": 2, "game_results": [{"seed": 10}]},
                        {"completed_games": 2, "game_results": [{"seed": 10}, {"seed": 10}]}):
                panel.write(path, doc)
                with self.assertRaises(ValueError):
                    panel.observed_count(path, 10, 128)

    def test_native_and_source_input_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"frozen")
            entry = dict(input_hashes={str(path): panel.digest(path)},
                         tree_hashes={}, repositories={})
            panel.verify(entry)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                panel.verify(entry)

    def test_compiled_simulator_additions_are_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.js").write_text("a")
            initial = panel.tree_digest(root)
            (root / "b.js").write_text("b")
            self.assertNotEqual(panel.tree_digest(root), initial)

    def test_actual_nonzero_exit_stays_nonzero(self):
        worker = Mock(pid=123456)
        worker.wait.return_value = 1
        with patch.object(panel, "group_exists", return_value=False):
            self.assertEqual(panel.stop_owned(worker), (1, True))
        worker.wait.assert_called_once_with(timeout=10)

    def test_owned_timeout_escalates_without_broad_kill(self):
        worker = Mock(pid=123456)
        worker.wait.side_effect = [subprocess.TimeoutExpired("owned", 10), -9]
        with patch.object(panel, "group_exists", side_effect=[True, True, False, False]), \
                patch.object(panel.os, "killpg") as kill:
            self.assertEqual(panel.stop_owned(worker), (-9, True))
        self.assertEqual([call.args[0] for call in kill.call_args_list], [123456, 123456])

    def test_unreviewed_entry_cannot_create_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "entry.json"
            review = Path(directory) / "review.json"
            output = Path(directory) / "run"
            panel.write(entry, {"run_directory": str(output)})
            panel.write(review, {"status": "CLEAR", "entry_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "exact entry"):
                panel.run(entry, review)
            self.assertFalse(output.exists())

    def fixture_entry(self, directory):
        root = Path(directory)
        entry = root / "entry.json"
        review = root / "review.json"
        panel.write(entry, dict(run_directory=str(root / "run"), environment={},
                    calendar_seconds=86400, no_completion_seconds=1800,
                    shards=[dict(name="candidate-p1", seat="p1", arguments=[])],
                    python="fixture-python", runtime=directory, common_arguments=[],
                    seed_count=128, seed_start=10, foulplay_seed_start=10))
        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
        return entry, review, root / "run"

    def test_calendar_cap_prevents_worker_launch_and_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            assertion = Mock()
            with patch.object(panel.sys, "platform", "darwin"), \
                    patch.object(panel, "verify"), \
                    patch.object(panel.time, "time", side_effect=[0, 86401, 86402]), \
                    patch.object(panel.subprocess, "Popen", return_value=assertion) as spawn:
                self.assertEqual(panel.run(entry, review), 1)
            self.assertEqual(spawn.call_count, 1)  # Sleep assertion, never a worker.
            self.assertEqual(spawn.call_args.args[0][0], "caffeinate")
            doc = json.loads((output / "terminal.json").read_text())
            self.assertEqual(doc["status"], "CALENDAR_CAP")
            self.assertEqual(doc["shards"], [])

    def test_failed_worker_stops_and_preserves_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            assertion, worker = Mock(), Mock(pid=123456)
            worker.poll.return_value = 1
            worker.wait.return_value = 1
            with patch.object(panel.sys, "platform", "darwin"), \
                    patch.object(panel, "verify"), \
                    patch.object(panel, "group_exists", return_value=False), \
                    patch.object(panel.subprocess, "Popen", side_effect=[assertion, worker]):
                self.assertEqual(panel.run(entry, review), 1)
            doc = json.loads((output / "terminal.json").read_text())
            self.assertEqual(doc["status"], "PROCESS_FAILURE")
            self.assertEqual(doc["shards"][0]["exit_code"], 1)
            self.assertEqual(doc["shards"][0]["observed_games"], 0)

    def test_post_exit_drift_cannot_erase_primary_worker_failure(self):
        for code, clean in ((1, True), (0, False)):
            with self.subTest(code=code, clean=clean), tempfile.TemporaryDirectory() as directory:
                entry, review, output = self.fixture_entry(directory)
                worker = Mock(pid=123456)
                worker.poll.return_value = code
                with patch.object(panel.sys, "platform", "linux"), \
                        patch.object(panel, "verify", side_effect=[None, None, ValueError("input changed: plan")]), \
                        patch.object(panel, "stop_owned", return_value=(code, clean)), \
                        patch.object(panel.subprocess, "Popen", return_value=worker) as spawn:
                    self.assertEqual(panel.run(entry, review), 1)
                terminal = json.loads((output / "terminal.json").read_text())
                shard = json.loads((output / "candidate-p1-terminal.json").read_text())
                self.assertEqual(terminal["status"], "PROCESS_FAILURE")
                self.assertEqual(shard["exit_code"], code)
                self.assertEqual(shard["cleanup_ok"], clean)
                self.assertEqual(shard["observed_games"], 0)
                self.assertEqual(terminal["shards"][0], shard)
                self.assertEqual(terminal["post_exit_verification_error"],
                                 {"error_type": "ValueError", "error": "input changed: plan"})
                spawn.assert_called_once()

    def test_post_exit_drift_on_clean_complete_child_remains_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            worker = Mock(pid=123456)
            worker.poll.return_value = 0
            with patch.object(panel.sys, "platform", "linux"), \
                    patch.object(panel, "verify", side_effect=[None, None, ValueError("input changed: plan")]), \
                    patch.object(panel, "stop_owned", return_value=(0, True)), \
                    patch.object(panel, "observed_count", return_value=128), \
                    patch.object(panel.subprocess, "Popen", return_value=worker), \
                    self.assertRaisesRegex(ValueError, "input changed: plan"):
                panel.run(entry, review)
            terminal = json.loads((output / "terminal.json").read_text())
            shard = json.loads((output / "candidate-p1-terminal.json").read_text())
            self.assertEqual(terminal["status"], "SUPERVISOR_FAILURE")
            self.assertEqual(shard["exit_code"], 0)
            self.assertEqual(shard["observed_games"], 128)
            self.assertTrue(shard["cleanup_ok"])
            self.assertIn("post_exit_verification_error", terminal)

    def test_linux_calendar_cap_never_spawns_sleep_assertion_or_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            with patch.object(panel.sys, "platform", "linux"), \
                    patch.object(panel, "verify"), \
                    patch.object(panel.time, "time", side_effect=[0, 86401, 86402]), \
                    patch.object(panel.subprocess, "Popen") as spawn:
                self.assertEqual(panel.run(entry, review), 1)
            spawn.assert_not_called()
            self.assertEqual(json.loads((output / "terminal.json").read_text())["status"],
                             "CALENDAR_CAP")

    def test_linux_failure_preserves_actual_worker_exit_without_caffeinate(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            worker = Mock(pid=123456)
            worker.poll.return_value = 1
            worker.wait.return_value = 1
            with patch.object(panel.sys, "platform", "linux"), \
                    patch.object(panel, "verify"), \
                    patch.object(panel, "group_exists", return_value=False), \
                    patch.object(panel.subprocess, "Popen", return_value=worker) as spawn:
                self.assertEqual(panel.run(entry, review), 1)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.args[0][0], "fixture-python")
            doc = json.loads((output / "terminal.json").read_text())
            self.assertEqual(doc["status"], "PROCESS_FAILURE")
            self.assertEqual(doc["shards"][0]["exit_code"], 1)
            self.assertTrue(doc["shards"][0]["cleanup_ok"])


if __name__ == "__main__":
    unittest.main()
