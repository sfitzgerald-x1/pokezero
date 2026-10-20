"""Authored supervisor fixtures. No scientific game, model or search execution."""
import importlib.util
import json
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
            with patch.object(panel, "verify"), \
                    patch.object(panel.time, "time", side_effect=[0, 86401, 86402]), \
                    patch.object(panel.subprocess, "Popen", return_value=assertion) as spawn:
                self.assertEqual(panel.run(entry, review), 1)
            self.assertEqual(spawn.call_count, 1)  # Sleep assertion, never a worker.
            doc = json.loads((output / "terminal.json").read_text())
            self.assertEqual(doc["status"], "CALENDAR_CAP")
            self.assertEqual(doc["shards"], [])

    def test_failed_worker_stops_and_preserves_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output = self.fixture_entry(directory)
            assertion, worker = Mock(), Mock(pid=123456)
            worker.poll.return_value = 1
            worker.wait.return_value = 1
            with patch.object(panel, "verify"), \
                    patch.object(panel, "group_exists", return_value=False), \
                    patch.object(panel.subprocess, "Popen", side_effect=[assertion, worker]):
                self.assertEqual(panel.run(entry, review), 1)
            doc = json.loads((output / "terminal.json").read_text())
            self.assertEqual(doc["status"], "PROCESS_FAILURE")
            self.assertEqual(doc["shards"][0]["exit_code"], 1)
            self.assertEqual(doc["shards"][0]["observed_games"], 0)


if __name__ == "__main__":
    unittest.main()
