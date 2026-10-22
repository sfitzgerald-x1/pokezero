"""Authored supervisor fixtures. No scientific game, model or search execution."""
import importlib.util
from contextlib import ExitStack
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


class ParallelSeatTests(unittest.TestCase):
    def fixture(self, directory):
        entry, review, output = PanelTests().fixture_entry(directory)
        doc = json.loads(entry.read_text())
        doc.update(shard_execution="parallel_seats_fixed_work", comparison_kind="raw_head_to_head",
                   common_arguments=["--policy-mode", "engine-mcts", "--opponent-policy-mode", "raw",
                                     "--device", "cpu", "--no-search-fallback", "--engine-worlds", "4",
                                     "--engine-sims", "256", "--engine-depth", "6", "--engine-batch", "16",
                                     "--engine-model-world-workers", "1"],
                   shards=[dict(name="candidate-" + seat, seat=seat, arguments=[])
                           for seat in ("p1", "p2")])
        self.save(entry, review, doc)
        return entry, review, output, doc

    def save(self, entry, review, doc):
        panel.write(entry, doc)
        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))

    def execute(self, entry, review, *, counts=None, workers=None, stop=None, verify=None,
                before_sleep=None, deadline=None):
        workers = workers or [Mock(pid=123001), Mock(pid=123002)]
        for worker in workers:
            if not isinstance(worker, Exception):
                worker.poll.return_value = 0
        clock = [0]

        def sleep(seconds):
            clock[0] += seconds
            if before_sleep:
                before_sleep(clock[0], workers)

        with ExitStack() as stack:
            stack.enter_context(patch.object(panel, "print", create=True))
            stack.enter_context(patch.object(panel.sys, "platform", "linux"))
            stack.enter_context(patch.object(panel, "verify", side_effect=verify))
            stack.enter_context(patch.object(panel.time, "time", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(panel.time, "sleep", side_effect=sleep))
            spawn = stack.enter_context(patch.object(panel.subprocess, "Popen", side_effect=workers))
            drain = stack.enter_context(patch.object(panel, "stop_owned", side_effect=stop,
                                                     return_value=(0, True)))
            stack.enter_context(patch.object(panel, "observed_count", side_effect=counts,
                                             return_value=128))
            result = panel.run(entry, review, deadline_unix=deadline)
        return result, spawn, drain

    def test_both_launch_before_poll_with_one_deadline_and_no_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)
            first, second = Mock(pid=123001), Mock(pid=123002)

            def first_poll():
                self.assertTrue((output / "candidate-p2-launch.json").exists())
                return 0

            first.poll.side_effect = first_poll
            result, spawn, drain = self.execute(entry, review, workers=[first, second], deadline=1000)
            self.assertEqual(result, 0)
            self.assertEqual(spawn.call_count, 2)
            self.assertEqual(drain.call_count, 2)
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "COLLECTION_ATTEMPT_COMPLETE")
            self.assertEqual(terminal["scientific_qualification"], "UNREVIEWED")
            self.assertEqual(terminal["shard_execution"], "parallel_seats_fixed_work")
            for seat in ("p1", "p2"):
                launch = json.loads((output / f"candidate-{seat}-launch.json").read_text())
                receipt = json.loads((output / f"candidate-{seat}-terminal.json").read_text())
                self.assertEqual(launch["deadline_unix"], 1000)
                self.assertEqual(receipt["exit_code"], 0)
                self.assertTrue(receipt["cleanup_ok"])
                self.assertEqual(receipt["observed_games"], 128)
            with patch.object(panel, "verify"), patch.object(panel.subprocess, "Popen") as retry:
                with self.assertRaises(FileExistsError):
                    panel.run(entry, review)
            retry.assert_not_called()

    def test_timed_dynamic_abbreviated_or_per_seat_overrides_cannot_opt_in(self):
        for arguments in (["--engine-early-stop"], ["--engine-early-st"],
                          ["--root-time-budget-ms", "1000"], ["--engine-depth-min", "2"],
                          ["--policy-mode", "trajectory-reference"], ["--device", "cuda"],
                          ["--engine-sims", "0"], ["--eng", "2"]):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                entry, review, output, doc = self.fixture(directory)
                doc["common_arguments"].extend(arguments)
                self.save(entry, review, doc)
                with patch.object(panel, "verify"), patch.object(panel.subprocess, "Popen") as spawn:
                    with self.assertRaises(ValueError):
                        panel.run(entry, review)
                spawn.assert_not_called()
                self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as directory:
            entry, review, _output, doc = self.fixture(directory)
            doc["shards"][1]["arguments"] = ["--engine-early-stop"]
            with self.assertRaisesRegex(ValueError, "own phase receipt"):
                panel.validate_execution(doc)
            doc["shards"][1]["arguments"] = ["--phase-progress-out",
                                                   str(Path(doc["run_directory"]) / "candidate-p2-phase.json")]
            self.assertEqual(panel.validate_execution(doc), "parallel_seats_fixed_work")

    def test_parallel_contract_does_not_expand_to_other_rosters_or_coordination(self):
        for field, value in (("shard_execution", "parallel"), ("comparison_kind", "foulplay"),
                             ("coordination", {}), ("shards", [])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                entry, review, output, doc = self.fixture(directory)
                doc[field] = value
                self.save(entry, review, doc)
                with patch.object(panel.subprocess, "Popen") as spawn:
                    with self.assertRaises(ValueError):
                        panel.run(entry, review)
                spawn.assert_not_called()
                self.assertFalse(output.exists())

    def test_either_nonzero_stops_peer_and_preserves_both_actual_exits(self):
        for failing_seat in (0, 1):
            with self.subTest(failing_seat=failing_seat), tempfile.TemporaryDirectory() as directory:
                entry, review, output, _doc = self.fixture(directory)
                workers = [Mock(pid=123001), Mock(pid=123002)]
                for i, worker in enumerate(workers):
                    worker.poll.side_effect = lambda i=i: 1 if i == failing_seat else None
                result, spawn, drain = self.execute(
                    entry, review, workers=workers, stop=[(1, True), (-15, True)])
                self.assertEqual(result, 1)
                self.assertEqual(spawn.call_count, 2)
                self.assertEqual([call.args[0] for call in drain.call_args_list],
                                 [workers[failing_seat], workers[1 - failing_seat]])
                terminal = json.loads((output / "terminal.json").read_text())
                self.assertEqual(terminal["status"], "PROCESS_FAILURE")
                self.assertEqual([s["exit_code"] for s in terminal["shards"]], [1, -15])

    def test_second_launch_failure_drains_first_and_records_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)
            with self.assertRaisesRegex(OSError, "fixture launch failed"):
                self.execute(entry, review, workers=[Mock(pid=123001), OSError("fixture launch failed")],
                             counts=lambda *_: 0, stop=[(-15, True)])
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "SUPERVISOR_FAILURE")
            self.assertEqual(terminal["shards"][0]["exit_code"], -15)
            self.assertFalse((output / "candidate-p2-launch.json").exists())

    def test_malformed_prefix_cannot_prevent_peer_drain_or_erase_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)

            def count(path, *_):
                if path.name == "candidate-p1.json":
                    raise ValueError("fixture malformed prefix")
                return 0

            with self.assertRaisesRegex(RuntimeError, "cleanup/receipt failure"):
                self.execute(entry, review, counts=count, stop=[(-15, True), (-15, True)])
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(len(terminal["shards"]), 2)
            self.assertEqual([s["exit_code"] for s in terminal["shards"]], [-15, -15])
            self.assertEqual(terminal["drain_errors"][0]["name"], "candidate-p1")
            self.assertIn("prefix_error", terminal["shards"][0])

    def test_one_seat_progress_cannot_renew_other_seats_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, doc = self.fixture(directory)
            doc["no_completion_seconds"] = 10
            self.save(entry, review, doc)
            workers = [Mock(pid=123001), Mock(pid=123002)]
            for worker in workers:
                worker.poll.side_effect = lambda: None
            seen = [0]

            def count(path, *_):
                if path.name == "candidate-p1.json":
                    seen[0] += 1
                    return seen[0]
                return 0

            result, _spawn, drain = self.execute(entry, review, workers=workers, counts=count,
                                               stop=[(-15, True), (-15, True)])
            self.assertEqual(result, 1)
            self.assertEqual(drain.call_count, 2)
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "NO_COMPLETION_TIMEOUT")
            self.assertEqual(terminal["stopping_shard"], "candidate-p2")
            self.assertTrue((output / "candidate-p1-progress.json").exists())
            self.assertFalse((output / "candidate-p2-progress.json").exists())

    def test_calendar_cap_drains_both_even_with_advancing_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)
            workers = [Mock(pid=123001), Mock(pid=123002)]
            for worker in workers:
                worker.poll.side_effect = lambda: None
            result, _spawn, drain = self.execute(entry, review, workers=workers, deadline=5,
                                               stop=[(-15, True), (-15, True)])
            self.assertEqual(result, 1)
            self.assertEqual(drain.call_count, 2)
            self.assertEqual(json.loads((output / "terminal.json").read_text())["status"], "CALENDAR_CAP")

    def test_zero_exit_incomplete_prefix_drains_peer_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "without the planned completed prefix"):
                self.execute(entry, review, counts=lambda *_: 2, stop=[(0, True), (-15, True)])
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "SUPERVISOR_FAILURE")
            self.assertEqual([s["exit_code"] for s in terminal["shards"]], [0, -15])

    def test_post_exit_provenance_failure_still_drains_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "fixture input drift"):
                self.execute(entry, review, verify=[None, None, None, ValueError("fixture input drift"), None],
                             stop=[(0, True), (-15, True)])
            terminal = json.loads((output / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "SUPERVISOR_FAILURE")
            self.assertEqual([s["exit_code"] for s in terminal["shards"]], [0, -15])


if __name__ == "__main__":
    unittest.main()
