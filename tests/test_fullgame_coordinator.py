"""Authored transport/process fixtures only: no scientific games or pods."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "coordinator", Path(__file__).resolve().parents[1] / "scripts" / "coordinate_fullgame_panels.py")
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)
panel = coordinator.panel
SHARDS = [["candidate-p1", "p1"], ["raw-p1", "p1"], ["candidate-p2", "p2"], ["raw-p2", "p2"]]


class FixtureTransport:
    def __init__(self, states=None, fail=None, drained=None, head_to_head=False):
        self.calls = []
        self.started = set()
        self.polls = {}
        self.states = states or {}
        self.fail = fail
        self.drained = drained
        self.observations = {}
        self.head_to_head = head_to_head

    def preflight_pod(self, target):
        self.calls.append((target["name"], "pod"))
        if self.fail == (target["name"], "pod"):
            raise ValueError("changed pod")

    def call(self, target, action, deadline=None, cursors=None):
        name = target["name"]
        self.calls.append((name, action))
        if self.fail == (name, action):
            raise subprocess.TimeoutExpired(action, 15)
        if action == "preflight":
            if self.head_to_head:
                return dict(entry_sha256=target["entry_sha256"], seed_start=10 if name == "a" else 42,
                            seed_count=32, calendar_seconds=86400,
                            shards=coordinator.HEAD_TO_HEAD_SHARDS,
                            comparison_kind=coordinator.HEAD_TO_HEAD_KIND,
                            candidate_policy_mode="trajectory-reference", opponent_policy_mode="raw")
            return dict(entry_sha256=target["entry_sha256"], seed_start=10 if name == "a" else 74,
                        seed_count=64, calendar_seconds=86400, shards=SHARDS)
        if action == "start":
            self.started.add(name)
            return dict(deadline_unix=deadline, identity=dict(pid=123, start_ticks=12))
        if action == "abort":
            return dict(abort_recorded=True)
        if name not in self.started:
            return dict(documents={}, counts={}, launcher_alive=False, supervisor_exit=None)
        counters = self.observations if action == "observe" else self.polls
        i = counters.get(name, 0)
        counters[name] = i + 1
        sequences = self.drained if action == "observe" and self.drained is not None else self.states
        sequence = sequences.get(name, ["complete"])
        state = sequence[min(i, len(sequence) - 1)]
        if state == "complete":
            return dict(documents={"terminal.json": dict(status="COLLECTION_ATTEMPT_COMPLETE",
                        deadline_unix=deadline, ended_unix=deadline - 1)}, counts={"candidate-p1": 64},
                        supervisor_exit=dict(exit_code=0, deadline_unix=deadline), launcher_alive=False)
        if state == "failed":
            return dict(documents={"terminal.json": dict(status="PROCESS_FAILURE")}, counts={},
                        supervisor_exit=None, launcher_alive=True)
        if state == "late":
            return dict(documents={"terminal.json": dict(status="COLLECTION_ATTEMPT_COMPLETE",
                        deadline_unix=deadline, ended_unix=deadline + 1)}, counts={},
                        supervisor_exit=dict(exit_code=0, deadline_unix=deadline), launcher_alive=False)
        if state == "stopped":
            return dict(documents={"terminal.json": dict(status="COORDINATOR_ABORT", cleanup_ok=True),
                        "candidate-p1.json": dict(completed_games=2, game_results=[dict(seed=10), dict(seed=11)])},
                        counts={"candidate-p1": 2}, supervisor_exit=dict(exit_code=1, deadline_unix=deadline),
                        launcher_alive=False)
        return dict(documents={}, counts={}, supervisor_exit=None, launcher_alive=(state != "lost"))


class CoordinatorTests(unittest.TestCase):
    def controller_fixture(self, directory):
        root = Path(directory)
        entry, review = root / "entry.json", root / "review.json"
        doc = dict(run_directory=str(root / "run"), calendar_seconds=86400,
                   coordinator_sha256=panel.digest(coordinator.__file__),
                   seed_start=10, seed_count=128, targets=[dict(name=name,
                   entry_sha256=name * 64, pod_uid="uid-" + name, node_name="node-" + name)
                   for name in ("a", "b")])
        panel.write(entry, doc)
        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
        return entry, review, root / "run", doc

    def execute(self, entry, review, transport):
        with patch.object(coordinator.sys, "platform", "linux"), \
                patch.object(coordinator.time, "time", return_value=10), \
                patch.object(coordinator.time, "sleep"):
            return coordinator.run(entry, review, transport)

    def test_complete_partitions_share_one_clock_and_export_outside_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport()
            self.assertEqual(self.execute(entry, review, transport), 0)
            terminal = coordinator.read(output / "terminal.json")
            self.assertEqual(terminal["deadline_unix"], 86410)
            self.assertEqual(terminal["status"], "COLLECTION_ATTEMPT_COMPLETE")
            self.assertEqual(terminal["scientific_qualification"], "UNREVIEWED")
            for name in ("a", "b"):
                self.assertEqual(transport.calls.count((name, "start")), 1)
                self.assertTrue((output / name / "terminal.json").exists())
                self.assertEqual(terminal["launches"][name]["deadline_unix"], 86410)

    def test_changed_pod_prevents_all_launches_and_clock_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(fail=("b", "pod"))
            with self.assertRaisesRegex(ValueError, "changed pod"):
                self.execute(entry, review, transport)
            self.assertFalse(output.exists())
            self.assertFalse(any(action == "start" for _name, action in transport.calls))

    def test_head_to_head_two_seat_partitions_keep_one_clock_and_fixed_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, doc = self.controller_fixture(directory)
            doc.update(comparison_kind=coordinator.HEAD_TO_HEAD_KIND, seed_count=64)
            panel.write(entry, doc)
            panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
            transport = FixtureTransport(head_to_head=True)
            self.assertEqual(self.execute(entry, review, transport), 0)
            terminal = coordinator.read(output / "terminal.json")
            self.assertEqual(terminal["deadline_unix"], 86410)
            self.assertEqual(terminal["scientific_qualification"], "UNREVIEWED")
            for name in ("a", "b"):
                self.assertEqual(transport.calls.count((name, "start")), 1)

    def test_head_to_head_mode_cannot_waive_geometry_or_worker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            _entry, _review, _output, doc = self.controller_fixture(directory)
            doc.update(comparison_kind=coordinator.HEAD_TO_HEAD_KIND, seed_count=64)
            transport = FixtureTransport(head_to_head=True)
            facts = [transport.call(t, "preflight") for t in doc["targets"]]
            coordinator.validate_partitions(doc, facts)
            defects = ("primary-default", "unknown-kind", "four-shards", "one-seat", "wrong-seat",
                       "missing-worker-kind", "worker-kind", "opponent", "raw-candidate",
                       "different-candidates", "missing-preflight", "extra-preflight", "gap", "overlap")
            for defect in defects:
                damaged_doc, damaged = copy.deepcopy(doc), copy.deepcopy(facts)
                if defect == "primary-default":
                    damaged_doc.pop("comparison_kind")
                elif defect == "unknown-kind":
                    damaged_doc["comparison_kind"] = "unregistered"
                elif defect == "four-shards":
                    damaged[0]["shards"] = SHARDS
                elif defect == "one-seat":
                    damaged[0]["shards"] = coordinator.HEAD_TO_HEAD_SHARDS[:1]
                elif defect == "wrong-seat":
                    damaged[0]["shards"][1][1] = "p1"
                elif defect == "missing-worker-kind":
                    damaged[0].pop("comparison_kind")
                elif defect == "worker-kind":
                    damaged[0]["comparison_kind"] = coordinator.PRIMARY_KIND
                elif defect == "opponent":
                    damaged[0]["opponent_policy_mode"] = "foulplay"
                elif defect == "raw-candidate":
                    damaged[0]["candidate_policy_mode"] = "raw"
                elif defect == "different-candidates":
                    damaged[0]["candidate_policy_mode"] = "engine-mcts"
                elif defect == "missing-preflight":
                    damaged.pop()
                elif defect == "extra-preflight":
                    damaged.append(copy.deepcopy(damaged[0]))
                elif defect == "gap":
                    damaged[1]["seed_start"] += 1
                else:
                    damaged[1]["seed_start"] -= 1
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    coordinator.validate_partitions(damaged_doc, damaged)

    def test_primary_default_still_requires_all_four_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            _entry, _review, _output, doc = self.controller_fixture(directory)
            transport = FixtureTransport()
            facts = [transport.call(t, "preflight") for t in doc["targets"]]
            coordinator.validate_partitions(doc, facts)
            for fact in facts:
                fact["shards"] = coordinator.HEAD_TO_HEAD_SHARDS
            with self.assertRaises(ValueError):
                coordinator.validate_partitions(doc, facts)

    def test_partition_overlap_gap_wrong_seats_and_same_node_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _entry, _review, _output, doc = self.controller_fixture(directory)
            transport = FixtureTransport()
            facts = [transport.call(t, "preflight") for t in doc["targets"]]
            for defect in ("overlap", "gap", "seats", "node", "binding", "cap"):
                damaged_doc = json.loads(json.dumps(doc))
                damaged = json.loads(json.dumps(facts))
                if defect == "overlap":
                    damaged[1]["seed_start"] -= 1
                elif defect == "gap":
                    damaged[1]["seed_start"] += 1
                elif defect == "seats":
                    damaged[1]["shards"] = SHARDS[:2]
                elif defect == "node":
                    damaged_doc["targets"][1]["node_name"] = "node-a"
                elif defect == "binding":
                    damaged[1]["entry_sha256"] = "wrong"
                else:
                    damaged_doc["calendar_seconds"] = 172800
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    coordinator.validate_partitions(damaged_doc, damaged)

    def test_launch_transport_failure_is_not_retried_and_aborts_both(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(fail=("b", "start"))
            self.assertEqual(self.execute(entry, review, transport), 1)
            for name in ("a", "b"):
                self.assertEqual(transport.calls.count((name, "start")), 1)
                self.assertEqual(transport.calls.count((name, "abort")), 1)
            self.assertFalse(coordinator.read(output / "terminal.json")["cleanup_verified"])

    def test_failure_loss_late_exit_or_observation_timeout_aborts_peer(self):
        for failure in ("failed", "lost", "late", "transport"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                entry, review, output, _doc = self.controller_fixture(directory)
                transport = FixtureTransport(states={"a": [failure], "b": ["running"]})
                if failure == "transport":
                    # Fail refresh only after the start action, not initial admission.
                    original = transport.call
                    def call(target, action, *args):
                        if target["name"] == "a" and action == "refresh" and "a" in transport.started:
                            raise subprocess.TimeoutExpired(action, 15)
                        return original(target, action, *args)
                    transport.call = call
                self.assertEqual(self.execute(entry, review, transport), 1)
                terminal = coordinator.read(output / "terminal.json")
                self.assertEqual(terminal["status"], "COORDINATOR_STOP")
                self.assertIn(("b", "abort"), transport.calls)
                self.assertFalse(terminal["cleanup_verified"])

    def test_completed_partition_is_not_repolled_while_peer_progresses(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, _output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(states={"b": ["running", "complete"]})
            self.assertEqual(self.execute(entry, review, transport), 0)
            self.assertEqual(transport.polls, {"a": 1, "b": 2})

    def test_abort_transport_failure_leaves_cleanup_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(states={"a": ["failed"]}, fail=("b", "abort"))
            self.assertEqual(self.execute(entry, review, transport), 1)
            terminal = coordinator.read(output / "terminal.json")
            self.assertEqual(terminal["abort_errors"], {"b": "TimeoutExpired"})
            self.assertFalse(terminal["cleanup_verified"])

    def test_stop_drains_without_renewing_and_exports_final_post_abort_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(states={"a": ["failed"], "b": ["running"]},
                                         drained={"a": ["stopped"], "b": ["running", "stopped"]})
            self.assertEqual(self.execute(entry, review, transport), 1)
            terminal = coordinator.read(output / "terminal.json")
            self.assertTrue(terminal["cleanup_verified"])
            self.assertEqual(transport.observations, {"a": 1, "b": 2})
            self.assertEqual(coordinator.read(output / "b" / "candidate-p1.json")["completed_games"], 2)
            for name in ("a", "b"):
                abort_index = transport.calls.index((name, "abort"))
                self.assertNotIn((name, "refresh"), transport.calls[abort_index + 1:])

    def test_unreachable_final_observation_is_bounded_and_does_not_claim_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, output, _doc = self.controller_fixture(directory)
            transport = FixtureTransport(states={"a": ["failed"]}, fail=("b", "observe"),
                                         drained={"a": ["stopped"]})
            self.assertEqual(self.execute(entry, review, transport), 1)
            terminal = coordinator.read(output / "terminal.json")
            self.assertEqual(transport.calls.count(("b", "observe")), 4)
            self.assertEqual(terminal["cleanup_unverified_partitions"], ["b"])
            self.assertFalse(terminal["cleanup_verified"])

    def agent_fixture(self, directory):
        root = Path(directory)
        entry, review = root / "worker.json", root / "review.json"
        doc = dict(run_directory=str(root / "run"), seed_start=10, seed_count=2,
                   calendar_seconds=86400, shards=[dict(name="candidate-p1", seat="p1")],
                   input_hashes={str(Path(source).resolve()): panel.digest(source)
                                 for source in (coordinator.__file__, panel.__file__)},
                   coordination=dict(lease_path=str(root / "lease.json"),
                                     abort_path=str(root / "abort.json"), lease_timeout_seconds=60))
        panel.write(entry, doc)
        panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
        self.enterContext(patch.dict(coordinator.os.environ,
                          POKEZERO_WORKER_POD_UID="uid-a", POKEZERO_WORKER_NODE_NAME="node-a"))
        return entry, review, doc

    def agent(self, *args):
        return coordinator.agent(*args, pod_uid="uid-a", node_name="node-a")

    def test_agent_refresh_writes_receiver_lease_bound_to_entry_and_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, doc = self.agent_fixture(directory)
            result = self.agent("refresh", entry, review, panel.digest(entry), 9999, "0-13", {})
            self.assertFalse(result["launcher_alive"])
            lease = coordinator.read(doc["coordination"]["lease_path"])
            self.assertEqual(lease, dict(entry_sha256=panel.digest(entry), deadline_unix=9999))
            self.agent("abort", entry, review, panel.digest(entry), 9999, "0-13", {})
            with self.assertRaisesRegex(ValueError, "COORDINATOR_ABORT"):
                self.agent("refresh", entry, review, panel.digest(entry), 9999, "0-13", {})

    def test_agent_start_exclusive_marker_prevents_duplicate_even_after_transport_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, _doc = self.agent_fixture(directory)
            entry_hash = panel.digest(entry)
            self.agent("refresh", entry, review, entry_hash, 9999999999, "0-13", {})
            worker = Mock(pid=123)
            with patch.object(coordinator.subprocess, "Popen", return_value=worker) as spawn, \
                    patch.object(coordinator, "process_identity", return_value=dict(pid=123, start_ticks=12)):
                result = self.agent("start", entry, review, entry_hash, 9999999999, "0-13", {})
                self.assertEqual(result["identity"]["start_ticks"], 12)
                with self.assertRaises(FileExistsError):
                    self.agent("start", entry, review, entry_hash, 9999999999, "0-13", {})
            spawn.assert_called_once()

    def test_agent_wait_records_actual_supervisor_nonzero_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, _doc = self.agent_fixture(directory)
            with patch.object(coordinator.subprocess, "run", return_value=Mock(returncode=1)):
                result = self.agent("wait", entry, review, panel.digest(entry), 9999, "0-13", {})
            self.assertEqual(result, dict(exit_code=1))
            self.assertEqual(coordinator.read(Path(directory) / "coordinator-exit.json")["exit_code"], 1)

    def test_snapshot_exports_new_prefix_and_refuses_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            _entry, _review, doc = self.agent_fixture(directory)
            output = Path(doc["run_directory"])
            output.mkdir()
            panel.write(output / "candidate-p1.json", dict(completed_games=1, game_results=[dict(seed=10)]))
            state = coordinator.snapshot(doc, Path(directory), {})
            self.assertEqual(state["counts"], {"candidate-p1": 1})
            self.assertIn("candidate-p1.json", state["documents"])
            unchanged = coordinator.snapshot(doc, Path(directory), {"candidate-p1": 1})
            self.assertNotIn("candidate-p1.json", unchanged["documents"])
            with self.assertRaisesRegex(ValueError, "shrank"):
                coordinator.snapshot(doc, Path(directory), {"candidate-p1": 2})

    def test_pid_reuse_or_zombie_is_not_a_live_launcher(self):
        identity = dict(pid=123, start_ticks=12)
        for current in (None, dict(pid=123, start_ticks=13, state="R"),
                        dict(pid=123, start_ticks=12, state="Z")):
            with self.subTest(current=current), patch.object(coordinator, "process_identity", return_value=current):
                self.assertFalse(coordinator.alive(identity))

    def test_transport_rejects_pod_replacement_or_node_change(self):
        target = dict(name="a", context="fixture", namespace="scott", pod="owned", pod_uid="uid", node_name="node")
        pod = dict(metadata=dict(uid="other"), spec=dict(nodeName="node"), status=dict(phase="Running"))
        with patch.object(coordinator.subprocess, "run", return_value=Mock(stdout=json.dumps(pod))):
            with self.assertRaisesRegex(ValueError, "pod identity"):
                coordinator.Transport().preflight_pod(target)

    def test_every_exec_rechecks_identity_before_start_or_abort(self):
        target = dict(name="a", context="fixture", namespace="scott", pod="owned", pod_uid="uid", node_name="node")
        pod = dict(metadata=dict(uid="replacement"), spec=dict(nodeName="node"), status=dict(phase="Running"))
        for action in ("start", "refresh", "observe", "abort"):
            with self.subTest(action=action), \
                    patch.object(coordinator.subprocess, "run", return_value=Mock(stdout=json.dumps(pod))) as execute:
                with self.assertRaisesRegex(ValueError, "pod identity"):
                    coordinator.Transport().call(target, action, 1000)
                execute.assert_called_once()
                self.assertIn("get", execute.call_args.args[0])
                self.assertNotIn("exec", execute.call_args.args[0])

    def test_replacement_after_get_pod_is_refused_by_worker_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, doc = self.agent_fixture(directory)
            with patch.dict(coordinator.os.environ, POKEZERO_WORKER_POD_UID="replacement"), \
                    patch.object(coordinator.subprocess, "Popen") as spawn:
                for action in ("refresh", "start", "abort", "observe"):
                    with self.subTest(action=action), self.assertRaisesRegex(ValueError, "downward API"):
                        self.agent(action, entry, review, panel.digest(entry), 9999999999, "0-13", {})
            spawn.assert_not_called()
            self.assertFalse(Path(doc["coordination"]["lease_path"]).exists())
            self.assertFalse(Path(doc["coordination"]["abort_path"]).exists())

    def test_agent_preflight_requires_both_bound_sources_and_unused_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, doc = self.agent_fixture(directory)
            with patch.object(panel, "verify"):
                facts = self.agent("preflight", entry, review, panel.digest(entry), None, "0-13", {})
                self.assertEqual(facts["seed_count"], 2)
                doc["input_hashes"] = {}
                panel.write(entry, doc)
                panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
                with self.assertRaisesRegex(ValueError, "source is not bound"):
                    self.agent("preflight", entry, review, panel.digest(entry), None, "0-13", {})

    def head_to_head_entry(self):
        return dict(comparison_kind=coordinator.HEAD_TO_HEAD_KIND,
                    common_arguments=["--opponent-policy-mode", "raw"],
                    shards=[dict(name=name, seat=seat,
                                 arguments=["--policy-mode", "trajectory-reference"])
                            for name, seat in coordinator.HEAD_TO_HEAD_SHARDS])

    def test_worker_preflight_derives_modes_from_actual_common_and_seat_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, review, doc = self.agent_fixture(directory)
            doc.update(self.head_to_head_entry())
            panel.write(entry, doc)
            panel.write(review, dict(status="CLEAR", entry_sha256=panel.digest(entry)))
            with patch.object(panel, "verify"):
                facts = self.agent("preflight", entry, review, panel.digest(entry), None, "0-13", {})
            self.assertEqual(facts["comparison_kind"], coordinator.HEAD_TO_HEAD_KIND)
            self.assertEqual(facts["candidate_policy_mode"], "trajectory-reference")
            self.assertEqual(facts["opponent_policy_mode"], "raw")
            self.assertEqual(facts["shards"], coordinator.HEAD_TO_HEAD_SHARDS)
            self.assertFalse(Path(doc["run_directory"]).exists())

    def test_head_to_head_actual_arguments_refuse_overrides_aliases_and_false_labels(self):
        for defect in ("foulplay", "raw-candidate", "different-seats", "four-shards", "duplicate",
                       "duplicate-common", "abbreviation", "equals", "alias-override", "missing",
                       "missing-value", "not-list", "non-string", "unknown-kind"):
            doc = self.head_to_head_entry()
            args = doc["shards"][0]["arguments"]
            if defect == "foulplay":
                doc["common_arguments"][1] = "foulplay"
            elif defect == "raw-candidate":
                args[1] = "raw"
            elif defect == "different-seats":
                args[1] = "engine-mcts"
            elif defect == "four-shards":
                doc["shards"].append(dict(name="raw-p1", seat="p1", arguments=[]))
            elif defect == "duplicate":
                args.extend(["--policy-mode", "raw"])
            elif defect == "duplicate-common":
                doc["common_arguments"].extend(["--policy-mode", "raw"])
            elif defect == "abbreviation":
                args[0] = "--policy-mod"
            elif defect == "equals":
                args[:] = ["--policy-mode=trajectory-reference"]
            elif defect == "alias-override":
                args.extend(["--opponent-policy-mod=sampled"])
            elif defect == "missing":
                args[:] = []
            elif defect == "missing-value":
                args[:] = ["--policy-mode"]
            elif defect == "not-list":
                doc["common_arguments"] = "--opponent-policy-mode raw"
            elif defect == "non-string":
                args.append(None)
            else:
                doc["comparison_kind"] = "unregistered"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                coordinator.head_to_head_contract(doc)

    def test_permission_denial_is_reported_without_repeated_requests_or_bypass(self):
        target = dict(name="a", context="fixture", namespace="scott", pod="owned")
        denied = subprocess.CalledProcessError(1, "kubectl", stderr="Forbidden: cannot get pods in namespace scott")
        transport = coordinator.Transport()
        with patch.object(coordinator.subprocess, "run", side_effect=denied) as execute:
            for action in ("start", "abort", "observe"):
                with self.assertRaisesRegex(PermissionError, "cannot get pods"):
                    transport.call(target, action, 1000)
        execute.assert_called_once()

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("taskset"),
                         "authored Linux process lifecycle fixture")
    def test_actual_linux_launcher_exit_and_abort_export_without_scientific_code(self):
        for abort in (False, True):
            with self.subTest(abort=abort), tempfile.TemporaryDirectory() as directory:
                entry, review, doc = self.agent_fixture(directory)
                root = Path(directory)
                package = root / "pokezero"
                package.mkdir()
                (package / "__init__.py").write_text("")
                fake = package / "foulplay_bridge.py"
                fake.write_text(
                    "import argparse,json,time\nfrom pathlib import Path\n"
                    "p=argparse.ArgumentParser()\n"
                    "p.add_argument('--summary-out');p.add_argument('--games',type=int)\n"
                    "p.add_argument('--seed-start',type=int);p.add_argument('--fixture-sleep',type=int)\n"
                    "a,_=p.parse_known_args()\n"
                    "time.sleep(a.fixture_sleep)\n"
                    "Path(a.summary_out).write_text(json.dumps({'completed_games':a.games,"
                    "'game_results':[{'seed':a.seed_start+i} for i in range(a.games)]}))\n")
                doc.update(python=sys.executable, runtime=str(root),
                           environment={"PYTHONPATH": str(root)}, repositories={}, tree_hashes={},
                           no_completion_seconds=1800, foulplay_seed_start=10,
                           common_arguments=["--fixture-sleep", "120" if abort else "0"])
                doc["shards"][0]["arguments"] = []
                doc["input_hashes"][str(fake)] = panel.digest(fake)
                panel.write(entry, doc)
                entry_hash = panel.digest(entry)
                panel.write(review, dict(status="CLEAR", entry_sha256=entry_hash))
                deadline = time.time() + 60
                cpus = str(min(os.sched_getaffinity(0)))
                self.agent("refresh", entry, review, entry_hash, deadline, cpus, {})
                # Agent start deliberately returns while its launcher is alive;
                # this fixture then observes the real exit and reaps that child.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="subprocess .* is still running",
                                            category=ResourceWarning)
                    launch = self.agent("start", entry, review, entry_hash, deadline, cpus, {})
                try:
                    if abort:
                        limit = time.monotonic() + 10
                        while not (Path(doc["run_directory"]) / "launch.json").exists():
                            self.assertLess(time.monotonic(), limit, "fixture supervisor did not launch")
                            time.sleep(0.05)
                        self.agent("abort", entry, review, entry_hash, deadline, cpus, {})
                    limit = time.monotonic() + 20
                    while True:
                        state = self.agent("observe", entry, review, entry_hash, deadline, cpus, {})
                        if state["supervisor_exit"] is not None and not state["launcher_alive"]:
                            break
                        self.assertLess(time.monotonic(), limit, "fixture did not reach actual terminal")
                        time.sleep(0.05)
                    self.assertEqual(state["supervisor_exit"]["exit_code"], 1 if abort else 0)
                    self.assertTrue(coordinator.cleanup_confirmed(state))
                    terminal = state["documents"]["terminal.json"]
                    self.assertEqual(terminal["status"], "COORDINATOR_ABORT" if abort
                                     else "COLLECTION_ATTEMPT_COMPLETE")
                    try:
                        os.waitpid(launch["identity"]["pid"], 0)
                    except ChildProcessError:
                        pass  # subprocess's own child bookkeeping already reaped it.
                    if not abort:
                        self.assertEqual(state["counts"], {"candidate-p1": 2})
                finally:
                    # Only this authored fixture's birth-bound process group.
                    if coordinator.alive(launch["identity"]):
                        self.agent("abort", entry, review, entry_hash, deadline, cpus, {})
                        os.killpg(launch["identity"]["pid"], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
