"""Authored instrument fixtures only: no champion, source panel or strength run."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("coverage_probe", SCRIPTS / "reference_materialization_coverage.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def entry():
    return dict(seed_start=100, seed_count=8, turns=[4, 12], calendar_seconds=3600,
                source_turn_cap=20, source_boundary_cap=256, sampling_seconds=10,
                binding=dict(joint_sleep_proposal_cap=32, allow_earlier_compatible_template=True))


def evidence(*statuses):
    return dict(draws=[dict(joint_sleep_proposals=[dict(status=s) for s in statuses],
                           error="PRIVATE_CANARY")], private="PRIVATE_CANARY")


class CoverageTests(unittest.TestCase):
    def test_geometry_is_exact_with_no_replacement_slots(self):
        rows = probe.slots(entry())
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({(r["seed"], r["turn"], r["seat"]) for r in rows}), 32)
        self.assertEqual([r["slot"] for r in rows], list(range(32)))
        self.assertEqual({r["status"] for r in rows}, {"UNAVAILABLE"})
        for field, value in (("seed_start", True), ("seed_count", 9), ("seed_count", True),
                             ("turns", [4, 13]), ("calendar_seconds", 3601),
                             ("sampling_seconds", 11), ("source_turn_cap", 21),
                             ("source_boundary_cap", 257)):
            bad = entry()
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.slots(bad)
        for key, value in (("joint_sleep_proposal_cap", 33),
                           ("joint_sleep_proposal_cap", True),
                           ("allow_earlier_compatible_template", 1)):
            bad = entry()
            bad["binding"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.slots(bad)

    def test_draw_seeds_are_stable_and_separate_for_every_fixed_slot(self):
        rows = probe.slots(entry())
        seeds = [probe.root_seed(r) for r in rows]
        self.assertEqual(seeds, [probe.root_seed(copy.deepcopy(r)) for r in rows])
        self.assertEqual(len(set(seeds)), 32)

    def test_liveness_uses_birth_not_scheduler_state(self):
        identity = dict(pid=17, start_ticks=123, state="R")
        self.assertTrue(probe.same_birth(identity, {**identity, "state": "S"}))
        for actual in (None, {**identity, "start_ticks": 124}, {**identity, "pid": 18},
                       {**identity, "state": "Z"}, {**identity, "state": "X"}):
            self.assertFalse(probe.same_birth(identity, actual))

    def test_child_requires_original_parent_birth_and_exact_finite_deadline(self):
        child, owner = dict(pid=1, start_ticks=7, state="R"), dict(pid=2, start_ticks=8, state="S")
        args = SimpleNamespace(deadline=1234., sha256="bound")
        launch = dict(entry_sha256="bound", deadline_unix=1234., identity=child, owner_identity=owner)
        with patch.object(probe, "process_identity", side_effect=[child, owner]):
            probe.validate_child_launch(args, launch)
        for deadline in (None, True, float("nan"), float("inf"), 1235.):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                probe.validate_child_launch(SimpleNamespace(deadline=deadline, sha256="bound"), launch)
        for actual in (({**child, "start_ticks": 9}, owner), (child, {**owner, "start_ticks": 9})):
            with patch.object(probe, "process_identity", side_effect=actual), self.assertRaises(ValueError):
                probe.validate_child_launch(args, launch)

    def test_proposal_costs_are_exact_and_do_not_export_raw_messages(self):
        counts = probe.sampling_counts(evidence("ACCEPTED", "REJECTED", "STARTED"))
        self.assertEqual(counts, dict(draws_started=1, proposals_started=3, proposals_accepted=1,
                                     proposals_rejected=1, proposals_incomplete=1))
        self.assertNotIn("PRIVATE_CANARY", json.dumps(counts))
        self.assertEqual(probe.sampling_counts({"draws": []})["draws_started"], 0)
        for bad in (None, {}, {"draws": None}, {"draws": [{}]}, evidence("UNKNOWN")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.sampling_counts(bad)

    def test_expected_refusals_are_distinct_and_unknown_bridge_errors_are_fatal(self):
        from pokezero.local_showdown import LocalShowdownError
        from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
        counts = probe.sampling_counts(evidence(*(["REJECTED"] * 32)))
        # Text without the actual trusted originating code is not a diagnosis.
        for error in (ReferenceRefusal("joint team/sleep proposal cap exhausted"),
                      SamplingDeadlineExceeded("PRIVATE_CANARY"),
                      ReferenceRefusal("joint sleep evidence unresolved: fixture")):
            with self.assertRaises(type(error)):
                probe.failure_status(error, counts)
        typed = LocalShowdownError("PRIVATE_CANARY", bridge_error_reason={
            "code": "UNSUPPORTED_PUBLIC_VOLATILE", "volatile": "confusion"})
        self.assertEqual(probe.failure_status(typed, counts), "TYPED_CONSTRUCTION_REFUSAL")
        for error in (ValueError("fixture bug"), LocalShowdownError("untyped bridge error")):
            with self.assertRaises(type(error)):
                probe.failure_status(error, counts)

    def runtime_fixture(self, *, error=None, root_mismatch=False):
        root = SimpleNamespace(key=b"root")
        world = Mock()
        world.subject = "p1"
        world.env.observe.return_value = object() if root_mismatch else root
        prepared = Mock(root=root)
        prepared.sample_world.side_effect = error
        prepared.sample_world.return_value = world
        prepared.evidence.return_value = evidence("ACCEPTED")
        runtime = Mock()
        runtime.prepare.return_value = prepared
        factory = Mock(return_value=runtime)
        return factory, runtime, prepared, world

    def test_real_cap_and_real_expiry_are_classified_without_forgiving_ownership_failure(self):
        from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
        from pokezero.mcts_eval.paper_reference import ReferenceRefusal
        from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
        import random
        factory = object.__new__(PublicRootWorldFactory)
        factory.active, factory.receipts = False, []
        factory.joint_sleep_proposal_cap, factory.deadline_at = 32, None
        factory.state = SimpleNamespace(player_id="p1")
        factory.known, factory.own_team = (), ()
        factory.sleep_ledger = public_sleep_history(("|start",), at_action_boundary=True)
        draw = SimpleNamespace(packed_team_sha256="fixture", known=(), unknown_party_seeds=(), team=())
        factory.sampler = SimpleNamespace(draw=Mock(return_value=draw))
        with patch("pokezero.mcts_eval.paper_reference_factory.joint_sleep_proposal", return_value=(None, {})):
            try:
                factory(random.Random(1))
            except ReferenceRefusal as error:
                counts = probe.sampling_counts(dict(draws=factory.receipts))
                self.assertEqual(probe.failure_status(error, counts), "PROPOSAL_CAP_EXHAUSTED")
            else:
                self.fail("cap did not refuse")
        factory.deadline_at = -1.
        try:
            factory._check_deadline()
        except ReferenceRefusal as error:
            self.assertEqual(probe.failure_status(error, counts, deadline_at=-1.), "DEADLINE_INTERRUPTION")
            with self.assertRaises(ReferenceRefusal):
                probe.failure_status(error, counts, deadline_at=float("inf"))
        factory.active = True
        try:
            factory(random.Random(1))
        except ReferenceRefusal as error:
            with self.assertRaises(ReferenceRefusal):
                probe.failure_status(error, counts)

    def test_real_prepare_knowledge_gap_has_certified_zero_draws(self):
        from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
        factory, runtime, prepared, world = self.runtime_fixture()
        def prepare(_request):
            return public_sleep_history(("|start", "|-start|p1a: Articuno|move: Yawn"), at_action_boundary=True)
        runtime.prepare.side_effect = prepare
        result = probe.probe_root(probe.slots(entry())[0], "fixture", {}, runtime_factory=factory)
        self.assertEqual(result["status"], "UNRESOLVED_EVIDENCE")
        self.assertEqual(result["counts"]["draws_started"], 0)
        prepared.sample_world.assert_not_called()
        world.frame.assert_not_called()
        runtime.close.assert_called_once()

    def test_real_joint_unresolved_evidence_is_distinct_from_other_refusals(self):
        from pokezero.mcts_eval.reference_sleep_joint import joint_sleep_proposal
        from pokezero.mcts_eval.paper_reference import ReferenceRefusal
        import random
        with patch("pokezero.mcts_eval.reference_sleep_joint.qualify_sleep_traits",
                   return_value=SimpleNamespace(status="unresolved", reason="fixture")):
            try:
                joint_sleep_proposal(None, {}, random.Random(1), check_deadline=lambda: None)
            except ReferenceRefusal as error:
                self.assertEqual(probe.failure_status(error, probe.sampling_counts({"draws": []})),
                                 "UNRESOLVED_EVIDENCE")
            else:
                self.fail("unresolved fixture did not refuse")

    def test_one_world_only_without_evaluation_and_both_owners_close(self):
        factory, runtime, prepared, world = self.runtime_fixture()
        row = {**probe.slots(entry())[0], "root_sha256": b"root".hex()}
        with patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", side_effect=lambda obs, player: obs):
            result = probe.probe_root(row, "saved-root", {}, runtime_factory=factory)
        self.assertEqual(result["status"], "ROOT_PRESERVED")
        factory.assert_called_once_with(row["slot"])
        runtime.prepare.assert_called_once_with("saved-root")
        prepared.sample_world.assert_called_once()
        prepared.bind_sampling_deadline.assert_called_once()
        prepared.evaluate_root.assert_not_called()
        world.frame.assert_not_called()
        world.evaluator.assert_not_called()
        world.close.assert_called_once()
        runtime.close.assert_called_once()
        self.assertGreaterEqual(result["elapsed_wall_seconds"], result["startup_wall_seconds"])

    def test_typed_refusal_preserves_public_token_and_never_retries(self):
        from pokezero.local_showdown import LocalShowdownError
        error = LocalShowdownError("PRIVATE_CANARY", bridge_error_reason={
            "code": "UNSUPPORTED_PUBLIC_VOLATILE", "volatile": "confusion"})
        factory, runtime, prepared, world = self.runtime_fixture(error=error)
        result = probe.probe_root(probe.slots(entry())[0], "root", {}, runtime_factory=factory)
        self.assertEqual(result["status"], "TYPED_CONSTRUCTION_REFUSAL")
        self.assertIn("confusion", json.dumps(result["error_code_sites"]))
        self.assertNotIn("PRIVATE_CANARY", json.dumps(result))
        prepared.sample_world.assert_called_once()
        world.close.assert_not_called()
        runtime.close.assert_called_once()

    def test_changed_root_is_fatal_and_closes_world_and_runtime(self):
        factory, runtime, prepared, world = self.runtime_fixture(root_mismatch=True)
        with patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", side_effect=lambda obs, player: obs), \
                self.assertRaisesRegex(ValueError, "differs"):
            probe.probe_root(probe.slots(entry())[0], "root", {}, runtime_factory=factory)
        prepared.sample_world.assert_called_once()
        world.close.assert_called_once()
        runtime.close.assert_called_once()

    def test_preparation_failure_stops_and_does_not_draw(self):
        factory, runtime, prepared, _ = self.runtime_fixture()
        runtime.prepare.side_effect = ValueError("malformed transport")
        with self.assertRaises(ValueError):
            probe.probe_root(probe.slots(entry())[0], "root", {}, runtime_factory=factory)
        prepared.sample_world.assert_not_called()
        runtime.close.assert_called_once()

    def test_source_capture_has_no_constructor_callback_and_keeps_unavailable_slots(self):
        class Env:
            def reset(self, *, seed):
                self.turn, self.seed = 1, seed

            def terminal(self):
                return "terminal" if self.turn > (6 if self.seed == 100 else 20) else None

            def requested_players(self):
                return ("p1", "p2")

            def public_materialization_state(self, player):
                return SimpleNamespace(replay=SimpleNamespace(turn_number=self.turn), self_request={"active": [{}]})

            def observe(self, player):
                return f"actor-{player}-turn-{self.turn}"

            def step(self, actions):
                self.turn += 1

        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        rows, policy = probe.slots(entry()), Mock()
        policy.select_action.return_value = SimpleNamespace(action_index=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(PublicRootRequest, "capture", side_effect=lambda state, observation: SimpleNamespace(observation=observation)), \
                patch("pokezero.mcts_eval.paper_reference_showdown.decision_state",
                      return_value=SimpleNamespace(key=b"root")):
            sources = probe.capture_sources(Env(), policy, rows, Path(directory), deadline=float("inf"))
            self.assertEqual(len(sources), 8)
            self.assertEqual(len(rows), 32)
            self.assertEqual(sum(r["status"] == "CAPTURED" for r in rows), 30)
            missing = [r for r in rows if r["status"] == "UNAVAILABLE"]
            self.assertEqual([(r["seed"], r["turn"]) for r in missing], [(100, 12)] * 2)
            self.assertEqual({r["unavailable_reason"] for r in missing}, {"SOURCE_TERMINAL"})
            self.assertEqual(len(list(Path(directory).glob("root-*.pickle"))), 30)
            self.assertNotIn("winner", json.dumps(sources))
        self.assertEqual(policy.reset.call_count, 8)

    def test_owner_signal_drains_exact_child_and_records_nonzero_terminal(self):
        for sig in (probe.signal.SIGTERM, probe.signal.SIGINT):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                doc = entry()
                doc.update(run_directory=str(Path(directory) / "run"), runtime=directory, environment={"OMP_NUM_THREADS": "1"})
                args = SimpleNamespace(child=False, sha256="bound", entry="entry", review="review")
                worker = Mock(pid=999999)
                worker.poll.side_effect = lambda: probe.owner_signal(sig, None)
                with patch.object(probe, "checked_entry", return_value=doc), \
                        patch.object(probe.subprocess, "Popen", return_value=worker) as spawn, \
                        patch.object(probe, "process_identity", return_value=dict(pid=999999, start_ticks=1, state="S")), \
                        patch.object(probe.panel, "stop_owned", return_value=(-15, True)) as stop:
                    with self.assertRaises(InterruptedError):
                        probe.main(args)
                stop.assert_called_once_with(worker)
                self.assertEqual(spawn.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")
                terminal = json.loads((Path(doc["run_directory"]) / "terminal.json").read_text())
                self.assertEqual(terminal["actual_child_exit"], -15)
                self.assertTrue(terminal["cleanup_ok"])
                self.assertEqual(terminal["reason"], "OWNER_FAILURE")

    def test_zero_child_exit_after_calendar_cap_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = {**entry(), "run_directory": str(Path(directory) / "run"),
                   "runtime": directory, "environment": {}}
            args = SimpleNamespace(child=False, sha256="bound", entry="entry", review="review")
            worker = Mock(pid=999999)
            worker.poll.return_value = 0
            with patch.object(probe, "checked_entry", return_value=doc), \
                    patch.object(probe.time, "time", side_effect=[0, 3600, 3601]), \
                    patch.object(probe.subprocess, "Popen", return_value=worker), \
                    patch.object(probe, "process_identity", return_value=dict(pid=999999, start_ticks=1, state="S")), \
                    patch.object(probe.panel, "stop_owned", return_value=(0, True)):
                self.assertEqual(probe.main(args), 1)
            terminal = json.loads((Path(doc["run_directory"]) / "terminal.json").read_text())
            self.assertEqual(terminal["actual_child_exit"], 0)
            self.assertEqual(terminal["reason"], "CALENDAR_CAP")

    def test_signal_inside_spawn_is_deferred_until_handle_can_be_drained(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = {**entry(), "run_directory": str(Path(directory) / "run"),
                   "runtime": directory, "environment": {}}
            args = SimpleNamespace(child=False, sha256="bound", entry="entry", review="review")
            worker = Mock(pid=999999)
            def spawn(*_args, **_kwargs):
                probe.signal.getsignal(probe.signal.SIGTERM)(probe.signal.SIGTERM, None)
                return worker
            with patch.object(probe, "checked_entry", return_value=doc), \
                    patch.object(probe.subprocess, "Popen", side_effect=spawn), \
                    patch.object(probe.panel, "stop_owned", return_value=(-15, True)) as stop:
                with self.assertRaises(InterruptedError):
                    probe.main(args)
            stop.assert_called_once_with(worker)
            terminal = json.loads((Path(doc["run_directory"]) / "terminal.json").read_text())
            self.assertEqual(terminal["reason"], "OWNER_FAILURE")

    def test_existing_output_cannot_resume_or_launch_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = {**entry(), "run_directory": directory}
            with patch.object(probe, "checked_entry", return_value=doc), \
                    patch.object(probe.subprocess, "Popen") as spawn, self.assertRaises(FileExistsError):
                probe.main(SimpleNamespace(child=False))
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
