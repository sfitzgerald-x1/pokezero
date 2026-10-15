"""Actual isolated model ABI and readers; random weights, no source games.

Run explicitly with the freshly built model wheel first on PYTHONPATH. These
checks fail rather than silently substituting an older installed extension.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from test_model_priors_search import _EncodedSearchFixture, pokezero_search
from test_rollout_model_priors import TIMING_FIELDS
from pokezero.mcts_eval.search_over_raw_native_atomic_work import state_hash, validate_native_report
from pokezero.mcts_eval.search_over_raw_native_atomic_compact import (
    SCHEMA, validate_compact_native_report, validate_timed_compact_native_report,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import NativeInputHandle, NativeWorldInputs
import test_search_over_raw_world_inputs as inputs


class NativeCompactRuntimeTests(_EncodedSearchFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False):
            raise AssertionError("fresh isolated model runtime required")
        signature = pokezero_search.NativeLeafModel.search_batched_multi_encoded.__text_signature__
        if not signature.endswith("record_compact_atomic_work=False)"):
            raise AssertionError("fresh compact ABI required")
        super().setUpClass()

    def arguments(self, position, iterations=32):
        return [position["state_str"], iterations, 16, self.tables_json,
            position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]), 3, 1.4, 17, True, True]

    def test_actual_default_legacy_and_compact_preserve_fixed_work_and_paths(self):
        for side, position in self.perspective_positions.items():
            with self.subTest(side=side):
                options = dict(record_joint_actions=True)
                ordinary = json.loads(self.native.search_batched_multi_encoded(*self.arguments(position), **options))
                full = json.loads(self.native.search_batched_multi_encoded(*self.arguments(position),
                    **options, record_atomic_work=True))
                compact = json.loads(self.native.search_batched_multi_encoded(*self.arguments(position),
                    **options, record_atomic_work=True, record_compact_atomic_work=True))
                self.assertNotIn("atomic_work", ordinary)
                self.assertNotEqual(full["atomic_work"]["schema"], SCHEMA)
                self.assertEqual(compact["atomic_work"]["schema"], SCHEMA)
                normalize = lambda value: {k:v for k,v in value.items()
                    if k not in TIMING_FIELDS and k != "atomic_work"}
                self.assertEqual(normalize(ordinary), normalize(full))
                self.assertEqual(normalize(ordinary), normalize(compact))
                self.assertEqual(full["atomic_work"]["final"], compact["atomic_work"]["final"])
                self.assertEqual([[c["path"] for c in r["commits"]] for r in full["atomic_work"]["rounds"]],
                    [[c["path"] for c in r["commits"]] for r in compact["atomic_work"]["rounds"]])
                binding = dict(state_sha256=state_hash(position["state_str"]),
                    requested_iterations=32, batch_size=16, chance_seed=17)
                self.assertEqual(validate_native_report(full, **binding)["completed_iterations"], 32)
                self.assertEqual(validate_compact_native_report(compact, **binding)["completed_iterations"], 32)
                with self.assertRaises(ValueError): validate_native_report(compact, **binding)
                with self.assertRaises(ValueError): validate_compact_native_report(full, **binding)

    def test_actual_compact_without_atomic_refuses_before_search(self):
        with self.assertRaisesRegex(ValueError, "compact.*atomic|atomic.*compact"):
            self.native.search_batched_multi_encoded(*self.arguments(self.position),
                record_compact_atomic_work=True)

    def test_actual_positive_and_zero_model_and_hp_through_input_observer(self):
        position = self.position
        ctx = json.loads(position["ctx"])
        world = inputs.FakeWorld(dict(authored=True),
            {s:tuple(ctx[s]) for s in ("p1", "p2")}, dict(p1="side_one", p2="side_two"))
        for mode in (None, "hp_fraction"):
            for budget, productive in ((1, False), (1000, True)):
                with self.subTest(mode=mode, budget=budget):
                    recorder = NativeWorldInputs(inputs.BINDING, 32, deadline_limited=True,
                        record_atomic_work=True, record_timed_atomic_work=True,
                        record_compact_atomic_work=True)
                    context = SimpleNamespace(player_id="p1")
                    recorder.attempt(context, 0, inputs.FakeOverride(dict(p1="authored", p2="authored")))
                    recorder.constructed(context, world,
                        SimpleNamespace(to_string=lambda:position["state_str"]))
                    args = self.arguments(position) + [0, True, False, None, False,
                        mode, 32, 200, "uniform", 0, 1, False, budget]
                    raw = NativeInputHandle(self.native, recorder).search_batched_multi_encoded(*args)
                    report = json.loads(raw)
                    self.assertEqual(report["atomic_work"]["schema"], SCHEMA)
                    checked = validate_timed_compact_native_report(report,
                        state_sha256=state_hash(position["state_str"]), requested_iterations=32,
                        batch_size=16, chance_seed=17, time_budget_ms=budget)
                    work = checked["completed_iterations"]
                    self.assertGreater(work, 0) if productive else self.assertEqual(work, 0)
                    manifest = recorder.finish(dict(total_iterations=work), attempted=1,
                        constructed=1, searched=int(work > 0))
                    self.assertEqual(manifest["calls"][0]["native_report_utf8"], raw)
                    self.assertEqual(manifest["calls"][0]["native_call_parameters"]["time_budget_ms"], budget)
                    self.assertTrue(manifest["timed_atomic_observation"]["complete_counter_reconciliation"])
                    self.assertFalse(manifest["runtime_qualified"])
                    changed = deepcopy(report)
                    changed["atomic_work"]["final"]["decisions"][0]["visits"] += 1
                    with self.assertRaises(ValueError):
                        validate_timed_compact_native_report(changed,
                            state_sha256=state_hash(position["state_str"]), requested_iterations=32,
                            batch_size=16, chance_seed=17, time_budget_ms=budget)


class NativeCompactRustFixtureTests(unittest.TestCase):
    def test_actual_native_deferred_commits_and_restored_cancellation_read_independently(self):
        repo = Path(__file__).resolve().parents[1]
        completed = subprocess.run(["cargo", "test", "--locked", "--offline", "--release",
            "--features", "model", "--lib", "atomic_work::tests::compact", "--", "--nocapture"],
            cwd=repo/"rust/pokezero-search", capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        self.assertIn("3 passed; 0 failed", completed.stdout)
        state = (repo/"rust/pokezero-search/src/test_fixtures/minimal.state").read_text().strip()
        binding = dict(state_sha256=state_hash(state), requested_iterations=16, batch_size=4, chance_seed=17)
        for marker, expected in (("ATOMIC_COMPACT_READER_FIXTURE:", 16),
                ("ATOMIC_COMPACT_CANCELLED_FIXTURE:", 0)):
            rows = [line.split(marker, 1)[1] for line in completed.stdout.splitlines() if marker in line]
            self.assertEqual(len(rows), 1)
            report = json.loads(rows[0])
            self.assertEqual(validate_compact_native_report(report, **binding)["completed_iterations"], expected)
            if expected:
                self.assertGreater(len(report["atomic_work"]["final"]["decisions"]), 1)
            else:
                cancelled, = report["atomic_work"]["rounds"]
                self.assertTrue(cancelled["attempted_paths"])
                self.assertEqual(cancelled["before_sha256"], cancelled["after_sha256"])


class NativeCompactActorOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_native_fixed_work_adapter import LoadedNativeValueAdapterTests
        cls.fixture = LoadedNativeValueAdapterTests
        cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_actual_model_and_hp_actor_owners_read_back_with_bounded_complete_terminals(self):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
        from pokezero.mcts_eval.search_over_raw_actor_timed_run import (
            ActorAnnotatedTimedJob, ActorAnnotatedTimedRunValidator,
            actor_timed_runtime_recipe, supervise_actor_timed,
        )
        from pokezero.mcts_eval.search_over_raw_annotation_boundary import ActorAnnotationBoundary
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        import pokezero_search.pokezero_search as native_search
        import poke_engine.poke_engine as native_engine
        helper = self.fixture(); self.addCleanup(helper.doCleanups)
        files = fixed_work_reader_source_hashes()
        paths = [Path(sys.executable).resolve(), helper.checkpoint,
            Path(helper.checkpoint_contract.model_path), Path(helper.checkpoint_contract.tables_path),
            Path(native_search.__file__), Path(native_engine.__file__)]
        paths.extend(p for p in (Path(helper.showdown_root)/"dist").rglob("*") if p.is_file())
        for path in paths: files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for leaf in ("model", "hp_fraction"):
            with self.subTest(leaf=leaf), tempfile.TemporaryDirectory() as temporary:
                env, context = helper.authored_context()
                cfg = SearchConfiguration("incumbent", leaf=leaf, seconds=1.)
                public_sha = public_request_witness(PublicRootRequest.capture(
                    context.public_materialization_state, context.observation))["public_request_sha256"]
                job = ActorAnnotatedTimedJob(cfg, helper.checkpoint_contract, helper.showdown_root, context,
                    "authored-compact-actor:"+cfg.identity, 7, "authored-compact-actor",
                    actor_timed_runtime_recipe(cfg, helper.checkpoint_contract, compact_native_atomic_work=True),
                    annotation_boundary=ActorAnnotationBoundary.capture(env, context),
                    expected_public_request_sha256=public_sha, synthetic_process_fixture=True)
                attempt = Path(temporary)/"owned-compact-actor"
                terminal = supervise_actor_timed(job, directory=attempt, binding_sha256=job._frozen_inputs,
                    verify_admission=lambda:None)
                self.assertEqual(terminal["supervisor_exit_code"], 0, terminal.get("error_type"))
                self.assertEqual(terminal["worker_exit_code"], 0)
                terminal_bytes = (attempt/"terminal.json").stat().st_size
                self.assertLessEqual(terminal_bytes, 16_777_216)
                result = terminal["result"]
                bindings = dict(directory=attempt, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                    binding_sha256=job._frozen_inputs, configuration=cfg, root_id=job.root_id,
                    statistics_root_id=job.statistics_root_id, selection_seed=job.selection_seed,
                    runtime_configuration=job.expected_runtime_configuration, runtime_sha256=job.expected_runtime_sha256,
                    consumed_binding=job._consumed_binding, legal_action_mask=job._legal_action_mask,
                    decision_round=context.decision_round_index, file_hashes=files,
                    checkpoint_path=helper.checkpoint_contract.checkpoint_path,
                    checkpoint_sha256=helper.checkpoint_contract.checkpoint_sha256, showdown_root=helper.showdown_root,
                    public_request_sha256=public_sha, annotation_sha256=job.annotation_boundary._original_sha)
                self.assertIsNone(ActorAnnotatedTimedRunValidator(**bindings)(result))
                reader = ActorAnnotatedTimedRunValidator(**bindings, allow_synthetic=True)
                self.assertEqual(reader(result), digest(result))
                self.assertGreater(reader.last_validation["actual_work"]["accepted_units"], 0)
                self.assertTrue(reader.last_validation["process_receipts_validated"])
                self.assertFalse(reader.last_validation["runtime_qualified"])
                self.assertIsNone(env._process)
                print("COMPACT_ACTOR_MEASUREMENT:"+json.dumps(dict(leaf=leaf, terminal_bytes=terminal_bytes,
                    accepted_units=reader.last_validation["actual_work"]["accepted_units"]), sort_keys=True))
