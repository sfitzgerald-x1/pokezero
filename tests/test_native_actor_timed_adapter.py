"""Actual random-model timed jobs on authored requests, never source games."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_model_priors_search import _crate_ready
import test_native_fixed_work_adapter as fixed_fixture
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_annotation_boundary import ActorAnnotationBoundary
from pokezero.mcts_eval.search_over_raw_actor_timed_run import (
    ActorAnnotatedTimedAdapter, ActorAnnotatedTimedJob, ActorAnnotatedTimedRunValidator,
    actor_timed_runtime_recipe, supervise_actor_timed, RUNTIME_KEY,
    prepare_archived_actor_timed_job,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest


@unittest.skipUnless(_crate_ready, "requires fresh isolated model runtime")
class LoadedActorTimedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixed_fixture.LoadedNativeValueAdapterTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixed_fixture.LoadedNativeValueAdapterTests.tearDownClass()

    def setUp(self):
        self.helper = fixed_fixture.LoadedNativeValueAdapterTests()
        self.addCleanup(self.helper.doCleanups)

    def job(self, leaf="model"):
        h = self.helper
        env, context = h.authored_context()
        request = PublicRootRequest.capture(context.public_materialization_state, context.observation)
        boundary = ActorAnnotationBoundary.capture(env, context)
        cfg = SearchConfiguration("incumbent", leaf=leaf, seconds=1.)
        job = ActorAnnotatedTimedJob(cfg, h.checkpoint_contract, h.showdown_root, context,
            "authored-actor-timed:"+cfg.identity, 7, "authored-actor-timed",
            actor_timed_runtime_recipe(cfg, h.checkpoint_contract), annotation_boundary=boundary,
            expected_public_request_sha256=public_request_witness(request)["public_request_sha256"],
            synthetic_process_fixture=True)
        return env, job

    def archive_fixture(self):
        from types import SimpleNamespace
        from pokezero.local_showdown import LocalShowdownSnapshot
        from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
        from pokezero.mcts_eval.search_over_raw_source import source_roots
        from pokezero.trajectory import BattleTrajectory, TrajectoryStep
        from pokezero.env import TerminalState
        import test_search_over_raw_reconciled as reconciled_fixture
        ledger = reconciled_fixture.ReconciledLedgerTest()
        ledger.setUp()
        self.addCleanup(ledger.tearDown)
        env, context = self.helper.authored_context()
        seed = ledger.contract["panels"]["exploration"]["seeds"][0]
        trajectory = BattleTrajectory(context.battle_id, context.format_id, seed)
        context = replace(context, seed=seed, trajectory=trajectory)
        archive = SealedSourceArchive()
        self.addCleanup(archive.close)
        archive.capture_public(context, 0)
        public = context.public_materialization_state
        snapshot = LocalShowdownSnapshot("authored-only-token", public.replay.battle_id,
            env._format_id, env._observation_format_id, {"private_canary":"never-policy-input"},
            tuple(env._lines), deepcopy(env._latest_requests), deepcopy(env._first_requests),
            deepcopy(env._request_history), env._parser.snapshot(), env._belief_engine.clone(),
            env._latest_turn, env._terminal, annotation_cache=env._annotation_cache())
        archive.capture_private(SimpleNamespace(requested_players=("p1",), decision_round_index=0,
            seed=seed, battle_id=context.battle_id, decisions={"p1":SimpleNamespace(action_index=0)}, snapshot=snapshot))
        trajectory.append(TrajectoryStep("p1", 0, context.observation, context.observation.legal_action_mask, 0))
        trajectory.record_terminal(TerminalState("p1", 1))
        source = source_roots(ledger.contract, panel="exploration", source_seed=seed, trajectory=trajectory)
        return env, context, archive, source, ledger.contract

    def prepare_archive(self, env, archive, source, contract, leaf="model", *, synthetic=False):
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        cfg = SearchConfiguration("incumbent", leaf=leaf, seconds=1.)
        return prepare_archived_actor_timed_job(archive, source, contract=contract,
            root_id=source["roots"][0]["root_id"], configuration=cfg,
            checkpoint_contract=self.helper.checkpoint_contract, showdown_root=self.helper.showdown_root,
            runtime_configuration=actor_timed_runtime_recipe(cfg, self.helper.checkpoint_contract),
            source_config=env.config, source_config_sha256=digest(_state_payload(env.config)),
            synthetic_process_fixture=synthetic)

    def test_archived_original_actor_job_reconstructs_cached_annotations_without_launch_or_private_transport(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        env, context, archive, source, contract = self.archive_fixture()
        expected = ActorAnnotationBoundary.capture(env, context)
        snapshot = archive._private[0]
        before = digest(_state_payload(snapshot))
        with patch("subprocess.Popen") as launch:
            job, binding = self.prepare_archive(env, archive, source, contract)
            sibling, _ = self.prepare_archive(env, archive, source, contract, "hp_fraction")
        launch.assert_not_called()
        self.assertEqual(job.annotation_boundary._original_sha, expected._original_sha)
        self.assertEqual(sibling.annotation_boundary._original_sha, expected._original_sha)
        self.assertIsNot(job.annotation_boundary, sibling.annotation_boundary)
        self.assertIsNot(job.annotation_boundary.state, sibling.annotation_boundary.state)
        self.assertEqual(digest(_state_payload(snapshot)), before)
        self.assertNotIn(b"never-policy-input", pickle.dumps(job))
        self.assertFalse(binding["source_policy_validated"])
        selected = job({}, lambda phase:None)
        self.assertGreater(job.read_selection(selected)["accepted_units"], 0)
        self.assertEqual(digest(_state_payload(snapshot)), before)
        self.assertIsNone(env._process)

    def test_archived_missing_rewound_and_post_observation_mismatching_cache_stays_unavailable(self):
        env, _, archive, source, contract = self.archive_fixture()
        original = archive._private[0]
        for mode in ("absent", "missing_actor", "rewound", "observation_mismatch"):
            with self.subTest(mode=mode):
                snapshot = deepcopy(original)
                if mode == "absent": snapshot = replace(snapshot, annotation_cache=None)
                elif mode == "missing_actor": snapshot.annotation_cache.tier2_trackers.pop("p1")
                elif mode == "rewound": snapshot.annotation_cache.tier2_trackers["p1"]._assessed_until = 0
                else:
                    from pokezero.showdown import _ReplayParser
                    parser = _ReplayParser.from_snapshot(snapshot.replay)
                    parser.feed(["|-status|p1a: Zangoose|par"])
                    snapshot = replace(snapshot, replay=parser.snapshot(),
                        protocol_lines=snapshot.protocol_lines+("|-status|p1a: Zangoose|par",))
                archive._private[0] = snapshot
                with patch("subprocess.Popen") as launch:
                    with self.assertRaises(ValueError): self.prepare_archive(env, archive, source, contract)
                launch.assert_not_called()
                archive._private[0] = original

    def test_original_actor_reconstruction_does_not_consume_opponent_private_request_fields(self):
        env, _, archive, source, contract = self.archive_fixture()
        job, _ = self.prepare_archive(env, archive, source, contract)
        snapshot = archive._private[0]
        private = deepcopy(snapshot.latest_requests)
        private["p2"] = {"private_canary":"opponent-private-never-input", "side":{"id":"p2", "pokemon":[]}}
        archive._private[0] = replace(snapshot, latest_requests=private)
        changed, _ = self.prepare_archive(env, archive, source, contract)
        self.assertEqual(changed.annotation_boundary._original_sha, job.annotation_boundary._original_sha)
        self.assertNotIn(b"opponent-private-never-input", pickle.dumps(changed))

    def test_archived_handoff_guards_initial_callback_returned_job_and_final_binding(self):
        import pokezero.mcts_eval.search_over_raw_archive as module
        env, _, archive, source, contract = self.archive_fixture()
        original_binding = module.archived_exploration_root_binding
        original_public = archive._public
        def early_mutation(*args, **kwargs):
            archive._public = deepcopy(original_public)
            return original_binding(*args, **kwargs)
        with patch.object(module, "archived_exploration_root_binding", early_mutation):
            with self.assertRaisesRegex(ValueError, "original archived actor timed preparation"):
                self.prepare_archive(env, archive, source, contract)
        archive._public = original_public
        original_post = ActorAnnotatedTimedJob.__post_init__
        for name in ("configuration", "checkpoint_contract"):
            with self.subTest(returned_job=name):
                def adopted_copy(job):
                    object.__setattr__(job, name, deepcopy(getattr(job, name)))
                    original_post(job)
                with patch.object(ActorAnnotatedTimedJob, "__post_init__", adopted_copy):
                    with self.assertRaisesRegex(ValueError, "original prospective archive handoff"):
                        self.prepare_archive(env, archive, source, contract)
        calls = []
        def late_binding_mutation(*args, **kwargs):
            value = original_binding(*args, **kwargs)
            calls.append(value)
            if len(calls) == 2: calls[0]["runtime_qualified"] = True
            return value
        with patch.object(module, "archived_exploration_root_binding", late_binding_mutation):
            with self.assertRaisesRegex(ValueError, "returned binding changed"):
                self.prepare_archive(env, archive, source, contract)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(env._process)

    def test_actual_owned_archive_derived_job_independent_readback(self):
        env, _, archive, source, contract = self.archive_fixture()
        job, _ = self.prepare_archive(env, archive, source, contract, synthetic=True)
        self.owned_readback(env, job, check_late_mutations=False)

    def test_actual_timed_model_and_hp_preserve_original_root_features_without_opponent_seed(self):
        for leaf in ("model", "hp_fraction"):
            with self.subTest(leaf=leaf):
                env, job = self.job(leaf)
                self.assertTrue(job.annotation_boundary.residuals_active)
                phases = []
                selected = job({}, phases.append)
                self.assertEqual(selected["status"], "SELECTED")
                readout = job.read_selection(selected)
                self.assertGreater(readout["accepted_units"], 0)
                self.assertEqual(selected["evidence"][RUNTIME_KEY]["original_boundary_sha256"],
                    job.annotation_boundary._original_sha)
                self.assertFalse(readout["runtime_qualified"])
                self.assertFalse(readout["scientific_collection_authorized"])
                self.assertFalse(selected["evidence"][RUNTIME_KEY]["sampled_opponent_history_implemented"])
                self.assertEqual(phases, ["startup", "selection", "cleanup"])
                self.assertIsNone(env._process)

    def test_actual_adapter_restores_annotation_source_and_cannot_retry(self):
        env, job = self.job()
        adapter = job._make_adapter()
        self.addCleanup(adapter.close)
        original = adapter._native._annotation_source
        state = job.context.public_materialization_state
        engine = state.belief_engine.clone()
        engine.set_source = self.helper.source
        context = replace(job.context, public_materialization_state=replace(state, belief_engine=engine))
        selected = adapter.select(context, root_id=job.root_id,
            statistics_root_id=job.statistics_root_id, selection_seed=job.selection_seed)
        self.assertIs(adapter._native._annotation_source, original)
        self.assertIsNone(adapter._annotation_restore)
        with self.assertRaisesRegex(ValueError, "single-attempt"):
            adapter.select(job.context, root_id=job.root_id,
                statistics_root_id=job.statistics_root_id, selection_seed=job.selection_seed)
        self.assertIsNone(env._process)
        self.assertIn(RUNTIME_KEY, selected["evidence"])

    def test_original_job_references_and_hidden_checkpoint_values_cannot_be_replaced(self):
        _, job = self.job()
        for name in ("checkpoint_contract", "configuration", "expected_runtime_configuration",
                "context", "contract", "annotation_boundary", "_public_request", "_consumed_binding"):
            with self.subTest(name=name):
                original = getattr(job, name)
                changed = deepcopy(original)
                self.assertIsNot(changed, original)
                object.__setattr__(job, name, changed)
                with self.assertRaisesRegex(ValueError, "original actor timed job"):
                    job._check()
                object.__setattr__(job, name, original)
                job._check()
        masks = job.checkpoint_contract.feature_masks
        original = masks["tier2_residuals"]
        masks["tier2_residuals"] = not original
        with self.assertRaisesRegex(ValueError, "original actor timed job"):
            job._check()
        masks["tier2_residuals"] = original
        job._check()

    def test_cleanup_runs_after_selection_or_cleanup_phase_mutation_or_error(self):
        close = ActorAnnotatedTimedAdapter.close
        for where, mode in (("selection", "mutation"), ("cleanup", "mutation"), ("cleanup", "error")):
            with self.subTest(where=where, mode=mode):
                _, job = self.job()
                closed = []
                def actual_close(adapter):
                    closed.append(adapter)
                    return close(adapter)
                def phase(name):
                    if name == where:
                        if mode == "error":
                            raise RuntimeError("authored cleanup notification failure")
                        object.__setattr__(job, "checkpoint_contract", deepcopy(job.checkpoint_contract))
                with patch.object(ActorAnnotatedTimedAdapter, "close", actual_close):
                    with self.assertRaises((ValueError, RuntimeError)):
                        job({}, phase)
                self.assertEqual(len(closed), 1)
                self.assertTrue(closed[0]._closed)
                self.assertIsNone(closed[0]._annotation_restore)

    def test_adapter_original_contract_and_native_config_guard_and_restoration_slot_attack(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputSearchAdapter
        for source_is_none in (False, True):
            with self.subTest(source_is_none=source_is_none):
                _, job = self.job()
                adapter = job._make_adapter()
                self.addCleanup(adapter.close)
                for owner, name in ((adapter, "contract"), (adapter, "configuration"), (adapter._native, "_config")):
                    original = getattr(owner, name)
                    setattr(owner, name, deepcopy(original))
                    with self.assertRaisesRegex(ValueError, "original actor timed adapter"):
                        adapter._actor_inputs()
                    setattr(owner, name, original)
                if source_is_none:
                    adapter._native._annotation_source = None
                original_source = adapter._native._annotation_source
                def corrupt_bookkeeping(actual, context, **kwargs):
                    # Negative cleanup seam only; no mocked success or policy-input claim.
                    actual._native._annotation_source = actual.annotation_boundary
                    actual._annotation_restore = None
                    return {"evidence":{}}
                with patch.object(WorldInputSearchAdapter, "select", corrupt_bookkeeping):
                    with self.assertRaisesRegex(ValueError, "actual original actor source"):
                        adapter.select(job.context)
                self.assertIs(adapter._native._annotation_source, original_source)
                self.assertTrue(adapter._poisoned)
                with self.assertRaisesRegex(ValueError, "single-attempt"):
                    adapter.select(job.context)

    def test_read_and_supervisor_callbacks_cannot_replace_original_job(self):
        import pokezero.mcts_eval.search_over_raw_actor_timed_run as module
        _, job = self.job()
        original = job.expected_runtime_configuration
        def mutation(*args, **kwargs):
            object.__setattr__(job, "expected_runtime_configuration", deepcopy(original))
            return {}
        with patch.object(module, "_read_actor_work", mutation):
            with self.assertRaisesRegex(ValueError, "original actor timed job"):
                job.read_selection({})
        object.__setattr__(job, "expected_runtime_configuration", original)
        def fake_supervision(*args, **kwargs):
            kwargs["verify_admission"]()
            return {"supervisor_exit_code":1}
        with patch.object(module, "_supervise_owned_job", fake_supervision):
            with self.assertRaisesRegex(ValueError, "original actor timed job"):
                supervise_actor_timed(job, directory="unused", binding_sha256=job._frozen_inputs,
                    verify_admission=mutation)
        object.__setattr__(job, "expected_runtime_configuration", original)
        with patch.object(module, "_supervise_owned_job", mutation):
            with self.assertRaisesRegex(ValueError, "original actor timed job"):
                supervise_actor_timed(job, directory="unused", binding_sha256=job._frozen_inputs,
                    verify_admission=lambda:None)

    def test_real_owning_timed_job_independent_readback_and_wrong_seed_rejection(self):
        env, job = self.job()
        self.owned_readback(env, job)

    def owned_readback(self, env, job, *, check_late_mutations=True):
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        import pokezero_search.pokezero_search as native_search
        import poke_engine.poke_engine as native_engine
        files = fixed_work_reader_source_hashes()
        h = self.helper
        paths = [Path(sys.executable).resolve(), h.checkpoint,
            Path(h.checkpoint_contract.model_path), Path(h.checkpoint_contract.tables_path),
            Path(native_search.__file__), Path(native_engine.__file__)]
        paths.extend(p for p in (Path(h.showdown_root)/"dist").rglob("*") if p.is_file())
        for path in paths: files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)/"owned-actor-timed"
            terminal = supervise_actor_timed(job, directory=attempt, binding_sha256=job._frozen_inputs,
                verify_admission=lambda:None)
            self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
            self.assertEqual(terminal["worker_exit_code"], 0)
            result = terminal["result"]
            inputs = dict(directory=attempt, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                binding_sha256=job._frozen_inputs, configuration=job.configuration,
                root_id=job.root_id, statistics_root_id=job.statistics_root_id,
                selection_seed=job.selection_seed, runtime_configuration=job.expected_runtime_configuration,
                runtime_sha256=job.expected_runtime_sha256, consumed_binding=job._consumed_binding,
                legal_action_mask=job._legal_action_mask, decision_round=job.context.decision_round_index,
                file_hashes=files, checkpoint_path=h.checkpoint_contract.checkpoint_path,
                checkpoint_sha256=h.checkpoint_contract.checkpoint_sha256, showdown_root=h.showdown_root,
                public_request_sha256=job.expected_public_request_sha256,
                annotation_sha256=job.annotation_boundary._original_sha)
            self.assertIsNone(ActorAnnotatedTimedRunValidator(**inputs)(result))
            reader = ActorAnnotatedTimedRunValidator(**inputs, allow_synthetic=True)
            self.assertEqual(reader(result), digest(result))
            self.assertTrue(reader.last_validation["process_receipts_validated"])
            self.assertFalse(reader.last_validation["resource_cost_complete"])
            with self.assertRaisesRegex(ValueError, "prospective handoff"):
                ActorAnnotatedTimedRunValidator(**dict(inputs, annotation_sha256="0"*64),
                    allow_synthetic=True)(result)
            changed = deepcopy(result)
            changed["evidence"][RUNTIME_KEY]["root_annotations_only"] = False
            with self.assertRaises(ValueError): reader(changed)
            original_files = ActorAnnotatedTimedRunValidator._files
            for name in (("configuration", "runtime", "directory", "contract", "_readout_schema")
                    if check_late_mutations else ()):
                with self.subTest(final_reader_mutation=name):
                    calls = []
                    def late_mutation(actual):
                        original_files(actual)
                        calls.append(None)
                        if len(calls) == 3:  # constructor, initial read, final read
                            setattr(actual, name, "forged-schema" if name == "_readout_schema"
                                else deepcopy(getattr(actual, name)))
                    with patch.object(ActorAnnotatedTimedRunValidator, "_files", late_mutation):
                        guarded = ActorAnnotatedTimedRunValidator(**inputs, allow_synthetic=True)
                        with self.assertRaisesRegex(ValueError, "original actor timed reader"):
                            guarded(result)
                        self.assertIsNone(guarded.last_validation)
                        self.assertEqual(len(calls), 3)
            self.assertFalse(list((attempt/"owned-groups/active").glob("group-*.json")))
        self.assertIsNone(env._process)
