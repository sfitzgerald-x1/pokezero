"""Loaded random-model adapter on an authored public request, no source game.

Real checkpoint loading, native search, parser, encoder and policy callbacks
remain unstubbed. This does not establish original-source or physics fidelity.
"""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, _load_export_module
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_native_fixed_values import NativeFixedWorkValueContract
from pokezero.mcts_eval.search_over_raw_annotation_boundary import ActorAnnotationBoundary
from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import (
    NativeFixedWorkValueAdapter, native_fixed_value_runtime_recipe,
    NativeFixedWorkValueJob,
)


@unittest.skipUnless(_crate_ready, "requires fresh isolated model runtime")
class LoadedNativeValueAdapterTests(_EncodedSearchFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.neural_policy import (
            save_transformer_checkpoint, TransformerTrainingConfig, TransformerTrainingResult,
        )
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        cls.showdown_root = str(Path(DEFAULT_SHOWDOWN_ROOT).resolve())
        cls.source = load_gen3_randbat_source_cached(cls.showdown_root)
        cls.checkpoint = Path(cls.tmpdir.name).resolve() / "authored-random-checkpoint.pt"
        cls.tables_path = cls.checkpoint.parent / "authored-matching-tables.json"
        cls.tables_path.write_text(cls.tables_json)
        save_transformer_checkpoint(cls.checkpoint, cls.python_model, result=TransformerTrainingResult(
            model_config=cls.config, training_config=TransformerTrainingConfig(), epochs=(),
            belief_set_source_hash=cls.source.metadata.source_hash))
        cls.checkpoint_contract = resolve_checkpoint_contract(cls.checkpoint,
            showdown_root=cls.showdown_root, showdown_source_sha256=cls.source.metadata.source_hash,
            model_path=cls.artifact.resolve(), tables_path=cls.tables_path)
        # A separately authored profile, not a mutation of the loaded model or
        # champion feature law. Equal random weights, distinct paired artifacts.
        from pokezero.neural_policy import (EntityTokenTransformerPolicy,
            observation_spec_from_model_config, feature_masks_from_model_config)
        from export_encoder_tables import build_tables
        cls.supported_config = replace(cls.config, tier2_residuals=False)
        supported = EntityTokenTransformerPolicy(cls.supported_config).eval()
        supported.load_state_dict(cls.python_model.state_dict())
        export = _load_export_module()
        artifact = cls.checkpoint.parent / "authored-residual-disabled-trace.pt"
        export.export_torchscript(export.build_exportable_module(supported),
            export.make_random_inputs(cls.supported_config, export.TRACE_BATCH, seed=7), artifact)
        tables = build_tables(cls.showdown_root, observation_schema_version=cls.supported_config.observation_schema_version,
            spec=observation_spec_from_model_config(cls.supported_config),
            masks=feature_masks_from_model_config(cls.supported_config),
            trained_tokens=cls.supported_config.category_vocab)
        table_path = cls.checkpoint.parent / "authored-residual-disabled-tables.json"
        table_path.write_text(json.dumps(tables))
        checkpoint = cls.checkpoint.parent / "authored-residual-disabled-checkpoint.pt"
        save_transformer_checkpoint(checkpoint, supported, result=TransformerTrainingResult(
            model_config=cls.supported_config, training_config=TransformerTrainingConfig(), epochs=(),
            belief_set_source_hash=cls.source.metadata.source_hash))
        cls.supported_contract = resolve_checkpoint_contract(checkpoint,
            showdown_root=cls.showdown_root, showdown_source_sha256=cls.source.metadata.source_hash,
            model_path=artifact, tables_path=table_path)

    def authored_context(self, *, model_config=None, active_name="zangoose"):
        from pokezero.local_showdown import (
            LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance,
        )
        from pokezero.neural_policy import (
            category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config,
        )
        from pokezero.showdown import _ReplayParser
        from pokezero.policy import PolicyContext
        from pokezero.trajectory import BattleTrajectory
        model_config = self.config if model_config is None else model_config
        config = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(showdown_root=self.showdown_root, set_belief_source=True),
            feature_masks_from_model_config(model_config), context="authored random native probe fixture",
            required_specs=observation_spec_from_model_config(model_config),
            required_vocabs=category_vocab_from_model_config(model_config, self.showdown_root))
        env = LocalShowdownEnv(config)
        self.addCleanup(env.close)
        env._parser = _ReplayParser("authored-native-probe", complete_prefix=True,
            hp_visibility={"p1":"exact", "p2":"percentage"})
        lines, request = self.authored_protocol(active_name=active_name)
        # All data is an explicitly authored parser/request fixture. No reset,
        # step, snapshot restoration, source allocation or protocol filtering.
        env._apply_event(dict(type="stream", stream="omniscient", lines=lines))
        env._apply_event(dict(type="stream", stream="p1", lines=["|request|"+json.dumps(request)]))
        observation = env.observe("p1")
        public = env.public_materialization_state("p1")
        self.assertIsNone(env._process)
        return env, PolicyContext("p1", 0, public.replay.battle_id, "gen3randombattle", 17,
            observation, ("p1",), BattleTrajectory(public.replay.battle_id, "gen3randombattle", 17),
            requested_observations={"p1":observation},
            requested_legal_action_masks={"p1":tuple(observation.legal_action_mask)},
            public_materialization_state=public)

    def authored_protocol(self, *, active_name="zangoose"):
        """Six member, closed-vocabulary request and parser history; no game."""
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.gen3_damage import gen3_hp_stat, gen3_stat
        from pokezero.determinization import _gen3_randbat_fixture_spread
        dex = load_showdown_dex_cached(self.showdown_root)
        names = ("zangoose", "heracross", "medicham", "charizard", "snorlax", "blastoise")
        names = (active_name,) + tuple(name for name in names if name != active_name)
        chosen = []
        for name in names:
            universe = self.source.universe_for(name)
            candidates = [v for v in universe.variants if all(not m.startswith("hiddenpower")
                and self.source.move_metadata.get(m, {}).get("category") in ("Physical", "Special") for m in v.moves)]
            variant = (candidates or list(universe.variants))[0]
            chosen.append(variant)
        party = []
        for i, variant in enumerate(chosen):
            info = dex.species_info(variant.species)
            spread = _gen3_randbat_fixture_spread({}, species=variant.species, moves=variant.moves,
                item=variant.item, level=variant.level, set_source=self.source)
            self.assertIsNotNone(spread)
            maxhp = gen3_hp_stat(info.base_stats["hp"], level=variant.level,
                iv=spread["ivs"]["hp"], ev=spread["evs"]["hp"])
            party.append(dict(ident="p1: "+info.name, details=f"{info.name}, L{variant.level}, M",
                condition=f"{maxhp}/{maxhp}" if i == 0 else "0 fnt", active=i == 0,
                moves=list(variant.moves), baseAbility=variant.ability, ability=variant.ability,
                item=variant.item, stats={s:gen3_stat(info.base_stats[s], level=variant.level,
                    iv=spread["ivs"][s], ev=spread["evs"][s])
                    for s in ("atk", "def", "spa", "spd", "spe")}))
        active = chosen[0]
        active_label = dex.species_info(active.species).name
        moves = [dict(id=m, move=dex.move_info(m).name, pp=dex.move_info(m).max_pp-1,
            maxpp=dex.move_info(m).max_pp, disabled=False) for m in active.moves]
        request = dict(active=[dict(moves=moves)], side=dict(id="p1", name="authored p1", pokemon=party))
        lines = ["|player|p1|authored p1|", "|player|p2|authored p2|", "|gen|3", "|start"]
        for variant in chosen[1:]:
            info = dex.species_info(variant.species)
            lines += [f"|switch|p2a: {info.name}|{info.name}, L{variant.level}, M|100/100",
                f"|faint|p2a: {info.name}"]
        lines += [f"|switch|p1a: {active_label}|{party[0]['details']}|{party[0]['condition']}",
            f"|switch|p2a: {active_label}|{active_label}, L{active.level}, M|100/100"]
        for turn, move in enumerate(moves, 1):
            lines += [f"|turn|{turn}", f"|move|p1a: {active_label}|{move['move']}|p2a: {active_label}",
                f"|move|p2a: {active_label}|{move['move']}|p1a: {active_label}"]
        lines += ["|turn|5"]
        return lines, request

    def test_original_annotation_handoff_preserves_unsupported_history_refusal(self):
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
        env, context = self.authored_context()
        request = PublicRootRequest.capture(context.public_materialization_state, context.observation)
        public_sha = public_request_witness(request)["public_request_sha256"]
        configuration = SearchConfiguration("incumbent", seconds=1.)
        value_contract = NativeFixedWorkValueContract(17)
        expected = native_fixed_value_runtime_recipe(configuration, self.checkpoint_contract,
            value_contract=value_contract)
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        adapter = NativeFixedWorkValueAdapter(configuration, value_contract=value_contract,
            annotation_boundary=ActorAnnotationBoundary.capture(env, context),
            expected_public_request_sha256=public_sha, fixed_work=FixedWorkDiagnosticContract(),
            checkpoint_contract=self.checkpoint_contract, showdown_root=self.showdown_root)
        self.addCleanup(adapter.close)
        self.assertEqual(digest(adapter.runtime_configuration), digest(expected))
        annotation_env = adapter._decider._env
        original_source = adapter._native._annotation_source
        selected = adapter.select(context, root_id="authored-native-probe:"+configuration.identity,
            statistics_root_id="authored-native-probe", selection_seed=7)
        self.assertEqual(selected["fixed_work_diagnostic"]["accepted_actual"], 64)
        self.assertIsNone(adapter._native._value_diagnostic_session)
        self.assertIsNone(adapter._native._world_attempt_observer)
        self.assertIs(adapter._native._annotation_source, original_source)
        with self.assertRaisesRegex(ValueError, "residual observer is not implemented"):
            adapter.finish_native_fixed_values(selected)
        self.assertTrue(adapter._poisoned)
        self.assertTrue(adapter._fixed_value_session.closed)
        self.assertIsNone(env._process)
        self.assertIsNone(annotation_env._process)
        with self.assertRaises(ValueError): adapter.finish_native_fixed_values(selected)

    def test_missing_active_annotation_handoff_refuses_before_native_work(self):
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        env, context = self.authored_context()
        public_sha = public_request_witness(PublicRootRequest.capture(
            context.public_materialization_state, context.observation))["public_request_sha256"]
        cfg = SearchConfiguration("incumbent", seconds=1.)
        adapter = NativeFixedWorkValueAdapter(cfg, value_contract=NativeFixedWorkValueContract(17),
            expected_public_request_sha256=public_sha, fixed_work=FixedWorkDiagnosticContract(),
            checkpoint_contract=self.checkpoint_contract, showdown_root=self.showdown_root)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(ValueError, "original actor boundary handoff"):
            adapter.select(context, root_id="authored-native-probe:"+cfg.identity,
                statistics_root_id="authored-native-probe", selection_seed=7)
        self.assertIsNone(adapter._fixed_value_session)
        self.assertIsNone(adapter._input_recorder)
        self.assertTrue(adapter._poisoned)

    def temporary_profile_job(self, *, certified_empty_delta_progress=False):
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
        env, context = self.authored_context(model_config=self.supported_config, active_name="heracross")
        public_sha = public_request_witness(PublicRootRequest.capture(
            context.public_materialization_state, context.observation))["public_request_sha256"]
        cfg = SearchConfiguration("incumbent", seconds=1.)
        contract = NativeFixedWorkValueContract(17, certified_empty_delta_progress=certified_empty_delta_progress)
        recipe = native_fixed_value_runtime_recipe(cfg, self.supported_contract, value_contract=contract)
        job = NativeFixedWorkValueJob(cfg, self.supported_contract, self.showdown_root, context,
            "authored-native-probe:"+cfg.identity, 7, "authored-native-probe", digest(recipe),
            expected_public_request_sha256=public_sha, value_contract=contract,
            synthetic_process_fixture=True)
        return env, job

    def test_explicit_v3_actual_job_labels_same_temporary_profile_without_redrawing_noops(self):
        from copy import deepcopy
        from pokezero.mcts_eval.search_over_raw_native_fixed_values import (
            EMPTY_DELTA_SCHEMA, EMPTY_DELTA_LAW, read_fixed_native_values,
        )
        env, job = self.temporary_profile_job(certified_empty_delta_progress=True)
        result = job({}, lambda _:None)
        self.assertEqual(result["selected"]["fixed_work_diagnostic"]["accepted_actual"], 64)
        self.assertEqual(result["native_values"]["schema"], EMPTY_DELTA_SCHEMA)
        rows = result["native_values"]["native_invocations"]
        self.assertGreater(sum(row["readout"]["certified_empty_delta_boundaries"] for row in rows), 0)
        for row in rows:
            self.assertEqual(row["readout"]["empty_delta_progress_law"], EMPTY_DELTA_LAW)
            self.assertEqual(row["readout"]["required_labels"], 16)
            self.assertEqual(row["readout"]["unknown_labels"], 0)
            self.assertFalse(row["readout"]["scientific_calibration_evidence"])
        row = rows[0]
        raw = row["raw_native_export"]
        value = json.loads(raw)
        inputs = dict(binding=row["binding"], root_key=value["root_information_key"], worker=value["worker"])
        with self.assertRaises(ValueError):
            read_fixed_native_values(raw, contract=replace(job.value_contract, certified_empty_delta_progress=False), **inputs)
        for kind in ("schema", "law", "boolean", "negative", "above_boundary", "drop"):
            altered = deepcopy(value)
            label = altered["leaves"][0]["labels"][0]
            if kind == "schema": altered["schema"] = "pokezero.native-visited-value.fixed-work-native.v2"
            if kind == "law": altered["empty_delta_progress_law"] = "redraw-away-noops"
            if kind == "boolean": label["certified_empty_delta_boundaries"] = True
            if kind == "negative": label["certified_empty_delta_boundaries"] = -1
            if kind == "above_boundary": label["certified_empty_delta_boundaries"] = label["boundaries"]+1
            if kind == "drop": del label["certified_empty_delta_boundaries"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                read_fixed_native_values(json.dumps(altered), contract=job.value_contract, **inputs)
        aliased = deepcopy(job.value_contract)
        object.__setattr__(aliased, "certified_empty_delta_progress", 1)
        with self.assertRaises(ValueError): read_fixed_native_values(raw, contract=aliased, **inputs)
        self.assertIsNone(env._process)

    def test_actual_temporary_profile_job_preserves_terminal_dead_end_refusal(self):
        env, job = self.temporary_profile_job()
        phases = []
        # This loaded profile passes the residual feature gate, but the real
        # rollout still refuses a dead end. Do not turn that into a score or
        # substitute another fixture and describe this profile as qualified.
        with self.assertRaisesRegex(ValueError, "nonterminal dead end"):
            job({}, phases.append)
        self.assertEqual(phases, ["startup", "selection", "cleanup"])
        self.assertFalse(self.supported_contract.feature_masks["tier2_residuals"])
        self.assertTrue(self.checkpoint_contract.feature_masks["tier2_residuals"])
        self.assertIsNone(env._process)
        job._check()

    def test_exact_failed_empty_delta_regenerates_as_an_unscored_progressing_turn(self):
        import pokezero_search
        env, job = self.temporary_profile_job()
        with self.assertRaisesRegex(ValueError, "nonterminal dead end") as caught:
            job({}, lambda _:None)
        # Native state stays inside this explicitly authored PRIVATE controller
        # test. It is not returned to policy providers or public run receipts.
        diagnostic = json.loads(caught.exception.private_native_empty_delta_json)
        self.assertEqual(diagnostic["schema"], "pokezero.native-empty-delta.private-diagnostic.v1")
        self.assertTrue(diagnostic["private_controller_only"])
        self.assertFalse(diagnostic["continuation_accepted"])
        self.assertFalse(diagnostic["scientific_calibration_evidence"])
        self.assertEqual(diagnostic["selected_instruction_count"], 0)
        context = dict(diagnostic["context"])
        context["hp_percent"] = [seat for seat, enabled in zip(("p1", "p2"), context["hp_percent"]) if enabled]
        regenerated = json.loads(pokezero_search.branch_events(diagnostic["pre_state"],
            diagnostic["side_one_choice"], diagnostic["side_two_choice"], json.dumps(context),
            branch_on_damage=diagnostic["branch_on_damage"], include_post_state=True))
        self.assertEqual([row["percentage"] for row in regenerated["branches"]], diagnostic["probabilities"])
        self.assertAlmostEqual(sum(diagnostic["probabilities"]), 100., places=3)
        selected = regenerated["branches"][diagnostic["selected_branch_index"]]
        self.assertEqual(selected["post_state"], diagnostic["pre_state"])
        self.assertTrue(selected["turn_completed"])
        self.assertFalse(selected["attribution_unsafe"])
        self.assertEqual(selected["events"][-1], "|turn|"+str(context["turn"]+1))
        self.assertGreaterEqual(sum(line.startswith("|move|") for line in selected["events"]), 1)
        self.assertIn("|upkeep", selected["events"])
        self.assertIsNone(env._process)

    def test_actual_owning_worker_reaps_failed_loaded_profile_without_retry(self):
        import tempfile
        from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import supervise_native_fixed_values
        env, job = self.temporary_profile_job()
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "authored-owned-native-probe"
            terminal = supervise_native_fixed_values(job, directory=attempt,
                binding_sha256=job._value_job_sha, verify_admission=lambda:job._check())
            self.assertNotEqual(terminal["supervisor_exit_code"], 0)
            self.assertIsNone(terminal["result"])
            self.assertFalse(terminal["resource_cost_complete"])
            self.assertTrue(json.loads((attempt / "claim.json").read_text())["synthetic_process_fixture"])
            worker = json.loads((attempt / "worker-result.json").read_text())
            self.assertEqual(worker["exit_code"], 1)
            self.assertEqual(worker["error_type"], "ValueError")
            self.assertEqual([row["phase"] for row in worker["phases"]], ["startup", "selection", "cleanup"])
            self.assertFalse(list((attempt / "owned-groups/active").glob("group-*.json")))
            with self.assertRaises((ValueError, FileExistsError)):
                supervise_native_fixed_values(job, directory=attempt,
                    binding_sha256=job._value_job_sha, verify_admission=lambda:job._check())
        self.assertIsNone(env._process)

    def test_explicit_v3_owned_job_and_independent_reader_remain_synthetic_nonclearing(self):
        import hashlib
        import pickle
        import sys
        import tempfile
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import (
            supervise_native_fixed_values, NativeFixedWorkValueRunValidator,
        )
        env, job = self.temporary_profile_job(certified_empty_delta_progress=True)
        files = fixed_work_reader_source_hashes()
        # Use the catalog's actual fingerprint, never fabricate a file digest.
        self.assertEqual(len(job._consumed_binding["set_source_hash"]), 16)
        self.assertEqual(job._consumed_binding["set_source_hash"], self.source.metadata.source_hash)
        import pokezero_search.pokezero_search as native_search
        import poke_engine.poke_engine as native_engine
        paths = [Path(sys.executable).resolve(), Path(job.checkpoint_contract.checkpoint_path),
            Path(job.checkpoint_contract.model_path), Path(job.checkpoint_contract.tables_path),
            Path(native_search.__file__), Path(native_engine.__file__)]
        paths.extend(path for path in (Path(job.showdown_root) / "dist").rglob("*") if path.is_file())
        for path in paths: files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "authored-owned-native-v3"
            terminal = supervise_native_fixed_values(job, directory=attempt,
                binding_sha256=job._value_job_sha, verify_admission=lambda:job._check())
            self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
            self.assertEqual(terminal["worker_exit_code"], 0)
            result = terminal["result"]
            inputs = dict(directory=attempt, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                binding_sha256=job._value_job_sha, configuration=job.configuration, root_id=job.root_id,
                statistics_root_id=job.statistics_root_id, selection_seed=job.selection_seed,
                runtime_sha256=job.expected_runtime_sha256, consumed_binding=job._consumed_binding,
                file_hashes=files, checkpoint_path=job.checkpoint_contract.checkpoint_path,
                checkpoint_sha256=job.checkpoint_contract.checkpoint_sha256, showdown_root=job.showdown_root,
                public_request_sha256=job.expected_public_request_sha256, value_contract=job.value_contract)
            self.assertIsNone(NativeFixedWorkValueRunValidator(**inputs)(result))
            reader = NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)
            self.assertEqual(reader(result), digest(result))
            self.assertTrue(reader.last_validation["process_receipts_validated"])
            self.assertFalse(reader.last_validation["scientific_calibration_evidence"])
            self.assertFalse(reader.last_validation["resource_cost_complete"])
            self.assertFalse(list((attempt / "owned-groups/active").glob("group-*.json")))
        self.assertIsNone(env._process)

    def test_full_annotation_state_and_root_mutants_fail_even_when_masked(self):
        import pickle
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from copy import deepcopy
        env, context = self.authored_context()
        request = PublicRootRequest.capture(context.public_materialization_state, context.observation)
        boundary = ActorAnnotationBoundary.capture(env, context)
        transported = pickle.loads(pickle.dumps(boundary))
        transported.validate(request, self.checkpoint_contract)
        self.assertEqual(transported.state.transition_tokens, boundary.state.transition_tokens)
        with self.assertRaisesRegex(ValueError, "acting-player only"):
            transported.overlay_for("p2")
        with self.assertRaises(ValueError):
            ActorAnnotationBoundary(boundary.state, boundary.observation,
                boundary.public_request_sha256, boundary.residuals_active)
        for kind in ("history", "request", "active", "root"):
            changed = deepcopy(boundary)
            if kind == "history": object.__setattr__(changed.state, "transition_tokens", ())
            if kind == "request": changed.state.request["side"]["name"] = "foreign"
            if kind == "active": object.__setattr__(changed, "residuals_active", False)
            if kind == "root": object.__setattr__(changed, "public_request_sha256", "0"*64)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                changed.validate(request, self.checkpoint_contract)

    def test_capture_rejects_masked_source_request_drift(self):
        env, context = self.authored_context()
        # The display name does not change any model tensor. It still changes
        # the exact original own request, so equal encodes are insufficient.
        env._latest_requests["p1"]["side"]["name"] = "later-source-request"
        with self.assertRaisesRegex(ValueError, "boundary changed"):
            ActorAnnotationBoundary.capture(env, context)

    def test_last_encoder_callback_cannot_mutate_request_or_checkpoint_law(self):
        from copy import deepcopy
        from unittest.mock import patch
        from pokezero.showdown import observation_from_player_state
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        env, context = self.authored_context()
        original_request = PublicRootRequest.capture(context.public_materialization_state, context.observation)
        boundary = ActorAnnotationBoundary.capture(env, context)
        for kind in ("request", "contract"):
            request = deepcopy(original_request)
            contract = deepcopy(self.checkpoint_contract)
            def mutate(*args, **kwargs):
                encoded = observation_from_player_state(*args, **kwargs)
                if kind == "request": request.state.self_request["side"]["name"] = "foreign"
                else: contract.feature_masks["tier2_residuals"] = False
                return encoded
            with self.subTest(kind=kind), patch("pokezero.showdown.observation_from_player_state", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed during annotation validation"):
                    boundary.validate(request, contract)

    def test_prospective_recipe_does_not_load_adapter_and_old_lane_stays_separate(self):
        from unittest.mock import patch
        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import fixed_work_runtime_recipe
        cfg = SearchConfiguration("incumbent", seconds=1.)
        contract = NativeFixedWorkValueContract(17)
        with patch("pokezero.mcts_eval.search_over_raw_native_fixed_value_run.NativeFixedWorkValueAdapter",
                side_effect=AssertionError("pure recipe must not load runtime")):
            recipe = native_fixed_value_runtime_recipe(cfg, self.checkpoint_contract, value_contract=contract)
        self.assertIn("native_fixed_values", recipe)
        with self.assertRaises(ValueError):
            native_fixed_value_runtime_recipe(object(), self.checkpoint_contract, value_contract=contract)
        with self.assertRaises(ValueError): fixed_work_runtime_recipe(cfg, self.checkpoint_contract)
        for changed in (SearchConfiguration("reference", seconds=1., workers=20),
                SearchConfiguration("incumbent", leaf="hp_fraction", seconds=1.),
                SearchConfiguration("incumbent", seconds=3.)):
            with self.subTest(configuration=changed), self.assertRaises(ValueError):
                native_fixed_value_runtime_recipe(changed, self.checkpoint_contract, value_contract=contract)


if __name__ == "__main__":
    unittest.main()
