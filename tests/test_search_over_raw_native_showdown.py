"""Genuine native captures -> real Showdown on an existing excluded fixture.

Temporary random model, unchanged fixed work; no champion or scientific panel.
The source policy is the existing fixture's first-legal policy, not a raw-policy
qualification. Every native construction/event/replay coordinate stays visible.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_supervision import FixedWorkAdapterJob,supervise_fixed_work


@dataclass(frozen=True)
class DiagnosedNativeFixtureJob(FixedWorkAdapterJob):
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        try:return super().__call__(ownership,phase)
        except BaseException as error:
            # Only this disposable fixture; no production exception disclosure.
            (Path(ownership["directory"]).parent/"fixture-error.json").write_text(
                json.dumps(dict(type=type(error).__name__,message=str(error))))
            raise


class ExcludedNativeShowdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pokezero_search
        if not getattr(pokezero_search,"MODEL_FEATURE_ENABLED",False) or not all(hasattr(pokezero_search,n)
                for n in ("NativeVisitedValueBank","native_scoped_boundary","native_edge_support","native_scoped_successor_support")):
            raise unittest.SkipTest("requires explicitly supplied prospective model-enabled native build")
        from test_native_fixed_work_adapter import LoadedNativeValueAdapterTests
        cls.model=LoadedNativeValueAdapterTests
        cls.model.setUpClass()

    @classmethod
    def tearDownClass(cls):cls.model.tearDownClass()

    def test_genuine_owned_native_capture_drives_original_paired_constructors_and_replays(self):
        from test_search_over_raw_archive import ActualSourceBoundaryArchiveTests
        import qualify_search_over_raw_source as source_driver
        from pokezero.local_showdown import LocalShowdownConfig,LocalShowdownEnv,env_config_from_checkpoint_provenance
        from pokezero.neural_policy import feature_masks_from_model_config,observation_spec_from_model_config,category_vocab_from_model_config
        from pokezero.mcts_eval.search_over_raw_ancestry import NativeAncestryContract
        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import fixed_work_runtime_recipe
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator,fixed_work_reader_source_hashes
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness,_state_payload
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
        from pokezero.mcts_eval.search_over_raw_native_chain import plan_owned_native_materializations,build_owned_native_materialization_jobs
        from pokezero.mcts_eval.search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS
        from pokezero.mcts_eval.search_over_raw_paired_materialization import supervise_paired_materialization_run,PairedMaterializationRunValidator
        from pokezero.mcts_eval.search_over_raw_consumed_materialization import (
            plan_owned_native_replays,build_owned_native_replay_jobs,reconcile_owned_replay_diagnostic)
        from pokezero.mcts_eval.search_over_raw_paired_replay import supervise_paired_prefix_run,PairedPrefixRunValidator
        import pokezero_search
        import pokezero_search.pokezero_search as native_search
        import poke_engine.poke_engine as native_engine

        fixture=ActualSourceBoundaryArchiveTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        profile=self.model.supported_config;checkpoint=self.model.supported_contract
        config=env_config_from_checkpoint_provenance(LocalShowdownConfig(showdown_root=self.model.showdown_root,set_belief_source=True),
            feature_masks_from_model_config(profile),context="excluded native/Showdown random-model test",
            required_specs=observation_spec_from_model_config(profile),
            required_vocabs=category_vocab_from_model_config(profile,self.model.showdown_root))
        env=LocalShowdownEnv(config);self.addCleanup(env.close);fixture.env=env
        root=fixture.root(fixture.collect());context,pending,_=fixture.archive.selected(root)
        self.assertIsNone(pending)
        request=PublicRootRequest.capture(context.public_materialization_state,context.observation)
        witness=public_request_witness(request)
        cfg=SearchConfiguration("incumbent",leaf="model",seconds=1.)
        ancestry=NativeAncestryContract(17,capture_native_edges=True)
        recipe=fixed_work_runtime_recipe(cfg,checkpoint,native_ancestry_contract=ancestry)
        statistics="excluded-native-showdown:"+str(source_driver.FIXTURE_SEED)
        job=DiagnosedNativeFixtureJob(cfg,checkpoint,self.model.showdown_root,context,
            statistics+":"+cfg.identity,source_driver.FIXTURE_SEED,statistics,digest(recipe),
            native_ancestry_contract=ancestry,expected_public_request_sha256=witness["public_request_sha256"])
        extras=[Path(checkpoint.model_path),Path(checkpoint.tables_path),Path(native_search.__file__),Path(native_engine.__file__)]
        files=fixed_work_reader_source_hashes()
        for path in [Path(sys.executable).resolve(),Path(checkpoint.checkpoint_path),*extras,
                *(p for p in (Path(self.model.showdown_root)/"dist").rglob("*") if p.is_file())]:
            files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve();native_path=directory/"native"
            terminal=supervise_fixed_work(job,directory=native_path,binding_sha256="a"*64,
                verify_admission=lambda:True)
            error=native_path/"fixture-error.json"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            selected=terminal["result"]
            reader=FixedWorkRunValidator(directory=native_path,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                binding_sha256="a"*64,configuration=cfg,root_id=job.root_id,statistics_root_id=statistics,
                selection_seed=source_driver.FIXTURE_SEED,runtime_sha256=digest(recipe),consumed_binding=job._consumed_binding,
                file_hashes=files,checkpoint_path=checkpoint.checkpoint_path,checkpoint_sha256=checkpoint.checkpoint_sha256,
                showdown_root=self.model.showdown_root,public_request_sha256=witness["public_request_sha256"],allow_synthetic=True)
            self.assertEqual(reader(selected),digest(selected))
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],64)
            runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,extra_files=extras)
            state=job.context.public_materialization_state
            options={k:False for k in BOOL_OPTIONS};options.update({k:None for k in MAP_OPTIONS})
            plan=plan_owned_native_materializations(selected,native_validator=reader,config=config,runtime=runtime,
                options=options,attempt_root=directory/"constructors",public_state_sha256=digest(_state_payload(state)),paired_context=state)
            constructors=build_owned_native_materialization_jobs(plan,selected,native_validator=reader,config=config,runtime=runtime,
                options=options,paired_context=state)
            self.assertEqual(len(plan.receipt()["records"]),4)
            self.assertTrue(constructors,"no consumed native overlay represented by original paired context")
            mats=[]
            for descriptor in constructors:
                current=descriptor["job"];path=Path(descriptor["attempt_directory"])
                result_terminal=supervise_paired_materialization_run(current,directory=path,binding_sha256="b"*64,
                    verify_admission=lambda:True,validate_result=lambda _:True)
                self.assertEqual(result_terminal["supervisor_exit_code"],0,result_terminal)
                result=result_terminal["result"]
                owner=PairedMaterializationRunValidator(directory=path,job_sha256=hashlib.sha256(pickle.dumps(current)).hexdigest(),
                    binding_sha256="b"*64,expected_inputs=current.input_binding(),runtime=runtime,
                    original_public_context=state,expected_context=result["paired_context"])
                self.assertEqual(owner(result["capture"]),digest(result["capture"]))
                mats.append(dict(construction_ordinal=descriptor["construction_ordinal"],result=result,validator=owner))
            replay_plan=plan_owned_native_replays(selected,mats,native_validator=reader,ancestry_contract=ancestry,
                native_extract=pokezero_search.native_scoped_boundary,config=config,attempt_root=directory/"replays",
                preserve_prefix=True,paired_contexts=True)
            replay_jobs=build_owned_native_replay_jobs(replay_plan,selected,mats,native_validator=reader,
                ancestry_contract=ancestry,native_extract=pokezero_search.native_scoped_boundary,config=config)
            self.assertTrue(replay_jobs,"all original captured native actions unavailable before replay")
            replays=[]
            for descriptor in replay_jobs:
                current=descriptor["job"];path=Path(descriptor["attempt_directory"])
                result_terminal=supervise_paired_prefix_run(current,directory=path,binding_sha256="c"*64,
                    verify_admission=lambda:True,validate_result=lambda _:True)
                self.assertEqual(result_terminal["supervisor_exit_code"],0,result_terminal)
                result=result_terminal["result"]
                owner=PairedPrefixRunValidator(directory=path,job_sha256=hashlib.sha256(pickle.dumps(current)).hexdigest(),
                    binding_sha256="c"*64,expected_inputs=current.input_binding(),expected_raw_root=current.raw_root,
                    runtime=runtime,snapshot=current.snapshot,expected_endpoint=result["paired_endpoint"])
                self.assertEqual(owner(result["capture"]),digest(result["capture"]))
                replays.append(dict(construction_ordinal=descriptor["construction_ordinal"],evaluation_ordinal=descriptor["evaluation_ordinal"],
                    result=result,validator=owner))
            joined=reconcile_owned_replay_diagnostic(selected,mats,replays,native_validator=reader,
                ancestry_contract=ancestry,native_extract=pokezero_search.native_scoped_boundary,
                native_support=pokezero_search.native_edge_support,native_successor_support=pokezero_search.native_scoped_successor_support)
            replay_join=joined["owned_replay_join"]
            self.assertEqual(len(replay_join["records"]),replay_plan.receipt()["required_construction_sample_records"])
            self.assertEqual(replay_join["actual_replay_runs"],len(replays))
            self.assertTrue(any(len(row["result"]["capture"]["paired_contexts"])>1 for row in replays))
            self.assertFalse(joined["runtime_qualified"])
            self.assertFalse(joined["scientific_collection_authorized"])
            print("EXCLUDED_NATIVE_SHOWDOWN:"+json.dumps(dict(accepted_native_units=64,
                constructor_coordinates=4,constructor_runs=len(mats),replay_coordinates=len(replay_join["records"]),
                replay_runs=len(replays),required_boundaries=replay_join["required_boundary_records"],
                field_counts=replay_join["observed_required_field_counts"],
                successor_counts=replay_join.get("projected_successor_support_counts"),
                scientific_measurement=False),sort_keys=True))
