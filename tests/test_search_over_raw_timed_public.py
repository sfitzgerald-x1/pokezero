"""Full public transport with actual Python hooks and disposable authored owners.

No model, scientific source game, production native search or holdout is run.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
import hashlib
import pickle
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_native_timed_job
from pokezero.mcts_eval.search_over_raw_timed_run import (
    NativeTimedAdapterJob,NativeTimedRunContract,NativeTimedRunValidator,read_native_timed_selection)
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    TIMED_NATIVE_PUBLIC_CAPTURE,NativeWorldInputs,WorldInputSearchAdapter,
    public_request_witness,validate_timed_public_request)
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
import test_search_over_raw_archive as archive_fixture
import test_search_over_raw_timed_run as timed_fixture
import test_search_over_raw_timed_native as native_fixture
import test_search_over_raw_root_plan as plan_fixture
import test_search_over_raw_world_inputs as world_fixture


PUBLIC_SHA="d"*64


def capture(binding,sha=PUBLIC_SHA):
    return dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
        public_request_sha256=sha,consumed_binding=deepcopy(binding),
        full_supported_public_state_and_pending=True,private_source_snapshot_included=False,runtime_qualified=False)


def full_selection(cfg,runtime,*,zero=False):
    selected=timed_fixture.selected_fixture(cfg,runtime,zero=zero)
    manifest=selected["evidence"]["consumed_world_inputs"]
    manifest["adapter_public_request_witness"]=capture(world_fixture.BINDING)
    manifest["public_request_observations"]=[dict(ordinal=i,witness=capture(world_fixture.BINDING))
        for i in range(len(manifest["rows"]))]
    return selected


@dataclass(frozen=True)
class AuthoredFullTimedOwner:
    cfg: object
    runtime: dict
    zero: bool=False
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        phase("startup");phase("selection")
        selected=full_selection(self.cfg,self.runtime,zero=self.zero)
        time.sleep(.04);phase("cleanup")
        return selected


@dataclass(frozen=True)
class AuthoredNativeSelectionOwner:
    selected: dict
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        phase("startup");phase("selection");time.sleep(self.selected["elapsed_seconds"]+.02)
        phase("cleanup");return deepcopy(self.selected)


class TimedPublicTests(unittest.TestCase):
    def setUp(self):
        f=archive_fixture.ExplorationRootCrosswalkTests();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.bound=f.bind();self.witness=self.bound["public_request_witness"]
        self.cfg=SearchConfiguration("incumbent",seconds=1.)

    def runtime(self,cfg=None):
        value=timed_fixture.runtime(cfg or self.cfg)
        value["set_source_hash"]=self.witness["consumed_binding"]["set_source_hash"]
        value["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        return value

    def job(self):
        with patch("subprocess.Popen") as launch:
            job,bound=prepare_archived_native_timed_job(self.f.archive,self.f.source,
                contract=self.f.contract,root_id=self.f.root_id,configuration=self.cfg,
                checkpoint_contract=SimpleNamespace(checkpoint_sha256="b"*64,
                    showdown_source_sha256=self.witness["consumed_binding"]["set_source_hash"]),
                showdown_root="authored-never-loaded",runtime_configuration=self.runtime())
        launch.assert_not_called();self.assertEqual(bound,self.bound)
        return job

    def test_actual_archived_job_is_public_only_full_bound_and_no_launch(self):
        job=self.job()
        self.assertEqual(job.expected_public_request_sha256,self.witness["public_request_sha256"])
        self.assertEqual(job._inputs(),job._frozen_inputs)
        self.assertEqual(set(job.context.requested_observations),{"p1"})
        self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
        self.assertNotIn(b"private-source-snapshot",pickle.dumps(job))

    def test_live_public_pp_pending_and_cache_changes_preserving_compact_keys_refuse(self):
        from pokezero.mcts_eval.paper_reference_pending import PendingPolicyTransition
        for mutation in ("pp","pending","cache"):
            job=self.job();before=job._consumed_binding
            if mutation == "pending":
                object.__setattr__(job,"pending_transition",PendingPolicyTransition.capture(job._public_request,0))
            else:
                state=job.context.public_materialization_state
                if mutation == "pp":state=replace(state,self_move_states={"p1:known":({"pp":14},)})
                else:state.belief_engine._authored_public_cache={"extra":"changed"}
                object.__setattr__(job,"context",replace(job.context,public_materialization_state=state))
            witness=public_request_witness(replace(job._public_request,state=job.context.public_materialization_state,
                observation=job.context.observation,pending_transition=job.pending_transition))
            self.assertEqual(witness["consumed_binding"],before)
            self.assertNotEqual(job._inputs(),job._frozen_inputs)
            with self.assertRaisesRegex(ValueError,"public inputs changed"):job.read_selection({})

    def test_actual_hooks_variable_roster_and_no_attempt_capture(self):
        for count in (0,1,3):
            recorder=NativeWorldInputs(self.witness["consumed_binding"],4096,
                expected_public_request=self.witness,record_atomic_work=True,record_timed_atomic_work=True)
            for ordinal in range(count):
                recorder.attempt(self.f.context,ordinal,world_fixture.FakeOverride(dict(p1="own",p2="hypothetical")))
                recorder.constructed(self.f.context,world_fixture.world(),SimpleNamespace(to_string=lambda:"complete-state"))
            evidence=dict(consumed_world_inputs=recorder.finish(dict(total_iterations=0),
                attempted=count,constructed=count,searched=0))
            readout=validate_timed_public_request(evidence,binding=self.witness["consumed_binding"],
                expected_sha256=self.witness["public_request_sha256"])
            self.assertEqual(readout["observed_attempts"],count)
            self.assertTrue(readout["adapter_capture_validated"])
            self.assertFalse(readout["runtime_qualified"])
            evidence["consumed_world_inputs"].pop("adapter_public_request_witness")
            with self.assertRaises(ValueError):validate_timed_public_request(evidence,
                binding=self.witness["consumed_binding"],expected_sha256=self.witness["public_request_sha256"])

    def test_actual_adapter_zero_return_emits_capture_and_all_sampling_witnesses(self):
        f=native_fixture.StrictZeroWorkSelectorTests();f.setUp();self.addCleanup(f.doCleanups)
        adapter=f.adapter()
        # Same already initialized prospective adapter; only inference/native
        # behavior is authored by this fixture. Capture/hooks/select are real.
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        witness=public_request_witness(PublicRootRequest.capture(f.context.public_materialization_state,f.context.observation))
        adapter._expected_public_request_sha256=witness["public_request_sha256"]
        adapter.runtime_configuration["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        adapter.runtime_sha256=digest(adapter.runtime_configuration)
        selected=f.select(adapter)
        readout=validate_timed_public_request(selected["evidence"],binding=witness["consumed_binding"],
            expected_sha256=witness["public_request_sha256"])
        self.assertEqual(readout["observed_attempts"],4)
        self.assertIsNone(selected["action"])

    def no_attempt_selection(self):
        from pokezero.engine_search import EngineSearchFallbackError
        f=native_fixture.StrictZeroWorkSelectorTests();f.setUp();self.addCleanup(f.doCleanups)
        f.context=self.f.context;adapter=f.adapter()
        adapter._expected_public_request_sha256=self.witness["public_request_sha256"]
        adapter.runtime_configuration["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        adapter.runtime_sha256=digest(adapter.runtime_configuration)
        def no_attempt(context,rng):
            raise EngineSearchFallbackError("authored exhausted capture",diagnostic=dict(
                schema="engine-search-fallback-refusal-v1",diagnostic_only_not_policy_input=True,
                battle_id=context.battle_id,round=context.decision_round_index,seat=context.player_id,
                reason="model_time_budget_no_completed_worlds",world_failures={},
                engine_mcts=dict(fallback="model_time_budget_no_completed_worlds",time_budget=dict(
                    scope="whole_model_decision",requested_ms=1000,native_batch_guard_ms=64,
                    deadline_elapsed_ms=1001.,deadline_overshoot_ms=1.,exhausted=True,
                    worlds_budget_skipped=0,native_invocations=[]))))
        adapter._native.select_action_with_context=no_attempt
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs.time.perf_counter",side_effect=[100.,101.2]):
            selected=adapter.select(self.f.context,root_id=self.f.root_id+":"+self.cfg.identity,
                statistics_root_id=self.f.root_id,selection_seed=self.bound["slot"]["source_seed"],
                pending_transition=getattr(self.f,"pending",None))
        return selected

    def test_actual_no_attempt_adapter_capture_and_job_dispatch(self):
        selected=self.no_attempt_selection()
        manifest=selected["evidence"]["consumed_world_inputs"]
        self.assertEqual(manifest["rows"],[]);self.assertEqual(manifest["public_request_observations"],[])
        self.assertEqual(manifest["adapter_public_request_witness"],self.witness)
        job=self.job()
        # Actual job method with a mock native/model loader; selected input
        # capture and zero classification above came from the actual adapter.
        wiring=timed_fixture.NativeTimedJobWiringTests();adapter=wiring.adapter(job,True)
        selected.update(runtime_sha256=job.expected_runtime_sha256,
            observed_runtime_configuration=deepcopy(job.expected_runtime_configuration),
            native_timed_run_contract=job.contract.receipt())
        adapter.select.return_value=selected
        source=SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=source), \
                patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter",return_value=adapter) as factory:
            result=job({},lambda p:None)
        self.assertEqual(factory.call_args.kwargs["expected_public_request_sha256"],self.witness["public_request_sha256"])
        self.assertEqual(job.read_selection(result)["public_request_consumption"]["observed_attempts"],0)
        adapter.close.assert_called_once()

    def test_owner_reads_productive_and_zero_full_requests_and_rejects_late_hash(self):
        f=timed_fixture.NativeTimedOwnerTests();f.setUp();self.addCleanup(f.doCleanups)
        runtime=deepcopy(f.runtime);runtime["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        for zero in (False,True):
            job=AuthoredFullTimedOwner(f.cfg,runtime,zero);directory=f.root/str(zero)
            args=dict(configuration=f.cfg,runtime_configuration=runtime,runtime_sha256=digest(runtime),
                root_id="root:"+f.cfg.identity,statistics_root_id="root",selection_seed=7,
                consumed_binding=world_fixture.BINDING,legal_action_mask=timed_fixture.MASK,decision_round=3,
                contract=NativeTimedRunContract(),public_request_sha256=PUBLIC_SHA)
            terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,
                verify_admission=lambda:None,contract=NativeTimedRunContract(),kind="native-timed-run",
                work_phase="selection",validate_result=lambda row:read_native_timed_selection(row,**args))
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            owner_args=dict(args);owner_args.update(directory=directory,
                job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256="a"*64,
                file_hashes=f.files,checkpoint_path=str(f.checkpoint),checkpoint_sha256=f.files[str(f.checkpoint)],
                showdown_root=str(f.showdown),allow_synthetic=True)
            reader=NativeTimedRunValidator(**owner_args)
            self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
            self.assertTrue(reader.last_validation["actual_work"]["full_public_request_consumption_validated"])
            self.assertFalse(reader.last_validation["runtime_qualified"])
            reader.timed_public_request_sha="e"*64
            with self.assertRaisesRegex(ValueError,"reader inputs changed"):reader(terminal["result"])

    def test_missing_duplicate_coerced_or_stale_capture_refuses_on_both_statuses(self):
        runtime=timed_fixture.runtime(self.cfg);runtime["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        for zero in (False,True):
            for mutation in ("capture","hash","missing","duplicate","bool","pending"):
                selected=full_selection(self.cfg,runtime,zero=zero);manifest=selected["evidence"]["consumed_world_inputs"]
                if mutation=="capture":manifest.pop("adapter_public_request_witness")
                if mutation=="hash":manifest["adapter_public_request_witness"]["public_request_sha256"]="e"*64
                if mutation=="missing":manifest["public_request_observations"].pop()
                if mutation=="duplicate":manifest["public_request_observations"].append(deepcopy(manifest["public_request_observations"][0]))
                if mutation=="bool":manifest["public_request_observations"][0]["ordinal"]=False
                if mutation=="pending":manifest["public_request_observations"][2]["witness"]["public_request_sha256"]="e"*64
                with self.assertRaises(ValueError):read_native_timed_selection(selected,configuration=self.cfg,
                    runtime_configuration=runtime,runtime_sha256=digest(runtime),root_id="root:"+self.cfg.identity,
                    statistics_root_id="root",selection_seed=7,consumed_binding=world_fixture.BINDING,
                    legal_action_mask=timed_fixture.MASK,decision_round=3,contract=NativeTimedRunContract(),
                    public_request_sha256=PUBLIC_SHA)

    def test_declared_full_scope_cannot_use_compact_only_reader(self):
        runtime=timed_fixture.runtime(self.cfg);runtime["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        with self.assertRaisesRegex(ValueError,"externally bound request hash"):
            timed_fixture.read_selection(full_selection(self.cfg,runtime),self.cfg,runtime)
        with self.assertRaises(ValueError):NativeTimedAdapterJob(self.cfg,
            SimpleNamespace(checkpoint_sha256="b"*64,showdown_source_sha256="a"*64),"unused",
            self.f.context,self.f.root_id+":"+self.cfg.identity,0,self.f.root_id,self.runtime())

    def test_actual_archive_plan_reports_native_correspondence_per_selector_without_qualification(self):
        f=plan_fixture.RootPlanTests();f.setUp();self.addCleanup(f.doCleanups)
        self.assertEqual(f.bound["public_request_witness"],self.witness)
        selected=self.no_attempt_selection();o=f.owner_fixture
        runtime=self.runtime();runtime["checkpoint_sha256"]=o.files[str(o.checkpoint)]
        selected.update(runtime_sha256=digest(runtime),observed_runtime_configuration=runtime,
            native_timed_run_contract=NativeTimedRunContract().receipt())
        job=AuthoredNativeSelectionOwner(selected);directory=o.root/"full-native-owner"
        reader=NativeTimedRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,configuration=self.cfg,root_id=selected["root_id"],
            statistics_root_id=f.archive_fixture.root_id,selection_seed=selected["selection_seed"],
            runtime_configuration=runtime,runtime_sha256=digest(runtime),
            consumed_binding=self.witness["consumed_binding"],legal_action_mask=f.mask,decision_round=0,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=self.witness["public_request_sha256"])
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
            contract=NativeTimedRunContract(),kind="native-timed-run",work_phase="selection",validate_result=reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        plan=f.plan(configurations=(f.cfg,self.cfg),selections={"raw":f.selected,self.cfg.identity:selected},
            selector_readers={"raw":f.reader,self.cfg.identity:reader})
        receipt=plan.revalidate()
        self.assertEqual(receipt["full_public_correspondence_by_selector"],{"raw":True,self.cfg.identity:True})
        self.assertTrue(receipt["full_timed_public_correspondence_validated"])
        self.assertFalse(receipt["runtime_qualified"]);self.assertFalse(receipt["collection_authorized"])
        self.assertEqual(receipt["root_evidence"]["root_intervals"][self.cfg.identity],[-1.,1.])
        self.assertEqual(len(receipt["continuation_cells"]),8) # raw only; no zero-work continuation
        original=reader.timed_public_request_sha;reader.timed_public_request_sha="e"*64
        with self.assertRaises(ValueError):plan.revalidate()
        reader.timed_public_request_sha=original


if __name__ == "__main__":unittest.main()
