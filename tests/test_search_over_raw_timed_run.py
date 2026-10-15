"""Actual disposable owners and authored native reports, never scientific runs."""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_timed_run import (
    NativeTimedRunContract,NativeTimedAdapterJob,NativeTimedRunValidator,read_native_timed_selection,
)
from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
from pokezero.mcts_eval.search_over_raw_world_inputs import NativeWorldInputs,NativeInputHandle,validate_timed_native_inputs
from test_search_over_raw_timed_native import timed_arguments,timed_report
from test_search_over_raw_supervision import SyntheticPublicState,SyntheticBelief
from test_search_over_raw_world_inputs import BINDING,construct,native_manifest
from test_policy_opponent_profile import telemetry,MASK

RUN_BINDING="a"*64


def runtime(configuration,checkpoint_sha="b"*64, *, compact=False):
    value=dict(configuration=asdict(configuration),checkpoint_sha256=checkpoint_sha,
        set_source_hash=BINDING["set_source_hash"],root_statistics="independent_per_root",
        incumbent=dict(depth=6,sims=4096,batch=16,worlds=4,native_batch_guard_ms=64),
        timed_native_atomic_work=dict(schema="pokezero.search-over-raw.timed-native-prefix.v1",
            native_budget_abi_slot=24,zero_work_not_scored=True,on_time_deployability_established=False,
            runtime_qualified=False),
        consumed_world_inputs=dict(schema="pokezero.search-over-raw.consumed-world-inputs.v1",initial_inputs_only=True,
            snapshot_fields_excluded=[],ownership_metadata_excluded=[],instrumentation_can_change_deadlines=True,
            qualifies_uninstrumented_runtime=False),
        native_atomic_work=dict(schema="native-atomic-backups-v1",
            scope="observed_float32_backup_consistency; not independent terminal labels",
            instrumentation_can_change_deadlines=True,runtime_qualified=False))
    if configuration.leaf=="hp_fraction":
        value["incumbent_leaf"]=dict(leaf="hp_fraction",tree="unchanged_encoded_model_tree",
            priors="unchanged_champion",model_forwards="retained",value_frame="side_one_absolute")
    if compact:
        from pokezero.mcts_eval.search_over_raw_world_inputs import native_atomic_runtime_scopes
        value["compact_native_atomic_work"] = True
        value["native_atomic_work"], value["timed_native_atomic_work"] = native_atomic_runtime_scopes(compact=True)
    return value


def selected_fixture(configuration,observed_runtime,*,zero=False,elapsed=.02,decision_round=3,compact=False,forced=False):
    recorder=NativeWorldInputs(BINDING,4096,record_atomic_work=True,record_timed_atomic_work=True,
        record_compact_atomic_work=compact)
    for _ in range(4):construct(recorder)
    report=timed_report(0 if zero else 16,requested=16384)
    report["model_evals"]=1
    if configuration.leaf=="hp_fraction":
        report.update(model_leaf_override="hp_fraction",hp_leaf_rows_priced=1,model_evals=1,
            hp_leaf_value_frame="side_one_absolute",hp_leaf_model_forwards_retained=True)
    if compact:
        from test_search_over_raw_native_atomic_compact import compact_fixture
        report = compact_fixture(report)
    NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a,**kw:json.dumps(report)),recorder) \
        .search_batched_multi_encoded(*timed_arguments(requested=16384,
            leaf="hp_fraction" if configuration.leaf=="hp_fraction" else None))
    evidence=telemetry("incumbent_mcts","matched_deadline",7)
    evidence.update(total_iterations=0 if zero else 16,root_action="action:0")
    evidence["consumed_world_inputs"]=native_manifest(recorder,0 if zero else 16)
    engine=evidence["engine_mcts"]
    if forced:
        engine["override"]["root_allocation"]["arms"]=[dict(action_index=0,
            visit_share=1.,model_prior=1.,reported_prior=1.)]
    engine["aggregated_choices_basis"]="deadline_prefix"
    engine["joint_actions"]["native_invocations"]=[dict(world_seed=7,belief_multiplicity=4,
        witness=deepcopy(report["joint_action_witness"]))]
    engine["time_budget"]=dict(scope="whole_model_decision",requested_ms=round(configuration.seconds*1000),
        native_batch_guard_ms=64,deadline_elapsed_ms=9.,deadline_overshoot_ms=0.,exhausted=True,
        worlds_budget_skipped=4 if zero else 0,native_invocations=[dict(status="completed",world_seed=7,
            multiplicity=4,requested_iterations=16384,completed_iterations=0 if zero else 16,
            remaining_iterations=16384 if zero else 16368,time_budget_ms=8,time_budget_elapsed_ms=8.5,
            time_budget_batch_overshoot_ms=.5,time_budget_exhausted=True,
            root_visits=dict(side_one=0 if zero else 16,side_two=0 if zero else 16))])
    if configuration.leaf=="hp_fraction":
        engine["model_leaf_override"]=dict(schema="pokezero.model-tree-hp-leaf.v1",mode="hp_fraction",
            value_frame="side_one_absolute",model_forwards_retained=True,
            scope="per_native_invocation_without_belief_reweighting",hp_leaf_rows_priced=1,
            native_invocations=[dict(world_seed=7,belief_multiplicity=4,completed_iterations=16,
                model_evals=1,hp_leaf_rows_priced=1)])
    selected=dict(root_id="root:"+configuration.identity,statistics_root_id="root",selection_seed=7,
        runtime_sha256=digest(observed_runtime),configuration_sha256=configuration.identity,
        native_timed_run_contract=NativeTimedRunContract().receipt(),
        observed_runtime_configuration=deepcopy(observed_runtime),
        action=None if zero else 0,status="UNQUALIFIED_AT_CEILING" if zero else "SELECTED",
        elapsed_seconds=elapsed,nominal_search_seconds=configuration.seconds,
        exceeded_nominal_seconds=elapsed>configuration.seconds,evidence=evidence,
        independent_root_statistics=True,scientific_strength_evidence=False)
    if zero:
        evidence=dict(total_iterations=0,engine_mcts=dict(fallback="model_time_budget_no_completed_worlds",
            time_budget=deepcopy(engine["time_budget"])),consumed_world_inputs=evidence["consumed_world_inputs"])
        diagnostic=dict(schema="engine-search-fallback-refusal-v1",diagnostic_only_not_policy_input=True,
            reason="model_time_budget_no_completed_worlds",battle_id="search-over-raw-root:root",
            seat=BINDING["subject"],round=decision_round,world_failures={},engine_mcts=deepcopy(evidence["engine_mcts"]))
        selected.update(evidence=evidence,strict_engine_diagnostic=diagnostic,
            original_failure_receipt=dict(root_id=selected["root_id"],error_type="EngineSearchFallbackError",
                status="UNCERTAIN_REFUSED",retry_authorized=False,engine_fallback_diagnostic=deepcopy(diagnostic)),
            attempt_consumed=True,retry_authorized=False,continuation_authorized=False,owning_process_validated=False,
            runtime_qualified=False,scientific_collection_authorized=False,
            timed_input_readout=validate_timed_native_inputs(evidence,binding=BINDING,iterations_per_world=4096,
                batch_size=16,expected_worlds=4,nominal_seconds=configuration.seconds,adapter_elapsed_seconds=elapsed,
                compact_native_atomic_work=compact))
    return selected


def read_selection(selected,configuration,observed_runtime,*,mask=MASK):
    return read_native_timed_selection(selected,configuration=configuration,runtime_configuration=observed_runtime,
        runtime_sha256=digest(observed_runtime),root_id="root:"+configuration.identity,
        statistics_root_id="root",selection_seed=7,consumed_binding=BINDING,legal_action_mask=mask,
        decision_round=3,contract=NativeTimedRunContract())


@dataclass(frozen=True)
class AuthoredTimedJob:
    configuration: object
    observed_runtime: dict
    zero: bool=False
    behavior: str="normal"
    adapter_elapsed: float=.02
    forced: bool=False
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        phase("startup");phase("selection")
        if self.behavior=="failure":raise RuntimeError("authored worker failure")
        if self.behavior=="timeout":time.sleep(2.)
        selected=selected_fixture(self.configuration,self.observed_runtime,zero=self.zero,elapsed=self.adapter_elapsed,
            compact=self.observed_runtime.get("compact_native_atomic_work") is True,forced=self.forced)
        time.sleep(self.adapter_elapsed+.01)
        phase("cleanup")
        if self.behavior=="cleanup":raise RuntimeError("authored cleanup failure")
        return selected


class NativeTimedReadoutTests(unittest.TestCase):
    def test_forced_positive_and_zero_work_have_distinct_truthful_status(self):
        mask=(True,)+(False,)*8
        for leaf in ("model","hp_fraction"):
            cfg=SearchConfiguration("incumbent",leaf=leaf,seconds=1.);observed=runtime(cfg)
            for zero in (False,True):
                selected=selected_fixture(cfg,observed,zero=zero,forced=True)
                result=read_selection(selected,cfg,observed,mask=mask)
                self.assertEqual(result["accepted_units"],0 if zero else 16)
                self.assertEqual(selected["action"],None if zero else 0)
                self.assertEqual(selected["status"],"UNQUALIFIED_AT_CEILING" if zero else "SELECTED")
                forged=deepcopy(selected)
                if zero:forged.update(status="SELECTED",action=0)
                else:forged["evidence"]["total_iterations"]=0
                with self.assertRaises((ValueError,RuntimeError)):read_selection(forged,cfg,observed,mask=mask)

    def test_empty_and_nonboolean_timed_masks_remain_rejected(self):
        cfg=SearchConfiguration("incumbent",seconds=1.);observed=runtime(cfg)
        for mask in ((False,)*9,(1,)+(False,)*8):
            with self.subTest(mask=mask),self.assertRaises(ValueError):
                read_selection(selected_fixture(cfg,observed,forced=True),cfg,observed,mask=mask)

    def test_distinct_contract_and_supported_configs_not_fixed_work(self):
        for values in (dict(iterations_per_world=16),dict(batch_size=True),dict(worlds=20),
                dict(emergency_watchdog_seconds=30)):
            with self.assertRaises(ValueError):NativeTimedRunContract(**values)
        for cfg in (SearchConfiguration("reference",seconds=1.,workers=20),
                SearchConfiguration("incumbent",seconds=2.),SearchConfiguration("incumbent",belief="oracle"),
                SearchConfiguration("incumbent",leaf="raw_rollout"),SearchConfiguration("incumbent",workers=2)):
            with self.assertRaises(ValueError):read_selection({},cfg,runtime(cfg))

    def test_positive_zero_and_hp_for_all_ceilings_preserve_scope(self):
        for seconds in (1.,3.,10.):
            for leaf,zero in (("model",False),("model",True),("hp_fraction",False),("hp_fraction",True)):
                cfg=SearchConfiguration("incumbent",leaf=leaf,seconds=seconds);observed=runtime(cfg)
                selected=selected_fixture(cfg,observed,zero=zero)
                result=read_selection(selected,cfg,observed)
                self.assertEqual(result["accepted_units"],0 if zero else 16)
                self.assertEqual(result["observed_native_model_evaluation_rows"],1)
                self.assertFalse(result["runtime_qualified"])
                self.assertFalse(result["scoring_authorized"])

    def test_zero_status_and_provenance_forgery_rejected(self):
        cfg=SearchConfiguration("incumbent",seconds=1.);observed=runtime(cfg)
        edits=[lambda s:s.update(status="SELECTED"),lambda s:s.update(action=0),lambda s:s.pop("action"),
            lambda s:s["strict_engine_diagnostic"].update(round=False),
            lambda s:s["strict_engine_diagnostic"].update(seat="p2"),
            lambda s:s["strict_engine_diagnostic"].update(world_failures={"operational":1}),
            lambda s:s["original_failure_receipt"].update(error_type="RuntimeError"),
            lambda s:s["original_failure_receipt"].pop("engine_fallback_diagnostic"),
            lambda s:s.update(attempt_consumed=False),lambda s:s.update(retry_authorized=True),
            lambda s:s["timed_input_readout"].update(completed_iterations=16)]
        for edit in edits:
            selected=selected_fixture(cfg,observed,zero=True);edit(selected)
            with self.assertRaises(ValueError):read_selection(selected,cfg,observed)

    def test_productive_selector_and_actual_joint_witness_checked(self):
        cfg=SearchConfiguration("incumbent",seconds=1.);observed=runtime(cfg)
        edits=[lambda s:s.update(action=True),lambda s:s.update(action=8),lambda s:s.update(status="UNQUALIFIED_AT_CEILING"),
            lambda s:s["evidence"].update(root_action="action:1"),
            lambda s:s["evidence"].update(fallbacks=1),
            lambda s:s["evidence"]["engine_mcts"].pop("joint_actions"),
            lambda s:s["evidence"]["engine_mcts"]["joint_actions"]["native_invocations"][0].update(world_seed=8)]
        for edit in edits:
            selected=selected_fixture(cfg,observed);edit(selected)
            with self.assertRaises((ValueError,RuntimeError)):read_selection(selected,cfg,observed)

    def test_runtime_receipt_and_actual_hp_leaf_join_not_caller_assertions(self):
        cfg=SearchConfiguration("incumbent",seconds=1.,leaf="hp_fraction");observed=runtime(cfg)
        selected=selected_fixture(cfg,observed)
        selected["evidence"]["engine_mcts"]["model_leaf_override"]["native_invocations"][0]["world_seed"]=8
        with self.assertRaisesRegex(ValueError,"HP selector witness"):read_selection(selected,cfg,observed)
        for edit in (lambda r:r["incumbent"].update(sims=16),
                lambda r:r["timed_native_atomic_work"].update(runtime_qualified=True),
                lambda r:r["incumbent_leaf"].update(model_forwards="not-retained"),
                lambda r:r.update(fixed_work_diagnostic={})):
            changed=deepcopy(observed);edit(changed)
            with self.assertRaises(ValueError):read_selection(selected,cfg,changed)

    def test_zero_raw_leaf_identity_cannot_be_forged_by_rehashing_the_report(self):
        for leaf in ("model","hp_fraction"):
            cfg=SearchConfiguration("incumbent",seconds=1.,leaf=leaf);observed=runtime(cfg)
            selected=selected_fixture(cfg,observed,zero=True)
            call=selected["evidence"]["consumed_world_inputs"]["calls"][0]
            if leaf=="model":call["native_report"]["model_leaf_override"]="hp_fraction"
            else:call["native_report"]["hp_leaf_model_forwards_retained"]=False
            call["native_report_utf8"]=json.dumps(call["native_report"])
            call["report_sha256"]=digest(call["native_report"])
            call["report_utf8_sha256"]=hashlib.sha256(call["native_report_utf8"].encode()).hexdigest()
            # The cached arithmetic readout is unchanged: leaf truth must still be read anew.
            with self.assertRaises((ValueError,RuntimeError)):read_selection(selected,cfg,observed)

    def test_model_row_count_binds_actual_calls_not_multiplicity_or_accepted_work(self):
        cfg=SearchConfiguration("incumbent",seconds=1.);observed=runtime(cfg)
        for zero in (False,True):
            selected=selected_fixture(cfg,observed,zero=zero)
            self.assertEqual(read_selection(selected,cfg,observed)["observed_native_model_evaluation_rows"],1)
            call=selected["evidence"]["consumed_world_inputs"]["calls"][0]
            call["native_report"]["model_evals"]=True
            call["native_report_utf8"]=json.dumps(call["native_report"])
            call["report_sha256"]=digest(call["native_report"])
            call["report_utf8_sha256"]=hashlib.sha256(call["native_report_utf8"].encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,"row count malformed"):read_selection(selected,cfg,observed)
        selected=selected_fixture(cfg,observed);selected["evidence"]["model_evals"]=4
        with self.assertRaisesRegex(ValueError,"differ from actual"):read_selection(selected,cfg,observed)


@dataclass(frozen=True)
class TimedContext:
    player_id: str
    observation: object
    requested_observations: dict
    public_materialization_state: object
    decision_round_index: int=3


class NativeTimedJobWiringTests(unittest.TestCase):
    def job(self,leaf="model",seconds=1.,mask=MASK):
        cfg=SearchConfiguration("incumbent",leaf=leaf,seconds=seconds)
        observed=runtime(cfg);state=SyntheticPublicState(SyntheticBelief())
        observation=SimpleNamespace(legal_action_mask=mask)
        context=TimedContext("p1",observation,{"p1":observation,"p2":"private-canary"},state)
        request=SimpleNamespace(state=state,observation=observation,set_source_hash=BINDING["set_source_hash"])
        with patch("pokezero.mcts_eval.head_to_head.public_only_context",side_effect=lambda c:c), \
                patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=request), \
                patch("pokezero.mcts_eval.search_over_raw_timed_run._binding",return_value=BINDING):
            return NativeTimedAdapterJob(cfg,SimpleNamespace(showdown_source_sha256=BINDING["set_source_hash"],
                checkpoint_sha256="b"*64),"authored-showdown",context,"root:"+cfg.identity,7,"root",observed)

    def test_forced_job_does_not_filter_or_replace_the_sampled_root(self):
        job=self.job(mask=(True,)+(False,)*8)
        self.assertEqual((job.root_id,job.statistics_root_id,job.selection_seed),
            ("root:"+job.configuration.identity,"root",7))
        self.assertEqual(job._legal_action_mask,(True,)+(False,)*8)

    def adapter(self,job,zero=False):
        cfg=SimpleNamespace(search_depth=6,search_sims=4096,search_batch=16,worlds=4,model_world_workers=1,
            model_decision_time_ms=round(job.configuration.seconds*1000),model_native_batch_guard_ms=64,
            early_stop=False,strict_fallbacks=True,model_priors=True,record_joint_actions=True,
            policy_opponent=False,leaf_eval="model",model_leaf_override="hp_fraction"
                if job.configuration.leaf=="hp_fraction" else None)
        adapter=Mock(runtime_configuration=deepcopy(job.expected_runtime_configuration),
            runtime_sha256=job.expected_runtime_sha256,_native=SimpleNamespace(_config=cfg,_ladder_sims_override=None))
        adapter.select.return_value=selected_fixture(job.configuration,job.expected_runtime_configuration,zero=zero)
        return adapter

    def dispatch(self,job,adapter,phases):
        source=SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING["set_source_hash"]))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=source), \
                patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter",return_value=adapter) as factory:
            result=job({},phases.append)
        self.assertTrue(factory.call_args.kwargs["timed_native_atomic_work"])
        self.assertNotIn("fixed_work",factory.call_args.kwargs)
        self.assertIs(adapter.select.call_args.args[0].public_materialization_state.belief_engine.set_source,source)
        self.assertEqual(set(adapter.select.call_args.args[0].requested_observations),{"p1"})
        self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
        return result

    def test_actual_job_closes_both_normal_branches_and_retains_complete_receipt(self):
        for leaf in ("model","hp_fraction"):
            for zero in (False,True):
                job=self.job(leaf);adapter=self.adapter(job,zero);phases=[]
                result=self.dispatch(job,adapter,phases)
                self.assertEqual(phases,["startup","selection","cleanup"])
                adapter.close.assert_called_once()
                self.assertEqual(job.read_selection(result)["accepted_units"],0 if zero else 16)
                self.assertEqual(result["status"],"UNQUALIFIED_AT_CEILING" if zero else "SELECTED")

    def test_actual_runtime_allocation_and_selector_failure_still_close(self):
        for mutation in ("allocation","depth","runtime","earlystop","guard","leaf","priors","failure","malformed","late-runtime"):
            job=self.job();adapter=self.adapter(job);phases=[]
            if mutation=="allocation":adapter._native._config.search_sims=16
            if mutation=="depth":adapter._native._config.search_depth=2
            if mutation=="runtime":adapter.runtime_configuration["incumbent"]["worlds"]=3
            if mutation=="earlystop":adapter._native._config.early_stop=True
            if mutation=="guard":adapter._native._config.model_native_batch_guard_ms=0
            if mutation=="leaf":adapter._native._config.model_leaf_override="hp_fraction"
            if mutation=="priors":adapter._native._config.model_priors=False
            if mutation=="failure":adapter.select.side_effect=RuntimeError("authored operational failure")
            if mutation=="malformed":adapter.select.return_value["evidence"]["fallbacks"]=1
            if mutation=="late-runtime":
                def late(*args,**kwargs):
                    adapter.runtime_configuration["incumbent"]["sims"]=16
                    return adapter.select.return_value
                adapter.select.side_effect=late
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,RuntimeError)):self.dispatch(job,adapter,phases)
            self.assertEqual(phases[-1],"cleanup");adapter.close.assert_called_once()

    def test_close_failure_is_operational_not_zero_work_rescue(self):
        job=self.job();adapter=self.adapter(job,True);adapter.close.side_effect=RuntimeError("authored close failure")
        with self.assertRaisesRegex(RuntimeError,"close failure"):self.dispatch(job,adapter,[])

    def test_cleanup_failure_preserves_the_original_selector_exception(self):
        job=self.job();adapter=self.adapter(job,True)
        original=RuntimeError("authored primary selection failure")
        adapter.select.side_effect=original;adapter.close.side_effect=ValueError("private cleanup details")
        with self.assertRaises(RuntimeError) as caught:self.dispatch(job,adapter,[])
        self.assertIs(caught.exception,original)
        self.assertEqual(caught.exception.__notes__,["Native timed adapter cleanup failed: ValueError"])

    def test_wrapper_keeps_distinct_law_and_never_infers_scientific_authority(self):
        from pokezero.mcts_eval.search_over_raw_timed_run import supervise_native_timed
        job=self.job()
        with patch("pokezero.mcts_eval.search_over_raw_timed_run._supervise_owned_job",return_value={}) as owner:
            supervise_native_timed(job,directory="never-created",binding_sha256=RUN_BINDING,verify_admission=lambda:False)
        self.assertEqual(owner.call_args.kwargs["kind"],"native-timed-run")
        self.assertEqual(owner.call_args.kwargs["work_phase"],"selection")
        self.assertIs(owner.call_args.kwargs["contract"],job.contract)
        self.assertEqual(owner.call_args.kwargs["validate_result"],job.read_selection)
        with self.assertRaises(ValueError):supervise_native_timed(AuthoredTimedJob(job.configuration,
            job.expected_runtime_configuration),directory="never-created",binding_sha256=RUN_BINDING,verify_admission=lambda:True)


class NativeTimedFileInventoryTests(unittest.TestCase):
    def setUp(self):
        fixture=NativeTimedOwnerTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture
        self.reader=object.__new__(NativeTimedRunValidator)
        self.reader.files=fixture.files
        self.reader.showdown_root=str(fixture.showdown)
        self.reader.checkpoint_path=str(fixture.checkpoint)
        self.reader.checkpoint_sha=fixture.files[str(fixture.checkpoint)]

    def test_each_declared_file_is_read_once_per_invocation(self):
        original=Path.read_bytes;reads={}
        def counted(path):
            name=str(path);reads[name]=reads.get(name,0)+1
            return original(path)
        with patch.object(Path,"read_bytes",counted):self.reader._files()
        self.assertEqual(reads,{path:1 for path in self.reader.files})
        with patch.object(Path,"read_bytes",counted):self.reader._files()
        self.assertEqual(reads,{path:2 for path in self.reader.files})

    def test_fresh_inventory_addition_deletion_and_required_file_loss(self):
        added=self.fixture.showdown/"dist/sim/new.js";added.write_text("authored new dependency")
        with self.assertRaisesRegex(ValueError,"file byte drift"):self.reader._files()
        self.reader.files[str(added)]=hashlib.sha256(added.read_bytes()).hexdigest()
        self.reader._files()
        added.unlink()
        with self.assertRaises(FileNotFoundError):self.reader._files()
        del self.reader.files[str(added)]
        (self.fixture.showdown/"dist/sim/state.js").unlink()
        with self.assertRaisesRegex(ValueError,"simulator binding"):self.reader._files()

    def test_new_call_rejects_changed_bytes_and_cannot_adopt_changed_checkpoint(self):
        self.reader._files()
        self.fixture.checkpoint.write_bytes(b"changed after prior validation")
        with self.assertRaisesRegex(ValueError,"file byte drift"):self.reader._files()
        self.reader.files[str(self.fixture.checkpoint)]=hashlib.sha256(self.fixture.checkpoint.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,"file byte drift"):self.reader._files()

    def test_same_bytes_at_symlink_path_cannot_substitute_canonical_binding(self):
        alias=self.fixture.root/"checkpoint-alias";alias.symlink_to(self.fixture.checkpoint)
        self.reader.files[str(alias)]=self.reader.checkpoint_sha
        with self.assertRaisesRegex(ValueError,"file byte drift"):self.reader._files()

    def test_path_inventory_preserves_all_controller_bindings(self):
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes,controller_input_paths
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_paths
        controllers=controller_input_hashes()
        self.assertEqual(set(controllers),{str(path) for path in controller_input_paths()})
        self.assertLessEqual(set(controllers),fixed_work_reader_source_paths())
        self.assertTrue(all(self.reader.files[path] == sha for path,sha in controllers.items()))


class NativeTimedOwnerTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.checkpoint=self.root/"not-a-model"
        self.checkpoint.write_bytes(b"unused authored checkpoint")
        self.showdown=self.root/"not-a-simulator";dist=self.showdown/"dist/sim";dist.mkdir(parents=True)
        for name in ("index.js","state.js"):(dist/name).write_text("// unused authored simulator")
        self.files=fixed_work_reader_source_hashes()
        for path in [self.checkpoint,Path(sys.executable).resolve(),*dist.glob("*.js")]:
            self.files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.cfg=SearchConfiguration("incumbent",seconds=1.)
        self.runtime=runtime(self.cfg,self.files[str(self.checkpoint)])

    def run_fixture(self,zero=False,behavior="normal",directory=None,adapter_elapsed=.02,forced=False):
        job=AuthoredTimedJob(self.cfg,self.runtime,zero,behavior,adapter_elapsed,forced)
        directory=directory or self.root/"attempt"
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256=RUN_BINDING,
            verify_admission=lambda:None,contract=NativeTimedRunContract(),kind="native-timed-run",work_phase="selection",
            validate_result=lambda s:read_selection(s,self.cfg,self.runtime,mask=(True,)+(False,)*8 if forced else MASK),
            _synthetic_watchdog_seconds=.5 if behavior=="timeout" else None)
        inputs=dict(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=RUN_BINDING,
            configuration=self.cfg,root_id="root:"+self.cfg.identity,statistics_root_id="root",selection_seed=7,
            runtime_configuration=self.runtime,runtime_sha256=digest(self.runtime),consumed_binding=BINDING,
            legal_action_mask=(True,)+(False,)*8 if forced else MASK,decision_round=3,file_hashes=self.files,checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.files[str(self.checkpoint)],showdown_root=str(self.showdown))
        return terminal,inputs

    def test_forced_owned_positive_and_zero_returns_keep_real_work_accounting(self):
        for zero in (False,True):
            terminal,inputs=self.run_fixture(zero=zero,forced=True,directory=self.root/str(zero))
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
            self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],0 if zero else 16)

    def load(self,inputs,name):return json.loads((inputs["directory"]/name).read_bytes())
    def write(self,inputs,name,value):(inputs["directory"]/name).write_text(json.dumps(value))

    def test_owned_compact_positive_and_zero_original_receipts_require_frozen_format(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import native_atomic_runtime_scopes
        self.runtime["compact_native_atomic_work"] = True
        self.runtime["native_atomic_work"], self.runtime["timed_native_atomic_work"] = native_atomic_runtime_scopes(compact=True)
        for zero in (False, True):
            terminal, inputs = self.run_fixture(zero=zero, directory=self.root/("compact-zero" if zero else "compact-positive"))
            self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
            self.assertEqual(terminal["status"], "NATIVE_TIMED_RUN_COMPLETE")
            selected = terminal["result"]
            reader = NativeTimedRunValidator(**inputs, allow_synthetic=True)
            self.assertEqual(reader(selected), digest(selected))
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"], 0 if zero else 16)
            self.assertFalse(reader.last_validation["runtime_qualified"])
            self.assertFalse(reader.last_validation["scientific_collection_authorized"])
            self.assertEqual(selected["evidence"]["consumed_world_inputs"]["native_atomic_encoding"],
                "native-atomic-backups-compact-v2")
            legacy = deepcopy(inputs); legacy["runtime_configuration"] = runtime(self.cfg, self.files[str(self.checkpoint)])
            legacy["runtime_sha256"] = digest(legacy["runtime_configuration"])
            with self.assertRaises(ValueError): NativeTimedRunValidator(**legacy, allow_synthetic=True)(selected)
            self.assertIsNone(NativeTimedRunValidator(**inputs)(selected))

    def test_catalog_fingerprint_format_preserves_exact_runtime_and_receipt_bindings(self):
        terminal, inputs = self.run_fixture()
        binding = dict(BINDING, set_source_hash="0123456789abcdef")
        observed_runtime = dict(self.runtime, set_source_hash=binding["set_source_hash"])
        changed = dict(inputs, consumed_binding=binding, runtime_configuration=observed_runtime,
            runtime_sha256=digest(observed_runtime))
        reader = NativeTimedRunValidator(**changed, allow_synthetic=True)
        self.assertEqual(reader.consumed, binding)
        with self.assertRaises(ValueError): reader(terminal["result"])
        self.assertIsNone(reader.last_validation)
        for fingerprint in ("", "a"*15, "a"*17, "a"*63, "a"*65, "A"*16,
                "g"*16, True, 123, {"hash":"a"*16}):
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(ValueError, "external timed job"):
                NativeTimedRunValidator(**dict(changed, consumed_binding=dict(binding,
                    set_source_hash=fingerprint)))
        with self.assertRaisesRegex(ValueError, "external timed job"):
            NativeTimedRunValidator(**dict(changed, consumed_binding=dict(binding, root_binding="a"*16)))
        with self.assertRaisesRegex(ValueError, "external timed job"):
            NativeTimedRunValidator(**dict(changed, job_sha256="a"*16))
        with self.assertRaisesRegex(ValueError, "catalog binding drift"):
            NativeTimedRunValidator(**dict(inputs, consumed_binding=binding))

    def test_owned_positive_and_zero_normal_exit_and_default_synthetic_none(self):
        for zero in (False,True):
            terminal,inputs=self.run_fixture(zero,directory=self.root/str(zero))
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            self.assertEqual(terminal["status"],"NATIVE_TIMED_RUN_COMPLETE")
            selected=terminal["result"]
            self.assertIsNone(NativeTimedRunValidator(**inputs)(selected))
            reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
            self.assertEqual(reader(selected),digest(selected))
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],0 if zero else 16)
            for flag in ("runtime_qualified","on_time_deployability_established","full_public_request_correspondence_validated",
                    "scientific_collection_authorized","resource_cost_complete"):
                self.assertFalse(reader.last_validation[flag])

    def test_actual_owned_overrun_is_reported_not_deadline_qualification(self):
        terminal,inputs=self.run_fixture(adapter_elapsed=1.01)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
        self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
        readout=reader.last_validation["actual_work"]["input_readout"]
        self.assertAlmostEqual(readout["adapter_overshoot_seconds"],.01)
        self.assertTrue(terminal["result"]["exceeded_nominal_seconds"])
        self.assertFalse(reader.last_validation["on_time_deployability_established"])

    def test_failures_timeout_and_cleanup_never_zero_unqualification(self):
        for behavior in ("failure","timeout","cleanup"):
            terminal,inputs=self.run_fixture(True,behavior,directory=self.root/behavior)
            self.assertNotEqual(terminal["supervisor_exit_code"],0)
            self.assertIsNone(terminal["result"])
            self.assertFalse(terminal["cpu_wall_cost_complete"])
            if behavior=="timeout":
                self.assertEqual(terminal["supervisor_exit_code"],124)
                self.assertEqual(terminal["status"],"NATIVE_TIMED_RUN_INCOMPLETE")
            reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
            with self.assertRaises((ValueError,FileNotFoundError)):reader(selected_fixture(self.cfg,self.runtime,zero=True))
            self.assertIsNone(reader.last_validation)

    def test_exact_job_input_source_and_return_substitutions_rejected(self):
        terminal,inputs=self.run_fixture();selected=terminal["result"]
        for key,value in (("job_sha256","f"*64),("binding_sha256","f"*64),
                ("selection_seed",8),("legal_action_mask",tuple([False,True,True]+[False]*6))):
            with self.assertRaises(ValueError):NativeTimedRunValidator(**dict(inputs,**{key:value}),allow_synthetic=True)(selected)
        reader=NativeTimedRunValidator(**inputs,allow_synthetic=True);reader.mask=tuple(reversed(MASK))
        with self.assertRaises(ValueError):reader(selected)
        selected["action"]=1
        with self.assertRaises(ValueError):NativeTimedRunValidator(**inputs,allow_synthetic=True)(selected)
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaises(ValueError):NativeTimedRunValidator(**inputs)

    def test_owned_exit_cost_status_generation_and_elapsed_mutations_rejected(self):
        terminal,inputs=self.run_fixture(True);selected=terminal["result"];original=self.load(inputs,"terminal.json")
        edits=[lambda t:t.update(worker_exit_code=True),lambda t:t.update(worker_exit_code=7),
            lambda t:t.update(watchdog_reached=True),lambda t:t.update(cleanup_error_type="ValueError"),
            lambda t:t.update(status="DIAGNOSTIC_COMPLETE"),lambda t:t.update(owned_groups=[]),
            lambda t:t.update(runtime_qualified=True),lambda t:t.update(resource_cost_complete=True),
            lambda t:t["observed_work_accounting"].update(accepted_units=16),
            lambda t:t.update(memory_measurement=None,resource_measurement_error_type=None)]
        for edit in edits:
            changed=deepcopy(original);edit(changed);self.write(inputs,"terminal.json",changed)
            with self.assertRaises(ValueError):NativeTimedRunValidator(**inputs,allow_synthetic=True)(selected)
        self.write(inputs,"terminal.json",original)
        extra=inputs["directory"]/"owned-groups/group-999-unexpected.json";extra.write_text("{}")
        with self.assertRaises(ValueError):NativeTimedRunValidator(**inputs,allow_synthetic=True)(selected)
        extra.unlink()
        changed=deepcopy(selected);changed["elapsed_seconds"]=2.;changed["exceeded_nominal_seconds"]=True
        self.write(inputs,"worker-result.json",dict(self.load(inputs,"worker-result.json"),result=changed))
        self.write(inputs,"terminal.json",dict(original,result=changed))
        with self.assertRaises(ValueError):NativeTimedRunValidator(**inputs,allow_synthetic=True)(changed)

    def test_late_receipt_mutation_and_create_only_attempt_no_retry(self):
        terminal,inputs=self.run_fixture();selected=terminal["result"]
        reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
        from pokezero.mcts_eval.search_over_raw_timed_run import validate_memory_receipt
        def mutate(*args,**kwargs):
            result=validate_memory_receipt(*args,**kwargs)
            value=self.load(inputs,"claim.json");value["retry_authorized"]=True;self.write(inputs,"claim.json",value)
            return result
        with patch("pokezero.mcts_eval.search_over_raw_timed_run.validate_memory_receipt",side_effect=mutate):
            with self.assertRaises(ValueError):reader(selected)
        self.assertIsNone(reader.last_validation)
        with self.assertRaises(FileExistsError):self.run_fixture(directory=inputs["directory"])

    def test_late_bound_source_change_invalidates_a_settled_receipt(self):
        terminal,inputs=self.run_fixture();selected=terminal["result"]
        reader=NativeTimedRunValidator(**inputs,allow_synthetic=True)
        from pokezero.mcts_eval.search_over_raw_timed_run import read_native_timed_selection
        def late(*args,**kwargs):
            result=read_native_timed_selection(*args,**kwargs)
            self.checkpoint.write_bytes(b"late changed authored checkpoint")
            return result
        with patch("pokezero.mcts_eval.search_over_raw_timed_run.read_native_timed_selection",side_effect=late):
            with self.assertRaisesRegex(ValueError,"file byte drift"):reader(selected)
        self.assertIsNone(reader.last_validation)
