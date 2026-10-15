"""Real Python original-team preparation and disposable workers; no science.

Positive prefixes below are authored protocol packets, not simulator/evaluator
validation. The empty owner uses all twenty actual Python worker preparations.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import hashlib
import json
import pickle
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_oracle import OracleRootRequest,_OracleRuntime,OracleWorkerFactory
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    public_request_witness,_state_payload,TIMED_REFERENCE_PUBLIC_CAPTURE,reference_manifest)
from pokezero.mcts_eval.search_over_raw_reference_oracle_run import (
    ReferenceOracleWorldInputRuntime,ReferenceOracleWorldInputWorkerFactory,
    ReferenceTimedOracleSearchAdapter,ReferenceOracleTimedRunContract,
    ReferenceOracleTimedAdapterJob,ReferenceOracleTimedRunValidator,
    REFERENCE_ORACLE_CAPTURE,read_reference_oracle_timed_selection,validate_reference_oracle_inputs)
from pokezero.mcts_eval.search_over_raw_reference_timed_run import ReferenceTimedRunContract
from pokezero.mcts_eval.search_over_raw_timed_reference import validate_reference_timed_inputs,read_reference_timed_selection
from pokezero.mcts_eval.paper_reference_runtime import _ShowdownRuntime
from pokezero.mcts_eval.paper_reference_showdown import decision_state
from pokezero.mcts_eval.paper_reference import ReferenceConfig,TrajectorySearch
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_reference_oracle_timed_job
from pokezero.mcts_eval.search_over_raw_belief_diagnostics import team_fingerprints
import test_search_over_raw_oracle_preparation as truth_fixture
import test_search_over_raw_reference_timed_run as owner_fixture
import test_search_over_raw_reference_public as public_fixture
import test_search_over_raw_timed_reference as timed_fixture
import test_search_over_raw_root_plan as plan_fixture


class EmptyPreparedOracleRuntime(_OracleRuntime):
    def prepare(self,request):
        with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootWorldFactory",truth_fixture.AuthoredFactory),\
                patch("pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory",truth_fixture.AuthoredFactory):
            return super().prepare(request)


@dataclass(frozen=True)
class AuthoredEmptyOracleFactory:
    source: object
    witness: dict
    oracle: object

    def __call__(self,index):
        base=object.__new__(_ShowdownRuntime)
        base.source=deepcopy(self.source);base.env=SimpleNamespace(close=Mock())
        base.evaluator=SimpleNamespace(forwards=0)
        base.allow_earlier_compatible_template=False;base.max_known_set_draws=10
        return ReferenceOracleWorldInputRuntime(EmptyPreparedOracleRuntime(base),worker_index=index,
            expected_public_request_sha256=self.witness["public_request_sha256"],
            expected_oracle_receipt=self.oracle.receipt(),
            expected_oracle_payload_sha256=digest(_state_payload(self.oracle)),
            expected_reconstruction_catalog_sha256=digest(_state_payload(self.source)))


@dataclass(frozen=True)
class FullEmptyOracleOwner(public_fixture.FullEmptyReferenceOwner):
    oracle: object
    source: object

    def __call__(self,ownership,phase):
        witness=public_request_witness(self.request)
        phase("startup")
        pool=ParallelTrajectorySearch(ReferenceConfig(.5,1.),
            AuthoredEmptyOracleFactory(self.source,witness,self.oracle),workers=20,
            initial_dispatch_workers=6,owned_process_receipts=ownership)
        try:
            phase("selection");began=time.perf_counter();time.sleep(1.01)
            result=pool.search(OracleRootRequest(self.request,self.oracle),
                decision_state(self.request.observation,player=self.request.state.player_id),
                battle_id="search-over-raw-root:"+self.statistics_root,seed=self.seed,
                deadline_seconds=1.,deadline_at=began+1.,adapter_started_at=began,timed_zero_work=True)
            evidence=asdict(result);evidence["worker_pids"]=list(evidence["worker_pids"])
            evidence["consumed_world_inputs"]=reference_manifest(evidence,witness["consumed_binding"])
            evidence["adapter_public_request_witness"]=witness;evidence["team_oracle"]=self.oracle.receipt()
            evidence["timed_reference_readout"]=validate_reference_timed_inputs(evidence,
                binding=witness["consumed_binding"],battle_id="search-over-raw-root:"+self.statistics_root,
                selection_seed=self.seed,nominal_seconds=1.,public_request_sha256=witness["public_request_sha256"])
            elapsed=time.perf_counter()-began
            return dict(root_id=self.statistics_root+":"+self.cfg.identity,statistics_root_id=self.statistics_root,
                selection_seed=self.seed,configuration_sha256=self.cfg.identity,runtime_sha256=digest(self.observed),
                independent_root_statistics=True,scientific_strength_evidence=False,evidence=evidence,
                status="UNQUALIFIED_AT_CEILING",action=None,elapsed_seconds=elapsed,
                nominal_search_seconds=1.,exceeded_nominal_seconds=elapsed>1.,attempt_consumed=True,
                retry_authorized=False,continuation_authorized=False,runtime_qualified=False,
                reference_timed_run_contract=ReferenceOracleTimedRunContract().receipt(),
                observed_runtime_configuration=deepcopy(self.observed))
        finally:
            phase("cleanup");pool.close()


class ReferenceOracleRunTests(unittest.TestCase):
    def setUp(self):
        self.t=truth_fixture.OraclePreparationTests();self.t.setUp();self.addCleanup(self.t.doCleanups)
        self.cfg=SearchConfiguration("reference",belief="oracle",seconds=1.,workers=20)
        self.witness=public_request_witness(self.t.request)
        self.catalog=digest(_state_payload(self.t.source));self.payload=digest(_state_payload(self.t.oracle))

    def wrapped(self,index=17):
        base=self.t.runtime()
        runtime=ReferenceOracleWorldInputRuntime(base,worker_index=index,
            expected_public_request_sha256=self.witness["public_request_sha256"],
            expected_oracle_receipt=self.t.oracle.receipt(),expected_oracle_payload_sha256=self.payload,
            expected_reconstruction_catalog_sha256=self.catalog)
        return runtime,self.t.prepare(runtime),base.base.prepared_factory

    def test_actual_wrapper_empty_exports_original_team_and_timing_proxy(self):
        runtime,prepared,factory=self.wrapped()
        self.assertEqual(factory.team_at_plan_construction,self.t.oracle.opponent_team_sha256)
        self.assertEqual(prepared.atomic_work.binding,self.witness["consumed_binding"])
        prepared.sample_world.collect_phase_timing=True
        tree=TrajectorySearch(ReferenceConfig(.5,1.));tree.reset_battle("authored")
        tree.search_batch(prepared.root,battle_id="authored",evaluate_root=prepared.evaluate_root,
            sample_world=prepared.sample_world,seed=0,trajectories=1,deadline_at=0.,clock=lambda:1.)
        first=prepared.evidence();second=prepared.evidence()
        self.assertEqual(first["draws"],[]);self.assertEqual(second["draws"],[])
        self.assertEqual(first["worker_public_request_preparation"],dict(worker=17,witness=self.witness))
        observed=first["worker_original_team_preparation"]
        self.assertEqual(observed["worker"],17);self.assertTrue(observed["materialized_draw_observer"])
        self.assertEqual(observed["preparation_sha256"],digest(first["team_oracle_preparation"]))
        self.assertEqual(len(first["search_phase_timing"]),1)
        self.assertEqual(factory.worlds,[])
        with self.assertRaises(AttributeError):prepared.sample_world._sample=lambda rng:None

    def test_wrong_wrapper_team_catalog_or_public_rejected_before_prepare(self):
        for mutation in ("wrapper","team","catalog","public"):
            runtime,_,_=self.wrapped();request=OracleRootRequest(self.t.request,self.t.oracle)
            if mutation == "wrapper":request=self.t.request
            elif mutation == "team":request=replace(request,oracle=replace(self.t.oracle,opponent_team_sha256="f"*64))
            elif mutation == "catalog":runtime.catalog_sha="f"*64
            else:runtime.expected_public_request_sha256="f"*64
            with self.subTest(mutation=mutation),patch.object(runtime.base,"prepare") as prepare:
                with self.assertRaises(ValueError):runtime.prepare(request)
                prepare.assert_not_called()

    def test_actual_observer_and_team_mutation_refuse_empty_export(self):
        for mutation in ("observer","catalog","sampler","full_payload"):
            runtime,prepared,factory=self.wrapped()
            if mutation == "observer":factory.collect_sampled_team_traits=False
            elif mutation == "catalog":self.t.source.move_metadata["late"]={"drift":True}
            elif mutation == "sampler":factory.sampler=None
            else:object.__setattr__(self.t.oracle,"opponent_team",self.t.oracle.opponent_team[:5])
            with self.assertRaises(ValueError):prepared.evidence()
            if mutation == "catalog":self.t.source.move_metadata.pop("late")
            if mutation == "full_payload":break

    def test_primary_identity_failure_survives_cleanup_error(self):
        runtime,prepared,factory=self.wrapped()
        def sample(actual_factory,rng):
            factory.receipts.append(dict(ordinal=0,status="ROOT_VALIDATED"))
            return SimpleNamespace(env=object(),subject="p1",close=Mock(side_effect=RuntimeError("PRIVATE_CLEANUP")))
        # Replace only the authored factory callable, retaining genuine oracle guards.
        with patch.object(truth_fixture.AuthoredFactory,"__call__",sample):
            with self.assertRaisesRegex(ValueError,"receipt identity drift") as raised:
                prepared.sample_world(None)
        self.assertEqual(raised.exception.__notes__,["Reference input world cleanup also failed: RuntimeError"])
        self.assertNotIn("PRIVATE_CLEANUP",str(raised.exception.__notes__))

    def authored(self,zero=False):
        fixture=public_fixture.ReferencePublicTests();fixture.cfg=self.cfg
        selected,runtime=fixture.authored_selection(zero)
        oracle=self.t.oracle.receipt();oracle.update(root_binding=timed_fixture.BINDING["root_binding"],
            set_source_hash=timed_fixture.BINDING["set_source_hash"])
        witness=deepcopy(selected["evidence"]["adapter_public_request_witness"])
        runtime,factory=owner_fixture.owner_runtime(self.cfg,"/authored-model","/authored-showdown","a"*64)
        runtime.update(full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE),
            reference_team_oracle_observation=deepcopy(REFERENCE_ORACLE_CAPTURE),team_oracle=oracle,
            oracle_payload_sha256=self.payload,oracle_reconstruction_catalog_sha256=self.catalog)
        selected.update(runtime_sha256=digest(runtime),observed_runtime_configuration=deepcopy(runtime),
            reference_timed_run_contract=ReferenceOracleTimedRunContract().receipt())
        evidence=selected["evidence"];evidence["team_oracle"]=oracle
        members=team_fingerprints(self.t.oracle.opponent_team)
        for row in evidence["worker_receipts"]:
            raw=row["evidence"]
            prep=dict(schema="pokezero.search-over-raw.reference-oracle-preparation.v1",oracle=oracle,
                public_request_witness=witness,actual_sampler_team_sha256=oracle["opponent_team_sha256"],
                actual_construction_sha256="c"*64,oracle_payload_sha256=self.payload,
                reconstruction_catalog_sha256=self.catalog,installed_before_chance_planning=True,
                source_snapshot_transported=False,committed_opponent_action_transported=False,
                deployable=False,runtime_qualified=False)
            raw.update(team_oracle=oracle,team_oracle_preparation=prep,
                worker_original_team_preparation=dict(worker=row["worker"],preparation_sha256=digest(prep),
                    materialized_draw_observer=True,root_binding=oracle["root_binding"]))
            for draw in raw["draws"]:
                draw.update(diagnostic_oracle_team_sha256=oracle["opponent_team_sha256"],
                    information_scope="original_opponent_team_only",packed_team_sha256=oracle["opponent_team_sha256"])
                if draw.get("consumed_world_input") is not None:
                    draw["sampled_team_origin"]=dict(schema="pokezero.search-over-raw.sampled-original-team.v1",
                        original_team_sha256=oracle["opponent_team_sha256"],members=members,members_sha256=digest(members))
                    draw["sampled_original_team"]=dict(schema="pokezero.search-over-raw.sampled-original-team.v1",
                        role="HYPOTHETICAL_DRAW_ONLY",**timed_fixture.BINDING,members=members,members_sha256=digest(members),
                        original_team_sha256=oracle["opponent_team_sha256"],instrumentation_can_change_deadlines=True,
                        qualifies_uninstrumented_runtime=False)
        evidence["timed_reference_readout"]=validate_reference_timed_inputs(evidence,
            binding=timed_fixture.BINDING,battle_id="search-over-raw-root:root",selection_seed=7,
            nominal_seconds=1.,public_request_sha256=witness["public_request_sha256"])
        return selected,dict(configuration=self.cfg,runtime_configuration=runtime,runtime_sha256=digest(runtime),
            root_id="root:"+self.cfg.identity,statistics_root_id="root",selection_seed=7,
            consumed_binding=timed_fixture.BINDING,legal_action_mask=(True,True),
            public_request_sha256=witness["public_request_sha256"],public_request=witness,oracle_receipt=oracle,
            expected_oracle_payload_sha256=self.payload,expected_reconstruction_catalog_sha256=self.catalog,
            contract=ReferenceOracleTimedRunContract())

    def test_authored_positive_and_empty_use_same_atomic_prefix_not_quota(self):
        for zero in (False,True):
            selected,inputs=self.authored(zero)
            readout=read_reference_oracle_timed_selection(selected,**inputs)
            self.assertEqual(readout["accepted_units"],0 if zero else 80)
            self.assertEqual(readout["original_team_consumption"]["prepared_workers"],20)
            self.assertIs(readout["original_team_consumption"]["team_consumed"],not zero)
            self.assertFalse(readout["archive_truth_correspondence_validated"])
            self.assertFalse(readout["runtime_qualified"])

    def test_missing_wrong_or_self_certified_truth_and_indexes_refuse(self):
        for mutation in ("missing_worker","bool_worker","missing_observer","payload","catalog","runtime_team","actual_team","origin","materialized"):
            selected,inputs=self.authored()
            evidence=selected["evidence"];row=evidence["worker_receipts"][17];raw=row["evidence"]
            if mutation == "missing_worker":evidence["worker_receipts"].pop()
            elif mutation == "bool_worker":raw["worker_original_team_preparation"]["worker"]=True
            elif mutation == "missing_observer":raw["worker_original_team_preparation"]["materialized_draw_observer"]=False
            elif mutation == "payload":inputs["expected_oracle_payload_sha256"]="f"*64
            elif mutation == "catalog":inputs["expected_reconstruction_catalog_sha256"]="f"*64
            elif mutation == "runtime_team":
                runtime=deepcopy(inputs["runtime_configuration"]);runtime["team_oracle"]["opponent_team_sha256"]="f"*64
                inputs.update(runtime_configuration=runtime,runtime_sha256=digest(runtime))
                selected.update(runtime_sha256=digest(runtime),observed_runtime_configuration=deepcopy(runtime))
            else:
                draw=next(d for d in raw["draws"] if d.get("consumed_world_input") is not None)
                if mutation == "actual_team":draw["packed_team_sha256"]="f"*64
                elif mutation == "origin":draw.pop("sampled_team_origin")
                else:draw.pop("sampled_original_team")
                with self.assertRaises(ValueError):validate_reference_oracle_inputs(evidence,
                    **{k:inputs[k] for k in ("public_request","oracle_receipt",
                        "expected_oracle_payload_sha256","expected_reconstruction_catalog_sha256")})
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                read_reference_oracle_timed_selection(selected,**inputs)

    def test_selected_history_anchor_is_observed_not_presampling_marker(self):
        selected,inputs=self.authored()
        for row in selected["evidence"]["worker_receipts"]:
            for draw in row["evidence"]["draws"]:
                if draw.get("consumed_world_input") is not None:
                    anchor=dict(packed_team_sha256=draw.pop("packed_team_sha256"),
                        sampled_team_origin=draw.pop("sampled_team_origin"))
                    draw["substitute_particle_conditioning"]=dict(anchor=anchor)
        selected["evidence"]["timed_reference_readout"]=validate_reference_timed_inputs(selected["evidence"],
            binding=inputs["consumed_binding"],battle_id="search-over-raw-root:root",selection_seed=7,
            nominal_seconds=1.,public_request_sha256=inputs["public_request_sha256"])
        read_reference_oracle_timed_selection(selected,**inputs)
        draw=next(d for d in selected["evidence"]["worker_receipts"][0]["evidence"]["draws"]
            if d.get("consumed_world_input") is not None)
        draw["substitute_particle_conditioning"]["anchor"]["packed_team_sha256"]="f"*64
        with self.assertRaises(ValueError):validate_reference_oracle_inputs(selected["evidence"],
            **{k:inputs[k] for k in ("public_request","oracle_receipt",
                "expected_oracle_payload_sha256","expected_reconstruction_catalog_sha256")})

    def test_public_reader_wrong_contract_or_configuration_cannot_certify_A1(self):
        selected,inputs=self.authored(True)
        with self.assertRaises(ValueError):read_reference_oracle_timed_selection(selected,
            **dict(inputs,contract=ReferenceTimedRunContract()))
        for cfg in (replace(self.cfg,belief="public"),replace(self.cfg,seconds=3.),replace(self.cfg,leaf="hp_fraction")):
            with self.assertRaises(ValueError):read_reference_oracle_timed_selection(selected,**dict(inputs,configuration=cfg))
        public={k:v for k,v in inputs.items() if k not in ("public_request","oracle_receipt", "contract",
            "expected_oracle_payload_sha256","expected_reconstruction_catalog_sha256")}
        with self.assertRaises(ValueError):read_reference_timed_selection(selected,**public)

    def test_late_outer_runtime_mutation_refuses_combined_read(self):
        import pokezero.mcts_eval.search_over_raw_reference_oracle_run as module
        selected,inputs=self.authored(True);original=module.validate_reference_oracle_inputs
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs);inputs["runtime_configuration"]["late"]=True;return result
        with patch.object(module,"validate_reference_oracle_inputs",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"changed during read"):
                read_reference_oracle_timed_selection(selected,**inputs)

    def test_distinct_factory_wraps_genuine_oracle_once(self):
        _,base=owner_fixture.owner_runtime(self.cfg,"/authored-model","/authored-showdown","a"*64)
        adapter=ReferenceTimedOracleSearchAdapter.__new__(ReferenceTimedOracleSearchAdapter)
        adapter.configuration=self.cfg;adapter.oracle=self.t.oracle
        adapter._expected_public_request_sha256=self.witness["public_request_sha256"]
        adapter._expected_catalog_sha=self.catalog
        factory=adapter._reference_worker_factory(base)
        self.assertIs(type(factory),ReferenceOracleWorldInputWorkerFactory)
        self.assertIs(type(factory.base),OracleWorkerFactory)
        self.assertIs(factory.base.base,base)

    def archive_job(self):
        o=owner_fixture.ReferenceOwnerTests();o.setUp();self.addCleanup(o.doCleanups)
        checkpoint=SimpleNamespace(checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_source_sha256=self.t.source.metadata.source_hash)
        runtime,factory=owner_fixture.owner_runtime(self.cfg,str(o.checkpoint),str(o.showdown),checkpoint.checkpoint_sha256)
        factory=replace(factory,set_source_hash=checkpoint.showdown_source_sha256)
        runtime.update(reference_factory=asdict(factory),set_source_hash=checkpoint.showdown_source_sha256,
            full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE),
            reference_team_oracle_observation=deepcopy(REFERENCE_ORACLE_CAPTURE),team_oracle=self.t.oracle.receipt(),
            oracle_payload_sha256=self.payload,oracle_reconstruction_catalog_sha256=self.catalog)
        f=self.t.f
        args=dict(contract=f.contract,root_id=f.root_id,configuration=self.cfg,checkpoint_contract=checkpoint,
            showdown_root=str(o.showdown),reference_factory=factory,runtime_configuration=runtime,set_source=self.t.source)
        with patch("subprocess.Popen") as launch:
            job,receipt=prepare_archived_reference_oracle_timed_job(self.t.archive,f.source,**args)
        launch.assert_not_called()
        return o,job,receipt,args

    def test_actual_archive_join_freezes_original_before_constructor_replacement(self):
        o,job,receipt,args=self.archive_job()
        self.assertEqual(receipt["team_projection"]["team_oracle"],self.t.oracle.receipt())
        self.assertEqual(job._inputs(),job._frozen_binding)
        for token in (b"PRIVATE_SNAPSHOT_NEVER_TRANSPORT",b"PRIVATE_OPPONENT_ACTION",b"LocalShowdownSnapshot"):
            self.assertNotIn(token,pickle.dumps(job))
        self.assertFalse(receipt["collection_authorized"]);self.assertFalse(receipt["holdout_authorized"])
        original=ReferenceOracleTimedAdapterJob
        def replacement(**kwargs):
            # Internally valid replacement truth/runtime must not replace archive origin.
            changed=replace(kwargs["oracle"],opponent_team=tuple(reversed(kwargs["oracle"].opponent_team)))
            from pokezero.mcts_eval.search_over_raw_oracle import team_sha256
            changed=replace(changed,opponent_team_sha256=team_sha256(changed.opponent_team))
            runtime=deepcopy(kwargs["expected_runtime_configuration"])
            runtime.update(team_oracle=changed.receipt(),oracle_payload_sha256=digest(_state_payload(changed)))
            return original(**dict(kwargs,oracle=changed,expected_oracle_receipt=changed.receipt(),
                expected_runtime_configuration=runtime))
        with patch("pokezero.mcts_eval.search_over_raw_reference_oracle_run.ReferenceOracleTimedAdapterJob",side_effect=replacement):
            with self.assertRaisesRegex(ValueError,"differs from original extracted"):
                prepare_archived_reference_oracle_timed_job(self.t.archive,self.t.f.source,**args)

    def test_actual_twenty_worker_empty_owner_job_wiring_and_indexed_closure(self):
        o,job,receipt,args=self.archive_job()
        witness=public_request_witness(job._public_request);runtime=job.expected_runtime_configuration
        owner=FullEmptyOracleOwner(self.cfg,runtime,job._public_request,job.statistics_root_id,
            job.selection_seed,job.oracle,self.t.source)
        directory=o.root/"original-team-empty-reference"
        reader=ReferenceOracleTimedRunValidator(directory=directory,
            job_sha256=hashlib.sha256(pickle.dumps(owner)).hexdigest(),binding_sha256="a"*64,
            configuration=self.cfg,root_id=job.root_id,statistics_root_id=job.statistics_root_id,
            selection_seed=job.selection_seed,runtime_configuration=runtime,runtime_sha256=digest(runtime),
            consumed_binding=witness["consumed_binding"],legal_action_mask=job._legal_action_mask,decision_round=None,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=witness["public_request_sha256"],
            oracle_receipt=job.expected_oracle_receipt,public_request=witness,
            expected_oracle_payload_sha256=self.payload,expected_reconstruction_catalog_sha256=self.catalog,
            contract=ReferenceOracleTimedRunContract())
        terminal=_supervise_owned_job(owner,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
            contract=ReferenceOracleTimedRunContract(),kind="reference-oracle-timed-run",
            work_phase="selection",validate_result=reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(terminal["status"],"REFERENCE_ORACLE_TIMED_RUN_COMPLETE")
        selected=terminal["result"];self.assertEqual(reader(selected),digest(selected))
        actual=reader.last_validation["actual_work"]
        self.assertEqual(actual["accepted_units"],0)
        self.assertEqual(actual["original_team_consumption"]["prepared_workers"],20)
        self.assertFalse(actual["original_team_consumption"]["team_consumed"])
        self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)
        self.assertTrue(all(not row["evidence"]["draws"] for row in selected["evidence"]["worker_receipts"]))
        self.assertIsNone(selected["action"]);self.assertFalse(selected["retry_authorized"])
        adapter=Mock(runtime_configuration=deepcopy(runtime),runtime_sha256=digest(runtime),oracle=job.oracle,
            _oracle_source=self.t.source,_pool=SimpleNamespace(workers=20,batch_size=10,initial_dispatch_workers=6))
        adapter.select.return_value=deepcopy(selected)
        phases=[]
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=self.t.source),\
                patch("pokezero.mcts_eval.search_over_raw_reference_oracle_run.ReferenceTimedOracleSearchAdapter",return_value=adapter) as constructor:
            returned=job({},phases.append)
        self.assertEqual(phases,["startup","selection","cleanup"]);adapter.close.assert_called_once()
        self.assertIs(constructor.call_args.kwargs["oracle"],job.oracle)
        self.assertEqual(job.read_selection(returned)["accepted_units"],0)
        closed=next((directory/"owned-groups").glob("closed-group-*.json"));original=closed.read_bytes()
        row=json.loads(original);row["worker_exit_code"]=1;closed.write_text(json.dumps(row))
        with self.assertRaises(ValueError):reader(selected)
        closed.write_bytes(original)
        launch=next((directory/"owned-groups").glob("group-*.json"));original=launch.read_bytes()
        row=json.loads(original);row["worker_index"]=True;launch.write_text(json.dumps(row))
        with self.assertRaises(ValueError):reader(selected)
        launch.write_bytes(original)
        object.__setattr__(job.oracle,"opponent_team_sha256","f"*64)
        with self.assertRaises(ValueError):job.read_selection(selected)
