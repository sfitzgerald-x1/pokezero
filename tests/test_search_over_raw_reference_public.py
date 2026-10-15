"""Actual public captures/worker preparation and disposable authored owners.

No model, source game, production simulator, collection or holdout is opened.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import pickle
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_reference_timed_job
from pokezero.mcts_eval.search_over_raw_reference_timed_run import (
    ReferenceTimedAdapterJob, ReferenceTimedRunContract, ReferenceTimedRunValidator)
from pokezero.mcts_eval.search_over_raw_timed_reference import (
    validate_reference_timed_inputs, read_reference_timed_selection)
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    TIMED_REFERENCE_PUBLIC_CAPTURE, WorldInputRuntime, WorldInputWorkerFactory,
    WorldInputSearchAdapter, public_request_witness, reference_manifest)
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
from pokezero.mcts_eval.paper_reference_showdown import decision_state
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision, ParallelTrajectorySearch
from pokezero.mcts_eval.paper_reference import ReferenceConfig
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
import test_search_over_raw_archive as archive_fixture
import test_search_over_raw_reference_timed_run as owner_fixture
import test_search_over_raw_timed_reference as timed_fixture
import test_search_over_raw_timed_public as public_fixture
import test_search_over_raw_root_plan as plan_fixture
from test_paper_reference import ToyWorld
from pokezero.mcts_eval.paper_reference import Evaluation
from test_search_over_raw_world_inputs import snapshot as authored_snapshot


class EmptyPublicRuntime:
    """Real prepare protocol with authored empty work, never a simulator."""
    def __init__(self, index): self.index = index

    def prepare(self, request):
        self.prepared_factory = SimpleNamespace(active=False, receipts=[])
        def forbidden(*args): raise AssertionError("expired worker must not start evaluator/world")
        return PreparedDecision(decision_state(request.observation,player=request.state.player_id), forbidden, forbidden,
            evidence=lambda:dict(draws=[]))

    def close(self): pass


@dataclass(frozen=True)
class FullEmptyReferenceOwner:
    cfg: object
    observed: dict
    request: object
    statistics_root: str
    seed: int
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        witness = public_request_witness(self.request)
        phase("startup")
        pool = ParallelTrajectorySearch(ReferenceConfig(.5,1.),
            WorldInputWorkerFactory(EmptyPublicRuntime,witness["public_request_sha256"]),
            workers=20,initial_dispatch_workers=6,owned_process_receipts=ownership)
        try:
            phase("selection"); began = time.perf_counter(); time.sleep(1.01)
            result = pool.search(self.request,decision_state(self.request.observation,player=self.request.state.player_id),
                battle_id="search-over-raw-root:"+self.statistics_root,seed=self.seed,
                deadline_seconds=1.,deadline_at=began+1.,adapter_started_at=began,timed_zero_work=True)
            evidence = asdict(result); evidence["worker_pids"] = list(evidence["worker_pids"])
            evidence["consumed_world_inputs"] = reference_manifest(evidence,witness["consumed_binding"])
            evidence["adapter_public_request_witness"] = witness
            evidence["timed_reference_readout"] = validate_reference_timed_inputs(evidence,
                binding=witness["consumed_binding"],battle_id="search-over-raw-root:"+self.statistics_root,
                selection_seed=self.seed,nominal_seconds=1.,public_request_sha256=witness["public_request_sha256"])
            elapsed = time.perf_counter()-began
            return dict(root_id=self.statistics_root+":"+self.cfg.identity,statistics_root_id=self.statistics_root,
                selection_seed=self.seed,configuration_sha256=self.cfg.identity,runtime_sha256=digest(self.observed),
                independent_root_statistics=True,scientific_strength_evidence=False,evidence=evidence,
                status="UNQUALIFIED_AT_CEILING",action=None,elapsed_seconds=elapsed,
                nominal_search_seconds=1.,exceeded_nominal_seconds=elapsed>1.,attempt_consumed=True,
                retry_authorized=False,continuation_authorized=False,runtime_qualified=False,
                reference_timed_run_contract=ReferenceTimedRunContract().receipt(),
                observed_runtime_configuration=deepcopy(self.observed))
        finally:
            phase("cleanup"); pool.close()


class TerminalPublicRuntime:
    """Actual public/atomic hooks over an explicitly authored terminal world.

    Cleanup pacing uses the worker's supplied absolute deadline to bound the
    toy evidence size, not to establish simulator/model timing or truth.
    """
    def __init__(self,index,*,fail_cleanup=False,expire_before_commit=False):
        self.index=index;self.fail_cleanup=fail_cleanup
        self.expire_before_commit=expire_before_commit;self.deadline=None

    def prepare(self,request):
        root=decision_state(request.observation,player=request.state.player_id)
        env=SimpleNamespace(_search_snapshot_permitted=True,snapshot=authored_snapshot)
        factory=SimpleNamespace(active=False,receipts=[],env=env,state=request.state)
        self.prepared_factory=factory;consumed=0
        def sample(rng):
            if factory.active:raise AssertionError("authored world overlap")
            row=dict(ordinal=len(factory.receipts),status="ROOT_VALIDATED",released=False)
            factory.receipts.append(row);factory.active=True
            world=ToyWorld(root=root,depth=1,value=1.,opponent=False)
            world.env=env;world.subject=request.state.player_id
            advance,close=world.advance,world.close
            def terminal(*args):
                if self.expire_before_commit:
                    time.sleep(max(0.,self.deadline+.01-time.perf_counter()))
                return advance(*args)
            def cleanup():
                time.sleep(max(0.,self.deadline+.01-time.perf_counter()))
                close();row["released"]=True;factory.active=False
                if self.fail_cleanup:raise RuntimeError("authored terminal cleanup failed")
            world.advance=terminal;world.close=cleanup
            return world
        def evidence():
            nonlocal consumed
            rows=deepcopy(factory.receipts[consumed:]);consumed=len(factory.receipts)
            return dict(draws=rows)
        def deadline(value):self.deadline=value
        return PreparedDecision(root,lambda state:Evaluation(tuple(1/len(state.actions) for _ in state.actions),0.),
            sample,evidence,set_sampling_deadline=deadline)

    def close(self):
        if hasattr(self,"prepared_factory") and self.prepared_factory.active:
            raise AssertionError("authored world was not released")


@dataclass(frozen=True)
class TerminalPublicFactory:
    fail_cleanup: bool=False
    expire_before_commit: bool=False
    def __call__(self,index):
        return TerminalPublicRuntime(index,fail_cleanup=self.fail_cleanup,
            expire_before_commit=self.expire_before_commit)


@dataclass(frozen=True)
class FullProductiveReferenceOwner:
    cfg: object
    observed: dict
    request: object
    statistics_root: str
    seed: int
    fail_cleanup: bool=False
    expire_before_commit: bool=False
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        witness=public_request_witness(self.request)
        phase("startup")
        pool=ParallelTrajectorySearch(ReferenceConfig(.5,1.),WorldInputWorkerFactory(
            TerminalPublicFactory(self.fail_cleanup,self.expire_before_commit),witness["public_request_sha256"]),
            workers=20,initial_dispatch_workers=6,owned_process_receipts=ownership)
        try:
            phase("selection");began=time.perf_counter()
            result=pool.search(self.request,decision_state(self.request.observation,player=self.request.state.player_id),
                battle_id="search-over-raw-root:"+self.statistics_root,seed=self.seed,
                deadline_seconds=self.cfg.seconds,deadline_at=began+self.cfg.seconds,
                adapter_started_at=began,timed_zero_work=True)
            evidence=asdict(result);evidence["worker_pids"]=list(evidence["worker_pids"])
            evidence["consumed_world_inputs"]=reference_manifest(evidence,witness["consumed_binding"])
            evidence["adapter_public_request_witness"]=witness
            readout=validate_reference_timed_inputs(evidence,binding=witness["consumed_binding"],
                battle_id="search-over-raw-root:"+self.statistics_root,selection_seed=self.seed,
                nominal_seconds=self.cfg.seconds,public_request_sha256=witness["public_request_sha256"])
            evidence["timed_reference_readout"]=readout
            elapsed=time.perf_counter()-began
            selected=dict(root_id=self.statistics_root+":"+self.cfg.identity,statistics_root_id=self.statistics_root,
                selection_seed=self.seed,configuration_sha256=self.cfg.identity,runtime_sha256=digest(self.observed),
                independent_root_statistics=True,scientific_strength_evidence=False,evidence=evidence,
                status=readout["status"],action=None if readout["accepted_units"] == 0 else int(evidence["result"]["action"][7:]),
                elapsed_seconds=elapsed,nominal_search_seconds=self.cfg.seconds,exceeded_nominal_seconds=elapsed>self.cfg.seconds,
                reference_timed_run_contract=ReferenceTimedRunContract().receipt(),
                observed_runtime_configuration=deepcopy(self.observed))
            if readout["accepted_units"] == 0:
                selected.update(attempt_consumed=True,retry_authorized=False,continuation_authorized=False,runtime_qualified=False)
            return selected
        finally:
            phase("cleanup");pool.close()


class FixedTerminalPublicRuntime(TerminalPublicRuntime):
    """Same toy world, with no timed pacing or decision-clock truncation."""
    def prepare(self,request):
        prepared=super().prepare(request)
        self.deadline=-1.  # Only disable the inherited fixture's cleanup pacing.
        def no_ceiling(value):
            if value is not None:raise AssertionError("fixed work acquired a decision ceiling")
            self.deadline=-1.
        return replace(prepared,set_sampling_deadline=no_ceiling)


@dataclass(frozen=True)
class AuthoredEmptyValueFactory:
    base: object
    recipe: object
    requires_owned_value_generation=True
    def __call__(self,index):
        from pokezero.mcts_eval.search_over_raw_reference_values import OwnedValueRuntime
        return OwnedValueRuntime(self.base(index),self.recipe,index)


class AuthoredNoLeafShowdownRuntime(FixedTerminalPublicRuntime):
    """Actual ShowdownTrajectoryWorld interface; authored one-step terminal, no visited leaf."""
    def prepare(self,request):
        from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld
        prepared=super().prepare(request)
        sample=prepared.sample_world
        def world(rng):
            original=sample(rng)
            ended=[False]
            terminal=SimpleNamespace(winner=request.state.player_id,capped=False)
            env=original.env  # Retain the actual factory's original owned shell.
            env.observe=lambda seat:deepcopy(request.observation)
            env.terminal=lambda:terminal if ended[0] else None
            env.requested_players=lambda:() if ended[0] else (request.state.player_id,)
            env.legal_actions=lambda seat:request.observation.legal_action_mask
            env.reseed_simulator_rng=lambda seed:None
            def step(actions):
                ended[0]=True
                return SimpleNamespace(terminal=terminal)
            env.step=step
            indices=tuple(i for i,v in enumerate(request.observation.legal_action_mask) if v)
            return ShowdownTrajectoryWorld(env,subject=request.state.player_id,
                evaluator=lambda obs:(indices,Evaluation(tuple(1/len(indices) for _ in indices),0.)),
                release=original.close)
        return replace(prepared,sample_world=world)


@dataclass(frozen=True)
class AuthoredExactLeafLabelFactory:
    """Actual fresh Node clone transport, with private authored leaf observations."""
    observations: object
    synthetic_process_fixture=True

    def __call__(self,config):
        from test_search_over_raw_outcomes import EntrypointSyntheticEnv
        observations=deepcopy(self.observations)
        class ExactLeafEnv(EntrypointSyntheticEnv):
            def observe(self,seat):return deepcopy(observations[seat])
        return ExactLeafEnv(config)


class AuthoredVisitedShowdownRuntime(FixedTerminalPublicRuntime):
    """Original source request -> distinct authored leaf -> terminal.

    Only transport/ownership joins are tested. Neither the authored search
    dynamics nor the clone's toy runtime establish simulator fidelity.
    """
    def __init__(self,index,config):
        super().__init__(index);self.config=config

    def prepare(self,request):
        from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld
        from test_search_over_raw_outcomes import EntrypointSyntheticEnv,parser_snapshot
        prepared=super().prepare(request)
        subject=request.state.player_id;opponent="p2" if subject=="p1" else "p1"
        leaf=replace(deepcopy(request.observation),numeric_features=
            tuple(tuple(float(v)+1. for v in row) for row in request.observation.numeric_features))
        other=deepcopy(leaf)
        other.metadata["belief_view"].update(self_slot=opponent,opponent_slot=subject)
        self.label_observations={subject:leaf,opponent:other}
        original_snapshot=parser_snapshot(EntrypointSyntheticEnv(self.config))
        # Match the two actionable indices at the private parser boundary.
        for seat in ("p1","p2"):
            boundary=dict(active=[dict(moves=[dict(id="tackle",pp=15),dict(id="protect",pp=15)])],side=dict(id=seat))
            original_snapshot.latest_requests[seat]=deepcopy(boundary)
            original_snapshot.first_requests[seat]=deepcopy(boundary)
            original_snapshot.request_history[seat]=(deepcopy(boundary),)
            original_snapshot.bridge_snapshot["boundaryRequests"][seat]=deepcopy(boundary)
        sample=prepared.sample_world
        def world(rng):
            original=sample(rng);env=original.env;env.index=0;env.config=self.config
            terminal=SimpleNamespace(winner="p1",capped=False)
            env.observe=lambda seat:deepcopy(request.observation if seat==subject and env.index==0
                else self.label_observations[seat])
            env.snapshot=lambda:replace(deepcopy(original_snapshot),latest_turn=3+env.index)
            env.terminal=lambda:terminal if env.index>=2 else None
            env.requested_players=lambda:() if env.index>=2 else (subject,opponent)
            env.legal_actions=lambda seat:env.observe(seat).legal_action_mask
            env.reseed_simulator_rng=lambda seed:None
            def step(actions):
                env.index+=1
                return SimpleNamespace(terminal=env.terminal())
            env.step=step
            def evaluate(obs):
                indices=tuple(i for i,v in enumerate(obs.legal_action_mask) if v)
                assert indices==(0,1),"authored positive leaf requires two original legal actions"
                return indices,Evaluation((.4,.6),.3)
            return ShowdownTrajectoryWorld(env,subject=subject,evaluator=evaluate,release=original.close)
        return replace(prepared,sample_world=world)


@dataclass(frozen=True)
class AuthoredVisitedRuntimeFactory:
    config: object
    def __call__(self,index):return AuthoredVisitedShowdownRuntime(index,self.config)


@dataclass(frozen=True)
class AuthoredNonemptyValueFactory:
    base: object
    recipe: object
    requires_owned_value_generation=True
    def __call__(self,index):
        from pokezero.mcts_eval.search_over_raw_reference_values import OwnedValueRuntime
        class ActualLeafRuntime(OwnedValueRuntime):
            def _new_bank(self,root_key):
                bank=super()._new_bank(root_key)
                bank.env_factory=AuthoredExactLeafLabelFactory(self.base.base.label_observations)
                return bank
        return ActualLeafRuntime(self.base(index),self.recipe,index)


@dataclass(frozen=True)
class FullFixedReferenceOwner:
    cfg: object
    request: object
    statistics_root: str
    seed: int
    owned_reference_value_recipe: object=None
    runtime_sha256: str="f"*64
    authored_label_config: object=None
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from pokezero.mcts_eval.search_over_raw_world_inputs import validate_fixed_work_inputs
        witness=public_request_witness(self.request);contract=FixedWorkDiagnosticContract()
        phase("startup")
        base=FixedTerminalPublicRuntime if self.owned_reference_value_recipe is None else AuthoredNoLeafShowdownRuntime
        if self.authored_label_config is not None:
            assert self.owned_reference_value_recipe is not None
            from pokezero.mcts_eval.search_over_raw_outcome_run import verify_outcome_runtime
            verify_outcome_runtime(self.owned_reference_value_recipe.runtime,self.authored_label_config)
            base=AuthoredVisitedRuntimeFactory(self.authored_label_config)
        factory=WorldInputWorkerFactory(base,witness["public_request_sha256"])
        if self.owned_reference_value_recipe is not None:
            value_factory=AuthoredEmptyValueFactory if self.authored_label_config is None else AuthoredNonemptyValueFactory
            factory=value_factory(factory,self.owned_reference_value_recipe)
        pool=ParallelTrajectorySearch(ReferenceConfig(.5,1.),factory,
            workers=20,initial_dispatch_workers=6,owned_process_receipts=ownership)
        try:
            phase("selection")
            measured=pool.search(self.request,decision_state(self.request.observation,player=self.request.state.player_id),
                battle_id="search-over-raw-root:"+self.statistics_root,seed=self.seed,trajectories_per_worker=4)
            evidence=asdict(measured);evidence["worker_pids"]=list(evidence["worker_pids"])
            evidence["consumed_world_inputs"]=reference_manifest(evidence,witness["consumed_binding"])
            evidence["adapter_public_request_witness"]=witness
            validate_fixed_work_inputs(evidence,arm="reference",binding=witness["consumed_binding"],
                statistics_root_id=self.statistics_root,expected_public_request_sha256=witness["public_request_sha256"])
            selected=dict(root_id=self.statistics_root+":"+self.cfg.identity,statistics_root_id=self.statistics_root,
                selection_seed=self.seed,configuration_sha256=self.cfg.identity,runtime_sha256=self.runtime_sha256,
                fixed_work_diagnostic=dict(contract=contract.receipt(),accepted_actual=80),evidence=evidence)
            if self.owned_reference_value_recipe is not None:values=pool.finish_diagnostics(measured)
        finally:
            phase("cleanup");pool.close()
        if self.owned_reference_value_recipe is not None:
            from pokezero.mcts_eval.search_over_raw_reference_values import read_completed_reference_values
            summary=read_completed_reference_values(pool,measured,values,self.owned_reference_value_recipe)
            selected["owned_reference_value_diagnostic"]=dict(schema="pokezero.selected-owned-reference-values.v1",
                root_id=selected["root_id"],selected_receipt_sha256=digest(selected),values=values,summary=summary)
        return selected


@dataclass(frozen=True,kw_only=True)
class AuthoredProductiveReferenceJob(ReferenceTimedAdapterJob):
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        return FullProductiveReferenceOwner(self.configuration,self.expected_runtime_configuration,
            self._public_request,self.statistics_root_id,self.selection_seed)(ownership,phase)


class ReferencePublicTests(unittest.TestCase):
    def setUp(self):
        f = archive_fixture.ExplorationRootCrosswalkTests(); f.setUp(); self.addCleanup(f.doCleanups)
        self.f = f; self.bound = f.bind(); self.witness = self.bound["public_request_witness"]
        self.request = PublicRootRequest.capture(f.context.public_materialization_state,f.context.observation)
        self.cfg = SearchConfiguration("reference",workers=20,seconds=1.)

    def test_actual_productive_public_owner_expiry_and_cleanup_failure(self):
        o=owner_fixture.ReferenceOwnerTests();o.setUp();self.addCleanup(o.doCleanups)
        runtime,factory=owner_fixture.owner_runtime(self.cfg,str(o.checkpoint),str(o.showdown),o.files[str(o.checkpoint)])
        factory=replace(factory,set_source_hash=self.witness["consumed_binding"]["set_source_hash"])
        runtime.update(reference_factory=asdict(factory),set_source_hash=factory.set_source_hash,
            full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
        root="authored-terminal-public"
        for fail_cleanup in (False,True):
            owner=FullProductiveReferenceOwner(self.cfg,runtime,self.request,root,7,
                fail_cleanup=fail_cleanup,expire_before_commit=not fail_cleanup)
            directory=o.root/("cleanup-failure" if fail_cleanup else "expired-before-commit")
            reader=ReferenceTimedRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(owner)).hexdigest(),
                binding_sha256="a"*64,configuration=self.cfg,root_id=root+":"+self.cfg.identity,statistics_root_id=root,
                selection_seed=7,runtime_configuration=runtime,runtime_sha256=digest(runtime),
                consumed_binding=self.witness["consumed_binding"],legal_action_mask=self.request.observation.legal_action_mask,
                decision_round=None,file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
                showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=self.witness["public_request_sha256"])
            terminal=_supervise_owned_job(owner,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
                contract=ReferenceTimedRunContract(),kind="reference-timed-run",work_phase="selection",validate_result=reader._read_work)
            if fail_cleanup:
                self.assertEqual(terminal["supervisor_exit_code"],1)
                self.assertEqual(terminal["status"],"OPERATIONAL_STOP")
                self.assertIsNone(terminal["result"])
                self.assertIsNone(reader.last_validation)
            else:
                self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
                selected=terminal["result"]
                self.assertEqual(reader(selected),digest(selected))
                self.assertEqual(selected["status"],"UNQUALIFIED_AT_CEILING")
                self.assertIsNone(selected["action"])
                self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],0)
                self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)

    def authored_selection(self, zero):
        selected = timed_fixture.selection_fixture(self.cfg,zero)
        expected = public_fixture.capture(timed_fixture.BINDING)
        evidence = selected["evidence"]; evidence["adapter_public_request_witness"] = expected
        for receipt in evidence["worker_receipts"]:
            receipt["evidence"]["worker_public_request_preparation"] = dict(
                worker=receipt["worker"],witness=deepcopy(expected))
        evidence["timed_reference_readout"] = validate_reference_timed_inputs(evidence,
            binding=timed_fixture.BINDING,battle_id="search-over-raw-root:root",selection_seed=7,
            nominal_seconds=1.,public_request_sha256=public_fixture.PUBLIC_SHA)
        runtime = timed_fixture.runtime(self.cfg)
        runtime["full_public_request_observation"] = deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE)
        selected["runtime_sha256"] = digest(runtime)
        return selected,runtime

    def read(self, selected, runtime, sha=public_fixture.PUBLIC_SHA):
        return read_reference_timed_selection(selected,configuration=self.cfg,runtime_configuration=runtime,
            runtime_sha256=digest(runtime),root_id="root:"+self.cfg.identity,statistics_root_id="root",
            selection_seed=7,consumed_binding=timed_fixture.BINDING,legal_action_mask=(True,True),
            public_request_sha256=sha)

    def test_productive_cancelled_and_empty_prefixes_require_every_worker(self):
        for zero in (False,True):
            selected,runtime = self.authored_selection(zero); readout = self.read(selected,runtime)
            self.assertTrue(readout["full_public_request_consumption_validated"])
            self.assertEqual(readout["public_request_consumption"]["observed_worker_preparations"],20)
            self.assertFalse(readout["runtime_qualified"])
            for mutation in ("capture","missing","sha","index","bool_index","pending"):
                altered = deepcopy(selected); evidence = altered["evidence"]
                row = evidence["worker_receipts"][17]["evidence"]["worker_public_request_preparation"]
                if mutation == "capture": evidence.pop("adapter_public_request_witness")
                elif mutation == "missing": evidence["worker_receipts"][17]["evidence"].pop("worker_public_request_preparation")
                elif mutation in ("sha","pending"): row["witness"]["public_request_sha256"] = "e"*64
                elif mutation == "index": row["worker"] = 16
                else: row["worker"] = True
                with self.assertRaises(ValueError): self.read(altered,runtime)

    def test_runtime_declaration_and_external_digest_cannot_substitute_capture(self):
        selected,runtime = self.authored_selection(True)
        with self.assertRaises(ValueError): self.read(selected,runtime,None)
        changed = deepcopy(runtime); changed.pop("full_public_request_observation")
        with self.assertRaises(ValueError): self.read(selected,changed)
        changed = deepcopy(runtime); changed["full_public_request_observation"]["includes_workers_without_draws"] = False
        with self.assertRaises(ValueError): self.read(selected,changed)

    def test_real_worker_prepare_and_every_empty_export_hash_same_live_request(self):
        for index in (0,19):
            base = EmptyPublicRuntime(index)
            runtime = WorldInputRuntime(base,expected_public_request_sha256=self.witness["public_request_sha256"],
                worker_index=index)
            wrapped = runtime.prepare(self.request)
            for _ in range(2):
                exported = wrapped.evidence()
                self.assertEqual(exported["draws"],[])
                self.assertEqual(exported["worker_public_request_preparation"],dict(worker=index,witness=self.witness))
            self.request.state.belief_engine._authored_public_cache = {"extra":"changed"}
            with self.assertRaisesRegex(ValueError,"changed after worker preparation"): wrapped.evidence()
            del self.request.state.belief_engine._authored_public_cache

    def test_worker_before_after_prepare_pp_pending_cache_drift_refuses(self):
        from pokezero.mcts_eval.paper_reference_pending import PendingPolicyTransition
        for mutation in ("pp","pending","cache"):
            changed = deepcopy(self.request)
            if mutation == "pp": changed = replace(changed,state=replace(changed.state,self_move_states={"p1:known":({"pp":14},)}))
            elif mutation == "pending": changed = replace(changed,pending_transition=PendingPolicyTransition.capture(changed,0))
            else: changed.state.belief_engine._authored_public_cache = {"extra":"changed"}
            self.assertEqual(public_request_witness(changed)["consumed_binding"],self.witness["consumed_binding"])
            base = Mock()
            runtime = WorldInputRuntime(base,expected_public_request_sha256=self.witness["public_request_sha256"],worker_index=17)
            with self.assertRaisesRegex(ValueError,"different full public request"): runtime.prepare(changed)
            base.prepare.assert_not_called()
        base = EmptyPublicRuntime(17); prepare = base.prepare
        def mutate(request):
            result = prepare(request); request.state.belief_engine._authored_public_cache = {"drift":True}; return result
        base.prepare = mutate
        with self.assertRaisesRegex(ValueError,"changed during worker preparation"):
            WorldInputRuntime(base,expected_public_request_sha256=self.witness["public_request_sha256"],worker_index=17).prepare(self.request)

    def test_worker_factory_binds_actual_index_and_default_stays_compact(self):
        factory = WorldInputWorkerFactory(EmptyPublicRuntime,self.witness["public_request_sha256"])
        runtime = factory(17)
        self.assertEqual(runtime.worker_index,17)
        self.assertEqual(runtime.base.index,17)
        for index in (True,-1,20):
            with self.assertRaises(ValueError): factory(index)
        self.assertNotIn("worker_public_request_preparation",WorldInputWorkerFactory(EmptyPublicRuntime)(0).prepare(self.request).evidence())

    def test_actual_adapter_capture_hashes_request_before_transport(self):
        adapter = WorldInputSearchAdapter.__new__(WorldInputSearchAdapter)
        adapter.configuration = self.cfg; adapter._timed_reference_atomic_work = True
        adapter._expected_public_request_sha256 = self.witness["public_request_sha256"]
        adapter._pool = Mock(search=Mock(return_value="authored"))
        self.assertEqual(adapter._search_reference(self.request,decision_state(self.request.observation,player=self.request.state.player_id),"root",7,1.),"authored")
        self.assertEqual(adapter._reference_public_capture,self.witness)
        changed = replace(self.request,state=replace(self.request.state,self_move_states={"p1:known":({"pp":14},)}))
        adapter._pool.search.reset_mock()
        with self.assertRaisesRegex(ValueError,"captured a different"): adapter._search_reference(changed,None,"root",7,1.)
        adapter._pool.search.assert_not_called()

    def test_fixed_work_reference_captures_request_without_a_decision_ceiling(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputFixedWorkAdapter
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from test_search_over_raw_supervision import selection_inputs
        adapter=WorldInputFixedWorkAdapter.__new__(WorldInputFixedWorkAdapter)
        adapter.configuration=self.cfg;adapter.fixed_work=FixedWorkDiagnosticContract()
        adapter._timed_reference_atomic_work=False
        adapter._expected_public_request_sha256=self.witness["public_request_sha256"]
        adapter._pool=Mock(search=Mock(return_value="authored-fixed-work"))
        self.assertEqual(adapter._search_reference(self.request,None,"root",7,1.),"authored-fixed-work")
        self.assertEqual(adapter._pool.search.call_args.kwargs["trajectories_per_worker"],4)
        self.assertFalse({"deadline_at","deadline_seconds"}&set(adapter._pool.search.call_args.kwargs))
        evidence=selection_inputs("reference")
        result=adapter._selection_evidence(evidence,self.request)
        self.assertEqual(result["adapter_public_request_witness"],self.witness)
        changed=replace(self.request,state=replace(self.request.state,self_move_states={"p1:known":({"pp":14},)}))
        self.assertEqual(public_request_witness(changed)["consumed_binding"],self.witness["consumed_binding"])
        with self.assertRaisesRegex(ValueError,"changed during selection"):
            adapter._selection_evidence(evidence,changed)
        adapter._pool.search.reset_mock()
        with self.assertRaisesRegex(ValueError,"captured a different"):
            adapter._search_reference(changed,None,"root",7,1.)
        adapter._pool.search.assert_not_called()

    def test_actual_fixed_work_twenty_preparations_and_eighty_commits_keep_original_archive_request(self):
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
        from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
        import test_search_over_raw_fixed_work_run as fixed_fixtures
        f=self.f;bound=self.bound;witness=self.witness
        o=fixed_fixtures.FixedWorkRunReaderTests();o.setUp();self.addCleanup(o.doCleanups)
        root=f.root_id;seed=bound["slot"]["source_seed"]
        owner=FullFixedReferenceOwner(self.cfg,self.request,root,seed)
        directory=o.root/"full-fixed-reference"
        terminal=_supervise_owned_job(owner,directory=directory,binding_sha256="a"*64,
            verify_admission=lambda:None,contract=FixedWorkDiagnosticContract(),kind="fixed-work",work_phase="selection",
            validate_result=lambda value:derive_fixed_work_accounting(value["evidence"],arm="reference",
                binding=witness["consumed_binding"],statistics_root_id=root,leaf="model"))
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader=FixedWorkRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(owner)).hexdigest(),
            binding_sha256="a"*64,configuration=self.cfg,root_id=root+":"+self.cfg.identity,
            statistics_root_id=root,selection_seed=seed,runtime_sha256="f"*64,consumed_binding=witness["consumed_binding"],
            public_request_sha256=witness["public_request_sha256"],file_hashes=o.files,
            checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True)
        selected=terminal["result"]
        self.assertEqual(reader(selected),digest(selected))
        self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)
        self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],80)
        self.assertEqual(len(selected["evidence"]["consumed_world_inputs"]["rows"]),80)
        self.assertEqual({row["worker"] for row in selected["evidence"]["worker_receipts"]},set(range(20)))
        self.assertTrue(all(row["evidence"]["worker_public_request_preparation"]["witness"]==witness
            for row in selected["evidence"]["worker_receipts"]))
        self.assertFalse(selected["evidence"]["result"]["deadline_exhausted"])
        self.assertFalse(reader.last_validation["runtime_qualified"])

    def test_original_archive_request_nonempty_labels_use_actual_twenty_worker_fixed_owner(self):
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from pokezero.mcts_eval.search_over_raw_reference_values import ReferenceValueRunValidator
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
        from test_search_over_raw_reference_values import OwnedReferenceValueTests
        fixture=OwnedReferenceValueTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        from pathlib import Path
        import sys
        file_hashes={**fixed_work_reader_source_hashes(),**fixture.recipe.runtime["files"]}
        executable=str(Path(sys.executable).resolve())
        file_hashes[executable]=hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        root=self.f.root_id;seed=self.bound["slot"]["source_seed"]
        owner=FullFixedReferenceOwner(self.cfg,self.request,root,seed,fixture.recipe,
            authored_label_config=fixture.config)
        directory=fixture.root/"original-request-positive"
        terminal=_supervise_owned_job(owner,directory=directory,binding_sha256="a"*64,
            verify_admission=lambda:None,contract=FixedWorkDiagnosticContract(),kind="fixed-work",work_phase="selection",
            validate_result=lambda value:derive_fixed_work_accounting(value["evidence"],arm="reference",
                binding=self.witness["consumed_binding"],statistics_root_id=root,leaf="model"))
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        runtime=fixture.recipe.runtime
        reader=ReferenceValueRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(owner)).hexdigest(),
            binding_sha256="a"*64,configuration=self.cfg,root_id=root+":"+self.cfg.identity,
            statistics_root_id=root,selection_seed=seed,runtime_sha256="f"*64,consumed_binding=self.witness["consumed_binding"],
            public_request_sha256=self.witness["public_request_sha256"],file_hashes=file_hashes,
            checkpoint_path=runtime["checkpoint_path"],checkpoint_sha256=runtime["files"][runtime["checkpoint_path"]],
            showdown_root=runtime["argv"][3],allow_synthetic=True,recipe=fixture.recipe)
        selected=terminal["result"]
        self.assertEqual(reader(selected),digest(selected))
        self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],80)
        summary=reader.last_validation["reference_value_labels"]
        self.assertEqual((summary["sampled_leaves"],summary["labels"],summary["complete_labels"]),(20,160,160))
        for row in selected["owned_reference_value_diagnostic"]["values"]:
            self.assertNotEqual(row["evidence"]["leaves"][0]["information_key"],self.witness["consumed_binding"]["information_key"])
            self.assertFalse(row["evidence"]["labels_used_for_backup"])

    def test_actual_archive_job_owned_twenty_empty_workers_and_plan(self):
        p = plan_fixture.RootPlanTests(); p.setUp(); self.addCleanup(p.doCleanups)
        f = p.archive_fixture; bound = p.bound; witness = bound["public_request_witness"]
        o = p.owner_fixture; checkpoint = p.checkpoint_contract()
        runtime,factory = owner_fixture.owner_runtime(self.cfg,str(o.checkpoint),str(o.showdown),checkpoint.checkpoint_sha256)
        factory = replace(factory,set_source_hash=witness["consumed_binding"]["set_source_hash"])
        runtime.update(reference_factory=asdict(factory),set_source_hash=factory.set_source_hash,
            full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
        with patch("subprocess.Popen") as launch:
            job,actual = prepare_archived_reference_timed_job(f.archive,f.source,contract=f.contract,root_id=f.root_id,
                configuration=self.cfg,checkpoint_contract=checkpoint,showdown_root=str(o.showdown),
                reference_factory=factory,runtime_configuration=runtime)
        launch.assert_not_called(); self.assertEqual(actual,bound)
        self.assertEqual(job._inputs(),job._frozen_binding)
        self.assertNotIn(b"private-source-snapshot",pickle.dumps(job))
        owner = FullEmptyReferenceOwner(self.cfg,runtime,job._public_request,f.root_id,bound["slot"]["source_seed"])
        directory = o.root/"full-empty-reference"
        reader = ReferenceTimedRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(owner)).hexdigest(),
            binding_sha256="a"*64,configuration=self.cfg,root_id=job.root_id,statistics_root_id=f.root_id,
            selection_seed=job.selection_seed,runtime_configuration=runtime,runtime_sha256=digest(runtime),
            consumed_binding=witness["consumed_binding"],legal_action_mask=job._legal_action_mask,decision_round=None,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=checkpoint.checkpoint_sha256,
            showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=witness["public_request_sha256"])
        terminal = _supervise_owned_job(owner,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
            contract=ReferenceTimedRunContract(),kind="reference-timed-run",work_phase="selection",validate_result=reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal); selected = terminal["result"]
        self.assertEqual(reader(selected),digest(selected))
        self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)
        self.assertTrue(all(not rec["evidence"]["draws"] for rec in selected["evidence"]["worker_receipts"]))
        adapter = Mock(runtime_configuration=deepcopy(runtime),runtime_sha256=digest(runtime),
            _pool=SimpleNamespace(workers=20,batch_size=10,initial_dispatch_workers=6))
        adapter.select.return_value = deepcopy(selected)
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=SimpleNamespace(metadata=SimpleNamespace(source_hash=factory.set_source_hash))), \
                patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter",return_value=adapter) as constructor:
            result = job({},lambda phase:None)
        self.assertEqual(constructor.call_args.kwargs["expected_public_request_sha256"],witness["public_request_sha256"])
        self.assertTrue(job.read_selection(result)["full_public_request_consumption_validated"])
        adapter.close.assert_called_once()
        with patch("subprocess.Popen") as launch:
            plan = p.plan(configurations=(p.cfg,self.cfg),selections={"raw":p.selected,self.cfg.identity:selected},
                selector_readers={"raw":p.reader,self.cfg.identity:reader})
            receipt = plan.revalidate()
        launch.assert_not_called()
        self.assertEqual(receipt["full_public_correspondence_by_selector"],{"raw":True,self.cfg.identity:True})
        self.assertTrue(receipt["full_timed_public_correspondence_validated"])
        self.assertIsNone(receipt["root_evidence"]["actions"].get(self.cfg.identity))
        self.assertEqual(len(receipt["continuation_cells"]),8)
        self.assertFalse(receipt["runtime_qualified"]); self.assertFalse(receipt["collection_authorized"])
        reader.reference_public_request_sha = "e"*64
        with self.assertRaises(ValueError): plan.revalidate()
        job.context.public_materialization_state.belief_engine._authored_public_cache = {"extra":"changed"}
        with self.assertRaisesRegex(ValueError,"bindings changed"): job.read_selection(selected)
