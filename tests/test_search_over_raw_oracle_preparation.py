"""Authored archive and real Python oracle preparation; no games or model load."""
from copy import deepcopy
from dataclasses import replace
import pickle
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.local_showdown import LocalShowdownSnapshot
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive,capture_archived_team_oracle
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest,_ShowdownRuntime
from pokezero.mcts_eval.search_over_raw_oracle import TeamOracle,OracleRootRequest,OracleTeamSampler,_OracleRuntime
from pokezero.randbat import Gen3RandbatSource,RandbatSourceMetadata
import test_search_over_raw_archive as archive_fixture
import pokezero.mcts_eval.search_over_raw_oracle as oracle_module


def opening(seat):
    species=("Pikachu","Charmander","Squirtle","Bulbasaur","Eevee","Snorlax")
    return dict(side=dict(id=seat,pokemon=[dict(ident=seat+": "+name,details=name+", L100, M",
        condition="",active=i == 0,moves=["tackle"],baseAbility="Fixture",item="Leftovers")
        for i,name in enumerate(species)]))


class AuthoredFactory(PublicRootWorldFactory):
    """Observe constructor order and real team sampler without simulation."""
    def __init__(self,**kwargs):
        self.set_source=kwargs["set_source"];self.state=kwargs["state"]
        self.env=kwargs["env"];self.evaluator=kwargs["evaluator"]
        self.pending_transition=kwargs["pending_transition"];self.known=()
        self.own_team=oracle_module.opening_team(self.state.self_initial_request,self.state.player_id,self.set_source)
        from pokezero.mcts_eval.paper_reference_showdown import decision_state
        self.root=decision_state(kwargs["observation"],player=self.state.player_id)
        self.receipts=[];self.active=False;self.history_population=None
        self.constant_chance_plan=None;self.diagnostic_oracle_team=None
        self.collect_phase_timing=False;self.sampling_deadline_at=None
        self.sampler=None
        team=kwargs.get("diagnostic_oracle_team")
        if team is not None:self.install_diagnostic_oracle(team)
        # Represents the constructor's prospective pending chance-plan step.
        self.team_at_plan_construction=None if self.sampler is None else self.sampler.sha256
        self.constant_chance_plan={"team_at_plan_construction":self.team_at_plan_construction}
        self.closed=False;self.worlds=[];self.on_sample=None

    def __call__(self,rng):
        draw=self.sampler.draw((),rng)
        self.receipts.append(dict(status="ROOT_VALIDATED",packed_team_sha256=draw.packed_team_sha256))
        world=SimpleNamespace(close=Mock());self.worlds.append(world)
        if self.on_sample is not None:self.on_sample()
        return world

    def close(self):self.closed=True


class OraclePreparationTests(unittest.TestCase):
    def setUp(self):
        f=archive_fixture.ExplorationRootCrosswalkTests();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f
        self.source=Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle",3,None,None,None,"a"*64),{})
        state=replace(f.context.public_materialization_state,self_initial_request=opening("p1"))
        state.belief_engine.set_source=self.source
        self.context=replace(f.context,public_materialization_state=state)
        self.snapshot=LocalShowdownSnapshot("private-battle-token","private-battle-id",
            "gen3randombattle","gen3randombattle",{"canary":"PRIVATE_SNAPSHOT_NEVER_TRANSPORT"},(),
            {"p1":deepcopy(state.self_request),"p2":{"committed":"PRIVATE_OPPONENT_ACTION"}},
            {"p1":deepcopy(state.self_initial_request),"p2":opening("p2")},{},
            deepcopy(state.replay),deepcopy(state.belief_engine),0,None)
        self.archive=SealedSourceArchive();self.addCleanup(self.archive.close)
        self.archive.capture_public(self.context,0)
        self.archive.capture_private(SimpleNamespace(requested_players=("p1",),decision_round_index=0,
            seed=f.context.seed,battle_id=f.context.battle_id,decisions={"p1":SimpleNamespace(action_index=0)},
            snapshot=self.snapshot))
        self.request=PublicRootRequest.capture(state,self.context.observation)
        self.oracle=TeamOracle.capture(self.snapshot,self.request,self.source)

    def capture(self):
        return capture_archived_team_oracle(self.archive,self.f.source,contract=self.f.contract,
            root_id=self.f.root_id,set_source=self.source)

    def runtime(self):
        base=object.__new__(_ShowdownRuntime)
        base.source=self.source;base.env=SimpleNamespace(close=Mock())
        base.evaluator=SimpleNamespace(forwards=0)
        base.allow_earlier_compatible_template=False;base.max_known_set_draws=10
        runtime=_OracleRuntime(base);self.addCleanup(runtime.close)
        return runtime

    def prepare(self,runtime):
        with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootWorldFactory",AuthoredFactory),\
                patch("pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory",AuthoredFactory):
            return runtime.prepare(OracleRootRequest(self.request,self.oracle))

    def test_actual_opening_extraction_returns_only_original_team(self):
        with patch("subprocess.Popen") as launch:
            oracle,receipt=self.capture()
        launch.assert_not_called()
        self.assertEqual(oracle,self.oracle)
        self.assertEqual(len(oracle.opponent_team),6)
        self.assertFalse(receipt["deployable"]);self.assertFalse(receipt["collection_authorized"])
        wire=pickle.dumps(OracleRootRequest(self.request,oracle))
        for private in (b"PRIVATE_SNAPSHOT_NEVER_TRANSPORT",b"PRIVATE_OPPONENT_ACTION",b"LocalShowdownSnapshot"):
            self.assertNotIn(private,wire)

    def test_canonical_source_request_alias_cannot_match_truth(self):
        request=replace(self.request,state=replace(self.request.state,
            self_request={"alias":True}))
        snapshot=replace(self.snapshot,latest_requests={"p1":{"alias":1}})
        self.assertEqual(snapshot.latest_requests["p1"],request.state.self_request)
        with self.assertRaisesRegex(ValueError,"does not match"):
            TeamOracle.capture(snapshot,request,self.source)

    def test_late_archive_and_catalog_mutation_reject_projection(self):
        for kind in ("archive","catalog"):
            with self.subTest(kind=kind):
                original=oracle_module.opening_team
                calls=0
                def mutate(*args):
                    nonlocal calls
                    result=original(*args);calls+=1
                    if calls == 4:
                        if kind == "archive":self.archive._private[0].bridge_snapshot["late"]=True
                        else:self.source.move_metadata["late"]={"changed":True}
                    return result
                with patch.object(oracle_module,"opening_team",side_effect=mutate):
                    with self.assertRaisesRegex(ValueError,"changed during extraction"):
                        self.capture()
                self.archive._private[0].bridge_snapshot.pop("late",None)
                self.source.move_metadata.pop("late",None)

    def test_team_installed_before_pending_chance_plan_and_empty_export(self):
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory
        self.assertEqual(factory.team_at_plan_construction,self.oracle.opponent_team_sha256)
        evidence=prepared.evidence()
        self.assertEqual(evidence["draws"],[])
        witness=evidence["team_oracle_preparation"]
        self.assertTrue(witness["installed_before_chance_planning"])
        self.assertEqual(witness["actual_sampler_team_sha256"],self.oracle.opponent_team_sha256)
        self.assertFalse(witness["deployable"])

    def test_public_constructor_receives_no_team_override(self):
        runtime=self.runtime()
        with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootWorldFactory",AuthoredFactory):
            runtime.base.prepare(self.request)
        self.assertIsNone(runtime.base.prepared_factory.team_at_plan_construction)

    def test_late_oracle_installation_cannot_rewrite_prebuilt_plan(self):
        runtime=self.runtime()
        with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootWorldFactory",AuthoredFactory):
            runtime.base.prepare(self.request)
        factory=runtime.base.prepared_factory
        with self.assertRaisesRegex(Exception,"cannot replace"):
            factory.install_diagnostic_oracle(self.oracle.opponent_team)
        self.assertIsNone(factory.sampler)

    def test_sampling_checks_actual_team_and_closes_mutated_world(self):
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory
        factory.on_sample=lambda:setattr(factory,"sampler",None)
        with self.assertRaisesRegex(ValueError,"construction/public transport changed"):
            prepared.sample_world(random.Random(0))
        self.assertEqual(len(factory.worlds),1)
        factory.worlds[0].close.assert_called_once()

    def test_sampler_drift_before_sampling_or_zero_export_refuses(self):
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory;factory.sampler=None
        for read in (lambda:prepared.sample_world(random.Random(0)),prepared.evidence):
            with self.assertRaisesRegex(ValueError,"construction/public transport changed"):read()
        self.assertEqual(factory.worlds,[])

    def test_tree_timing_on_guard_reaches_actual_factory_evidence(self):
        from pokezero.mcts_eval.paper_reference import ReferenceConfig,TrajectorySearch
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory;factory.collect_phase_timing=True
        tree=TrajectorySearch(ReferenceConfig(.5,1.));tree.reset_battle("authored")
        tree.search_batch(prepared.root,battle_id="authored",evaluate_root=prepared.evaluate_root,
            sample_world=prepared.sample_world,seed=0,trajectories=1,deadline_at=0.,clock=lambda:1.)
        self.assertEqual(factory.worlds,[])
        self.assertEqual(len(factory.search_phase_timing),1)
        self.assertEqual(prepared.evidence()["search_phase_timing"],factory.search_phase_timing)
        self.assertEqual(prepared.evidence()["search_phase_timing"],[])
        with self.assertRaisesRegex(AttributeError,"immutable"):
            prepared.sample_world._sample=lambda rng:None

    def test_cleanup_error_does_not_mask_primary_truth_drift(self):
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory
        def mutate():
            factory.worlds[-1].close.side_effect=RuntimeError("private cleanup detail")
            factory.sampler=None
        factory.on_sample=mutate
        with self.assertRaisesRegex(ValueError,"construction/public transport changed") as raised:
            prepared.sample_world(random.Random(0))
        self.assertEqual(raised.exception.__notes__,["oracle world cleanup also failed: RuntimeError"])
        self.assertNotIn("private cleanup detail",str(raised.exception.__notes__))
        factory.worlds[-1].close.assert_called_once()

    def test_catalog_and_full_public_drift_refuse_empty_export(self):
        runtime=self.runtime();prepared=self.prepare(runtime)
        self.source.move_metadata["late"]={"changed":True}
        with self.assertRaisesRegex(ValueError,"construction/public transport changed"):prepared.evidence()
        self.source.move_metadata.pop("late")
        self.request.state.self_move_states["late"]=(dict(pp=1),)
        with self.assertRaisesRegex(ValueError,"construction/public transport changed"):prepared.evidence()

    def test_late_evidence_callback_mutation_refuses_export(self):
        runtime=self.runtime();original=runtime.base._prepare
        def altered(*args,**kwargs):
            prepared=original(*args,**kwargs)
            def evidence():
                result=prepared.evidence();runtime.base.prepared_factory.sampler=None
                return result
            return replace(prepared,evidence=evidence)
        with patch.object(runtime.base,"_prepare",side_effect=altered):prepared=self.prepare(runtime)
        with self.assertRaisesRegex(ValueError,"construction/public transport changed"):prepared.evidence()

    def test_added_prepared_hooks_are_preserved(self):
        runtime=self.runtime();original=runtime.base._prepare
        finish=Mock();atomic=object()
        def altered(*args,**kwargs):
            return replace(original(*args,**kwargs),finish_diagnostics=finish,atomic_work=atomic)
        with patch.object(runtime.base,"_prepare",side_effect=altered):prepared=self.prepare(runtime)
        self.assertIs(prepared.finish_diagnostics,finish);self.assertIs(prepared.atomic_work,atomic)

    def test_actual_factory_only_inputs_cannot_drift_behind_transport(self):
        for field in ("state","pending","source","own_team","known","env","evaluator"):
            with self.subTest(field=field):
                runtime=self.runtime();prepared=self.prepare(runtime)
                factory=runtime.base.prepared_factory
                if field == "state":factory.state=replace(factory.state,self_move_states={"late":(dict(pp=1),)})
                elif field == "pending":factory.pending_transition=object()
                elif field == "source":factory.set_source=deepcopy(self.source)
                elif field == "own_team":factory.own_team=factory.own_team[:5]
                elif field in ("env","evaluator"):setattr(factory,field,object())
                else:factory.known=(dict(unobserved="changed"),)
                for check in (lambda:prepared.sample_world(random.Random(0)),prepared.evidence):
                    with self.assertRaises(ValueError):check()
                self.assertEqual(factory.worlds,[])

    def test_actual_pending_ancestor_is_compared_with_original_transport(self):
        from pokezero.mcts_eval.paper_reference_pending import PendingPolicyTransition
        pending=PendingPolicyTransition.capture(self.request,0)
        self.request=replace(self.request,pending_transition=pending)
        runtime=self.runtime();prepared=self.prepare(runtime)
        factory=runtime.base.prepared_factory
        self.assertIsNot(factory.pending_transition,pending)
        factory.pending_transition=replace(factory.pending_transition,
            before_state=replace(factory.pending_transition.before_state,self_move_states={"late":(dict(pp=1),)}))
        self.assertNotIn("late",pending.before_state.self_move_states)
        with self.assertRaisesRegex(ValueError,"construction/public transport changed"):
            prepared.evidence()

    def test_wrong_constructor_public_inputs_cannot_establish_success_baseline(self):
        class WrongFactory(AuthoredFactory):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                self.state=replace(self.state,self_move_states={"constructor-only":(dict(pp=1),)})
        runtime=self.runtime()
        with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootWorldFactory",WrongFactory),\
                patch("pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory",WrongFactory):
            with self.assertRaisesRegex(ValueError,"construction/public transport changed"):
                runtime.prepare(OracleRootRequest(self.request,self.oracle))

    def test_retained_public_prefix_is_not_reused_as_oracle(self):
        runtime=self.runtime();runtime.base.history_prefix_reuse=True
        with self.assertRaisesRegex(Exception,"fresh construction"):
            self.prepare(runtime)
        self.assertFalse(hasattr(runtime.base,"prepared_factory"))


if __name__ == "__main__":unittest.main()
