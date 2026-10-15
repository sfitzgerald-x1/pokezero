"""Original-team native hooks and actual Python constructors; no games/models."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.engine_world import battle_spec_from_payload
from pokezero.poke_engine_adapter import build_poke_engine_state
from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_native_oracle import (NativeOracleWorldInputs,
    NativeTimedOracleSearchAdapter,validate_native_oracle_inputs)
from pokezero.mcts_eval.search_over_raw_world_inputs import (NativeInputHandle,WorldInputSearchAdapter,
    public_request_witness,validate_timed_native_inputs,_state_payload)
from pokezero.mcts_eval.search_over_raw_oracle import DiagnosticTeamOracleSearchAdapter
import test_search_over_raw_oracle_preparation as oracle_fixture
import test_engine_world as world_fixture
import test_poke_engine_adapter as engine_fixture
import test_search_over_raw_timed_native as timed_fixture


class NativeOracleTests(unittest.TestCase):
    def setUp(self):
        f=oracle_fixture.OraclePreparationTests();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.witness=public_request_witness(f.request)
        from pokezero.env import BattleStartOverride
        from pokezero.showdown_fixture import pack_team
        from pokezero.mcts_eval.search_over_raw_oracle import opening_team
        own=opening_team(f.request.state.self_initial_request,"p1",f.source)
        self.override=BattleStartOverride(player_teams={"p1":pack_team(own),"p2":pack_team(f.oracle.opponent_team)},
            observation_format_id="gen3randombattle")
        self.policy=SimpleNamespace(_fixed_override=self.override,_set_source=f.source)
        self.catalog_sha=digest(_state_payload(f.source))

    def recorder(self):
        return NativeOracleWorldInputs(self.witness["consumed_binding"],16,
            record_atomic_work=True,record_timed_atomic_work=True,expected_public_request=self.witness,
            oracle=self.f.oracle,oracle_request=self.f.request,oracle_source=self.f.source,oracle_policy=self.policy)

    def construct(self,recorder,text="complete-state"):
        # Actual assembly with authored dex/HP data and a fake native State.
        dex=world_fixture._dex();template=dex.species["swampert"]
        names=[p.species for p in self.f.oracle.opponent_team]
        dex=replace(dex,species={name.lower():replace(template,id=name.lower(),name=name) for name in names},
            moves={"tackle":replace(dex.moves["earthquake"],id="tackle",name="Tackle",pp=35)})
        payload=world_fixture._payload(world_fixture._dex());payload["selfTeamOrder"]=names
        for seat in ("p1","p2"):
            side=payload["sides"][seat]
            side.update(boosts={},sideConditions={},sideConditionSetTurns={})
            side["pokemon"]=[dict(species=name,condition="362/362" if seat == "p1" else "100/100",
                active=i == 0,**(dict(moves=[dict(id="tackle",pp=56,maxpp=56,disabled=False)])
                    if seat == "p1" else {})) for i,name in enumerate(names)]
        recorder.attempt(self.f.context,len(recorder.rows),self.override)
        world=battle_spec_from_payload(payload,self.override,dex=dex,construction_observer=recorder.world_built)
        module=engine_fixture.fake_construction_module()
        def make_state(**kwargs):
            state=engine_fixture.RecordingState(**kwargs);state.serialized=text;return state
        module.State=make_state
        state=build_poke_engine_state(world.spec,module=module,construction_observer=recorder.native_built)
        recorder.constructed(self.f.context,world,state)
        return world,state

    def dispatch(self,recorder,world,completed=16,cancelled=False):
        report=timed_fixture.timed_report(completed,requested=16*len(recorder.rows),cancelled=cancelled)
        args=timed_fixture.timed_arguments(requested=16*len(recorder.rows))
        args[5]=json.dumps(dict(**{seat:list(world.party_species[seat]) for seat in ("p1","p2")},turn=3))
        handle=NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=Mock(return_value=json.dumps(report))),recorder)
        handle.search_batched_multi_encoded(*args)

    def finish(self,recorder,completed=0):
        return recorder.finish(dict(total_iterations=completed),attempted=len(recorder.rows),
            constructed=len(recorder.rows),searched=sum(c["multiplicity"] for c in recorder.calls
                if c["completed_iterations"] > 0))

    def read(self,manifest):
        return validate_native_oracle_inputs(manifest,oracle_receipt=self.f.oracle.receipt(),
            public_request=self.witness,expected_override_sha256=digest(asdict(self.override)),
            expected_reconstruction_catalog_sha256=self.catalog_sha)

    def test_zero_attempts_are_preparation_not_consumption(self):
        recorder=self.recorder();manifest=self.finish(recorder)
        result=self.read(manifest)
        self.assertTrue(result["prepared"]);self.assertFalse(result["team_consumed"])
        self.assertEqual((result["attempted_worlds"],result["native_calls"]),(0,0))
        self.assertFalse(result["atomic_prefix_validated"])
        self.assertFalse(result["owning_process_validated"])

    def test_actual_python_constructor_native_dispatch_and_collapsed_truth_join(self):
        recorder=self.recorder();world,_=self.construct(recorder);self.construct(recorder)
        self.dispatch(recorder,world,cancelled=True)
        manifest=self.finish(recorder,16);result=self.read(manifest)
        self.assertEqual((result["attempted_worlds"],result["dispatched_worlds"],result["native_calls"]),(2,2,1))
        self.assertTrue(result["team_consumed"])
        self.assertEqual(manifest["calls"][0]["construction_ordinals"],[0,1])

    def test_constructed_but_not_dispatched_is_not_consumed(self):
        recorder=self.recorder();self.construct(recorder)
        result=self.read(self.finish(recorder))
        self.assertEqual(result["constructed_worlds"],1);self.assertEqual(result["dispatched_worlds"],0)
        self.assertFalse(result["team_consumed"])

    def test_zero_commit_call_still_consumes_original_team(self):
        recorder=self.recorder();world,_=self.construct(recorder)
        self.dispatch(recorder,world,completed=0,cancelled=True)
        result=self.read(self.finish(recorder))
        self.assertTrue(result["team_consumed"])
        self.assertEqual(result["native_calls"],1)

    def test_sampled_or_none_override_and_swallowed_hook_failure_cannot_certify(self):
        for override in (None,deepcopy(self.override),replace(self.override,player_teams={"p1":"wrong","p2":"wrong"})):
            recorder=self.recorder()
            with self.subTest(override=override),self.assertRaises(ValueError):
                recorder.attempt(self.f.context,0,override)
            with self.assertRaises(ValueError):self.finish(recorder)

    def test_preparation_source_and_installed_override_drift_refuse_empty_export(self):
        for change in ("source","catalog","override","public"):
            recorder=self.recorder()
            with self.subTest(change=change):
                if change == "source":self.policy._set_source=deepcopy(self.f.source)
                elif change == "catalog":self.f.source.move_metadata["late"]={"changed":True}
                elif change == "override":self.policy._fixed_override=deepcopy(self.override)
                else:self.f.request.state.self_move_states["late"]=(dict(pp=1),)
                with self.assertRaises(ValueError):self.finish(recorder)
                self.policy._set_source=self.f.source;self.policy._fixed_override=self.override
                self.f.source.move_metadata.pop("late",None);self.f.request.state.self_move_states.pop("late",None)

    def test_missing_or_mutated_constructor_and_dispatch_joins_refuse(self):
        recorder=self.recorder();world,_=self.construct(recorder);self.dispatch(recorder,world)
        original=self.finish(recorder,16)
        for field in ("constructor","state","team","attempt","call","multiplicity","bool-multiplicity",
                "bool-call","consumed","catalog","catalog-hex","extra-private"):
            manifest=deepcopy(original);row=manifest["rows"][0]
            if field == "constructor":row.pop("construction_receipt")
            elif field == "state":row["native_state_utf8_sha256"]="f"*64
            elif field == "team":manifest["native_team_oracle"]["preparation"]["oracle"]["opponent_team_sha256"]="f"*64
            elif field == "attempt":manifest["native_team_oracle"]["attempts"][0]["ordinal"]=False
            elif field == "call":manifest["calls"][0]["native_call_parameters"]["state_sha256"]="f"*64
            elif field == "multiplicity":manifest["calls"][0]["multiplicity"]=2
            elif field == "bool-multiplicity":manifest["calls"][0]["multiplicity"]=True
            elif field == "bool-call":row["call_ordinal"]=False
            elif field == "catalog":manifest["native_team_oracle"]["preparation"]["reconstruction_catalog_sha256"]="f"*64
            elif field == "catalog-hex":manifest["native_team_oracle"]["preparation"]["reconstruction_catalog_sha256"]="z"*64
            elif field == "extra-private":manifest["native_team_oracle"]["preparation"]["snapshot"]="private"
            else:row["consumed"]=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.read(manifest)

    def test_actual_prefix_reader_accepts_variable_original_team_population(self):
        for completed in (0,16):
            recorder=self.recorder();world,_=self.construct(recorder);self.construct(recorder)
            self.dispatch(recorder,world,completed=completed,cancelled=True)
            manifest=self.finish(recorder,completed)
            invocations=[]
            for call in manifest["calls"]:
                row=call["atomic_work_validation"]
                invocations.append(dict(status="completed",world_seed=7,multiplicity=call["multiplicity"],
                    requested_iterations=call["requested_iterations"],completed_iterations=completed,
                    remaining_iterations=row["remaining_iterations"],time_budget_ms=row["native_budget_ms"],
                    time_budget_elapsed_ms=row["native_elapsed_ms"],time_budget_batch_overshoot_ms=row["native_overshoot_ms"],
                    time_budget_exhausted=row["exhausted"],root_visits=dict(side_one=completed,side_two=completed)))
            evidence=dict(total_iterations=completed,consumed_world_inputs=manifest,
                engine_mcts=dict(time_budget=dict(scope="whole_model_decision",requested_ms=1000,
                    native_batch_guard_ms=64,deadline_elapsed_ms=1001.,deadline_overshoot_ms=1.,exhausted=True,
                    worlds_budget_skipped=2 if completed == 0 else 0,native_invocations=invocations)))
            result=validate_timed_native_inputs(evidence,binding=self.witness["consumed_binding"],
                iterations_per_world=16,batch_size=16,expected_worlds=4,nominal_seconds=1.,adapter_elapsed_seconds=1.2)
            self.assertEqual(result["attempted_worlds"],2)
            self.assertEqual(result["completed_iterations"],completed)
            self.assertEqual(result["not_attempted_worlds"],2)
            self.assertTrue(self.read(manifest)["team_consumed"])

    def test_public_lane_still_rejects_oracle_and_A1_rejects_other_scopes(self):
        with self.assertRaises(ValueError):WorldInputSearchAdapter(SearchConfiguration("incumbent",belief="oracle"))
        for cfg in (SearchConfiguration("incumbent"),SearchConfiguration("incumbent",belief="oracle",seconds=3.),
                SearchConfiguration("incumbent",belief="oracle",leaf="hp_fraction"),
                SearchConfiguration("reference",belief="oracle",workers=20)):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):
                NativeTimedOracleSearchAdapter(cfg,expected_public_request_sha256=self.witness["public_request_sha256"])

    def test_actual_A1_adapter_strict_zero_keeps_preparation_and_cleans_hooks(self):
        from pokezero.engine_search import EngineSearchFallbackError
        cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.)
        def authored_init(adapter,configuration,**kwargs):
            adapter.configuration=configuration;adapter.runtime_configuration={}
            adapter.oracle=self.f.oracle;adapter._oracle_source=self.f.source
        with patch.object(DiagnosticTeamOracleSearchAdapter,"__init__",authored_init):
            adapter=NativeTimedOracleSearchAdapter(cfg,
                expected_public_request_sha256=self.witness["public_request_sha256"])
        adapter.contract=SimpleNamespace(showdown_source_sha256="a"*64)
        adapter._closed=adapter._poisoned=False
        adapter._pending_value_diagnostics=adapter._value_diagnostics=None
        adapter._attempted=set();adapter._attempted_statistics_roots=set();adapter.last_failure=None
        class Policy:
            _fixed_override=_ladder_sims_override=_value_diagnostic_session=None
            _world_attempt_observer=_world_observer=None
            _world_construction_observer=_native_construction_observer=None
            _set_source=self.f.source
            _config=SimpleNamespace(model_world_workers=1,early_stop=False,search_sims=4096,
                model_decision_time_ms=1000,search_batch=16,worlds=4)
            stats=SimpleNamespace(worlds_attempted=0,worlds_constructed=0,worlds_searched=0,total_iterations=0)
            def reset(self):pass
            def _native(self):raise AssertionError("no native call at expired ceiling")
            def select_action_with_context(policy,context,rng):
                diagnostic=dict(schema="engine-search-fallback-refusal-v1",diagnostic_only_not_policy_input=True,
                    battle_id=context.battle_id,round=context.decision_round_index,seat=context.player_id,
                    reason="model_time_budget_no_completed_worlds",world_failures={},
                    engine_mcts=dict(fallback="model_time_budget_no_completed_worlds",time_budget=dict(
                        scope="whole_model_decision",requested_ms=1000,native_batch_guard_ms=64,
                        deadline_elapsed_ms=1001.,deadline_overshoot_ms=1.,exhausted=True,
                        worlds_budget_skipped=0,native_invocations=[])))
                raise EngineSearchFallbackError("authored expired ceiling",diagnostic=diagnostic)
        adapter._native=Policy()
        adapter._decider=SimpleNamespace(_snapshot_stats=lambda policy:dict(depth_reached_histogram={}))
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs.time.perf_counter",side_effect=[100.,101.2]):
            result=adapter.select(self.f.context,root_id="root:"+cfg.identity,
                statistics_root_id="root",selection_seed=7)
        self.assertEqual(result["status"],"UNQUALIFIED_AT_CEILING");self.assertIsNone(result["action"])
        observed=result["evidence"]["consumed_world_inputs"]["native_team_oracle"]
        self.assertTrue(observed["prepared"]);self.assertFalse(observed["team_consumed"])
        self.assertEqual(result["evidence"]["team_oracle"],self.f.oracle.receipt())
        self.assertFalse(result["continuation_authorized"]);self.assertTrue(adapter._poisoned)
        for name in ("_world_attempt_observer","_world_observer","_world_construction_observer",
                "_native_construction_observer"):
            self.assertIsNone(getattr(adapter._native,name))
        self.assertNotIn("_native",vars(adapter._native))
        with self.assertRaises(ValueError):adapter.select(self.f.context,root_id="other",selection_seed=7)


if __name__ == "__main__":unittest.main()
