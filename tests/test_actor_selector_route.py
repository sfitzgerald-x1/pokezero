"""Owned authored source protocol plus actual random-model selectors; no games.

Source policy rows and simulator messages are authored. Native/model/raw
selector execution and all ownership, preparation and independent readers are
real. Nothing here qualifies source policy, physics, costs or collection.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
import traceback
import sys
import unittest
from unittest.mock import patch

import test_native_fixed_work_adapter as native_fixture
import test_search_over_raw_continuation_admission as fresh_fixture
from test_model_priors_search import _crate_ready
from test_search_over_raw_source import FakeChampion
from test_search_over_raw_source_run import AuthoredConfig
from pokezero.local_showdown import LocalShowdownEnv, showdown_seed_from_int
from pokezero.env import StepResult
from pokezero.policy import PolicyDecision
from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_raw_run import RawSelectionJob,raw_runtime
from pokezero.mcts_eval.search_over_raw_actor_timed_run import actor_timed_runtime_recipe
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_source_run import (
    collect_owned_exploration_panel,source_recipe,source_review_template)
from pokezero.mcts_eval.search_over_raw_selector_producer import (produce_archived_selector,
    read_recorded_actor_input_unavailable,read_actor_input_unavailable,_actor_unavailable_path)
from pokezero.mcts_eval.search_over_raw_reconciled import (
    ReconciledPhaseALedger,configuration_review_template,selector_runtime_recipe,controller_input_hashes)
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work


class AuthoredMaskedSourcePolicy(FakeChampion):
    def __init__(self,checkpoint):
        super().__init__();self.weights_sha256=checkpoint.checkpoint_sha256
        self.result.model_config=AuthoredConfig()
    def select_action(self,observation,*,rng):
        legal=[i for i,v in enumerate(observation.legal_action_mask) if v]
        probabilities=[1/len(legal) if i in legal else 0. for i in range(len(observation.legal_action_mask))]
        return PolicyDecision(legal[0],"authored-source-row",metadata={"policy_distribution":probabilities})


class AuthoredCachedSourceEnv(LocalShowdownEnv):
    """Actual pipe ownership and encoder; fixed one-boundary protocol fixture."""
    def __init__(self,config):
        super().__init__(config);self.n=0
    def reset(self,*,seed,format_id="gen3randombattle"):
        self._source_seed=seed
        return super().reset(seed=seed,format_id=format_id)
    def _bridge_request_boundary(self,payload):
        self._send_command(payload);return self._read_until_event_type("ready")
    def requested_players(self):
        if self.n:return ()
        return ("p1","p2") if self._source_seed in (0,1) else ("p2",)
    def step(self,actions):
        self._send_command(dict(type="choices",battleId=self._battle_token,
            choices={p:"move "+str(a+1) for p,a in actions.items()}))
        self._read_until_event_type("terminal");self.n=1
        return StepResult(observations={p:self.observe(p) for p in ("p1","p2")},
            rewards=self._rewards(),terminal=self._terminal,requested_players=self.requested_players())
    def snapshot_actionable_boundary(self):
        snapshot=super().snapshot_actionable_boundary()
        # Explicit legacy-cache absence, authored before the archive/owner seals.
        if self._source_seed == 1: snapshot=replace(snapshot,annotation_cache=None)
        return snapshot


class AuthoredCachedSourceFactory:
    synthetic_process_fixture=True
    def __new__(cls,config,checkpoint):
        return AuthoredCachedSourceEnv(config),{p:AuthoredMaskedSourcePolicy(checkpoint) for p in ("p1","p2")}


@dataclass(frozen=True)
class ActualAuthoredRawJob(RawSelectionJob):
    synthetic_process_fixture=True


@dataclass(frozen=True)
class ActorRouteAuditor:
    ledger_directory:str
    checkpoint:object
    config:object
    source_runtime:dict
    source_review_path:str
    history_review_path:str
    configuration_reviews:dict
    files:dict
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup")
        ledger=ReconciledPhaseALedger(self.ledger_directory)
        cfg=SearchConfiguration("incumbent",seconds=1.)
        runtime=actor_timed_runtime_recipe(cfg,self.checkpoint)
        control=raw_runtime(SearchConfiguration("raw",seconds=1.),self.checkpoint.checkpoint_sha256,
            self.checkpoint.showdown_source_sha256)
        panel=None
        try:
            phase("selection")
            panel=collect_owned_exploration_panel(ledger,stage="A1",configuration=cfg.identity,
                config=self.config,checkpoint_contract=self.checkpoint,bridge_runtime=self.source_runtime,
                source_review_path=self.source_review_path,history_review_path=self.history_review_path,
                _synthetic_factory=AuthoredCachedSourceFactory)
            assert len(panel.by_root)==2, (len(panel.by_root),list(panel.by_root))
            plans,gaps,raws,searches={},{},{},{}
            launches=dict(raw=0,search=0,reused=0,continuation=0)
            for root,source_owner in panel.by_root.items():
                common=dict(archive=source_owner.archive,source=source_owner.source,source_owner=source_owner,
                    root_id=root,checkpoint_contract=self.checkpoint,showdown_root=self.checkpoint.showdown_root,
                    file_hashes=self.files,history_review_path=self.history_review_path,
                    source_config=self.config,source_review_path=self.source_review_path)
                raw,receipt=produce_archived_selector(ledger,"A1",cfg.identity,**common,runtime=control,selector="raw",
                    qualification_review_path=self.configuration_reviews["A1"],
                    _synthetic_job_class=ActualAuthoredRawJob,_synthetic_job_kwargs={})
                raws[root]=raw;launches["raw"]+=int(receipt["selector_launched"])
                invalid=dict(common,source_config=replace(self.config,read_timeout_seconds=6.))
                with unittest.TestCase().assertRaisesRegex(ValueError,"source configuration/checkpoint"):
                    produce_archived_selector(ledger,"A1",cfg.identity,**invalid,runtime=runtime,
                        qualification_review_path=self.configuration_reviews["A1"],_synthetic_actual_actor=True)
                assert not _actor_unavailable_path(ledger,root,cfg.identity).exists()
                search,receipt=produce_archived_selector(ledger,"A1",cfg.identity,**common,runtime=runtime,
                    qualification_review_path=self.configuration_reviews["A1"],_synthetic_actual_actor=True)
                if search is None:
                    gaps[root]=receipt
                    assert receipt["observed_zero_work"] is False and receipt["selector_launched"] is False
                else:
                    searches[root]=search;launches["search"]+=1
                    assert search.reader.last_validation["actual_work"]["accepted_units"]>0
                    plans[root]=ArchivedRootEvaluationPlan(source_owner.archive,source_owner.source,
                        contract=ledger.contract,root_id=root,configurations=(SearchConfiguration("raw",seconds=1.),cfg),
                        selections={"raw":raw.selection,cfg.identity:search.selection},
                        selector_readers={"raw":raw.reader,cfg.identity:search.reader},source_owner=source_owner,
                        allow_synthetic=True)
                reused,reuse=produce_archived_selector(ledger,"A2",cfg.identity,**common,runtime=runtime,
                    qualification_review_path=self.configuration_reviews["A2"],prior_owner=search,
                    _synthetic_actual_actor=True)
                assert reused is search and reuse["selector_launched"] is False
                launches["reused"]+=1
            assert len(plans)==len(gaps)==1
            rows={}
            for stage in ("A1","A2"):
                rows[stage]=ledger.record_owned_stage_evidence(stage,cfg.identity,plans=plans,
                    continuations={r:None for r in plans},source_panel=panel,input_unavailable=gaps,
                    qualification_review_path=self.configuration_reviews[stage])
                assert len(ledger.contract["panels"]["exploration"]["root_slots"])==200
                assert set(rows[stage]["root_intervals"])==set(plans) and not set(gaps)&set(rows[stage]["root_intervals"])
                with patch.object(ledger,"_stage_configs",return_value=(cfg,)):
                    ledger._rows(stage)  # Completed durable partition readback, no private cache reconstruction.
            for root,receipt in gaps.items():
                read_recorded_actor_input_unavailable(ledger,receipt,configuration=cfg)
                for field,value in (("source_seed",True),("selector_attempted",0),
                        ("observed_zero_work",True),("source_sha256","f"*64),("reason","unexpected_error")):
                    wrong=deepcopy(receipt);wrong[field]=value
                    with unittest.TestCase().assertRaises(ValueError):
                        read_recorded_actor_input_unavailable(ledger,wrong,configuration=cfg)
                path=_actor_unavailable_path(ledger,root,cfg.identity)
                original=path.read_bytes();changed=deepcopy(receipt);changed["reason"]="actor_tracker_absent"
                try:
                    path.write_text(json.dumps(changed))
                    with unittest.TestCase().assertRaisesRegex(ValueError,"binding drift"):
                        read_recorded_actor_input_unavailable(ledger,receipt,configuration=cfg)
                finally:path.write_bytes(original)
                claim=path.with_name(path.name.replace("-input-unavailable.json",".json"))
                try:
                    claim.write_text("{}")
                    with unittest.TestCase().assertRaisesRegex(ValueError,"candidate absence"):
                        read_recorded_actor_input_unavailable(ledger,receipt,configuration=cfg)
                finally:claim.unlink()
                verify=ledger.verify
                def late_claim():
                    verify()
                    if sys._getframe(1).f_code.co_name=="read_actor_input_unavailable":claim.write_text("{}")
                try:
                    with patch.object(ledger,"verify",new=late_claim):
                        with unittest.TestCase().assertRaisesRegex(ValueError,"changed during callbacks"):
                            read_actor_input_unavailable(ledger,panel.by_root[root],receipt,configuration=cfg)
                finally:
                    if claim.exists():claim.unlink()
            panel.validate()
            for owner in panel.owners:owner.close()
            with patch.object(ledger,"_stage_configs",return_value=(cfg,)):
                for stage in ("A1","A2"):ledger._rows(stage)
            for receipt in gaps.values():read_recorded_actor_input_unavailable(ledger,receipt,configuration=cfg)
            return dict(launches=launches,owned_roots=len(plans),unavailable_roots=len(gaps),
                source_missing_roots=len(panel.missing_root_ids),rows=rows,
                scientific_measurements=0,collection_authorized=False,holdout_authorized=False)
        except BaseException as error:
            (Path(ownership["directory"]).parent/"fixture-error.txt").write_text(traceback.format_exc())
            raise
        finally:
            phase("cleanup")
            if panel is not None:
                for owner in panel.owners:owner.close()


@unittest.skipUnless(_crate_ready,"requires existing isolated native runtime")
class ActorRouteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):native_fixture.LoadedNativeValueAdapterTests.setUpClass()
    @classmethod
    def tearDownClass(cls):native_fixture.LoadedNativeValueAdapterTests.tearDownClass()
    def test_owned_available_and_unavailable_roots_with_actual_selectors_and_a1_a2_reuse(self):
        helper=native_fixture.LoadedNativeValueAdapterTests();self.addCleanup(helper.doCleanups)
        env,_=helper.authored_context()
        lines,request=helper.authored_protocol()
        lines[:2]=["|player|p1|PokeZero p1|","|player|p2|PokeZero p2|"]
        other=deepcopy(request)
        other["side"]["id"]="p2"
        for pokemon in other["side"]["pokemon"]:pokemon["ident"]=pokemon["ident"].replace("p1:","p2:")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            settings=dict(mode="complete",rounds=1,snapshotTransport=True,actorPrefix=lines,
                actorRequests={"p1":request,"p2":other},availableSeeds=[showdown_seed_from_int(i) for i in (0,1)])
            settings_path=root/"authored-settings.json";settings_path.write_text(json.dumps(settings))
            script=Path(__file__).with_name("source_bridge_fixture.mjs").read_text()
            old='`${root}/dist/sim/index.js`'
            assert script.count(old)==1
            script=script.replace(old,json.dumps(str(settings_path)))
            script=script.replace('let token, n = 0;','let token, n = 0; let seed;')
            script=script.replace('token = command.battleId;','token = command.battleId; seed=command.seed;')
            script=script.replace('const value = structuredClone(settings.actorRequests[player]);',
                'if (player === "p1" && !settings.availableSeeds.includes(seed))\n'
                '  return {wait:true,side:{id:"p1"}};\n'
                'const value = structuredClone(settings.actorRequests[player]);')
            # Other authored sources request only p2 at their one boundary;
            # no p1 record exists. Forced p1 moves are still catalog records
            # under the source law and must not be invented as missing roots.
            script=script.replace('requested:["p1","p2"]',
                'requested:(settings.availableSeeds.includes(seed) ? ["p1","p2"] : ["p2"])')
            bridge=root/"battle_bridge.mjs";bridge.write_text(script)
            scripts=Path(__file__).resolve().parents[1]/"scripts"
            for name in ("battle_bridge_owned_continuation.mjs","battle_bridge_owned_snapshot.mjs"):
                (root/name).write_bytes((scripts/name).read_bytes())
            config=replace(env.config,bridge_path=str(bridge),read_timeout_seconds=5.)
            checkpoint=helper.checkpoint_contract
            source_runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,
                extra_files=(checkpoint.model_path,checkpoint.tables_path))
            h=fresh_fixture.FreshLifecycleTests();h.setUp();self.addCleanup(h.doCleanups)
            h.raw_sha=digest(raw_runtime(SearchConfiguration("raw",seconds=1.),checkpoint.checkpoint_sha256,
                checkpoint.showdown_source_sha256));h.review["raw_runtime_sha256"]=h.raw_sha
            evidence=root/"authored-evidence.json";evidence.write_text('{"scientific":false}')
            evidence_hash={str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()}
            source_path=root/"source-review.json"
            review=source_review_template(h.contract,source_recipe(config,checkpoint,source_runtime),
                qualification_evidence_hashes=evidence_hash)
            review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
                source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
            source_path.write_text(json.dumps(review))
            cfg=SearchConfiguration("incumbent",seconds=1.);runtime=actor_timed_runtime_recipe(cfg,checkpoint)
            paths={};extra={**evidence_hash,str(source_path):hashlib.sha256(source_path.read_bytes()).hexdigest()}
            for stage in ("A1","A2"):
                value=configuration_review_template(h.contract,stage=stage,configuration=cfg.identity,
                    raw_runtime_sha256=h.raw_sha,selector_recipe_sha256=digest(selector_runtime_recipe(cfg,runtime)),
                    qualification_evidence_hashes=evidence_hash,cost_scope="Authored source protocol and actual random selectors only.")
                value.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
                    atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
                path=root/(stage+"-review.json");path.write_text(json.dumps(value));paths[stage]=str(path)
                extra[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(fresh_fixture,"controller_input_hashes",return_value={**controller_input_hashes(),**extra}):
                h.initialize()
            from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
            import pokezero_search.pokezero_search as native_search
            import poke_engine.poke_engine as native_engine
            import sys
            files={**fixed_work_reader_source_hashes(),**source_runtime["files"]}
            for path in (Path(sys.executable).resolve(),Path(native_search.__file__),Path(native_engine.__file__)):
                files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
            job=ActorRouteAuditor(str(h.directory),checkpoint,config,source_runtime,str(source_path),
                str(h.review_path),paths,files)
            directory=root/"auditor"
            terminal=supervise_fixed_work(job,directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            error=directory/"fixture-error.txt"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            result=terminal["result"]
            self.assertEqual(result["launches"],dict(raw=2,search=1,reused=2,continuation=0))
            self.assertEqual((result["owned_roots"],result["unavailable_roots"],result["source_missing_roots"]),(1,1,198))
            self.assertFalse(result["collection_authorized"])
