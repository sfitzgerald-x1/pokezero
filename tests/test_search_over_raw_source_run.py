"""Real owned driver/bridge with authored states and policies, no science."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import pickle
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig,LocalShowdownEnv,PublicBattleMaterializationState
from pokezero.belief import PublicBattleBeliefEngine
from pokezero.showdown import parse_showdown_replay
from pokezero.mcts_eval.resolver import CheckpointContract
from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_continuation_admission import initialize_fresh_namespace
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_raw_run import raw_runtime
from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger,controller_input_hashes
from pokezero.mcts_eval.search_over_raw_source_run import (
    collect_owned_exploration_source,source_recipe,source_review_template,read_source_owner,
)
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from pokezero.mcts_eval.paper_reference_parallel import live_owned_groups
from test_search_over_raw_source import SourceEnv,FakeChampion
import test_search_over_raw_continuation_admission as admission_fixture


@dataclass(frozen=True)
class AuthoredConfig:
    window_size: int = 1


@dataclass(frozen=True)
class AuthoredSourceSnapshot:
    audit_only: dict
    bridge_snapshot: dict


class AuthoredPolicy(FakeChampion):
    def __init__(self, checkpoint, mode):
        super().__init__();self.weights_sha256=checkpoint.checkpoint_sha256
        self.result.model_config=AuthoredConfig();self.mode=mode
    def select_action(self,observation,*,rng):
        if self.mode == "policy_hang":signal.pause()
        return super().select_action(observation,rng=rng)


class AuthoredSourceEnv(LocalShowdownEnv):
    """Actual start/send/read/pumps/wait; authored state and rollout surfaces."""
    def __init__(self, config):
        super().__init__(config)
        info=json.loads((Path(config.resolved_showdown_root())/"dist/sim/index.js").read_text())
        self.rounds,self.mode,self.n=info["rounds"],info["mode"],0;self.calls=[]
        self.snapshot_transport=info.get("snapshotTransport",False)
    def _bridge_request_boundary(self,payload):
        self._send_command(payload)
        return self._read_until_event_type("ready")
    def observe(self,seat):
        observation=SourceEnv.observe(self,seat)
        return replace(observation,metadata={**observation.metadata,"self_team":[],"opponent_team":[]})
    def requested_players(self):return SourceEnv.requested_players(self)
    def terminal(self):return SourceEnv.terminal(self)
    winner="p1"
    def step(self,actions):
        self._send_command(dict(type="choices",battleId=self._battle_token,
            choices={seat:"move "+str(action+1) for seat,action in actions.items()}))
        self._read_until_event_type("terminal" if self.n+1 == self.rounds else "ready")
        return SourceEnv.step(self,actions)
    def public_materialization_state(self,seat):
        engine=PublicBattleBeliefEngine(format_id="gen3randombattle",
            set_source=SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64)))
        return PublicBattleMaterializationState(seat,"gen3randombattle","gen3randombattle",
            parse_showdown_replay([],battle_id="authored-source"),engine,
            self_request={"active":[{"moves":[{"id":"tackle","pp":15},{"id":"protect","pp":15}]}]})
    def snapshot_actionable_boundary(self):
        if self.snapshot_transport:
            event=self._bridge_request_event(dict(type="snapshot_actionable_boundary",battleId=self._battle_token),
                "snapshot_actionable_boundary")
            return AuthoredSourceSnapshot({"canary":"private-source-driver-canary","n":self.n},event["snapshot"])
        from test_search_over_raw_archive import AuthoredPrivateBoundary
        return AuthoredPrivateBoundary({"canary":"private-source-driver-canary","n":self.n})


class AuthoredSourceFactory:
    synthetic_process_fixture=True
    def __new__(cls,config,checkpoint):
        env=AuthoredSourceEnv(config)
        return env,{seat:AuthoredPolicy(checkpoint,env.mode) for seat in ("p1","p2")}


class AuthoredActorSourceEnv(AuthoredSourceEnv):
    """Real checkpoint encoder over authored messages; no model/simulator."""
    def observe(self, seat):
        from test_showdown import _phase2_fake_dex
        with patch("pokezero.local_showdown.load_showdown_dex_cached", return_value=_phase2_fake_dex()):
            return LocalShowdownEnv.observe(self, seat)


class AuthoredActorSourceFactory:
    synthetic_process_fixture=True
    def __new__(cls, config, checkpoint):
        env=AuthoredActorSourceEnv(config)
        return env,{seat:AuthoredPolicy(checkpoint,env.mode) for seat in ("p1","p2")}


@dataclass(frozen=True)
class AuthoredSourceJob:
    ledger_directory: str
    source_review_path: str
    history_review_path: str
    config: object
    checkpoint: object
    runtime: dict
    seed: int
    mode: str="complete"
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        ledger=ReconciledPhaseALedger(self.ledger_directory)
        phase("startup");phase("selection")
        handle=None;panel=None
        try:
            import pokezero.mcts_eval.search_over_raw_source_run as source_run
            if self.mode in {"full_panel","staged_empty_panel"}:
                panel=source_run.collect_owned_exploration_panel(ledger,stage="A1",
                    configuration=SearchConfiguration("incumbent","public","model",1.).identity,
                    config=self.config,checkpoint_contract=self.checkpoint,bridge_runtime=self.runtime,
                    source_review_path=self.source_review_path,history_review_path=self.history_review_path,
                    _synthetic_factory=AuthoredSourceFactory)
                if self.mode == "staged_empty_panel":
                    from dataclasses import asdict
                    from pokezero.mcts_eval.search_over_raw_stage_run import (
                        StageSelectorRecipe,exploration_recipe_keys,run_owned_exploration_stages,
                    )
                    from pokezero.mcts_eval.search_over_raw_reconciled import (
                        a1_configurations,a2_configurations,deployment_configurations,
                    )
                    from test_search_over_raw_outcome_run import SyntheticContinuationJob
                    class UnusedTransport:
                        synthetic_process_fixture=True
                        def __call__(self,**kwargs):raise AssertionError("empty panel launched a selector")
                    configs={c.identity:c for c in a1_configurations()+a2_configurations()+deployment_configurations()}
                    # No selector is attempted; these are explicitly authored
                    # unused recipe placeholders, not qualification clearances.
                    recipes={key:StageSelectorRecipe(runtime=dict(configuration=asdict(configs[key[1]])),
                        qualification_review_path=self.source_review_path) for key in exploration_recipe_keys()}
                    result=run_owned_exploration_stages(ledger,panel,recipes=recipes,
                        checkpoint_contract=self.checkpoint,showdown_root=str(self.config.resolved_showdown_root()),
                        file_hashes=self.runtime["files"],raw_runtime={},history_review_path=self.history_review_path,
                        continuation_config=self.config,continuation_runtime=self.runtime,set_source=None,
                        _synthetic_selector_transport=UnusedTransport(),
                        _synthetic_continuation_job_class=SyntheticContinuationJob)
                    return result
                try:pickle.dumps(panel)
                except TypeError:pass
                else:raise ValueError("private source panel unexpectedly serialized")
                # Partial source lists and omitted available roots cannot clear.
                try:source_run.TrustedExplorationPanel(ledger,panel.owners[:-1])
                except ValueError:pass
                else:raise ValueError("partial source roster accepted")
                try:panel.require_stage_inputs(ledger,{})
                except ValueError:pass
                else:raise ValueError("available root omission accepted")
                receipt=panel.receipt()
                original=panel.owners[-1].archive.selected
                def late_selected(root):
                    value=original(root)
                    first=panel.owners[0]
                    index=first.source["roots"][0]["source_request_index"]
                    first.archive._private[index].audit_only["n"]=-1
                    return value
                panel.owners[-1].archive.selected=late_selected
                try:panel.validate()
                except ValueError as failure:
                    if not any(message in str(failure) for message in (
                            "original retained source archive changed","original fixed panel identities changed")):raise
                else:raise ValueError("later source callback mutation accepted")
                return dict(source_count=receipt["source_count"],roots=len(receipt["observed_root_ids"]),
                    missing=len(receipt["missing_root_ids"]),source_roster_complete=receipt["source_roster_complete"],
                    partial_roster_rejected=True,omitted_roots_rejected=True,cross_source_mutation_rejected=True,
                    scientific_measurements=0,collection_authorized=False,holdout_authorized=False)
            original=source_run.archived_exploration_root_binding
            def initial_callback(*args,**kwargs):
                value=original(*args,**kwargs)
                path=ledger.directory/"source-attempts"/str(self.seed)/"driver.json"
                data=json.loads(path.read_text());data["trajectory_sha256"]="f"*64
                path.write_text(json.dumps(data))
                return value
            scope=patch.object(source_run,"archived_exploration_root_binding",side_effect=initial_callback) \
                if self.mode == "mutate_initial_receipt" else nullcontext()
            with scope:
                handle=collect_owned_exploration_source(ledger,stage="A1",configuration=SearchConfiguration("incumbent", "public", "model", 1.).identity,
                    source_seed=self.seed,config=self.config,checkpoint_contract=self.checkpoint,bridge_runtime=self.runtime,
                    source_review_path=self.source_review_path,history_review_path=self.history_review_path,
                    _synthetic_factory=None if self.mode == "production_not_admitted" else
                        AuthoredActorSourceFactory if self.mode in {"actor_history","capture_unreviewed"} else AuthoredSourceFactory,
                    capture_actor_history=self.mode in {"actor_history","capture_unreviewed"})
            for obj in (handle,handle.archive):
                try:pickle.dumps(obj)
                except TypeError:pass
                else:raise ValueError("private source unexpectedly serialized")
            if self.mode == "public_history":
                from pokezero.mcts_eval.search_over_raw_source_public_history import rebuild_public_fold
                histories=[]
                for root in handle.source["roots"]:
                    history=handle.public_history(root["root_id"])
                    replay,belief=rebuild_public_fold(history,expected_history_sha256=digest(history))
                    if replay.requests or belief.set_source is not None:
                        raise ValueError("public fold acquired original private requests/catalog")
                    histories.append(history)
                return dict(roots=len(histories),public_histories=histories,scientific_measurements=0,
                    labels_produced=False,collection_authorized=False,holdout_authorized=False)
            if self.mode == "actor_history":
                receipts=[]
                for root in handle.source["roots"]:
                    # Authored static dex only; actual transport/call chronology
                    # and owning handle are not replaced by fabricated input.
                    from test_showdown import _phase2_fake_dex
                    with patch("pokezero.local_showdown.load_showdown_dex_cached", return_value=_phase2_fake_dex()):
                        result=handle.actor_program(root["root_id"],config=self.config)
                    if not result.receipt["process_ownership_validated"] or not result.receipt["source_prefix_join_validated"]:
                        raise ValueError("source actor ownership/snapshot join was not checked")
                    receipts.append(result.receipt)
                return dict(roots=len(receipts),actor_receipts=receipts,scientific_measurements=0,
                    labels_produced=False,collection_authorized=False,holdout_authorized=False)
            if self.mode == "retry":
                collect_owned_exploration_source(ledger,stage="A1",configuration=SearchConfiguration("incumbent", "public", "model", 1.).identity,
                    source_seed=self.seed,config=self.config,checkpoint_contract=self.checkpoint,bridge_runtime=self.runtime,
                    source_review_path=self.source_review_path,history_review_path=self.history_review_path,
                    _synthetic_factory=AuthoredSourceFactory)
            if self.mode == "mutate_archive":
                root=handle.source["roots"][0]
                handle.archive._private[root["source_request_index"]].audit_only["n"]=-1
            if self.mode == "mutate_receipt_callback":
                original=handle.archive.selected
                def selected(root):
                    value=original(root)
                    path=handle.directory/"driver.json"
                    data=json.loads(path.read_text());data["trajectory_sha256"]="f"*64
                    path.write_text(json.dumps(data))
                    return value
                handle.archive.selected=selected
            if self.mode == "mutate_earlier_root_callback":
                original=handle.archive.selected
                first,last=handle.source["roots"][0],handle.source["roots"][-1]
                def selected(root):
                    value=original(root)
                    if root["root_id"]==last["root_id"]:
                        handle.archive._private[first["source_request_index"]].audit_only["n"]=-1
                    return value
                handle.archive.selected=selected
            if self.mode == "replace_ledger":
                handle.ledger=ReconciledPhaseALedger(self.ledger_directory)
            if self.mode == "replace_runtime":
                handle.runtime=deepcopy(handle.runtime)
            if self.mode == "replace_archive":
                from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
                other=SealedSourceArchive()
                other.__dict__.update(handle.archive.__dict__)
                handle.archive=other
            handle.validate()
            if handle.source["status"] == "UNCERTAIN_SOURCE":
                if handle.archive._public or handle.archive._private:
                    raise AssertionError("capped source retained unusable private requests")
            return dict(status=handle.source["status"],roots=len(handle.source["roots"]),
                missing=len(handle.source["missing_root_ids"]),private_archive_returned=False,
                scientific_measurements=0,collection_authorized=False,holdout_authorized=False)
        except BaseException as error:
            # Authored inputs only; production workers retain sanitized errors.
            (Path(ownership["directory"]).parent/"authored-error.json").write_text(
                json.dumps(dict(error=type(error).__name__,message=str(error),notes=getattr(error,"__notes__",[]))))
            raise
        finally:
            phase("cleanup")
            if handle is not None:handle.close()
            if panel is not None:
                for owner in panel.owners:owner.close()


class SourceRunTests(unittest.TestCase):
    def setUp(self):
        self.h=admission_fixture.FreshLifecycleTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.root=self.h.base.root.resolve()

    def prepare(self,*,rounds=12,mode="complete",job_mode="complete"):
        showdown=self.root/"source-showdown";sim=showdown/"dist/sim";sim.mkdir(parents=True)
        actor_capture=job_mode in {"actor_history","capture_unreviewed"}
        snapshots=job_mode=="public_history" or actor_capture
        settings=dict(rounds=rounds,mode=mode,snapshotTransport=snapshots)
        if actor_capture:
            import test_search_over_raw_actor_projection as actors
            settings.update(actorPrefix=actors.prefix("p1"),
                actorRequests={"p1":actors.own_request(pp=56),"p2":actors.own_request("p2",active="Snorlax",hp="350/350",pp=56)})
        (sim/"index.js").write_text(json.dumps(settings))
        (sim/"state.js").write_text("state.log = battle.log;" if snapshots else "authored-not-engine")
        if snapshots:
            official=Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/battle.js").read_text()
            start=official.index("const splitRegex = ");stop=official.index("\nclass Battle {",start)
            (sim/"battle.js").write_text(official[start:stop]+"\nclass Battle {\n}")
        bridge_dir=self.root/"source-bridge";bridge_dir.mkdir()
        script_root=Path(__file__).resolve().parents[1]/"scripts"
        (bridge_dir/"battle_bridge.mjs").write_bytes(Path(__file__).with_name("source_bridge_fixture.mjs").read_bytes())
        for name in ("battle_bridge_owned_snapshot.mjs","battle_bridge_owned_continuation.mjs"):
            (bridge_dir/name).write_bytes((script_root/name).read_bytes())
        checkpoint=self.root/"authored-source-checkpoint";checkpoint.write_bytes(b"not-a-source-model")
        self.config=LocalShowdownConfig(showdown_root=str(showdown),bridge_path=str(bridge_dir/"battle_bridge.mjs"),
            set_belief_source=False,read_timeout_seconds=.5)
        if actor_capture:
            from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC
            from pokezero.randbat_vocab import gen3_category_vocabulary
            vocab=gen3_category_vocabulary(LocalShowdownConfig().resolved_showdown_root(),include_turn_merged=True,extra_moves=("tackle",))
            self.config=replace(self.config,observation_spec=V2_2_REPLAY_OBSERVATION_SPEC,category_vocab=vocab)
        self.checkpoint=CheckpointContract(checkpoint_path=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            policy_id="authored",schema_version="authored",token_count=1,categorical_feature_count=1,numeric_feature_count=1,
            transition_token_count=0,category_vocab=(),architecture={"window_size":1},feature_masks={},model_device="cpu",
            showdown_root=str(showdown),showdown_source_sha256="a"*64)
        self.runtime=outcome_runtime_binding(self.config,checkpoint_path=checkpoint)
        self.h.raw_sha=digest(raw_runtime(SearchConfiguration("raw",seconds=1.),self.checkpoint.checkpoint_sha256,"a"*64))
        self.h.review["raw_runtime_sha256"]=self.h.raw_sha
        qualification=self.root/"source-engineering-review.txt";qualification.write_text("authored only; not production qualification")
        self.source_review=self.root/"source-review.json"
        review=source_review_template(self.h.contract,source_recipe(self.config,self.checkpoint,self.runtime,
            capture_actor_history=job_mode=="actor_history"),
            qualification_evidence_hashes={str(qualification):hashlib.sha256(qualification.read_bytes()).hexdigest()})
        review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
            source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
        self.source_review.write_text(json.dumps(review))
        original=admission_fixture.controller_input_hashes
        with patch.object(admission_fixture,"controller_input_hashes",return_value={**original(),
            str(self.source_review):hashlib.sha256(self.source_review.read_bytes()).hexdigest()}):self.h.initialize()
        self.seed=self.h.contract["panels"]["exploration"]["seeds"][0]
        self.directory=self.root/"source-auditor"
        self.job=AuthoredSourceJob(str(self.h.directory),str(self.source_review),str(self.h.review_path),
            self.config,self.checkpoint,self.runtime,self.seed,job_mode)
        return self.job

    def run_job(self,job,**kwargs):
        return supervise_fixed_work(job,directory=self.directory,binding_sha256="c"*64,
            verify_admission=lambda:True,**kwargs)

    def test_actual_driver_retains_exact_archive_closes_bridge_and_redacts_result(self):
        result=self.run_job(self.prepare())
        diagnostic=self.directory/"authored-error.json"
        self.assertEqual(result["supervisor_exit_code"],0,json.loads(diagnostic.read_text()) if diagnostic.exists() else result)
        self.assertEqual((result["result"]["roots"],result["result"]["missing"]),(7,0))
        source_dir=self.h.directory/"source-attempts"/str(self.seed)
        source=json.loads((source_dir/"source.json").read_text())
        readout=read_source_owner(source_dir,source,self.runtime,self.h.contract)
        self.assertTrue(readout["source_pipeline_validated"])
        self.assertFalse(readout["full_resource_cost_complete"])
        for path in (self.directory/"worker-result.json",source_dir/"bridge-transport.json"):
            self.assertNotIn("private-source-driver-canary",path.read_text())
        closed=json.loads((source_dir/"bridge-close.json").read_text())
        self.assertEqual(closed["exit_code"],0);self.assertTrue(closed["pumps_drained"])
        self.assertFalse(closed["forced_cleanup"])

    def test_short_source_keeps_fixed_missing_quota(self):
        result=self.run_job(self.prepare(rounds=2))
        self.assertEqual(result["supervisor_exit_code"],0,result)
        self.assertEqual((result["result"]["roots"],result["result"]["missing"]),(2,5))

    def test_capture_requires_its_prospective_review_before_source_claim(self):
        result=self.run_job(self.prepare(rounds=2,job_mode="capture_unreviewed"))
        self.assertNotEqual(result["supervisor_exit_code"],0)
        self.assertFalse((self.h.directory/"source-attempts").exists())
        error=json.loads((self.directory/"authored-error.json").read_text())
        self.assertIn("mismatched source recipe review",error["message"])

    def test_owned_actor_stream_and_actual_inputs_join_each_original_selected_boundary(self):
        result=self.run_job(self.prepare(rounds=3,job_mode="actor_history"))
        diagnostic=self.directory/"authored-error.json"
        self.assertEqual(result["supervisor_exit_code"],0,json.loads(diagnostic.read_text()) if diagnostic.exists() else result)
        receipts=result["result"]["actor_receipts"]
        self.assertEqual(len(receipts),3)
        self.assertTrue(all(row["process_ownership_validated"] and row["source_prefix_join_validated"] for row in receipts))
        self.assertTrue(all(row["status"] in {"INPUT_CORRESPONDENCE_MATCHED","ENCODING_MISMATCH"} for row in receipts))
        self.assertTrue(all(row["actual_raw_input_binding_validated"] for row in receipts))
        self.assertTrue(all(not row["original_source_p2_history_used"] and not row["independent_terminal_labels"] for row in receipts))
        source_dir=self.h.directory/"source-attempts"/str(self.seed)
        trace=json.loads((source_dir/"bridge-transport.json").read_text())["records"]
        policy_rows=[r for r in trace if r["kind"]=="actor_policy_input"]
        self.assertEqual(len(policy_rows),3)
        self.assertEqual(sorted(row["raw_input_count"] for row in receipts),[1,2,3])
        self.assertEqual(len({row["cutoff_record_index"] for row in receipts}),3)
        self.assertFalse(result["result"]["labels_produced"])

    def test_actual_snapshot_transport_joins_archive_and_builds_no_leak_public_prefix(self):
        result=self.run_job(self.prepare(job_mode="public_history"))
        diagnostic=self.directory/"authored-error.json"
        self.assertEqual(result["supervisor_exit_code"],0,json.loads(diagnostic.read_text()) if diagnostic.exists() else result)
        self.assertEqual(result["result"]["roots"],7)
        rendered=json.dumps(result["result"])
        for secret in ("SOURCE_SECRET_P1","SOURCE_SECRET_P2","SOURCE_PRIVATE_POKEMON","SOURCE_PRIVATE_MOVE",
                "private-source-driver-canary"):
            self.assertNotIn(secret,rendered)
        source_dir=self.h.directory/"source-attempts"/str(self.seed)
        source=json.loads((source_dir/"source.json").read_text())
        readout=read_source_owner(source_dir,source,self.runtime,self.h.contract)
        self.assertTrue(readout["source_snapshot_transport_validated"])
        trace=json.loads((source_dir/"bridge-transport.json").read_text())["records"]
        snapshots=[r for r in trace if r["kind"]=="event" and r["type"]=="snapshot_actionable_boundary"]
        self.assertEqual([r["decision_round_index"] for r in snapshots],list(range(12)))
        self.assertEqual(len({r["snapshot_sha256"] for r in snapshots}),12)
        self.assertFalse(result["result"]["labels_produced"])

    def test_full_fixed_authored_panel_keeps_short_quotas_and_original_owners(self):
        result=self.run_job(self.prepare(rounds=2,job_mode="full_panel"))
        diagnostic=self.directory/"authored-error.json"
        self.assertEqual(result["supervisor_exit_code"],0,json.loads(diagnostic.read_text()) if diagnostic.exists() else result)
        self.assertEqual(result["result"]["source_count"],32)
        self.assertEqual((result["result"]["roots"],result["result"]["missing"]),(64,136))
        self.assertTrue(result["result"]["cross_source_mutation_rejected"])
        self.assertEqual(len(list((self.h.directory/"source-attempts").iterdir())),32)

    def test_normally_capped_source_has_all_original_slots_missing(self):
        result=self.run_job(self.prepare(rounds=300))
        self.assertEqual(result["supervisor_exit_code"],0,result)
        self.assertEqual(result["result"]["status"],"UNCERTAIN_SOURCE")
        self.assertEqual((result["result"]["roots"],result["result"]["missing"]),(0,7))
        self.assertFalse((self.h.directory/"operational-stop.json").exists())

    def test_actual_all_cap_32_source_panel_scheduler_keeps_all_200_slots_unstarted(self):
        result=self.run_job(self.prepare(rounds=300,job_mode="staged_empty_panel"))
        diagnostic=self.directory/"authored-error.json"
        self.assertEqual(result["supervisor_exit_code"],0,json.loads(diagnostic.read_text()) if diagnostic.exists() else result)
        summary=result["result"]
        self.assertEqual(summary["source_panel"]["source_count"],32)
        self.assertEqual(len(summary["source_panel"]["missing_root_ids"]),200)
        self.assertEqual(summary["source_panel"]["observed_root_ids"],[])
        for stage in ("A1","A2"):
            self.assertEqual(len(summary["reports"][stage]["configurations"]),4)
            for row in summary["reports"][stage]["configurations"]:
                self.assertEqual(row["status"],"UNSTARTED")
                self.assertEqual(row["uncertain_roots"],200)
                self.assertEqual(row["identification_interval"],[-1.,1.])
        self.assertEqual(summary["reports"]["A3"]["configurations"],[])
        self.assertFalse(summary["holdout_authorized"])
        self.assertEqual(summary["launches"]["raw"],0)
        self.assertFalse((self.h.directory/"operational-stop.json").exists())

    def test_production_not_admitted_before_source_claim_or_dependency_load(self):
        result=self.run_job(self.prepare(job_mode="production_not_admitted"))
        self.assertEqual(result["supervisor_exit_code"],1,result)
        self.assertFalse((self.h.directory/"source-attempts").exists())

    def test_second_source_attempt_cannot_reset_or_redraw(self):
        result=self.run_job(self.prepare(job_mode="retry"))
        self.assertEqual(result["supervisor_exit_code"],1,result)
        self.assertEqual(len(list((self.h.directory/"source-attempts").iterdir())),1)
        source_dir=self.h.directory/"source-attempts"/str(self.seed)
        self.assertTrue((source_dir/"source.json").exists())
        self.assertEqual(len([r for r in json.loads((source_dir/"bridge-transport.json").read_text())["records"]
            if r["kind"] == "command" and r["type"] == "start"]),1)

    def test_retained_archive_mutation_is_not_validated_source(self):
        result=self.run_job(self.prepare(job_mode="mutate_archive"))
        self.assertEqual(result["supervisor_exit_code"],1,result)

    def test_later_root_callback_cannot_mutate_an_earlier_validated_archive(self):
        result=self.run_job(self.prepare(job_mode="mutate_earlier_root_callback"))
        self.assertEqual(result["supervisor_exit_code"],1,result)
        self.assertRegex(json.loads((self.directory/"authored-error.json").read_text())["message"],
            "original retained source archive changed|source inputs changed during archive callbacks")

    def assert_bridge_failure(self,mode,*,exit_code=None,limit=None):
        job=self.prepare(mode=mode)
        kwargs={} if limit is None else {"_synthetic_watchdog_seconds":limit}
        result=self.run_job(job,**kwargs)
        self.assertEqual(result["supervisor_exit_code"],124 if mode == "policy_hang" else 1,result)
        self.assertIsNone(result["cleanup_error_type"],result)
        self.assertFalse(live_owned_groups(set(result["owned_groups"])))
        directory=self.h.directory/"source-attempts"/str(self.seed)
        self.assertTrue((directory/"claim.json").exists())
        self.assertFalse((directory/"source.json").exists())
        if mode != "policy_hang":
            closed=json.loads((directory/"bridge-close.json").read_text())
            self.assertTrue(closed["reaped"])
            self.assertTrue(closed["pumps_drained"])
            self.assertFalse(closed["normal_close"])
            self.assertTrue(closed["forced_cleanup"])
            if exit_code is not None:self.assertEqual(closed["exit_code"],exit_code)
            self.assertTrue((self.h.directory/"operational-stop.json").exists())
        else:
            self.assertTrue(result["watchdog_reached"])
        return result

    def test_nonzero_bridge_exit_is_failure_not_disappearance_success(self):
        self.assert_bridge_failure("bridge_failure",exit_code=9)

    def test_bridge_hang_is_failure_and_cleanup_reaps_actual_child(self):
        self.assert_bridge_failure("bridge_hang")

    def test_missing_stream_end_is_failure(self):
        self.assert_bridge_failure("missing_end")

    def test_wrong_close_acknowledgement_is_failure(self):
        self.assert_bridge_failure("bad_close")

    def test_close_acknowledgement_with_nonzero_exit_is_failure(self):
        self.assert_bridge_failure("nonzero_close",exit_code=7)

    def test_outer_watchdog_cleans_hung_source_policy_and_bridge(self):
        self.assert_bridge_failure("policy_hang",limit=1.5)

    def test_receipt_mutation_from_archive_callback_is_rejected(self):
        result=self.run_job(self.prepare(job_mode="mutate_receipt_callback"))
        self.assertEqual(result["supervisor_exit_code"],1,result)

    def test_initial_extraction_callback_cannot_establish_altered_receipt_baseline(self):
        result=self.run_job(self.prepare(job_mode="mutate_initial_receipt"))
        self.assertEqual(result["supervisor_exit_code"],1,result)
        error=json.loads((self.directory/"authored-error.json").read_text())
        self.assertIn("constructing archive crosswalks",error["message"])

    def test_equal_content_replacement_ledger_is_not_original_owner(self):
        result=self.run_job(self.prepare(job_mode="replace_ledger"))
        self.assertEqual(result["supervisor_exit_code"],1,result)

    def test_equal_content_replacement_runtime_is_not_original_owner(self):
        result=self.run_job(self.prepare(job_mode="replace_runtime"))
        self.assertEqual(result["supervisor_exit_code"],1,result)

    def test_equal_content_replacement_archive_is_not_original_owner(self):
        result=self.run_job(self.prepare(job_mode="replace_archive"))
        self.assertEqual(result["supervisor_exit_code"],1,result)

    def successful_source(self):
        result=self.run_job(self.prepare())
        self.assertEqual(result["supervisor_exit_code"],0,result)
        directory=self.h.directory/"source-attempts"/str(self.seed)
        return directory,json.loads((directory/"source.json").read_text())

    def test_independent_reader_rejects_rehashed_missing_teardown(self):
        directory,source=self.successful_source()
        path=directory/"bridge-transport.json";trace=json.loads(path.read_text())
        trace["records"]=[r for r in trace["records"] if r["type"] not in {"end","ended","stream_end"}]
        path.write_text(json.dumps(trace))
        closepath=directory/"bridge-close.json";closed=json.loads(closepath.read_text())
        closed["transport_sha256"]=digest(trace["records"]);closepath.write_text(json.dumps(closed))
        with self.assertRaisesRegex(ValueError,"order differs"):
            read_source_owner(directory,source,self.runtime,self.h.contract)

    def test_independent_reader_rejects_rehashed_substituted_bridge_parent(self):
        directory,source=self.successful_source()
        path=directory/"bridge-launch.json";launch=json.loads(path.read_text())
        launch["parent_pid"]+=1;launch["group"]=launch["parent_pid"];path.write_text(json.dumps(launch))
        path=directory/"bridge-close.json";closed=json.loads(path.read_text())
        closed["launch_sha256"]=digest(launch);closed["parent_pid"]=launch["parent_pid"];path.write_text(json.dumps(closed))
        with self.assertRaisesRegex(ValueError,"normally reaped source"):
            read_source_owner(directory,source,self.runtime,self.h.contract)


if __name__ == "__main__":unittest.main()
