"""Actual Python materializer/send/read, synthetic transport only; no simulator."""
from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownEnv,LocalShowdownConfig,PublicBattleMaterializationState
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_materialization import (
    materialization_inputs,MaterializationCapture,observe_materialized_world,read_materialization,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import _redact
from test_search_over_raw_construction import map_fixture
from test_engine_world import _override,_payload,_dex

BINDING=dict(root="synthetic-public-root",subject="p1")


def fixture(*,mode="complete"):
    env=LocalShowdownEnv(LocalShowdownConfig(set_belief_source=False,read_timeout_seconds=.2))
    root,_,_=map_fixture()
    raw_requests={seat:dict(active=[dict(moves=[dict(id="surf",pp=5)])],side=dict(id=seat,pokemon=[]))
        for seat in ("p1","p2")}
    root["boundaryRequests"]=deepcopy(raw_requests)
    source_request=dict(active=[dict(moves=[dict(id="surf",pp=99)])],side=dict(id="p1",pokemon=[]))
    state=PublicBattleMaterializationState(player_id="p1",format_id="gen3randombattle",
        observation_format_id="gen3randombattle",replay=replace(env._parser.snapshot(),requests={}),
        belief_engine=env._belief_engine.clone(),self_request=source_request)
    commands=[]
    def send_events(rows):
        for row in rows:env._stdout_queue.put(json.dumps(row))
    class Writer:
        def write(self,line):
            command=json.loads(line);commands.append(command);token=command.get("battleId")
            if command["type"]=="start":send_events([dict(type="started",battleId=token,
                formatid=command["formatid"],seed=command["seed"]),dict(type="ready",battleId=token,requested=["p1","p2"])])
            elif command["type"]=="materialize":
                if mode=="failure":send_events([dict(type="error",battleId=token,message="synthetic refusal")])
                else:send_events([dict(type="materialized",battleId=token,boundaryRequests=deepcopy(raw_requests),
                    requested=["p1","p2"])])
            elif command["type"]=="snapshot":
                if mode=="snapshot-refresh":root["boundaryRequests"]["p1"]["rqid"]=3
                send_events([dict(type="snapshot",battleId=token,snapshot=deepcopy(root))])
            elif command["type"]=="end":send_events([dict(type="stream_end",battleId=token,stream=s)
                for s in ("omniscient","p1","p2")]+[dict(type="ended",battleId=token)])
        def flush(self):pass
    def start():
        env._stdout_queue=queue.Queue();env._process=SimpleNamespace(stdin=Writer(),poll=lambda:None)
    env._validate_runtime=lambda:None;env._start_bridge=start
    # No sync/observation builder is involved in this authored public fixture.
    env._sync_incremental_state=lambda:None
    return env,state,_override(),commands


def run_fixture(mode="complete",options=None):
    env,state,override,commands=fixture(mode=mode)
    try:
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
            result=observe_materialized_world(env,state=state,override=override,seed=17,root_binding=BINDING,options=options)
        return env,state,override,commands,result
    finally:
        # No child exists. Do not let a fake process enter actual cleanup.
        env._process=None


class MaterializationTests(unittest.TestCase):
    def test_actual_materializer_preserves_raw_requests_not_python_actor_substitution(self):
        env,state,override,commands,result=run_fixture()
        capture=result["capture"];root=result["snapshot"].bridge_snapshot
        checked=read_materialization(capture,expected_inputs=result["inputs"],process_records=capture["records"],expected_snapshot=root)
        self.assertTrue(checked["transport_consistent"])
        self.assertEqual([v["type"] for v in commands],["start","materialize","snapshot","end"])
        self.assertEqual(root["boundaryRequests"]["p1"]["active"][0]["moves"][0]["pp"],5)
        self.assertEqual(result["snapshot"].latest_requests["p1"]["active"][0]["moves"][0]["pp"],99)
        self.assertIsNone(env._bridge_evidence_observer);self.assertIsNone(env._battle_token)
        for flag in ("process_ownership_validated","construction_provenance_validated","full_world_equivalence",
            "branch_transition_validated","runtime_qualified","scientific_collection_authorized"):
            self.assertFalse(checked[flag])
        serialized=json.dumps(capture["records"])
        self.assertNotIn('"team":',serialized);self.assertNotIn('"publicState":',serialized)
        self.assertNotIn('"boundaryRequests":',serialized)

    def test_all_actual_materializer_options_are_frozen_without_normalization(self):
        options=dict(reference_rest_sleep=True,reference_consumed_items=True,reference_turn_clocks=True,
            reference_attract=True,reference_yawn=True,reference_encore_durations={"swampert":2},
            reference_induced_sleep={},deferred_opponent_actions={},deferred_opponent_action_priors={})
        # Reference actor legality is independently tested elsewhere; this test
        # observes exact option forwarding, not those synthetic request semantics.
        with patch("pokezero.local_showdown._validate_reference_actor_request"):
            _,_,_,commands,result=run_fixture(options=options)
        self.assertEqual(commands[1],result["inputs"]["materialize"])
        self.assertEqual(result["inputs"]["options"],options)
        self.assertEqual(commands[1]["referenceRulesFormat"],"gen3randombattle")

    def test_existing_snapshot_request_refresh_is_explicit_not_hidden(self):
        *_,result=run_fixture("snapshot-refresh")
        checked=read_materialization(result["capture"],expected_inputs=result["inputs"],
            process_records=result["capture"]["records"],expected_snapshot=result["snapshot"].bridge_snapshot)
        self.assertTrue(checked["request_refresh_changed"])
        self.assertEqual(checked["snapshot_law"],"existing_snapshot_command_may_refresh_requests")

    def test_reused_shell_unknown_options_and_missing_explicit_team_rejected(self):
        env,state,override,commands=fixture()
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
            for value in (dict(unknown=True),dict(reference_yawn=1)):
                with self.assertRaises(ValueError):materialization_inputs(state=state,override=override,seed=17,
                    root_binding=BINDING,options=value)
            with self.assertRaises(ValueError):materialization_inputs(state=state,
                override=replace(override,player_teams={"p1":override.player_teams["p1"]}),seed=17,root_binding=BINDING)
            inputs=materialization_inputs(state=state,override=override,seed=17,root_binding=BINDING)
            env._battle_counter=1
            with self.assertRaises(ValueError):
                with MaterializationCapture(inputs).installed(env):pass
        self.assertEqual(commands,[])

    def test_missing_reordered_foreign_or_late_events_cannot_be_promoted(self):
        *_,result=run_fixture();root=result["snapshot"].bridge_snapshot
        for mode in ("raw-request","snapshot","command","drop-end","duplicate","late-stream","foreign","inputs"):
            changed=deepcopy(result["capture"]);inputs=deepcopy(result["inputs"])
            if mode=="raw-request":changed["raw_events"][0]["boundaryRequests"]["p1"]["wait"]=True
            if mode=="snapshot":changed["raw_events"][1]["snapshot"]["battle"]["ended"]=True
            if mode=="command":changed["records"][2]["payload"]["type"]="choices"
            if mode=="drop-end":changed["records"].pop()
            if mode=="duplicate":changed["records"].append(deepcopy(changed["records"][-1]))
            if mode=="late-stream":changed["records"].insert(-1,_redact("event",dict(type="stream",battleId="b1",stream="p1",lines=["private"])))
            if mode=="foreign":changed["records"][1]["payload"]["battleId"]="foreign"
            if mode=="inputs":inputs["materialize"]["referenceYawn"]=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                read_materialization(changed,expected_inputs=inputs,process_records=changed["records"],expected_snapshot=root)

    def test_constructor_failure_poisoned_and_observer_removed(self):
        env,state,override,commands=fixture(mode="failure")
        try:
            with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
                with self.assertRaises(Exception):observe_materialized_world(env,state=state,override=override,
                    seed=17,root_binding=BINDING)
            self.assertEqual([v["type"] for v in commands],["start","materialize"])
            self.assertIsNone(env._bridge_evidence_observer)
        finally:env._process=None

    def test_self_consistent_transport_cannot_hide_conflicting_external_inputs(self):
        *_,result=run_fixture();root=result["snapshot"].bridge_snapshot
        for mode in ("draw-hash","draw-team","start-team","seed","option","option-type",
            "deferred-option","public-hash","observation-format","token","extra"):
            inputs=deepcopy(result["inputs"]);capture=deepcopy(result["capture"])
            if mode=="draw-hash":inputs["draw_sha256"]="0"*64
            if mode=="draw-team":
                inputs["draw"]["player_teams"]["p1"]+="unconsumed"
                inputs["draw_sha256"]=digest(inputs["draw"])
            if mode=="start-team":inputs["start"]["players"]["p1"]["team"]+="unfrozen"
            if mode=="seed":inputs["seed"]+=1
            if mode=="option":inputs["options"]["reference_yawn"]=True
            if mode=="option-type":inputs["options"]["reference_yawn"]=0
            if mode=="deferred-option":inputs["options"]["deferred_opponent_actions"]={"p2":1}
            if mode=="public-hash":inputs["public_payload_sha256"]="0"*64
            if mode=="observation-format":inputs["observation_format_id"]=None
            if mode=="token":
                inputs["start"]["battleId"]=inputs["materialize"]["battleId"]="b2"
            if mode=="extra":inputs["extra"]="not-declared"
            # Forge the same input hash and command/ack trace the old reader
            # accepted: rejecting this must depend on internal input laws.
            capture["input_sha256"]=digest(inputs)
            for record in capture["records"]:
                tag=record["payload"].get("type")
                if record["kind"]=="command" and tag in {"start","materialize"}:
                    record["payload"]=_redact("command",inputs[tag])["payload"]
                elif tag=="started":
                    record["payload"].update(seed=inputs["start"]["seed"],formatid=inputs["start"]["formatid"])
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                read_materialization(capture,expected_inputs=inputs,process_records=capture["records"],expected_snapshot=root)

    def test_opaque_source_state_digest_is_not_claimed_reconstructable(self):
        *_,result=run_fixture();inputs=deepcopy(result["inputs"]);capture=deepcopy(result["capture"])
        inputs["public_state_sha256"]="0"*64;capture["input_sha256"]=digest(inputs)
        checked=read_materialization(capture,expected_inputs=inputs,process_records=capture["records"],
            expected_snapshot=result["snapshot"].bridge_snapshot)
        self.assertTrue(checked["transport_consistent"])
        self.assertFalse(checked["construction_provenance_validated"])

    def test_malformed_external_digest_bindings_are_rejected(self):
        *_,result=run_fixture()
        for key in ("public_state_sha256","public_payload_sha256","draw_sha256"):
            for value in ("","not-a-digest",True,0,None,"G"*64):
                inputs=deepcopy(result["inputs"]);inputs[key]=value
                capture=deepcopy(result["capture"]);capture["input_sha256"]=digest(inputs)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    read_materialization(capture,expected_inputs=inputs,process_records=capture["records"],
                        expected_snapshot=result["snapshot"].bridge_snapshot)


if __name__=="__main__":unittest.main()
