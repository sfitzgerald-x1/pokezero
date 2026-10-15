"""Actual disposable bridge transport and encoder; authored history/physics."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownEnv
from pokezero.policy import PolicyDecision
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_after_raw import (
    after_step_seed, run_after_raw_cell, read_after_raw_cell, AfterRawCellCapture,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_actor_projection import SCHEMA, ENCODING_LAW
import test_search_over_raw_materialization_run as material_fixture
from test_search_over_raw_actor_replay_projection import coherent_actor_root
from test_search_over_raw_source import FakeChampion, SHA
from test_search_over_raw_construction import _dex


class MaskedChampion(FakeChampion):
    def __init__(self):
        super().__init__(); self.calls=[]

    def select_action(self, observation, *, rng):
        legal=[i for i,v in enumerate(observation.legal_action_mask) if v]
        distribution=[0.] * len(observation.legal_action_mask)
        # Highest legal index is intentionally different from the first/legal
        # initial forced action in the old continuation fixture.
        action=max(legal); distribution[action]=1.
        self.calls.append(observation)
        return PolicyDecision(action,"authored",metadata={"policy_distribution":distribution})


class Trace:
    def __init__(self): self.records=[]
    def started(self,*args): pass
    def command(self,payload):
        from pokezero.mcts_eval.search_over_raw_outcome_run import _redact
        self.records.append(_redact("command",payload))
    def event(self,payload):
        from pokezero.mcts_eval.search_over_raw_outcome_run import _redact
        self.records.append(_redact("event",payload))
    def close(self,env):
        # This is not an owning-process fixture. Remove the trace before the
        # legacy close path; no reap/ownership clearance is claimed.
        env._bridge_process_observer=None
        env.close()


class AfterRawTests(unittest.TestCase):
    def setUp(self):
        h=material_fixture.MaterializationRunTests(); h.setUp(); self.addCleanup(h.doCleanups); self.h=h
        from test_search_over_raw_actor_projection import ActorProjectionTests
        vocabulary=ActorProjectionTests(); vocabulary.setUp(); self.addCleanup(vocabulary.doCleanups)
        self.h.config=replace(self.h.config,category_vocab=vocabulary.config.category_vocab,
            observation_spec=vocabulary.config.observation_spec)
        bridge=material_fixture.BRIDGE.replace("else if(c.type==='choices')", "else if(c.type==='reseed') send({type:'reseeded',battleId:b,seed:c.seed});\n else if(c.type==='choices')")
        bridge=bridge.replace("current.battle.ended=true;current.boundaryRequests={};",
            "current.battle.log.push('|win|PokeZero p1'); send({type:'stream',battleId:b,stream:'omniscient',lines:['|win|PokeZero p1']}); current.battle.ended=true;current.boundaryRequests={};")
        self.h.bridge.write_text(bridge)
        official=Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/battle.js").read_text()
        start=official.index("const splitRegex = "); stop=official.index("\nclass Battle {",start)
        sim=self.h.root/"showdown/dist/sim"
        (sim/"battle.js").write_text(official[start:stop]+"\nclass Battle {\n}")
        (sim/"state.js").write_text("state.log = battle.log;")
        self.root=coherent_actor_root()
        self.root["battle"]["log"]=["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|"]
        self.programs={}
        for seat in ("p1","p2"):
            request=deepcopy(self.root["boundaryRequests"][seat])
            actor_prefix=["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|", "|gen|3", "|start",
                "|switch|p1a: Swampert|Swampert, L84|100/100",
                "|switch|p2a: Snorlax|Snorlax, L80|100/100", "|turn|1"]
            self.programs[seat]=dict(schema=SCHEMA,actor=seat,encoding_law=ENCODING_LAW,opening_request=request,
                events=[dict(kind="protocol",lines=actor_prefix),dict(kind="own_request",request=request),
                    dict(kind="observe",coordinate=0)])
        self.h.fixture_path.write_text(json.dumps(dict(mode="complete",snapshot=self.root)))
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        initial=json.dumps(self.root,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        self.clone=dict(battleId="after-test",cloneId="after-cell",snapshotJsonSha256=hashlib.sha256(initial.encode()).hexdigest())
        self.ids=["after-test:"+str(i) for i in range(251)]
        self.policies={s:MaskedChampion() for s in ("p1","p2")}
        self.checkpoint_sha=self.runtime["files"][self.runtime["checkpoint_path"]]
        for policy in self.policies.values():policy.weights_sha256=self.checkpoint_sha
        self.dex=patch("pokezero.local_showdown.load_showdown_dex_cached",return_value=_dex())
        self.dex.start(); self.addCleanup(self.dex.stop)

    def run_cell(self,programs=None):
        env=LocalShowdownEnv(self.h.config); trace=Trace(); env._bridge_process_observer=trace
        result=run_after_raw_cell(env,raw_after=self.root,programs=self.programs if programs is None else programs,
            config=self.h.config,clone_binding=self.clone,request_ids=self.ids,sample_seed=17,event_id="a"*64,
            replicate=0,subject="p1",policies=self.policies,checkpoint_sha256=self.checkpoint_sha,runtime=self.runtime)
        self.assertIsNone(env._process)
        return result, trace.records

    def read(self,capture,records):
        # The private trace stops at release; close is a separate lifecycle.
        return read_after_raw_cell(capture,process_records=records,raw_after=self.root,programs=self.programs,
            config=self.h.config,expected_binding=deepcopy(capture.binding),runtime=self.runtime)

    def test_real_empty_bridge_raw_argmax_at_step_zero_and_independent_terminal_read(self):
        capture,records=self.run_cell()
        self.assertEqual([c["seat"] for c in capture.policy_calls],["p1","p2"])
        self.assertEqual([c["step"] for c in capture.policy_calls],[0,0])
        self.assertTrue(all(c["action"]==max(i for i,v in enumerate(c["mask"]) if v) for c in capture.policy_calls))
        self.assertTrue(all(len(p.calls)==1 for p in self.policies.values()))
        result=self.read(capture,records)
        self.assertEqual((result["status"],result["signed_outcome"],result["boundaries"]),("COMPLETE",1,1))
        for flag in ("forced_initial_action","sampled_initial_opponent","process_ownership_validated",
                "actor_history_provenance_validated","independent_terminal_labels","runtime_qualified",
                "scientific_collection_authorized"):
            self.assertFalse(result[flag])
        with self.assertRaises(TypeError): pickle.dumps(capture)

    def test_unavailable_opponent_refuses_before_bridge_or_policy(self):
        with patch.object(LocalShowdownEnv,"_start_bridge",side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError,"both separately frozen actor histories"):
                self.run_cell(dict(self.programs,p2=None))
        self.assertTrue(all(not p.calls for p in self.policies.values()))

    def test_step_rng_is_exact_distinct_and_rejects_numeric_aliases(self):
        seeds=[after_step_seed(17,"a"*64,rep,step) for rep in range(8) for step in range(250)]
        self.assertEqual(len(set(seeds)),2000)
        for args in ((True,"a"*64,0,0),(17,"bad",0,0),(17,"a"*64,True,0),
                (17,"a"*64,0,False),(17,"a"*64,8,0),(17,"a"*64,0,250)):
            with self.assertRaises(ValueError): after_step_seed(*args)

    def test_independent_reader_rejects_policy_input_action_seed_and_extra_calls(self):
        capture,records=self.run_cell()
        for target in ("observation","action","seed","extra"):
            calls=deepcopy(capture.policy_calls); modified=deepcopy(records)
            if target=="observation":calls[0]["observation_sha256"]="f"*64
            if target=="action":calls[0]["action"]=min(i for i,v in enumerate(calls[0]["mask"]) if v)
            if target=="seed":
                command=next(r for r in modified if r["kind"]=="command" and r["payload"]["type"]=="reseed")
                command["payload"]["seed"]="1,2,3,4"
                next(r for r in modified if r["kind"]=="event" and r["payload"]["type"]=="reseeded")["payload"]["seed"]="1,2,3,4"
                transport=deepcopy(capture.transport); transport["records"]=modified
            else:transport=capture.transport
            if target=="extra":calls=(*calls,deepcopy(calls[0]))
            with self.subTest(target=target),self.assertRaises(ValueError):
                self.read(AfterRawCellCapture(capture.binding,transport,tuple(calls)),modified)

    def two_steps(self):
        text=self.h.bridge.read_text().replace("if(fixture.nonterminalReplay)","if(fixture.nonterminalReplay && generation<2)")
        self.h.bridge.write_text(text)
        self.h.fixture_path.write_text(json.dumps(dict(mode="complete",snapshot=self.root,nonterminalReplay=True)))
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])

    def test_two_actual_boundaries_use_one_policy_call_per_requested_seat_and_step(self):
        self.two_steps(); capture,records=self.run_cell()
        self.assertEqual([(r["step"],r["seat"]) for r in capture.policy_calls],[(0,"p1"),(0,"p2"),(1,"p1"),(1,"p2")])
        result=self.read(capture,records)
        self.assertEqual((result["status"],result["boundaries"]),("COMPLETE",2))
        self.assertTrue(all(len(p.calls)==2 for p in self.policies.values()))

    def test_later_policy_callback_cannot_mutate_an_earlier_actual_input(self):
        self.two_steps()
        original=self.policies["p1"].select_action
        def mutate(observation,*,rng):
            result=original(observation,rng=rng)
            if len(self.policies["p1"].calls)==2:
                self.policies["p1"].calls[0].metadata["late_mutation"]="different earlier input"
            return result
        self.policies["p1"].select_action=mutate
        with self.assertRaisesRegex(ValueError,"previous observations changed"):
            self.run_cell()

    def test_terminal_event_without_protocol_stays_uncertain_not_draw(self):
        bridge=self.h.bridge.read_text().replace("current.battle.log.push('|win|PokeZero p1'); send({type:'stream',battleId:b,stream:'omniscient',lines:['|win|PokeZero p1']}); ","")
        self.h.bridge.write_text(bridge)
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        capture,records=self.run_cell(); result=self.read(capture,records)
        self.assertEqual(result["status"],"TERMINAL_UNVALIDATED")
        self.assertIsNone(result["signed_outcome"])

    def test_midboundary_stop_cannot_be_relabelled_as_a_terminal_prefix(self):
        self.two_steps(); capture,records=self.run_cell()
        transport=deepcopy(capture.transport)
        transport["snapshots"].pop(); transport["request_ids"].pop()
        with self.assertRaises(ValueError):
            self.read(AfterRawCellCapture(capture.binding,transport,capture.policy_calls),records)

    def test_later_requested_actor_without_own_side_refuses_before_second_inference(self):
        self.two_steps()
        self.h.bridge.write_text(self.h.bridge.read_text().replace("current.battle.log.push('|turn|2');",
            "delete current.boundaryRequests.p1.side; current.battle.log.push('|turn|2');"))
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        with self.assertRaisesRegex(ValueError,"requested actor lacks actual full own request"):
            self.run_cell()
        self.assertTrue(all(len(p.calls)==1 for p in self.policies.values()))

    def test_release_phase_win_cannot_supply_an_independent_terminal(self):
        capture,records=self.run_cell()
        modified=deepcopy(records)
        win=next(r for r in modified if r["kind"]=="event" and r["payload"].get("type")=="stream")
        modified.remove(win)
        release=next(i for i,r in enumerate(modified) if r["kind"]=="command" and r["payload"]["type"]=="release_clone")
        modified.insert(release+1,win)
        transport=deepcopy(capture.transport); transport["records"]=modified
        with self.assertRaisesRegex(ValueError,"terminal protocol outside actual acknowledged transition"):
            self.read(AfterRawCellCapture(capture.binding,transport,capture.policy_calls),modified)

    def test_observed_win_missing_from_final_snapshot_refuses(self):
        capture,records=self.run_cell()
        transport=deepcopy(capture.transport); modified=deepcopy(records)
        last=transport["snapshots"][-1]; raw=json.loads(last["currentSnapshotJson"])
        raw["battle"]["log"].remove("|win|PokeZero p1")
        last["currentSnapshotJson"]=json.dumps(raw)
        last["currentSnapshotJsonSha256"]=hashlib.sha256(last["currentSnapshotJson"].encode()).hexdigest()
        matching=next(r for r in modified if r["payload"].get("type")=="owned_boundary_snapshot"
            and r["payload"]["snapshotIndex"]==1)
        matching["payload"]["currentSnapshotJsonSha256"]=last["currentSnapshotJsonSha256"]
        transport["records"]=modified
        with self.assertRaisesRegex(ValueError,"final snapshot and transition terminal protocol disagree"):
            self.read(AfterRawCellCapture(capture.binding,transport,capture.policy_calls),modified)

    def test_actual_waiting_actor_is_not_encoded_again_or_sent_a_choice(self):
        self.root["boundaryRequests"]["p2"]["wait"]=True
        self.programs["p2"]["events"] += [dict(kind="own_request",request=deepcopy(self.root["boundaryRequests"]["p2"])),
            dict(kind="observe",coordinate=1)]
        raw=json.dumps(self.root,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        self.clone["snapshotJsonSha256"]=hashlib.sha256(raw.encode()).hexdigest()
        self.h.bridge.write_text(self.h.bridge.read_text().replace("send({type:'cloned',...owned,requested:['p1','p2']})",
            "send({type:'cloned',...owned,requested:Object.entries(current.boundaryRequests).filter(([s,r])=>!r.wait).map(([s,r])=>s)})"))
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        capture,records=self.run_cell()
        self.assertEqual([c["seat"] for c in capture.policy_calls],["p1"])
        self.assertFalse(self.policies["p2"].calls)
        self.assertEqual(self.read(capture,records)["status"],"COMPLETE")

    def test_fixed_250_boundary_cap_remains_null_after_full_observed_prefix(self):
        self.h.fixture_path.write_text(json.dumps(dict(mode="complete",snapshot=self.root,nonterminalReplay=True)))
        self.h.bridge.write_text(self.h.bridge.read_text().replace("current.boundaryRequests[seat].active[0].moves[0].pp--;",
            "current.boundaryRequests[seat].active[0].moves[0].pp=current.boundaryRequests[seat].active[0].moves[0].pp;"))
        self.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        capture,records=self.run_cell(); result=self.read(capture,records)
        self.assertEqual((result["status"],result["boundaries"]),("CAPPED_UNCERTAIN",250))
        self.assertIsNone(result["signed_outcome"])
        self.assertEqual(len(capture.transport["snapshots"]),251)

    def test_final_reader_callback_cannot_flip_external_subject_or_filter_bytes(self):
        import pokezero.mcts_eval.search_over_raw_after_raw as module
        capture,records=self.run_cell()
        original=module._actors
        for target in ("subject","filter"):
            expected=deepcopy(capture.binding)
            path=self.h.root/"showdown/dist/sim/battle.js"; original_bytes=path.read_bytes()
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs)
                if target=="subject":expected["subject"]="p2"
                else:path.write_bytes(original_bytes+b"changed filter bytes")
                return result
            try:
                with self.subTest(target=target),patch.object(module,"_actors",side_effect=mutate),self.assertRaises(ValueError):
                    read_after_raw_cell(capture,process_records=records,raw_after=self.root,programs=self.programs,
                        config=self.h.config,expected_binding=expected,runtime=self.runtime)
            finally:path.write_bytes(original_bytes)

    def test_cleanup_callback_cannot_mutate_original_clone_request_ids_or_env_config(self):
        original=Trace.close
        for target in ("clone","requests","config"):
            clone=deepcopy(self.clone); ids=list(self.ids)
            def mutate(trace,env):
                original(trace,env)
                if target=="clone":self.clone["cloneId"]="late-other-clone"
                if target=="requests":self.ids.append("late-extra-id")
                if target=="config":env.config=replace(env.config,read_timeout_seconds=3.)
            try:
                with self.subTest(target=target),patch.object(Trace,"close",new=mutate),self.assertRaisesRegex(ValueError,"original AFTER cell inputs"):
                    self.run_cell()
            finally:self.clone=clone; self.ids=ids
