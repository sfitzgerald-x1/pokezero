"""Synthetic transport only; no scientific root, model or Showdown runtime."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_boundary_transport import (
    BoundaryTransportCapture, read_owned_boundary_transport, SCHEMA,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import _redact
from pokezero.mcts_eval.search_over_raw_outcomes import ContinuationCapture
from test_search_over_raw_outcomes import EntrypointSyntheticEnv, parser_snapshot


def encoded(value): return json.dumps(value,sort_keys=True,separators=(",",":"))
def sha(value): return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    root = dict(battle=dict(ended=False,private="fixture-hidden"),
        boundaryRequests={"p1":dict(active=[{}]),"p2":dict(wait=True)},terminalScheduled=False)
    initial = encoded(root)
    binding = dict(battleId="boundary-fixture",cloneId="clone-fixture",snapshotJsonSha256=sha(initial))
    ids = ["root","next","terminal"]
    records, snapshots = [], []
    def row(kind,**payload): records.append(_redact(kind,payload))
    row("command",type="clone_snapshot",battleId=binding["battleId"],cloneId=binding["cloneId"],snapshotJson=initial)
    row("event",type="cloned",**binding,requested=["p1"])
    for index,request in enumerate(ids):
        if index:
            row("command",type="choices",battleId=binding["battleId"],choices=dict(p1="move 1"))
            row("event",type="choice_ack",battleId=binding["battleId"],player="p1",choice="move 1")
            row("event",type="stream",battleId=binding["battleId"],stream="p1",lines=["|request|private"])
            row("event",type="terminal" if index==2 else "ready",battleId=binding["battleId"],
                **({} if index==2 else dict(requested=["p1"])))
        if index == 2:
            root["battle"]["ended"],root["boundaryRequests"],root["terminalScheduled"]=True,{},True
        row("command",type="snapshot_owned_boundary",**binding,requestId=request)
        current = encoded(root)
        event = dict(type="owned_boundary_snapshot",**binding,requestId=request,snapshotIndex=index,
            boundaryGeneration=index,currentSnapshotJson=current,currentSnapshotJsonSha256=sha(current))
        snapshots.append(event); row("event",**event)
    row("command",type="release_clone",**binding)
    for seat in ("omniscient","p1","p2"):row("event",type="stream_end",battleId=binding["battleId"],stream=seat)
    row("event",type="clone_released",**binding)
    capture = dict(schema=SCHEMA,clone_binding=binding,request_ids=ids,records=records,
        snapshots=snapshots,private_controller_only=True)
    return capture,deepcopy(records),initial,deepcopy(binding),list(ids)


def read(values):
    cap,records,initial,binding,ids=values
    return read_owned_boundary_transport(cap,process_records=records,initial_snapshot_json=initial,
        clone_binding=binding,request_ids=ids)


class BoundaryTransportTests(unittest.TestCase):
    def test_exact_trace_retains_every_root_and_transition_without_promoting_provenance(self):
        values=fixture(); before=deepcopy(values); result=read(values)
        self.assertEqual(result["capture_sha256"],digest(values[0])); self.assertEqual(values,before)
        self.assertEqual([v["boundary_kind"] for v in result["observed_snapshots"]],["root","ready","terminal"])
        self.assertEqual([v["observed_generation"] for v in result["observed_snapshots"]],[0,1,2])
        self.assertTrue(result["snapshot_transport_consistent"])
        for key in ("process_ownership_validated","construction_provenance_validated","branch_transition_validated",
            "native_ancestry_coverage_validated","independent_terminal_truth","runtime_qualified",
            "exact_rng_coupling_claim","scientific_collection_authorized"):
            self.assertFalse(result[key])
        self.assertNotIn("fixture-hidden",encoded(result))

    def test_raw_snapshot_mutation_and_rehashed_substitution_cannot_join_external_trace(self):
        for mode in ("raw","rehash","index","generation","request","token","clone","root-hash","boolean"):
            values=fixture(); event=values[0]["snapshots"][1]
            if mode in {"raw","rehash"}:
                raw=json.loads(event["currentSnapshotJson"]);raw["battle"]["private"]="substitution"
                event["currentSnapshotJson"]=encoded(raw)
                if mode=="rehash":event["currentSnapshotJsonSha256"]=sha(event["currentSnapshotJson"])
            if mode=="index":event["snapshotIndex"]=0
            if mode=="generation":event["boundaryGeneration"]=0
            if mode=="request":event["requestId"]="wrong"
            if mode=="token":event["battleId"]="wrong"
            if mode=="clone":event["cloneId"]="wrong"
            if mode=="root-hash":event["snapshotJsonSha256"]="b"*64
            if mode=="boolean":event["snapshotIndex"]=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):read(values)

    def test_unsettled_duplicate_missing_foreign_and_post_release_transport_rejected(self):
        # Mutate BOTH stored traces, so equality alone cannot catch the defect.
        for mode in ("no-root","no-ready","no-ack","duplicate-ack","missing-end","duplicate-end",
            "after-release","foreign","extra-choice","wrong-command","swap-snapshots","bool-generation"):
            values=fixture(); records=values[0]["records"]
            index=lambda tag:next(i for i,r in enumerate(records) if r["payload"].get("type")==tag)
            if mode=="no-root":del records[index("snapshot_owned_boundary"):index("owned_boundary_snapshot")+1]
            if mode=="no-ready":records.pop(index("ready"))
            if mode=="no-ack":records.pop(index("choice_ack"))
            if mode=="duplicate-ack":records.insert(index("choice_ack"),deepcopy(records[index("choice_ack")]))
            if mode=="missing-end":records.pop(index("stream_end"))
            if mode=="duplicate-end":records.insert(index("stream_end"),deepcopy(records[index("stream_end")]))
            if mode=="after-release":records.append(deepcopy(records[index("stream")]))
            if mode=="foreign":records[index("stream")]["payload"]["battleId"]="old-token"
            if mode=="extra-choice":records[index("choices")]["payload"]["choices"]["p2"]="move 1"
            if mode=="wrong-command":records[index("choices")]["payload"]["type"]="restore"
            if mode=="swap-snapshots":values[0]["snapshots"].reverse()
            if mode=="bool-generation":
                event=values[0]["snapshots"][0];event["boundaryGeneration"]=False
                records[index("owned_boundary_snapshot")]=_redact("event",event)
            values[1][:]=deepcopy(records)
            with self.subTest(mode=mode),self.assertRaises(ValueError):read(values)

    def test_missing_nonterminal_requests_not_repaired_even_if_both_traces_rehashed(self):
        for mode in ("missing","empty","terminal-conflict"):
            values=fixture();event=values[0]["snapshots"][1];raw=json.loads(event["currentSnapshotJson"])
            if mode=="missing":raw["boundaryRequests"].pop("p2")
            if mode=="empty":raw["boundaryRequests"]={}
            if mode=="terminal-conflict":raw["battle"]["ended"]=True
            event["currentSnapshotJson"]=encoded(raw);event["currentSnapshotJsonSha256"]=sha(event["currentSnapshotJson"])
            for records in (values[0]["records"],values[1]):
                row=next(v for v in records if v["payload"].get("type")=="owned_boundary_snapshot"
                    and v["payload"].get("snapshotIndex")==1)
                row.update(_redact("event",event))
            with self.subTest(mode=mode),self.assertRaises(ValueError):read(values)

    def test_explicit_reseed_does_not_increment_boundary_generation_or_claim_rng_coupling(self):
        values=fixture();records=values[0]["records"]
        at=next(i for i,r in enumerate(records) if r["payload"].get("type")=="choices")
        token=values[3]["battleId"]
        records[at:at]=[_redact("command",dict(type="reseed",battleId=token,seed="1,2,3,4")),
            _redact("event",dict(type="reseeded",battleId=token,seed="1,2,3,4"))]
        values[1][:]=deepcopy(records)
        self.assertEqual(read(values)["observed_snapshots"][1]["observed_generation"],1)
        records.pop(at+1);values[1][:]=deepcopy(records)
        with self.assertRaises(ValueError):read(values)

    def test_late_private_request_streams_cannot_clear_ready_snapshot_or_next_action(self):
        for stage in ("after-ready","during-snapshot","after-snapshot"):
            values=fixture();records=values[0]["records"]
            if stage=="after-ready":at=next(i for i,r in enumerate(records) if r["payload"].get("type")=="ready")+1
            elif stage=="during-snapshot":at=next(i for i,r in enumerate(records) if r["payload"].get("type")
                =="snapshot_owned_boundary" and r["payload"].get("requestId")=="next")+1
            else:at=next(i for i,r in enumerate(records) if r["payload"].get("type")
                =="owned_boundary_snapshot" and r["payload"].get("requestId")=="next")+1
            records.insert(at,_redact("event",dict(type="stream",battleId=values[3]["battleId"],
                stream="p1",lines=["|request|late-private-request"])))
            values[1][:]=deepcopy(records)
            with self.subTest(stage=stage),self.assertRaisesRegex(ValueError,"late private"):read(values)

    def test_private_snapshot_redacted_in_both_process_and_old_cell_observers(self):
        event=fixture()[0]["snapshots"][0]
        self.assertNotIn("currentSnapshotJson",_redact("event",event)["payload"])
        observer=ContinuationCapture({},"a"*64);observer.active=dict(records=[])
        observer.event(event)
        self.assertNotIn("fixture-hidden",encoded(observer.active))
        self.assertEqual(observer.active["records"][0],_redact("event",event))

    def test_actual_python_and_bridge_transport_snapshots_root_and_terminal_with_separate_observers(self):
        with tempfile.TemporaryDirectory(prefix="pokezero-boundary-transport-synthetic-") as temp:
            sim=Path(temp)/"dist/sim";sim.mkdir(parents=True)
            text=Path(__file__).with_name("test_owned_continuation.mjs").read_text()
            for variable,name in (("fakeRuntime","index.js"),("fakeState","state.js")):
                (sim/name).write_text(re.search(r"const "+variable+r" = `([\s\S]*?)`;",text).group(1))
            env=EntrypointSyntheticEnv(LocalShowdownConfig(showdown_root=temp,set_belief_source=False,
                read_timeout_seconds=2.))
            state=parser_snapshot(env);initial=encoded(state.bridge_snapshot)
            binding=dict(battleId="private-boundary",cloneId="path-1",snapshotJsonSha256=sha(initial))
            cap=BoundaryTransportCapture(clone_binding=binding,request_ids=["root","terminal"])
            separate=[]
            try:
                with cap.installed(env):
                    env._start_bridge()
                    # Independent transport observer only, NOT a process owner.
                    env._bridge_process_observer=SimpleNamespace(
                        command=lambda value:separate.append(_redact("command",value)),
                        event=lambda value:separate.append(_redact("event",value)))
                    env._battle_token=binding["battleId"]
                    ack=env._bridge_request_event(dict(type="clone_snapshot",battleId=binding["battleId"],
                        cloneId=binding["cloneId"],snapshotJson=initial),"cloned")
                    self.assertEqual(ack["snapshotJsonSha256"],binding["snapshotJsonSha256"])
                    env._restore_local_snapshot_state(state)
                    self.assertFalse(cap.snapshot()["battle"]["ended"])
                    env.step(dict(p1=0,p2=0))
                    self.assertTrue(cap.snapshot()["battle"]["ended"])
                    env._bridge_request_event(dict(type="release_clone",**binding),"clone_released")
                    env._battle_token=None
                value=cap.export()
                result=read_owned_boundary_transport(value,process_records=separate,initial_snapshot_json=initial,
                    clone_binding=binding,request_ids=["root","terminal"])
                self.assertEqual([v["observed_generation"] for v in result["observed_snapshots"]],[0,1])
                self.assertFalse(result["process_ownership_validated"])
                self.assertNotIn("currentSnapshotJson",encoded(separate).replace("currentSnapshotJsonSha256",""))
                with self.assertRaises(ValueError):
                    with cap.installed(env):pass
            finally:
                env._bridge_process_observer=None
                env.close()

    def test_ambiguous_read_poisoned_no_next_request_retry(self):
        values=fixture();cap=BoundaryTransportCapture(clone_binding=values[3],request_ids=values[4])
        def failed(command,event):
            cap.command(command)
            raise TimeoutError("synthetic observation failure")
        cap.env=SimpleNamespace(_battle_token=values[3]["battleId"],_bridge_request_event=failed)
        with self.assertRaises(TimeoutError):cap.snapshot()
        with self.assertRaises(ValueError):cap.snapshot()
        self.assertEqual(len(cap.records),1)
        # A fully valid-looking delayed ack consumed during cleanup is not a recovery.
        cap.records,cap.snapshots=deepcopy(values[0]["records"]),deepcopy(values[0]["snapshots"])
        cap.env=None
        with self.assertRaisesRegex(ValueError,"failed observation"):cap.export()

    def test_actual_bridge_late_private_request_after_root_snapshot_invalidates_transport(self):
        with tempfile.TemporaryDirectory(prefix="pokezero-late-request-synthetic-") as temp:
            sim=Path(temp)/"dist/sim";sim.mkdir(parents=True)
            text=Path(__file__).with_name("test_owned_continuation.mjs").read_text()
            for variable,name in (("fakeRuntime","index.js"),("fakeState","state.js")):
                source=re.search(r"const "+variable+r" = `([\s\S]*?)`;",text).group(1)
                if variable=="fakeRuntime":
                    source=source.replace("if (data.startsWith('>reseed ')) return;",
                        "if (data.startsWith('>reseed ')) { this.channels.p1.push("+
                        "'|request|'+JSON.stringify({active:[{}],side:{id:'p1'}})); return; }")
                (sim/name).write_text(source)
            env=EntrypointSyntheticEnv(LocalShowdownConfig(showdown_root=temp,set_belief_source=False,
                read_timeout_seconds=2.));state=parser_snapshot(env);initial=encoded(state.bridge_snapshot)
            binding=dict(battleId="late-boundary",cloneId="late-path",snapshotJsonSha256=sha(initial))
            cap=BoundaryTransportCapture(clone_binding=binding,request_ids=["root"]);separate=[]
            try:
                with cap.installed(env):
                    env._start_bridge();env._bridge_process_observer=SimpleNamespace(
                        command=lambda v:separate.append(_redact("command",v)),
                        event=lambda v:separate.append(_redact("event",v)))
                    env._battle_token=binding["battleId"]
                    env._bridge_request_event(dict(type="clone_snapshot",battleId=binding["battleId"],
                        cloneId=binding["cloneId"],snapshotJson=initial),"cloned")
                    env._restore_local_snapshot_state(state);cap.snapshot()
                    env.reseed_simulator_rng(5)
                    # Consume the late synthetic request/ready without any
                    # action, observation construction or simulator source.
                    env._read_until_boundary()
                    env._bridge_request_event(dict(type="release_clone",**binding),"clone_released")
                    env._battle_token=None
                self.assertTrue(any(r["payload"].get("stream")=="p1" for r in separate))
                with self.assertRaisesRegex(ValueError,"late private"):
                    read_owned_boundary_transport(cap.export(),process_records=separate,
                        initial_snapshot_json=initial,clone_binding=binding,request_ids=["root"])
            finally:
                env._bridge_process_observer=None;env.close()


if __name__ == "__main__":unittest.main()
