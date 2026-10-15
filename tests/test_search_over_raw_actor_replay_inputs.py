"""Authored replay input transport only; no actor inference or labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_outcome_run import _redact
from pokezero.mcts_eval.search_over_raw_actor_replay_inputs import read_actor_replay_inputs
from pokezero.mcts_eval.search_over_raw_source_public_history import channel_lines
from test_search_over_raw_boundary_transport import fixture, encoded, sha


def actor_fixture():
    cap,records,initial,binding,ids=fixture()
    raw=json.loads(initial)
    shell=["|split|p2","|debug|NEVER_IMPORT_SHELL_PRIVATE","|message|shell"]
    raw["battle"]["log"]=shell
    raw["boundaryRequests"]["p1"]["side"]={"id":"p1","pokemon":[{"ident":"p1: actor-own"}]}
    raw["boundaryRequests"]["p2"]["side"]={"id":"p2","pokemon":[{"ident":"p2: OPPONENT_PRIVATE_TEAM"}]}
    initial=encoded(raw);binding["snapshotJsonSha256"]=sha(initial)
    cap["clone_binding"]=deepcopy(binding)
    logs=[shell,shell+["|split|p1","|-damage|p1a: actor-own|37/91","|-damage|p1a: actor-own|41/100",
        "|split|p2","|-damage|p2a: rival|22/177","|-damage|p2a: rival|13/100"],None]
    logs[2]=logs[1]+["|win|A"]
    for index,event in enumerate(cap["snapshots"]):
        current=deepcopy(raw)
        current["battle"]["log"]=deepcopy(logs[index])
        if index==2:
            current["battle"]["ended"]=True;current["boundaryRequests"]={};current["terminalScheduled"]=True
        event.update(binding)
        event["currentSnapshotJson"]=encoded(current);event["currentSnapshotJsonSha256"]=sha(event["currentSnapshotJson"])
    for row in cap["records"]:
        payload=row["payload"]
        if payload["type"]=="clone_snapshot":
            payload["snapshotJsonSha256"]=sha(initial)
        elif payload["type"]=="owned_boundary_snapshot":
            row.update(_redact("event",cap["snapshots"][payload["snapshotIndex"]]))
        elif "snapshotJsonSha256" in payload:payload["snapshotJsonSha256"]=sha(initial)
    records=deepcopy(cap["records"])
    return cap,records,initial,binding,ids


class ActorReplayInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="pokezero-actor-input-authored-")
        self.addCleanup(self.temp.cleanup)
        sim=Path(self.temp.name)/"dist/sim";sim.mkdir(parents=True)
        official=Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/battle.js").read_text()
        start=official.index("const splitRegex = ");stop=official.index("\nclass Battle {",start)
        (sim/"battle.js").write_text(official[start:stop]+"\nclass Battle {\n}")
        (sim/"state.js").write_text("state.log = battle.log;")
        import pokezero.mcts_eval.search_over_raw_source_public_history as source
        import pokezero.mcts_eval.search_over_raw_protocol_channels as channels
        paths=[sim/"battle.js",sim/"state.js",Path(source.__file__).resolve(),Path(channels.__file__).resolve()]
        self.runtime=dict(argv=["not-launched","not-launched","--showdown-root",self.temp.name],
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def read(self,values,actor="p1"):
        cap,records,initial,binding,ids=values
        return read_actor_replay_inputs(cap,process_records=records,initial_snapshot_json=initial,
            clone_binding=binding,request_ids=ids,actor=actor,runtime=self.runtime)

    def rewrite_snapshot(self,values,index,change):
        cap,records,initial,binding,ids=values
        event=cap["snapshots"][index];raw=json.loads(event["currentSnapshotJson"]);change(raw)
        event["currentSnapshotJson"]=encoded(raw);event["currentSnapshotJsonSha256"]=sha(event["currentSnapshotJson"])
        for trace in (cap["records"],records):
            row=next(r for r in trace if r["payload"]["type"]=="owned_boundary_snapshot"
                and r["payload"]["snapshotIndex"]==index)
            row.update(_redact("event",event))

    def test_both_seat_suffixes_preserve_own_exact_hp_and_opponent_public_hp_without_shell_prefix(self):
        values=actor_fixture();before=deepcopy(values)
        p1=self.read(values,"p1");p2=self.read(values,"p2")
        self.assertEqual(values,before)
        self.assertEqual(p1["rows"][0]["visible_suffix"],[])
        self.assertIn("37/91",str(p1["rows"][1]["visible_suffix"]))
        self.assertNotIn("22/177",str(p1))
        self.assertIn("13/100",str(p1["rows"][1]["visible_suffix"]))
        self.assertIn("22/177",str(p2["rows"][1]["visible_suffix"]))
        self.assertNotIn("37/91",str(p2))
        self.assertNotIn("NEVER_IMPORT_SHELL_PRIVATE",json.dumps(p1)+json.dumps(p2))
        self.assertNotIn("OPPONENT_PRIVATE_TEAM",json.dumps(p1))
        self.assertEqual([r["observed_generation"] for r in p1["rows"]],[0,1,2])
        self.assertTrue(p1["rows"][0]["actor_requested"])
        self.assertFalse(p2["rows"][0]["actor_requested"])
        self.assertIsNone(p1["rows"][-1]["own_request"])
        for key in ("process_ownership_validated","construction_provenance_validated","source_prefix_join_validated",
                "full_actor_observation_validated","independent_terminal_labels","runtime_qualified"):
            self.assertFalse(p1[key])

    def test_rewritten_truncated_and_new_root_logs_refuse_even_with_consistent_transport_hashes(self):
        changes=[(1,lambda raw:raw["battle"]["log"].__setitem__(0,"|message|rewrite")),
            (1,lambda raw:raw["battle"].__setitem__("log",[])),
            (0,lambda raw:raw["battle"]["log"].append("|message|new-root")),
            (2,lambda raw:raw["battle"]["log"].__setitem__(4,"|-damage|p1a: actor-own|38/91"))]
        for index,change in changes:
            values=actor_fixture();self.rewrite_snapshot(values,index,change)
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,"rewrote|clone root log"):
                self.read(values)

    def test_private_request_split_and_missing_log_cannot_enter_actor_suffix(self):
        for change in (lambda raw:raw["battle"]["log"].append("|request|original-private"),
                lambda raw:raw["battle"]["log"].append("|split|p1"),
                lambda raw:raw["battle"].pop("log"),
                lambda raw:raw["boundaryRequests"]["p1"]["side"].__setitem__("id","p2")):
            values=actor_fixture();self.rewrite_snapshot(values,1,change)
            with self.assertRaises(ValueError):self.read(values)

    def test_rehashed_capture_without_independent_observation_and_changed_filter_bytes_refuse(self):
        values=actor_fixture()
        values[0]["snapshots"][1]["currentSnapshotJsonSha256"]="b"*64
        with self.assertRaises(ValueError):self.read(values)
        path=Path(self.temp.name)/"dist/sim/battle.js";path.write_text("unsupported filter")
        with self.assertRaisesRegex(ValueError,"runtime bytes drifted"):self.read(actor_fixture())

    def test_older_runtime_cannot_omit_the_new_shared_filter_bytes(self):
        import pokezero.mcts_eval.search_over_raw_protocol_channels as channels
        import pokezero.mcts_eval.search_over_raw_source_public_history as source
        self.assertIs(source.channel_lines,channels.channel_lines)
        self.runtime["files"].pop(str(Path(channels.__file__).resolve()))
        with self.assertRaisesRegex(ValueError,"runtime bytes drifted"):self.read(actor_fixture())

    def test_boolean_channel_alias_and_foreign_actor_refuse(self):
        for channel in (True,False,-1,3,"1"):
            with self.assertRaises(ValueError):channel_lines([],channel=channel)
        for actor in ("p3",1,True):
            with self.assertRaises(ValueError):self.read(actor_fixture(),actor)


if __name__ == "__main__":unittest.main()
