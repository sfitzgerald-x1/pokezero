"""Authored transport/public histories only; no sources, labels, or physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
from pokezero.mcts_eval.search_over_raw_source_run import (
    _source_trace, read_source_snapshot_trace, read_archived_source_snapshot_join)
from pokezero.mcts_eval.search_over_raw_source_public_history import (
    SCHEMA, FILTER_SHA256, channel_zero_lines, channel_lines, rebuild_public_fold)


class SourcePublicHistoryTests(unittest.TestCase):
    def records(self):
        records=[]
        for index in range(2):
            records.append(_source_trace("command",dict(type="snapshot_actionable_boundary",battleId="owned")))
            event=_source_trace("event",dict(type="snapshot_actionable_boundary",battleId="owned",
                available=True,requested=["p1","p2"],snapshot={"n":index},nodeProcMs=.5))
            event.update(record_index=len(records),decision_round_index=index)
            records.append(event)
            records.append(_source_trace("command",dict(type="choices",battleId="owned",choices={"p1":"move 1","p2":"move 1"})))
        return records

    def test_actual_received_snapshot_digest_is_independent_of_timing_and_private_bytes_not_copied(self):
        payload=dict(type="snapshot_actionable_boundary",battleId="owned",available=True,requested=["p1"],
            snapshot={"private":"never-copy-source-canary"},nodeProcMs=1.)
        first=_source_trace("event",payload)
        payload["nodeProcMs"]=2.
        second=_source_trace("event",payload)
        self.assertNotEqual(first["payload_sha256"],second["payload_sha256"])
        self.assertEqual(first["snapshot_sha256"],second["snapshot_sha256"])
        self.assertNotIn("never-copy-source-canary",json.dumps(first))
        payload["snapshot"]["private"]="changed"
        self.assertNotEqual(first["snapshot_sha256"],_source_trace("event",payload)["snapshot_sha256"])

    def test_exact_observed_boundary_roster_and_legacy_non_upgrade(self):
        records=self.records()
        observed=read_source_snapshot_trace(records,decision_boundaries=2)
        self.assertEqual(set(observed),{0,1})
        legacy=[row for row in records if row["type"]=="choices"]
        self.assertEqual(read_source_snapshot_trace(legacy,decision_boundaries=2),{})
        self.assertTrue(all("snapshot" not in row for row in observed.values()))

    def test_wrong_or_partial_coordinates_unsolicited_duplicate_unavailable_and_seat_drift_refuse(self):
        candidates=[]
        for key,value in (("record_index",2),("record_index",True),("decision_round_index",1),
                ("decision_round_index",False),("snapshot_sha256","x"*64),("available",False),
                ("requested",["p2"]),("requested",["p2","p1"])):
            rows=self.records();rows[1][key]=value;candidates.append(rows)
        rows=self.records();del rows[1]["snapshot_sha256"];candidates.append(rows)
        candidates += [self.records()[1:],self.records()[:3]+self.records()[5:],
            self.records()[:2]+[deepcopy(self.records()[1])]+self.records()[2:],
            self.records()[:1]+self.records()[2:]]
        for rows in candidates:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                read_source_snapshot_trace(rows,decision_boundaries=2)

    def test_archive_join_rejects_another_snapshot_and_legacy_missing_transport(self):
        archive=SealedSourceArchive()
        archive._private={0:SimpleNamespace(bridge_snapshot={"n":0})}
        source=dict(decision_boundaries=2,roots=[dict(root_id="selected",source_request_index=0)])
        with patch("pokezero.mcts_eval.search_over_raw_source_run.retained_archive_root_identity",return_value="a"*64):
            result=read_archived_source_snapshot_join(archive,source,self.records())
            self.assertEqual(result["selected"]["source_snapshot_sha256"],digest({"n":0}))
            archive._private[0].bridge_snapshot["n"]=1
            with self.assertRaisesRegex(ValueError,"differs from received"):
                read_archived_source_snapshot_join(archive,source,self.records())
            with self.assertRaisesRegex(ValueError,"LABEL_INPUT_UNAVAILABLE"):
                read_archived_source_snapshot_join(archive,source,[r for r in self.records() if r["type"]=="choices"])

    def test_channel_zero_matches_exact_pinned_upstream_function_on_authored_families(self):
        # Execute ONLY the inspected upstream pure filter function, not a
        # simulator module or scientific source. Node receives authored lines.
        node=shutil.which("node")
        self.assertIsNotNone(node)
        text=Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/battle.js").read_text()
        start=text.index("const splitRegex = ");stop=text.index("\nclass Battle {",start)
        function=text[start:stop]
        self.assertEqual(hashlib.sha256(function.encode()).hexdigest(),FILTER_SHA256)
        families=[[],["", "|start", ""],["|split|p1","SECRET","|message|visible"],
            ["|split|p2","SECRET",""],["|player|p1|A", "|split|p1","|-damage|p1a: A|117/253",
                "|-damage|p1a: A|47/100", "|split|p2","|-damage|p2a: B|19/212","|-damage|p2a: B|9/100",
                "|turn|2"]]
        program=function+"\nconst fs=require('fs');const families=JSON.parse(fs.readFileSync(0,'utf8'));" \
            +"process.stdout.write(JSON.stringify(families.map(lines=>extractChannelMessages(lines.join('\\n'),[0,1,2]))));"
        result=subprocess.run([node,"-e",program],input=json.dumps(families),text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        expected=json.loads(result.stdout)
        for channel in (0,1,2):
            self.assertEqual([channel_lines(lines,channel=channel) for lines in families],
                [row[str(channel)] for row in expected])

    def test_malformed_truncated_nested_request_and_multiline_splits_refuse(self):
        for lines in (["|split|p1"],["|split|p2","secret"],["|split|p3","secret","shared"],
                ["|split|p1","|split|p2","shared"],["|split|p1","secret","|split|p2"],
                ["|split|garbage","secret","shared"],["|request|{}"],
                ["|split|p1","|request|private","shared"],["|turn|1\n|request|{}"],
                ["|message|unicode\u2028separator"],["|message|\x00hidden"]):
            with self.subTest(lines=lines),self.assertRaises(ValueError):channel_zero_lines(lines)

    def test_rebuild_discards_exact_private_hp_and_has_no_original_requests_or_catalog(self):
        raw=["|player|p1|A|","|player|p2|B|","|start", "|split|p1",
            "|switch|p1a: Pikachu|Pikachu, L80|117/253","|switch|p1a: Pikachu|Pikachu, L80|47/100",
            "|split|p2","|switch|p2a: Snorlax|Snorlax, L80|211/287",
            "|switch|p2a: Snorlax|Snorlax, L80|74/100","|turn|1"]
        public=channel_zero_lines(raw)
        history=dict(schema=SCHEMA,root_id="authored",channel=0,public_lines=public,
            public_lines_sha256=digest(public),complete_reset_prefix=True,old_omniscient_fold_used=False,
            original_source_requests_used=False,private_source_snapshot_included=False,
            labels_produced=False,runtime_qualified=False)
        replay,belief=rebuild_public_fold(history,expected_history_sha256=digest(history))
        self.assertEqual(replay.requests,{})
        self.assertIsNone(belief.set_source)
        rendered=str(replay)
        self.assertNotIn("117/253",rendered);self.assertNotIn("211/287",rendered)
        self.assertIn("47/100",rendered);self.assertIn("74/100",rendered)
        expected=digest(history)
        history["public_lines"]+=["|request|{}"]
        with self.assertRaises(ValueError):rebuild_public_fold(history,expected_history_sha256=expected)


if __name__ == "__main__":unittest.main()
