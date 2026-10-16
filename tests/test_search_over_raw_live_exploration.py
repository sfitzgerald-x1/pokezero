"""Focused live-controller joins and uncertainty rules; no real science."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_live_exploration import (
    PRIOR_SCHEMA,_write_new,gate_reports,read_live_source,read_gates,
)


class LiveExplorationTests(unittest.TestCase):
    def test_external_calendar_cap_stops_an_actual_hanging_worker(self):
        from test_search_over_raw_supervision import SyntheticJob
        from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
        from pokezero.mcts_eval.search_over_raw_exploration_entry import ExplorationAuditorContract
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);deadline=time.time()+1.
            receipt=_supervise_owned_job(SyntheticJob("hang"),directory=folder/"attempt",binding_sha256="a"*64,
                verify_admission=lambda:True,contract=ExplorationAuditorContract(60),kind="staged-exploration-run",
                work_phase="selection",validate_result=lambda v:True,calendar_deadline_utc_seconds=deadline)
            self.assertEqual(receipt["supervisor_exit_code"],124)
            self.assertTrue(receipt["watchdog_reached"])
            self.assertEqual(receipt["calendar_deadline_utc_seconds"],deadline)
            self.assertIsNotNone(receipt["worker_exit_code"])

    def report(self):
        binding=dict(source_configuration={"format":"gen3randombattle"},
            selector_configurations={"sampled4":{"worlds":4,"search_sims":80}},
            world_rng="per-decision",source_files={"source.py":"a"*64},
            **{k+"_sha256":"b"*64 for k in ("checkpoint","model","tables","engine","search")})
        return dict(schema=PRIOR_SCHEMA,complete=True,bindings=binding,missing_root_ids=[],
            expected=[dict(seed=7,round=3,variant="sampled4")],rows=[dict(seed=7,round=3,
                raw_action=1,boundary_sha256="c"*64,legal=[0,1],variants={"sampled4":dict(
                    status="MEASURED",action=0,iterations=320,worlds_searched=4,
                    native_prior_pre_display={"0":.25,"1":.75})})])

    def test_exact_selected_roster_passes_but_unknown_or_missing_does_not(self):
        collector=self.report();live=deepcopy(collector)
        self.assertEqual(gate_reports(collector,live)["status"],"PASSED")
        unavailable=deepcopy(collector["rows"][0]);unavailable["round"]=4
        unavailable["variants"]["sampled4"]={"status":"UNAVAILABLE","error":"constructor refused"}
        collector["rows"].append(unavailable);collector["expected"].append(dict(seed=7,round=4,variant="sampled4"))
        gate=gate_reports(collector,live)
        self.assertEqual(gate["status"],"INCOMPLETE")
        self.assertTrue(gate["measured_rows"]["passed"])
        self.assertEqual(gate["unavailable_selected_roots"],1)
        collector=self.report();collector["missing_root_ids"]=["exploration:7:6"]
        self.assertEqual(gate_reports(collector,live)["status"],"INCOMPLETE")

    def test_tiny_changed_live_value_blocks_contrast_admission(self):
        collector=self.report();live=deepcopy(collector)
        live["rows"][0]["variants"]["sampled4"]["native_prior_pre_display"]={"0":.250000001,"1":.749999999}
        with self.assertRaisesRegex(ValueError,"correspondence failed"):gate_reports(collector,live)

    def fixture(self,folder):
        report=self.report();row=report["rows"][0]
        source=dict(source_seed=7,missing_root_ids=[],roots=[dict(root_id="exploration:7:0",source_request_index=3,
            public_record={"recorded_action_index":1,"current_legal_action_mask":[True,True]})],
            live_shadow={"invocations":[dict(round=3,rows={"model":digest(row)})]})
        s=dict(output_directory=str(folder),contract={"leaves":["model"],"live_shadow":{"authored":True}},
            source_runtime={},prior_bindings={"model":report["bindings"]})
        _write_new(folder/"live-selections/7/model/3.json",row)
        _write_new(folder/"sources/7/bridge-transport.json",{"records":[
            dict(kind="command",type="choices",players=["p2"]),
            dict(kind="command",type="choices",players=["p2"]),
            dict(kind="command",type="choices",players=["p2"]),
            dict(kind="command",type="choices",players=["p1","p2"])]})
        for name,value in (("collector",report),("live",deepcopy(report)),("gate",gate_reports(report,report))):
            _write_new(folder/("live-checks/7/model/"+name+".json"),value)
        return s,source,row

    def test_source_manifest_joins_persisted_rows_original_catalog_and_actual_p1_schedule(self):
        with tempfile.TemporaryDirectory() as tmp,patch(
                "pokezero.mcts_eval.search_over_raw_source_run.read_source_owner",return_value={}):
            folder=Path(tmp);s,source,row=self.fixture(folder)
            rows=read_live_source(s,source)
            self.assertEqual(rows["model"][3],row)
            self.assertEqual(read_gates(s,source,rows)["model"]["status"],"PASSED")
            path=folder/"live-selections/7/model/3.json"
            changed=deepcopy(row);changed["raw_action"]=0;path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"manifest"):read_live_source(s,source)

    def test_extra_rows_wrong_schedule_and_incomplete_gate_roster_fail(self):
        for mode in ("extra","schedule","gate_roster"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch(
                    "pokezero.mcts_eval.search_over_raw_source_run.read_source_owner",return_value={}):
                folder=Path(tmp);s,source,row=self.fixture(folder)
                if mode=="extra":_write_new(folder/"live-selections/7/model/99.json",row)
                elif mode=="schedule":
                    path=folder/"sources/7/bridge-transport.json";v=json.loads(path.read_bytes())
                    v["records"][0]["players"]=["p1"];path.write_text(json.dumps(v))
                else:
                    rows=read_live_source(s,source);path=folder/"live-checks/7/model/collector.json"
                    v=json.loads(path.read_bytes());v["expected"]=[];path.write_text(json.dumps(v))
                with self.assertRaises(ValueError):
                    if mode=="gate_roster":read_gates(s,source,rows)
                    else:read_live_source(s,source)


if __name__=="__main__":unittest.main()
