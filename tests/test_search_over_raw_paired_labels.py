"""Constructor -> paired replay -> eight owned raw/raw clones, authored only."""
from copy import deepcopy
from dataclasses import dataclass,fields,replace
import hashlib
import json
from pathlib import Path
import pickle
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_outcomes import (
    paired_player_names,validate_observed_outcome,
)
from pokezero.mcts_eval.search_over_raw_paired_labels import (
    OwnedPairedLabelJob,PairedLabelRunValidator,supervise_paired_label_run,
    prepare_paired_label_job,label_contract,read_labels,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
import test_search_over_raw_paired_replay as replay


class AuthoredLabelEnv(replay.paired.PairedSyntheticEnv):
    def _observe(self,seat,**kwargs):return self.observe(seat)


@dataclass(frozen=True)
class AuthoredPairedLabelJob(OwnedPairedLabelJob):
    synthetic_process_fixture=True
    def _components(self):return AuthoredLabelEnv(self.config),lambda obs:([0,1],[.25,.75])
    def __call__(self,ownership,phase):
        try:return super().__call__(ownership,phase)
        except BaseException as error:
            (Path(ownership["directory"]).parent/"authored-label-error.json").write_text(str(error))
            raise


class PairedLabelTests(unittest.TestCase):
    def setUp(self):
        self.r=replay.PairedReplayTests();self.r.setUp();self.addCleanup(self.r.doCleanups)
        self.h=self.r.h
        # Prospectively authored source context, not a repair of a produced
        # snapshot: the smaller constructor fixture has no player identities.
        self.h.state=replace(self.h.state,replay=replace(self.h.state.replay,players=dict(p1="A",p2="B")))
        self.r.p.inputs["public_state_sha256"]=digest(_state_payload(self.h.state))
        for side in self.h.snapshot["battle"]["sides"]:
            side["name"]=self.h.state.replay.players[side["id"]]
        text=self.h.bridge.read_text().replace("if(fixture.nonterminalReplay) {",
            "if(fixture.nonterminalReplay && b==='paired-replay-fixture') {")
        text=text.replace("send({type:'terminal',battleId:b});",
            "send({type:'stream',battleId:b,stream:'omniscient',lines:['|win|'+fixture.snapshot.battle.sides[0].name]}); "+
            "send({type:'terminal',battleId:b});")
        text=text.replace("else if(c.type==='release_clone') {",
            "else if(c.type==='reseed') send({type:'reseeded',battleId:b,seed:c.seed});\n else if(c.type==='release_clone') {")
        self.h.bridge.write_text(text)

    def setup_labels(self,*,disable=False):
        native_job,_=self.r.setup_job(disable=disable)
        replay_result=self.r.run_job(native_job);self.replay_reader=self.r.reader(native_job)
        root_binding=dict(root_id="authored-after",source_seed=17,paired_replay_sha256=digest(replay_result),
            native_coordinate_sha256=digest(native_job.native_boundaries[-1]))
        prepared=prepare_paired_label_job(self.replay_reader,replay_result,config=self.h.config,runtime=self.h.runtime,
            root_binding=root_binding,subject="p1",namespace="paired-native-after.v1")
        job=AuthoredPairedLabelJob(**{field.name:getattr(prepared,field.name) for field in fields(prepared)})
        self.replay_result=replay_result
        return job

    def run_labels(self,job):
        self.directory=self.h.root/"paired-labels"
        return supervise_paired_label_run(job,directory=self.directory,binding_sha256="a"*64,
            verify_admission=lambda:True,validate_result=lambda _:True)

    def reader(self,job,allow_synthetic=True):
        return PairedLabelRunValidator(directory=self.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,root_binding=job.root_binding,expected_snapshot=job.snapshot,runtime=job.runtime,
            expected_inputs=job.input_binding(),allow_synthetic=allow_synthetic)

    def test_constructor_replay_and_eight_independently_validated_raw_from_start_cells(self):
        job=self.setup_labels();terminal=self.run_labels(job)
        detail=self.directory/"authored-label-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,detail.read_text() if detail.exists() else terminal)
        result=terminal["result"];reader=self.reader(job)
        self.assertEqual(reader(result["bank"]),digest(result["bank"]))
        self.assertEqual(len(result["outcomes"]),8)
        self.assertEqual([r["signed_outcome"] for r in result["outcomes"]],[1]*8)
        self.assertEqual([r["boundaries"] for r in result["outcomes"]],[1]*8)
        self.assertTrue(all(row["action"]==0 for row in result["outcomes"]))
        cells=result["bank"]["cells"]
        self.assertEqual(len({cell["battle_token"] for cell in cells}),8)
        # The chosen raw action is 1 for BOTH seats, including subject step0;
        # a forced action0 route would fail this assertion.
        first=[next(row["payload"] for row in cell["records"] if row["kind"]=="boundary") for cell in cells]
        self.assertTrue(all(row["actions"]==dict(p1=1,p2=1) for row in first))
        self.assertTrue(all(cell["root_player_lines"]==[] for cell in cells))
        self.assertEqual(job.root_binding["paired_replay_origin"]["result_sha256"],digest(self.replay_result))
        self.assertEqual(job.snapshot.protocol_lines[0],"|turn|2")
        self.assertFalse(result["runtime_qualified"])
        self.assertFalse(reader.last_validation["scientific_collection_authorized"])
        self.assertIsNone(self.reader(job,False)(result["bank"]))
        with self.assertRaises(FileExistsError):self.run_labels(job)
        # The legacy contract is unchanged: no implicit empty-line fallback.
        contract=label_contract(job.input_binding());contract.pop("paired_player_names")
        with self.assertRaisesRegex(ValueError,"root player protocol"):
            validate_observed_outcome(result["outcomes"][0],contract,cells[0],expected_snapshot=job.snapshot,
                runtime_sha256=job.input_binding()["runtime_sha256"])

    def test_unknown_or_conflicting_player_identity_never_becomes_draw(self):
        job=self.setup_labels();terminal=self.run_labels(job)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        result=terminal["result"];expected=job.input_binding();contract=label_contract(expected)
        cell=deepcopy(result["bank"]["cells"][0])
        row=deepcopy(result["outcomes"][0])
        event=next(r for r in cell["records"] if r["kind"]=="event" and r["payload"].get("stream")=="omniscient")
        event["payload"]["lines"]=["|win|Unknown"]
        with self.assertRaisesRegex(ValueError,"unknown winner"):
            validate_observed_outcome(row,contract,cell,expected_snapshot=job.snapshot,runtime_sha256=expected["runtime_sha256"])
        for mode in ("duplicate","parser","protocol","missing"):
            snapshot=deepcopy(job.snapshot)
            if mode=="duplicate":snapshot.bridge_snapshot["battle"]["sides"][1]["name"]=snapshot.bridge_snapshot["battle"]["sides"][0]["name"]
            elif mode=="parser":snapshot=replace(snapshot,replay=replace(snapshot.replay,players=dict(p1="Other",p2="Other2")))
            elif mode=="protocol":snapshot=replace(snapshot,protocol_lines=("|player|p1|Wrong","|player|p2|Wrong2"))
            else:snapshot.bridge_snapshot["battle"]["sides"][0].pop("name")
            with self.subTest(mode=mode),self.assertRaises(ValueError):paired_player_names(snapshot)

    def test_prefix_cannot_prepare_a_label_job(self):
        with self.assertRaisesRegex(ValueError,"cannot substitute"):self.setup_labels(disable=True)

    def test_caps_keep_all_eight_required_cells_null(self):
        text=self.h.bridge.read_text().replace("if(fixture.nonterminalReplay && b==='paired-replay-fixture') {",
            "if(fixture.nonterminalReplay) {")
        # Do not simulate a 2,000-turn PP violation in the authored label shell.
        text=text.replace("for(const seat of ['p1','p2']) current.boundaryRequests[seat].active[0].moves[0].pp--;",
            "if(b==='paired-replay-fixture') for(const seat of ['p1','p2']) current.boundaryRequests[seat].active[0].moves[0].pp--;")
        self.h.bridge.write_text(text)
        job=self.setup_labels();terminal=self.run_labels(job)
        detail=self.directory/"authored-label-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,detail.read_text() if detail.exists() else terminal)
        result=terminal["result"]
        self.assertEqual(self.reader(job)(result["bank"]),digest(result["bank"]))
        self.assertEqual(len(result["outcomes"]),8)
        self.assertTrue(all(r["status"]=="CAPPED" and r["signed_outcome"] is None and r["boundaries"]==250
            for r in result["outcomes"]))

    def test_opponent_only_after_request_does_not_bypass_existing_subject_gate(self):
        job=self.setup_labels()
        snapshot=deepcopy(job.snapshot)
        snapshot.bridge_snapshot["boundaryRequests"]["p1"]=dict(wait=True)
        snapshot.latest_requests["p1"]=dict(wait=True)
        job=replace(job,snapshot=snapshot)
        terminal=self.run_labels(job)
        self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
        self.assertIsNone(terminal["result"])

    def test_missing_cell_or_rehashed_terminal_label_is_rejected(self):
        job=self.setup_labels();terminal=self.run_labels(job)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        result=terminal["result"]
        missing=deepcopy(result);missing["outcomes"].pop()
        with self.assertRaises(ValueError):read_labels(missing,expected=job.input_binding(),snapshot=job.snapshot)
        changed=deepcopy(result);changed["outcomes"][0]["signed_outcome"]=-1
        changed["bank"]["cells"][0]["row_sha256"]=digest(changed["outcomes"][0]);changed["bank_sha256"]=digest(changed["bank"])
        with self.assertRaises(ValueError):read_labels(changed,expected=job.input_binding(),snapshot=job.snapshot)

    def test_failed_label_bridge_cleanup_cannot_bank_a_complete_run(self):
        text=self.h.bridge.read_text().replace("if(fixture.mode==='missing-close')",
            "if(fixture.mode==='missing-close' || owned?.battleId?.startsWith('outcome-'))")
        self.h.bridge.write_text(text)
        job=self.setup_labels();terminal=self.run_labels(job)
        self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
        self.assertIsNone(terminal["result"])
        with self.assertRaises(ValueError):self.reader(job)(dict())
