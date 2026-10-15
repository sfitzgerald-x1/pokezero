"""Actual integrated collector/controller ownership over authored science data."""
from copy import deepcopy
from dataclasses import dataclass,replace
import json
import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pokezero.mcts_eval.search_over_raw import digest,rng_seed
from pokezero.mcts_eval.search_over_raw_paired_chain import (
    paired_chain_law,paired_leaf_law,paired_label_coordinates,request_correspondence,pin_original_result,
    paired_label_descriptors,paired_label_report,paired_value_diagnostics)
from pokezero.mcts_eval.search_over_raw_native_leaf_cells import plan_native_leaf_cells
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
import test_search_over_raw_native_chain as native
import test_search_over_raw_native_leaf_cells as cells
import test_search_over_raw_paired_materialization as material
import test_search_over_raw_paired_replay as replay
import test_search_over_raw_paired_labels as labels


def paired_bridge(original):
    text=original.replace("send({type:'ready',battleId:b,requested:['p1','p2']}); }",
        "for(const seat of ['p1','p2']) send({type:'stream',battleId:b,stream:seat,lines:['|request|'+JSON.stringify(fixture.snapshot.boundaryRequests[seat])]}); "+
        "send({type:'ready',battleId:b,requested:['p1','p2']}); }",1)
    text=text.replace("if(fixture.nonterminalReplay) {","if(b.startsWith('nr-')) {")
    text=text.replace("current.battle.log.push('|turn|2');",
        "current.battle.log.push('|turn|4'); send({type:'stream',battleId:b,stream:'omniscient',lines:['|turn|4']});")
    text=text.replace("send({type:'ready',battleId:b,requested:['p1','p2']});\n   } else {",
        "for(const seat of ['p1','p2']) send({type:'stream',battleId:b,stream:seat,lines:['|request|'+JSON.stringify(current.boundaryRequests[seat])]}); "+
        "send({type:'ready',battleId:b,requested:['p1','p2']});\n   } else {")
    text=text.replace("send({type:'terminal',battleId:b});",
        "send({type:'stream',battleId:b,stream:'omniscient',lines:['|win|'+fixture.snapshot.battle.sides[0].name]}); send({type:'terminal',battleId:b});")
    return text.replace("else if(c.type==='release_clone') {",
        "else if(c.type==='reseed') send({type:'reseeded',battleId:b,seed:c.seed});\n else if(c.type==='release_clone') {")


@dataclass(frozen=True)
class AuthoredPairedNativeJob(native.AuthoredRetainedNativeJob):
    def _run(self,ownership,phase):
        original=native.construction_fixture._override
        def draw():return replace(original(),observation_format_id=self.context.public_materialization_state.observation_format_id)
        with patch.object(native.construction_fixture,"_override",side_effect=draw):
            return super()._run(ownership,phase)


class AuthoredPairedChainTransport:
    synthetic_process_fixture=True
    def __call__(self,*,role,job,native_owner):
        cls={"materialization":material.AuthoredPairedJob,"replay":replay.AuthoredPairedPrefixJob,
            "labels":labels.AuthoredPairedLabelJob}[role]
        return dict(job_class=cls,job_kwargs={})


class PairedChainLawTests(unittest.TestCase):
    def population(self):
        from pokezero.mcts_eval.search_over_raw_native_edge import ENDPOINT_FIELDS
        h=cells.NativeLeafCellPlanTests();h.setUp();self.addCleanup(h.doCleanups)
        plan=plan_native_leaf_cells(h.fixture(),run_binding=cells.RUN,consumed_binding=cells.INPUT_BINDING,
            ancestry_contract=cells.CONTRACT,paired_labels=True)
        reports=[];records=[]
        for event in plan.receipt()["events"]:
            reports.append(paired_label_report(dict(event_id=event["event_id"],event=deepcopy(event),
                construction_ordinal=event["canonical_construction_ordinal"],evaluation_ordinal=event["evaluation_ordinal"],
                job=None,status="UNSTARTED_UNCERTAIN",reason="CANONICAL_REPLAY_UNAVAILABLE")))
            count=event["required_after_snapshot_index"]+1
            for alias in event["construction_aliases"]:
                records.append(dict(construction_ordinal=alias,evaluation_ordinal=event["evaluation_ordinal"],
                    native_capture_sha256=event["native_capture_sha256"],required_boundaries=count,
                    boundary_observations=[dict(ordinal=i,observed_required_fields=dict(
                        **{field:"MATCH" for field in ENDPOINT_FIELDS},
                        action_identity="MATCH" if i else "NOT_APPLICABLE",branch_transition="UNCHECKABLE")) for i in range(count)],
                    supported_prefix_observations=[dict(ordinal=i,observed_prefix_match=False) for i in range(count)]))
        return plan,reports,dict(ancestry_root_join=dict(native_selected_sha256=plan.receipt()["native_selected_sha256"]),records=records)

    def test_successful_noncanonical_alias_cannot_replace_missing_or_unreached_canonical_endpoint(self):
        from pokezero.mcts_eval.search_over_raw_paired_replay import PairedPrefixRunValidator
        plan,_,_=self.population();event=plan.receipt()["events"][0]
        canonical=event["canonical_construction_ordinal"]
        alternate=next(a for a in event["construction_aliases"] if a!=canonical)
        def replay_row(alias,status):
            reader=object.__new__(PairedPrefixRunValidator)
            reader.expected=dict(root_origin=dict(native_selected_sha256=event["native_selected_sha256"],
                native_capture_sha256=event["native_capture_sha256"],construction_ordinal=alias,
                evaluation_ordinal=event["evaluation_ordinal"]),request_ids=list(range(event["required_after_snapshot_index"]+1)))
            return dict(construction_ordinal=alias,evaluation_ordinal=event["evaluation_ordinal"],validator=reader,
                result=dict(capture=dict(termination=dict(status=status))))
        successful=replay_row(alternate,"FULL_PATH_OBSERVED")
        with tempfile.TemporaryDirectory() as temporary,patch.object(PairedPrefixRunValidator,"__call__",lambda _,capture:digest(capture)), \
                patch("pokezero.mcts_eval.search_over_raw_paired_chain.prepare_paired_label_job",
                    side_effect=AssertionError("noncanonical endpoint used")) as prepare:
            for rows,reason in (([successful],"CANONICAL_REPLAY_UNAVAILABLE"),
                    ([replay_row(canonical,"PREFIX_OBSERVED"),successful],"DESIGNATED_AFTER_UNREACHED")):
                descriptors=paired_label_descriptors(plan,rows,config=None,runtime=None,attempt_root=Path(temporary).resolve())
                self.assertEqual(descriptors[0]["reason"],reason)
                self.assertIsNone(descriptors[0]["job"])
                report=paired_label_report(descriptors[0])
                self.assertEqual(len(report["cells"]),8)
                self.assertTrue(all(c["signed_outcome"] is None for c in report["cells"]))
            prepare.assert_not_called()

    def test_prediction_bounds_keep_all_missing_cells_and_separate_scoped_mismatches(self):
        plan,reports,joined=self.population()
        for report in reports:
            report.update(status="LABELS_RECORDED",reason=None)
            for cell in report["cells"]:cell.update(status="COMPLETE",signed_outcome=1,boundaries=1)
        # One cap and a completely unavailable second event remain uncertain.
        reports[0]["cells"][0].update(status="CAPPED_UNCERTAIN",signed_outcome=None)
        reports[1]["status"]="UNSTARTED_UNCERTAIN";reports[1]["reason"]="SUBJECT_REQUEST_UNAVAILABLE"
        for cell in reports[1]["cells"]:cell.update(status="UNSTARTED_UNCERTAIN",signed_outcome=None,boundaries=0)
        first=plan.receipt()["events"][0]
        canonical=next(row for row in joined["records"] if row["construction_ordinal"]==first["canonical_construction_ordinal"]
            and row["evaluation_ordinal"]==first["evaluation_ordinal"])
        canonical["boundary_observations"][-1]["observed_required_fields"]["hp"]="MISMATCH"
        before=digest(dict(reports=reports,joined=joined))
        result=paired_value_diagnostics(plan,reports,joined)
        self.assertEqual((result["required_events"],result["required_labels"],result["unknown_labels"]),(2,16,9))
        self.assertEqual(result["events"][0]["win_score_interval"],[7/8,1])
        self.assertEqual(result["events"][1]["win_score_interval"],[0,1])
        self.assertEqual(result["scoped_strata"]["SCOPED_FIELDS_MISMATCH"]["required_labels"],8)
        self.assertEqual(result["scoped_strata"]["SCOPED_FIELDS_MATCH"]["required_labels"],8)
        self.assertEqual(sum(b["labels"] for b in result["calibration"]),16)
        self.assertEqual(sum(b["unknown_labels"] for b in result["calibration"]),9)
        self.assertFalse(result["native_evaluated_state_equivalence"])
        self.assertFalse(result["scientific_calibration_evidence"])
        self.assertEqual(digest(dict(reports=reports,joined=joined)),before)
        canonical["boundary_observations"][-1]["observed_required_fields"]["hp"]="UNCHECKABLE"
        result=paired_value_diagnostics(plan,reports,joined)
        self.assertEqual(result["scoped_strata"]["SCOPED_FIELDS_UNCHECKABLE"]["required_labels"],8)

    def test_calibration_rejects_shortened_alias_or_label_rosters_foreign_capture_and_boolean_outcomes(self):
        for mutation in ("alias","cell","capture","boolean","replicate"):
            plan,reports,joined=self.population()
            if mutation=="alias":joined["records"].pop()
            elif mutation=="cell":reports[0]["cells"].pop()
            elif mutation=="capture":joined["records"][0]["native_capture_sha256"]="0"*64
            elif mutation=="boolean":reports[0]["cells"][0].update(status="COMPLETE",signed_outcome=True)
            else:reports[0]["cells"][0]["replicate"]=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):paired_value_diagnostics(plan,reports,joined)

    def test_action_numbers_cannot_hide_swapped_move_or_party_identities(self):
        import test_search_over_raw_showdown_boundary as fixture
        raw=fixture.snapshot()
        for request in raw["boundaryRequests"].values():
            request["active"][0]["moves"].append(dict(id="tackle",disabled=False))
            request["side"]["pokemon"]=[dict(active=i==0,condition="100/100",ident="same nickname",
                details="same species",item="item-"+str(i)) for i in range(3)]
        original=SimpleNamespace(bridge_snapshot=raw,latest_requests=deepcopy(raw["boundaryRequests"]),
            replay=SimpleNamespace(requests=deepcopy(raw["boundaryRequests"])))
        self.assertIsNotNone(request_correspondence(original))
        for field in ("latest_requests","parsed_requests"):
            for kind in ("moves","bench"):
                snapshot=deepcopy(original)
                requests=snapshot.latest_requests if field=="latest_requests" else snapshot.replay.requests
                request=requests["p1"]
                if kind=="moves":request["active"][0]["moves"].reverse()
                else:request["side"]["pokemon"][1:]=reversed(request["side"]["pokemon"][1:])
                from pokezero.showdown import showdown_choices_for_request
                self.assertEqual(showdown_choices_for_request(request,"p1"),
                    showdown_choices_for_request(raw["boundaryRequests"]["p1"],"p1"))
                with self.subTest(field=field,kind=kind):self.assertIsNone(request_correspondence(snapshot))

    def test_original_file_pin_precedes_any_rehashed_private_artifact_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()
            job=b"original private job";artifact=b"original context"
            terminal=json.dumps(dict(result=dict(paired_context=dict(artifact_sha256=hashlib.sha256(artifact).hexdigest())))).encode()
            (directory/"job.pickle").write_bytes(job);(directory/"paired-context.pickle").write_bytes(artifact)
            (directory/"terminal.json").write_bytes(terminal)
            readout=dict(process_receipts_validated=True,receipt_file_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                for name in ("terminal.json","paired-context.pickle")})
            self.assertIn("paired_context",pin_original_result(directory,job_sha256=hashlib.sha256(job).hexdigest(),readout=readout))
            changed=b"replacement context";(directory/"paired-context.pickle").write_bytes(changed)
            (directory/"terminal.json").write_text(json.dumps(dict(result=dict(paired_context=dict(
                artifact_sha256=hashlib.sha256(changed).hexdigest())))))
            with patch("pickle.loads",side_effect=AssertionError("untrusted artifact deserialized")) as load:
                with self.assertRaisesRegex(ValueError,"independently frozen"):
                    pin_original_result(directory,job_sha256=hashlib.sha256(job).hexdigest(),readout=readout)
            load.assert_not_called()
    def test_explicit_seed_law_uses_actual_continuation_formula_and_preserves_events(self):
        h=cells.NativeLeafCellPlanTests();h.setUp();self.addCleanup(h.doCleanups)
        # Reuse only authored original native exports; no source or label jobs.
        selected,run,consumed,contract=h.fixture(),cells.RUN,cells.INPUT_BINDING,cells.CONTRACT
        old=plan_native_leaf_cells(selected,run_binding=run,consumed_binding=consumed,ancestry_contract=contract)
        new=plan_native_leaf_cells(selected,run_binding=run,consumed_binding=consumed,ancestry_contract=contract,paired_labels=True)
        self.assertEqual(new.receipt()["schema"],"pokezero.search-over-raw.native-after-leaf-cell-plan.v2")
        self.assertEqual([e["event_id"] for e in old.receipt()["events"]],[e["event_id"] for e in new.receipt()["events"]])
        for event in new.receipt()["events"]:
            root,namespace=paired_label_coordinates(event,contract.sample_seed)
            self.assertEqual(event["labels"][0]["chance_stream_key"],rng_seed(namespace,root,0))
            self.assertNotEqual(root,paired_label_coordinates(event,contract.sample_seed+1)[0])
        self.assertEqual(paired_chain_law()["leaf_cells"],paired_leaf_law())

    def test_v6_entry_requires_explicit_paired_recipe_and_preserves_legacy_scope(self):
        import pokezero.mcts_eval.search_over_raw_exploration_entry as entry
        from pokezero.mcts_eval.search_over_raw_exploration_workflow import NativeChainRecipe
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        from test_search_over_raw_workflow_preflight import DiagnosticWorkflowPreflightTests
        helper=DiagnosticWorkflowPreflightTests();self.addCleanup(helper.doCleanups)
        h=helper.prepare(capture_actor_history=False)
        spec=deepcopy(h.spec);spec["schema"]=entry.PAIRED_VALUE_WORKFLOW_SCHEMA
        spec["diagnostics"]["native_chain_recipe"]=_state_payload(NativeChainRecipe(
            unqualified_reason="authored unavailable runtime",preserve_replay_prefix=True,paired_continuation_labels=True))
        with patch("subprocess.Popen") as launch:
            prepared=entry._prepared(h.ledger,spec)
        launch.assert_not_called()
        self.assertTrue(prepared["native_chain_recipe"].paired_continuation_labels)
        legacy=deepcopy(spec);legacy["schema"]=entry.ACTOR_WORKFLOW_SCHEMA
        with self.assertRaisesRegex(ValueError,"explicit prospective v6"):entry._prepared(h.ledger,legacy)
        spec["diagnostics"]["native_chain_recipe"]=_state_payload(NativeChainRecipe(unqualified_reason="not paired"))
        with self.assertRaisesRegex(ValueError,"explicit prospective v6"):entry._prepared(h.ledger,spec)


class PairedFullWorkflowTests(unittest.TestCase):
    def test_actual_full_collector_labels_and_completed_reader_rejects_prediction_tamper(self):
        from test_search_over_raw_productive_panel import ProductivePanelAuditor
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            terminal=supervise_fixed_work(ProductivePanelAuditor(integrated_diagnostics=True,
                paired_native_values=True,preserve_replay_prefix=True),directory=directory,binding_sha256="a"*64,
                verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            inputs=terminal["result"]["authored_external_readback_inputs"]
            from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode
            from pokezero.mcts_eval.search_over_raw_exploration_workflow import read_recorded_workflow
            from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger
            from test_engine_world import _payload,_dex
            recipes={row["configuration"]:_decode(row["recipe"]) for row in inputs["diagnostic_recipes"]}
            def read():
                with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
                    return read_recorded_workflow(ReconciledPhaseALedger(inputs["ledger_directory"]),inputs["report"],
                        diagnostic_recipes=recipes,native_chain_recipe=_decode(inputs["native_chain_recipe"]),
                        diagnostic_review_path=inputs["diagnostic_review_path"],diagnostic_file_hashes=inputs["diagnostic_file_hashes"],
                        checkpoint_contract=_decode(inputs["checkpoint_contract"]),auditor_directory=directory,
                        auditor_pid=terminal["worker_pid"],allow_synthetic=True,_synthetic_helpers=native.AuthoredChainHelpers(),
                        expected_outer_terminal=terminal)
            readback=read();self.assertEqual(readback["fixed_slots"],192)
            chain=next(row["diagnostic"] for row in inputs["report"]["native_chains"] if row["status"]=="CHAIN_RECORDED")
            self.assertEqual((chain["retained_unique_leaf_events"],chain["required_leaf_label_cells"],
                chain["leaf_label_process_launches"],chain["completed_leaf_label_cells"]),(2,16,2,16))
            self.assertTrue(all(c["signed_outcome"]==1 for row in chain["paired_labels"] for c in row["cells"]))
            self.assertEqual(len(chain["attempt_costs"]),4+8+2)
            diagnostics=chain["paired_value_diagnostics"]
            self.assertEqual((diagnostics["required_labels"],diagnostics["unknown_labels"]),(16,0))
            self.assertEqual(diagnostics["scoped_strata"]["SCOPED_FIELDS_MISMATCH"]["required_labels"],16)
            self.assertFalse(diagnostics["scientific_calibration_evidence"])
            self.assertTrue(readback["paired_value_readback"]["scoped_diagnostics_revalidated"])
            event=chain["paired_labels"][0]
            path=Path(event["attempt_directory"])/"terminal.json";original=path.read_bytes()
            value=json.loads(original);value["result"]["root_binding"]["model_signed_value"]=-.999
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):read()
            path.write_bytes(original);self.assertEqual(read(),readback)
            constructor=next(item["owner"] for item in chain["owning_jobs"] if item["owner"]["role"]=="materialization")
            folder=Path(constructor["directory"]);artifact=folder/"paired-context.pickle"
            originals={name:(folder/name).read_bytes() for name in ("paired-context.pickle","terminal.json","worker-result.json")}
            changed=originals["paired-context.pickle"]+b"replacement"
            artifact.write_bytes(changed)
            for name in ("terminal.json","worker-result.json"):
                value=json.loads(originals[name]);value["result"]["paired_context"]["artifact_sha256"]=hashlib.sha256(changed).hexdigest()
                (folder/name).write_text(json.dumps(value))
            load=pickle.loads
            def reject_changed(raw,*args,**kwargs):
                if raw==changed:raise AssertionError("replacement paired context was loaded")
                return load(raw,*args,**kwargs)
            with patch("pickle.loads",side_effect=reject_changed),self.assertRaisesRegex(ValueError,"independently frozen"):
                read()
            for name,raw in originals.items():(folder/name).write_bytes(raw)
            self.assertFalse(readback["scientific_measurement_validated"])
