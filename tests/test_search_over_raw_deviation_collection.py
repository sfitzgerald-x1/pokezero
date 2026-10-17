"""Authored boundaries only; no real seed, search, collection or model load."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval import search_over_raw_deviation_collection as m
from pokezero.mcts_eval.search_over_raw_live_capture import shadow_recipe
import test_search_over_raw_live_exploration as live_tests


class CollectionLawTests(unittest.TestCase):
    def checkpoint(self):
        return SimpleNamespace(model_path="model",checkpoint_path="checkpoint",tables_path="tables")

    def test_new_recipe_is_explicit_and_old_contract_leaves_do_not_change(self):
        checkpoint=self.checkpoint()
        for total,leaves in ((320,list(m.LEAVES)),(1024,["model"]),(4096,["model"])):
            recipe=shadow_recipe(checkpoint,total=total,leaves=leaves)
            self.assertEqual(recipe["schema"],"pokezero.search-over-raw.live-shadow-recipe.v1")
            self.assertNotIn("scan_policy",recipe)
        with self.assertRaises(ValueError):shadow_recipe(checkpoint,total=4096,leaves=list(m.LEAVES))
        new=shadow_recipe(checkpoint,total=4096,leaves=list(m.LEAVES),scan_policy=m.SCAN_POLICY)
        self.assertEqual(new["leaves"],list(m.LEAVES))
        self.assertEqual(new["root_rule"],"all scanned eligible boundaries in source/request order; no quota sampling")

    def test_derived_source_contract_capacity_is_not_a_sampling_quota(self):
        s=dict(contract=dict(panels={"exploration":dict(seeds=[7,8])},namespace="test"))
        ss=m.source_spec(s,7)
        self.assertNotIn("root_slots",s["contract"]["panels"]["exploration"])
        self.assertEqual(len(ss["contract"]["panels"]["exploration"]["root_slots"]),249)
        self.assertEqual({r["source_seed"] for r in ss["contract"]["panels"]["exploration"]["root_slots"]},{7})
        with self.assertRaises(ValueError):m.source_spec(s,9)

    def test_full_catalog_reordered_chronologically_not_six_root_sampled(self):
        source=dict(roots=[dict(source_request_index=i) for i in (4,1,8,2,7,3,6,5)],missing_root_ids=["capacity"])
        reserved=[dict(request_index=i) for i in range(1,9)]
        selected=m._selected_source(source,reserved)
        self.assertEqual([r["source_request_index"] for r in selected["roots"]],list(range(1,9)))
        self.assertEqual(selected["missing_root_ids"],[])
        self.assertEqual(source["missing_root_ids"],["capacity"])
        with self.assertRaises(ValueError):m._selected_source(source,reserved+[dict(request_index=9)])

    def test_prefix_requires_every_attempt_for_both_leaves(self):
        def rows(indices):
            return {leaf:{i:dict(variants={"sampled4":dict(status="MEASURED" if i<3 else "NOT_SCANNED")})
                          for i in indices} for leaf in m.LEAVES}
        reserved=[dict(request_index=i) for i in (1,2)]
        m._check_prefix(rows(range(5)),reserved)
        for bad in ([dict(request_index=2)],[dict(request_index=i) for i in (1,2,3)],list(reversed(reserved))):
            with self.assertRaises(ValueError):m._check_prefix(rows(range(5)),bad)
        changed=rows(range(5));changed["hp_fraction"][1]["variants"]["sampled4"]["status"]="NOT_SCANNED"
        with self.assertRaises(ValueError):m._check_prefix(changed,reserved)

    def test_prior_mismatch_blocks_even_if_action_stays_the_same(self):
        report=live_tests.LiveExplorationTests().report();live=deepcopy(report)
        live["rows"][0]["variants"]["sampled4"]["native_prior_pre_display"]={"0":.250000001,"1":.749999999}
        with self.assertRaisesRegex(ValueError,"correspondence failed"):m.gate_reports(report,live)

    def test_complete_case_is_explicit_diagnostic_beside_full_uncertainty(self):
        roots=[m.ScannedRoot(7,1,0,1,None),m.ScannedRoot(7,2,0,0,None),m.ScannedRoot(8,1,0,None,None)]
        bounds=[dict(model=(.5,.5),hp_fraction=(-1.,1.)),dict(model=(0.,0.),hp_fraction=(-1.,1.)),
                dict(model=(-1.,1.),hp_fraction=(-1.,1.))]
        result=m._complete_case_diagnostics(roots,bounds,108)
        self.assertEqual(result["model"]["identified_roots"],2)
        self.assertEqual(result["model"]["uncertain_roots"],1)
        self.assertEqual(result["model"]["mean"],dict(pooled_roots=.25,equal_seed=.25))
        self.assertEqual(result["hp_fraction"]["identified_roots"],0)
        self.assertIsNone(result["hp_fraction"]["mean"])
        self.assertFalse(result["model"]["complete_case_inference_authorized"])

    def test_action_mismatch_is_an_independent_additional_stop(self):
        row=dict(round=1,raw_action=0,variants={"sampled4":dict(status="MEASURED",action=1,iterations=4096,worlds_searched=4)})
        live=dict(rows=[dict(round=1,raw_action=0,variants={"sampled4":dict(engine_action=1)})])
        m._check_actions([row],live)
        live["rows"][0]["variants"]["sampled4"]["engine_action"]=0
        with self.assertRaisesRegex(ValueError,"action changed"):m._check_actions([row],live)

    def test_measured_action_cannot_hide_wrong_fixed_work(self):
        row=dict(round=1,raw_action=0,variants={"sampled4":dict(status="MEASURED",action=1,iterations=1024,worlds_searched=4)})
        live=dict(rows=[dict(round=1,raw_action=0,variants={"sampled4":dict(engine_action=1)})])
        with self.assertRaisesRegex(ValueError,"work telemetry changed"):m._check_actions([row],live)


class CaptureTests(unittest.TestCase):
    def fixture(self,folder,prior=()):
        capture=m.DeviationCapture.__new__(m.DeviationCapture)
        capture.seed=7;capture.state=dict(roots=list(prior),stop=None,unavailable=0)
        capture.rows={leaf:{} for leaf in m.LEAVES};capture.selectors={leaf:Mock() for leaf in m.LEAVES}
        capture.output=Path(folder);capture.started=time.time();capture._guard=Mock()
        capture.publish=Mock()
        context=SimpleNamespace(seed=7,player_id="p1",decision_round_index=1,
            observation=SimpleNamespace(legal_action_mask=[True,True]))
        decision=SimpleNamespace(action_index=0,metadata=dict(policy_distribution=[.75,.25]))
        return capture,context,decision

    def fake_parent(self,status="MEASURED",fail=False):
        def run(capture,context,decision):
            index=context.decision_round_index
            if index>0:
                reserved=m._reservations(capture.output)
                self.assertEqual(reserved[-1]["request_index"],index)
                self.assertEqual(len(capture.state["roots"]),len(reserved)-1)
            for leaf in m.LEAVES:
                capture.rows[leaf][index]=dict(variants={"sampled4":dict(status=status,action=1)})
            if fail:raise RuntimeError("authored second selector failure")
        return run

    def test_reserve_precedes_both_calls_and_target_makes_rest_raw_only(self):
        prior=[m.ScannedRoot(6,i+1,0,1,1) for i in range(99)]
        with tempfile.TemporaryDirectory() as tmp:
            c,context,decision=self.fixture(tmp,prior)
            # The durable journal for the previous source is original state,
            # not recreated by the next source callback.
            for i in range(99):
                m._write_new(Path(tmp)/"scan-reservations"/(f"{i:04d}.json"),dict(ordinal=i,seed=6,
                    request_index=i+1,raw=0,boundary_sha256="b"*64,both_leaves_required=True,retry_authorized=False))
            with patch.object(m.LiveBoundaryCapture,"__call__",self.fake_parent()),patch(
                    "pokezero.root_prior_correspondence.boundary_fingerprint",return_value="b"*64):
                c(context,decision)
                self.assertEqual(c.state["stop"],"DEVIATION_TARGET")
                self.assertEqual(len(m._reservations(tmp)),100)
                context.decision_round_index=2;c(context,decision)
            self.assertEqual(len(m._reservations(tmp)),100)
            self.assertEqual(len(c.state["roots"]),100)
            for leaf in m.LEAVES:
                self.assertEqual(c.rows[leaf][2]["variants"]["sampled4"]["status"],"NOT_SCANNED")

    def test_opening_warms_selectors_without_reserving_or_counting_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,context,decision=self.fixture(tmp);context.decision_round_index=0
            with patch.object(m.LiveBoundaryCapture,"__call__",self.fake_parent()):c(context,decision)
            self.assertEqual(m._reservations(tmp),[]);self.assertEqual(c.state["roots"],[])

    def test_failed_selector_leaves_reserved_uncertain_attempt_not_redraw(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,context,decision=self.fixture(tmp)
            with patch.object(m.LiveBoundaryCapture,"__call__",self.fake_parent(fail=True)),patch(
                    "pokezero.root_prior_correspondence.boundary_fingerprint",return_value="b"*64):
                with self.assertRaises(RuntimeError):c(context,decision)
            self.assertEqual(len(m._reservations(tmp)),1)
            self.assertEqual(c.state["roots"],[])

    def test_three_unavailable_boundaries_stop_without_outcome_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,context,decision=self.fixture(tmp)
            with patch.object(m.LiveBoundaryCapture,"__call__",self.fake_parent(status="UNAVAILABLE")),patch(
                    "pokezero.root_prior_correspondence.boundary_fingerprint",return_value="b"*64):
                for i in range(1,4):context.decision_round_index=i;c(context,decision)
            self.assertEqual(c.state["stop"],"THREE_CONSECUTIVE_UNAVAILABLE")
            self.assertEqual(len(m._reservations(tmp)),3)
            self.assertTrue(all(root.model is None and root.hp_fraction is None for root in c.state["roots"]))


class ReaderTests(unittest.TestCase):
    def reservation(self,folder,seed=7,ordinal=0,index=1):
        (Path(folder)/"sources"/str(seed)).mkdir(parents=True,exist_ok=True)
        m._write_new(Path(folder)/"scan-reservations"/(f"{ordinal:04d}.json"),dict(ordinal=ordinal,
            seed=seed,request_index=index,raw=0,boundary_sha256="b"*64,both_leaves_required=True,retry_authorized=False))

    def test_failed_source_keeps_every_reserved_root_in_denominator_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp)
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7])}))
            terminal=dict(status="OPERATIONAL_STOP",supervisor_exit_code=1,worker_exit_code=1,result=None)
            with patch.object(m,"summarize",side_effect=lambda roots,intervals,**kwargs:dict(
                    roots=len(roots),intervals=intervals,actions=[(r.model,r.hp_fraction) for r in roots])):
                result=m.readout(s,terminal)
            self.assertEqual(result["scanned_roots"],1)
            self.assertEqual(result["status"],"INCOMPLETE")
            self.assertEqual(result["summaries"]["pooled_roots"]["actions"],[(None,None)])
            self.assertEqual(result["summaries"]["pooled_roots"]["intervals"],
                             [{leaf:(-1.,1.) for leaf in m.LEAVES}])

    def test_clean_worker_cannot_hide_missing_source_or_gate_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp)
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7])}))
            terminal=dict(status="COMPLETE",supervisor_exit_code=0,worker_exit_code=0,result=m._final_binding(s,"ROOT_CAP"))
            with self.assertRaises((ValueError,FileNotFoundError)):m.readout(s,terminal)

    def test_non_contiguous_journal_and_out_of_roster_order_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp,ordinal=1)
            with self.assertRaisesRegex(ValueError,"contiguous"):m._reservations(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp,seed=8);self.reservation(tmp,seed=7,ordinal=1)
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7,8])}))
            terminal=dict(status="STOP",supervisor_exit_code=1,worker_exit_code=1,result=None)
            with self.assertRaisesRegex(ValueError,"source order"):m.readout(s,terminal)

    def test_deleting_last_source_journal_cannot_drop_actual_live_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp,seed=7)
            (Path(tmp)/"sources/8").mkdir()
            m._write_new(Path(tmp)/"live-selections/8/model/1.json",dict(seed=8,round=1,
                variants={"sampled4":dict(status="MEASURED")}))
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7,8])}))
            with self.assertRaisesRegex(ValueError,"missing from reservation denominator"):
                m._check_actual_scan_inventory(s,m._reservations(tmp))

    def test_actual_source_order_includes_zero_eligible_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"sources/8").mkdir(parents=True)
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7,8])}))
            with self.assertRaisesRegex(ValueError,"frozen source prefix"):m._attempted_sources(s)
            (Path(tmp)/"sources/7").mkdir()
            self.assertEqual(m._attempted_sources(s),[7,8])

    def test_original_worker_digest_detects_deleted_or_changed_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp)
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7])}))
            bound=m._final_binding(s,"ROOT_CAP")
            path=Path(tmp)/"scan-reservations/0000.json";row=json.loads(path.read_bytes());row["raw"]=1
            # Mutation is only to a disposable synthetic fixture.
            path.write_text(json.dumps(row))
            terminal=dict(status="COMPLETE",supervisor_exit_code=0,worker_exit_code=0,result=bound)
            with self.assertRaisesRegex(ValueError,"inventories changed"):m.readout(s,terminal)

    def test_agreement_runs_no_games_and_roundtrips_completed_reader(self):
        from pokezero.mcts_eval.search_over_raw_fixed_exploration import _pins
        from pokezero.mcts_eval import search_over_raw_live_exploration as live_controller
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp,index=3)
            report=live_tests.LiveExplorationTests().report()
            original=report["rows"][0];original["raw_action"]=0
            original["boundary_sha256"]="b"*64
            original["variants"]["sampled4"].update(action=0,iterations=4096)
            rows={leaf:{3:deepcopy(original)} for leaf in m.LEAVES}
            public=dict(recorded_action_index=0,current_legal_action_mask=[True,True])
            root=dict(root_id="exploration:7:0",source_request_index=3,
                public_record=public,public_record_sha256=digest(public))
            source=dict(source_seed=7,status="COMPLETE",roots=[root],missing_root_ids=["unused_capacity"])
            s=dict(output_directory=tmp,source_runtime={},prior_bindings={leaf:report["bindings"] for leaf in m.LEAVES},
                contract=dict(panels={"exploration":dict(seeds=[7])},namespace="fixture",
                    leaves=list(m.LEAVES),accepted_native_iterations_total=4096,deviation_triggered_continuations=True))
            ss=m.source_spec(s,7)
            m._write_new(Path(tmp)/"sources/7/source.json",source)
            for leaf in m.LEAVES:
                m._write_new(Path(tmp)/f"live-selections/7/{leaf}/3.json",rows[leaf][3])
                collector={**report,"rows":[rows[leaf][3]]}
                live=deepcopy(collector);live["rows"][0]["variants"]["sampled4"]["engine_action"]=0
                folder=Path(tmp)/f"live-checks/7/{leaf}"
                m._write_new(folder/"collector.json",collector);m._write_new(folder/"live.json",live)
                m._write_new(folder/"gate.json",m.gate_reports(collector,live))
            context=SimpleNamespace(public_materialization_state={},observation=SimpleNamespace(legal_action_mask=[True,True]))
            archive=Mock();archive.selected.return_value=(context,None,{})
            with patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=None),patch(
                    "pokezero.mcts_eval.search_over_raw_archive.retained_archive_root_identity",return_value="a"*64),patch(
                    "pokezero.mcts_eval.search_over_raw_live_exploration.public_request_witness",return_value=dict(
                        consumed_binding=dict(information_key="fixture"))),patch(
                    "pokezero.mcts_eval.search_over_raw_outcome_run.supervise_outcome_run") as launch:
                row=live_controller.measure_root(ss,None,archive,source,root,rows,Mock())
            launch.assert_not_called()
            self.assertEqual(row["cells"],[])
            row.update(receipt_files=_pins(Path(tmp)/"roots"/digest(root["root_id"])),
                       source_receipt_files=_pins(Path(tmp)/"sources/7"))
            m._write_new(Path(tmp)/"completed-roots"/(digest(root["root_id"])+".json"),row)
            live_controller.read_completed_root(ss,row,source,rows)
            terminal=dict(status="COMPLETE",supervisor_exit_code=0,worker_exit_code=0,
                          result=m._final_binding(s,"SOURCE_CAPACITY_EXHAUSTED"))
            with patch.object(m,"read_live_source",return_value=rows):
                result=m.readout(s,terminal)
            self.assertEqual(result["scanned_roots"],1)
            self.assertEqual(result["status"],"DIAGNOSTIC_INSUFFICIENT_SOURCE_COVERAGE")
            self.assertEqual(result["validated_source_clusters"],1)
            self.assertEqual(result["summaries"]["pooled_roots"]["leaves"]["model"][
                "identification_intervals"]["overall_effect"],(0.,0.))

    def test_successful_root_cap_cannot_be_claimed_after_one_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reservation(tmp)
            for leaf in m.LEAVES:
                m._write_new(Path(tmp)/f"live-selections/7/{leaf}/1.json",dict(seed=7,round=1,
                    raw_action=0,boundary_sha256="b"*64,variants={"sampled4":dict(status="MEASURED",action=1)}))
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7])}))
            with self.assertRaisesRegex(ValueError,"stopping law"):
                m._check_successful_stop(s,m._reservations(tmp),"ROOT_CAP")

    def test_three_unavailable_stop_matches_first_boundary_and_saved_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(1,4):
                self.reservation(tmp,ordinal=i-1,index=i)
                for leaf in m.LEAVES:
                    m._write_new(Path(tmp)/f"live-selections/7/{leaf}/{i}.json",dict(seed=7,round=i,
                        raw_action=0,boundary_sha256="b"*64,variants={"sampled4":dict(status="UNAVAILABLE")}))
            s=dict(output_directory=tmp,contract=dict(panels={"exploration":dict(seeds=[7])}))
            m._write_new(Path(tmp)/"scan-stop.json",dict(reason="THREE_CONSECUTIVE_UNAVAILABLE",scanned_roots=3,
                last_seed=7,last_request_index=3,counts_provisional_until_gate=True))
            m._check_successful_stop(s,m._reservations(tmp),"THREE_CONSECUTIVE_UNAVAILABLE")
            with self.assertRaisesRegex(ValueError,"stopping law"):
                m._check_successful_stop(s,m._reservations(tmp),"DEVIATION_TARGET")
            self.reservation(tmp,ordinal=3,index=4)
            for leaf in m.LEAVES:
                m._write_new(Path(tmp)/f"live-selections/7/{leaf}/4.json",dict(seed=7,round=4,
                    raw_action=0,boundary_sha256="b"*64,variants={"sampled4":dict(status="MEASURED",action=0)}))
            with self.assertRaisesRegex(ValueError,"continued after"):
                m._check_successful_stop(s,m._reservations(tmp),"THREE_CONSECUTIVE_UNAVAILABLE")


if __name__=="__main__":unittest.main()
