"""Scheduler control-flow tests, explicitly NOT owner/runtime qualification.

The separate owned integration tests exercise genuine supervisors/readers;
these replace producers only to probe scheduling and irreversible boundaries.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_ledger import _load
from pokezero.mcts_eval.search_over_raw_reconciled import (
    ReconciledPhaseALedger, a1_configurations,a2_configurations,deployment_configurations,controller_input_hashes,
)
import pokezero.mcts_eval.search_over_raw_stage_run as runner
import test_search_over_raw_reconciled as ledger_fixture


class AuthoredPanel:
    def __init__(self,ledger,roots):
        self.ledger=ledger; self.frozen="f"*64; self.by_root={}; self.owners=[]
        for root in roots:
            owner=SimpleNamespace(archive=object(),source=dict(root_id=root),closed=False)
            owner.close=lambda owner=owner:setattr(owner,"closed",True)
            self.by_root[root]=owner; self.owners.append(owner)
        self.owners=tuple(self.owners)
        self.missing_root_ids=tuple(r["root_id"] for r in ledger.contract["panels"]["exploration"]["root_slots"]
            if r["root_id"] not in roots)
    def validate(self):return self
    def _original_inputs(self):return None
    def require_stage_inputs(self,ledger,plans,*,input_unavailable=None,configuration=None):
        gaps={} if input_unavailable is None else input_unavailable
        if set(plans)&set(gaps) or set(plans)|set(gaps)!=set(self.by_root):raise ValueError("authored partition drift")
        return self
    def receipt(self):return dict(contract_sha256=self.ledger.contract_sha256,
        source_count=32,observed_root_ids=list(self.by_root),
        missing_root_ids=list(self.missing_root_ids),source_roster_complete=True)


class Transport:
    synthetic_process_fixture=True
    def __call__(self,**kwargs):
        if runner.RUNTIME_KEY in kwargs["runtime"]:
            return dict(_synthetic_job_class=None,_synthetic_job_kwargs=None)
        return dict(_synthetic_job_class=object,_synthetic_job_kwargs={})


class StageRunTests(unittest.TestCase):
    def setUp(self):
        base=ledger_fixture.ReconciledLedgerTest(); base.setUp();self.addCleanup(base.tearDown)
        self.contract=deepcopy(base.contract);self.contract["input_hashes"]=controller_input_hashes()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.ledger=ReconciledPhaseALedger.create(Path(self.temp.name)/"ledger",self.contract)
        self.root=self.contract["panels"]["exploration"]["root_slots"][0]["root_id"]
        configs={c.identity:c for c in a1_configurations()+a2_configurations()+deployment_configurations()}
        self.recipes={k:runner.StageSelectorRecipe(runtime=dict(configuration=asdict(configs[k[1]])),
            qualification_review_path=str(Path(self.temp.name)/"review.json")) for k in runner.exploration_recipe_keys()}
        self.args=dict(recipes=self.recipes,checkpoint_contract=None,showdown_root="authored",file_hashes={},
            raw_runtime={},history_review_path=str(Path(self.temp.name)/"history.json"),
            continuation_config=None,continuation_runtime={},set_source=None,
            _synthetic_selector_transport=Transport(),_synthetic_continuation_job_class=object)

    def test_empty_full_panel_is_unstarted_and_no_workers_or_a3_configs(self):
        panel=AuthoredPanel(self.ledger,[])
        with patch.object(runner,"TrustedExplorationPanel",AuthoredPanel),patch.object(runner,"produce_archived_selector") as select,\
                patch.object(runner,"produce_reserved_continuation") as continuation:
            result=runner.run_owned_exploration_stages(self.ledger,panel,**self.args)
        select.assert_not_called();continuation.assert_not_called()
        self.assertEqual(result["selection"]["choices"],(("incumbent",None,None),("reference",None,None)))
        self.assertEqual(result["reports"]["A3"]["configurations"],[])
        for stage in ("A1","A2"):
            for row in result["reports"][stage]["configurations"]:
                self.assertEqual(row["status"],"UNSTARTED");self.assertEqual(row["uncertain_roots"],200)
                self.assertEqual(row["identification_interval"],[-1.,1.])
        self.assertFalse(result["holdout_authorized"])

    def test_missing_candidate_recipe_refuses_before_claim_and_launch(self):
        panel=AuthoredPanel(self.ledger,[self.root]);self.recipes.pop(next(iter(self.recipes)))
        with patch.object(runner,"TrustedExplorationPanel",AuthoredPanel),patch.object(runner,"produce_archived_selector") as select,\
                self.assertRaisesRegex(ValueError,"complete prospective"):
            runner.run_owned_exploration_stages(self.ledger,panel,**self.args)
        select.assert_not_called();self.assertTrue(panel.owners[0].closed)
        self.assertFalse((self.ledger.directory/"staged-run-claim.json").exists())

    def test_production_champion_or_simulator_mismatch_refuses_before_claim(self):
        # Admission and the retained panel are mocked solely to reach the real
        # production continuity guard; this is not scientific clearance.
        for mismatch in ("champion","simulator"):
            with self.subTest(mismatch=mismatch):
                panel=AuthoredPanel(self.ledger,[self.root]);args=deepcopy(self.args)
                args.pop("_synthetic_selector_transport");args.pop("_synthetic_continuation_job_class")
                args["checkpoint_contract"]=SimpleNamespace(checkpoint_sha256="a"*64,showdown_root="/authored-source")
                args["continuation_runtime"]=dict(checkpoint_path="/authored-model",
                    files={"/authored-model":"b"*64 if mismatch=="champion" else "a"*64})
                args["continuation_config"]=SimpleNamespace(resolved_showdown_root=lambda:
                    Path("/authored-different" if mismatch=="simulator" else "/authored-source"))
                with patch.object(runner,"TrustedExplorationPanel",AuthoredPanel),\
                        patch.object(self.ledger,"verify_exploration",return_value={}),\
                        patch.object(runner,"produce_archived_selector") as select,\
                        patch("subprocess.Popen") as launch:
                    with self.assertRaisesRegex(ValueError,"retain source champion and simulator"):
                        runner.run_owned_exploration_stages(self.ledger,panel,**args)
                select.assert_not_called();launch.assert_not_called()
                self.assertFalse((self.ledger.directory/"staged-run-claim.json").exists())
                self.assertTrue(panel.owners[0].closed)

    def exercise_productive_schedule(self,*,mutate_final=False,fail=False,gap_mode=None,late_gap=None):
        roots=[self.root]
        if gap_mode=="partial":roots.append(self.contract["panels"]["exploration"]["root_slots"][1]["root_id"])
        gap_roots=set(roots if gap_mode=="all" else roots[1:] if gap_mode=="partial" else ())
        panel=AuthoredPanel(self.ledger,roots);events=[]; retained=[];journal=object();gap_files=[]
        if gap_mode:
            for (stage,identity),spec in tuple(self.recipes.items()):
                cfg=SearchConfiguration(**spec.runtime["configuration"])
                if cfg.arm=="incumbent" and cfg.belief=="public":
                    self.recipes[(stage,identity)]=runner.StageSelectorRecipe(
                        runtime={**spec.runtime,runner.RUNTIME_KEY:True},
                        qualification_review_path=spec.qualification_review_path)
        class Selector:
            def __init__(owner,cfg,status,role):
                owner.job=SimpleNamespace(configuration=cfg);owner.selection=dict(status=status,action=None if status=="UNQUALIFIED_AT_CEILING" else 1)
                owner.reader=object();owner.frozen="frozen";owner.changed=False
            def _inputs(owner):return "changed" if owner.changed else owner.frozen
            def validate(owner,*args):
                if mutate_final and owner is retained[-1]:retained[0].changed=True
        def select(ledger,stage,cfgid,**kwargs):
            cfg=next(c for c in ledger._stage_configs(stage) if c.identity==cfgid)
            events.append(("selector",stage,cfgid,kwargs["selector"]))
            if fail and cfg.belief=="oracle":raise RuntimeError("authored integrity failure")
            if kwargs["selector"]=="search" and cfg.arm=="incumbent" and cfg.belief=="public" and kwargs["root_id"] in gap_roots:
                root=kwargs["root_id"]
                receipt=dict(schema="authored-control-flow-only",status="INPUT_UNAVAILABLE",root_id=root,
                    configuration=cfgid,selector_launched=False,selector_attempted=False,observed_zero_work=False)
                path=runner._actor_unavailable_path(ledger,root,cfgid)
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_text(json.dumps(receipt));gap_files.append(path)
                return None,receipt
            if kwargs["prior_owner"] is not None:return kwargs["prior_owner"],dict(selector_launched=False)
            status="UNQUALIFIED_AT_CEILING" if kwargs["selector"]=="search" and cfg.leaf=="model" else "SELECTED"
            owner=Selector(cfg,status,kwargs["selector"]);retained.append(owner)
            return owner,dict(selector_launched=True)
        class Plan:
            def __init__(plan,archive,source,**kwargs):
                plan.root_id=kwargs["root_id"];plan.selections=kwargs["selections"];plan.configurations=kwargs["configurations"]
                plan.archive=archive;plan.source=source
            def revalidate(plan):return None
            def accept_continuation_result(plan,*args,**kwargs):return None
        def continuation(ledger,stage,cfgid,**kwargs):
            events.append(("continuation",stage,cfgid));return None,dict(continuation_launched=False)
        def ingest(stage,cfgid,**kwargs):
            cfg=next(c for c in self.ledger._stage_configs(stage) if c.identity==cfgid)
            expected=set(roots)-gap_roots if cfg.arm=="incumbent" and cfg.belief=="public" else set(roots)
            self.assertIs(kwargs["source_panel"],panel);self.assertEqual(set(kwargs["plans"]),expected)
            zero=cfg.leaf=="model"
            return self.ledger.record_stage(stage,cfgid,{} if zero else {self.root:[0.,0.]},
                accepted_units=0 if zero else 1,status="UNQUALIFIED_AT_CEILING" if zero else "QUALIFIED",
                evidence_sha256=digest(dict(authored_control_flow=stage,configuration=cfgid)))
        original_freeze=self.ledger.freeze
        def freeze():
            result=original_freeze()
            if late_gap:
                path=gap_files[0]
                if late_gap=="bytes":path.write_text(path.read_text()+"\n")
                else:path.with_name(path.name.replace("-input-unavailable.json",".json")).write_text("{}")
            return result
        state_payload=runner._state_payload
        with patch.object(runner,"TrustedExplorationPanel",AuthoredPanel),patch.object(runner,"produce_archived_selector",side_effect=select),\
                patch.object(runner,"ArchivedRootEvaluationPlan",Plan),patch.object(runner,"produce_reserved_continuation",side_effect=continuation),\
                patch.object(runner,"ContinuationAdmissionJournal",SimpleNamespace(create=lambda *a,**k:journal)),\
                patch.object(runner,"_owner_inputs",return_value="authored control-flow only"),\
                patch.object(runner,"_state_payload",side_effect=lambda value:
                    {} if isinstance(value,SimpleNamespace) and hasattr(value,"receipt") else state_payload(value)),\
                patch.object(self.ledger,"record_owned_stage_evidence",side_effect=ingest),\
                patch.object(runner,"read_actor_input_unavailable",side_effect=lambda ledger,owner,receipt,**kw:receipt),\
                patch("pokezero.mcts_eval.search_over_raw_selector_producer.read_recorded_actor_input_unavailable",
                    side_effect=lambda ledger,receipt,**kw:receipt),\
                patch.object(self.ledger,"freeze",side_effect=freeze),\
                patch("pokezero.mcts_eval.search_over_raw_archive.capture_archived_team_oracle",
                    return_value=(SimpleNamespace(receipt=lambda:{}),{})):
            try:result=runner.run_owned_exploration_stages(self.ledger,panel,**self.args)
            finally:self.assertTrue(panel.owners[0].closed)
        return result,events

    def test_independent_zero_a1_a2_actual_freeze_and_only_eligible_a3(self):
        result,events=self.exercise_productive_schedule()
        self.assertEqual(result["evaluator_freeze"]["evaluators"],(("incumbent","hp_fraction"),("reference","hp_fraction")))
        self.assertEqual(result["launches"]["raw"],1)
        self.assertEqual(result["launches"]["reused_a3_records"],2)
        self.assertEqual(result["launches"]["search"],10)
        self.assertFalse(any(e[0]=="continuation" and e[1]=="A1" for e in events))
        for event in events:
            if event[1]=="A3":
                cfg=next(c for c in deployment_configurations() if c.identity==event[2])
                self.assertEqual(cfg.leaf,"hp_fraction");self.assertIn(cfg.seconds,(3.,10.))
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        a2=_load(self.ledger.directory/"A2"/f"{hp.identity}.json")
        a3=_load(self.ledger.directory/"A3"/f"{hp.identity}.json")
        self.assertEqual(a2["evidence_sha256"],a3["evidence_sha256"])

    def test_late_owner_mutation_refuses_instead_of_freezing_selection(self):
        with self.assertRaisesRegex(ValueError,"earlier retained selector"):
            self.exercise_productive_schedule(mutate_final=True)
        self.assertFalse((self.ledger.directory/"selection.json").exists())
        self.assertTrue((self.ledger.directory/"operational-stop.json").exists())

    def test_integrity_failure_stops_without_retry_or_a2_promotion(self):
        with self.assertRaisesRegex(RuntimeError,"authored integrity failure"):
            self.exercise_productive_schedule(fail=True)
        self.assertTrue((self.ledger.directory/"operational-stop.json").exists())
        self.assertFalse((self.ledger.directory/"a2-freeze.json").exists())

    def test_all_unavailable_incumbent_is_unstarted_without_blocking_reference(self):
        result,events=self.exercise_productive_schedule(gap_mode="all")
        for stage in ("A1","A2"):
            for row in result["reports"][stage]["configurations"]:
                if row["arm"]=="incumbent" and row["belief"]=="public":
                    self.assertEqual(row["status"],"UNSTARTED")
                    self.assertEqual(row["identification_interval"],[-1.,1.])
                    self.assertEqual(row["uncertain_roots"],200)
        self.assertEqual(result["evaluator_freeze"]["evaluators"],(("incumbent",None),("reference","hp_fraction")))
        self.assertTrue(result["input_unavailable_by_root_configuration"])
        self.assertFalse(any(e[0]=="continuation" and e[2] in {
            cfg.identity for cfg in a1_configurations()+a2_configurations() if cfg.arm=="incumbent"} for e in events))

    def test_partial_unavailable_roots_are_skipped_without_changing_roster(self):
        result,events=self.exercise_productive_schedule(gap_mode="partial")
        self.assertEqual(result["evaluator_freeze"]["evaluators"],(("incumbent","hp_fraction"),("reference","hp_fraction")))
        self.assertEqual(len(result["input_unavailable_by_root_configuration"]),4)
        self.assertTrue(all(row["root_slots"]==200 and row["source_seeds"]==32
            for stage in result["reports"].values() for row in stage["configurations"]))

    def test_last_freeze_cannot_rewrite_an_unavailable_receipt(self):
        with self.assertRaisesRegex(ValueError,"candidate-attempt absence changed"):
            self.exercise_productive_schedule(gap_mode="all",late_gap="bytes")
        self.assertTrue((self.ledger.directory/"operational-stop.json").exists())

    def test_last_freeze_cannot_create_a_candidate_claim_for_an_unavailable_root(self):
        with self.assertRaisesRegex(ValueError,"candidate-attempt absence changed"):
            self.exercise_productive_schedule(gap_mode="all",late_gap="claim")
        self.assertTrue((self.ledger.directory/"operational-stop.json").exists())


if __name__ == "__main__":unittest.main()
