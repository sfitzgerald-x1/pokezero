"""Public entry and scheduler availability; authored data, no scientific work."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_reconciled import (
    ReconciledPhaseALedger, a1_configurations, a2_configurations, deployment_configurations)
import pokezero.mcts_eval.search_over_raw_stage_run as stage
import pokezero.mcts_eval.search_over_raw_exploration_entry as entry
import test_search_over_raw_exploration_entry as entry_fixture
import test_search_over_raw_stage_run as stage_fixture


class ProspectiveAvailabilityTests(unittest.TestCase):
    def make(self, predicate=None, mutate_review=None):
        h=entry_fixture.ExplorationEntryTests();h.setUp();self.addCleanup(h.doCleanups)
        configs={c.identity:c for c in a1_configurations()+a2_configurations()+deployment_configurations()}
        unavailable=[]
        def alter(spec,contract):
            for row in spec["recipes"]:
                cfg=configs[row["configuration"]]
                disabled=(cfg.arm=="reference" or cfg.belief=="oracle"
                    or row["stage"]=="A3" and cfg.seconds==10.) if predicate is None else predicate(row["stage"],cfg)
                if not disabled:continue
                stage_name,identity,path=row["stage"],row["configuration"],row["review_path"]
                reason="authored prospectively unavailable configuration"
                review=stage.unavailable_configuration_review_template(contract,stage=stage_name,
                    configuration=identity,reason=reason)
                review.update(status="CLEAR_PROSPECTIVE_CONFIGURATION_UNAVAILABLE",
                    independent_review_completed=True,reviewed_before_first_source_work=True)
                if mutate_review:mutate_review(review)
                Path(path).write_text(json.dumps(review))
                row.clear();row.update(stage=stage_name,configuration=identity,
                    unavailable_reason=reason,review_path=path)
                unavailable.append((stage_name,identity))
        h.make(alter=alter)
        return h,unavailable

    def test_public_entry_preflights_unavailable_candidates_without_runtime_or_work(self):
        h,unavailable=self.make()
        before=entry._evidence_inventory(h.directory)
        with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
            _,prepared=h.preflight()
        launch.assert_not_called();catalog.assert_not_called()
        self.assertEqual(before,entry._evidence_inventory(h.directory))
        self.assertEqual(len(prepared["recipes"]),len(stage.exploration_recipe_keys()))
        self.assertTrue(all(type(prepared["recipes"][k]) is stage.ProspectivelyUnavailableStageRecipe
                            for k in unavailable))
        self.assertTrue(any(type(r) is stage.StageSelectorRecipe for r in prepared["recipes"].values()))

    def productive(self, *, late_claim=False, late_review=False, unused_a3=False):
        h,unavailable=self.make();_,prepared=h.preflight()
        scheduler=stage_fixture.StageRunTests();scheduler.setUp();self.addCleanup(scheduler.doCleanups)
        contract=deepcopy(h.ledger.contract)
        contract.update(execution_ready=False,collection_authorized=False,remaining_gates=["authored not admitted"])
        ledger=ReconciledPhaseALedger.create(Path(scheduler.temp.name)/"prepared-public-entry",contract)
        scheduler.contract=contract;scheduler.ledger=ledger
        scheduler.root=contract["panels"]["exploration"]["root_slots"][0]["root_id"]
        scheduler.recipes=prepared["recipes"];scheduler.args["recipes"]=scheduler.recipes
        original_freeze=ledger.freeze
        first=(next(key for key in unavailable if key[0]=="A3" and key[1]
            ==SearchConfiguration("incumbent",leaf="model",seconds=10.).identity)
            if unused_a3 else unavailable[0])
        def mutate_after_freeze():
            result=original_freeze()
            if late_claim:
                token=digest(dict(root_id=scheduler.root,configuration=first[1]))
                path=ledger.directory/"selector-producer-claims"/(token+".json")
                path.parent.mkdir(exist_ok=True);path.write_text("{}")
            if late_review:
                path=Path(scheduler.recipes[first].review_path)
                path.write_text(path.read_text()+"\n")
            return result
        if late_claim or late_review:
            ledger.freeze=mutate_after_freeze
        result,events=scheduler.exercise_productive_schedule()
        return h,ledger,unavailable,result,events

    def test_public_prepared_roster_runs_available_arm_and_never_launches_disabled_candidates(self):
        h,ledger,unavailable,result,events=self.productive()
        self.assertEqual(result["evaluator_freeze"]["evaluators"],(("incumbent","hp_fraction"),("reference",None)))
        self.assertEqual(result["launches"]["raw"],1)
        disabled_ids={identity for _,identity in unavailable}
        self.assertFalse(any(event[2] in disabled_ids for event in events))
        for name in ("A1","A2"):
            for row in result["reports"][name]["configurations"]:
                if (name,row["configuration_sha256"]) not in unavailable:continue
                self.assertEqual(row["status"],"UNSTARTED")
                self.assertEqual(row["uncertain_roots"],200)
                self.assertEqual(row["identification_interval"],[-1.,1.])
                evidence=json.loads((ledger.directory/name/(row["configuration_sha256"]+"-root-evidence.json")).read_text())
                self.assertIs(evidence["selector_attempted"],False)
                self.assertIs(evidence["observed_zero_work"],False)
        self.assertFalse(any(row["arm"]=="reference" for row in result["reports"]["A3"]["configurations"]))
        unused=SearchConfiguration("incumbent",leaf="model",seconds=10.)
        self.assertFalse((ledger.directory/"A3"/(unused.identity+"-root-evidence.json")).exists())
        self.assertFalse(result["holdout_authorized"])

    def test_shared_identity_cannot_be_unavailable_in_a1_but_executable_in_a2(self):
        h,_=self.make(predicate=lambda name,cfg:name=="A1" and cfg.belief=="public")
        h.refuse_without_write("consistent prospective availability")

    def test_unreviewed_or_falsely_qualified_unavailability_refuses(self):
        for mutate in (lambda r:r.update(independent_review_completed=False),
                       lambda r:r.update(runtime_qualified=True),
                       lambda r:r.update(observed_zero_work=True)):
            h,_=self.make(mutate_review=mutate)
            h.refuse_without_write("reviewed unavailability")

    def test_unavailable_row_cannot_smuggle_an_executable_runtime(self):
        h,_=self.make()
        spec=deepcopy(h.spec)
        row=next(r for r in spec["recipes"] if "unavailable_reason" in r)
        row["runtime"]={"hidden execution":True}
        with self.assertRaisesRegex(ValueError,"exact selector recipe fields"):
            entry._prepared(h.ledger,spec)

    def test_missing_available_clearance_is_not_implicitly_unavailable(self):
        h,_=self.make(predicate=lambda *_:False)
        row=h.spec["recipes"][0]
        review=json.loads(Path(row["review_path"]).read_text())
        review["resource_cost_qualification_reviewed"]=False
        Path(row["review_path"]).write_text(json.dumps(review))
        with self.assertRaises(ValueError):h.preflight()

    def test_last_freeze_cannot_attach_candidate_claim_to_unavailable_configuration(self):
        with self.assertRaisesRegex(ValueError,"acquired candidate work"):
            self.productive(late_claim=True)

    def test_last_freeze_cannot_attach_work_to_unavailable_unused_a3_candidate(self):
        with self.assertRaisesRegex(ValueError,"acquired candidate work"):
            self.productive(late_claim=True,unused_a3=True)

    def test_last_freeze_cannot_change_original_unavailability_review_bytes(self):
        with self.assertRaises(ValueError):self.productive(late_review=True)

    def test_durable_stage_report_rejects_unavailability_with_an_observed_work_claim(self):
        _,ledger,unavailable,_,_=self.productive()
        name,identity=unavailable[0]
        evidence_path=ledger.directory/name/(identity+"-root-evidence.json")
        artifact_path=ledger.directory/name/(identity+".json")
        evidence=json.loads(evidence_path.read_text());evidence["observed_zero_work"]=True
        artifact=json.loads(artifact_path.read_text());artifact["evidence_sha256"]=digest(evidence)
        evidence_path.write_text(json.dumps(evidence));artifact_path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError,"observed zero work"):
            ledger.report_stage(name)


if __name__=="__main__":unittest.main()
