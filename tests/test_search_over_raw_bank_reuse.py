"""Multi-bank action ownership using authored states and disposable toy owners.

No scientific roots, production search, registration or holdout access.
"""
import hashlib
import json
import pickle
import unittest
from copy import copy
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_root_plan import OwnedContinuationBank
from pokezero.mcts_eval.search_over_raw_outcome_run import OwningRunValidator,supervise_outcome_run
import test_search_over_raw_owned_stage as stage_fixture
import test_search_over_raw_outcome_run as outcome_fixture
import pokezero.mcts_eval.search_over_raw_outcome_run as outcome_run
import test_search_over_raw_root_plan as plan_fixture
from pokezero.mcts_eval.search_over_raw_raw_run import RawSelectionRunContract,RawSelectionRunValidator
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job


class MissingCellsSyntheticJob(outcome_fixture.SyntheticContinuationJob):
    """Authored incomplete producer: seven real toy captures, no eighth attempt."""
    def __call__(self,ownership,phase):
        original=outcome_run.paired_continuations
        def partial(**kwargs):
            with patch("pokezero.mcts_eval.search_over_raw.range",
                side_effect=lambda *args:range(7) if args == (8,) else range(*args),create=True):
                rows=original(**kwargs)
            rows["status"]="UNCERTAIN"
            return rows
        with patch.object(outcome_run,"paired_continuations",side_effect=partial):
            return super().__call__(ownership,phase)


class BankReuseTests(unittest.TestCase):
    def setUp(self):
        self.helper=stage_fixture.OwnedStageTests();self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)

    def setup_banks(self,*,new_action=1,capped=False,missing=False):
        h=self.helper;f=h.fixture()
        selected,reader=h.selector(f,action=0)
        cfg=SearchConfiguration("incumbent",seconds=3.)
        other,other_reader=h.selector(f,cfg=cfg,action=new_action)
        ledger=h.ledger([(f,selected,reader),(f,other,other_reader)])
        old,data=h.plan(f,selected,reader,outcome=True,
            job_class=MissingCellsSyntheticJob if missing else stage_fixture.CappedSyntheticJob
                if capped else outcome_fixture.SyntheticContinuationJob)
        h.ingest(ledger,{old.root_id:old},{old.root_id:data})
        for unstarted in ledger._stage_configs("A2"):
            if unstarted.identity != h.cfg.identity:
                ledger.record_stage("A2",unstarted.identity,{},accepted_units=0,
                    status="UNSTARTED",evidence_sha256="a"*64)
        ledger.freeze_evaluators()
        current,_=h.plan(f,other,other_reader)
        return ledger,cfg,old,OwnedContinuationBank(old,*data),current

    def prepare(self,current,donors):
        old=donors[0]
        return current.prepare_new_continuation_job(donor_banks=tuple(donors),
            config=old.job.config,runtime=old.job.runtime,
            _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)

    def run_toy(self,plan,job,directory):
        binding=plan.continuation_attempt_binding(job)
        terminal=supervise_outcome_run(job,directory=directory,binding_sha256=binding,
            verify_admission=lambda:None,validate_result=lambda result:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader=OwningRunValidator(directory=directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=binding,
            root_binding=job.root_binding,expected_snapshot=job.snapshot,runtime=job.runtime,
            allow_synthetic=True)
        return OwnedContinuationBank(plan,job,terminal["result"],reader)

    def ingest(self,ledger,cfg,current,banks):
        return self.helper.ingest(ledger,{current.root_id:current},
            {current.root_id:tuple(banks)},cfg=cfg,stage="A3")

    def evidence(self,ledger,cfg):
        return json.loads((ledger.directory/"A3"/(cfg.identity+"-root-evidence.json")).read_text())

    def test_a3_new_action_reuses_exact_a2_raw_owner(self):
        ledger,cfg,old,donor,current=self.setup_banks()
        with patch("subprocess.Popen") as launch:
            job,preparation=self.prepare(current,[donor])
        launch.assert_not_called()
        self.assertEqual(job.actions,{cfg.identity:1})
        self.assertEqual(current._prepared[id(job)]["target_actions"],(1,))
        self.assertEqual(preparation["new_action_values"],[1])
        self.assertFalse(preparation["collection_authorized"])
        self.assertFalse(preparation["donor_roster_exhaustive"])
        new=self.run_toy(current,job,donor.reader.directory.parent/"new-action")
        accepted=current.accept_continuation_result(job,new.result,owning_reader=new.reader)
        self.assertEqual(accepted["root_evidence"]["uncertain_action_cells"],8)
        with patch("subprocess.Popen") as launch:
            self.ingest(ledger,cfg,current,[donor,new])
        launch.assert_not_called()
        evidence=self.evidence(ledger,cfg);root=evidence["root_evidence"][current.root_id]
        self.assertEqual((root["unique_action_cells"],root["uncertain_action_cells"]),(16,0))
        before=self.helper.evidence()["owned_input_handoff"]["shared_cell_provenance"]
        after=evidence["owned_input_handoff"]["shared_cell_provenance"]
        self.assertEqual(len(before),8);self.assertEqual(len(after),16)
        for key,sha in before.items():self.assertEqual(after[key],sha)
        handoff=evidence["owned_input_handoff"]
        self.assertTrue(handoff["new_actions_with_reused_prior_banks_supported"])
        self.assertFalse(handoff["collection_authorized"])
        self.assertEqual(len(ledger.contract["panels"]["exploration"]["root_slots"]),200)

    def test_changed_configuration_same_action_requires_no_job(self):
        ledger,cfg,old,donor,current=self.setup_banks(new_action=0)
        with patch("subprocess.Popen") as launch:
            job,receipt=self.prepare(current,[donor])
            self.ingest(ledger,cfg,current,[donor])
        launch.assert_not_called();self.assertIsNone(job)
        self.assertTrue(receipt["no_job_required"])
        root=self.evidence(ledger,cfg)["root_evidence"][current.root_id]
        self.assertEqual((root["unique_action_cells"],root["uncertain_action_cells"]),(8,0))

    def test_capped_target_reserves_all_cells_without_retry(self):
        ledger,cfg,old,donor,current=self.setup_banks(new_action=0,capped=True)
        with patch("subprocess.Popen") as launch:
            job,receipt=self.prepare(current,[donor])
            self.ingest(ledger,cfg,current,[donor])
        launch.assert_not_called();self.assertIsNone(job)
        self.assertTrue(receipt["missing_cells_not_retried"])
        root=self.evidence(ledger,cfg)["root_evidence"][current.root_id]
        self.assertEqual(root["uncertain_action_cells"],8)

    def test_duplicate_bank_cannot_double_count_or_fill_caps(self):
        ledger,cfg,old,donor,current=self.setup_banks(capped=True)
        with self.assertRaisesRegex(ValueError,"overlap"):
            self.prepare(current,[donor,donor])
        with self.assertRaisesRegex(ValueError,"overlap"):
            self.ingest(ledger,cfg,current,[donor,donor])
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A3"/(cfg.identity+"-root-evidence.json")).exists())

    def test_missing_capture_is_owned_uncertainty_not_new_work(self):
        ledger,cfg,old,donor,current=self.setup_banks(new_action=0,missing=True)
        self.assertEqual(len(donor.result["bank"]["cells"]),7)
        with patch("subprocess.Popen") as launch:
            job,receipt=self.prepare(current,[donor])
            self.ingest(ledger,cfg,current,[donor])
        launch.assert_not_called();self.assertIsNone(job)
        self.assertEqual(receipt["reused_target_action_values"],[0])
        evidence=self.evidence(ledger,cfg)
        self.assertEqual(evidence["root_evidence"][current.root_id]["uncertain_action_cells"],1)
        before=self.helper.evidence()["owned_input_handoff"]["shared_cell_provenance"]
        after=evidence["owned_input_handoff"]["shared_cell_provenance"]
        self.assertEqual(before,after)
        cells=[json.loads((ledger.directory/path).read_text()) for path in after]
        missing=[cell for cell in cells if not cell["supplied"]]
        self.assertEqual(len(missing),1)
        self.assertEqual(missing[0]["replicate"],7)
        self.assertIsNone(missing[0]["capture_sha256"])
        self.assertEqual(missing[0]["directory"],str(donor.reader.directory.resolve()))

    def test_redrawn_identical_donor_cannot_replace_prior_raw_cells(self):
        ledger,cfg,old,donor,current=self.setup_banks(new_action=0)
        redraw=self.run_toy(old,donor.job,donor.reader.directory.parent/"redrawn")
        self.assertEqual(redraw.result["outcomes"],donor.result["outcomes"])
        before=self.helper.evidence()
        with self.assertRaisesRegex(ValueError,"actual shared continuation cell"):
            self.ingest(ledger,cfg,current,[redraw])
        self.assertEqual(self.helper.evidence(),before)

    def test_new_action_preparation_rejects_late_donor_readout_mutation(self):
        ledger,cfg,old,donor,current=self.setup_banks()
        actual=current.prepare_continuation_job
        def mutate(**kwargs):
            result=actual(**kwargs)
            donor.reader.last_validation["late_mutation"]=True
            return result
        with patch.object(current,"prepare_continuation_job",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"readout changed"):
                self.prepare(current,[donor])

    def test_stage_rejects_donor_changed_after_current_selector_callback(self):
        ledger,cfg,old,donor,current=self.setup_banks(new_action=0)
        actual=current._read_selector
        def mutate(configuration,row,binding):
            result=actual(configuration,row,binding)
            if donor.reader.last_validation is not None:
                donor.reader.last_validation["later_callback_mutation"]=True
            return result
        with patch.object(current,"_read_selector",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"readout"):
                self.ingest(ledger,cfg,current,[donor])
        self.assertFalse((ledger.directory/"A3"/(cfg.identity+"-root-evidence.json")).exists())

    def test_subset_requires_nonempty_exact_selected_action_values(self):
        ledger,cfg,old,donor,current=self.setup_banks()
        for targets in ((),[1],(True,),(2,),(1,0),(1,1)):
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError,"selected-action subset"):
                    current.prepare_continuation_job(config=donor.job.config,runtime=donor.job.runtime,
                        actions_to_collect=targets,_synthetic_job_class=outcome_fixture.SyntheticContinuationJob)

    def unbanked_consumers(self):
        h=self.helper;f=h.fixture()
        selected,reader=h.selector(f)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=h.selector(f,cfg=hp)
        ledger=h.ledger([(f,selected,reader),(f,other,other_reader)])
        old,_=h.plan(f,selected,reader)
        h.ingest(ledger,{old.root_id:old},{old.root_id:None})
        return f,ledger,hp,other,other_reader

    def test_no_bank_still_rejects_second_raw_process_with_equal_receipt(self):
        f,ledger,hp,selected,reader=self.unbanked_consumers()
        raw=f.reader;job=plan_fixture.AuthoredRawOwner(f.selected)
        directory=raw.directory.parent/"second-raw-owner"
        redraw=RawSelectionRunValidator(directory=directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=raw.binding,
            configuration=raw.configuration,root_id=raw.root_id,statistics_root_id=raw.statistics_root_id,
            selection_seed=raw.seed,runtime_configuration=raw.runtime,runtime_sha256=raw.runtime_sha,
            consumed_binding=raw.consumed,legal_action_mask=raw.mask,decision_round=raw.round,
            file_hashes=raw.files,checkpoint_path=raw.checkpoint_path,checkpoint_sha256=raw.checkpoint_sha,
            showdown_root=raw.showdown_root,allow_synthetic=True,public_request_sha256=raw.public_request_sha)
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256=raw.binding,
            verify_admission=lambda:None,contract=RawSelectionRunContract(),kind="raw-selection-run",
            work_phase="selection",validate_result=redraw._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(terminal["result"],f.selected)
        f.reader=redraw
        current,_=self.helper.plan(f,selected,reader)
        with self.assertRaisesRegex(ValueError,"actual shared raw owner changed"):
            self.helper.ingest(ledger,{current.root_id:current},{current.root_id:None},cfg=hp)
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A2"/(hp.identity+"-root-evidence.json")).exists())

    def test_absent_bank_allows_first_attempt_without_upgrading_old_uncertainty(self):
        h=self.helper;f=h.fixture()
        selected,reader=h.selector(f,action=1)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=h.selector(f,cfg=hp,action=0)
        ledger=h.ledger([(f,selected,reader),(f,other,other_reader)])
        plan,data=h.plan(f,selected,reader,outcome=True,extras=((other,other_reader),))
        h.ingest(ledger,{plan.root_id:plan},{plan.root_id:None})
        before=h.evidence()
        self.assertEqual(before["root_intervals"][plan.root_id],[-1.,1.])
        self.assertEqual(before["owned_input_handoff"]["attempted_actions_by_root"],{plan.root_id:[]})
        self.assertEqual(list((ledger.directory/"shared-actions").glob("*/cell-*.json")),[])
        h.ingest(ledger,{plan.root_id:plan},{plan.root_id:data},cfg=hp)
        self.assertEqual(h.evidence(),before)
        with patch.object(ledger,"_stage_configs",return_value=(h.cfg,hp)):
            ledger._rows("A2")
        self.assertEqual(len(list((ledger.directory/"shared-actions").glob("*/cell-*.json"))),8)

    def test_unselected_bank_targets_keep_missing_capture_ownership(self):
        h=self.helper;f=h.fixture()
        selected,reader=h.selector(f,action=0)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=h.selector(f,cfg=hp,action=1)
        ledger=h.ledger([(f,selected,reader),(f,other,other_reader)])
        plan,data=h.plan(f,selected,reader,outcome=True,extras=((other,other_reader),),
            job_class=MissingCellsSyntheticJob)
        h.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        handoff=h.evidence()["owned_input_handoff"]
        self.assertEqual(handoff["attempted_actions_by_root"],{plan.root_id:[0]})
        self.assertEqual(len(handoff["shared_cell_provenance"]),16)
        records=[json.loads((ledger.directory/p).read_text()) for p in handoff["shared_cell_provenance"]]
        missing=[r for r in records if r["action"] == 1 and not r["supplied"]]
        self.assertEqual(len(missing),1);self.assertEqual(missing[0]["replicate"],7)
        bank=OwnedContinuationBank(plan,*data)
        with patch("subprocess.Popen") as launch:
            job,receipt=self.prepare(plan,[bank])
            h.ingest(ledger,{plan.root_id:plan},{plan.root_id:(bank,)},cfg=hp)
        launch.assert_not_called();self.assertIsNone(job)
        self.assertTrue(receipt["missing_cells_not_retried"])
        self.assertEqual(h.evidence(hp)["root_evidence"][plan.root_id]["uncertain_action_cells"],2)

    def test_fresh_validator_for_same_actual_raw_run_is_allowed(self):
        f,ledger,hp,selected,reader=self.unbanked_consumers()
        f.reader=copy(f.reader)
        current,_=self.helper.plan(f,selected,reader)
        self.helper.ingest(ledger,{current.root_id:current},{current.root_id:None},cfg=hp)
        before=self.helper.evidence()["owned_input_handoff"]["shared_raw_provenance"]
        after=self.helper.evidence(hp)["owned_input_handoff"]["shared_raw_provenance"]
        self.assertEqual(before,after)
        self.assertEqual(len(before),1)

    def test_raw_provenance_is_checked_again_on_stage_report(self):
        f,ledger,hp,selected,reader=self.unbanked_consumers()
        current,_=self.helper.plan(f,selected,reader)
        self.helper.ingest(ledger,{current.root_id:current},{current.root_id:None},cfg=hp)
        for cfg in ledger._stage_configs("A2"):
            if cfg.arm == "reference":
                ledger.record_stage("A2",cfg.identity,{},accepted_units=0,status="UNSTARTED",evidence_sha256="a"*64)
        provenance=self.helper.evidence()["owned_input_handoff"]["shared_raw_provenance"]
        path=ledger.directory/next(iter(provenance))
        receipt=json.loads(path.read_text());receipt["directory"]+="-substitution"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,"actual shared raw owner provenance changed"):
            ledger.report_stage("A2")


if __name__ == "__main__":unittest.main()
