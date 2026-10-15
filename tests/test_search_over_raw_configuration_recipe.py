"""Prospective ordering with disposable authored owners, never scientific work."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_reconciled import (
    configuration_review_template, selector_runtime_recipe, controller_input_hashes,
)
from pokezero.mcts_eval.search_over_raw_root_plan import _OwnedStageInputs
from pokezero.mcts_eval.search_over_raw_world_inputs import TIMED_NATIVE_PUBLIC_CAPTURE
import test_search_over_raw_continuation_admission as fresh_fixture
import test_search_over_raw_owned_stage as stage_fixture
import test_search_over_raw_timed_run as timed_fixture
import test_search_over_raw_oracle_stage as oracle_fixture
import test_search_over_raw_native_oracle_run as native_oracle_fixture
import test_search_over_raw_reconciled as ledger_fixture


class ConfigurationRecipeTests(unittest.TestCase):
    def setUp(self):
        self.h=fresh_fixture.FreshLifecycleTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.s=stage_fixture.OwnedStageTests();self.s.setUp();self.addCleanup(self.s.doCleanups)
        self.cfg=SearchConfiguration("incumbent",seconds=1.)
        self.stage="A2"
        self.checkpoint=hashlib.sha256(b"authored unused model").hexdigest()
        self.evidence=self.h.directory.parent/"authored-qualification-evidence.json"
        self.evidence.write_text(json.dumps(dict(synthetic=True,production_qualification=False)))
        self.review_path=self.h.directory.parent/"configuration-review.json"

    def recipe(self,cfg=None):
        cfg=cfg or self.cfg
        runtime=(native_oracle_fixture.observed_runtime(cfg,self.checkpoint) if cfg.belief == "oracle"
            else timed_fixture.runtime(cfg,self.checkpoint))
        runtime["set_source_hash"]="a"*64
        runtime["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        return runtime

    def initialize(self,mutate=None):
        # No actual root, archive, truth, selection or owner exists at review.
        self.assertFalse(hasattr(self.h,"fixture"));self.assertFalse(self.h.directory.exists())
        evidence_hashes={str(self.evidence):hashlib.sha256(self.evidence.read_bytes()).hexdigest()}
        self.review=configuration_review_template(self.h.contract,stage=self.stage,configuration=self.cfg.identity,
            raw_runtime_sha256=self.h.raw_sha,selector_recipe_sha256=digest(self.recipe()),
            qualification_evidence_hashes=evidence_hashes,
            cost_scope="Authored toy CPU/wall/cleanup fixture only; no production or complete memory-cost qualification.")
        self.review.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
            atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
        if mutate:mutate(self.review)
        self.review_path.write_text(json.dumps(self.review))
        extra={str(self.review_path):hashlib.sha256(self.review_path.read_bytes()).hexdigest(),**evidence_hashes}
        # Extend only fixture ledger inputs, not the controller code recipe.
        with patch.object(fresh_fixture,"controller_input_hashes",return_value={**controller_input_hashes(),**extra}):
            self.h.initialize()

    def observed(self,zero=False,action=1):
        self.h.observed_plan();f=self.h.fixture
        selected,reader=self.s.selector(f,cfg=self.cfg,zero=zero,action=action)
        self.plan=f.plan(configurations=(f.cfg,self.cfg),
            selections={"raw":f.selected,self.cfg.identity:selected},
            selector_readers={"raw":f.reader,self.cfg.identity:reader})
        return self.plan

    def ingest(self,plan=None):
        plan=plan or self.plan
        with patch("subprocess.Popen") as launch:
            row=self.h.ledger.record_owned_stage_evidence(self.stage,self.cfg.identity,
                plans={plan.root_id:plan},continuations={plan.root_id:None},qualification_review_path=self.review_path)
        launch.assert_not_called();return row

    def test_genuine_prospective_public_recipe_keeps_full_missing_roster(self):
        self.initialize();contract_sha=self.h.ledger.contract_sha256;self.observed()
        self.assertEqual(digest(self.plan.contract),contract_sha)
        self.assertNotIn("public_bindings_sha256",self.review)
        self.assertNotIn("runtime_by_root_sha256",self.review)
        row=self.ingest()
        self.assertEqual(row["summary"]["lower_mean"],-1.)
        self.assertEqual(len(self.h.ledger.contract["panels"]["exploration"]["root_slots"]),200)
        self.assertFalse(self.h.ledger.contract["collection_authorized"])
        evidence=json.loads((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).read_text())
        self.assertEqual(evidence["root_intervals"][self.plan.root_id],[-1.,1.])
        self.assertFalse(evidence["runtime_qualified_by_this_code"])

    def test_same_action_is_exact_zero_even_unknown_outcome(self):
        self.initialize();self.observed(action=0);self.ingest()
        evidence=json.loads((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).read_text())
        self.assertEqual(evidence["root_intervals"][self.plan.root_id],[0.,0.])

    def test_same_prospective_review_classifies_observed_zero_work_locally(self):
        self.initialize();self.observed(zero=True);row=self.ingest()
        self.assertEqual(row["summary"]["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(row["root_intervals"],{})
        self.assertFalse((self.h.directory/"operational-stop.json").exists())

    def test_generic_callbacks_and_wrapped_owned_callbacks_cannot_adopt_v2(self):
        self.initialize();self.observed()
        inputs=_OwnedStageInputs(self.h.ledger,self.stage,self.cfg.identity,
            {self.plan.root_id:self.plan},{self.plan.root_id:None})
        for owned,callback in ((None,inputs.validate_selector),(inputs,lambda *a:inputs.validate_selector(*a))):
            with self.assertRaisesRegex(ValueError,"exact owned ingestion"):
                self.h.ledger.record_stage_evidence(self.stage,self.cfg.identity,inputs.bundles,
                    public_bindings=inputs.public_bindings,runtime_hashes=inputs.runtime_hashes,
                    qualification_review_path=self.review_path,validate_selector=callback,
                    validate_outcome=inputs.validate_outcome,_owned_stage_inputs=owned)
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-evidence-claim.json")).exists())

    def test_observed_runtime_recipe_drift_stops_claimed_stage(self):
        self.initialize(lambda r:r.update(selector_recipe_sha256="f"*64));self.observed()
        with self.assertRaisesRegex(ValueError,"observed owner runtime"):
            self.ingest()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        self.assertTrue((self.h.directory/self.stage/(self.cfg.identity+"-evidence-claim.json")).exists())

    def test_qualification_evidence_drift_rejected(self):
        self.initialize();self.observed();self.evidence.write_text("changed")
        with self.assertRaises(ValueError):self.ingest()
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_boolean_authority_alias_and_future_root_fields_rejected(self):
        self.initialize(lambda r:r.update(collection_authorized=0,public_bindings_sha256="a"*64));self.observed()
        with self.assertRaisesRegex(ValueError,"exact prospective"):self.ingest()

    def test_unreviewed_cost_scope_does_not_clear(self):
        self.initialize(lambda r:r.update(resource_cost_qualification_reviewed=False));self.observed()
        with self.assertRaisesRegex(ValueError,"exact prospective"):self.ingest()

    def test_future_a3_candidate_review_does_not_override_a2_freeze(self):
        self.stage="A3";self.cfg=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=3.)
        self.initialize();self.observed()
        with self.assertRaises((ValueError,FileNotFoundError)):self.ingest()
        self.assertFalse((self.h.directory/"A3"/(self.cfg.identity+"-evidence-claim.json")).exists())

    def test_oracle_normalization_removes_only_root_truth_fields(self):
        cfg=SearchConfiguration("reference",belief="oracle",seconds=1.,workers=20)
        runtime=dict(configuration=asdict(cfg),team_oracle={"truth":"private"},
            oracle_payload_sha256="a"*64,oracle_reconstruction_catalog_sha256="b"*64,allocation={"workers":20})
        expected={k:v for k,v in runtime.items() if k not in {"team_oracle","oracle_payload_sha256"}}
        self.assertEqual(digest(selector_runtime_recipe(cfg,runtime)),digest(expected))
        other=deepcopy(runtime);other["oracle_reconstruction_catalog_sha256"]="c"*64
        self.assertNotEqual(digest(selector_runtime_recipe(cfg,other)),digest(expected))

    def observed_oracle(self,seed=0):
        helper=oracle_fixture.OracleStageTests();self.addCleanup(helper.doCleanups)
        original=ledger_fixture.ReconciledLedgerTest.setUp
        def static_recipe(base):
            original(base);base.contract=deepcopy(self.h.contract)
        with patch.object(ledger_fixture.ReconciledLedgerTest,"setUp",static_recipe):
            f=helper.fixture(seed)
        selected,reader=helper.selector(f)
        return f.plan(configurations=(f.cfg,self.cfg),
            selections={"raw":f.selected,self.cfg.identity:selected},
            selector_readers={"raw":f.reader,self.cfg.identity:reader},oracle_set_source=f.truth.source)
    def test_genuine_original_team_recipe_review_precedes_private_archive(self):
        self.stage="A1";self.cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.)
        self.initialize();contract_sha=self.h.ledger.contract_sha256
        self.plan=self.observed_oracle()
        self.assertEqual(digest(self.plan.contract),contract_sha)
        self.ingest()
        evidence=json.loads((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).read_text())
        self.assertTrue(evidence["owned_input_handoff"]["original_team_archive_correspondence_validated"])
        self.assertEqual(set(evidence["selector_runtime_by_root"]),{self.plan.root_id})

    def test_two_future_original_teams_vary_truth_not_recipe(self):
        self.stage="A1";self.cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.)
        self.initialize();first=self.observed_oracle();second=self.observed_oracle(1)
        self.assertNotEqual(first.readers[self.cfg.identity].runtime_sha,second.readers[self.cfg.identity].runtime_sha)
        plans={p.root_id:p for p in (first,second)}
        self.h.ledger.record_owned_stage_evidence(self.stage,self.cfg.identity,
            plans=plans,continuations={r:None for r in plans},qualification_review_path=self.review_path)
        evidence=json.loads((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).read_text())
        self.assertEqual(len(evidence["selector_runtime_by_root"]),2)

    def test_oracle_catalog_substitution_is_not_excused_by_truth_normalization(self):
        self.stage="A1";self.cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.)
        self.initialize();self.plan=self.observed_oracle()
        self.plan._oracle_source.move_metadata["tackle"]={"authored":"drift"}
        with self.assertRaises(ValueError):self.ingest()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_bad_owned_preparation_is_durably_stopped_not_retryable(self):
        self.initialize();self.observed()
        with self.assertRaisesRegex(ValueError,"exact supplied"):
            self.h.ledger.record_owned_stage_evidence(self.stage,self.cfg.identity,
                plans={},continuations={},qualification_review_path=self.review_path)
        self.assertTrue((self.h.directory/self.stage/(self.cfg.identity+"-owned-preparation-claim.json")).exists())
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        with self.assertRaisesRegex(ValueError,"integrity failure"):self.ingest()

    def test_controller_code_is_not_independent_qualification_evidence(self):
        with self.assertRaisesRegex(ValueError,"separate from controller"):
            configuration_review_template(self.h.contract,stage=self.stage,configuration=self.cfg.identity,
                raw_runtime_sha256=self.h.raw_sha,selector_recipe_sha256=digest(self.recipe()),
                qualification_evidence_hashes=controller_input_hashes(),cost_scope="not measured")

    def test_final_owner_callback_cannot_rewrite_prospective_review(self):
        self.initialize();self.observed();original=_OwnedStageInputs.finalize
        def late(inputs,rows):
            result=original(inputs,rows);self.review_path.write_text(self.review_path.read_text()+" ");return result
        with patch.object(_OwnedStageInputs,"finalize",late),self.assertRaises(ValueError):self.ingest()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_exact_private_inputs_cannot_adopt_a_stranded_preparation_claim(self):
        self.initialize();self.observed()
        inputs=_OwnedStageInputs(self.h.ledger,self.stage,self.cfg.identity,
            {self.plan.root_id:self.plan},{self.plan.root_id:None})
        # Even the exact owning coordinator is insufficient outside its live
        # ingestion call; an old durable claim is not a reusable capability.
        path=self.h.directory/self.stage/(self.cfg.identity+"-owned-preparation-claim.json")
        path.write_text(json.dumps(dict(stage=self.stage,configuration=self.cfg.identity,
            contract_sha256=self.h.ledger.contract_sha256,
            qualification_review_sha256=hashlib.sha256(self.review_path.read_bytes()).hexdigest(),retry_authorized=False)))
        with self.assertRaisesRegex(ValueError,"current owned preparation handoff"):
            self.h.ledger.record_stage_evidence(self.stage,self.cfg.identity,inputs.bundles,
                public_bindings=inputs.public_bindings,runtime_hashes=inputs.runtime_hashes,
                qualification_review_path=self.review_path,validate_selector=inputs.validate_selector,
                validate_outcome=inputs.validate_outcome,_owned_stage_inputs=inputs)
        with self.assertRaises(FileExistsError):self.ingest()
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_final_owner_callback_cannot_delete_preparation_claim(self):
        self.initialize();self.observed();original=_OwnedStageInputs.finalize
        def late(inputs,rows):
            result=original(inputs,rows)
            (self.h.directory/self.stage/(self.cfg.identity+"-owned-preparation-claim.json")).unlink()
            return result
        with patch.object(_OwnedStageInputs,"finalize",late),self.assertRaises(FileNotFoundError):self.ingest()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_final_owner_callback_cannot_mutate_preparation_claim(self):
        self.initialize();self.observed();original=_OwnedStageInputs.finalize
        def late(inputs,rows):
            result=original(inputs,rows)
            path=self.h.directory/self.stage/(self.cfg.identity+"-owned-preparation-claim.json")
            path.write_text(path.read_text()+" ");return result
        with patch.object(_OwnedStageInputs,"finalize",late),self.assertRaisesRegex(ValueError,"preparation claim"):
            self.ingest()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        self.assertFalse((self.h.directory/self.stage/(self.cfg.identity+"-root-evidence.json")).exists())


if __name__ == "__main__":unittest.main()
