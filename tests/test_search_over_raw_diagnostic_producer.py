"""Prospective rosters and actual authored owned panels; no scientific work."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_supervision import FixedWorkAdapterJob, supervise_fixed_work
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
    FixedWorkRecipe, diagnostic_configurations, diagnostic_manifest, diagnostic_review_template,
)
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
from test_search_over_raw_reference_public import FullFixedReferenceOwner
import test_search_over_raw_reconciled as ledger_fixture
from test_search_over_raw_productive_panel import ProductivePanelAuditor


@dataclass(frozen=True,kw_only=True)
class AuthoredPanelFixedReferenceJob(FixedWorkAdapterJob):
    authored_label_config: object=None  # Explicit synthetic-extra field; no base-envelope override.
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        request=PublicRootRequest(state=self.context.public_materialization_state,observation=self.context.observation,
            set_source_hash=self._set_source_hash,pending_transition=self.pending_transition)
        try:
            result=FullFixedReferenceOwner(self.configuration,request,self.statistics_root_id,self.selection_seed,
                self.owned_reference_value_recipe,self.expected_runtime_sha256,self.authored_label_config)(ownership,phase)
        except BaseException as error:
            import traceback
            from pokezero.mcts_eval.search_over_raw_ledger import _write_new
            _write_new(Path(ownership["directory"]).parent/"authored-reference-error.json",
                dict(error=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            raise
        return result


class DiagnosticProducerTests(unittest.TestCase):
    def setUp(self):
        helper=ledger_fixture.ReconciledLedgerTest();helper.setUp();self.addCleanup(helper.tearDown)
        self.contract=deepcopy(helper.contract)
        self.checkpoint=SimpleNamespace(checkpoint_path="/authored/no-model",checkpoint_sha256="a"*64)
        self.recipes={c.identity:FixedWorkRecipe(unqualified_reason="authored unavailable recipe; never attempted")
            for c in diagnostic_configurations()}

    def manifest(self):
        return diagnostic_manifest(self.contract,self.recipes,checkpoint_contract=self.checkpoint,
            showdown_root="/authored/no-simulator",file_hashes={})

    def test_all_six_recipes_and_original_slot_zero_roster_without_all_pass_gate(self):
        before=deepcopy(self.contract)
        with patch("subprocess.Popen") as spawn:
            manifest=self.manifest()
        spawn.assert_not_called()
        self.assertEqual(self.contract,before)
        self.assertEqual(len(manifest["recipes"]),6)
        self.assertEqual(len(manifest["slots"]),192)
        self.assertTrue(all(s["root"]["root_slot"]==0 for s in manifest["slots"]))
        self.assertEqual(len({s["root"]["source_seed"] for s in manifest["slots"]}),32)
        self.assertFalse(manifest["all_recipes_must_qualify"])
        self.assertIsNone(manifest["decision_ceiling"])
        self.assertFalse(manifest["diagnostic_results_used_for_selection"])
        self.assertEqual(diagnostic_review_template(manifest)["status"],"REVIEW_REQUIRED")

    def test_no_omitted_recipe_or_unqualified_executable_relabel(self):
        key=next(iter(self.recipes));original=self.recipes.pop(key)
        with self.assertRaises(ValueError):self.manifest()
        self.recipes[key]=replace(original,runtime={})
        with self.assertRaises(ValueError):self.manifest()

    def test_review_pinning_is_acyclic_and_scientific_design_changes_still_bind(self):
        before=self.manifest()
        self.contract["input_hashes"]["/authored/prospective-diagnostic-review"]="b"*64
        self.assertEqual(digest(before),digest(self.manifest()))
        self.contract["namespace"]+="-changed"
        self.assertNotEqual(digest(before),digest(self.manifest()))

    def test_actual_owned_panel_two_reference_jobs_all_missing_slots_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            value=supervise_fixed_work(ProductivePanelAuditor(available_sources=2,fixed_diagnostics=True),
                directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            self.assertEqual(value["supervisor_exit_code"],0,value)
            result=value["result"]
            self.assertEqual(result["required_slots"],192)
            self.assertEqual(result["status_counts"],dict(DIAGNOSTIC_COMPLETE=2,SKIPPED_UNQUALIFIED_RECIPE=10,UNSTARTED=180))
            self.assertEqual(result["actual_accepted_work"],160)
            self.assertEqual(result["owners_validated_while_archives_live"],2)
            self.assertTrue(result["deployment_selection_untouched"])
            self.assertEqual(result["owner_mutation_checks"],4)
            self.assertFalse(result["terminal_truth_validated"])
            self.assertFalse(result["runtime_qualified"])

    def test_owned_label_recipe_enters_actual_panel_but_empty_events_do_not_become_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            value=supervise_fixed_work(ProductivePanelAuditor(fixed_diagnostics=True,reference_values=True),
                directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(value["supervisor_exit_code"],0,error.read_text() if error.exists() else value)
            result=value["result"]
            self.assertEqual(result["status_counts"],dict(DIAGNOSTIC_COMPLETE=1,SKIPPED_UNQUALIFIED_RECIPE=5,UNSTARTED=186))
            self.assertEqual(result["reference_value_summaries"],1)
            self.assertEqual(result["reference_value_labels"],0)
            self.assertEqual(result["reference_terminal_label_populations"],0)
            self.assertEqual(result["owners_validated_while_archives_live"],1)
            self.assertTrue(result["deployment_selection_untouched"])
