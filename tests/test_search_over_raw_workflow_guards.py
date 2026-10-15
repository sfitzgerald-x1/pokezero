"""Mocked zero-work receipt shapes isolate final callback guards, no science."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
import pokezero.mcts_eval.search_over_raw_exploration_workflow as workflow
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
    FixedWorkRecipe,diagnostic_configurations,diagnostic_manifest,diagnostic_review_template)
import test_search_over_raw_continuation_admission as admission


class UnusedHelpers:
    synthetic_process_fixture=True
    @staticmethod
    def extract(*args):raise AssertionError("zero-work fixture invoked native helper")


class CompletedWorkflowInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.h=admission.FreshLifecycleTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.root=self.h.base.root.resolve()
        self.external=self.root/"external-declaration.txt";self.external.write_text("original")
        self.files={str(self.external):hashlib.sha256(self.external.read_bytes()).hexdigest()}
        self.checkpoint=SimpleNamespace(checkpoint_path="/authored/no-loaded-model",checkpoint_sha256="a"*64,
            showdown_root="/authored/no-simulator",showdown_source_sha256="a"*64)
        self.recipes={c.identity:FixedWorkRecipe(unqualified_reason="authored unavailable") for c in diagnostic_configurations()}
        self.chain=workflow.NativeChainRecipe(unqualified_reason="original unavailable chain")
        manifest=diagnostic_manifest(self.h.contract,self.recipes,checkpoint_contract=self.checkpoint,
            showdown_root=self.checkpoint.showdown_root,file_hashes=self.files)
        self.review=self.root/"diagnostic-review.json"
        review=diagnostic_review_template(manifest)
        review.update(status="CLEAR_FIXED_WORK_DIAGNOSTIC_ROSTER",independent_review_completed=True,
            reviewed_before_first_source_work=True)
        self.review.write_text(json.dumps(review))
        original=admission.controller_input_hashes
        with patch.object(admission,"controller_input_hashes",return_value={**original(),**self.files,
                str(self.review):hashlib.sha256(self.review.read_bytes()).hexdigest()}):self.h.initialize()
        for seed in self.h.contract["panels"]["exploration"]["seeds"]:
            path=self.h.directory/"source-attempts"/str(seed);path.mkdir(parents=True)
            (path/"source.json").write_text(json.dumps(dict(roots=[])))
        rows=[dict(root_id=s["root"]["root_id"],source_seed=s["root"]["source_seed"],
            configuration_sha256=s["configuration_sha256"],status="UNSTARTED",attempt_directory=None,
            accepted_work=None,selected_sha256=None,cost=None) for s in manifest["slots"]]
        population=dict(manifest_sha256=digest(manifest),required_slots=len(rows),slots=rows)
        (self.h.directory/"fixed-work-diagnostics.json").write_text(json.dumps(population))
        native=next(c for c in diagnostic_configurations() if c.arm=="incumbent" and c.leaf=="model")
        chains=[dict(root_id=r["root_id"],source_seed=r["source_seed"],native_status="UNSTARTED",status="UNSTARTED",diagnostic=None)
            for r in rows if r["configuration_sha256"]==native.identity]
        self.report=dict(schema="pokezero.search-over-raw.exploration-workflow.v1",diagnostic_population=population,
            native_chains=chains,staged={},**{k:False for k in ("diagnostic_results_used_for_selection","private_archive_returned",
                "runtime_qualified","resource_cost_complete","scientific_measurement_validated","collection_authorized",
                "holdout_authorized","phase_b_authorized")})
        (self.h.directory/"exploration-workflow.json").write_text(json.dumps(self.report))

    def read(self):
        return workflow.read_recorded_workflow(self.h.ledger,self.report,diagnostic_recipes=self.recipes,
            native_chain_recipe=self.chain,diagnostic_review_path=str(self.review),diagnostic_file_hashes=self.files,
            checkpoint_contract=self.checkpoint,auditor_directory=self.root/"authored-auditor",auditor_pid=123,
            allow_synthetic=True,_synthetic_helpers=UnusedHelpers())

    def test_zero_work_retains_all_uncertain_slots_without_launch(self):
        with patch("subprocess.Popen") as launch:result=self.read()
        launch.assert_not_called()
        self.assertEqual((result["fixed_slots"],result["native_chain_rows"],result["completed_fixed_jobs"]),(192,32,0))

    def test_recipe_declaration_changed_by_last_callback_is_rejected(self):
        actual=workflow.preflight_diagnostic_workflow
        def changed(*args,**kwargs):
            manifest=actual(*args,**kwargs)
            object.__setattr__(self.chain,"unqualified_reason","changed after preflight")
            return manifest
        with patch.object(workflow,"preflight_diagnostic_workflow",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"external workflow inputs") :self.read()

    def test_external_file_changed_by_last_callback_is_rejected(self):
        actual=workflow.preflight_diagnostic_workflow
        def changed(*args,**kwargs):
            manifest=actual(*args,**kwargs);self.external.write_text("changed after preflight")
            return manifest
        with patch.object(workflow,"preflight_diagnostic_workflow",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"external workflow inputs") :self.read()
