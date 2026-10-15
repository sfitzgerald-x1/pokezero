"""No-load prospective laws; authored review flags are not qualification."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_ancestry import NativeAncestryContract
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
    FixedWorkRecipe,diagnostic_configurations,diagnostic_manifest,diagnostic_review_template,fixed_work_runtime_recipe)
from pokezero.mcts_eval.search_over_raw_exploration_workflow import NativeChainRecipe
from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputFixedWorkAdapter,_state_payload
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
import pokezero.mcts_eval.search_over_raw_exploration_entry as entry
import test_search_over_raw_exploration_entry as fixture


class DiagnosticWorkflowPreflightTests(unittest.TestCase):
    def prepare(self,mutation=None,*,capture_actor_history=None,reviewed_capture=None,
            after_population=False,after_unqualified=False,after_config_mutation=False,after_review_mutation=False):
        h=fixture.ExplorationEntryTests();h.setUp();self.addCleanup(h.doCleanups)
        pinned={}
        def alter(spec,contract):
            spec["schema"]=entry.WORKFLOW_SCHEMA
            checkpoint=entry._decode(spec["checkpoint_contract"])
            if capture_actor_history is not None:
                spec["schema"]=entry.ACTOR_WORKFLOW_SCHEMA
                spec["capture_actor_history"]=capture_actor_history
                reviewed=capture_actor_history if reviewed_capture is None else reviewed_capture
                path=Path(spec["source_review_path"])
                previous=json.loads(path.read_bytes())
                from pokezero.mcts_eval.search_over_raw_source_run import source_recipe,source_review_template
                review=source_review_template(contract,source_recipe(entry._decode(spec["source_config"]),
                    checkpoint,spec["source_runtime"],capture_actor_history=reviewed),
                    qualification_evidence_hashes=previous["qualification_evidence_hashes"])
                review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
                    source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
                path.write_text(json.dumps(review))
            factories={row["configuration"]:entry._decode(row["reference_factory"]) for row in spec["recipes"]}
            recipes={}
            for cfg in diagnostic_configurations():
                factory=factories.get(cfg.identity) if cfg.arm=="reference" else None
                ancestry=NativeAncestryContract(17,capture_native_edges=True) if cfg.arm=="incumbent" and cfg.leaf=="model" else None
                if cfg.arm=="reference" and factory is None:
                    # Raw-rollout is a diagnostic, not a wall-clock selector.
                    factory=next(v for v in factories.values() if v is not None)
                runtime=fixed_work_runtime_recipe(cfg,checkpoint,reference_factory=factory,native_ancestry_contract=ancestry)
                if mutation and mutation[0]==cfg.arm and mutation[1]==cfg.leaf:mutation[2](runtime)
                recipes[cfg.identity]=FixedWorkRecipe(runtime=runtime,reference_factory=factory,native_ancestry_contract=ancestry)
            manifest=diagnostic_manifest(contract,recipes,checkpoint_contract=checkpoint,
                showdown_root=checkpoint.showdown_root,file_hashes=spec["selector_files"])
            path=h.root/"fixed-diagnostic-review.json"
            review=diagnostic_review_template(manifest)
            review.update(status="CLEAR_FIXED_WORK_DIAGNOSTIC_ROSTER",independent_review_completed=True,
                reviewed_before_first_source_work=True)
            path.write_text(json.dumps(review));pinned[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            spec["diagnostics"]=dict(recipes=[dict(configuration=k,recipe=_state_payload(v)) for k,v in recipes.items()],
                native_chain_recipe=_state_payload(NativeChainRecipe(unqualified_reason="authored missing constructor runtime")),
                review_path=str(path))
            if after_population:
                from pokezero.mcts_eval.search_over_raw_after_population import NativeAfterRecipe,after_population_review_template
                spec["schema"]=entry.AFTER_WORKFLOW_SCHEMA
                spec["capture_actor_history"]=False if capture_actor_history is None else capture_actor_history
                if after_unqualified:after=NativeAfterRecipe(unqualified_reason="authored not qualified at this ceiling")
                else:
                    config=entry._decode(spec["source_config"])
                    if after_config_mutation:config=replace(config,read_timeout_seconds=config.read_timeout_seconds+1.)
                    from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
                    runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path)
                    review=after_population_review_template(contract,config=config,runtime=runtime)
                    review.update(status="CLEAR_NULL_ONLY_AFTER_POPULATION_RECIPE",independent_review_completed=True,
                        reviewed_before_first_source_work=True)
                    if after_review_mutation:review["law"]["null_only"]=False
                    after_path=h.root/"after-population-review.json"
                    after_path.write_text(json.dumps(review));pinned[str(after_path)]=hashlib.sha256(after_path.read_bytes()).hexdigest()
                    after=NativeAfterRecipe(config=config,runtime=runtime,review_path=str(after_path))
                spec["diagnostics"]["native_after_recipe"]=_state_payload(after)
            self.recipes,self.checkpoint=recipes,checkpoint
        with patch.object(fixture,"controller_input_hashes",side_effect=lambda:{**controller_input_hashes(),**pinned}):h.make(alter=alter)
        return h

    def test_v4_prospective_ready_and_unqualified_recipes_preflight_without_work(self):
        for unqualified in (False,True):
            h=self.prepare(capture_actor_history=False,after_population=True,after_unqualified=unqualified)
            before=entry._evidence_inventory(h.directory)
            with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
                _,prepared=h.preflight()
            launch.assert_not_called();catalog.assert_not_called()
            self.assertEqual(prepared["native_after_recipe"].unqualified_reason is not None,unqualified)
            self.assertEqual(before,entry._evidence_inventory(h.directory))

    def test_v4_source_encoder_or_review_law_drift_refuses_before_work(self):
        for options,message in ((dict(after_config_mutation=True),"original source configuration"),
                (dict(after_review_mutation=True),"prospective null-only AFTER review")):
            h=self.prepare(capture_actor_history=False,after_population=True,**options)
            h.refuse_without_write(message)

    def test_old_workflow_cannot_gain_after_option_and_v4_cannot_omit_it(self):
        h=self.prepare(capture_actor_history=False,after_population=True)
        spec=deepcopy(h.spec)
        del spec["diagnostics"]["native_after_recipe"]
        with self.assertRaisesRegex(ValueError,"exact prospective diagnostic registry"):entry._prepared(h.ledger,spec)
        for schema in (entry.WORKFLOW_SCHEMA,entry.ACTOR_WORKFLOW_SCHEMA):
            spec=deepcopy(h.spec);spec["schema"]=schema
            if schema==entry.WORKFLOW_SCHEMA:del spec["capture_actor_history"]
            with self.assertRaises(ValueError):entry._prepared(h.ledger,spec)

    def test_v2_complete_recipe_registry_preflights_without_launch_and_v1_stays_distinct(self):
        h=self.prepare()
        with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
            job,prepared=h.preflight()
        launch.assert_not_called();catalog.assert_not_called()
        self.assertEqual(len(prepared["diagnostic_recipes"]),6)
        self.assertIsNotNone(prepared["native_chain_recipe"].unqualified_reason)
        self.assertNotEqual(entry.SCHEMA,entry.WORKFLOW_SCHEMA)
        self.assertFalse(prepared["capture_actor_history"])

    def test_v3_capture_option_is_reviewed_without_loading_or_launching(self):
        for capture in (False,True):
            h=self.prepare(capture_actor_history=capture)
            before=entry._evidence_inventory(h.directory)
            with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
                _,prepared=h.preflight()
            launch.assert_not_called();catalog.assert_not_called()
            self.assertIs(prepared["capture_actor_history"],capture)
            self.assertEqual(len(prepared["diagnostic_recipes"]),6)
            self.assertEqual(before,entry._evidence_inventory(h.directory))

    def test_capture_requires_the_matching_source_review_in_both_directions(self):
        for capture in (False,True):
            h=self.prepare(capture_actor_history=capture,reviewed_capture=not capture)
            h.refuse_without_write("exact source recipe/code/qualification review required")

    def test_only_the_prepared_capture_flag_reaches_the_panel_collector(self):
        h=self.prepare(capture_actor_history=True);_,prepared=h.preflight()
        with patch.object(entry,"collect_owned_exploration_panel",return_value="authored panel") as collect:
            self.assertEqual(entry._collect_entry_panel(h.ledger,h.spec,prepared),"authored panel")
        self.assertTrue(collect.call_args.kwargs["capture_actor_history"])
        changed=deepcopy(h.spec);changed["capture_actor_history"]=False
        with patch.object(entry,"collect_owned_exploration_panel") as collect, \
                self.assertRaisesRegex(ValueError,"differs from prepared"):
            entry._collect_entry_panel(h.ledger,changed,prepared)
        collect.assert_not_called()

    def test_capture_boolean_cannot_be_inferred_or_added_to_historical_entries(self):
        for value in (None,0,1,"true",[],{}):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"capture boolean"):
                entry._capture_actor_history(dict(schema=entry.ACTOR_WORKFLOW_SCHEMA,capture_actor_history=value))
        with self.assertRaisesRegex(ValueError,"capture boolean"):
            entry._capture_actor_history(dict(schema=entry.ACTOR_WORKFLOW_SCHEMA))
        for schema in (entry.SCHEMA,entry.WORKFLOW_SCHEMA):
            self.assertFalse(entry._capture_actor_history(dict(schema=schema)))
            with self.assertRaisesRegex(ValueError,"historical entry"):
                entry._capture_actor_history(dict(schema=schema,capture_actor_history=False))

    def test_v3_still_requires_the_complete_diagnostic_registry(self):
        h=self.prepare(capture_actor_history=True)
        del h.spec["diagnostics"]
        with self.assertRaisesRegex(ValueError,"exact data-only"):
            entry._prepared(h.ledger,h.spec)

    def test_internally_reviewed_but_wrong_runtime_law_refuses_before_source(self):
        changes=[("incumbent","model",lambda r:r["incumbent"].update(native_batch_guard_ms=False)),
            ("incumbent","raw_rollout",lambda r:r["incumbent_leaf"].update(deadline="wall-clock")),
            ("reference","model",lambda r:r.update(initial_dispatch_workers=20)),
            ("reference","hp_fraction",lambda r:r["reference_leaf"].update(rollout_cap=249))]
        for mutation in changes:
            with self.subTest(arm=mutation[0],leaf=mutation[1]):
                h=self.prepare(mutation)
                h.refuse_without_write("complete fixed-work diagnostic runtime recipe drift")

    def test_no_load_recipe_matches_actual_adapter_assembly_for_all_six_lanes(self):
        h=self.prepare()
        # Actual adapter assembly executes, but model loading/pool construction
        # are replaced. This checks recipe parity, not loaded runtime behavior.
        for cfg in diagnostic_configurations():
            recipe=self.recipes[cfg.identity]
            with (self.subTest(arm=cfg.arm,leaf=cfg.leaf),patch("subprocess.Popen") as launch,
                    patch.object(WorldInputFixedWorkAdapter,"_make_incumbent_runtime",return_value=(None,None,None)),
                    patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch",return_value=SimpleNamespace(close=lambda:None))):
                adapter=WorldInputFixedWorkAdapter(cfg,checkpoint_contract=self.checkpoint,
                    showdown_root=self.checkpoint.showdown_root,reference_factory=recipe.reference_factory,
                    fixed_work=FixedWorkDiagnosticContract(),native_ancestry_contract=recipe.native_ancestry_contract,
                    expected_public_request_sha256="a"*64,
                    **({"owned_process_receipts":{}} if cfg.arm=="reference" else {}))
                self.assertEqual(digest(adapter.runtime_configuration),digest(recipe.runtime))
                adapter.close()
                launch.assert_not_called()
