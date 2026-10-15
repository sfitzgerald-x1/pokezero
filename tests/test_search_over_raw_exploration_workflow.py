"""Authored full-panel integration; no scientific collection or runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from test_search_over_raw_productive_panel import ProductivePanelAuditor


class FullDiagnosticWorkflowTests(unittest.TestCase):
    def test_reference_label_recipe_full_workflow_readback_and_completed_job_swap_refusal(self):
        self._reference_label_workflow(nonempty=False)

    def test_nonempty_reference_labels_full_collector_readback_and_tamper_refusal(self):
        self._reference_label_workflow(nonempty=True)

    def _reference_label_workflow(self,*,nonempty):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            terminal=supervise_fixed_work(ProductivePanelAuditor(integrated_diagnostics=True,reference_values=True,
                nonempty_reference_values=nonempty),
                directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            result=terminal["result"];inputs=result["authored_external_readback_inputs"]
            from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode
            from pokezero.mcts_eval.search_over_raw_exploration_workflow import read_recorded_workflow
            from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger
            from pokezero.mcts_eval.search_over_raw_diagnostic_producer import ReferenceValueFixedWorkRecipe
            from test_search_over_raw_native_chain import AuthoredChainHelpers
            recipes={r["configuration"]:_decode(r["recipe"]) for r in inputs["diagnostic_recipes"]}
            def read():
                return read_recorded_workflow(ReconciledPhaseALedger(inputs["ledger_directory"]),inputs["report"],
                    diagnostic_recipes=recipes,native_chain_recipe=_decode(inputs["native_chain_recipe"]),
                    diagnostic_review_path=inputs["diagnostic_review_path"],diagnostic_file_hashes=inputs["diagnostic_file_hashes"],
                    checkpoint_contract=_decode(inputs["checkpoint_contract"]),auditor_directory=directory,
                    auditor_pid=terminal["worker_pid"],allow_synthetic=True,_synthetic_helpers=AuthoredChainHelpers(),
                    expected_outer_terminal=terminal)
            readback=read()
            self.assertEqual((readback["fixed_slots"],readback["completed_fixed_jobs"]),(192,2))
            self.assertEqual(readback["reference_value_readback"]["completed_populations"],1)
            self.assertEqual(readback["reference_value_readback"]["labels"],160 if nonempty else 0)
            self.assertEqual(readback["reference_value_readback"]["complete_labels"],160 if nonempty else 0)
            self.assertEqual(readback["reference_value_readback"]["uncertain_labels"],0)
            self.assertEqual(readback["reference_value_readback"]["terminal_populations"],int(nonempty))
            self.assertFalse(readback["reference_value_readback"]["labels_used_for_selection"])
            self.assertFalse(readback["reference_value_readback"]["runtime_qualified"])
            self.assertFalse(readback["reference_value_readback"]["resource_cost_complete"])
            self.assertEqual(result["scientific_measurements"],0)
            # No current reader may accept a replacement of the completed
            # original job with its own label recipe omitted.
            import pickle
            labeled=next(key for key,recipe in recipes.items() if type(recipe) is ReferenceValueFixedWorkRecipe)
            row=next(r for r in inputs["report"]["diagnostic_population"]["slots"]
                if r["configuration_sha256"]==labeled and r["status"]=="DIAGNOSTIC_COMPLETE")
            if nonempty:
                saved=json.loads((Path(row["attempt_directory"])/"terminal.json").read_bytes())["result"]
                bundle=saved["owned_reference_value_diagnostic"]
                self.assertEqual(bundle["summary"]["sampled_leaves"],20)
                self.assertTrue(all(v["evidence"]["model_evaluations_seen"]>0 for v in bundle["values"]))
                proof=bundle["values"][0]["evidence"]["proofs"][0]
                leaf_path=Path(proof["directory"])/"labels-readout.json"
                leaf_bytes=leaf_path.read_bytes()
                changed=json.loads(leaf_bytes)
                changed["labels"][0]["signed_outcome"]=-1
                leaf_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):read()
                leaf_path.write_bytes(leaf_bytes)
                self.assertEqual(read()["reference_value_readback"],readback["reference_value_readback"])
            path=Path(row["attempt_directory"])/"job.pickle";original=path.read_bytes()
            job=pickle.loads(original)
            # Do not re-capture an already sanitized request: the deliberate
            # negative changes only this disposable original job's recipe.
            object.__setattr__(job,"owned_reference_value_recipe",None)
            path.write_bytes(pickle.dumps(job))
            with self.assertRaises(ValueError):read()
            path.write_bytes(original)
            self.assertEqual(read()["reference_value_readback"],readback["reference_value_readback"])

    def test_full_after_population_is_null_only_and_read_after_actual_outer_reap(self):
        self._full_after_population(prefix=False,unavailable=False)

    def test_unavailable_prefix_full_workflow_keeps_roster_null_after_and_completed_readback(self):
        self._full_after_population(prefix=True,unavailable=True)

    def test_complete_v2_prefix_full_workflow_keeps_actual_after_endpoint_and_null_labels(self):
        self._full_after_population(prefix=True,unavailable=False)

    def _full_after_population(self,*,prefix,unavailable):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            terminal=supervise_fixed_work(ProductivePanelAuditor(integrated_diagnostics=True,after_population=True,
                preserve_replay_prefix=prefix,unavailable_native_replay_root=unavailable),
                directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            result=terminal["result"];inputs=result["authored_external_readback_inputs"]
            self.assertTrue(result["archives_closed"])
            self.assertEqual(result["launches"],dict(raw=1,search=8,continuation=2,reused_selectors=11,reused_a3_records=1))
            self.assertEqual(result["scientific_measurements"],0)
            population=inputs["report"]["native_after_population"]
            self.assertEqual((population["required_native_slots"],population["known_retained_events"],
                population["known_required_cells"],population["unknown_native_slots"],population["consumed_cells"]),(32,2,16,31,16))
            self.assertTrue(all(c["signed_outcome"] is None for e in population["events"] for c in e["bank"]["cells"]))
            if prefix:
                if unavailable:
                    self.assertTrue(all(e["admission"]["status"]=="CANONICAL_PATH_UNAVAILABLE"
                        and e["admission"]["raw_after_sha256"] is None
                        and not e["admission"]["original_endpoint_provenance_validated"] for e in population["events"]))
                else:
                    self.assertTrue(all(e["admission"]["raw_after_sha256"] is not None
                        and e["admission"]["original_endpoint_provenance_validated"] for e in population["events"]))
                chains=[r for r in inputs["report"]["native_chains"] if r["status"]=="CHAIN_RECORDED"]
                joined=chains[0]["diagnostic"]["owned_replay_join"]
                self.assertEqual((joined["required_construction_sample_records"],joined["replay_ordinal_records"],
                    joined["required_boundary_records"]),(8,8,16))
                self.assertTrue(joined["schema"].endswith(".v2"))
                expected_status="OWNED_PREFIX_AND_UNAVAILABLE_SUFFIX" if unavailable else "OWNED_RAW_ROOT_AND_COORDINATES_NOT_TRANSITIONS"
                self.assertTrue(all(r["status"]==expected_status
                    and len(r["reference_boundaries"])==2 and (r["reference_boundaries"][1] is None)==unavailable
                    for r in joined["records"]))
            from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode
            from pokezero.mcts_eval.search_over_raw_exploration_workflow import read_recorded_workflow
            from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger
            from test_search_over_raw_native_chain import AuthoredChainHelpers
            import pokezero.mcts_eval.search_over_raw_after_population as after_module
            from unittest.mock import patch
            captured=[]
            original_reader=after_module.read_recorded_after_population
            def record_context(*args,**kwargs):
                captured.append((args,kwargs))
                return original_reader(*args,**kwargs)
            def read(outer=terminal):
                return read_recorded_workflow(ReconciledPhaseALedger(inputs["ledger_directory"]),inputs["report"],
                    diagnostic_recipes={r["configuration"]:_decode(r["recipe"]) for r in inputs["diagnostic_recipes"]},
                    native_chain_recipe=_decode(inputs["native_chain_recipe"]),diagnostic_review_path=inputs["diagnostic_review_path"],
                    diagnostic_file_hashes=inputs["diagnostic_file_hashes"],checkpoint_contract=_decode(inputs["checkpoint_contract"]),
                    auditor_directory=directory,auditor_pid=terminal["worker_pid"],allow_synthetic=True,
                    _synthetic_helpers=AuthoredChainHelpers(),native_after_recipe=_decode(inputs["native_after_recipe"]),
                    expected_outer_terminal=outer)
            with patch.object(after_module,"read_recorded_after_population",side_effect=record_context):
                readback=read()["native_after_readback"]
            self.assertEqual((readback["consumed_cells"],readback["after_process_launches"],readback["scored_terminal_labels"]),(16,0,0))
            self.assertTrue(readback["enclosing_auditor_terminal_validated"])
            self.assertFalse(readback["p1_history_externally_reconstructed"])
            self.assertFalse(readback["resource_cost_complete"])
            if prefix:
                # The completed reader must independently reject changes to
                # an original child's stopping witness and full frozen roster.
                ledger=Path(inputs["ledger_directory"])
                replay_root=next((ledger/"native-diagnostic-chains").iterdir())/"replays"
                path=next(replay_root.glob("*/terminal.json"));saved_terminal=path.read_bytes()
                changed=json.loads(saved_terminal)
                changed["result"]["capture"]["termination"]["reason"]="REFERENCE_TERMINAL"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):read()
                path.write_bytes(saved_terminal)
                plan=replay_root.parent/"replay-plan.json";saved_plan=plan.read_bytes()
                changed=json.loads(saved_plan)
                next(r for r in changed["private_plan"]["records"] if r["job_spec"] is not None)["job_spec"]["request_ids"].pop()
                plan.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):read()
                plan.write_bytes(saved_plan)
                self.assertEqual(read()["native_after_readback"],readback)
            # Changing durable bytes cannot acquire a new baseline even when
            # the caller offers an otherwise valid owning terminal.
            banks=list((directory/"after-banks").iterdir());claim=banks[0]/"claim.json";saved=claim.read_bytes()
            claim.write_bytes(saved+b"\n")
            # Formatting is not semantic drift; an extra launch directory is.
            extra=banks[0]/"unexpected-bridge";extra.mkdir()
            args,kwargs=captured[0]
            with self.assertRaises(ValueError):original_reader(*args,**kwargs)
            extra.rmdir();claim.write_bytes(saved)
            from copy import deepcopy
            bad=deepcopy(terminal);bad["worker_exit_code"]=1
            with self.assertRaises(ValueError):original_reader(*args,**dict(kwargs,expected_outer_terminal=bad))
            with self.assertRaises(ValueError):original_reader(*args,**dict(kwargs,expected_outer_terminal=None))
            bank_result=banks[0]/"result.json";original=bank_result.read_bytes()
            invalid=json.loads(original);invalid["cells"][0]["signed_outcome"]=1
            bank_result.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):original_reader(*args,**kwargs)
            bank_result.write_bytes(original)
            memory=directory/"memory-trace.json"
            self.assertTrue(memory.exists())
            original_memory=memory.read_bytes();value=json.loads(original_memory)
            value["binding_sha256"]="b"*64;memory.write_text(json.dumps(value))
            with self.assertRaises(ValueError):original_reader(*args,**kwargs)
            memory.write_bytes(original_memory)
            import pokezero.mcts_eval.search_over_raw_resources as resource_module
            original_validate=resource_module.validate_memory_receipt
            recipe=kwargs["recipe"];old_runtime=recipe.runtime
            def replace_equal_runtime(*args,**kw):
                answer=original_validate(*args,**kw)
                object.__setattr__(recipe,"runtime",deepcopy(recipe.runtime))
                return answer
            with patch.object(resource_module,"validate_memory_receipt",side_effect=replace_equal_runtime):
                with self.assertRaisesRegex(ValueError,"class/field references"):original_reader(*args,**kwargs)
            object.__setattr__(recipe,"runtime",old_runtime)

    def test_authored_boundary_export_bytes_are_identical_across_hash_seeds(self):
        code = ("from test_search_over_raw_native_chain import AuthoredChainHelpers; "
                "import hashlib; "
                "print(hashlib.sha256(AuthoredChainHelpers.extract('complete-state',True).encode()).hexdigest())")
        exports = [subprocess.check_output([sys.executable, "-B", "-c", code],
            env=dict(os.environ, PYTHONHASHSEED=seed), text=True).strip() for seed in ("1", "2", "3")]
        self.assertEqual(len(set(exports)), 1)

    def test_actual_full_panel_diagnostics_precede_unchanged_stages_and_archives_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()/"auditor"
            terminal=supervise_fixed_work(ProductivePanelAuditor(integrated_diagnostics=True),
                directory=directory,binding_sha256="a"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
            result=terminal["result"]
            self.assertEqual(result["authored_workflow_summary"],dict(required_diagnostic_slots=192,
                native_chain_rows=32,recorded_chains=1,constructor_runs=4,replay_runs=8,
                diagnostic_results_used_for_selection=False))
            self.assertTrue(result["archives_closed"])
            self.assertTrue(result["authored_prior_owner_capture_reuse_validated"])
            self.assertEqual(result["launches"],dict(raw=1,search=8,continuation=2,reused_selectors=11,reused_a3_records=1))
            self.assertEqual(result["scientific_measurements"],0)
            from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode
            from pokezero.mcts_eval.search_over_raw_exploration_workflow import read_recorded_workflow
            from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger
            from test_search_over_raw_native_chain import AuthoredChainHelpers
            inputs=result["authored_external_readback_inputs"]
            readback=read_recorded_workflow(ReconciledPhaseALedger(inputs["ledger_directory"]),inputs["report"],
                diagnostic_recipes={r["configuration"]:_decode(r["recipe"]) for r in inputs["diagnostic_recipes"]},
                native_chain_recipe=_decode(inputs["native_chain_recipe"]),diagnostic_review_path=inputs["diagnostic_review_path"],
                diagnostic_file_hashes=inputs["diagnostic_file_hashes"],checkpoint_contract=_decode(inputs["checkpoint_contract"]),
                auditor_directory=directory,auditor_pid=terminal["worker_pid"],allow_synthetic=True,
                _synthetic_helpers=AuthoredChainHelpers())
            self.assertEqual((readback["fixed_slots"],readback["native_chain_rows"],readback["completed_fixed_jobs"],
                readback["materializer_jobs"],readback["replay_jobs"]),(192,32,1,4,8))
            self.assertFalse(readback["closed_source_archive_reconstructed"])
