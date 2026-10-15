"""Disposable data/ownership tests only: no scientific authority or collection.

Authored review flags and checkpoint bytes are deliberately NOT production
qualification. Mocked readback fixtures test guards, not simulator/model truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import unittest
from unittest.mock import patch

import pokezero.mcts_eval.search_over_raw_exploration_entry as entry
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_reconciled import (
    ReconciledPhaseALedger, controller_input_hashes, configuration_review_template,
    a1_configurations, a2_configurations, deployment_configurations, selector_runtime_recipe,
    HOLDOUT_ONLY_GATE,
)
from pokezero.mcts_eval.search_over_raw_continuation_admission import fresh_history_review_template
from pokezero.mcts_eval.search_over_raw_source_run import source_recipe, source_review_template
from pokezero.mcts_eval.search_over_raw_raw_run import raw_runtime
from pokezero.mcts_eval.search_over_raw_stage_run import exploration_recipe_keys
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    _state_payload, TIMED_NATIVE_PUBLIC_CAPTURE, TIMED_REFERENCE_PUBLIC_CAPTURE,
)
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_resources import measurement_protocol,resource_source_hash
from pokezero.mcts_eval.resolver import CheckpointContract
import test_search_over_raw_continuation_admission as admission_fixture
from test_search_over_raw_reference_timed_run import owner_runtime
from test_search_over_raw_timed_run import runtime as native_runtime
from test_search_over_raw_native_oracle_run import observed_runtime as native_oracle_runtime


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value))


@dataclass(frozen=True)
class AuthoredEntryPhases:
    delay: float=0.
    fail: bool=False
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup");phase("selection")
        try:
            if self.fail:raise RuntimeError("authored refusal")
            time.sleep(self.delay)
            return dict(authored=True,scientific_measurements=0)
        finally:phase("cleanup")


class ExplorationEntryTests(unittest.TestCase):
    def setUp(self):
        self.h=admission_fixture.FreshLifecycleTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.root=self.h.base.root.resolve();self.directory=self.h.directory

    def make(self,*,admitted=True,alter=None):
        h=self.h;o=h.outcome
        contract=deepcopy(h.contract)
        contract.update(execution_ready=admitted,collection_authorized=admitted,
            remaining_gates=[HOLDOUT_ONLY_GATE] if admitted else ["runtime is not admitted"])
        checkpoint=CheckpointContract(checkpoint_path=o.runtime["checkpoint_path"],
            checkpoint_sha256=o.runtime["files"][o.runtime["checkpoint_path"]],
            policy_id="authored",schema_version="authored",token_count=1,categorical_feature_count=1,
            numeric_feature_count=1,transition_token_count=0,category_vocab=(),architecture={"window_size":1},
            feature_masks={},model_device="cpu",showdown_root=str(o.config.resolved_showdown_root()),
            showdown_source_sha256="a"*64)
        raw=raw_runtime(SearchConfiguration("raw",seconds=1.),checkpoint.checkpoint_sha256,"a"*64)
        evidence=self.root/"authored-qualification.json"
        save(evidence,dict(authored=True,production_qualified=False,scientific_measurements=0))
        evidence_hashes={str(evidence):sha(evidence)}
        paths=[evidence];recipes=[]
        configs={cfg.identity:cfg for cfg in a1_configurations()+a2_configurations()+deployment_configurations()}
        for stage,identity in exploration_recipe_keys():
            cfg=configs[identity];factory=None
            if cfg.arm=="reference":
                runtime,factory=owner_runtime(cfg,checkpoint.checkpoint_path,checkpoint.showdown_root,checkpoint.checkpoint_sha256)
                factory=replace(factory,set_source_hash="a"*64)
                runtime.update(reference_factory=asdict(factory),set_source_hash="a"*64,
                    full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
                if cfg.belief=="oracle":
                    from pokezero.mcts_eval.search_over_raw_reference_oracle_run import REFERENCE_ORACLE_CAPTURE
                    runtime.update(reference_team_oracle_observation=deepcopy(REFERENCE_ORACLE_CAPTURE),
                        oracle_reconstruction_catalog_sha256="f"*64)
            else:
                runtime=(native_oracle_runtime(cfg,checkpoint.checkpoint_sha256) if cfg.belief=="oracle"
                    else native_runtime(cfg,checkpoint.checkpoint_sha256))
                runtime.update(set_source_hash="a"*64,full_public_request_observation=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE))
            review_path=self.root/(stage+"-"+identity+"-review.json")
            review=configuration_review_template(contract,stage=stage,configuration=identity,
                raw_runtime_sha256=digest(raw),selector_recipe_sha256=digest(selector_runtime_recipe(cfg,runtime)),
                qualification_evidence_hashes=evidence_hashes,cost_scope="Authored preflight fixture, not production qualification.")
            review.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
                atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
            save(review_path,review);paths.append(review_path)
            recipes.append(dict(stage=stage,configuration=identity,runtime=runtime,
                review_path=str(review_path),reference_factory=_state_payload(factory)))
        source_review_path=self.root/"source-review.json"
        source_review=source_review_template(contract,source_recipe(o.config,checkpoint,o.runtime),
            qualification_evidence_hashes=evidence_hashes)
        source_review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
            source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
        save(source_review_path,source_review);paths.append(source_review_path)
        history=fresh_history_review_template(contract,ledger_directory=self.directory,
            raw_runtime_sha256=digest(raw),continuation_runtime_sha256=digest(o.runtime))
        history.update(status="CLEAR_FRESH_DECLARED_NAMESPACE",reviewed_before_first_source_work=True)
        save(h.review_path,history);paths.append(h.review_path)
        self.spec_path=self.root/"entry-spec.json";review_path=self.root/"entry-review.json"
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        files={**fixed_work_reader_source_hashes(),**o.runtime["files"],
            str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve())}
        spec=dict(schema=entry.SCHEMA,ledger_directory=str(self.directory),
            contract_recipe_sha256=digest(entry._core(contract)),review_path=str(review_path),
            source_review_path=str(source_review_path),history_review_path=str(h.review_path),
            emergency_watchdog_seconds=20.,source_config=_state_payload(o.config),
            checkpoint_contract=_state_payload(checkpoint),source_runtime=deepcopy(o.runtime),
            selector_files=files,raw_runtime=raw,continuation_config=_state_payload(o.config),
            continuation_runtime=deepcopy(o.runtime),recipes=recipes)
        if alter:alter(spec,contract)
        save(self.spec_path,spec);paths.append(self.spec_path)
        review=entry.entry_review_template(contract,spec)
        review.update(status="CLEAR_EXPLORATION_ENTRY",reviewed_before_first_source_work=True,
            complete_recipe_registry_reviewed=True,whole_auditor_lifecycle_reviewed=True)
        save(review_path,review);paths.append(review_path)
        # An authored alter callback may freeze a real checkpoint/bridge
        # inventory instead of this default fixture's placeholders.
        contract["input_hashes"]={**controller_input_hashes(),**spec["selector_files"],**{str(p):sha(p) for p in paths}}
        self.ledger=ReconciledPhaseALedger.create(self.directory,contract)
        self.spec=spec;self.checkpoint=checkpoint
        return self.ledger

    def preflight(self):return entry.prepare_exploration_entry(self.directory,self.spec_path)

    def refuse_without_write(self,match):
        before=entry._evidence_inventory(self.directory)
        with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
            with self.assertRaisesRegex(ValueError,match):self.preflight()
        launch.assert_not_called();catalog.assert_not_called()
        self.assertEqual(before,entry._evidence_inventory(self.directory))

    def test_closed_codec_and_explicit_watchdog(self):
        from pokezero.local_showdown import LocalShowdownConfig
        cfg=LocalShowdownConfig()
        self.assertEqual(entry._decode(_state_payload(cfg)),cfg)
        for value in (float("nan"),float("inf"),{"path":False},
                {"dataclass":"os.system","fields":{}},{"mapping":[["x",1],["x",2]]}):
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):entry._decode(value)
        for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
            with self.assertRaises(ValueError):entry._json(raw)
        for value in (True,0,-1,float("nan"),float("inf"),None):
            with self.assertRaises(ValueError):entry.ExplorationAuditorContract(value)

    def test_original_entry_cost_readback_survives_prelaunch_failure_and_stopped_ledger(self):
        self.make(admitted=False)
        job=entry.ExplorationAuditorJob(str(self.directory),str(self.spec_path),sha(self.spec_path),
            self.ledger.contract_sha256,sha(self.spec["review_path"]))
        outer=entry.exploration_attempt_cost_spec(job);attempts=(outer,);roster_sha=digest([asdict(outer)])
        with patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored failure")):
            terminal=_supervise_owned_job(job,directory=outer.directory,binding_sha256=outer.binding_sha256,
                verify_admission=lambda:True,contract=entry.ExplorationAuditorContract(20.),
                kind="staged-exploration-run",work_phase="selection",validate_result=lambda r:None)
        self.ledger.stop("OSError")
        before=entry._evidence_inventory(self.directory)
        with patch("subprocess.Popen") as spawn,patch("os.wait4") as wait,patch.object(self.ledger,"verify") as verify:
            result=entry.read_exploration_attempt_cost(job,attempts,roster_sha256=roster_sha,expected_terminal=terminal)
        spawn.assert_not_called();wait.assert_not_called();verify.assert_not_called()
        self.assertEqual(before,entry._evidence_inventory(self.directory))
        self.assertEqual(result["attempts"][0]["state"],"PRELAUNCH_FAILURE_COST_OBSERVED")
        self.assertEqual(result["attempts"][0]["supervisor_thread_cpu_seconds"],terminal["supervisor_thread_cpu_seconds"])
        self.assertTrue(result["original_entry_inputs_bound"])
        self.assertTrue(result["supplied_roster_only"])
        self.assertFalse(result["attempt_population_exhaustiveness_validated"])
        self.assertFalse(result["completed_entry_work_validated"])
        self.assertFalse(result["scientific_collection_authorized"])
        for changed in (replace(outer,binding_sha256="a"*64),replace(outer,kind="fixed-work"),
                replace(outer,effective_watchdog_seconds=19.)):
            with self.assertRaisesRegex(ValueError,"outer job/watchdog"):
                entry.read_exploration_attempt_cost(job,(changed,),roster_sha256=digest([asdict(changed)]))
        wrong=dict(terminal);wrong["elapsed_wall_seconds"]+=1
        with self.assertRaisesRegex(ValueError,"returned outer terminal"):
            entry.read_exploration_attempt_cost(job,attempts,roster_sha256=roster_sha,expected_terminal=wrong)

    def test_entry_cost_reader_rejects_late_original_input_mutation(self):
        self.make(admitted=False)
        job=entry.ExplorationAuditorJob(str(self.directory),str(self.spec_path),sha(self.spec_path),
            self.ledger.contract_sha256,sha(self.spec["review_path"]))
        outer=entry.exploration_attempt_cost_spec(job);attempts=(outer,);roster_sha=digest([asdict(outer)])
        import pokezero.mcts_eval.search_over_raw_attempt_cost as costs
        actual=costs.reconcile_attempt_cost_roster
        def mutate(*args,**kwargs):
            value=actual(*args,**kwargs);self.spec_path.write_bytes(self.spec_path.read_bytes()+b" ");return value
        with patch.object(costs,"reconcile_attempt_cost_roster",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"inputs changed during readback"):
                entry.read_exploration_attempt_cost(job,attempts,roster_sha256=roster_sha)
        self.assertEqual(entry.ExplorationAuditorContract(20.).receipt()["emergency_watchdog_seconds"],20.)

    def test_read_only_prospective_complete_registry_before_a2_freeze(self):
        self.make();before=entry._evidence_inventory(self.directory)
        with patch("subprocess.Popen") as launch,patch("pokezero.randbat.load_gen3_randbat_source_cached") as catalog:
            job,prepared=self.preflight()
        launch.assert_not_called();catalog.assert_not_called()
        self.assertEqual(set(prepared["recipes"]),set(exploration_recipe_keys()))
        self.assertEqual(job.contract_sha256,self.ledger.contract_sha256)
        self.assertEqual(before,entry._evidence_inventory(self.directory))
        self.assertFalse((self.directory/"a2-freeze.json").exists())

    def test_unadmitted_refuses_before_spec_or_launch(self):
        self.make(admitted=False);self.spec_path.unlink()  # Authority check must precede data decoding.
        self.refuse_without_write("admission is incomplete|bound input drift")

    def test_unknown_gate_and_legacy_contract_are_not_admitted(self):
        self.make(alter=lambda s,c:c.update(remaining_gates=["unreviewed dependency"]))
        self.refuse_without_write("admission is incomplete")

    def test_valid_same_class_alternate_factory_rejected_before_work(self):
        def alter(spec,contract):
            row=next(r for r in spec["recipes"] if entry._decode(r["reference_factory"]) is not None)
            factory=entry._decode(row["reference_factory"])
            row["reference_factory"]=_state_payload(replace(factory,max_known_set_draws=factory.max_known_set_draws+1))
        self.make(alter=alter)
        self.refuse_without_write("actual reference factory differs")

    def test_missing_recipe_refuses_without_work(self):
        self.make(alter=lambda s,c:s["recipes"].pop())
        self.refuse_without_write("complete A1/A2")

    def test_duplicate_recipe_refuses_without_work(self):
        self.make(alter=lambda s,c:s["recipes"].append(deepcopy(s["recipes"][0])))
        self.refuse_without_write("duplicate prospective recipe")

    def test_prospectively_rebound_truncated_selector_files_refuse_before_work(self):
        def truncate(spec,contract):
            checkpoint=entry._decode(spec["checkpoint_contract"])
            spec["selector_files"]={checkpoint.checkpoint_path:checkpoint.checkpoint_sha256}
        self.make(alter=truncate)
        self.refuse_without_write("complete current selector file inventory")

    def test_simulator_executable_and_exported_model_bindings_are_required(self):
        self.make();files=deepcopy(self.spec["selector_files"])
        import sys
        for path in (str(Path(sys.executable).resolve()),self.checkpoint.showdown_root+"/dist/sim/state.js",
                     self.checkpoint.checkpoint_path):
            with self.subTest(path=Path(path).name):
                truncated={p:sha for p,sha in files.items() if p!=path}
                with self.assertRaisesRegex(ValueError,"complete current selector file inventory"):
                    entry._selector_files(truncated,self.checkpoint)
        model=self.root/"authored-export";model.write_bytes(b"not-an-export")
        exported=replace(self.checkpoint,model_path=str(model),model_sha256=sha(model))
        with self.assertRaisesRegex(ValueError,"complete current selector file inventory"):
            entry._selector_files(files,exported)
        entry._selector_files({**files,str(model):sha(model)},exported)

    def test_checkpoint_catalog_and_continuation_disagreement_refuse(self):
        self.make(alter=lambda s,c:s["continuation_runtime"].update(checkpoint_path="/unreviewed-model"))
        self.refuse_without_write("runtime|checkpoint")

    def test_original_review_byte_drift_is_not_readmitted(self):
        self.make();path=Path(self.spec["review_path"]);path.write_bytes(path.read_bytes()+b" ")
        self.refuse_without_write("bound input drift")

    def test_occupied_paths_do_not_enable_resumption(self):
        self.make();(self.directory/"source-attempts").mkdir()
        self.refuse_without_write("already occupied")

    def test_postwait_original_authority_allows_expected_completed_outputs(self):
        self.make();job,_=self.preflight()
        calls=[]
        def supervisor(actual,**kwargs):
            self.assertEqual(actual,job);kwargs["verify_admission"]();calls.append("fresh")
            directory=kwargs["directory"];directory.mkdir()
            payload=pickle.dumps(actual);(directory/"job.pickle").write_bytes(payload)
            save(directory/"claim.json",dict(job_sha256=hashlib.sha256(payload).hexdigest(),
                binding_sha256=digest(actual.__dict__),contract=kwargs["contract"].receipt(),
                resource_measurement_protocol=measurement_protocol(),resource_measurement_source_sha256=resource_source_hash(),
                retry_authorized=False,synthetic_process_fixture=False))
            (self.directory/"source-attempts").mkdir();save(self.directory/"selection.json",dict(authored=True))
            kwargs["verify_admission"]();calls.append("original")
            return dict(supervisor_exit_code=0,status="AUTHORED_CALLBACK_TEST")
        with patch.object(entry,"_supervise_owned_job",side_effect=supervisor),\
                patch.object(entry,"read_exploration_entry") as readback,\
                patch.object(entry,"_publish_entry_observer_window") as cost_publication:
            terminal=entry.run_exploration_entry(self.directory,self.spec_path)
        self.assertEqual(calls,["fresh","original"]);readback.assert_called_once_with(job,terminal)
        cost_publication.assert_called_once()  # Callback shapes only, no real cost receipt.
        self.refuse_without_write("frozen|already occupied")

    def test_postwait_drift_still_refuses(self):
        self.make()
        def supervisor(actual,**kwargs):
            kwargs["verify_admission"]()
            Path(actual.spec_path).write_bytes(Path(actual.spec_path).read_bytes()+b" ")
            kwargs["verify_admission"]()
        with patch.object(entry,"_supervise_owned_job",side_effect=supervisor),\
                patch.object(entry,"read_exploration_entry") as readback:
            with self.assertRaisesRegex(ValueError,"bound input drift"):
                entry.run_exploration_entry(self.directory,self.spec_path)
        readback.assert_not_called()

    def test_actual_distinct_outer_lifecycle_and_no_retry(self):
        for name,job,watchdog,status,exitcode in (
            ("normal",AuthoredEntryPhases(),20.,"STAGED_EXPLORATION_RUN_COMPLETE",0),
            ("failure",AuthoredEntryPhases(fail=True),20.,"OPERATIONAL_STOP",1),
            ("cap",AuthoredEntryPhases(delay=3.),.2,"STAGED_EXPLORATION_RUN_INCOMPLETE",124)):
            with self.subTest(name=name):
                directory=self.root/name;calls=[]
                def admission():calls.append(True);return True
                terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,
                    verify_admission=admission,contract=entry.ExplorationAuditorContract(watchdog),
                    kind="staged-exploration-run",work_phase="selection",validate_result=lambda v:v)
                self.assertEqual(terminal["status"],status,terminal)
                self.assertEqual(terminal["supervisor_exit_code"],exitcode,terminal)
                self.assertFalse(terminal["resource_cost_complete"])
                self.assertEqual(len(calls),2 if name=="normal" else 1)
                with self.assertRaises(FileExistsError):
                    _supervise_owned_job(job,directory=directory,binding_sha256="a"*64,
                        verify_admission=lambda:True,contract=entry.ExplorationAuditorContract(watchdog),
                        kind="staged-exploration-run",work_phase="selection",validate_result=lambda v:v)

    def test_actual_typed_worker_failure_consumes_only_disposable_attempt(self):
        self.make();captured=[]
        def supervise(*args,**kwargs):
            terminal=_supervise_owned_job(*args,**kwargs);captured.append(terminal);return terminal
        # Authored incomplete simulator files cannot load the catalog. This
        # exercises the REAL entry worker/claim/namespace/stop/wait4 path, not
        # scientific collection, a fake successful source, or a production model.
        with patch.object(entry,"_supervise_owned_job",side_effect=supervise):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry"):
                entry.run_exploration_entry(self.directory,self.spec_path)
        self.assertEqual(len(captured),1)
        terminal=captured[0]
        self.assertEqual(terminal["status"],"OPERATIONAL_STOP",terminal)
        self.assertEqual(terminal["supervisor_exit_code"],1,terminal)
        self.assertEqual(terminal["worker_exit_code"],1,terminal)
        self.assertFalse(terminal["watchdog_reached"])
        self.assertTrue((self.directory/"continuation-admission/namespace.json").is_file())
        self.assertTrue((self.directory/"operational-stop.json").is_file())
        from pokezero.mcts_eval.search_over_raw_attempt_journal import _permission_handle,read_attempt_journal_cost
        claim=json.loads((self.directory/"exploration-entry-run/claim.json").read_text())
        handle,_=_permission_handle(claim["attempt_declaration"])
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],1,value)
        self.assertEqual(value["attempts"][0]["state"],"REAPED_FAILURE_COST_OBSERVED",value)
        self.assertEqual(value["known_nonoverlapping_job_cpu_seconds"],terminal["cpu_seconds_including_reaped_children"])
        self.assertEqual(value["writer_close_errors"],[],value)
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        self.assertFalse((self.directory/"source-attempts").exists())
        self.assertFalse((self.directory/"staged-run-claim.json").exists())
        with patch("subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError,"operational integrity failure"):
                entry.run_exploration_entry(self.directory,self.spec_path)
        launch.assert_not_called()

    def test_real_guarded_entry_failed_spawn_is_declared_before_claim_and_never_retried(self):
        self.make();captured=[]
        def supervise(*args,**kwargs):
            terminal=_supervise_owned_job(*args,**kwargs);captured.append(terminal);return terminal
        with patch.object(entry,"_supervise_owned_job",side_effect=supervise),\
                patch("pokezero.mcts_eval.search_over_raw_supervision.subprocess.Popen",side_effect=OSError("authored spawn refusal")):
            with self.assertRaisesRegex(ValueError,"auditor failed; no retry"):
                entry.run_exploration_entry(self.directory,self.spec_path)
        terminal=captured[0]
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertIsNone(terminal["worker_pid"])
        from pokezero.mcts_eval.search_over_raw_attempt_journal import _permission_handle,read_attempt_journal_cost
        claim=json.loads((self.directory/"exploration-entry-run/claim.json").read_text())
        handle,_=_permission_handle(claim["attempt_declaration"])
        value=read_attempt_journal_cost(handle,expected_outer_terminal=terminal)
        self.assertEqual(value["required_attempt_slots"],1,value)
        self.assertEqual(value["attempts"][0]["state"],"PRELAUNCH_FAILURE_COST_OBSERVED")
        self.assertFalse((self.directory/"source-attempts").exists())
        self.assertFalse((self.directory/"continuation-admission").exists())
        self.assertFalse(value["declared_supervised_attempt_census_complete"])
        with patch("subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError,"journal already occupied"):
                entry.run_exploration_entry(self.directory,self.spec_path)
        spawn.assert_not_called()

    def readback_fixture(self):
        """Authored receipt shapes ONLY; ownership/physics mocked explicitly."""
        self.make();job,_=self.preflight();directory=self.directory/"exploration-entry-run";directory.mkdir()
        payload=pickle.dumps(job);(directory/"job.pickle").write_bytes(payload)
        binding=digest(job.__dict__);contract=entry.ExplorationAuditorContract(self.spec["emergency_watchdog_seconds"])
        save(directory/"claim.json",dict(job_sha256=hashlib.sha256(payload).hexdigest(),binding_sha256=binding,
            resource_measurement_protocol=measurement_protocol(),resource_measurement_source_sha256=resource_source_hash(),
            contract=contract.receipt(),retry_authorized=False,synthetic_process_fixture=False))
        save(directory/"launch.json",dict(pid=1,group=1,supervisor_pid=2,birth_identity="authored-not-live",binding_sha256=binding))
        seeds=self.ledger.contract["panels"]["exploration"]["seeds"]
        parent=self.directory/"source-attempts";parent.mkdir();source_hashes={}
        for seed in seeds:
            source_dir=parent/str(seed);source_dir.mkdir()
            missing=[r["root_id"] for r in self.ledger.contract["panels"]["exploration"]["root_slots"] if r["source_seed"]==seed]
            source=dict(roots=[],missing_root_ids=missing,authored_receipt_shape=True)
            save(source_dir/"source.json",source);source_hashes[str(seed)]=digest(source)
            save(source_dir/"claim.json",dict(auditor_pid=1,auditor_binding_sha256=binding,
                synthetic_process_fixture=False,source_seed=seed,
                recipe_sha256=digest(source_recipe(self.h.outcome.config,self.checkpoint,self.spec["source_runtime"])),
                source_review_sha256=sha(self.spec["source_review_path"])))
        result=dict(status="STAGED_EXPLORATION_RECORDED",reports={stage:{} for stage in ("A1","A2","A3")},
            evaluator_freeze={"evaluators":[]},selection={},launches={},source_panel=dict(source_count=len(seeds),
                schema="pokezero.search-over-raw.retained-exploration-panel.v1",contract_sha256=job.contract_sha256,
                source_recipe_sha256=digest(source_recipe(self.h.outcome.config,self.checkpoint,self.spec["source_runtime"])),
                source_runtime_sha256=digest(self.spec["source_runtime"]),source_review_sha256=sha(self.spec["source_review_path"]),
                source_roster_complete=True,source_pipeline_validated=True,runtime_qualified=False,
                collection_authorized=False,holdout_authorized=False,observed_root_ids=[],
                missing_root_ids=[r["root_id"] for seed in seeds for r in
                    self.ledger.contract["panels"]["exploration"]["root_slots"] if r["source_seed"]==seed],
                source_sha256_by_seed=source_hashes),
            synthetic_process_fixture=False,elapsed_seconds=0.,auditor_cpu_seconds=0.,resource_cost_complete=False,
            cost_scope="AUTHORED UNIT SHAPES ONLY",private_archive_returned=False,runtime_qualified_by_this_code=False,
            collection_authorized=False,holdout_authorized=False,phase_b_authorized=False)
        worker=dict(binding_sha256=binding,worker_pid=1,exit_code=0,error_type=None,result=result)
        terminal=dict(schema="pokezero.search-over-raw.staged-exploration-run-supervision.v1",
            status="STAGED_EXPLORATION_RUN_COMPLETE",binding_sha256=binding,worker_pid=1,
            supervisor_exit_code=0,worker_exit_code=0,error_type=None,cleanup_error_type=None,
            watchdog_reached=False,watchdog_seconds=contract.emergency_watchdog_seconds,
            memory_measurement=None,resource_measurement_error_type="AuthoredUnavailable",result=result)
        save(directory/"worker-result.json",worker);save(directory/"terminal.json",terminal)
        self.addCleanup(patch.stopall)
        patch.object(ReconciledPhaseALedger,"report_stage",side_effect=lambda stage:{}).start()
        patch.object(ReconciledPhaseALedger,"_a2",return_value={"evaluators":[]}).start()
        patch.object(ReconciledPhaseALedger,"_selection",return_value={}).start()
        patch.object(entry,"read_completed_process_cost",return_value={}).start()
        return job,terminal,parent

    def test_readback_late_source_callback_mutation_rejected(self):
        job,terminal,parent=self.readback_fixture();calls=[]
        def source_reader(*args):
            calls.append(args[0])
            if len(calls)==32:save(parent/"0/source.json",dict(late_mutation=True))
        with patch.object(entry,"verify_delegated_closure",return_value="authored-closure"),\
                patch.object(entry,"read_source_owner",side_effect=source_reader):
            with self.assertRaisesRegex(ValueError,"evidence changed during independent validation"):
                entry.read_exploration_entry(job,terminal)
        self.assertEqual(len(calls),32)

    def test_readback_final_closure_callback_stage_addition_rejected(self):
        job,terminal,parent=self.readback_fixture();calls=[]
        def closure(*args):
            calls.append(True)
            if len(calls)==2:save(self.directory/"A1/late-evidence.json",dict(authored=True))
            return "authored-closure"
        with patch.object(entry,"verify_delegated_closure",side_effect=closure),\
                patch.object(entry,"read_source_owner",return_value={}):
            with self.assertRaisesRegex(ValueError,"evidence changed during independent validation"):
                entry.read_exploration_entry(job,terminal)
        self.assertEqual(len(calls),2)

    def test_readback_unchanged_shapes_only_not_production_clearance(self):
        job,terminal,parent=self.readback_fixture()
        with patch.object(entry,"verify_delegated_closure",return_value="authored-closure"),\
                patch.object(entry,"read_source_owner",return_value={}):
            value=entry.read_exploration_entry(job,terminal)
        self.assertEqual(value["source_count"],32)
        self.assertFalse(value["runtime_qualified"])
        self.assertFalse(value["full_resource_cost_complete"])

    def test_inventory_tracks_sibling_source_stage_and_added_entries(self):
        self.make();before=entry._evidence_inventory(self.directory)
        (self.directory/"source-attempts").mkdir()
        self.assertNotEqual(before,entry._evidence_inventory(self.directory))
        source=self.directory/"source-attempts/source.json";save(source,dict(authored=True))
        before=entry._evidence_inventory(self.directory);save(source,dict(authored=False))
        self.assertNotEqual(before,entry._evidence_inventory(self.directory))
        alias=self.directory/"alias";alias.symlink_to(source)
        with self.assertRaises(ValueError):entry._evidence_inventory(self.directory)


if __name__=="__main__":unittest.main()
