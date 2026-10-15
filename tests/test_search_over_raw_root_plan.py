"""Actual Python archive plus disposable raw owner; authored model outcomes.

Continuation job binding and search readers are mocked only in the structural
alias/zero test. No simulator, source game, scientific root or holdout is run.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle
import time
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_raw_job
from pokezero.mcts_eval.search_over_raw_raw_run import RawSelectionRunContract,RawSelectionRunValidator,raw_runtime
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedRunValidator
from pokezero.mcts_eval.search_over_raw_outcome_run import OwnedContinuationJob,OwningRunValidator,supervise_outcome_run
import test_search_over_raw_archive as archive_fixture_module
import test_search_over_raw_raw_run as raw_fixture_module
import test_search_over_raw_outcome_run as outcome_fixture_module


@dataclass(frozen=True)
class AuthoredRawOwner:
    selection: dict
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        phase("startup");phase("selection");time.sleep(.01);phase("cleanup")
        return deepcopy(self.selection)


class RootPlanTests(unittest.TestCase):
    def setUp(self):
        self.archive_fixture=archive_fixture_module.ExplorationRootCrosswalkTests();self.archive_fixture.setUp()
        self.addCleanup(self.archive_fixture.doCleanups)
        self.owner_fixture=raw_fixture_module.RawOwnerTests();self.owner_fixture.setUp();self.addCleanup(self.owner_fixture.doCleanups)
        f=self.archive_fixture;o=self.owner_fixture
        self.bound=f.bind();self.cfg=o.cfg;self.mask=self.bound["public_evidence_binding"]["legal_action_mask"]
        self.runtime=raw_runtime(self.cfg,o.files[str(o.checkpoint)],self.bound["public_request_witness"]["consumed_binding"]["set_source_hash"])
        legal=[i for i,v in enumerate(self.mask) if v]
        self.selected=dict(root_id=f.root_id+":raw",statistics_root_id=f.root_id,selection_seed=self.bound["slot"]["source_seed"],
            configuration_sha256=self.cfg.identity,runtime_sha256=digest(self.runtime),action=legal[0],status="SELECTED",
            elapsed_seconds=.005,nominal_search_seconds=1.,exceeded_nominal_seconds=False,
            independent_root_statistics=True,scientific_strength_evidence=False,
            raw_selection_run_contract=RawSelectionRunContract().receipt(),observed_runtime_configuration=deepcopy(self.runtime),
            evidence=dict(selector="deterministic_masked_argmax",model_forwards=1,legal_actions=legal,
                priors=[1/len(legal)]*len(legal),signed_value=.1,
                consumed_binding=deepcopy(self.bound["public_request_witness"]["consumed_binding"]),
                public_request_witness=deepcopy(self.bound["public_request_witness"])))
        job=AuthoredRawOwner(self.selected);directory=o.root/"plan-raw-owner"
        self.reader=RawSelectionRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,configuration=self.cfg,root_id=self.selected["root_id"],statistics_root_id=f.root_id,
            selection_seed=self.selected["selection_seed"],runtime_configuration=self.runtime,runtime_sha256=digest(self.runtime),
            consumed_binding=self.selected["evidence"]["consumed_binding"],legal_action_mask=self.mask,decision_round=None,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True,
            public_request_sha256=self.bound["public_request_witness"]["public_request_sha256"])
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
            contract=RawSelectionRunContract(),kind="raw-selection-run",work_phase="selection",validate_result=self.reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)

    def plan(self,**changes):
        f=self.archive_fixture
        args=dict(contract=f.contract,root_id=f.root_id,configurations=(self.cfg,),selections={"raw":self.selected},
            selector_readers={"raw":self.reader},allow_synthetic=True)
        args.update(changes)
        return ArchivedRootEvaluationPlan(f.archive,f.source,**args)

    def test_actual_archive_owned_raw_join_is_no_launch_and_no_private_export(self):
        with patch("subprocess.Popen") as launch:
            plan=self.plan();receipt=plan.revalidate()
            job,bound=prepare_archived_raw_job(self.archive_fixture.archive,self.archive_fixture.source,
                contract=self.archive_fixture.contract,root_id=self.archive_fixture.root_id,configuration=self.cfg,
                checkpoint_contract=self.checkpoint_contract(),
                showdown_root=str(self.owner_fixture.showdown),runtime_configuration=self.runtime)
        launch.assert_not_called()
        self.assertEqual(job.expected_public_request_sha256,bound["public_request_witness"]["public_request_sha256"])
        self.assertEqual(len(receipt["continuation_cells"]),8)
        self.assertEqual(receipt["root_evidence"]["uncertain_action_cells"],8)
        self.assertFalse(receipt["execution_ready"]);self.assertFalse(receipt["collection_authorized"])
        import json
        self.assertNotIn("private-source-snapshot",json.dumps(receipt))
        with self.assertRaises(TypeError):pickle.dumps(plan)
        with self.assertRaisesRegex(ValueError,"synthetic evidence"):self.plan(allow_synthetic=False)

    def checkpoint_contract(self):
        from types import SimpleNamespace
        return SimpleNamespace(checkpoint_path=str(self.owner_fixture.checkpoint),
            checkpoint_sha256=self.owner_fixture.files[str(self.owner_fixture.checkpoint)],
            showdown_source_sha256=self.runtime["set_source_hash"])

    def prepared_owned_continuations(self):
        # Actual archive, actual raw owning reader and actual disposable
        # continuation bridge; authored private state and policy, not science.
        fixture=outcome_fixture_module.OutcomeRunTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        f=self.archive_fixture
        index=f.source["roots"][0]["source_request_index"]
        f.archive._private[index]=deepcopy(fixture.snapshot)
        plan=self.plan()
        job,_=plan.prepare_continuation_job(config=fixture.config,runtime=fixture.runtime,
            _synthetic_job_class=outcome_fixture_module.SyntheticContinuationJob)
        binding=plan.continuation_attempt_binding(job)
        terminal=supervise_outcome_run(job,directory=fixture.directory,binding_sha256=binding,
            verify_admission=lambda:None,validate_result=lambda result:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader=OwningRunValidator(directory=fixture.directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=binding,
            root_binding=job.root_binding,expected_snapshot=job.snapshot,runtime=job.runtime,allow_synthetic=True)
        return plan,job,terminal["result"],reader

    def test_actual_archive_raw_owner_and_continuation_owner_acceptance(self):
        plan,job,result,reader=self.prepared_owned_continuations()
        with patch("subprocess.Popen") as launch:
            accepted=plan.accept_continuation_result(job,result,owning_reader=reader)
        launch.assert_not_called()
        self.assertEqual(accepted["attempt_binding_sha256"],plan.continuation_attempt_binding(job))
        self.assertEqual(accepted["root_evidence"]["uncertain_action_cells"],0)
        self.assertEqual(accepted["root_evidence"]["unique_action_cells"],8)
        self.assertTrue(accepted["synthetic_process_fixture"])
        for flag in ("execution_ready","runtime_qualified","collection_authorized","holdout_authorized",
                "evaluator_correctness_established","dynamics_correctness_established"):
            self.assertFalse(accepted[flag])

    def test_supplied_rows_and_attempt_identity_cannot_replace_actual_producer(self):
        plan,job,result,reader=self.prepared_owned_continuations()
        changed=deepcopy(result);changed["outcomes"]["outcomes"][0]["signed_outcome"]=-1.
        with self.assertRaisesRegex(ValueError,"actual owned producer result"):
            plan.accept_continuation_result(job,changed,owning_reader=reader)
        original=reader.binding;reader.binding="e"*64
        with self.assertRaisesRegex(ValueError,"owning continuation reader drift"):
            plan.accept_continuation_result(job,result,owning_reader=reader)
        reader.binding=original
        self.assertTrue(plan.accept_continuation_result(job,result,owning_reader=reader)["synthetic_process_fixture"])

    def test_late_reader_identity_and_synthetic_scope_cannot_be_substituted(self):
        plan,job,result,reader=self.prepared_owned_continuations()
        actual=plan._read_selector
        def mutate(cfg,selected,binding):
            value=actual(cfg,selected,binding)
            if getattr(reader,"last_validation",None) is not None:reader.binding="f"*64
            return value
        with patch.object(plan,"_read_selector",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"owning evidence changed"):
                plan.accept_continuation_result(job,result,owning_reader=reader)
        reader.binding=plan.continuation_attempt_binding(job)
        changed=deepcopy(result);changed["synthetic_process_fixture"]=False
        with self.assertRaisesRegex(ValueError,"synthetic scope mismatch"):
            plan.accept_continuation_result(job,changed,owning_reader=reader)

    def test_exact_owner_and_full_raw_archive_binding_not_a_digest_callback(self):
        with self.assertRaises(ValueError):self.plan(selector_readers={"raw":lambda r:digest(r)})
        before=self.reader.public_request_sha
        self.reader.public_request_sha="f"*64
        with self.assertRaises(ValueError):self.plan()
        self.reader.public_request_sha=before
        plan=self.plan()
        self.selected["action"]=next(i for i,v in enumerate(self.mask) if v and i!=self.selected["action"])
        self.assertEqual(plan.revalidate()["root_evidence"]["actions"]["raw"],0) # caller edits do not mutate frozen copy
        plan.selections["raw"]["action"]=self.selected["action"]
        with self.assertRaisesRegex(ValueError,"frozen root evaluation"):plan.revalidate()

    def test_owner_terminal_or_actual_private_archive_drift_blocks_handoff(self):
        plan=self.plan()
        path=self.reader.directory/"terminal.json";original=path.read_bytes();path.write_text("{}")
        with self.assertRaises((KeyError,ValueError)):plan.revalidate()
        path.write_bytes(original)
        index=self.archive_fixture.source["roots"][0]["source_request_index"]
        self.archive_fixture.archive._private[index].audit_only["canary"]="changed-private-state"
        with self.assertRaisesRegex(ValueError,"private snapshot changed"):plan.revalidate()

    def test_structural_alias_zero_and_prepared_job_guards_never_launch(self):
        # Mock only search-reader and continuation input-binding seams: actual
        # raw owner above remains independently read from its terminal files.
        inc=SearchConfiguration("incumbent",seconds=1.);hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=3.)
        names=[inc.identity,hp.identity];rows={"raw":deepcopy(self.selected)};readers={"raw":self.reader}
        for cfg in (inc,hp):
            name=cfg.identity;row=deepcopy(self.selected);row.update(root_id=self.archive_fixture.root_id+":"+name,
                configuration_sha256=name,nominal_search_seconds=cfg.seconds,evidence=dict(total_iterations=16))
            rows[name]=row
            reader=object.__new__(NativeTimedRunValidator)
            reader.configuration=cfg;reader.root_id=row["root_id"];reader.statistics_root_id=self.archive_fixture.root_id
            reader.seed=row["selection_seed"];reader.mask=self.mask
            reader.consumed=self.selected["evidence"]["consumed_binding"];reader.runtime_sha=row["runtime_sha256"]
            reader._inputs=lambda name=name:digest(dict(authored_reader=name))
            reader._files=lambda:None;reader.directory=self.reader.directory;reader.job_sha=self.reader.job_sha
            readers[name]=reader
        rows[hp.identity].update(status="UNQUALIFIED_AT_CEILING",action=None,attempt_consumed=True,
            retry_authorized=False,continuation_authorized=False,evidence=dict(total_iterations=0))
        actual_owned_read=NativeTimedRunValidator.__call__
        def authored_owned_read(reader,row):
            if type(reader) is RawSelectionRunValidator:return actual_owned_read(reader,row)
            reader.last_validation=dict(selected_sha256=digest(row),process_receipts_validated=True,synthetic_process_fixture=True,
                receipt_file_sha256={})
            return digest(row)
        def authored_job_binding(job):
            return dict(root=job.root_binding,snapshot=job.snapshot.audit_only,actions=job.actions,
                namespace=job.namespace,config=job.config,runtime=job.runtime,subject=job.subject)
        with patch.object(NativeTimedRunValidator,"__call__",autospec=True,side_effect=authored_owned_read), \
                patch.object(OwnedContinuationJob,"input_binding",autospec=True,side_effect=authored_job_binding), \
                patch("subprocess.Popen") as launch:
            plan=self.plan(configurations=(self.cfg,inc,hp),selections=rows,selector_readers=readers)
            receipt=plan.revalidate()
            self.assertEqual(len(receipt["continuation_cells"]),8) # raw/native aliases share all cells
            self.assertEqual(receipt["root_evidence"]["root_intervals"][inc.identity],[0.,0.])
            self.assertEqual(receipt["root_evidence"]["root_intervals"][hp.identity],[-1.,1.])
            job,receipt=plan.prepare_continuation_job(config={"authored":True},runtime={"authored":True})
            self.assertNotIn(hp.identity,job.actions)
            plan.verify_prepared_job(job)
            job.actions["raw"]=999
            with self.assertRaisesRegex(ValueError,"changed continuation job"):plan.verify_prepared_job(job)
        launch.assert_not_called()

    def test_later_validation_cannot_change_prepared_private_snapshot(self):
        # Mutate after the initial job check but before the final handoff. This
        # is a callback-race guard, not independent continuation qualification.
        plan=self.plan()
        def binding(job):
            return dict(root=job.root_binding,snapshot=job.snapshot.audit_only,config=job.config,
                runtime=job.runtime,actions=job.actions,namespace=job.namespace,subject=job.subject)
        actual_revalidate=plan.revalidate
        with patch.object(OwnedContinuationJob,"input_binding",autospec=True,side_effect=binding):
            job,_=plan.prepare_continuation_job(config={},runtime={})
            def mutate():
                receipt=actual_revalidate();job.snapshot.audit_only["canary"]="late-change"
                return receipt
            with patch.object(plan,"revalidate",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"prepared root/actions"):plan.verify_prepared_job(job)
            # Also protect preparation's final revalidation, before the caller
            # ever receives the newly created job.
            captured=[]
            def capture(job):captured.append(job);return binding(job)
            def mutate_new():
                receipt=actual_revalidate()
                if captured:captured[-1].runtime["late"]="changed"
                return receipt
            with patch.object(OwnedContinuationJob,"input_binding",new=capture), \
                    patch.object(plan,"revalidate",side_effect=mutate_new):
                with self.assertRaisesRegex(ValueError,"final preparation"):plan.prepare_continuation_job(config={},runtime={})

    def test_later_owned_reader_cannot_rewrite_earlier_raw_terminal(self):
        cfg=SearchConfiguration("incumbent",seconds=1.)
        row=deepcopy(self.selected);row.update(root_id=self.archive_fixture.root_id+":"+cfg.identity,
            configuration_sha256=cfg.identity,evidence=dict(total_iterations=16))
        reader=object.__new__(NativeTimedRunValidator)
        reader.configuration=cfg;reader.root_id=row["root_id"];reader.statistics_root_id=self.archive_fixture.root_id
        reader.seed=row["selection_seed"];reader.mask=self.mask;reader.consumed=self.selected["evidence"]["consumed_binding"]
        reader.runtime_sha=row["runtime_sha256"];reader._inputs=lambda:"authored-later-owner"
        reader._files=lambda:None;reader.directory=self.reader.directory;reader.job_sha=self.reader.job_sha
        actual=NativeTimedRunValidator.__call__;path=self.reader.directory/"terminal.json";before=path.read_bytes()
        def later(owner,selection):
            if type(owner) is RawSelectionRunValidator:return actual(owner,selection)
            owner.last_validation=dict(selected_sha256=digest(selection),process_receipts_validated=True,
                synthetic_process_fixture=True,receipt_file_sha256={})
            path.write_bytes(before+b" ") # semantically unchanged JSON, different exact durable bytes
            return digest(selection)
        with patch.object(NativeTimedRunValidator,"__call__",autospec=True,side_effect=later):
            with self.assertRaisesRegex(ValueError,"earlier owner receipts changed"):
                self.plan(configurations=(self.cfg,cfg),selections={"raw":self.selected,cfg.identity:row},
                    selector_readers={"raw":self.reader,cfg.identity:reader})


if __name__ == "__main__":unittest.main()
