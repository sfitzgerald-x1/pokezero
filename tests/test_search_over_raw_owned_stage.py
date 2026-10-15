"""Actual archived/owned handoff into the fixed-roster ledger.

Temporary authored states/native reports and disposable synthetic processes only.
No model, scientific source game, production search or holdout is run.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import pickle
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger
from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
from pokezero.mcts_eval.search_over_raw_source import source_roots
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedRunContract,NativeTimedRunValidator
from pokezero.mcts_eval.search_over_raw_world_inputs import TIMED_NATIVE_PUBLIC_CAPTURE
from pokezero.mcts_eval.search_over_raw_outcome_run import OwningRunValidator,supervise_outcome_run
from pokezero.env import TerminalState
from pokezero.trajectory import BattleTrajectory,TrajectoryStep
from types import SimpleNamespace
import test_search_over_raw_archive as archive_fixture
import test_search_over_raw_root_plan as plan_fixture
import test_search_over_raw_timed_public as public_fixture
import test_search_over_raw_timed_run as timed_fixture
import test_search_over_raw_outcome_run as outcome_fixture


class OwnedStageTests(unittest.TestCase):
    def setUp(self):
        self.cfg=SearchConfiguration("incumbent",seconds=1.)
        self.fixtures=[]

    def fixture(self,seed=0):
        original=archive_fixture.ExplorationRootCrosswalkTests.setUp
        def setup(f):
            original(f)
            if not seed:return
            f.context=replace(f.context,seed=seed,
                trajectory=BattleTrajectory("authored-boundary","gen3randombattle",seed))
            f.archive=SealedSourceArchive();f.addCleanup(f.archive.close)
            f.archive.capture_public(f.context,0)
            f.archive.capture_private(SimpleNamespace(requested_players=("p1",),decision_round_index=0,
                seed=seed,battle_id="authored-boundary",decisions={"p1":SimpleNamespace(action_index=0)},
                snapshot=archive_fixture.AuthoredPrivateBoundary({"canary":"private-source-snapshot"})))
            trajectory=BattleTrajectory("authored-boundary","gen3randombattle",seed)
            trajectory.append(TrajectoryStep("p1",0,f.context.observation,
                f.context.observation.legal_action_mask,0))
            trajectory.record_terminal(TerminalState("p1",1))
            f.source=source_roots(f.contract,panel="exploration",source_seed=seed,trajectory=trajectory)
            f.root_id=f.source["roots"][0]["root_id"]
        f=plan_fixture.RootPlanTests()
        with patch.object(archive_fixture.ExplorationRootCrosswalkTests,"setUp",setup):f.setUp()
        self.addCleanup(f.doCleanups);self.fixtures.append(f)
        return f

    def selector(self,f,*,zero=False,cfg=None,action=0):
        cfg=cfg or self.cfg;o=f.owner_fixture;a=f.archive_fixture
        witness=f.bound["public_request_witness"]
        runtime=timed_fixture.runtime(cfg,o.files[str(o.checkpoint)])
        runtime["set_source_hash"]=witness["consumed_binding"]["set_source_hash"]
        runtime["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
        if zero:
            helper=public_fixture.TimedPublicTests();helper.f=a;helper.bound=f.bound
            helper.witness=witness;helper.cfg=cfg;self.addCleanup(helper.doCleanups)
            selected=helper.no_attempt_selection()
        else:
            with patch.object(timed_fixture,"BINDING",witness["consumed_binding"]):
                selected=timed_fixture.selected_fixture(cfg,runtime,decision_round=0)
            manifest=selected["evidence"]["consumed_world_inputs"]
            manifest["adapter_public_request_witness"]=deepcopy(witness)
            manifest["public_request_observations"]=[dict(ordinal=i,witness=deepcopy(witness))
                for i in range(len(manifest["rows"]))]
            selected.update(action=action)
            selected["evidence"]["root_action"]=f"action:{action}"
        selected.update(root_id=a.root_id+":"+cfg.identity,statistics_root_id=a.root_id,
            selection_seed=f.bound["slot"]["source_seed"],runtime_sha256=digest(runtime),
            observed_runtime_configuration=runtime,native_timed_run_contract=NativeTimedRunContract().receipt())
        job=public_fixture.AuthoredNativeSelectionOwner(selected)
        directory=o.root/("native-"+cfg.identity)
        reader=NativeTimedRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,configuration=cfg,root_id=selected["root_id"],statistics_root_id=a.root_id,
            selection_seed=selected["selection_seed"],runtime_configuration=runtime,runtime_sha256=digest(runtime),
            consumed_binding=witness["consumed_binding"],legal_action_mask=f.mask,decision_round=0,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=witness["public_request_sha256"])
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,
            verify_admission=lambda:None,contract=NativeTimedRunContract(),kind="native-timed-run",
            work_phase="selection",validate_result=reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        return selected,reader

    def ledger(self,items):
        # Reviews are bound BEFORE constructing these authored fixture plans.
        # This is not scientific qualification or registration.
        bindings={f.archive_fixture.root_id:f.bound["public_record_sha256"] for f,_,_ in items}
        hashes=controller_input_hashes();self.reviews={}
        for f,selected,reader in items:
            cfg=reader.configuration
            path=f.owner_fixture.root/(cfg.identity+"-review.json")
            review=dict(status="CLEAR_UNQUALIFIED_AT_CEILING" if selected["action"] is None
                else "CLEAR_CONFIGURATION_AT_CEILING",configuration_sha256=cfg.identity,
                runtime_sha256=reader.runtime_sha,public_bindings_sha256=digest(bindings),
                atomic_work_and_cleanup_validated=True,zero_work_is_local_unqualified=True)
            if cfg.identity not in self.reviews:
                path.write_text(json.dumps(review));hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                self.reviews[cfg.identity]=path
        contract=dict(items[0][0].archive_fixture.contract,input_hashes=hashes)
        self.ledger_object=PhaseALedger.create(items[0][0].owner_fixture.root/"owned-ledger",contract)
        for f,_,_ in items:
            f.archive_fixture.contract=deepcopy(contract)
            f.archive_fixture.source["contract_sha256"]=digest(contract)
        return self.ledger_object

    def plan(self,f,selected,reader,*,outcome=False,extras=(),job_class=outcome_fixture.SyntheticContinuationJob):
        if outcome:
            fixture=outcome_fixture.OutcomeRunTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
            index=f.archive_fixture.source["roots"][0]["source_request_index"]
            f.archive_fixture.archive._private[index]=deepcopy(fixture.snapshot)
        configs=(f.cfg,reader.configuration)+tuple(r.configuration for _,r in extras)
        selections={"raw":f.selected,reader.configuration.identity:selected}
        readers={"raw":f.reader,reader.configuration.identity:reader}
        for row,r in extras:selections[r.configuration.identity]=row;readers[r.configuration.identity]=r
        plan=f.plan(configurations=configs,selections=selections,selector_readers=readers)
        if not outcome:return plan,None
        job,_=plan.prepare_continuation_job(config=fixture.config,runtime=fixture.runtime,
            _synthetic_job_class=job_class)
        binding=plan.continuation_attempt_binding(job)
        terminal=supervise_outcome_run(job,directory=fixture.directory,binding_sha256=binding,
            verify_admission=lambda:None,validate_result=lambda result:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        owner=OwningRunValidator(directory=fixture.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=binding,root_binding=job.root_binding,expected_snapshot=job.snapshot,
            runtime=job.runtime,allow_synthetic=True)
        return plan,(job,terminal["result"],owner)

    def ingest(self,ledger,plans,continuations,cfg=None,stage="A2"):
        cfg=cfg or self.cfg
        return ledger.record_owned_stage_evidence(stage,cfg.identity,plans=plans,continuations=continuations,
            qualification_review_path=self.reviews.get(cfg.identity,self.reviews[self.cfg.identity]))

    def evidence(self,cfg=None):
        return json.loads((self.ledger_object.directory/"A2"/
            ((cfg or self.cfg).identity+"-root-evidence.json")).read_text())

    def test_productive_without_continuation_keeps_fixed_roster_uncertain(self):
        f=self.fixture();selected,reader=self.selector(f,action=1)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        with patch("subprocess.Popen") as launch:
            row=self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        launch.assert_not_called()
        evidence=self.evidence();handoff=evidence["owned_input_handoff"]
        self.assertEqual(len(ledger.contract["panels"]["exploration"]["root_slots"]),200)
        self.assertEqual(len(ledger.contract["panels"]["exploration"]["seeds"]),32)
        self.assertEqual(row["summary"]["lower_mean"],-1.)
        self.assertEqual(evidence["root_intervals"][plan.root_id],[-1.,1.])
        self.assertEqual(handoff["unsupplied_continuation_roots"],[plan.root_id])
        self.assertFalse(handoff["unsupplied_attempts_proven_absent"])
        for flag in ("runtime_qualified","collection_authorized","holdout_authorized"):
            self.assertFalse(handoff[flag])

    def test_actual_continuation_aliases_and_global_owner_join(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader,outcome=True)
        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        evidence=self.evidence();root=evidence["root_evidence"][plan.root_id]
        self.assertEqual(root["unique_action_cells"],8)
        self.assertEqual(root["uncertain_action_cells"],0)
        self.assertEqual(evidence["root_intervals"][plan.root_id],[0.,0.])
        self.assertTrue(evidence["owned_input_handoff"]["supplied_continuation_owners_validated"])

    def test_two_configurations_reuse_one_actual_owned_bank(self):
        f=self.fixture();selected,reader=self.selector(f)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=self.selector(f,cfg=hp,action=1)
        ledger=self.ledger([(f,selected,reader),(f,other,other_reader)])
        plan,data=self.plan(f,selected,reader,outcome=True,extras=((other,other_reader),))
        plans={plan.root_id:plan};continuations={plan.root_id:data}
        self.ingest(ledger,plans,continuations)
        self.ingest(ledger,plans,continuations,cfg=hp)
        model=self.evidence();hp_evidence=self.evidence(hp)
        self.assertEqual(model["root_evidence"][plan.root_id]["unique_action_cells"],8)
        self.assertEqual(hp_evidence["root_evidence"][plan.root_id]["unique_action_cells"],16)
        for name,sha in model["owned_input_handoff"]["shared_cell_provenance"].items():
            self.assertEqual(hp_evidence["owned_input_handoff"]["shared_cell_provenance"][name],sha)

    def test_identical_labels_from_redrawn_owner_cannot_replace_shared_cells(self):
        f=self.fixture();selected,reader=self.selector(f)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=self.selector(f,cfg=hp)
        ledger=self.ledger([(f,selected,reader),(f,other,other_reader)])
        plan,data=self.plan(f,selected,reader,outcome=True,extras=((other,other_reader),))
        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        job,result,owner=data
        directory=owner.directory.parent/"redraw-attempt"
        binding=plan.continuation_attempt_binding(job)
        terminal=supervise_outcome_run(job,directory=directory,binding_sha256=binding,
            verify_admission=lambda:None,validate_result=lambda result:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        redraw=OwningRunValidator(directory=directory,job_sha256=owner.job_sha,binding_sha256=binding,
            root_binding=job.root_binding,expected_snapshot=job.snapshot,runtime=job.runtime,allow_synthetic=True)
        self.assertEqual(result["outcomes"],terminal["result"]["outcomes"])
        with self.assertRaisesRegex(ValueError,"actual shared continuation cell"):
            self.ingest(ledger,{plan.root_id:plan},{plan.root_id:(job,terminal["result"],redraw)},cfg=hp)
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A2"/(hp.identity+"-root-evidence.json")).exists())

    def test_original_mapping_mutation_after_claim_is_not_a_stage_success(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        plans={plan.root_id:plan};continuations={plan.root_id:data}
        actual=plan._read_selector
        def mutate(cfg,row,binding):
            result=actual(cfg,row,binding);continuations.clear();return result
        with patch.object(plan,"_read_selector",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"inputs/evidence changed"):
                self.ingest(ledger,plans,continuations)
        self.assertTrue((ledger.directory/"operational-stop.json").exists())

    def test_wrong_stage_and_a3_one_second_reuse_are_not_fresh_ingestion(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        with self.assertRaises(ValueError):
            self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data},cfg=SearchConfiguration("incumbent",seconds=3.))
        with patch.object(ledger,"_stage_configs",return_value=(self.cfg,)):
            with patch.object(plan,"revalidate",side_effect=AssertionError("one-second A3 cannot reread owner")):
                with self.assertRaisesRegex(ValueError,"must reuse"):
                    self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data},stage="A3")

    def test_zero_does_not_block_other_configuration_evaluator_freeze(self):
        f=self.fixture();selected,reader=self.selector(f,zero=True)
        hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
        other,other_reader=self.selector(f,cfg=hp)
        ledger=self.ledger([(f,selected,reader),(f,other,other_reader)])
        plan,data=self.plan(f,selected,reader,extras=((other,other_reader),))
        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data},cfg=hp)
        for cfg in ledger._stage_configs("A2"):
            if cfg.arm=="reference":
                ledger.record_stage("A2",cfg.identity,{},accepted_units=0,status="UNSTARTED",evidence_sha256="a"*64)
        frozen=ledger.freeze_evaluators()
        self.assertEqual(dict(frozen["evaluators"])["incumbent"],"hp_fraction")
        self.assertIsNone(dict(frozen["evaluators"])["reference"])
        report=ledger.report_stage("A2")
        for row in report["configurations"]:
            self.assertEqual((row["root_slots"],row["source_seeds"]),(200,32))
        self.assertFalse(report["collection_authorized"])

    def test_capped_actual_owned_results_remain_unknown(self):
        f=self.fixture();selected,reader=self.selector(f,action=1)
        ledger=self.ledger([(f,selected,reader)])
        plan,data=self.plan(f,selected,reader,outcome=True,job_class=CappedSyntheticJob)
        self.assertTrue(all(row["status"]=="CAPPED" for row in data[1]["outcomes"]["outcomes"]))
        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        root=self.evidence()["root_evidence"][plan.root_id]
        self.assertEqual(root["uncertain_action_cells"],16)
        self.assertEqual(root["root_intervals"][self.cfg.identity],[-1.,1.])

    def test_late_owner_receipt_inventory_and_readout_mutations_refuse(self):
        for mutation in ("receipt","inventory","readout","reader","result","job"):
            with self.subTest(mutation=mutation):
                f=self.fixture();g=self.fixture(1)
                selected,reader=self.selector(f);other,other_reader=self.selector(g)
                ledger=self.ledger([(f,selected,reader),(g,other,other_reader)])
                plan,data=self.plan(f,selected,reader,outcome=True)
                later,later_data=self.plan(g,other,other_reader)
                actual=later._read_selector
                def mutate(cfg,row,binding):
                    value=actual(cfg,row,binding)
                    if mutation=="receipt":(reader.directory/"terminal.json").write_text("{}")
                    if mutation=="inventory":(reader.directory/"extra-receipt.json").write_text("{}")
                    if mutation=="readout":reader.last_validation["canary"]="changed"
                    if mutation=="reader":reader.directory=other_reader.directory
                    if mutation=="result":data[1]["resource_cost_complete"]=True
                    if mutation=="job":data[0].actions["raw"]=1
                    return value
                with patch.object(later,"_read_selector",side_effect=mutate):
                    with self.assertRaises(ValueError):
                        self.ingest(ledger,{plan.root_id:plan,later.root_id:later},
                            {plan.root_id:data,later.root_id:later_data})
                self.assertTrue((ledger.directory/"operational-stop.json").exists())
                self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_contract_full_transport_and_unsupported_extra_config_refuse(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        reader.timed_public_request_sha="f"*64
        with self.assertRaisesRegex(ValueError,"frozen root evaluation"):
            self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        for mutation in ("contract","oracle","fixed"):
            with self.subTest(mutation=mutation):
                f=self.fixture();selected,reader=self.selector(f)
                ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
                if mutation=="contract":plan.contract["namespace"]="changed"
                elif mutation=="oracle":plan.configurations+=(SearchConfiguration("incumbent",belief="oracle",seconds=1.),)
                else:plan.configurations+=(SearchConfiguration("incumbent",leaf="raw_rollout",seconds=1.),)
                with patch.object(plan,"revalidate",side_effect=AssertionError("structural mismatch cannot validate owner")):
                    with self.assertRaises(ValueError):self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
                self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-evidence-claim.json")).exists())

    def test_post_final_hook_review_drift_prevents_success_write(self):
        from pokezero.mcts_eval.search_over_raw_root_plan import _OwnedStageInputs
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        actual=_OwnedStageInputs.finalize
        def mutate(inputs,rows):
            result=actual(inputs,rows)
            self.reviews[self.cfg.identity].write_text("{}")
            return result
        with patch.object(_OwnedStageInputs,"finalize",mutate):
            with self.assertRaisesRegex(ValueError,"bound input drift"):
                self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_digest_callbacks_cannot_attach_owned_scope_to_generic_api(self):
        from pokezero.mcts_eval.search_over_raw_root_plan import _OwnedStageInputs
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        inputs=_OwnedStageInputs(ledger,"A2",self.cfg.identity,{plan.root_id:plan},{plan.root_id:data})
        with self.assertRaisesRegex(ValueError,"validation roster changed"):
            ledger.record_stage_evidence("A2",self.cfg.identity,inputs.bundles,
                public_bindings=inputs.public_bindings,runtime_hashes=inputs.runtime_hashes,
                qualification_review_path=self.reviews[self.cfg.identity],
                validate_selector=lambda cfg,row,binding:digest(row),validate_outcome=lambda row,law:None,
                _owned_stage_inputs=inputs)
        self.assertTrue((ledger.directory/"operational-stop.json").exists())

    def test_zero_configuration_is_local_unqualified_and_duplicate_claim_refuses(self):
        f=self.fixture();selected,reader=self.selector(f,zero=True)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        row=self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.assertEqual(row["summary"]["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(self.evidence()["root_intervals"],{})
        self.assertFalse((ledger.directory/"operational-stop.json").exists())
        with patch.object(plan,"revalidate",side_effect=AssertionError("duplicate must not invoke owner")):
            with self.assertRaises(FileExistsError):self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})

    def test_bad_attempt_poison_follows_claim_and_no_stage_success(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader,outcome=True)
        job,result,owner=data;result.pop("outcomes")
        claim=ledger.directory/"A2"/(self.cfg.identity+"-evidence-claim.json")
        actual=plan.revalidate
        def after_claim():
            self.assertTrue(claim.exists());return actual()
        with patch.object(plan,"revalidate",side_effect=after_claim):
            with self.assertRaises(ValueError):self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_malformed_actions_are_rejected_after_durable_claim(self):
        for kind in ("selector","producer"):
            for value in ([],{},True):
                with self.subTest(kind=kind,value=value):
                    f=self.fixture();selected,reader=self.selector(f)
                    ledger=self.ledger([(f,selected,reader)])
                    plan,data=self.plan(f,selected,reader,outcome=kind=="producer")
                    if kind=="selector":plan.selections[self.cfg.identity]["action"]=value
                    else:data[1]["outcomes"]["outcomes"][0]["action"]=value
                    with self.assertRaises(ValueError):
                        self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
                    self.assertTrue((ledger.directory/"A2"/(self.cfg.identity+"-evidence-claim.json")).exists())
                    self.assertTrue((ledger.directory/"operational-stop.json").exists())
                    self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-root-evidence.json")).exists())

    def test_later_selector_cannot_reset_earlier_readout_baseline(self):
        f=self.fixture();selected,reader=self.selector(f)
        ledger=self.ledger([(f,selected,reader)]);plan,data=self.plan(f,selected,reader)
        from pokezero.mcts_eval.search_over_raw_root_plan import _OwnedStageInputs
        actual=_OwnedStageInputs.validate_selector
        def mutate(inputs,cfg,row,binding):
            result=actual(inputs,cfg,row,binding)
            if cfg==self.cfg:f.reader.last_validation["canary"]="changed-after-earlier-return"
            return result
        with patch.object(_OwnedStageInputs,"validate_selector",mutate):
            with self.assertRaisesRegex(ValueError,"earlier owned readout changed"):
                self.ingest(ledger,{plan.root_id:plan},{plan.root_id:data})
        self.assertTrue((ledger.directory/"operational-stop.json").exists())

    def test_later_root_cannot_mutate_earlier_archive_or_owner(self):
        f=self.fixture();g=self.fixture(1)
        selected,reader=self.selector(f);other,other_reader=self.selector(g)
        ledger=self.ledger([(f,selected,reader),(g,other,other_reader)])
        # Both fixtures use the same authored checkpoint bytes/runtime.
        plan,data=self.plan(f,selected,reader);later,later_data=self.plan(g,other,other_reader)
        actual=later._read_selector
        def mutate(cfg,row,binding):
            result=actual(cfg,row,binding)
            plan.archive._private[0].audit_only["canary"]="changed-by-later-root"
            return result
        with patch.object(later,"_read_selector",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"inputs/evidence changed"):
                self.ingest(ledger,{plan.root_id:plan,later.root_id:later},
                    {plan.root_id:data,later.root_id:later_data})
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A2"/(self.cfg.identity+"-root-evidence.json")).exists())


class CappedSyntheticJob(outcome_fixture.SyntheticContinuationJob):
    def _components(self):
        env,evaluator=super()._components();original=env.terminal
        def capped():
            terminal=original()
            return None if terminal is None else SimpleNamespace(capped=True,winner=None)
        env.terminal=capped
        return env,evaluator


if __name__=="__main__":unittest.main()
