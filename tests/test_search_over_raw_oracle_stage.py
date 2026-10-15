"""Authored archive/original teams and disposable owners only; no science."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.env import BattleStartOverride, TerminalState
from pokezero.showdown_fixture import pack_team
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger
from pokezero.mcts_eval.search_over_raw_oracle import TeamOracle, opening_team
from pokezero.mcts_eval.search_over_raw_source import source_roots
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
from pokezero.mcts_eval.search_over_raw_root_plan import _OwnedStageInputs
from pokezero.mcts_eval.search_over_raw_native_oracle_run import NativeOracleTimedRunContract, NativeOracleTimedRunValidator
from pokezero.mcts_eval.search_over_raw_reference_oracle_run import ReferenceOracleTimedRunContract, ReferenceOracleTimedRunValidator
from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness, _state_payload
import test_search_over_raw_oracle_preparation as truth_fixture
import test_search_over_raw_native_oracle as native_truth
import test_search_over_raw_native_oracle_run as native_fixture
import test_search_over_raw_reference_oracle_run as reference_fixture
import test_search_over_raw_root_plan as plan_fixture
import test_search_over_raw_owned_stage as stage_fixture


class OracleStageTests(unittest.TestCase):
    def fixture(self, seed=0):
        t=truth_fixture.OraclePreparationTests();t.setUp();self.addCleanup(t.doCleanups)
        a=t.f;a.context=replace(t.context,seed=seed,
            trajectory=BattleTrajectory("authored-boundary","gen3randombattle",seed))
        if seed:
            # Genuinely different original team, not a changed SHA assertion.
            t.snapshot.first_requests["p2"]["side"]["pokemon"][1]["item"]="Oran Berry"
        a.archive=SealedSourceArchive();self.addCleanup(a.archive.close)
        a.archive.capture_public(a.context,0)
        a.archive.capture_private(SimpleNamespace(requested_players=("p1",),decision_round_index=0,
            seed=seed,battle_id=a.context.battle_id,decisions={"p1":SimpleNamespace(action_index=0)},
            snapshot=t.snapshot))
        trajectory=BattleTrajectory("authored-boundary","gen3randombattle",seed)
        trajectory.append(TrajectoryStep("p1",0,a.context.observation,a.context.observation.legal_action_mask,0))
        trajectory.record_terminal(TerminalState("p1",1))
        a.source=source_roots(a.contract,panel="exploration",source_seed=seed,trajectory=trajectory)
        a.root_id=a.source["roots"][0]["root_id"]
        t.context=a.context;t.archive=a.archive;t.oracle=TeamOracle.capture(t.snapshot,t.request,t.source)
        a.setUp=lambda:None
        p=plan_fixture.RootPlanTests()
        with patch.object(plan_fixture,"archive_fixture_module",SimpleNamespace(ExplorationRootCrosswalkTests=lambda:a)):
            p.setUp()
        self.addCleanup(p.doCleanups)
        p.truth=t
        return p

    def selector(self,p,arm="incumbent",zero=False):
        t=p.truth;o=p.owner_fixture;witness=p.bound["public_request_witness"]
        cfg=SearchConfiguration(arm,belief="oracle",seconds=1.,workers=20 if arm == "reference" else 1)
        catalog=digest(_state_payload(t.source));payload=digest(_state_payload(t.oracle))
        if arm == "incumbent":
            truth=native_truth.NativeOracleTests();truth.f=t;truth.witness=witness
            truth.override=BattleStartOverride(player_teams={"p1":pack_team(opening_team(t.request.state.self_initial_request,"p1",t.source)),
                "p2":pack_team(t.oracle.opponent_team)},observation_format_id="gen3randombattle")
            truth.policy=SimpleNamespace(_fixed_override=truth.override,_set_source=t.source);truth.catalog_sha=catalog
            helper=native_fixture.NativeOracleRunTests();helper.truth=truth;helper.cfg=cfg
            runtime=native_fixture.observed_runtime(cfg,o.files[str(o.checkpoint)])
            runtime["team_oracle"]=t.oracle.receipt();helper.runtime=runtime
            helper.inputs=dict(decision_round=0,consumed_binding=witness["consumed_binding"])
            selected=helper.selected(zero=zero)
            selected.update(root_id=p.archive_fixture.root_id+":"+cfg.identity,
                statistics_root_id=p.archive_fixture.root_id,selection_seed=p.bound["slot"]["source_seed"])
            if zero:
                selected["strict_engine_diagnostic"]["battle_id"]="search-over-raw-root:"+p.archive_fixture.root_id
                selected["original_failure_receipt"].update(root_id=selected["root_id"],
                    engine_fallback_diagnostic=deepcopy(selected["strict_engine_diagnostic"]))
            job=native_fixture.AuthoredOracleOwner(selected)
            klass=NativeOracleTimedRunValidator;contract=NativeOracleTimedRunContract();kind="native-oracle-timed-run"
            extra=dict(expected_override_sha256=digest(asdict(truth.override)))
        else:
            runtime,factory=reference_fixture.owner_fixture.owner_runtime(cfg,str(o.checkpoint),str(o.showdown),o.files[str(o.checkpoint)])
            factory=replace(factory,set_source_hash=t.source.metadata.source_hash)
            runtime.update(reference_factory=asdict(factory),set_source_hash=t.source.metadata.source_hash,
                full_public_request_observation=deepcopy(reference_fixture.TIMED_REFERENCE_PUBLIC_CAPTURE),
                reference_team_oracle_observation=deepcopy(reference_fixture.REFERENCE_ORACLE_CAPTURE),
                team_oracle=t.oracle.receipt(),oracle_payload_sha256=payload,oracle_reconstruction_catalog_sha256=catalog)
            job=reference_fixture.FullEmptyOracleOwner(cfg,runtime,t.request,p.archive_fixture.root_id,
                p.bound["slot"]["source_seed"],t.oracle,t.source)
            klass=ReferenceOracleTimedRunValidator;contract=ReferenceOracleTimedRunContract();kind="reference-oracle-timed-run"
            extra=dict(expected_oracle_payload_sha256=payload)
        directory=o.root/("A1-"+arm)
        reader=klass(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256="a"*64,
            configuration=cfg,root_id=p.archive_fixture.root_id+":"+cfg.identity,statistics_root_id=p.archive_fixture.root_id,
            selection_seed=p.bound["slot"]["source_seed"],runtime_configuration=runtime,runtime_sha256=digest(runtime),
            consumed_binding=witness["consumed_binding"],legal_action_mask=p.mask,decision_round=0,
            file_hashes=o.files,checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
            showdown_root=str(o.showdown),allow_synthetic=True,public_request_sha256=witness["public_request_sha256"],
            public_request=witness,oracle_receipt=t.oracle.receipt(),expected_reconstruction_catalog_sha256=catalog,
            contract=contract,**extra)
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,verify_admission=lambda:None,
            contract=contract,kind=kind,work_phase="selection",validate_result=reader._read_work)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        return terminal["result"],reader

    def ledger(self,items,mutate_review=None,public_items=()):
        cfg=items[0][2].configuration
        bindings={p.archive_fixture.root_id:p.bound["public_record_sha256"] for p,_,_ in items}
        runtimes={p.archive_fixture.root_id:{"raw":p.reader.runtime_sha,cfg.identity:r.runtime_sha} for p,_,r in items}
        review=dict(status="CLEAR_UNQUALIFIED_AT_CEILING" if all(s["action"] is None for _,s,_ in items)
            else "CLEAR_CONFIGURATION_AT_CEILING",configuration_sha256=cfg.identity,
            runtime_binding_schema="pokezero.search-over-raw.a1-original-team-root-runtimes.v1",
            runtime_by_root_sha256=digest(runtimes),public_bindings_sha256=digest(bindings),
            atomic_work_and_cleanup_validated=True,zero_work_is_local_unqualified=True)
        if mutate_review:mutate_review(review,runtimes)
        path=items[0][0].owner_fixture.root/"A1-review.json";path.write_text(json.dumps(review))
        hashes=controller_input_hashes();hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.public_reviews={}
        for p,s,r in public_items:
            public_path=p.owner_fixture.root/(r.configuration.identity+"-public-review.json")
            public_review=dict(status="CLEAR_CONFIGURATION_AT_CEILING",configuration_sha256=r.configuration.identity,
                runtime_sha256=r.runtime_sha,public_bindings_sha256=digest(bindings),
                atomic_work_and_cleanup_validated=True,zero_work_is_local_unqualified=True)
            public_path.write_text(json.dumps(public_review))
            hashes[str(public_path)]=hashlib.sha256(public_path.read_bytes()).hexdigest()
            self.public_reviews[r.configuration.identity]=public_path
        contract=dict(items[0][0].archive_fixture.contract,input_hashes=hashes)
        ledger=PhaseALedger.create(items[0][0].owner_fixture.root/"A1-ledger",contract)
        for p,_,_ in items:
            p.archive_fixture.contract=deepcopy(contract);p.archive_fixture.source["contract_sha256"]=digest(contract)
        return ledger,path

    def plan(self,p,s,r):
        return p.plan(configurations=(p.cfg,r.configuration),selections={"raw":p.selected,r.configuration.identity:s},
            selector_readers={"raw":p.reader,r.configuration.identity:r},oracle_set_source=p.truth.source)

    def ingest(self,ledger,review,plans,cfg,stage="A1"):
        return ledger.record_owned_stage_evidence(stage,cfg.identity,plans=plans,
            continuations={root:None for root in plans},qualification_review_path=review)

    def test_two_original_teams_have_reviewed_distinct_runtime_and_uncertain_fixed_roster(self):
        items=[]
        for seed in (0,1):
            p=self.fixture(seed);s,r=self.selector(p);items.append((p,s,r))
        self.assertNotEqual(items[0][0].truth.oracle.opponent_team_sha256,items[1][0].truth.oracle.opponent_team_sha256)
        self.assertNotEqual(items[0][2].runtime_sha,items[1][2].runtime_sha)
        ledger,review=self.ledger(items);plans={p.archive_fixture.root_id:self.plan(p,s,r) for p,s,r in items}
        with patch("subprocess.Popen") as launch:
            row=self.ingest(ledger,review,plans,items[0][2].configuration)
        launch.assert_not_called()
        evidence=json.loads((ledger.directory/"A1"/(items[0][2].configuration.identity+"-root-evidence.json")).read_text())
        self.assertEqual(len(evidence["selector_runtime_by_root"]),2)
        self.assertEqual(len(ledger.contract["panels"]["exploration"]["root_slots"]),200)
        self.assertLess(row["summary"]["lower_mean"],0.)
        self.assertEqual(row["summary"]["accepted_units"],32)
        self.assertTrue(evidence["owned_input_handoff"]["original_team_archive_correspondence_validated"])
        self.assertFalse(evidence["owned_input_handoff"]["collection_authorized"])
        self.assertNotIn("PRIVATE_SNAPSHOT",json.dumps(evidence))

    def test_reference_twenty_worker_zero_ingests_as_local_unqualified(self):
        p=self.fixture();s,r=self.selector(p,"reference",zero=True)
        ledger,review=self.ledger([(p,s,r)]);plan=self.plan(p,s,r)
        row=self.ingest(ledger,review,{plan.root_id:plan},r.configuration)
        self.assertIsNone(s["action"]);self.assertFalse(s["retry_authorized"])
        self.assertEqual(r.last_validation["actual_work"]["original_team_consumption"]["prepared_workers"],20)
        self.assertEqual(plan.revalidate()["continuation_cells"],[dict(action=0,replicate=i) for i in range(8)])
        self.assertEqual(row["summary"]["status"],"UNQUALIFIED_AT_CEILING")

    def test_native_zero_does_not_require_continuation_or_qualify_runtime(self):
        p=self.fixture();s,r=self.selector(p,zero=True)
        ledger,review=self.ledger([(p,s,r)]);plan=self.plan(p,s,r)
        row=self.ingest(ledger,review,{plan.root_id:plan},r.configuration)
        self.assertEqual(row["summary"]["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(plan.revalidate()["root_evidence"]["selector_statuses"][r.configuration.identity],"UNQUALIFIED_AT_CEILING")

    def test_swapped_review_runtime_assignments_refuse_before_stage_success(self):
        items=[]
        for seed in (0,1):
            p=self.fixture(seed);s,r=self.selector(p);items.append((p,s,r))
        def swap(review,runtimes):
            roots=list(runtimes);cfg=items[0][2].configuration.identity
            runtimes[roots[0]][cfg],runtimes[roots[1]][cfg]=runtimes[roots[1]][cfg],runtimes[roots[0]][cfg]
            review["runtime_by_root_sha256"]=digest(runtimes)
        ledger,review=self.ledger(items,swap);plans={p.archive_fixture.root_id:self.plan(p,s,r) for p,s,r in items}
        with self.assertRaises(ValueError):self.ingest(ledger,review,plans,items[0][2].configuration)
        self.assertFalse((ledger.directory/"A1"/(items[0][2].configuration.identity+"-root-evidence.json")).exists())

    def test_full_truth_catalog_and_exact_reader_scope_cannot_be_substituted(self):
        p=self.fixture();s,r=self.selector(p)
        with self.assertRaises(ValueError):
            p.plan(configurations=(p.cfg,r.configuration),selections={"raw":p.selected,r.configuration.identity:s},
                selector_readers={"raw":p.reader,r.configuration.identity:r})
        original=r.catalog_sha;r.catalog_sha="f"*64
        with self.assertRaises(ValueError):self.plan(p,s,r)
        r.catalog_sha=original
        plan=self.plan(p,s,r)
        p.truth.source.move_metadata["tackle"]={"authored":"drift"}
        # The plan privately froze the source; actual archive remains bound to
        # its separately captured catalog. Internal catalog substitution fails.
        plan._oracle_source.move_metadata["tackle"]={"authored":"drift"}
        with self.assertRaises(ValueError):plan.revalidate()

    def test_final_truth_extraction_cannot_mutate_earlier_owner_receipts(self):
        p=self.fixture();s,r=self.selector(p);plan=self.plan(p,s,r)
        original=plan._check_original_team;calls=0
        def late():
            nonlocal calls
            original();calls+=1
            if calls == 3:(p.reader.directory/"terminal.json").write_text("{}")
        with patch.object(plan,"_check_original_team",side_effect=late):
            with self.assertRaises((ValueError,KeyError)):plan.revalidate()

    def test_per_root_truth_exception_does_not_permit_raw_or_recipe_drift(self):
        items=[]
        for seed in (0,1):
            p=self.fixture(seed);s,r=self.selector(p);items.append((p,s,r))
        ledger,_=self.ledger(items);plans={p.archive_fixture.root_id:self.plan(p,s,r) for p,s,r in items}
        last=items[-1];last[0].reader.runtime_sha="f"*64
        with self.assertRaisesRegex(ValueError,"common raw runtime"):
            _OwnedStageInputs(ledger,"A1",last[2].configuration.identity,plans,{root:None for root in plans})
        last[0].reader.runtime_sha=items[0][0].reader.runtime_sha
        last[2].runtime["unreviewed_search_change"]=True
        with self.assertRaisesRegex(ValueError,"only in original-team truth"):
            _OwnedStageInputs(ledger,"A1",last[2].configuration.identity,plans,{root:None for root in plans})

    def test_oracle_plan_is_never_an_A2_arm(self):
        p=self.fixture();s,r=self.selector(p);ledger,review=self.ledger([(p,s,r)]);plan=self.plan(p,s,r)
        with patch.object(plan,"revalidate",side_effect=AssertionError("A2 cannot touch original-team owner")):
            with self.assertRaises(ValueError):self.ingest(ledger,review,{plan.root_id:plan},r.configuration,stage="A2")

    def test_local_oracle_zero_does_not_block_shared_public_A1_A2_freeze(self):
        p=self.fixture();s,r=self.selector(p,zero=True)
        helper=stage_fixture.OwnedStageTests();helper.setUp();self.addCleanup(helper.doCleanups)
        public,public_reader=helper.selector(p)
        ledger,review=self.ledger([(p,s,r)],public_items=((p,public,public_reader),))
        oracle_plan=self.plan(p,s,r)
        public_plan=helper.plan(p,public,public_reader)[0]
        public_review=self.public_reviews[public_reader.configuration.identity]
        self.ingest(ledger,review,{oracle_plan.root_id:oracle_plan},r.configuration)
        self.ingest(ledger,public_review,{public_plan.root_id:public_plan},public_reader.configuration)
        self.ingest(ledger,public_review,{public_plan.root_id:public_plan},public_reader.configuration,stage="A2")
        for cfg in ledger._stage_configs("A2"):
            if cfg != public_reader.configuration:
                ledger.record_stage("A2",cfg.identity,{},accepted_units=0,status="UNSTARTED",evidence_sha256="a"*64)
        frozen=ledger.freeze_evaluators()
        self.assertEqual(dict(frozen["evaluators"])["incumbent"],"model")
        self.assertIsNone(dict(frozen["evaluators"])["reference"])
        for cfg in ledger._stage_configs("A1"):
            if cfg not in (r.configuration,public_reader.configuration):
                ledger.record_stage("A1",cfg.identity,{},accepted_units=0,status="UNSTARTED",evidence_sha256="a"*64)
        self.assertFalse(ledger.report_stage("A1")["collection_authorized"])

    def test_generic_map_roster_and_public_scope_are_exact_before_callbacks(self):
        p=self.fixture();s,r=self.selector(p)
        ledger,review=self.ledger([(p,s,r)]);plan=self.plan(p,s,r)
        inputs=_OwnedStageInputs(ledger,"A1",r.configuration.identity,{plan.root_id:plan},{plan.root_id:None})
        for change in ("missing","extra","extra-selector","different-raw"):
            runtime=deepcopy(inputs.runtime_hashes)
            if change == "missing":runtime.clear()
            elif change == "extra":runtime["extra-root"]=deepcopy(runtime[plan.root_id])
            elif change == "extra-selector":runtime[plan.root_id]["extra-selector"]="a"*64
            else:
                # Direct API cannot excuse a changed raw policy across roots.
                runtime["extra-root"]={"raw":"f"*64,r.configuration.identity:r.runtime_sha}
            with self.subTest(change=change),patch.object(plan,"revalidate",side_effect=AssertionError("invalid admission cannot read owner")):
                with self.assertRaises(ValueError):
                    ledger.record_stage_evidence("A1",r.configuration.identity,inputs.bundles,
                        public_bindings=inputs.public_bindings,runtime_hashes=runtime,qualification_review_path=review,
                        validate_selector=inputs.validate_selector,validate_outcome=inputs.validate_outcome)
        public_cfg=SearchConfiguration("incumbent",seconds=1.)
        # Even a prospectively hashed review cannot widen a public arm's scope.
        with self.assertRaisesRegex(ValueError,"public configuration cannot adopt"):
            ledger.record_stage_evidence("A2",public_cfg.identity,inputs.bundles,public_bindings=inputs.public_bindings,
                runtime_hashes=inputs.runtime_hashes,qualification_review_path=review,
                validate_selector=lambda *a:None,validate_outcome=lambda *a:None)

    def test_later_root_cannot_mutate_earlier_truth_or_owned_bytes(self):
        items=[]
        for seed in (0,1):
            p=self.fixture(seed);s,r=self.selector(p);items.append((p,s,r))
        ledger,review=self.ledger(items);plans={p.archive_fixture.root_id:self.plan(p,s,r) for p,s,r in items}
        first,second=plans.values();actual=second._read_selector
        def mutate(cfg,row,binding):
            result=actual(cfg,row,binding)
            first._oracle_source.move_metadata["tackle"]={"later_root":"drift"}
            return result
        with patch.object(second,"_read_selector",side_effect=mutate):
            with self.assertRaises(ValueError):self.ingest(ledger,review,plans,items[0][2].configuration)
        self.assertTrue((ledger.directory/"operational-stop.json").exists())
        self.assertFalse((ledger.directory/"A1"/(items[0][2].configuration.identity+"-root-evidence.json")).exists())


if __name__ == "__main__":unittest.main()
