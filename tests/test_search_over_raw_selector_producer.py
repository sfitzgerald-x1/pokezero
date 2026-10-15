"""Actual disposable selector owners, fresh authored archives; no science."""
from copy import deepcopy
from dataclasses import dataclass,asdict,replace,fields
import pickle
import time
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration,digest
from pokezero.mcts_eval.search_over_raw_raw_run import RawSelectionJob,raw_runtime,RawSelectionRunContract
from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedAdapterJob
from pokezero.mcts_eval.search_over_raw_reference_timed_run import ReferenceTimedAdapterJob
from pokezero.mcts_eval.search_over_raw_native_oracle_run import NativeOracleTimedAdapterJob
from pokezero.mcts_eval.search_over_raw_reference_oracle_run import ReferenceOracleTimedAdapterJob,REFERENCE_ORACLE_CAPTURE
from pokezero.mcts_eval.search_over_raw_selector_producer import produce_archived_selector
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
from pokezero.mcts_eval.search_over_raw_continuation_admission import ContinuationAdmissionJournal
from pokezero.mcts_eval.search_over_raw_continuation_producer import produce_reserved_continuation
from pokezero.mcts_eval.search_over_raw_world_inputs import TIMED_REFERENCE_PUBLIC_CAPTURE,_state_payload
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
import pokezero.mcts_eval.search_over_raw_selector_producer as producer
import test_search_over_raw_configuration_recipe as recipe_fixture
import test_search_over_raw_archive as archive_fixture
import test_search_over_raw_reconciled as ledger_fixture
import test_search_over_raw_raw_run as raw_fixture
import test_search_over_raw_root_plan as plan_fixture
import test_search_over_raw_owned_stage as stage_fixture
import test_search_over_raw_reference_timed_run as reference_fixture
import test_search_over_raw_reference_public as reference_public
import test_search_over_raw_oracle_stage as oracle_fixture
import test_search_over_raw_reference_oracle_run as reference_oracle
import test_search_over_raw_outcome_run as outcome_fixture


@dataclass(frozen=True,kw_only=True)
class AuthoredRawJob(RawSelectionJob):
    authored_selection: dict
    fail: bool=False
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup");phase("selection");time.sleep(.03)
        if self.fail:raise RuntimeError("authored raw failure")
        phase("cleanup");return deepcopy(self.authored_selection)


@dataclass(frozen=True,kw_only=True)
class AuthoredNativeJob(NativeTimedAdapterJob):
    authored_selection: dict
    fail: bool=False
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup");phase("selection");time.sleep(max(.03,self.authored_selection["elapsed_seconds"]+.01))
        if self.fail:raise RuntimeError("authored selector failure")
        phase("cleanup");return deepcopy(self.authored_selection)


@dataclass(frozen=True,kw_only=True)
class AuthoredNativeOracleJob(NativeOracleTimedAdapterJob):
    authored_selection: dict
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup");phase("selection");time.sleep(.03)
        phase("cleanup");return deepcopy(self.authored_selection)


@dataclass(frozen=True,kw_only=True)
class AuthoredReferenceJob(ReferenceTimedAdapterJob):
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        return reference_public.FullEmptyReferenceOwner(self.configuration,self.expected_runtime_configuration,
            self._public_request,self.statistics_root_id,self.selection_seed)(ownership,phase)


@dataclass(frozen=True,kw_only=True)
class AuthoredReferenceOracleJob(ReferenceOracleTimedAdapterJob):
    authored_source: object
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        return reference_oracle.FullEmptyOracleOwner(self.configuration,self.expected_runtime_configuration,
            self._public_request,self.statistics_root_id,self.selection_seed,self.oracle,self.authored_source)(ownership,phase)


@dataclass(frozen=True)
class AuthoredCheckpoint:
    checkpoint_path: str
    checkpoint_sha256: str
    showdown_source_sha256: str


class SelectorProducerTests(unittest.TestCase):
    def setUp(self):
        self.c=recipe_fixture.ConfigurationRecipeTests();self.c.setUp();self.addCleanup(self.c.doCleanups)
        self.h=self.c.h;self.cfg=self.c.cfg;self.stage="A2"
        self.o=raw_fixture.RawOwnerTests();self.o.setUp();self.addCleanup(self.o.doCleanups)

    def initialize(self,*,cfg=None,stage="A2",oracle=False,continuation=False):
        self.cfg=cfg or self.cfg;self.stage=stage;self.c.cfg=self.cfg;self.c.stage=stage
        self.factory=None;self.set_source=None
        if self.cfg.arm == "reference":
            runtime,factory=reference_fixture.owner_runtime(self.cfg,str(self.o.checkpoint),str(self.o.showdown),
                self.o.files[str(self.o.checkpoint)])
            factory=replace(factory,set_source_hash="a"*64)
            runtime.update(reference_factory=asdict(factory),set_source_hash="a"*64,
                full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
            if oracle:
                # Catalog is a static empty authored fixture, known before roots.
                from pokezero.randbat import Gen3RandbatSource,RandbatSourceMetadata
                static_source=Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle",3,None,None,None,"a"*64),{})
                runtime.update(reference_team_oracle_observation=deepcopy(REFERENCE_ORACLE_CAPTURE),
                    oracle_reconstruction_catalog_sha256=digest(_state_payload(static_source)))
            self.factory=factory;self.c.recipe=lambda cfg=None:deepcopy(runtime)
        self.c.initialize()
        original=ledger_fixture.ReconciledLedgerTest.setUp
        def static_recipe(base):original(base);base.contract=deepcopy(self.h.contract)
        if oracle:
            helper=oracle_fixture.OracleStageTests();self.addCleanup(helper.doCleanups)
            # Suppress only its incidental fixture raw supervisor. The producer
            # below owns the first actual raw worker in the fresh namespace.
            with patch.object(ledger_fixture.ReconciledLedgerTest,"setUp",static_recipe),patch.object(
                    plan_fixture,"_supervise_owned_job",return_value=dict(supervisor_exit_code=0)):
                self.f=helper.fixture()
            self.a=self.f.archive_fixture;self.set_source=self.f.truth.source
            self.f.owner_fixture=self.o
        else:
            self.a=archive_fixture.ExplorationRootCrosswalkTests()
            with patch.object(ledger_fixture.ReconciledLedgerTest,"setUp",static_recipe):self.a.setUp()
            self.addCleanup(self.a.doCleanups)
            if continuation:
                # New authored snapshot is installed before any binding,
                # selector owner or accepted baseline exists.
                index=self.a.source["roots"][0]["source_request_index"]
                self.a.archive._private[index]=deepcopy(self.h.outcome.snapshot)
            self.f=plan_fixture.RootPlanTests();self.f.archive_fixture=self.a;self.f.owner_fixture=self.o
            self.f.bound=self.a.bind();self.f.cfg=SearchConfiguration("raw",seconds=1.)
            self.f.mask=self.f.bound["public_evidence_binding"]["legal_action_mask"]
        self.f.runtime=raw_runtime(self.f.cfg,self.o.files[str(self.o.checkpoint)],"a"*64)
        self.checkpoint=AuthoredCheckpoint(str(self.o.checkpoint),self.o.files[str(self.o.checkpoint)],"a"*64)
        self.search_runtime=self.c.recipe()
        self.authoring_helper=None
        if oracle:
            helper=oracle_fixture.OracleStageTests();self.addCleanup(helper.doCleanups)
            with patch.object(oracle_fixture,"_supervise_owned_job",side_effect=lambda job,**kwargs:
                    dict(supervisor_exit_code=0,result=deepcopy(job.selected))):
                # Native helper returns its authored selected packet separately;
                # reference's packet is made inside the actual twenty-worker job.
                if self.cfg.arm == "incumbent":
                    selected,reader=helper.selector(self.f)
                    self.search_runtime=deepcopy(reader.runtime);self.authored_selected=selected
            if self.cfg.arm == "reference":
                self.search_runtime.update(team_oracle=self.f.truth.oracle.receipt(),
                    oracle_payload_sha256=digest(_state_payload(self.f.truth.oracle)))

    def raw_packet(self):
        bound=self.f.bound;witness=bound["public_request_witness"]
        legal=[i for i,v in enumerate(self.f.mask) if v]
        return dict(root_id=self.a.root_id+":raw",statistics_root_id=self.a.root_id,
            selection_seed=bound["slot"]["source_seed"],configuration_sha256=self.f.cfg.identity,
            runtime_sha256=digest(self.f.runtime),action=legal[0],status="SELECTED",elapsed_seconds=.005,
            nominal_search_seconds=1.,exceeded_nominal_seconds=False,independent_root_statistics=True,
            scientific_strength_evidence=False,raw_selection_run_contract=RawSelectionRunContract().receipt(),
            observed_runtime_configuration=deepcopy(self.f.runtime),evidence=dict(selector="deterministic_masked_argmax",
                model_forwards=1,legal_actions=legal,priors=[1/len(legal)]*len(legal),signed_value=.1,
                consumed_binding=deepcopy(witness["consumed_binding"]),public_request_witness=deepcopy(witness)))

    def native_packet(self,zero=False):
        helper=stage_fixture.OwnedStageTests();helper.setUp();self.addCleanup(helper.doCleanups)
        with patch.object(stage_fixture,"_supervise_owned_job",return_value=dict(supervisor_exit_code=0)):
            selected,_=helper.selector(self.f,cfg=self.cfg,zero=zero,action=1)
        return selected

    def produce(self,*,raw=False,zero=False,fail=False,prior=None,scientific=False):
        runtime=self.f.runtime if raw else self.search_runtime
        kwargs={}
        if raw:cls=AuthoredRawJob;kwargs=dict(authored_selection=self.raw_packet(),fail=fail)
        elif self.cfg.arm == "reference":
            cls=AuthoredReferenceOracleJob if self.cfg.belief == "oracle" else AuthoredReferenceJob
            if self.cfg.belief == "oracle":kwargs=dict(authored_source=self.set_source)
        elif self.cfg.belief == "oracle":cls=AuthoredNativeOracleJob;kwargs=dict(authored_selection=self.authored_selected)
        else:cls=AuthoredNativeJob;kwargs=dict(authored_selection=self.native_packet(zero),fail=fail)
        return produce_archived_selector(self.h.ledger,self.stage,self.cfg.identity,
            archive=self.a.archive,source=self.a.source,root_id=self.a.root_id,checkpoint_contract=self.checkpoint,
            source_owner=getattr(self,"source_owner",None),
            showdown_root=str(self.o.showdown),runtime=runtime,file_hashes=self.o.files,
            qualification_review_path=self.c.review_path,history_review_path=self.h.review_path,
            selector="raw" if raw else "search",reference_factory=self.factory,set_source=self.set_source,
            prior_owner=prior,_synthetic_job_class=None if scientific else cls,
            _synthetic_job_kwargs=None if scientific else kwargs)

    def test_genuine_raw_and_native_chain_into_owned_stage_no_repeat(self):
        self.initialize();raw,_=self.produce(raw=True);search,_=self.produce()
        plan=ArchivedRootEvaluationPlan(self.a.archive,self.a.source,contract=self.h.contract,root_id=self.a.root_id,
            configurations=(self.f.cfg,self.cfg),selections={"raw":raw.selection,self.cfg.identity:search.selection},
            selector_readers={"raw":raw.reader,self.cfg.identity:search.reader},allow_synthetic=True)
        row=self.h.ledger.record_owned_stage_evidence(self.stage,self.cfg.identity,plans={plan.root_id:plan},
            continuations={plan.root_id:None},qualification_review_path=self.c.review_path)
        self.assertEqual(row["summary"]["lower_mean"],-1.)
        self.assertTrue(search.reader.last_validation["process_receipts_validated"])
        with self.assertRaises(FileExistsError):self.produce()
        with self.assertRaises(TypeError):pickle.dumps(raw)

    def test_exact_shared_raw_owner_reuse_launches_nothing(self):
        self.initialize();raw,_=self.produce(raw=True)
        with patch("subprocess.Popen") as launch:
            reused,receipt=self.produce(raw=True,prior=raw)
        launch.assert_not_called();self.assertIs(reused,raw);self.assertFalse(receipt["selector_launched"])

    def test_real_selector_reservation_continuation_and_stage_chain(self):
        self.initialize(continuation=True);raw,_=self.produce(raw=True);search,_=self.produce()
        plan=ArchivedRootEvaluationPlan(self.a.archive,self.a.source,contract=self.h.contract,root_id=self.a.root_id,
            configurations=(self.f.cfg,self.cfg),selections={"raw":raw.selection,self.cfg.identity:search.selection},
            selector_readers={"raw":raw.reader,self.cfg.identity:search.reader},allow_synthetic=True)
        journal=ContinuationAdmissionJournal.create(self.h.ledger,plan,history_review_path=self.h.review_path)
        bank,receipt=produce_reserved_continuation(self.h.ledger,self.stage,self.cfg.identity,plan=plan,
            journal=journal,config=self.h.outcome.config,runtime=self.h.outcome.runtime,
            _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        self.assertEqual(receipt["reserved_cells"],16)
        self.assertEqual(journal.consumed_actions(plan),(0,1))
        self.assertTrue(bank.reader.last_validation["process_receipts_validated"])
        with patch("subprocess.Popen") as launch:
            row=self.h.ledger.record_owned_stage_evidence(self.stage,self.cfg.identity,plans={plan.root_id:plan},
                continuations={plan.root_id:(bank,)},qualification_review_path=self.c.review_path)
        launch.assert_not_called();self.assertEqual(len(row["root_intervals"]),1)
        self.assertFalse(self.h.ledger.contract["collection_authorized"])

    def test_directory_existence_never_implies_reusable_owner(self):
        self.initialize();self.produce(raw=True)
        with patch("subprocess.Popen") as launch,self.assertRaises(FileExistsError):self.produce(raw=True)
        launch.assert_not_called()

    def test_prior_owner_mode_substitution_rejected_without_launch(self):
        self.initialize();owner,_=self.produce(raw=True);owner.reader.allow_synthetic=False
        with patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"exact original canonical"):
            self.produce(raw=True,prior=owner)
        launch.assert_not_called()

    def test_equal_content_reader_cannot_replace_original_reusable_owner(self):
        self.initialize();owner,_=self.produce(raw=True);owner.reader=deepcopy(owner.reader)
        with patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"original selector owner"):
            self.produce(raw=True,prior=owner)
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_reader_class_substitution_cannot_report_reuse(self):
        self.initialize();owner,_=self.produce(raw=True)
        class ForgedReader(type(owner.reader)):
            def __call__(self,selection):raise AssertionError("forged reader must never be invoked")
        owner.reader.__class__=ForgedReader
        with patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"exact selector job"):
            self.produce(raw=True,prior=owner)
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_final_archive_callback_cannot_replace_reused_reader(self):
        self.initialize();owner,_=self.produce(raw=True);original=producer.archived_exploration_root_binding
        def replace_reader(*args,**kwargs):
            value=original(*args,**kwargs);owner.reader=deepcopy(owner.reader);return value
        with patch.object(producer,"archived_exploration_root_binding",side_effect=replace_reader),\
                patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"selector owner changed"):
            self.produce(raw=True,prior=owner)
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_first_archive_callback_claims_before_source_drift_and_launch(self):
        self.initialize();original=producer.archived_exploration_root_binding
        def drift(*args,**kwargs):
            self.assertTrue((self.h.directory/"selector-producer-claims").is_dir())
            value=original(*args,**kwargs);self.a.source["unreviewed"]=True;return value
        with patch.object(producer,"archived_exploration_root_binding",side_effect=drift),\
                patch("subprocess.Popen") as launch,self.assertRaises(ValueError):self.produce()
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_synthetic_transport_cannot_replace_prepared_inputs(self):
        self.initialize();original=deepcopy
        def replace_fields(value):
            if type(value) is dict and "authored_selection" in value:return {**original(value),"context":"forged"}
            return original(value)
        with patch.object(producer,"deepcopy",side_effect=replace_fields),patch("subprocess.Popen") as launch,\
                self.assertRaisesRegex(ValueError,"cannot replace production"):
            self.produce()
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def substituted_job(self,*,raw=False,reference=False,oracle=False):
        if reference:self.initialize(cfg=SearchConfiguration("reference",belief="oracle" if oracle else "public",
            seconds=1.,workers=20),stage="A1" if oracle else "A2",oracle=oracle)
        else:self.initialize(cfg=SearchConfiguration("incumbent",belief="oracle" if oracle else "public",
            seconds=1.),stage="A1" if oracle else "A2",oracle=oracle)
        original=producer._route
        def route(cfg, runtime=None):
            cls,builder,reader,supervisor=original(cfg, runtime)
            def alternate(*args,**kwargs):
                job,bound=builder(*args,**kwargs)
                context,pending,_=self.a.archive.selected(self.a.source["roots"][0])
                values={f.name:getattr(job,f.name) for f in fields(job)}
                values.update(context=context,pending_transition=pending,selection_seed=job.selection_seed+1)
                wrong=cls(**values)
                # This alternate is internally well formed, not a broken guard.
                self.assertEqual(wrong.selection_seed,job.selection_seed+1)
                return wrong,bound
            return cls,alternate,reader,supervisor
        with patch.object(producer,"_route",side_effect=route),patch("subprocess.Popen") as launch,\
                self.assertRaisesRegex(ValueError,"original prospective job envelope"):
            self.produce(raw=raw)
        launch.assert_not_called();self.assertTrue((self.h.directory/"operational-stop.json").exists())

    def test_same_class_raw_alternate_cannot_launch(self):self.substituted_job(raw=True)
    def test_same_class_native_alternate_cannot_launch(self):self.substituted_job()
    def test_same_class_reference_alternate_cannot_launch(self):self.substituted_job(reference=True)
    def test_same_class_native_oracle_alternate_cannot_launch(self):self.substituted_job(oracle=True)
    def test_same_class_reference_oracle_alternate_cannot_launch(self):self.substituted_job(reference=True,oracle=True)

    def test_a3_one_second_selector_cannot_launch(self):
        self.initialize()
        with patch.object(self.h.ledger,"configuration_recipe_review",return_value=(self.c.review,b"unused")),\
                patch.object(self.h.ledger,"_stage_configs",return_value=(self.cfg,)),\
                patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"reuse frozen A2"):
            produce_archived_selector(self.h.ledger,"A3",self.cfg.identity,archive=self.a.archive,
                source=self.a.source,root_id=self.a.root_id,checkpoint_contract=self.checkpoint,
                showdown_root=str(self.o.showdown),runtime=self.search_runtime,file_hashes=self.o.files,
                qualification_review_path=self.c.review_path,history_review_path=self.h.review_path)
        launch.assert_not_called();self.assertFalse((self.h.directory/"selector-producer-claims").exists())

    def test_zero_owned_search_local_and_no_continuation(self):
        self.initialize();owner,receipt=self.produce(zero=True)
        self.assertEqual(receipt["status"],"UNQUALIFIED_AT_CEILING");self.assertIsNone(owner.selection["action"])
        self.assertFalse((self.h.directory/"operational-stop.json").exists())
        self.assertFalse((self.h.directory/"continuation-admission"/digest(dict(root_id=self.a.root_id))).exists())

    def test_operational_failure_stops_and_does_not_reopen(self):
        self.initialize()
        with self.assertRaisesRegex(ValueError,"operational failure"):self.produce(fail=True)
        self.assertTrue((self.h.directory/"operational-stop.json").exists())
        with patch("subprocess.Popen") as launch,self.assertRaises(ValueError):self.produce()
        launch.assert_not_called()

    def test_scientific_scope_unadmitted_before_any_launch(self):
        self.initialize()
        with patch("subprocess.Popen") as launch,self.assertRaises(ValueError):self.produce(scientific=True)
        launch.assert_not_called();self.assertFalse((self.h.directory/"selector-producer-claims").exists())

    def test_recipe_drift_refused_before_claim_or_launch(self):
        self.initialize();self.search_runtime["unreviewed"]=True
        with patch("subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"prospective recipe"):self.produce()
        launch.assert_not_called();self.assertFalse((self.h.directory/"selector-producer-claims").exists())

    def test_prospective_reference_route_has_twenty_normal_empty_workers(self):
        self.initialize(cfg=SearchConfiguration("reference",seconds=1.,workers=20))
        owner,receipt=self.produce()
        self.assertEqual(receipt["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(len(owner.reader.last_validation["owned_generation_pids"]),20)

    def test_prospective_native_original_team_route(self):
        self.initialize(cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.),stage="A1",oracle=True)
        owner,receipt=self.produce()
        self.assertEqual(receipt["status"],"SELECTED")
        self.assertTrue(owner.reader.last_validation["process_receipts_validated"])

    def test_prospective_reference_original_team_route(self):
        self.initialize(cfg=SearchConfiguration("reference",belief="oracle",seconds=1.,workers=20),stage="A1",oracle=True)
        owner,receipt=self.produce()
        self.assertEqual(receipt["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(len(owner.reader.last_validation["owned_generation_pids"]),20)

    def test_final_archive_callback_cannot_rewrite_accepted_owner(self):
        self.initialize();original=producer.OwnedSelectorRun.validate
        def late(owner,archive,source):
            result=original(owner,archive,source)
            path=owner.reader.directory/"terminal.json";path.write_text(path.read_text()+" ");return result
        with patch.object(producer.OwnedSelectorRun,"validate",late),self.assertRaisesRegex(ValueError,"accepted selector"):
            self.produce()
        self.assertTrue((self.h.directory/"operational-stop.json").exists())


if __name__ == "__main__":unittest.main()
