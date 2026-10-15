"""Actual consumed constructions/owned bridges; no scientific model or source."""
from copy import deepcopy
from dataclasses import dataclass,replace,FrozenInstanceError
import hashlib
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_native_chain import (
    OwnedNativeMaterializationPlan,plan_owned_native_materializations,build_owned_native_materialization_jobs)
from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator,fixed_work_reader_source_hashes
from pokezero.mcts_eval.search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS
from pokezero.mcts_eval.search_over_raw_materialization_run import MaterializationRunValidator,supervise_materialization_run
from pokezero.mcts_eval.search_over_raw_consumed_materialization import join_consumed_materializations
from test_search_over_raw_recorded_materialization import AuthoredRecordedJob
import test_search_over_raw_fixed_work_run as native_fixture
import test_search_over_raw_materialization_run as material_fixture
import test_search_over_raw_construction as construction_fixture
import test_search_over_raw_source_run as source_fixture
import test_search_over_raw_consumed_materialization as consumed_fixture
import test_search_over_raw_ancestry as ancestry_fixture
import test_search_over_raw_native_edge as edge_fixture
from test_engine_world import _payload,_dex

from pokezero.mcts_eval.search_over_raw_supervision import FixedWorkAdapterJob,supervise_fixed_work
from pokezero.mcts_eval.search_over_raw_ancestry import NativeAncestryContract
from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_fixed_work_job
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import OwnedFixedWorkDiagnostic
from pokezero.mcts_eval.search_over_raw_native_chain import native_chain_review_template,produce_owned_native_diagnostic_chain
from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_source_run import collect_owned_exploration_source


@dataclass(frozen=True)
class AuthoredRetainedNativeJob(FixedWorkAdapterJob):
    public_witness: dict=None
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        try:return self._run(ownership,phase)
        except BaseException as error:
            (Path(ownership["directory"])/"authored-error.json").write_text(json.dumps(
                dict(error=type(error).__name__,message=str(error))))
            raise
    def _run(self,ownership,phase):
        selected=consumed_fixture.AuthoredArchivedNativeJob(self.configuration,self.context,
            self.statistics_root_id,self.selection_seed,self.public_witness)(ownership,phase)
        selected["runtime_sha256"]=self.expected_runtime_sha256
        for worker,(row,call) in enumerate(zip(selected["evidence"]["native_searched_ancestry"],
                selected["evidence"]["consumed_world_inputs"]["calls"])):
            value=json.loads(row["export_utf8"]);value["schema"]=edge_fixture.ANCESTRY_EDGE_SCHEMA
            for leaf in value["evaluations"]:
                for step in leaf["capture"]["steps"]:edge_fixture.add_edge(step)
            selected["evidence"]["native_searched_ancestry"][worker]=ancestry_fixture.row_from_payload(value,call,worker)
        return selected


class AuthoredChainHelpers:
    synthetic_process_fixture=True
    extract=staticmethod(consumed_fixture.OwnedAncestryRootJoinTests.extractor)


@dataclass
class AuthoredChainTransport:
    mutation: str="none"
    calls: int=0
    synthetic_process_fixture=True
    def __call__(self,*,role,job,native_owner):
        self.calls+=1
        root=native_owner.source_owner.ledger.directory/"native-diagnostic-chains"/digest(native_owner.job.statistics_root_id)
        claim=json.loads((root/"claim.json").read_bytes())
        leaves=claim["leaf_cell_plan"]
        if leaves["retained_unique_evaluation_events"]!=2 or leaves["required_label_cells"]!=16 \
                or any(e["canonical_construction_ordinal"]!=0 or e["construction_aliases"]!=[0,1,2,3]
                    for e in leaves["events"]):
            raise ValueError("unique leaf roster was not frozen before constructor/replay transport")
        if any(c["signed_outcome"] is not None or c["status"]!="UNSTARTED_UNCERTAIN"
                for e in leaves["events"] for c in e["labels"]):
            raise ValueError("unowned terminal labels appeared in frozen leaf roster")
        if self.calls==2 and self.mutation!="none":
            root=native_owner.source_owner.ledger.directory/"native-diagnostic-chains"/digest(native_owner.job.statistics_root_id)
            reservation=next((root/"reservations").iterdir())
            if self.mutation=="bytes":reservation.write_bytes(reservation.read_bytes()+b" ")
            elif self.mutation=="delete":reservation.unlink()
            elif self.mutation=="extra":(root/"reservations"/"extra.json").write_text("{}")
        replay_cls=(consumed_fixture.AuthoredNativePrefixReplayJob
            if type(job) is consumed_fixture.OwnedNativePrefixBoundaryJob else consumed_fixture.AuthoredNativeReplayJob)
        return dict(job_class=AuthoredRecordedJob if role=="materialization" else replay_cls,
            job_kwargs={})


@dataclass(frozen=True)
class AuthoredRetainedChainJob(source_fixture.AuthoredSourceJob):
    chain_config: object=None
    chain_runtime: dict=None
    options: dict=None
    helper_files: dict=None
    chain_review_path: str=""
    mutation: str="none"
    preserve_replay_prefix: bool=False
    def __call__(self,ownership,phase):
        ledger=ReconciledPhaseALedger(self.ledger_directory);handle=None
        phase("startup");phase("selection")
        try:
            cfg=SearchConfiguration("incumbent",seconds=1.)
            handle=collect_owned_exploration_source(ledger,stage="A1",configuration=cfg.identity,
                source_seed=self.seed,config=self.config,checkpoint_contract=self.checkpoint,bridge_runtime=self.runtime,
                source_review_path=self.source_review_path,history_review_path=self.history_review_path,
                _synthetic_factory=source_fixture.AuthoredSourceFactory)
            root_id=handle.source["roots"][0]["root_id"]
            original,bound=prepare_archived_fixed_work_job(handle.archive,handle.source,contract=ledger.contract,
                root_id=root_id,configuration=cfg,checkpoint_contract=self.checkpoint,
                showdown_root=self.checkpoint.showdown_root,runtime_sha256=digest(self.chain_runtime),
                native_ancestry_contract=NativeAncestryContract(17,capture_native_edges=True))
            # Re-sanitize the ORIGINAL retained context, not an already stripped
            # request whose public catalog is intentionally absent.
            from dataclasses import fields
            context,_,_=handle.archive.selected(handle.source["roots"][0])
            args={f.name:getattr(original,f.name) for f in fields(FixedWorkAdapterJob)};args["context"]=deepcopy(context)
            job=AuthoredRetainedNativeJob(**args,public_witness=deepcopy(bound["public_request_witness"]))
            directory=ledger.directory/"authored-native";job_sha=hashlib.sha256(pickle.dumps(job)).hexdigest()
            witness=bound["public_request_witness"]
            files={**fixed_work_reader_source_hashes(),**self.chain_runtime["files"]}
            import sys
            executable=str(Path(sys.executable).resolve())
            files[executable]=hashlib.sha256(Path(executable).read_bytes()).hexdigest()
            reader=FixedWorkRunValidator(directory=directory,job_sha256=job_sha,binding_sha256="b"*64,
                configuration=cfg,root_id=job.root_id,statistics_root_id=root_id,selection_seed=self.seed,
                runtime_sha256=digest(self.chain_runtime),consumed_binding=witness["consumed_binding"],
                public_request_sha256=witness["public_request_sha256"],file_hashes=files,
                checkpoint_path=self.checkpoint.checkpoint_path,checkpoint_sha256=self.checkpoint.checkpoint_sha256,
                showdown_root=self.checkpoint.showdown_root,allow_synthetic=True)
            terminal=supervise_fixed_work(job,directory=directory,binding_sha256="b"*64,verify_admission=lambda:True)
            if terminal["supervisor_exit_code"]!=0:
                error=directory/"owned-groups"/"authored-error.json"
                raise ValueError("authored native failed: "+(error.read_text() if error.exists() else str(terminal)))
            selected=terminal["result"];selected_sha=reader(selected)
            if selected_sha!=digest(selected):raise ValueError("authored native independent read failed")
            owner=OwnedFixedWorkDiagnostic(handle,job,reader,selected,selected_sha)
            private,report=produce_owned_native_diagnostic_chain(ledger,owner,config=self.chain_config,
                runtime=self.chain_runtime,options=self.options,helper_files=self.helper_files,review_path=self.chain_review_path,
                preserve_prefix=self.preserve_replay_prefix,
                _synthetic_helpers=AuthoredChainHelpers(),_synthetic_job_transport=AuthoredChainTransport(self.mutation))
            owner.validate(ledger)
            return dict(constructor_runs=report["actual_constructor_runs"],replay_runs=report["actual_replay_runs"],
                retained_leaf_events=report["retained_unique_leaf_events"],leaf_cells=report["required_leaf_label_cells"],
                completed_leaf_cells=report["completed_leaf_label_cells"],
                required_samples=report["owned_replay_join"]["required_construction_sample_records"],
                attached=report["root_to_panel_attachment_validated"],archive_live=bool(handle.archive._public),
                scientific_collection_authorized=report["scientific_collection_authorized"],
                deployment_accepted_units_added=report["deployment_accepted_units_added"],
                replay_prefix_stops=sum(r["status"]=="OWNED_PREFIX_AND_UNAVAILABLE_SUFFIX"
                    for r in report["owned_replay_join"]["records"]),
                required_boundaries=report["owned_replay_join"]["required_boundary_records"])
        except BaseException as error:
            (Path(ownership["directory"]).parent/"authored-error.json").write_text(json.dumps(
                dict(error=type(error).__name__,message=str(error),notes=getattr(error,"__notes__",[]))))
            raise
        finally:
            phase("cleanup")
            if handle is not None:handle.close()


class RetainedNativeChainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.s=source_fixture.SourceRunTests();self.s.setUp();self.addCleanup(self.s.doCleanups)
    def prepare(self,mutation="none",*,preserve_prefix=False,unavailable_root=False):
        original_initialize=self.s.h.initialize
        def initialize():
            bridge_dir=self.s.root/"constructor-bridge";bridge_dir.mkdir()
            bridge=bridge_dir/"fixture.mjs";bridge.write_text(material_fixture.BRIDGE)
            repo=Path(__file__).resolve().parents[1]
            for name in ("battle_bridge_owned_continuation.mjs","battle_bridge_owned_snapshot.mjs"):
                (bridge_dir/name).write_bytes((repo/"scripts"/name).read_bytes())
            snapshot=construction_fixture.map_fixture()[0]
            import test_search_over_raw_showdown_boundary as boundary_fixture
            for side in snapshot["battle"]["sides"]:
                for mon in side["pokemon"]:
                    mon.update(moveSlots=[dict(id="move",pp=15,disabled=False)],hpType="Ice",hpPower=70)
            snapshot["boundaryRequests"]=boundary_fixture.requests(snapshot["battle"]["sides"])
            if unavailable_root:snapshot["boundaryRequests"]["p1"]["active"][0]["moves"][0]["disabled"]=True
            fixture=Path(self.s.config.resolved_showdown_root())/"fixture.json"
            fixture.write_text(json.dumps(dict(mode="complete",snapshot=snapshot)))
            self.config=replace(self.s.config,bridge_path=str(bridge),read_timeout_seconds=2.)
            self.runtime=outcome_runtime_binding(self.config,checkpoint_path=self.s.checkpoint.checkpoint_path,extra_files=[fixture])
            self.options={k:False for k in BOOL_OPTIONS};self.options.update({k:None for k in MAP_OPTIONS})
            path=str(Path(__file__).resolve());self.helpers={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            self.review=self.s.root/"chain-review.json"
            review=native_chain_review_template(self.s.h.contract,config=self.config,runtime=self.runtime,
                options=self.options,helper_files=self.helpers,preserve_prefix=preserve_prefix)
            review.update(status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",independent_review_completed=True,
                reviewed_before_first_source_work=True)
            self.review.write_text(json.dumps(review))
            hashes=source_fixture.admission_fixture.controller_input_hashes()
            with patch.object(source_fixture.admission_fixture,"controller_input_hashes",return_value={**hashes,
                    str(self.review):hashlib.sha256(self.review.read_bytes()).hexdigest()}):original_initialize()
        with patch.object(self.s.h,"initialize",side_effect=initialize):original=self.s.prepare(rounds=2)
        return AuthoredRetainedChainJob(**vars(original),chain_config=self.config,chain_runtime=self.runtime,
            options=self.options,helper_files=self.helpers,chain_review_path=str(self.review),mutation=mutation,
            preserve_replay_prefix=preserve_prefix)

    def test_prospective_unavailable_prefixes_preserve_all_eight_original_replays_and_null_labels(self):
        terminal=self.s.run_job(self.prepare(preserve_prefix=True,unavailable_root=True))
        error=self.s.directory/"authored-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        result=terminal["result"]
        self.assertEqual((result["constructor_runs"],result["required_samples"],result["replay_runs"]),(4,8,8))
        self.assertEqual((result["replay_prefix_stops"],result["required_boundaries"]),(8,16))
        self.assertEqual((result["leaf_cells"],result["completed_leaf_cells"]),(16,0))
        self.assertTrue(result["attached"])
        self.assertFalse(result["scientific_collection_authorized"])
    def test_actual_live_owned_source_runs_complete_materialization_and_replay_chain(self):
        terminal=self.s.run_job(self.prepare())
        error=self.s.directory/"authored-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        result=terminal["result"]
        self.assertEqual((result["constructor_runs"],result["required_samples"]),(4,8))
        self.assertEqual(result["replay_runs"],8)
        self.assertEqual((result["retained_leaf_events"],result["leaf_cells"],result["completed_leaf_cells"]),(2,16,0))
        self.assertTrue(result["attached"]);self.assertTrue(result["archive_live"])
        self.assertFalse(result["scientific_collection_authorized"])
        self.assertEqual(result["deployment_accepted_units_added"],0)

    def test_later_transport_cannot_change_earlier_reservation_bytes_roster_or_presence(self):
        for mutation in ("bytes","delete","extra"):
            with self.subTest(mutation=mutation):
                h=RetainedNativeChainIntegrationTests();h.setUp();self.addCleanup(h.doCleanups)
                terminal=h.s.run_job(h.prepare(mutation))
                self.assertEqual(terminal["supervisor_exit_code"],1)
                error=json.loads((h.s.directory/"authored-error.json").read_text())
                self.assertIn("original native-chain reservation bytes or roster changed",error["message"])
                root=h.s.h.directory/"native-diagnostic-chains"
                terminals=list(root.rglob("terminal.json"))
                self.assertEqual(len(terminals),1)  # Only the original first materialization was attempted.
                self.assertFalse(list(root.rglob("diagnostic.json")))
                self.assertTrue((h.s.h.directory/"operational-stop.json").is_file())


@dataclass(frozen=True)
class AuthoredPendingNativeJob(native_fixture.AuthoredFixedWorkJob):
    pending_payload: dict | None = None
    def __call__(self,ownership,phase):
        with patch.object(construction_fixture,"_payload",return_value=deepcopy(self.pending_payload)):
            return super().__call__(ownership,phase)


class NativeMaterializationPlanTests(unittest.TestCase):
    def setUp(self):
        self.m=material_fixture.MaterializationRunTests();self.m.setUp();self.addCleanup(self.m.doCleanups)
        self.n=native_fixture.FixedWorkRunReaderTests();self.n.setUp();self.addCleanup(self.n.doCleanups)
        # The native owner and prospective constructors share original declared
        # champion/simulator bytes. Neither model nor simulator is actually loaded.
        self.n.checkpoint=self.m.checkpoint.resolve();self.n.showdown=Path(self.m.config.resolved_showdown_root()).resolve()
        self.n.files.update(fixed_work_reader_source_hashes());self.n.files.update(self.m.runtime["files"])
        self.root_payload=_payload(_dex());pending=deepcopy(self.root_payload)
        pending["turn"]=self.root_payload["turn"]+1
        self.pending_turn=pending["turn"]
        # Nonempty deferred actions are explicitly refused by this native
        # constructor. Keep that scientific restriction; use a supported
        # differing overlay, not a fixture-only waiver of the refusal.
        pending["deferredOpponentActions"]={};pending["deferredOpponentActionPriors"]={}
        # This modifies the input to the REAL Python constructor hooks, not the
        # saved snapshot or capture. The native FFI/report alone is authored.
        with patch.object(native_fixture,"AuthoredFixedWorkJob",side_effect=lambda cfg,sha:
                AuthoredPendingNativeJob(cfg,sha,pending)):
            self.selected,inputs=self.n.run_fixture(public_request_sha256="e"*64)
        self.reader=FixedWorkRunValidator(**inputs,allow_synthetic=True)
        self.options={k:False for k in BOOL_OPTIONS};self.options.update({k:None for k in MAP_OPTIONS})
        self.attempt_root=self.n.root/"recorded-materializations"

    def plan(self):
        return plan_owned_native_materializations(self.selected,native_validator=self.reader,
            config=self.m.config,runtime=self.m.runtime,options=self.options,attempt_root=self.attempt_root,
            public_state_sha256="f"*64)

    def build(self,plan):
        return build_owned_native_materialization_jobs(plan,self.selected,native_validator=self.reader,
            config=self.m.config,runtime=self.m.runtime,options=self.options)

    def test_exact_four_actual_draws_pending_overlays_and_seeds_without_sampling_or_launch(self):
        with patch("subprocess.Popen",side_effect=AssertionError("planner launched")):
            plan=self.plan();jobs=self.build(plan);again=self.plan()
        self.assertEqual(plan,again)
        saved=plan.receipt();self.assertEqual(len(jobs),4)
        self.assertEqual([r["construction_ordinal"] for r in saved["records"]],list(range(4)))
        self.assertEqual(saved["actual_native_calls"],1)
        self.assertFalse(self.attempt_root.exists())
        for row in saved["records"]:
            ordinal=row["construction_ordinal"]
            captured=self.selected["evidence"]["consumed_world_inputs"]["rows"][ordinal]["construction_receipt"]
            world=captured["events"][0]["value"]
            inputs=row["recorded_inputs"]
            self.assertEqual(inputs["draw"],world["override"])
            self.assertEqual(inputs["materialize"]["publicState"],world["payload"])
            self.assertNotEqual(world["payload"],self.root_payload)
            self.assertEqual(inputs["materialize"]["publicState"]["turn"],self.pending_turn)
            self.assertEqual(inputs["options"]["deferred_opponent_actions"],{})
            call=self.selected["evidence"]["consumed_world_inputs"]["calls"][row["native_call_ordinal"]]
            self.assertEqual(inputs["seed"],call["native_call_parameters"]["chance_seed"])
        with self.assertRaises(FrozenInstanceError):plan.payload_utf8="{}"

    def test_all_four_owned_recorded_constructors_join_actual_consumed_population(self):
        plan=self.plan();jobs=self.build(plan);runs=[]
        for descriptor in jobs:
            original=descriptor["job"]
            job=AuthoredRecordedJob(original.config,original.recorded_inputs,original.runtime)
            directory=Path(descriptor["attempt_directory"]);binding=digest(dict(plan=plan.sha256,ordinal=descriptor["construction_ordinal"]))
            expected=original.input_binding()
            reader=MaterializationRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                binding_sha256=binding,expected_inputs=expected,runtime=self.m.runtime,allow_synthetic=True)
            terminal=supervise_materialization_run(job,directory=directory,binding_sha256=binding,
                verify_admission=lambda:True,validate_result=lambda _:True)
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            self.assertEqual(reader(terminal["result"]["capture"]),digest(terminal["result"]["capture"]))
            runs.append(dict(construction_ordinal=descriptor["construction_ordinal"],result=terminal["result"],validator=reader))
        joined=join_consumed_materializations(self.selected,runs,native_validator=self.reader)
        self.assertEqual(joined["required_construction_ordinals"],4)
        self.assertEqual(joined["matched_constructor_ordinals"],4)
        self.assertEqual(joined["materialization_runs_recorded"],4)
        self.assertEqual(joined["actual_native_calls"],1)
        self.assertFalse(joined["full_world_equivalence"])
        self.assertFalse(joined["scientific_collection_authorized"])
        # A genuinely owned, internally valid constructor using a substituted
        # seed must not match the ACTUAL original native invocation seed.
        from pokezero.mcts_eval.search_over_raw_materialization import recorded_materialization_inputs
        original=jobs[0]["job"];inputs=original.recorded_inputs
        changed=recorded_materialization_inputs(draw=inputs["draw"],
            public_payload=inputs["materialize"]["publicState"],seed=inputs["seed"]+1,
            root_binding=inputs["root_binding"],public_state_sha256=inputs["public_state_sha256"],
            recorded_native_origin=inputs["recorded_native_origin"],options=inputs["options"])
        substitute=AuthoredRecordedJob(original.config,changed,original.runtime)
        directory=self.n.root/"independently-owned-substituted-seed"
        binding="9"*64
        terminal=supervise_materialization_run(substitute,directory=directory,binding_sha256=binding,
            verify_admission=lambda:True,validate_result=lambda _:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader=MaterializationRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(substitute)).hexdigest(),
            binding_sha256=binding,expected_inputs=substitute.input_binding(),runtime=self.m.runtime,allow_synthetic=True)
        replaced=[dict(construction_ordinal=0,result=terminal["result"],validator=reader)]+runs[1:]
        mismatch=join_consumed_materializations(self.selected,replaced,native_validator=self.reader)
        self.assertEqual(mismatch["matched_constructor_ordinals"],3)
        self.assertEqual(len(mismatch["rows"]),4)
        self.assertEqual(mismatch["rows"][0]["status"],"INPUT_MISMATCH")
        self.assertIn("recorded_native_chance_seed",mismatch["rows"][0]["mismatched_inputs"])

    def test_forged_overlay_origin_seed_or_dropped_ordinal_cannot_change_frozen_plan(self):
        plan=self.plan()
        for mode in ("overlay","origin","seed","dropped"):
            saved=plan.receipt()
            if mode=="overlay":saved["records"][0]["recorded_inputs"]["materialize"]["publicState"]["turn"]=78
            elif mode=="origin":saved["records"][0]["recorded_inputs"]["recorded_native_origin"]["native_selected_sha256"]="1"*64
            elif mode=="seed":saved["records"][0]["recorded_inputs"]["seed"]+=1
            else:saved["records"].pop()
            import json
            forged=OwnedNativeMaterializationPlan(json.dumps(saved,sort_keys=True,separators=(",",":"),ensure_ascii=False))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.build(forged)

    def test_existing_original_attempt_is_not_replaced_or_resumed(self):
        plan=self.plan();first=plan.receipt()["records"][0]
        Path(first["attempt_directory"]).mkdir(parents=True)
        with self.assertRaises(ValueError):self.build(plan)
        after=self.plan().receipt()
        self.assertEqual(len(after["records"]),4)
        self.assertEqual(after["records"][0]["attempt_directory"],first["attempt_directory"])
        self.assertEqual(after["records"][0]["status"],"ATTEMPT_ALREADY_CLAIMED")

    def test_late_native_reader_cannot_redirect_or_occupy_original_plan_path(self):
        baseline=self.plan();first=baseline.receipt()["records"][0]
        real=FixedWorkRunValidator.__call__;calls=0
        def late(reader,selected):
            nonlocal calls
            answer=real(reader,selected)
            if reader is self.reader:
                calls+=1
                if calls==2:Path(first["attempt_directory"]).mkdir(parents=True)
            return answer
        with patch.object(FixedWorkRunValidator,"__call__",late),self.assertRaisesRegex(ValueError,"paths or occupancy"):
            self.plan()
        self.assertEqual(calls,2)
