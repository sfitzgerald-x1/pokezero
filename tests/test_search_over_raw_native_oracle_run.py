"""Authored original teams, Python constructors and disposable owners only."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_native_oracle import NativeOracleWorldInputs, NATIVE_ORACLE_CAPTURE
from pokezero.mcts_eval.search_over_raw_native_oracle_run import (NativeOracleTimedRunContract,
    NativeOracleTimedAdapterJob, NativeOracleTimedRunValidator, read_native_oracle_timed_selection,
    supervise_native_oracle_timed)
from pokezero.mcts_eval.search_over_raw_timed_run import read_native_timed_selection, NativeTimedRunContract
from pokezero.mcts_eval.search_over_raw_construction import SCHEMA, construction_source_hashes
from pokezero.mcts_eval.search_over_raw_world_inputs import (NativeInputHandle, TIMED_NATIVE_PUBLIC_CAPTURE,
    validate_timed_native_inputs)
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
import test_search_over_raw_native_oracle as truth_fixture
import test_search_over_raw_timed_run as owner_fixture


def observed_runtime(configuration, checkpoint_sha="b"*64):
    value=owner_fixture.runtime(configuration,checkpoint_sha)
    value["set_source_hash"]="a"*64
    value["full_public_request_observation"]=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE)
    value["native_team_oracle_observation"]=deepcopy(NATIVE_ORACLE_CAPTURE)
    value["native_oracle_construction"]=dict(schema=SCHEMA,source_hashes=construction_source_hashes(),
        private_controller_only=True,instrumentation_can_change_deadlines=True,runtime_qualified=False)
    return value


@dataclass(frozen=True)
class AuthoredOracleOwner:
    selected: dict
    behavior: str="normal"
    synthetic_process_fixture=True

    def __call__(self, ownership, phase):
        phase("startup");phase("selection")
        if self.behavior == "failure":raise RuntimeError("authored operational failure")
        if self.behavior == "timeout":time.sleep(2.)
        time.sleep(self.selected["elapsed_seconds"]+.01)
        phase("cleanup")
        if self.behavior == "cleanup":raise ValueError("authored cleanup failure")
        return deepcopy(self.selected)


class NativeOracleRunTests(unittest.TestCase):
    def setUp(self):
        self.truth=truth_fixture.NativeOracleTests();self.truth.setUp();self.addCleanup(self.truth.doCleanups)
        self.cfg=SearchConfiguration("incumbent",belief="oracle",seconds=1.)
        self.runtime=observed_runtime(self.cfg)
        t=self.truth
        self.runtime["team_oracle"]=t.f.oracle.receipt()
        self.inputs=dict(configuration=self.cfg,runtime_configuration=self.runtime,runtime_sha256=digest(self.runtime),
            root_id="root:"+self.cfg.identity,statistics_root_id="root",selection_seed=7,
            consumed_binding=t.witness["consumed_binding"],legal_action_mask=t.f.context.observation.legal_action_mask,
            decision_round=t.f.context.decision_round_index,contract=NativeOracleTimedRunContract(),
            public_request_sha256=t.witness["public_request_sha256"],public_request=t.witness,
            oracle_receipt=t.f.oracle.receipt(),expected_override_sha256=digest(asdict(t.override)),
            expected_reconstruction_catalog_sha256=t.catalog_sha)

    def selected(self, *, zero=False, attempts=4):
        t=self.truth
        recorder=NativeOracleWorldInputs(t.witness["consumed_binding"],4096,record_atomic_work=True,
            record_timed_atomic_work=True,expected_public_request=t.witness,
            oracle=t.f.oracle,oracle_request=t.f.request,oracle_source=t.f.source,oracle_policy=t.policy)
        for _ in range(attempts):world,_=t.construct(recorder)
        completed=0 if zero else 16
        if attempts:
            report=truth_fixture.timed_fixture.timed_report(completed,requested=4096*attempts)
            report["model_evals"]=1
            args=truth_fixture.timed_fixture.timed_arguments(requested=4096*attempts)
            args[5]=json.dumps(dict(**{seat:list(world.party_species[seat]) for seat in ("p1","p2")},turn=3))
            NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a,**kw:json.dumps(report)),recorder) \
                .search_batched_multi_encoded(*args)
        manifest=t.finish(recorder,completed)
        selected=owner_fixture.selected_fixture(self.cfg,self.runtime,zero=zero,
            elapsed=1.2 if zero else .02,decision_round=self.inputs["decision_round"])
        selected["native_timed_run_contract"]=NativeOracleTimedRunContract().receipt()
        evidence=selected["evidence"];evidence["consumed_world_inputs"]=manifest
        evidence["team_oracle"]=t.f.oracle.receipt()
        budget=evidence["engine_mcts"]["time_budget"]
        if attempts == 0:budget.update(native_invocations=[],worlds_budget_skipped=0)
        elif attempts != 4:
            budget["native_invocations"][0].update(multiplicity=attempts,requested_iterations=4096*attempts,
                remaining_iterations=4096*attempts-completed)
            if not zero:evidence["engine_mcts"]["joint_actions"]["native_invocations"][0]["belief_multiplicity"]=attempts
        if zero:
            budget.update(deadline_elapsed_ms=1001.,deadline_overshoot_ms=1.)
            selected["strict_engine_diagnostic"]["engine_mcts"]=deepcopy(evidence["engine_mcts"])
            selected["original_failure_receipt"]["engine_fallback_diagnostic"]=deepcopy(selected["strict_engine_diagnostic"])
            selected["timed_input_readout"]=validate_timed_native_inputs(evidence,
                binding=self.inputs["consumed_binding"],iterations_per_world=4096,batch_size=16,
                expected_worlds=4,nominal_seconds=1.,adapter_elapsed_seconds=1.2)
        return selected

    def read(self, selected, **changes):
        return read_native_oracle_timed_selection(selected,**{**self.inputs,**changes})

    def job(self, **changes):
        t=self.truth
        return NativeOracleTimedAdapterJob(configuration=self.cfg,
            checkpoint_contract=SimpleNamespace(showdown_source_sha256="a"*64,checkpoint_sha256="b"*64,
                checkpoint_path="not-a-model"),showdown_root="not-a-simulator",context=t.f.context,
            root_id=self.inputs["root_id"],selection_seed=7,statistics_root_id="root",
            expected_runtime_configuration=self.runtime,
            expected_public_request_sha256=self.inputs["public_request_sha256"],
            oracle=t.f.oracle,expected_oracle_receipt=self.inputs["oracle_receipt"],
            expected_override_sha256=self.inputs["expected_override_sha256"],
            expected_reconstruction_catalog_sha256=changes.pop("catalog",t.catalog_sha),**changes)

    def test_same_prefix_law_positive_zero_and_preparation_only(self):
        for zero,attempts in ((False,4),(False,2),(True,4),(True,0)):
            with self.subTest(zero=zero,attempts=attempts):
                result=self.read(self.selected(zero=zero,attempts=attempts))
                self.assertEqual(result["accepted_units"],0 if zero else 16)
                self.assertIs(result["original_team_consumption"]["team_consumed"],attempts > 0)
                self.assertFalse(result["deployable"]);self.assertFalse(result["archive_truth_correspondence_validated"])
                self.assertFalse(result["runtime_qualified"])

    def test_public_reader_and_wrong_contract_or_configuration_reject_A1(self):
        selected=self.selected()
        public={k:v for k,v in self.inputs.items() if k not in ("oracle_receipt","public_request",
            "expected_override_sha256","expected_reconstruction_catalog_sha256")}
        with self.assertRaises(ValueError):read_native_timed_selection(selected,**public)
        with self.assertRaises(ValueError):self.read(selected,contract=NativeTimedRunContract())
        for cfg in (SearchConfiguration("incumbent"),replace(self.cfg,seconds=3.),
                replace(self.cfg,leaf="hp_fraction"),replace(self.cfg,workers=2)):
            with self.assertRaises(ValueError):self.read(selected,configuration=cfg)
        public_cfg=SearchConfiguration("incumbent")
        public["configuration"]=public_cfg;public["contract"]=NativeTimedRunContract()
        runtime=observed_runtime(public_cfg);public.update(runtime_configuration=runtime,runtime_sha256=digest(runtime))
        with self.assertRaisesRegex(ValueError,"cannot carry oracle"):read_native_timed_selection(selected,**public)

    def test_missing_truth_and_external_bindings_never_certify(self):
        for field in ("team_oracle","native_team_oracle","constructor","catalog","override","public","runtime"):
            selected=self.selected();changes={}
            if field == "team_oracle":selected["evidence"].pop(field)
            elif field == "native_team_oracle":selected["evidence"]["consumed_world_inputs"].pop(field)
            elif field == "constructor":selected["evidence"]["consumed_world_inputs"]["rows"][0].pop("construction_receipt")
            elif field == "catalog":changes["expected_reconstruction_catalog_sha256"]="f"*64
            elif field == "override":changes["expected_override_sha256"]="f"*64
            elif field == "public":changes["public_request_sha256"]="f"*64
            else:
                runtime=deepcopy(self.runtime);runtime["native_oracle_construction"]["source_hashes"]={}
                changes.update(runtime_configuration=runtime,runtime_sha256=digest(runtime))
            with self.subTest(field=field),self.assertRaises(ValueError):self.read(selected,**changes)

    def test_runtime_team_declaration_is_independently_bound_not_self_certified(self):
        for missing in (True,False):
            selected=self.selected();runtime=deepcopy(self.runtime)
            if missing:runtime.pop("team_oracle")
            else:runtime["team_oracle"]["opponent_team_sha256"]="f"*64
            selected.update(runtime_sha256=digest(runtime),observed_runtime_configuration=deepcopy(runtime))
            with self.assertRaises(ValueError):self.read(selected,runtime_configuration=runtime,runtime_sha256=digest(runtime))
            with self.assertRaises(ValueError):replace(self.job(),expected_runtime_configuration=runtime)

    def test_job_public_projection_and_full_truth_are_frozen(self):
        job=self.job();self.assertEqual(job.read_selection(self.selected())["accepted_units"],16)
        wire=pickle.dumps(job)
        for token in (b"PRIVATE_SNAPSHOT_NEVER_TRANSPORT",b"PRIVATE_OPPONENT_ACTION",b"LocalShowdownSnapshot"):
            self.assertNotIn(token,wire)
        self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
        job._check_source(self.truth.f.source)
        with self.assertRaises(ValueError):self.job(catalog="f"*64)._check_source(self.truth.f.source)
        object.__setattr__(job.oracle,"opponent_team_sha256","f"*64)
        with self.assertRaises(ValueError):job.read_selection(self.selected())

    def test_actual_job_wires_oracle_closes_and_checks_loaded_catalog(self):
        job=self.job();public_job=owner_fixture.NativeTimedJobWiringTests()
        adapter=public_job.adapter(job);adapter.oracle=job.oracle;adapter._oracle_source=self.truth.f.source
        adapter.select.return_value=self.selected(zero=True,attempts=0)
        phases=[]
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=self.truth.f.source), \
                patch("pokezero.mcts_eval.search_over_raw_native_oracle_run.NativeTimedOracleSearchAdapter",
                    return_value=adapter) as factory:
            result=job({},phases.append)
        self.assertEqual(phases,["startup","selection","cleanup"]);adapter.close.assert_called_once()
        self.assertIs(factory.call_args.kwargs["oracle"],job.oracle)
        self.assertEqual(result["status"],"UNQUALIFIED_AT_CEILING")
        self.assertFalse(job.read_selection(result)["original_team_consumption"]["team_consumed"])

    def test_source_drift_and_cleanup_failures_are_operational(self):
        for mutation in ("source","late-source","cleanup","allocation","truth"):
            job=self.job();adapter=owner_fixture.NativeTimedJobWiringTests().adapter(job)
            adapter.oracle=job.oracle;adapter._oracle_source=deepcopy(self.truth.f.source)
            adapter.select.return_value=self.selected(zero=True,attempts=0)
            if mutation == "source":adapter._oracle_source.move_metadata["late"]={"wrong":True}
            elif mutation == "late-source":adapter.close.side_effect=lambda:adapter._oracle_source.move_metadata.update(late={})
            elif mutation == "cleanup":adapter.close.side_effect=RuntimeError("cleanup failure")
            elif mutation == "allocation":adapter._native._config.search_sims=16
            else:adapter.oracle=replace(job.oracle,opponent_team_sha256="f"*64)
            with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=self.truth.f.source), \
                    patch("pokezero.mcts_eval.search_over_raw_native_oracle_run.NativeTimedOracleSearchAdapter",return_value=adapter), \
                    self.subTest(mutation=mutation),self.assertRaises((ValueError,RuntimeError)):
                job({},lambda p:None)
            adapter.close.assert_called_once()

    def test_supervisor_wrapper_distinct_kind_no_public_job_alias(self):
        job=self.job()
        with patch("pokezero.mcts_eval.search_over_raw_native_oracle_run._supervise_owned_job",return_value={}) as owner:
            supervise_native_oracle_timed(job,directory="never-created",binding_sha256="a"*64,verify_admission=lambda:False)
        self.assertEqual(owner.call_args.kwargs["kind"],"native-oracle-timed-run")
        self.assertEqual(owner.call_args.kwargs["validate_result"],job.read_selection)
        with self.assertRaises(ValueError):supervise_native_oracle_timed(object(),directory="never-created",
            binding_sha256="a"*64,verify_admission=lambda:True)

    def test_real_archive_projection_prepares_job_without_launch_or_private_snapshot(self):
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_native_oracle_timed_job
        f=self.truth.f
        with patch("subprocess.Popen") as launch:
            job,receipt=prepare_archived_native_oracle_timed_job(f.archive,f.f.source,contract=f.f.contract,
                root_id=f.f.root_id,configuration=self.cfg,checkpoint_contract=SimpleNamespace(
                    showdown_source_sha256="a"*64,checkpoint_sha256="b"*64,checkpoint_path="not-a-model"),
                showdown_root="not-a-simulator",runtime_configuration=self.runtime,set_source=f.source)
        launch.assert_not_called()
        self.assertEqual(job.oracle,f.oracle)
        self.assertEqual(job.statistics_root_id,f.f.root_id)
        self.assertEqual(job._inputs(),receipt["job_inputs_sha256"])
        self.assertEqual(receipt["team_projection"]["team_oracle"],self.inputs["oracle_receipt"])
        self.assertFalse(receipt["private_snapshot_transported"])
        self.assertFalse(receipt["collection_authorized"])
        wire=pickle.dumps(job)
        for token in (b"PRIVATE_SNAPSHOT_NEVER_TRANSPORT",b"PRIVATE_OPPONENT_ACTION",b"LocalShowdownSnapshot"):
            self.assertNotIn(token,wire)

    def test_archive_drift_during_job_construction_rejects_preparation(self):
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_native_oracle_timed_job
        f=self.truth.f
        def changed(**kwargs):
            job=NativeOracleTimedAdapterJob(**kwargs)
            f.snapshot.first_requests["p2"]["side"]["pokemon"][0]["item"]="Changed"
            return job
        with patch("pokezero.mcts_eval.search_over_raw_native_oracle_run.NativeOracleTimedAdapterJob",side_effect=changed), \
                self.assertRaises(ValueError):
            prepare_archived_native_oracle_timed_job(f.archive,f.f.source,contract=f.f.contract,
                root_id=f.f.root_id,configuration=self.cfg,checkpoint_contract=SimpleNamespace(
                    showdown_source_sha256="a"*64,checkpoint_sha256="b"*64,checkpoint_path="not-a-model"),
                showdown_root="not-a-simulator",runtime_configuration=self.runtime,set_source=f.source)

    def test_alternate_self_consistent_job_cannot_replace_extracted_archive_truth(self):
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_native_oracle_timed_job
        from pokezero.mcts_eval.search_over_raw_oracle import team_sha256
        from pokezero.env import BattleStartOverride
        from pokezero.showdown_fixture import pack_team
        f=self.truth.f
        def changed(**kwargs):
            old=kwargs["oracle"];team=(replace(old.opponent_team[0],item="Choice Band"),*old.opponent_team[1:])
            alternate=replace(old,opponent_team=team,opponent_team_sha256=team_sha256(team))
            alternate.validate(f.request,f.source)
            kwargs.update(oracle=alternate,expected_oracle_receipt=alternate.receipt(),
                expected_runtime_configuration={**kwargs["expected_runtime_configuration"],"team_oracle":alternate.receipt()},
                expected_override_sha256=digest(asdict(BattleStartOverride(player_teams={
                    "p1":self.truth.override.player_teams["p1"],"p2":pack_team(team)},observation_format_id="gen3randombattle"))))
            job=NativeOracleTimedAdapterJob(**kwargs)
            job._check_source(f.source)  # Independently consistent, but not archive truth.
            self.assertEqual(job._inputs(),job._frozen_inputs)
            return job
        with patch("pokezero.mcts_eval.search_over_raw_native_oracle_run.NativeOracleTimedAdapterJob",side_effect=changed), \
                self.assertRaisesRegex(ValueError,"returned A1 job differs"):
            prepare_archived_native_oracle_timed_job(f.archive,f.f.source,contract=f.f.contract,
                root_id=f.f.root_id,configuration=self.cfg,checkpoint_contract=SimpleNamespace(
                    showdown_source_sha256="a"*64,checkpoint_sha256="b"*64,checkpoint_path="not-a-model"),
                showdown_root="not-a-simulator",runtime_configuration=self.runtime,set_source=f.source)

    def owner(self, *, zero=False, attempts=4, behavior="normal"):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        f=owner_fixture.NativeTimedOwnerTests();f.setUp();self.addCleanup(f.doCleanups)
        self.runtime=observed_runtime(self.cfg,f.files[str(f.checkpoint)])
        self.runtime["team_oracle"]=self.truth.f.oracle.receipt()
        self.inputs.update(runtime_configuration=self.runtime,runtime_sha256=digest(self.runtime))
        selected=self.selected(zero=zero,attempts=attempts);job=AuthoredOracleOwner(selected,behavior)
        directory=root/"attempt"
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="a"*64,
            verify_admission=lambda:None,contract=NativeOracleTimedRunContract(),kind="native-oracle-timed-run",
            work_phase="selection",validate_result=self.read,
            _synthetic_watchdog_seconds=.2 if behavior == "timeout" else None)
        inputs={k:v for k,v in self.inputs.items() if k != "contract"}
        inputs.update(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,file_hashes=f.files,checkpoint_path=str(f.checkpoint),
            checkpoint_sha256=f.files[str(f.checkpoint)],showdown_root=str(f.showdown))
        return terminal,inputs

    def test_actual_owned_zero_and_positive_receipts_read_independently(self):
        for zero,attempts in ((False,2),(True,4),(True,0)):
            terminal,inputs=self.owner(zero=zero,attempts=attempts)
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            self.assertEqual(terminal["status"],"NATIVE_ORACLE_TIMED_RUN_COMPLETE")
            reader=NativeOracleTimedRunValidator(**inputs,allow_synthetic=True)
            self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],0 if zero else 16)
            self.assertFalse(reader.last_validation["runtime_qualified"])
            self.assertIsNone(NativeOracleTimedRunValidator(**inputs)(terminal["result"]))
            reader.catalog_sha="f"*64
            with self.assertRaises(ValueError):reader(terminal["result"])

    def test_actual_owned_failures_and_watchdog_do_not_become_unqualification(self):
        for behavior in ("failure","cleanup","timeout"):
            terminal,inputs=self.owner(zero=True,attempts=0,behavior=behavior)
            self.assertEqual(terminal["supervisor_exit_code"],124 if behavior == "timeout" else 1)
            self.assertIsNone(terminal["result"])
            with self.assertRaises((ValueError,FileNotFoundError)):
                NativeOracleTimedRunValidator(**inputs,allow_synthetic=True)(self.selected(zero=True,attempts=0))
