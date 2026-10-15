"""Actual reference workers and fresh toy bridge clones, no scientific games."""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceConfig
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision, ParallelTrajectorySearch, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_reference_values import (
    ReferenceValueLabelRecipe, OwnedValueRuntime, ReferenceValueRunValidator, read_completed_reference_values,
)
from pokezero.local_showdown import LocalShowdownConfig
from test_search_over_raw_outcomes import EntrypointSyntheticEnv, parser_snapshot
from test_search_over_raw_leaves import LeafEnv, evaluator
from tests.test_paper_reference_showdown import observation
from pokezero.public_decision_corpus import PublicObservation
from unittest.mock import patch
from pokezero.mcts_eval.search_over_raw_value_diagnostics import calibration_bounds
import pickle
import sys
from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract,FixedWorkDiagnosticAdapter
from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
from test_search_over_raw_supervision import BINDING,INPUT_BINDING,selection_inputs


def fixture_observation(seat, turn):
    obs=deepcopy(observation(seat))
    obs.numeric_features=((float(turn),),)
    return PublicObservation.from_observation(obs).to_observation(belief_view=obs.metadata["belief_view"])


PRIVATE_SENTINEL="PRIVATE-SNAPSHOT-REQUEST-SENTINEL-DO-NOT-EXPORT"


class FixtureLabelEnv(EntrypointSyntheticEnv):
    synthetic_process_fixture = True
    def observe(self, seat):
        return fixture_observation(seat,self._latest_turn)


class PrivateFailureLabelEnv(FixtureLabelEnv):
    def _read_event(self,*,timeout):
        event=super()._read_event(timeout=timeout)
        if event and event.get("type")=="cloned":
            raise ValueError(PRIVATE_SENTINEL+repr(event))
        return event


from test_search_over_raw_outcomes import OwnedSyntheticTransport,snapshot,BINDING as RAW_BINDING


class CappedRawTransport(OwnedSyntheticTransport):
    def terminal(self):return None
    def _submit_step_choices(self,*,choices,payload):
        self.index+=1
        self._exchange(payload,[dict(type="choice_ack",player=seat,choice=choice) for seat,choice in choices.items()]
            +[dict(type="ready",requested=["p1"])])


class CapturedFixtureEnv(LeafEnv):
    def __init__(self, config, mode="complete"):
        super().__init__()
        self.config = config
        self.mode=mode
        self.original = parser_snapshot(FixtureLabelEnv(config))
    def observe(self, seat):
        return fixture_observation(seat,3+self.index)
    def snapshot(self):
        if self.mode=="snapshot-error":
            from pokezero.local_showdown import LocalShowdownError
            raise LocalShowdownError(PRIVATE_SENTINEL)
        return replace(deepcopy(self.original), latest_turn=3+self.index)
    def legal_actions(self, seat):
        return (True,True)
    def step(self, actions):
        from types import SimpleNamespace
        super().step(actions)
        return SimpleNamespace(terminal=self.terminal())


class FixtureBaseRuntime:
    def __init__(self, config, mode="complete"):
        self.config,self.mode=config,mode
    def prepare(self, request):
        def sample(rng):
            return ShowdownTrajectoryWorld(CapturedFixtureEnv(self.config,self.mode), subject="p1",
                evaluator=evaluator, release=lambda:None)
        prepared=PreparedDecision(sample(None).frame().subject,
            lambda state:Evaluation((.4,.6),.3), sample)
        if self.mode=="evidence-error":
            def evidence():raise ValueError(PRIVATE_SENTINEL)
            prepared.evidence=evidence
        return prepared
    def close(self):
        if self.mode=="close-error":raise RuntimeError(PRIVATE_SENTINEL)


class FixtureOwnedRuntime(OwnedValueRuntime):
    def _new_bank(self, root_key):
        bank = super()._new_bank(root_key)
        bank.env_factory = PrivateFailureLabelEnv if self.base.mode=="label-error" else FixtureLabelEnv
        return bank


@dataclass(frozen=True)
class FixtureOwnedFactory:
    config: object
    recipe: object
    mode: str = "complete"
    requires_owned_value_generation = True
    def __call__(self, index):
        return FixtureOwnedRuntime(FixtureBaseRuntime(self.config,self.mode), self.recipe, index)


def fixture_binding(config):
    return dict(INPUT_BINDING,information_key=decision_state(CapturedFixtureEnv(config).observe("p1"),player="p1").key.hex())


@dataclass(frozen=True)
class AuthoredOwnedValueJob:
    """Real fixed-work pool/bridges; input/atomic manifest is authored, not qualified."""
    config: object
    recipe: object
    configuration: object
    synthetic_process_fixture=True
    def __call__(self,ownership,phase):
        phase("startup")
        with ParallelTrajectorySearch(ReferenceConfig(.5,1),FixtureOwnedFactory(self.config,self.recipe),
                workers=20,owned_process_receipts=ownership) as pool:
            phase("selection")
            root=decision_state(CapturedFixtureEnv(self.config).observe("p1"),player="p1")
            measured=pool.search(None,root,battle_id="authored-fixed-root",seed=7,trajectories_per_worker=4)
            assert measured.result.trajectories==80
            values=pool.finish_diagnostics(measured)
        summary=read_completed_reference_values(pool,measured,values,self.recipe)
        import test_search_over_raw_atomic_work as atomic_fixture
        from test_paper_reference import ToyWorld
        authored_root=replace(atomic_fixture.ROOT,key=root.key)
        with patch.dict(INPUT_BINDING,fixture_binding(self.config)), \
                patch.object(atomic_fixture,"ROOT",authored_root), \
                patch.object(atomic_fixture,"ToyWorld",side_effect=lambda **kw:ToyWorld(root=authored_root,**kw)):
            evidence=selection_inputs("reference")  # Generate all authored manifest hashes with the same binding.
        evidence["worker_pids"]=list(pool.worker_pids)
        evidence["decision_id"]=measured.decision_id
        selected=dict(root_id="same-root:"+self.configuration.identity,statistics_root_id="same-root",
            configuration_sha256=self.configuration.identity,selection_seed=7,runtime_sha256=BINDING,
            fixed_work_diagnostic=dict(contract=FixedWorkDiagnosticContract().receipt(),accepted_actual=80),
            evidence=evidence)
        bundle=dict(schema="pokezero.selected-owned-reference-values.v1",root_id=selected["root_id"],
            selected_receipt_sha256=digest(selected),values=values,summary=summary)
        selected["owned_reference_value_diagnostic"]=bundle
        phase("cleanup")
        return selected


class OwnedReferenceValueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pokezero-owned-reference-values-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        sim = self.root / "showdown/dist/sim"
        sim.mkdir(parents=True)
        text = Path(__file__).with_name("test_owned_continuation.mjs").read_text()
        for variable, name in (("fakeRuntime","index.js"),("fakeState","state.js")):
            (sim/name).write_text(re.search(r"const "+variable+r" = `([\s\S]*?)`;",text).group(1))
        checkpoint = self.root / "not-a-model"
        checkpoint.write_bytes(b"authored unused weights")
        self.config = LocalShowdownConfig(showdown_root=str(self.root/"showdown"),
            set_belief_source=False, read_timeout_seconds=2.)
        self.recipe = ReferenceValueLabelRecipe(71, outcome_runtime_binding(self.config,checkpoint_path=checkpoint),
            leaves_per_worker=1)

    def run_pool(self):
        owned = self.root / "owned-groups"
        owned.mkdir()
        ownership = dict(directory=str(owned),controller_pid=os.getpid(),registration_sha256="b"*64,
            protocol="active-generations-v1")
        pool = ParallelTrajectorySearch(ReferenceConfig(.5,1), FixtureOwnedFactory(self.config,self.recipe),
            workers=2, owned_process_receipts=ownership)
        self.addCleanup(pool.close)
        root = decision_state(CapturedFixtureEnv(self.config).observe("p1"),player="p1")
        try:
            selected = pool.search(None,root,battle_id="authored-root",seed=7,trajectories_per_worker=1)
            values = pool.finish_diagnostics(selected)
        except ReferenceRefusal as error:
            raise AssertionError(pool.last_evidence) from error
        return pool,selected,values

    def test_actual_worker_fresh_clones_raw_step_zero_and_normal_parent_readback(self):
        pool, selected, values = self.run_pool()
        with self.assertRaisesRegex(ValueError,"normally closed"):
            read_completed_reference_values(pool,selected,values,self.recipe)
        pool.close()
        summary = read_completed_reference_values(pool,selected,values,self.recipe)
        self.assertEqual(summary["sampled_leaves"],2)
        self.assertEqual(summary["labels"],16)
        self.assertTrue(summary["independent_terminal_labels"])
        self.assertFalse(summary["resource_cost_complete"])
        for worker in values:
            evidence = worker["evidence"]
            self.assertFalse(evidence["private_snapshots_exported"])
            self.assertFalse(evidence["labels_used_for_backup"])
            leaf,proof = evidence["leaves"][0],evidence["proofs"][0]
            self.assertEqual(len(leaf["labels"]),8)
            self.assertTrue(all(l["status"]=="COMPLETE" and l["signed_outcome"]==1 for l in leaf["labels"]))
            directory = Path(proof["directory"])
            transport = json.loads((directory/"bridge-transport.json").read_bytes())["records"]
            clones = [r["payload"] for r in transport if r["kind"]=="command" and r["payload"]["type"]=="clone_snapshot"]
            choices = [r["payload"]["choices"] for r in transport if r["kind"]=="command" and r["payload"]["type"]=="choices"]
            self.assertEqual(len(clones),8)
            self.assertEqual(len({c["battleId"] for c in clones}),8)
            self.assertTrue(all(c=={"p1":"move 2","p2":"move 2"} for c in choices))
            encoded = json.dumps(values)
            self.assertNotIn("snapshotJson\"",encoded)
            self.assertNotIn("annotation_cache",encoded)
            self.assertFalse(list(directory.parent.parent.parent.parent.joinpath("active").glob("*.json")))
        with self.assertRaises(ReferenceRefusal):
            pool.finish_diagnostics(selected)

    def test_modified_prediction_and_changed_receipt_cannot_bank_labels(self):
        pool,selected,values = self.run_pool()
        pool.close()
        changed = deepcopy(values)
        changed[0]["evidence"]["leaves"][0]["model_signed_value"] = -.9
        with self.assertRaisesRegex(ValueError,"immutable|prediction"):
            read_completed_reference_values(pool,selected,changed,self.recipe)
        proof=values[0]["evidence"]["proofs"][0]
        path=Path(proof["directory"])/"bridge-close.json"
        original=path.read_bytes()
        row=json.loads(original);row["forced_cleanup"]=True
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,"receipt bytes"):
            read_completed_reference_values(pool,selected,values,self.recipe)
        path.write_bytes(original)
        self.assertEqual(read_completed_reference_values(pool,selected,values,self.recipe)["labels"],16)

    def test_caller_cannot_rehash_changed_label_rows_or_subject_into_acceptance(self):
        pool,selected,values=self.run_pool();pool.close()
        for field,value in (("signed_outcome",-1),("status","CAPPED_UNCERTAIN"),("boundaries",249),("subject","p2")):
            changed=deepcopy(values)
            leaf=changed[0]["evidence"]["leaves"][0]
            if field=="subject":leaf[field]=value
            else:
                leaf["labels"][0][field]=value
                if field=="status":leaf["labels"][0]["signed_outcome"]=None
            changed[0]["evidence"]["proofs"][0]["labels_sha256"]=digest(leaf["labels"])
            changed[0]["evidence"]["calibration"]=calibration_bounds(changed[0]["evidence"]["leaves"])
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"immutable"):
                read_completed_reference_values(pool,selected,changed,self.recipe)

    def test_actual_adapter_finish_closes_pool_then_reads_original_generation(self):
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        pool,measured,values=self.run_pool()
        # The pool's diagnostics is one-shot. Use its actual retained output at
        # this adapter boundary, not a fabricated label producer.
        adapter=object.__new__(PublicModelSearchAdapter)
        selected=dict(root_id="authored-selected",runtime_sha256="c"*64)
        adapter._owned_value_recipe=self.recipe
        adapter._closed=adapter._poisoned=False
        adapter._pending_value_diagnostics=(selected,digest(selected),measured)
        adapter._pool=pool;adapter._decider=None
        with patch.object(pool,"finish_diagnostics",return_value=values):
            bundle=adapter.finish_owned_reference_values(selected)
        self.assertTrue(adapter._closed)
        self.assertEqual(bundle["selected_receipt_sha256"],digest(selected))
        self.assertEqual(bundle["summary"]["labels"],16)
        with self.assertRaises(ValueError):adapter.finish_owned_reference_values(selected)

    def test_external_reader_requires_enclosing_terminal_before_label_read(self):
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
        reader=object.__new__(ReferenceValueRunValidator)
        with patch.object(FixedWorkRunValidator,"__call__",side_effect=ValueError("original enclosing run not terminal")), \
                patch("pokezero.mcts_eval.search_over_raw_reference_values.read_reference_value_receipts") as read:
            with self.assertRaisesRegex(ValueError,"not terminal"):
                reader({})
        read.assert_not_called()

    def test_world_input_wrapper_keeps_original_atomic_observer_and_model_capture(self):
        import random,time
        from pokezero.mcts_eval.search_over_raw_value_diagnostics import _ValueRuntime,ValueDiagnosticContract
        from pokezero.mcts_eval.search_over_raw_world_inputs import ReferenceInputWorld
        original=object()
        config=self.config
        class WrappedBase(FixtureBaseRuntime):
            def prepare(self,request):
                prepared=super().prepare(request)
                sample=prepared.sample_world
                prepared.sample_world=lambda rng:ReferenceInputWorld(sample(rng),dict(status="ROOT_VALIDATED"))
                prepared.atomic_work=original
                return prepared
        runtime=_ValueRuntime(WrappedBase(config),ValueDiagnosticContract(71,time.perf_counter()+60),0)
        prepared=runtime.prepare(None)
        self.assertIs(prepared.atomic_work,original)
        world=prepared.sample_world(random.Random(7))
        self.assertIsInstance(world,ReferenceInputWorld)
        state=world.frame().subject
        world.evaluate(state)
        self.assertEqual(runtime.bank.seen,1)
        self.assertTrue(world.draw["consumed_world_input"]["consumed"])
        world.close();runtime.close()

    def test_raw_label_reader_validates_boundary_cap_without_scoring_it(self):
        from pokezero.mcts_eval.search_over_raw import _run_continuation_cell
        from pokezero.mcts_eval.search_over_raw_outcomes import OwnedContinuationLifecycle,RAW_LABEL_POLICY,validate_observed_outcome
        from pokezero.mcts_eval.search_over_raw_reference_values import RawLabelCapture
        env,state=CappedRawTransport(),snapshot()
        capture=RawLabelCapture(RAW_BINDING,"a"*64)
        lifecycle=OwnedContinuationLifecycle(env,expected_snapshot=state,root_binding=RAW_BINDING)
        law=dict(root_id=RAW_BINDING["root_id"],action=0,replicate=0,subject="p1",namespace="cap-fixture",
            max_boundaries=250,policy=RAW_LABEL_POLICY)
        with capture.installed(env):
            lifecycle.prepare(state,law);capture.begin_cell(state,**law)
            result=_run_continuation_cell(env,state,"p1",0,0,lambda obs:([0,1],[.4,.6]),
                law["namespace"],law["root_id"],250,capture,lifecycle,raw_from_start=True)
            lifecycle.release()
            row=dict(root_id=law["root_id"],action=0,replicate=0,**result)
            capture.finish_cell(row)
        contract=dict(root_binding=RAW_BINDING,root_id=law["root_id"],subject="p1",namespace=law["namespace"],
            max_boundaries=250,replicates=8,selection_seed=RAW_BINDING["source_seed"],policy=RAW_LABEL_POLICY)
        self.assertEqual(row["status"],"CAPPED")
        self.assertIsNone(row["signed_outcome"])
        cell=capture.export()["cells"][0]
        self.assertEqual(validate_observed_outcome(row,contract,cell,expected_snapshot=state,runtime_sha256="a"*64),digest(row))
        changed=deepcopy(row);changed["signed_outcome"]=0
        with self.assertRaises(ValueError):validate_observed_outcome(changed,contract,cell,expected_snapshot=state,runtime_sha256="a"*64)

    def test_checkpoint_mismatch_refuses_adapter_before_pool_spawn(self):
        from types import SimpleNamespace
        cfg=SearchConfiguration("reference",seconds=1.,workers=20)
        for contract in (SimpleNamespace(checkpoint_path=str(self.root/"another"),checkpoint_sha256="d"*64),
                SimpleNamespace(checkpoint_path=self.recipe.runtime["checkpoint_path"],checkpoint_sha256="d"*64)):
            with patch("pokezero.mcts_eval.paper_reference_parallel.mp.get_context") as spawn:
                with self.assertRaisesRegex(ValueError,"actual reference champion"):
                    FixedWorkDiagnosticAdapter(cfg,fixed_work=FixedWorkDiagnosticContract(),checkpoint_contract=contract,
                        showdown_root=str(self.root/"showdown"),owned_process_receipts={"protocol":"active-generations-v1"},
                        owned_value_recipe=self.recipe)
            spawn.assert_not_called()

    def test_actual_completed_outer_run_twenty_generations_and_bridge_labels(self):
        cfg=SearchConfiguration("reference",seconds=1.,workers=20)
        job=AuthoredOwnedValueJob(self.config,self.recipe,cfg)
        binding=fixture_binding(self.config)
        directory=self.root/"outer"
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256=BINDING,
            verify_admission=lambda:True,contract=FixedWorkDiagnosticContract(),kind="fixed-work",work_phase="selection",
            validate_result=lambda selected:derive_fixed_work_accounting(selected["evidence"],arm="reference",
                binding=binding,statistics_root_id="same-root",leaf="model"))
        if terminal["supervisor_exit_code"] != 0:
            failed=json.loads((directory/"worker-result.json").read_bytes())
            derive_fixed_work_accounting(failed["result"]["evidence"],arm="reference",binding=binding,
                statistics_root_id="same-root",leaf="model")
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        files=fixed_work_reader_source_hashes()
        files.update(self.recipe.runtime["files"])
        files[str(Path(sys.executable).resolve())]=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
        checkpoint=self.recipe.runtime["checkpoint_path"]
        reader=ReferenceValueRunValidator(recipe=self.recipe,directory=directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=BINDING,configuration=cfg,
            root_id="same-root:"+cfg.identity,statistics_root_id="same-root",selection_seed=7,runtime_sha256=BINDING,
            consumed_binding=binding,file_hashes=files,checkpoint_path=checkpoint,
            checkpoint_sha256=files[checkpoint],showdown_root=str(self.root/"showdown"),allow_synthetic=True)
        selected=terminal["result"]
        self.assertEqual(reader(selected),digest(selected))
        self.assertEqual(reader.last_validation["reference_value_labels"]["labels"],160)
        self.assertFalse(reader.last_validation["runtime_qualified"])
        path=directory/"terminal.json"
        original=path.read_bytes();data=json.loads(original);data["worker_exit_code"]=7
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"externally reaped"):
            reader(selected)
        path.write_bytes(original)

    def test_owned_factory_requires_durable_generations_before_any_start(self):
        with self.assertRaisesRegex(ReferenceRefusal,"immutable"):
            ParallelTrajectorySearch(ReferenceConfig(.5,1),FixtureOwnedFactory(self.config,self.recipe),workers=2)

    def prospective_recipe(self):
        from types import SimpleNamespace
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
            ReferenceValueFixedWorkRecipe,fixed_work_runtime_recipe)
        checkpoint=SimpleNamespace(checkpoint_path=self.recipe.runtime["checkpoint_path"],
            checkpoint_sha256=self.recipe.runtime["files"][self.recipe.runtime["checkpoint_path"]],
            showdown_root=str(self.root/"showdown"),showdown_source_sha256="a"*64)
        cfg=SearchConfiguration("reference",seconds=1.,workers=20)
        factory=ShowdownWorkerFactory(checkpoint.checkpoint_path,checkpoint.checkpoint_sha256,
            checkpoint.showdown_root,checkpoint.showdown_source_sha256)
        runtime=fixed_work_runtime_recipe(cfg,checkpoint,reference_factory=factory,
            owned_reference_value_recipe=self.recipe)
        return cfg,checkpoint,ReferenceValueFixedWorkRecipe(runtime=runtime,reference_factory=factory,
            owned_reference_value_recipe=self.recipe)

    def test_prospective_label_recipe_round_trip_and_historical_entry_rejection(self):
        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import FixedWorkRecipe
        from pokezero.mcts_eval.search_over_raw_exploration_entry import (
            _decode,_entry_diagnostic_recipe,REFERENCE_VALUE_WORKFLOW_SCHEMA,WORKFLOW_SCHEMAS)
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        _,_,recipe=self.prospective_recipe()
        self.assertEqual(_state_payload(_decode(_state_payload(recipe))),_state_payload(recipe))
        self.assertIs(_entry_diagnostic_recipe(REFERENCE_VALUE_WORKFLOW_SCHEMA,recipe),self.recipe)
        for schema in WORKFLOW_SCHEMAS[:-1]:
            with self.subTest(schema=schema),self.assertRaises(ValueError):_entry_diagnostic_recipe(schema,recipe)
            self.assertIsNone(_entry_diagnostic_recipe(schema,FixedWorkRecipe(unqualified_reason="not available")))

    def test_labeled_manifest_preserves_all_six_recipes_and_original_slots(self):
        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
            FixedWorkRecipe,diagnostic_configurations,diagnostic_manifest)
        import test_search_over_raw_reconciled as ledger_fixture
        helper=ledger_fixture.ReconciledLedgerTest();helper.setUp();self.addCleanup(helper.tearDown)
        cfg,checkpoint,recipe=self.prospective_recipe()
        recipes={c.identity:FixedWorkRecipe(unqualified_reason="authored unavailable") for c in diagnostic_configurations()}
        recipes[cfg.identity]=recipe
        def manifest(files=self.recipe.runtime["files"]):
            return diagnostic_manifest(helper.contract,recipes,checkpoint_contract=checkpoint,
                showdown_root=checkpoint.showdown_root,file_hashes=files)
        with patch("subprocess.Popen") as spawn:
            value=manifest()
            self.assertEqual(value["schema"],"pokezero.search-over-raw.fixed-diagnostic-manifest.v2")
            self.assertEqual((len(value["recipes"]),len(value["slots"])),(6,192))
            self.assertFalse(value["diagnostic_results_used_for_selection"])
            self.assertFalse(value["collection_authorized"])
            missing=dict(self.recipe.runtime["files"]);missing.pop(next(iter(missing)))
            with self.assertRaisesRegex(ValueError,"prospectively pinned"):manifest(missing)
            recipes[cfg.identity]=FixedWorkRecipe(runtime=recipe.runtime,reference_factory=recipe.reference_factory)
            with self.assertRaisesRegex(ValueError,"silently acquire"):manifest()
        spawn.assert_not_called()

    def test_archived_recipe_handoff_binds_labels_before_construction_without_launch(self):
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_reference_fixed_work_job
        from pokezero.mcts_eval.search_over_raw_supervision import FixedWorkAdapterJob
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        import test_search_over_raw_archive as archive_fixture
        f=archive_fixture.ExplorationRootCrosswalkTests();f.setUp();self.addCleanup(f.doCleanups)
        cfg,checkpoint,recipe=self.prospective_recipe()
        args=dict(contract=f.contract,root_id=f.root_id,configuration=cfg,checkpoint_contract=checkpoint,
            showdown_root=checkpoint.showdown_root,runtime_sha256=digest(recipe.runtime),
            reference_factory=recipe.reference_factory,owned_reference_value_recipe=self.recipe)
        with patch("subprocess.Popen") as spawn:
            job,_=prepare_archived_reference_fixed_work_job(f.archive,f.source,**args)
            self.assertEqual(_state_payload(job.owned_reference_value_recipe),_state_payload(self.recipe))
            def dropped(**kwargs):
                kwargs["owned_reference_value_recipe"]=None
                return FixedWorkAdapterJob(**kwargs)
            with patch("pokezero.mcts_eval.search_over_raw_supervision.FixedWorkAdapterJob",side_effect=dropped):
                with self.assertRaisesRegex(ValueError,"prospective envelope"):
                    prepare_archived_reference_fixed_work_job(f.archive,f.source,**args)
        spawn.assert_not_called()

    def test_original_request_zero_event_job_closes_and_reads_own_label_generation(self):
        from dataclasses import fields
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_reference_fixed_work_job
        from test_search_over_raw_diagnostic_producer import AuthoredPanelFixedReferenceJob
        import test_search_over_raw_archive as archive_fixture
        f=archive_fixture.ExplorationRootCrosswalkTests();f.setUp();self.addCleanup(f.doCleanups)
        cfg,checkpoint,recipe=self.prospective_recipe()
        prepared,bound=prepare_archived_reference_fixed_work_job(f.archive,f.source,contract=f.contract,
            root_id=f.root_id,configuration=cfg,checkpoint_contract=checkpoint,showdown_root=checkpoint.showdown_root,
            runtime_sha256=digest(recipe.runtime),reference_factory=recipe.reference_factory,
            owned_reference_value_recipe=self.recipe)
        args={field.name:getattr(prepared,field.name) for field in fields(prepared)}
        args["context"]=deepcopy(f.context)
        job=AuthoredPanelFixedReferenceJob(**args)
        directory=self.root/"empty-outer"
        value=_supervise_owned_job(job,directory=directory,binding_sha256=BINDING,verify_admission=lambda:True,
            contract=FixedWorkDiagnosticContract(),kind="fixed-work",work_phase="selection",
            validate_result=lambda selected:derive_fixed_work_accounting(selected["evidence"],arm="reference",
                binding=job._consumed_binding,statistics_root_id=f.root_id,leaf="model"))
        error=directory/"authored-reference-error.json"
        self.assertEqual(value["supervisor_exit_code"],0,error.read_text() if error.exists() else value)
        summary=value["result"]["owned_reference_value_diagnostic"]["summary"]
        self.assertEqual(summary["labels"],0)
        self.assertFalse(summary["independent_terminal_labels"])

    def test_recipe_rejects_work_or_sampling_changes(self):
        for kwargs in ({"sample_seed":True},{"replicates":7},{"maximum_boundaries":249},
                {"leaves_per_worker":0},{"maximum_snapshot_bytes":9*1024*1024}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                ReferenceValueLabelRecipe(**{ "sample_seed":71,"runtime":self.recipe.runtime,**kwargs})

    def test_owned_worker_failure_channels_redact_private_capture_label_evidence_and_close(self):
        root=decision_state(CapturedFixtureEnv(self.config).observe("p1"),player="p1")
        for mode in ("snapshot-error","label-error","evidence-error","close-error"):
            owned=self.root/mode;owned.mkdir()
            ownership=dict(directory=str(owned),controller_pid=os.getpid(),registration_sha256="b"*64,
                protocol="active-generations-v1")
            pool=ParallelTrajectorySearch(ReferenceConfig(.5,1),FixtureOwnedFactory(self.config,self.recipe,mode),
                workers=1,owned_process_receipts=ownership)
            try:
                with self.subTest(mode=mode),self.assertRaises(ReferenceRefusal):
                    selected=pool.search(None,root,battle_id="authored-failure",seed=7,trajectories_per_worker=1)
                    if mode=="label-error":pool.finish_diagnostics(selected)
                    elif mode=="close-error":pool.close()
                self.assertNotIn(PRIVATE_SENTINEL,json.dumps(pool.last_evidence,default=str))
            finally:
                try:pool.close()
                except ReferenceRefusal:pass  # Actual forced close remains failure, never success.
            self.assertNotIn(PRIVATE_SENTINEL,json.dumps(pool.cleanup_evidence,default=str))


if __name__ == "__main__":
    unittest.main()
