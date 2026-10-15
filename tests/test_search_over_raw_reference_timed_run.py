"""Real disposable toy owners plus authored model/simulator/runtime bindings."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_reference_timed_run import (
    ReferenceTimedRunContract, ReferenceTimedAdapterJob, ReferenceTimedRunValidator, supervise_reference_timed,
)
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
from pokezero.mcts_eval.paper_reference import ReferenceConfig
from pokezero.mcts_eval.search_over_raw_timed_reference import validate_reference_timed_inputs
from pokezero.mcts_eval.search_over_raw_world_inputs import reference_manifest
from test_search_over_raw_timed_reference import runtime, selection_fixture, TimedAtomicToyRuntime
from test_search_over_raw_timed_run import TimedContext
from test_search_over_raw_supervision import SyntheticPublicState, SyntheticBelief
from test_search_over_raw_atomic_work import BINDING
from test_paper_reference_parallel import Request
from test_paper_reference import ROOT

RUN_BINDING = "a"*64


def owner_runtime(cfg, checkpoint, showdown, checkpoint_sha):
    value = runtime(cfg)
    factory = ShowdownWorkerFactory(checkpoint, checkpoint_sha, showdown, BINDING["set_source_hash"])
    value.update(checkpoint_sha256=checkpoint_sha, reference_factory=asdict(factory),
        owned_process_receipts=dict(enabled=True, publishes_before_detachment=True, qualifies_uninstrumented_runtime=False))
    return value, factory


@dataclass(frozen=True)
class ToyReferenceOwner:
    cfg: object
    observed: dict
    zero: bool = False
    failure: bool = False
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup")
        pool = ParallelTrajectorySearch(ReferenceConfig(.5,1.), TimedAtomicToyRuntime, workers=20,
            initial_dispatch_workers=6, owned_process_receipts=ownership)
        try:
            phase("selection")
            began = time.perf_counter()
            if self.zero: time.sleep(1.01)
            result = pool.search(Request(failure=self.failure), ROOT, battle_id="search-over-raw-root:root", seed=7,
                deadline_seconds=1., deadline_at=began+1., adapter_started_at=began, timed_zero_work=True)
            evidence = asdict(result)
            # Toy labels are mapped to the authored encoded-action surface;
            # this is not actual Pokémon encoder/evaluator correspondence.
            def convert(value):
                if type(value) is dict:
                    for key,item in value.items():
                        if key == "actions" and type(item) in (list,tuple) and tuple(item) == ROOT.actions:
                            value[key] = ["action:0", "action:1"]
                        elif key == "action" and item in ROOT.actions:
                            value[key] = "action:"+str(ROOT.actions.index(item))
                        else: convert(item)
                elif type(value) in (list,tuple):
                    for item in value: convert(item)
            convert(evidence)
            evidence["worker_pids"] = list(evidence["worker_pids"])
            evidence["consumed_world_inputs"] = reference_manifest(evidence,BINDING)
            readout = validate_reference_timed_inputs(evidence,binding=BINDING,
                battle_id="search-over-raw-root:root",selection_seed=7,nominal_seconds=1.)
            evidence["timed_reference_readout"] = readout
            elapsed = time.perf_counter()-began
            selected = dict(root_id="root:"+self.cfg.identity,statistics_root_id="root",selection_seed=7,
                configuration_sha256=self.cfg.identity,runtime_sha256=digest(self.observed),
                independent_root_statistics=True,scientific_strength_evidence=False,evidence=evidence,
                status=readout["status"],action=None if self.zero else int(evidence["result"]["action"][7:]),
                elapsed_seconds=elapsed,nominal_search_seconds=1.,exceeded_nominal_seconds=elapsed>1.,
                reference_timed_run_contract=ReferenceTimedRunContract().receipt(),
                observed_runtime_configuration=deepcopy(self.observed))
            if self.zero: selected.update(attempt_consumed=True,retry_authorized=False,
                continuation_authorized=False,runtime_qualified=False)
            return selected
        finally:
            phase("cleanup");pool.close()


class ReferenceOwnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.checkpoint = self.root/"unused-model";self.checkpoint.write_bytes(b"authored unused checkpoint")
        self.showdown = self.root/"unused-showdown";dist = self.showdown/"dist/sim";dist.mkdir(parents=True)
        for name in ("index.js","state.js"): (dist/name).write_text("// unused authored simulator")
        self.files = fixed_work_reader_source_hashes()
        for path in [self.checkpoint,Path(sys.executable).resolve(),*dist.glob("*.js")]:
            self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cfg = SearchConfiguration("reference",workers=20,seconds=1.)
        self.observed,self.factory = owner_runtime(self.cfg,str(self.checkpoint),str(self.showdown),self.files[str(self.checkpoint)])

    def inputs(self,directory,job):
        return dict(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256=RUN_BINDING,
            configuration=self.cfg,root_id="root:"+self.cfg.identity,statistics_root_id="root",selection_seed=7,
            runtime_configuration=self.observed,runtime_sha256=digest(self.observed),consumed_binding=BINDING,
            legal_action_mask=(True,True),decision_round=3,file_hashes=self.files,checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.files[str(self.checkpoint)],showdown_root=str(self.showdown),allow_synthetic=True)

    def run_owner(self,zero=False,failure=False):
        job = ToyReferenceOwner(self.cfg,self.observed,zero,failure)
        directory = self.root/("zero" if zero else "positive")
        reader = ReferenceTimedRunValidator(**self.inputs(directory,job))
        terminal = _supervise_owned_job(job,directory=directory,binding_sha256=RUN_BINDING,
            verify_admission=lambda:None,contract=ReferenceTimedRunContract(),kind="reference-timed-run",
            work_phase="selection",validate_result=reader._read_work)
        return terminal,reader,directory

    def test_real_twenty_owned_generations_positive_and_zero(self):
        for zero in (False,True):
            terminal,reader,directory = self.run_owner(zero)
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            self.assertEqual(terminal["status"],"REFERENCE_TIMED_RUN_COMPLETE")
            selected = terminal["result"]
            self.assertEqual(reader(selected),digest(selected))
            self.assertEqual(reader.last_validation["owned_generation_pids"],selected["evidence"]["worker_pids"])
            self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)
            self.assertEqual(reader.last_validation["actual_work"]["accepted_units"] == 0,zero)
            for flag in ("runtime_qualified","resource_cost_complete","scientific_collection_authorized",
                    "loaded_binary_dependency_closure_validated","independent_evaluator_correctness"):
                self.assertFalse(reader.last_validation[flag])
            reader.allow_synthetic=False
            with self.assertRaisesRegex(ValueError,"inputs changed"):reader(selected)

    def test_generation_index_exit_extra_and_dynamic_mutation_refuse(self):
        terminal,reader,directory=self.run_owner(True);selected=terminal["result"]
        path=next((directory/"owned-groups").glob("group-*.json"));original=path.read_bytes()
        closed=directory/"owned-groups"/("closed-"+path.name);closed_original=closed.read_bytes()
        for key,value in (("worker_index",True),("controller_pid",True),("pid",True)):
            row=json.loads(original);row[key]=value;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):reader(selected)
            path.write_bytes(original)
        row=json.loads(closed_original);row["worker_exit_code"]=1;closed.write_text(json.dumps(row))
        with self.assertRaises(ValueError):reader(selected)
        closed.write_bytes(closed_original)
        for key,value in (("clean_acknowledgment",False),("forced_cleanup",True),
                ("runtime_close_failed",True),("cleanup_exception_count",True),("cleanup_protocol","old-exit-only")):
            row=json.loads(closed_original);row[key]=value;closed.write_text(json.dumps(row))
            with self.subTest(cleanup_field=key),self.assertRaises(ValueError):reader(selected)
            closed.write_bytes(closed_original)
        extra=directory/"owned-groups/extra.json";extra.write_text("{}")
        with self.assertRaisesRegex(ValueError,"no active/extras"):reader(selected)
        extra.unlink()
        original_read=reader._read_work
        def mutate(saved):
            result=original_read(saved);closed.write_text("{}");return result
        with patch.object(reader,"_read_work",side_effect=mutate),self.assertRaisesRegex(ValueError,"changed during"):
            reader(selected)
        closed.write_bytes(closed_original)
        original_owned=reader._read_owned
        def late_generation(*args):
            result=original_owned(*args);extra.write_text("{}");return result
        with patch.object(reader,"_read_owned",side_effect=late_generation),self.assertRaisesRegex(ValueError,"changed during"):
            reader(selected)

    def test_operational_refusal_is_not_zero_rescue(self):
        terminal,reader,_=self.run_owner(failure=True)
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(terminal["status"],"OPERATIONAL_STOP")
        with self.assertRaises((ValueError,KeyError)):reader(terminal["result"])

    def job(self):
        state=SyntheticPublicState(SyntheticBelief());observation=SimpleNamespace(legal_action_mask=(True,True))
        context=TimedContext("p1",observation,{"p1":observation,"p2":"private-canary"},state)
        request=SimpleNamespace(state=state,observation=observation,set_source_hash=BINDING["set_source_hash"])
        contract=SimpleNamespace(checkpoint_path=str(self.checkpoint),checkpoint_sha256=self.files[str(self.checkpoint)],
            showdown_source_sha256=BINDING["set_source_hash"])
        with patch("pokezero.mcts_eval.head_to_head.public_only_context",side_effect=lambda c:c), \
                patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=request), \
                patch("pokezero.mcts_eval.search_over_raw_reference_timed_run._binding",return_value=BINDING):
            return ReferenceTimedAdapterJob(self.cfg,contract,str(self.showdown),self.factory,context,
                "root:"+self.cfg.identity,7,"root",self.observed)

    def test_real_job_mocked_adapter_both_branches_cleanup_and_wrapper(self):
        for zero in (False,True):
            job=self.job();selected=selection_fixture(self.cfg,zero);selected["runtime_sha256"]=digest(self.observed)
            adapter=Mock(runtime_configuration=deepcopy(self.observed),runtime_sha256=digest(self.observed),
                _pool=SimpleNamespace(workers=20,batch_size=10,initial_dispatch_workers=6))
            adapter.select.return_value=selected;phases=[]
            source=SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING["set_source_hash"]))
            with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=source), \
                    patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter",return_value=adapter) as factory:
                result=job({},phases.append)
            self.assertEqual(phases,["startup","selection","cleanup"]);adapter.close.assert_called_once()
            self.assertTrue(factory.call_args.kwargs["timed_reference_atomic_work"])
            self.assertEqual(factory.call_args.kwargs["initial_dispatch_workers"],6)
            self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
            self.assertEqual(set(adapter.select.call_args.args[0].requested_observations),{"p1"})
            self.assertEqual(job.read_selection(result)["accepted_units"],0 if zero else 80)
            with patch("pokezero.mcts_eval.search_over_raw_reference_timed_run._supervise_owned_job",return_value={}) as owner:
                supervise_reference_timed(job,directory="never-created",binding_sha256=RUN_BINDING,verify_admission=lambda:False)
            self.assertEqual(owner.call_args.kwargs["kind"],"reference-timed-run")

    def test_job_primary_failure_survives_cleanup_failure(self):
        job=self.job();original=RuntimeError("primary failure")
        adapter=Mock(runtime_configuration=deepcopy(self.observed),runtime_sha256=digest(self.observed),
            _pool=SimpleNamespace(workers=20,batch_size=10,initial_dispatch_workers=6))
        adapter.select.side_effect=original;adapter.close.side_effect=ValueError("private details")
        source=SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING["set_source_hash"]))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=source), \
                patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter",return_value=adapter), \
                self.assertRaises(RuntimeError) as caught:job({},lambda _:None)
        self.assertIs(caught.exception,original)
        self.assertEqual(original.__notes__,["Reference timed adapter cleanup failed: ValueError"])

    def test_contract_runtime_factory_drift_rejected_before_launch(self):
        for kwargs in (dict(workers=True),dict(batch_size=1),dict(initial_dispatch_workers=20),dict(emergency_watchdog_seconds=1)):
            with self.assertRaises(ValueError):ReferenceTimedRunContract(**kwargs)
        job=self.job();changed=deepcopy(self.observed);changed["reference_factory"]["checkpoint_sha256"]="c"*64
        with self.assertRaises(ValueError):ReferenceTimedRunValidator(**dict(self.inputs(self.root/"none",ToyReferenceOwner(self.cfg,changed)),
            runtime_configuration=changed,runtime_sha256=digest(changed)))
