"""Disposable actual process ownership; authored inference, never model truth."""
from copy import deepcopy
from dataclasses import dataclass, replace
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
from pokezero.mcts_eval.search_over_raw_raw_run import (
    RawSelectionRunContract, RawSelectionJob, RawSelectionAdapter, RawSelectionRunValidator,
    raw_runtime, read_raw_selection, supervise_raw_selection,
    _runtime,
)
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
from test_search_over_raw_timed_run import TimedContext
from test_search_over_raw_supervision import SyntheticPublicState, SyntheticBelief
from test_search_over_raw_atomic_work import BINDING

PUBLIC_SHA = "d"*64
RUN_SHA = "a"*64


def witness():
    return dict(schema="pokezero.search-over-raw.public-request-consumption.v1", public_request_sha256=PUBLIC_SHA,
        consumed_binding=deepcopy(BINDING), full_supported_public_state_and_pending=True,
        private_source_snapshot_included=False, runtime_qualified=False)


def selected_fixture(cfg, runtime, *, forced=False):
    selected = dict(root_id="root:raw", statistics_root_id="root", selection_seed=7,
        configuration_sha256=cfg.identity, runtime_sha256=digest(runtime), action=0, status="SELECTED",
        elapsed_seconds=.005, nominal_search_seconds=cfg.seconds, exceeded_nominal_seconds=False,
        independent_root_statistics=True, scientific_strength_evidence=False,
        raw_selection_run_contract=RawSelectionRunContract().receipt(), observed_runtime_configuration=deepcopy(runtime),
        evidence=dict(selector="deterministic_masked_argmax", model_forwards=1, legal_actions=[0,2],
            priors=[.5,.5], signed_value=.2, consumed_binding=deepcopy(BINDING), public_request_witness=witness()))
    if forced:
        selected["evidence"].update(legal_actions=[0], priors=[1.])
    return selected


@dataclass(frozen=True)
class ToyRawOwner:
    cfg: object
    runtime: dict
    fail: bool = False
    forced: bool = False
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup"); phase("selection")
        if self.fail: raise RuntimeError("authored raw failure")
        selected = selected_fixture(self.cfg, self.runtime, forced=self.forced)
        time.sleep(.01)
        phase("cleanup")
        return selected


class RawOwnerTests(unittest.TestCase):
    def test_original_catalog_fingerprint_is_accepted_without_padding_or_aliases(self):
        cfg=SearchConfiguration("raw",seconds=1.)
        for fingerprint in ("1a2b3c4d5e6f7890","a"*64):
            runtime=raw_runtime(cfg,"b"*64,fingerprint)
            _runtime(cfg,runtime,digest(runtime))
            self.assertEqual(runtime["set_source_hash"],fingerprint)
        for fingerprint in ("a"*15,"a"*17,"A"*16,False,0):
            runtime=raw_runtime(cfg,"b"*64,fingerprint)
            with self.subTest(fingerprint=fingerprint),self.assertRaisesRegex(ValueError,"raw runtime drift"):
                _runtime(cfg,runtime,digest(runtime))
        runtime=raw_runtime(cfg,"b"*64,"1a2b3c4d5e6f7890")
        original=digest(runtime)
        runtime["set_source_hash"]="0"*48+runtime["set_source_hash"]
        with self.assertRaisesRegex(ValueError,"raw runtime drift"):_runtime(cfg,runtime,original)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.checkpoint = self.root/"unused-model"; self.checkpoint.write_bytes(b"authored unused model")
        self.showdown = self.root/"unused-showdown"; dist = self.showdown/"dist/sim"; dist.mkdir(parents=True)
        for name in ("index.js", "state.js"): (dist/name).write_text("// authored unused simulator")
        self.files = fixed_work_reader_source_hashes()
        for path in [self.checkpoint, Path(sys.executable).resolve(), *dist.glob("*.js")]:
            self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cfg = SearchConfiguration("raw", seconds=1.)
        self.runtime = raw_runtime(self.cfg, self.files[str(self.checkpoint)], BINDING["set_source_hash"])

    def reader(self, directory, job, allow_synthetic=True, mask=(True,False,True)):
        return RawSelectionRunValidator(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=RUN_SHA, configuration=self.cfg, root_id="root:raw", statistics_root_id="root",
            selection_seed=7, runtime_configuration=self.runtime, runtime_sha256=digest(self.runtime),
            consumed_binding=BINDING, legal_action_mask=mask, decision_round=None,
            file_hashes=self.files, checkpoint_path=str(self.checkpoint), checkpoint_sha256=self.files[str(self.checkpoint)],
            showdown_root=str(self.showdown), public_request_sha256=PUBLIC_SHA, allow_synthetic=allow_synthetic)

    def run_owner(self, fail=False, *, forced=False):
        job = ToyRawOwner(self.cfg, self.runtime, fail, forced)
        directory = self.root/"owner"
        reader = self.reader(directory,job,mask=(True,False,False) if forced else (True,False,True))
        terminal = _supervise_owned_job(job,directory=directory,binding_sha256=RUN_SHA,
            verify_admission=lambda:None,contract=RawSelectionRunContract(),kind="raw-selection-run",
            work_phase="selection",validate_result=reader._read_work)
        return terminal,reader,directory,job

    def test_forced_owned_raw_still_requires_one_forward_and_no_search(self):
        terminal,reader,_,_=self.run_owner(forced=True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
        work=reader.last_validation["actual_work"]
        self.assertEqual((work["observed_model_forwards"],work["accepted_units"]),(1,0))
        selected=deepcopy(terminal["result"]);selected["evidence"]["model_forwards"]=0
        with self.assertRaises(ValueError):reader._read_work(selected)

    def test_forced_raw_job_preserves_root_seed_and_exact_legal_mask(self):
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        state=SyntheticPublicState(SyntheticBelief())
        observation=SimpleNamespace(legal_action_mask=(True,False,False))
        context=TimedContext("p1",observation,{"p1":observation},state)
        request=PublicRootRequest(state=state,observation=observation,set_source_hash=BINDING["set_source_hash"])
        with patch("pokezero.mcts_eval.head_to_head.public_only_context",side_effect=lambda c:c), \
                patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=request), \
                patch("pokezero.mcts_eval.search_over_raw_raw_run._binding",return_value=BINDING), \
                patch("pokezero.mcts_eval.search_over_raw_raw_run.public_request_witness",return_value=witness()):
            job=RawSelectionJob(self.cfg,SimpleNamespace(checkpoint_path=str(self.checkpoint),
                checkpoint_sha256=self.files[str(self.checkpoint)],showdown_source_sha256=BINDING["set_source_hash"]),
                str(self.showdown),context,"root:raw",7,"root",self.runtime,PUBLIC_SHA)
            self.assertEqual(job.read_selection(selected_fixture(self.cfg,self.runtime,forced=True))["action"],0)
        self.assertEqual((job.root_id,job.statistics_root_id,job.selection_seed),("root:raw","root",7))
        self.assertEqual(job._mask,(True,False,False))

    def test_empty_and_nonboolean_raw_masks_remain_rejected(self):
        for mask in ((False,False,False),(1,False,False)):
            with self.subTest(mask=mask),self.assertRaises(ValueError):
                read_raw_selection(selected_fixture(self.cfg,self.runtime,forced=True),configuration=self.cfg,
                    runtime=self.runtime,root_id="root:raw",statistics_root_id="root",selection_seed=7,
                    consumed_binding=BINDING,legal_action_mask=mask,public_request_sha256=PUBLIC_SHA)

    def test_owned_control_is_distinct_normal_completion_not_search_or_qualification(self):
        terminal,reader,directory,job = self.run_owner()
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        self.assertEqual(terminal["status"],"RAW_SELECTION_RUN_COMPLETE")
        self.assertEqual(reader(terminal["result"]),digest(terminal["result"]))
        validation = reader.last_validation
        self.assertEqual(validation["actual_work"]["observed_model_forwards"],1)
        self.assertEqual(validation["actual_work"]["accepted_units"],0)
        self.assertEqual(validation["owned_generation_pids"],[])
        self.assertTrue(validation["process_receipts_validated"])
        for name in ("runtime_qualified","scientific_collection_authorized","independent_evaluator_correctness",
                "loaded_binary_dependency_closure_validated","full_public_request_correspondence_validated"):
            self.assertFalse(validation[name])
        self.assertIsNone(self.reader(directory,job,False)(terminal["result"]))
        with self.assertRaises(FileExistsError):
            _supervise_owned_job(job,directory=directory,binding_sha256=RUN_SHA,verify_admission=lambda:None,
                contract=RawSelectionRunContract(),kind="raw-selection-run",work_phase="selection",validate_result=reader._read_work)

    def test_selected_argmax_mask_forward_request_runtime_and_tampering_refuse(self):
        terminal,reader,directory,job = self.run_owner()
        original = terminal["result"]
        cases = [("action",2),("action",True),("status","UNQUALIFIED_AT_CEILING"),
            ("selection_seed",True),("runtime_sha256","e"*64),("elapsed_seconds",float("nan"))]
        for key,value in cases:
            selected=deepcopy(original);selected[key]=value
            with self.assertRaises(ValueError):reader._read_work(selected)
        for key,value in (("model_forwards",True),("model_forwards",2),("legal_actions",[0,1]),
                ("priors",[.3,.3]),("signed_value",2.),("public_request_witness",dict(witness(),public_request_sha256="e"*64))):
            selected=deepcopy(original);selected["evidence"][key]=value
            with self.assertRaises(ValueError):reader._read_work(selected)
        path=directory/"terminal.json";saved=path.read_bytes();value=json.loads(saved);value["worker_exit_code"]=1
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):reader(original)
        path.write_bytes(saved)
        (directory/"owned-groups"/"unexpected.json").write_text("{}")
        with self.assertRaises(ValueError):reader(original)

    def test_operational_failure_is_not_a_selected_raw_control(self):
        terminal,reader,directory,job=self.run_owner(True)
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(terminal["status"],"OPERATIONAL_STOP")
        with self.assertRaises(ValueError):reader(selected_fixture(self.cfg,self.runtime))

    def test_actual_job_dispatch_sanitized_root_and_canonical_loader_cleanup(self):
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        state=SyntheticPublicState(SyntheticBelief());observation=SimpleNamespace(legal_action_mask=(True,False,True))
        context=TimedContext("p1",observation,{"p1":observation,"p2":"private-canary"},state)
        request=PublicRootRequest(state=state,observation=observation,set_source_hash=BINDING["set_source_hash"])
        with patch("pokezero.mcts_eval.head_to_head.public_only_context",side_effect=lambda c:c), \
                patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=request), \
                patch("pokezero.mcts_eval.search_over_raw_raw_run._binding",return_value=BINDING), \
                patch("pokezero.mcts_eval.search_over_raw_raw_run.public_request_witness",return_value=witness()):
            job=RawSelectionJob(self.cfg,SimpleNamespace(checkpoint_path=str(self.checkpoint),
                checkpoint_sha256=self.files[str(self.checkpoint)],showdown_source_sha256=BINDING["set_source_hash"]),
                str(self.showdown),context,"root:raw",7,"root",self.runtime,PUBLIC_SHA)
            self.assertEqual(set(job.context.requested_observations),{"p1"})
            adapter=Mock(runtime_configuration=deepcopy(self.runtime),runtime_sha256=digest(self.runtime))
            adapter.select.return_value=selected_fixture(self.cfg,self.runtime)
            phases=[]
            with patch("pokezero.neural_policy.load_transformer_policy",return_value="canonical-policy") as load, \
                    patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=SimpleNamespace(
                        metadata=SimpleNamespace(source_hash=BINDING["set_source_hash"]))), \
                    patch("pokezero.mcts_eval.paper_reference_showdown.ChampionEvaluator",return_value="mock-evaluator") as evaluate, \
                    patch("pokezero.mcts_eval.search_over_raw_raw_run.RawSelectionAdapter",return_value=adapter) as factory:
                self.assertEqual(job({},phases.append)["action"],0)
                load.assert_called_once_with(self.checkpoint,device="cpu",deterministic=True,exploration_epsilon=0.,
                    sampling_temperature=1.,family_gated_selection=False)
                evaluate.assert_called_once_with("canonical-policy")
                self.assertEqual(factory.call_args.kwargs["evaluator"],"mock-evaluator")
                adapter.close.assert_called_once()
                self.assertEqual(phases,["startup","selection","cleanup"])
                adapter.select.side_effect=RuntimeError("primary");adapter.close.side_effect=ValueError("private detail")
                with self.assertRaisesRegex(RuntimeError,"primary") as caught:job({},phases.append)
                self.assertEqual(caught.exception.__notes__,["Raw adapter cleanup failed: ValueError"])
            job.expected_runtime_configuration["checkpoint_sha256"]="e"*64
            with self.assertRaisesRegex(ValueError,"changed before startup"):job({},lambda p:None)

    def test_capture_adapter_reads_actual_request_and_wrapper_never_accepts_toy(self):
        adapter=object.__new__(RawSelectionAdapter)
        with patch("pokezero.mcts_eval.search_over_raw_raw_run._binding",return_value=BINDING), \
                patch("pokezero.mcts_eval.search_over_raw_raw_run.public_request_witness",return_value=witness()) as capture:
            evidence=adapter._selection_evidence(dict(selector="test"),"actual-request")
            capture.assert_called_once_with("actual-request")
            self.assertEqual(evidence["public_request_witness"],witness())
        with self.assertRaises(ValueError):supervise_raw_selection(ToyRawOwner(self.cfg,self.runtime),
            directory=self.root/"not-launched",binding_sha256=RUN_SHA,verify_admission=lambda:None)
        self.assertFalse((self.root/"not-launched").exists())
        with self.assertRaises(ValueError):RawSelectionRunContract(True)

    def test_real_public_capture_and_actual_adapter_dispatch_without_source_game(self):
        from test_search_over_raw_archive import ExplorationRootCrosswalkTests
        from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
        from pokezero.mcts_eval.paper_reference import Evaluation
        fixture=ExplorationRootCrosswalkTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        bound=fixture.bind();consumed=bound["public_request_witness"]["consumed_binding"]
        runtime=raw_runtime(self.cfg,self.files[str(self.checkpoint)],consumed["set_source_hash"])
        job=RawSelectionJob(self.cfg,SimpleNamespace(checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.files[str(self.checkpoint)],showdown_source_sha256=consumed["set_source_hash"]),
            str(self.showdown),fixture.context,fixture.root_id+":raw",bound["slot"]["source_seed"],fixture.root_id,
            runtime,bound["public_request_witness"]["public_request_sha256"])
        self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
        self.assertNotIn(b"private-source-snapshot",pickle.dumps(job))
        self.assertNotIn(b"not-actor-input",pickle.dumps(job))
        forward=object()
        policy=SimpleNamespace(weights_sha256=self.files[str(self.checkpoint)],result=SimpleNamespace(
            model_config=SimpleNamespace(window_size=1,value_activation="tanh")),deterministic=True,
            exploration_epsilon=0.,sampling_temperature=1.,family_gated_selection=False,history_mask_k=None,
            forward_fn=forward,_default_forward=forward)
        observations=[]
        def authored_forward(evaluator,observation):
            evaluator.forwards+=1;observations.append(observation)
            legal=tuple(i for i,v in enumerate(observation.legal_action_mask) if v)
            return legal,Evaluation(tuple(1/len(legal) for _ in legal),.1)
        with patch("pokezero.neural_policy.load_transformer_policy",return_value=policy), \
                patch("pokezero.randbat.load_gen3_randbat_source_cached",return_value=SimpleNamespace(
                    metadata=SimpleNamespace(source_hash=consumed["set_source_hash"]))), \
                patch.object(ChampionEvaluator,"__call__",autospec=True,side_effect=authored_forward), \
                patch("subprocess.Popen") as launch:
            selected=job({},lambda p:None)
        launch.assert_not_called();self.assertEqual(len(observations),1)
        self.assertEqual(selected["evidence"]["public_request_witness"],bound["public_request_witness"])
        self.assertEqual(job.read_selection(selected)["observed_model_forwards"],1)
        # A public field omitted by the compact information key still changes
        # the request and blocks launch, rather than matching only the key.
        object.__setattr__(job,"context",replace(job.context,public_materialization_state=replace(
            job.context.public_materialization_state,self_move_states={"p1:known":({"pp":14},)})))
        with self.assertRaisesRegex(ValueError,"changed before startup"):job({},lambda p:None)


if __name__ == "__main__":unittest.main()
