"""Actual paired initializer and disposable owners; no scientific model/game."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import pickle
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownSnapshot,LocalShowdownEnv
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS,recorded_materialization_inputs
from pokezero.mcts_eval.search_over_raw_materialization_run import MaterializationRunValidator
from pokezero.mcts_eval.search_over_raw_paired_materialization import (
    ARTIFACT, RESET_ARTIFACT, OwnedPairedMaterializationJob, PairedMaterializationRunValidator,
    ContinuationContextUnavailable, context_sources, paired_materialization_inputs,
    supervise_paired_materialization_run,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload,_snapshot_payload
import test_search_over_raw_materialization_run as fixture


@dataclass(frozen=True)
class AuthoredPairedJob(OwnedPairedMaterializationJob):
    synthetic_process_fixture=True
    def _components(self):return PairedSyntheticEnv(self.config)
    def input_binding(self):
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=fixture._payload(fixture._dex())):
            return super().input_binding()
    def __call__(self,ownership,phase):
        try:
            with patch("pokezero.local_showdown._public_materialization_payload",return_value=fixture._payload(fixture._dex())):
                return super().__call__(ownership,phase)
        except BaseException as error:
            (Path(ownership["directory"]).parent/"authored-error.json").write_text(json.dumps(
                dict(error=type(error).__name__,message=str(error))))
            raise


class PairedSyntheticEnv(fixture.SyntheticEnv):
    _sync_incremental_state=LocalShowdownEnv._sync_incremental_state


class PairedMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.MaterializationRunTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        # Real Python reset/event folding over predeclared authored Node streams.
        # These are sampled-shell requests, not source-opponent history.
        bridge=fixture.BRIDGE.replace("send({type:'ready',battleId:b,requested:['p1','p2']}); }",
            "send({type:'stream',battleId:b,stream:'omniscient',lines:['|player|p1|A','|player|p2|B','|turn|1']}); "
            "for(const seat of ['p1','p2']) send({type:'stream',battleId:b,stream:seat,lines:['|request|'+JSON.stringify(fixture.snapshot.boundaryRequests[seat])]}); "
            "send({type:'ready',battleId:b,requested:['p1','p2']}); }",1)
        self.h.bridge.write_text(bridge);self.h.configure("complete")
        self.options={k:False for k in BOOL_OPTIONS};self.options.update({k:None for k in MAP_OPTIONS})
        draw=asdict(fixture._override());payload=fixture._payload(fixture._dex())
        draw.update(observation_format_id=self.h.state.observation_format_id)
        origin=dict(schema="pokezero.search-over-raw.recorded-native-materialization.v1",
            native_selected_sha256="c"*64,construction_capture_sha256="d"*64,construction_ordinal=0,
            original_public_request_sha256="e"*64,draw_sha256=digest(draw),public_payload_sha256=digest(payload))
        self.inputs=recorded_materialization_inputs(draw=draw,public_payload=payload,seed=17,
            root_binding=deepcopy(fixture.BINDING),public_state_sha256=digest(_state_payload(self.h.state)),
            recorded_native_origin=origin,options=self.options)

    def job(self):
        return AuthoredPairedJob(self.h.config,deepcopy(self.inputs),deepcopy(self.h.runtime),deepcopy(self.h.state))

    def run_job(self):
        job=self.job()
        terminal=supervise_paired_materialization_run(job,directory=self.h.directory,binding_sha256="a"*64,
            verify_admission=lambda:True,validate_result=lambda _:True)
        if terminal["result"] is not None:self.context_report=deepcopy(terminal["result"]["paired_context"])
        detail=self.h.directory/"authored-error.json"
        return job,terminal,detail.read_text() if detail.exists() else terminal

    def reader(self,job,*,allow_synthetic=True):
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=fixture._payload(fixture._dex())):
            return PairedMaterializationRunValidator(directory=self.h.directory,
                job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),binding_sha256="a"*64,
                expected_inputs=job.input_binding(),runtime=self.h.runtime,original_public_context=self.h.state,
                expected_context=self.context_report,
                allow_synthetic=allow_synthetic)

    def test_actual_paired_snapshot_retained_by_distinct_owner_and_independent_reader(self):
        job,terminal,detail=self.run_job()
        self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        result=terminal["result"];reader=self.reader(job)
        self.assertEqual(reader(result["capture"]),digest(result["capture"]))
        snapshot=reader.retained_context(result)
        self.assertIs(type(snapshot),LocalShowdownSnapshot)
        self.assertEqual(digest(snapshot.bridge_snapshot),result["produced_bridge_snapshot_sha256"])
        self.assertEqual(snapshot.latest_requests["p1"],self.h.state.self_request)
        self.assertEqual(snapshot.latest_requests["p2"],result["capture"]["raw_events"][0]["boundaryRequests"]["p2"])
        self.assertTrue(all(snapshot.request_history[s] for s in ("p1","p2")))
        self.assertIn(ARTIFACT,reader.last_validation["receipt_file_sha256"])
        self.assertIn(RESET_ARTIFACT,reader.last_validation["receipt_file_sha256"])
        self.assertTrue(result["paired_context"]["request_substitution_changed"])
        self.assertFalse(result["paired_context"]["request_correspondence_validated"])
        self.assertFalse(result["paired_context"]["paired_replay_validated"])
        self.assertFalse(result["scientific_collection_authorized"])
        self.assertEqual([r["payload"]["type"] for r in result["capture"]["records"] if r["kind"]=="command"],
            ["start","materialize","snapshot","end"])
        self.assertTrue(self.h.load("bridge-close.json")["normal_close"])
        with self.assertRaises(FileExistsError):self.run_job()
        self.assertIsNone(self.reader(job,allow_synthetic=False)(result["capture"]))
        with self.assertRaises(ValueError):self.reader(job,allow_synthetic=False).retained_context(result)
        with self.assertRaises(ValueError):
            MaterializationRunValidator(directory=self.h.directory,job_sha256=reader.job_sha,
                binding_sha256="a"*64,expected_inputs=job.input_binding(),runtime=self.h.runtime,allow_synthetic=True)

    def test_well_formed_different_captured_overlay_has_no_source_root_fallback_or_launch(self):
        value=deepcopy(self.inputs);value["materialize"]["publicState"]["turn"]+=1
        sha=digest(value["materialize"]["publicState"])
        value["public_payload_sha256"]=sha;value["recorded_native_origin"]["public_payload_sha256"]=sha
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=fixture._payload(fixture._dex())), \
                patch("subprocess.Popen") as launch:
            with self.assertRaises(ContinuationContextUnavailable):paired_materialization_inputs(self.h.state,value,self.h.runtime)
        launch.assert_not_called()

    def test_malformed_native_origin_is_not_normal_context_unavailability(self):
        value=deepcopy(self.inputs);value["recorded_native_origin"]["construction_ordinal"]=False
        with patch("subprocess.Popen") as launch:
            try:paired_materialization_inputs(self.h.state,value,self.h.runtime)
            except ContinuationContextUnavailable:self.fail("malformed origin became normal missing context")
            except ValueError:pass
            else:self.fail("malformed origin accepted")
        launch.assert_not_called()

    def test_retained_bytes_or_python_request_substitution_cannot_change_after_owner(self):
        job,terminal,detail=self.run_job();self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        result=terminal["result"];reader=self.reader(job)
        path=self.h.directory/ARTIFACT;original=path.read_bytes()
        path.write_bytes(original+b" ")
        with self.assertRaises(ValueError):reader.retained_context(result)
        self.assertIsNone(reader.last_validation)
        path.write_bytes(original)
        self.assertIs(type(reader.retained_context(result)),LocalShowdownSnapshot)
        # Recompute every exposed digest and both durable result wrappers. The
        # reader must still derive the original initializer's actual requests.
        changed=pickle.loads(original);changed.latest_requests["p2"]=deepcopy(changed.latest_requests["p1"])
        raw=pickle.dumps(changed,protocol=5);path.write_bytes(raw)
        changed_result=deepcopy(result)
        changed_result["paired_context"].update(artifact_sha256=hashlib.sha256(raw).hexdigest(),
            snapshot_sha256=digest(_snapshot_payload(changed)),latest_requests_sha256=digest(changed.latest_requests))
        saved={name:(self.h.directory/name).read_bytes() for name in ("terminal.json","worker-result.json")}
        for name,body in saved.items():
            value=json.loads(body);value["result"]=changed_result;self.h.save(name,value)
        with patch("pokezero.mcts_eval.search_over_raw_paired_materialization.pickle.loads",side_effect=AssertionError("untrusted pickle loaded")):
            with self.assertRaisesRegex(ValueError,"externally frozen"):
                reader(changed_result["capture"])
        for name,body in saved.items():(self.h.directory/name).write_bytes(body)
        path.write_bytes(original)
        self.assertIs(type(reader.retained_context(result)),LocalShowdownSnapshot)

    def test_rehashed_request_history_still_requires_original_observed_reset_events(self):
        job,terminal,detail=self.run_job();self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        result=deepcopy(terminal["result"]);path=self.h.directory/ARTIFACT
        snapshot=pickle.loads(path.read_bytes())
        snapshot.request_history["p2"][0]["active"][0]["moves"][0]["pp"]=123
        raw=pickle.dumps(snapshot,protocol=5);path.write_bytes(raw)
        result["paired_context"].update(artifact_sha256=hashlib.sha256(raw).hexdigest(),
            snapshot_sha256=digest(_snapshot_payload(snapshot)))
        # Even a pure adversarial recomputation cannot detach request history
        # from independently supplied original transport, though it is not an
        # externally owned result and cannot qualify through the real reader.
        from pokezero.mcts_eval.search_over_raw_paired_materialization import _join_paired_result
        with self.assertRaisesRegex(ValueError,"retained paired state"):
            _join_paired_result(result,job.input_binding(),result["capture"],self.h.directory.resolve(),self.h.state,
                result["paired_context"])

    def test_public_context_provenance_drift_is_operational_not_ordinary_unavailability(self):
        for key in ("public_state_sha256","observation_format_id","subject","selfPlayer"):
            value=deepcopy(self.inputs)
            if key=="subject":value["root_binding"][key]="p2"
            elif key=="selfPlayer":
                value["materialize"]["publicState"][key]="p2"
                sha=digest(value["materialize"]["publicState"])
                value["public_payload_sha256"]=sha;value["recorded_native_origin"]["public_payload_sha256"]=sha
            else:value[key]="f"*64 if key.endswith("sha256") else "other-format"
            with self.subTest(key=key),patch("subprocess.Popen") as launch:
                try:paired_materialization_inputs(self.h.state,value,self.h.runtime)
                except ContinuationContextUnavailable:self.fail("provenance drift became normal unavailability")
                except ValueError:pass
                else:self.fail("provenance drift accepted")
                launch.assert_not_called()

    def test_rehashed_battle_metadata_cannot_rename_original_initializer(self):
        job,terminal,detail=self.run_job();self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        result=deepcopy(terminal["result"]);path=self.h.directory/ARTIFACT
        snapshot=pickle.loads(path.read_bytes())
        from dataclasses import replace
        snapshot=replace(snapshot,battle_id="another-battle",replay=replace(snapshot.replay,battle_id="another-battle"))
        raw=pickle.dumps(snapshot,protocol=5);path.write_bytes(raw)
        result["paired_context"].update(artifact_sha256=hashlib.sha256(raw).hexdigest(),snapshot_sha256=digest(_snapshot_payload(snapshot)))
        from pokezero.mcts_eval.search_over_raw_paired_materialization import _join_paired_result
        with self.assertRaisesRegex(ValueError,"declared original initializer"):
            _join_paired_result(result,job.input_binding(),result["capture"],self.h.directory.resolve(),self.h.state,
                result["paired_context"])

    def test_late_capture_callback_cannot_change_uninventoried_private_context(self):
        job,terminal,detail=self.run_job();self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        reader=self.reader(job);path=self.h.directory/ARTIFACT;original=path.read_bytes()
        check=reader._match_capture
        def mutate(*args):
            check(*args);path.write_bytes(original+b" ")
        with patch.object(reader,"_match_capture",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"private paired artifacts changed"):
                reader(terminal["result"]["capture"])
        self.assertIsNone(reader.last_validation)
        path.write_bytes(original)
        self.assertIs(type(reader.retained_context(terminal["result"])),LocalShowdownSnapshot)

    def test_source_scope_and_context_bytes_are_prospectively_pinned(self):
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        inventory=controller_input_hashes()
        self.assertTrue(all(inventory.get(p)==sha for p,sha in context_sources().items()))
        job=self.job();binding=job.input_binding()
        changed=deepcopy(binding);changed["context_law"]["independent_labels"]=0
        from pokezero.mcts_eval.search_over_raw_paired_materialization import _validate_inputs
        with self.assertRaises(ValueError):_validate_inputs(changed,self.h.state)
        changed=deepcopy(binding);changed["context_sources"].pop(next(iter(context_sources())))
        with self.assertRaises(ValueError):_validate_inputs(changed,self.h.state)

    def test_failed_normal_close_does_not_clear_a_retained_context(self):
        self.h.configure("missing-close")
        _,terminal,_=self.run_job()
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertIsNone(terminal["result"])
