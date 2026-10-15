"""Exact recorded overlays through real disposable bridge ownership, no science."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import pickle
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_materialization import (
    BOOL_OPTIONS,MAP_OPTIONS,recorded_materialization_inputs,validate_materialization_inputs)
from pokezero.mcts_eval.search_over_raw_materialization_run import (
    OwnedRecordedMaterializationJob,MaterializationRunValidator,supervise_materialization_run)
import test_search_over_raw_materialization_run as fixtures
from test_engine_world import _override,_payload,_dex


@dataclass(frozen=True)
class AuthoredRecordedJob(OwnedRecordedMaterializationJob):
    synthetic_process_fixture=True
    def _components(self):return fixtures.SyntheticEnv(self.config)


class RecordedMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.MaterializationRunTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.payload=deepcopy(_payload(_dex()))
        self.payload["turn"]=77  # A pending-state overlay that is not the archived/root fixture payload.
        self.payload["deferredOpponentActions"]={"p2":0}
        self.payload["deferredOpponentActionPriors"]={"p2":[1.,0.]}
        self.options={k:False for k in BOOL_OPTIONS};self.options.update({k:None for k in MAP_OPTIONS})
        self.options.update(deferred_opponent_actions=deepcopy(self.payload["deferredOpponentActions"]),
            deferred_opponent_action_priors=deepcopy(self.payload["deferredOpponentActionPriors"]))
        self.draw=asdict(_override())
        self.origin=dict(schema="pokezero.search-over-raw.recorded-native-materialization.v1",
            native_selected_sha256="c"*64,construction_capture_sha256="d"*64,construction_ordinal=0,
            original_public_request_sha256="e"*64,draw_sha256=digest(self.draw),public_payload_sha256=digest(self.payload))
        self.inputs=recorded_materialization_inputs(draw=self.draw,public_payload=self.payload,seed=17,
            root_binding=deepcopy(fixtures.BINDING),public_state_sha256="f"*64,
            recorded_native_origin=self.origin,options=self.options)

    def test_actual_recorded_pending_overlay_is_sent_without_root_reconstruction(self):
        job=AuthoredRecordedJob(self.h.config,self.inputs,self.h.runtime)
        with patch("pokezero.local_showdown._public_materialization_payload",side_effect=AssertionError("root overlay rebuilt")):
            terminal=supervise_materialization_run(job,directory=self.h.directory,binding_sha256="a"*64,
                verify_admission=lambda:True,validate_result=lambda _:True)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        result=terminal["result"]
        reader=MaterializationRunValidator(directory=self.h.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,expected_inputs=job.input_binding(),runtime=self.h.runtime,allow_synthetic=True)
        self.assertEqual(reader(result["capture"]),digest(result["capture"]))
        command=next(r["payload"] for r in result["capture"]["records"]
            if r["kind"]=="command" and r["payload"]["type"]=="materialize")
        # Redaction hashes the complete actual command; the separately frozen
        # input includes the exact overlay and original captured origin.
        self.assertEqual(result["materialization_inputs"]["materialize"]["publicState"],self.payload)
        self.assertEqual(result["materialization_inputs"]["recorded_native_origin"],self.origin)
        self.assertEqual(command["publicState_sha256"],digest(self.payload))
        self.assertNotEqual(self.payload,_payload(_dex()))
        self.assertFalse(result["full_world_equivalence"])
        self.assertFalse(result["scientific_collection_authorized"])

    def test_changed_capture_origin_overlay_or_numeric_ordinal_is_rejected(self):
        for mode in ("draw","payload","ordinal"):
            value=deepcopy(self.inputs)
            if mode=="draw":value["recorded_native_origin"]["draw_sha256"]="1"*64
            elif mode=="payload":value["materialize"]["publicState"]["turn"]=78
            else:value["recorded_native_origin"]["construction_ordinal"]=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_materialization_inputs(value)

    def test_recorded_job_requires_explicit_origin_and_does_not_sample(self):
        value=deepcopy(self.inputs);value.pop("recorded_native_origin")
        job=OwnedRecordedMaterializationJob(self.h.config,value,self.h.runtime)
        with patch("subprocess.Popen") as spawn,self.assertRaises(ValueError):job.input_binding()
        spawn.assert_not_called()
