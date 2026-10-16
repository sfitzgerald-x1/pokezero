"""Explicit post-collection ABI smoke using authored state and random weights.

No source games, champion inference, continuations or old-root replay. Run this
separately after the live scientific worker closes, with the pinned model wheel
and built Showdown checkout selected explicitly.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import test_native_fixed_work_adapter as runtime_fixture
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
from pokezero.mcts_eval.search_over_raw_annotation_boundary import ActorAnnotationBoundary
from pokezero.mcts_eval.search_over_raw_native_atomic_compact import SCHEMA
from pokezero.mcts_eval.search_over_raw_root_priors import native_measure
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    _binding, public_request_witness, validate_fixed_work_inputs,
)


class FixedCompactPublicFoldRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=runtime_fixture.LoadedNativeValueAdapterTests
        if not runtime_fixture._crate_ready:
            raise AssertionError("explicit pinned compact model runtime required")
        cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_actual_fixed4096_dispatch_and_independent_readback_have_bounded_evidence(self):
        helper=self.fixture();self.addCleanup(helper.doCleanups)
        env,context=helper.authored_context()
        boundary=ActorAnnotationBoundary.capture(env,context)
        request=PublicRootRequest.capture(context.public_materialization_state,context.observation)
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path=Path(temporary)/"actual-compact-inputs.json"
            start=time.perf_counter()
            selected=native_measure(context,boundary,helper.checkpoint_contract,7,
                accepted_iterations=4096,compact_native_atomic_work=True,evidence_path=evidence_path)
            elapsed=time.perf_counter()-start
            self.assertEqual(selected["accepted_target"],4096)
            self.assertEqual(selected["fixed_iterations"],4096)
            manifest=selected["actual_native_evidence"]
            self.assertEqual(manifest["native_atomic_encoding"],SCHEMA)
            self.assertEqual([r["ordinal"] for r in manifest["rows"]],[0,1,2,3])
            self.assertTrue(all(r["consumed"] and "public_fold_capture" in r for r in manifest["rows"]))
            for call in manifest["calls"]:
                self.assertEqual(call["atomic_work_validation"]["schema"],SCHEMA)
                self.assertEqual(call["native_report"]["atomic_work"]["schema"],SCHEMA)
                self.assertIn("dispatch_inputs",call)
                self.assertEqual(hashlib.sha256(call["native_report_utf8"].encode()).hexdigest(),
                    call["report_utf8_sha256"])
            checked=validate_fixed_work_inputs(dict(total_iterations=4096,consumed_world_inputs=manifest),
                arm="incumbent",binding=_binding(request),statistics_root_id=context.battle_id,
                construction_required=True,expected_public_request_sha256=public_request_witness(request)["public_request_sha256"],
                native_public_fold_required=True,compact_native_atomic_work=True,
                iterations_per_world=1024,compact_fixed_public_fold=True)
            self.assertIsInstance(checked,str)
            size=evidence_path.stat().st_size
            selected_bytes=len((json.dumps(selected,indent=2,sort_keys=True,allow_nan=False)+"\n").encode())
            self.assertLess(size,64*1024*1024)
            self.assertLess(selected_bytes,64*1024*1024)
            self.assertIsNone(env._process)
            print("AUTHORED_FIXED4096_COMPACT:"+json.dumps(dict(accepted=4096,worlds=4,
                inputs_bytes=size,selection_bytes=selected_bytes,elapsed_seconds=elapsed,
                scientific_source_roots=0),sort_keys=True))


if __name__=="__main__":unittest.main()
