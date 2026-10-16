"""Authored compact/public-fold integration; no games, models or native calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pokezero.mcts_eval import search_over_raw_fixed_exploration as fixed
from pokezero.mcts_eval.search_over_raw_native_atomic_compact import SCHEMA as ENCODING
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    NativeWorldInputs, NativeInputHandle, validate_fixed_work_inputs,
)
import test_search_over_raw_public_fold as fold_fixture
from test_search_over_raw_world_inputs import arguments
from test_search_over_raw_native_atomic_work import native_report
from test_search_over_raw_native_atomic_compact import compact_fixture


class FixedCompactPublicFoldTests(unittest.TestCase):
    def contract(self, total=4096, compact=False):
        return fixed.contract("51f8beec-d601-4fdc-ab73-f0517cd05001",list(range(100,132)),[1,2,3],
            dict(path="/authored/steer",sha256="a"*64),total,compact=compact)

    def test_encoding_is_explicit_and_legacy_contracts_remain_separate(self):
        legacy=self.contract();prospective=self.contract(compact=True)
        fixed.validate_contract(legacy);fixed.validate_contract(prospective)
        self.assertEqual(legacy["schema"],fixed.SCHEMA)
        self.assertNotIn("native_atomic_encoding",legacy)
        self.assertEqual(prospective["schema"],fixed.COMPACT_SCHEMA)
        self.assertEqual(prospective["native_atomic_encoding"],ENCODING)
        for total in (64,320,1024):
            with self.assertRaises(ValueError):self.contract(total,True)
        for key,value in (("schema",fixed.SCHEMA),("native_atomic_encoding","v1"),
                ("native_iterations_per_world",256)):
            changed={**prospective,key:value}
            with self.assertRaises(ValueError):fixed.validate_contract(changed)
        terminal=dict(supervisor_exit_code=0,worker_exit_code=0,status="COMPLETE",
            elapsed_wall_seconds=1.,cpu_seconds_including_reaped_children=.8)
        for c in (legacy,prospective):
            readout=fixed.summarize(dict(contract=c),[],terminal)
            self.assertEqual(readout["schema"],c["schema"]+".readout")
            self.assertEqual(readout.get("native_atomic_encoding"),c.get("native_atomic_encoding"))

    def fixture(self):
        helper=fold_fixture.ConstructionPublicFoldTests();helper.setUp();self.addCleanup(helper.doCleanups)
        witness=helper.witness
        recorder=NativeWorldInputs(witness["consumed_binding"],1024,deadline_limited=False,
            record_atomic_work=True,record_compact_atomic_work=True,
            record_construction=True,record_public_fold=True,expected_public_request=witness)
        for _ in range(4):world,state=helper.construct(recorder)
        args=arguments(text=state.to_string(),iterations=4096)
        args[5]=json.dumps({seat:list(world.party_species[seat]) for seat in ("p1","p2")} | {"turn":3})
        raw=json.dumps(compact_fixture(native_report(4096,text=state.to_string())))
        base=SimpleNamespace(search_batched_multi_encoded=Mock(return_value=raw))
        NativeInputHandle(base,recorder).search_batched_multi_encoded(*args)
        evidence=dict(total_iterations=4096)
        evidence["consumed_world_inputs"]=recorder.finish(evidence,attempted=4,constructed=4,searched=4)
        kwargs=dict(arm="incumbent",binding=witness["consumed_binding"],statistics_root_id="authored",
            construction_required=True,expected_public_request_sha256=witness["public_request_sha256"],
            native_public_fold_required=True,compact_native_atomic_work=True,
            iterations_per_world=1024,compact_fixed_public_fold=True)
        return evidence,kwargs,base,raw

    def test_original_observer_and_independent_reader_join_exact_compact_work_to_full_public_fold(self):
        evidence,kwargs,base,raw=self.fixture()
        self.assertEqual(base.search_batched_multi_encoded.call_args.kwargs,
            dict(record_atomic_work=True,record_compact_atomic_work=True))
        manifest=evidence["consumed_world_inputs"]
        self.assertEqual(manifest["calls"][0]["native_report_utf8"],raw)
        self.assertEqual(manifest["native_atomic_encoding"],ENCODING)
        self.assertEqual(manifest["calls"][0]["construction_ordinals"],[0,1,2,3])
        self.assertIsInstance(validate_fixed_work_inputs(evidence,**kwargs),str)
        for key,value in (("compact_fixed_public_fold",False),("compact_fixed_public_fold",1),
                ("compact_native_atomic_work",False),("native_public_fold_required",False),
                ("construction_required",False),("iterations_per_world",256)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_fixed_work_inputs(evidence,**{**kwargs,key:value})
        for mode in ("fold","dispatch","encoding","exact-return"):
            changed=deepcopy(evidence);m=changed["consumed_world_inputs"]
            if mode=="fold":m["rows"][0].pop("public_fold_capture")
            elif mode=="dispatch":m["calls"][0].pop("dispatch_inputs")
            elif mode=="encoding":m.pop("native_atomic_encoding")
            else:m["calls"][0]["native_report_utf8"]+=" "
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                validate_fixed_work_inputs(changed,**kwargs)


if __name__=="__main__":unittest.main()
