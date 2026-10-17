"""Exact gate arithmetic and boundary joins, without games or native search."""
from copy import deepcopy
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.root_prior_correspondence import (
    SCHEMA, boundary_fingerprint, exact_collector_gate, native_extension_path,
)


class ExactCollectorGateTests(unittest.TestCase):
    def report(self):
        binding = dict(source_configuration={"format_id": "gen3randombattle"},
            selector_configurations={"sampled4": {"worlds": 4, "search_sims": 80}},
            world_rng="per-decision", source_files={"engine_search.py": "a"*64},
            **{k+"_sha256": "b"*64 for k in ("checkpoint", "model", "tables", "engine", "search")})
        return dict(schema=SCHEMA, complete=True, bindings=binding,
            expected=[dict(seed=12, round=7, variant="sampled4")],
            rows=[dict(seed=12, round=7, boundary_sha256="c"*64, legal=[0, 2],
                variants={"sampled4": dict(native_prior_pre_display={"0": .75, "2": .25})})])

    def test_exact_join_passes_and_live_extras_do_not_replace_expected_boundaries(self):
        left=self.report();right=deepcopy(left)
        extra=deepcopy(right["rows"][0]);extra["round"]=8;right["rows"].append(extra)
        verdict=exact_collector_gate(left,right)
        self.assertTrue(verdict["passed"])
        self.assertEqual((verdict["expected"],verdict["compared"]),(1,1))
        self.assertEqual(verdict["gate_tv"],0.)

    def test_tiny_difference_fails_before_display_rounding(self):
        left=self.report();right=deepcopy(left)
        right["rows"][0]["variants"]["sampled4"]["native_prior_pre_display"]={"0":.750000001,"2":.249999999}
        verdict=exact_collector_gate(left,right)
        self.assertFalse(verdict["passed"])
        self.assertGreater(verdict["mismatches"][0]["tv"],0.)
        self.assertEqual(round(verdict["mismatches"][0]["tv"],4),0.)

    def test_never_renormalizes_changed_recorded_values_to_make_them_equal(self):
        left=self.report();right=deepcopy(left)
        right["rows"][0]["variants"]["sampled4"]["native_prior_pre_display"]={"0":.75000075,"2":.25000025}
        self.assertFalse(exact_collector_gate(left,right)["passed"])

    def test_requires_closed_reports_exact_inputs_and_nonempty_unique_roster(self):
        for mode in ("open", "schema", "binding", "empty", "duplicate", "rng", "unknown_config", "hash"):
            left=self.report();right=deepcopy(left)
            if mode=="open":right["complete"]=False
            elif mode=="schema":right["schema"]="old-rounded-report"
            elif mode=="binding":right["bindings"]["model_sha256"]="d"*64
            elif mode=="empty":left["expected"]=[]
            elif mode=="duplicate":left["expected"]*=2
            elif mode=="rng":
                left["bindings"]["world_rng"]=right["bindings"]["world_rng"]="per-game"
            elif mode=="unknown_config":left["expected"][0]["variant"]="unbound"
            else:
                left["bindings"]["engine_sha256"]=right["bindings"]["engine_sha256"]="not-a-hash"
            with self.subTest(mode=mode):self.assertFalse(exact_collector_gate(left,right)["passed"])

    def test_missing_duplicate_wrong_boundary_and_partial_mapping_do_not_pass(self):
        for mode in ("missing", "duplicate", "boundary", "legal", "rounded", "unmapped", "error",
                "negative", "boolean", "nan", "not_normalized", "boolean_index"):
            left=self.report();right=deepcopy(left)
            row=right["rows"][0];item=row["variants"]["sampled4"]
            if mode=="missing":right["rows"]=[]
            elif mode=="duplicate":right["rows"].append(deepcopy(row))
            elif mode=="boundary":row["boundary_sha256"]="d"*64
            elif mode=="legal":row["legal"]=[0,1]
            elif mode=="rounded":item["native_prior"]=item.pop("native_prior_pre_display")
            elif mode=="unmapped":item["unmapped_arms"]=1
            elif mode=="error":item["error"]="fallback"
            elif mode=="boolean_index":row["round"]=True
            else:item["native_prior_pre_display"]["2"]={
                "negative":-.25,"boolean":True,"nan":float("nan"),"not_normalized":.5}[mode]
            with self.subTest(mode=mode):self.assertFalse(exact_collector_gate(left,right)["passed"])


class FullBoundaryIdentityTests(unittest.TestCase):
    def context(self):
        return SimpleNamespace(format_id="gen3randombattle",player_id="p1",
            public_materialization_state={"self_request":{"forceSwitch":False}},
            observation=SimpleNamespace(schema_version="authored",categorical_ids=[[1]],
                numeric_features=[[.1]],token_type_ids=[2],attention_mask=[True],
                legal_action_mask=[True,False],metadata={"turn_number":7,"public_stat":3},perspective=None))

    def test_full_arrays_metadata_and_public_request_affect_identity(self):
        with patch("pokezero.local_showdown._public_materialization_payload",side_effect=lambda state:state):
            context=self.context();expected=boundary_fingerprint(context)
            for mode in ("categorical", "numeric", "token", "attention", "legal", "metadata", "request"):
                changed=deepcopy(context)
                if mode=="request":changed.public_materialization_state["self_request"]["forceSwitch"]=True
                elif mode=="metadata":changed.observation.metadata["public_stat"]=4
                else:
                    key={"categorical":"categorical_ids","numeric":"numeric_features","token":"token_type_ids",
                        "attention":"attention_mask","legal":"legal_action_mask"}[mode]
                    setattr(changed.observation,key,[[9]] if mode in ("categorical","numeric") else [9])
                with self.subTest(mode=mode):self.assertNotEqual(boundary_fingerprint(changed),expected)

    def test_runtime_pin_resolves_loaded_extension_not_reexport_wrapper(self):
        wrapper=SimpleNamespace(__name__="authored_native",__file__="/authored/__init__.py")
        extension=SimpleNamespace(__name__="authored_native.authored_native",
            __file__="/authored/authored_native"+EXTENSION_SUFFIXES[0])
        with patch.dict("sys.modules",{"authored_native":wrapper,"authored_native.authored_native":extension}):
            self.assertEqual(native_extension_path(wrapper),Path(extension.__file__).resolve())
            self.assertEqual(native_extension_path(extension),Path(extension.__file__).resolve())
        with patch.dict("sys.modules",{"authored_native":wrapper}):
            with self.assertRaises(ValueError):native_extension_path(wrapper)


if __name__=="__main__":unittest.main()
