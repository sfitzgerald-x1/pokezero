"""Full denominator/null-only contracts on authored outcomes; no science."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_after_population import (
    NativeAfterRecipe,plan_after_population,population_law,validate_null_admission_reason,completed_slots,_original_row_crosswalk,
    _freeze_after_recipe,preflight_after_population,produce_owned_after_population)
from pokezero.mcts_eval.search_over_raw_after_admission import OwnedNativeAfterAdmission,execute_original_native_after_bank,_payload
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import diagnostic_configurations
from pokezero.mcts_eval.search_over_raw_native_leaf_cells import NativeLeafCellPlan
from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode,_capture_actor_history,AFTER_WORKFLOW_SCHEMA
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
from test_search_over_raw_native_leaf_cells import NativeLeafCellPlanTests,RUN


class AfterPopulationContractsTests(unittest.TestCase):
    def fixture(self):
        h=NativeLeafCellPlanTests();plan=h.plan(h.fixture())
        cfg=next(c for c in diagnostic_configurations() if c.arm=="incumbent" and c.leaf=="model")
        roots=[dict(root_id=RUN["statistics_root_id"] if i==0 else "missing-"+str(i),
            source_seed=RUN["selection_seed"]+i) for i in range(32)]
        manifest=dict(slots=[dict(root=r,configuration_sha256=cfg.identity) for r in roots])
        rows=[dict(**r,status="UNSTARTED",diagnostic=None) for r in roots]
        rows[0].update(status="CHAIN_RECORDED",diagnostic=dict(leaf_cell_plan_sha256=plan.sha256,
            retained_unique_leaf_events=2,required_leaf_label_cells=16))
        return manifest,rows,{roots[0]["root_id"]:plan}

    def run_plan(self,manifest,rows,plans,recipe=None):
        return plan_after_population(manifest,rows,plans,recipe=NativeAfterRecipe() if recipe is None else recipe)

    def test_all_32_slots_two_unique_events_sixteen_cells_31_unknown_counts(self):
        args=self.fixture();before=digest([args[0],args[1],{k:v.payload_utf8 for k,v in args[2].items()}])
        result=self.run_plan(*args)
        self.assertEqual((result["required_native_slots"],result["known_retained_events"],
            result["known_required_cells"],result["unknown_native_slots"]),(32,2,16,31))
        self.assertTrue(all(s["retained_events"] is None and s["required_cells"] is None for s in result["slots"][1:]))
        self.assertEqual(result["law"],population_law())
        self.assertEqual(len({c for e in result["events"] for c in e["cell_ids"]}),16)
        self.assertEqual(before,digest([args[0],args[1],{k:v.payload_utf8 for k,v in args[2].items()}]))

    def test_unqualified_preserves_known_events_without_blocking_other_slots(self):
        result=self.run_plan(*self.fixture(),recipe=NativeAfterRecipe(unqualified_reason="not qualified"))
        self.assertEqual(result["slots"][0]["status"],"SKIPPED_UNQUALIFIED_AFTER")
        self.assertEqual((len(result["events"]),result["known_required_cells"]),(2,16))
        self.assertEqual(result["slots"][1]["status"],"NATIVE_EVENT_POPULATION_UNKNOWN")

    def test_missing_extra_reordered_or_duplicate_slots_refuse(self):
        for mutation in (lambda rows:rows.pop(),lambda rows:rows.append(deepcopy(rows[0])),
                lambda rows:rows.reverse(),lambda rows:rows.__setitem__(1,deepcopy(rows[0]))):
            manifest,rows,plans=self.fixture();mutation(rows)
            with self.assertRaises(ValueError):self.run_plan(manifest,rows,plans)

    def test_extra_or_absent_leaf_plan_refuses(self):
        for extra in (False,True):
            manifest,rows,plans=self.fixture()
            if extra:plans["another-root"]=next(iter(plans.values()))
            else:plans.clear()
            with self.assertRaises(ValueError):self.run_plan(manifest,rows,plans)

    def test_bad_original_cell_alias_coordinate_or_counts_refuse(self):
        for mutation in (lambda p:p["events"][0]["labels"].pop(),
                lambda p:p["events"][0].update(canonical_construction_ordinal=1),
                lambda p:p["events"][0]["labels"][0].update(replicate=False),
                lambda p:p["events"][0]["labels"][0].update(signed_outcome=1),
                lambda p:p.update(required_label_cells=8)):
            manifest,rows,plans=self.fixture();key=next(iter(plans));payload=plans[key].receipt();mutation(payload)
            plans[key]=NativeLeafCellPlan(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False))
            rows[0]["diagnostic"]["leaf_cell_plan_sha256"]=plans[key].sha256
            with self.assertRaises(ValueError):self.run_plan(manifest,rows,plans)

    def test_missing_native_count_does_not_become_an_empty_known_population(self):
        manifest,rows,plans=self.fixture();rows[0].update(status="SKIPPED_UNQUALIFIED_CHAIN",diagnostic=None);plans.clear()
        result=self.run_plan(manifest,rows,plans)
        self.assertEqual((result["known_retained_events"],result["unknown_native_slots"]),(0,32))
        self.assertTrue(all(s["retained_events"] is None for s in result["slots"]))

    def test_typed_recipe_codec_and_explicit_v4_capture(self):
        value=NativeAfterRecipe(unqualified_reason="authored unavailable runtime")
        self.assertEqual(_decode(_state_payload(value)),value)
        for flag in (True,False):
            self.assertIs(_capture_actor_history(dict(schema=AFTER_WORKFLOW_SCHEMA,capture_actor_history=flag)),flag)
        with self.assertRaises(ValueError):_capture_actor_history(dict(schema=AFTER_WORKFLOW_SCHEMA))

    def test_legitimate_missing_anchor_reasons_are_null_not_integrity_failures(self):
        raw=dict(battle=dict(ended=False));programs=dict(p1=None,p2=None)
        for reason in ("OWN_REQUEST_ANCHOR_UNAVAILABLE","OWN_KNOWLEDGE_ANCHOR_MISMATCH"):
            validate_null_admission_reason(raw,programs,reason)
            with self.assertRaises(ValueError):validate_null_admission_reason(raw,dict(p1="a"*64,p2=None),reason)
        with self.assertRaises(ValueError):validate_null_admission_reason(raw,programs,"SAMPLED_ACTOR_HISTORY_UNAVAILABLE")
        validate_null_admission_reason(raw,dict(p1="a"*64,p2=None),"SAMPLED_ACTOR_HISTORY_UNAVAILABLE")

    def test_completed_null_slot_is_not_reported_as_never_processed(self):
        plan=self.run_plan(*self.fixture());slots=completed_slots(plan)
        self.assertEqual(slots[0]["status"],"ALL_RESERVED_CELLS_UNCERTAIN")
        self.assertEqual(plan["slots"][0]["status"],"UNSTARTED_UNCERTAIN")
        self.assertEqual(slots[1:],plan["slots"][1:])

    def test_later_callback_cannot_promote_mutated_copied_rows_to_final_baseline(self):
        plan=self.run_plan(*self.fixture())
        pairs=[(SimpleNamespace(receipt=dict(status="unavailable")),SimpleNamespace(result=dict(scored_terminal_labels=0)))
            for _ in plan["events"]]
        rows=[dict(**deepcopy(e),admission=deepcopy(a.receipt),bank=deepcopy(b.result))
            for e,(a,b) in zip(plan["events"],pairs)]
        original=[digest(row) for row in rows]
        _original_row_crosswalk(rows,pairs,plan,original)
        rows[0]["bank"]["scored_terminal_labels"]=1
        with self.assertRaisesRegex(ValueError,"copied AFTER rows"):_original_row_crosswalk(rows,pairs,plan,original)
        rows[0]["bank"]["scored_terminal_labels"]=0
        pairs[0][0].receipt["status"]="promoted"
        with self.assertRaisesRegex(ValueError,"copied AFTER rows"):_original_row_crosswalk(rows,pairs,plan,original)

    def test_new_paired_history_refuses_before_runtime_or_bank_launch(self):
        raw={};programs=dict(p1={},p2={});receipt=dict(complete_actor_histories_validated=True)
        admitted=OwnedNativeAfterAdmission(raw,programs,receipt,_payload(raw,programs,receipt),lambda:None)
        with patch("pokezero.mcts_eval.search_over_raw_after_admission.admit_owned_native_after",return_value=admitted), \
                patch("pokezero.mcts_eval.search_over_raw_after_bank.execute_owned_after_bank") as bank, \
                patch("pokezero.mcts_eval.search_over_raw_outcome_run.verify_outcome_runtime") as runtime:
            with self.assertRaisesRegex(ValueError,"newly available paired histories"):
                execute_original_native_after_bank({},ledger=None,ownership=None,event_id="event",expected_plan_sha256="a"*64,
                    source_config=None,native_extract=None,runtime=None,policies=None,null_only=True)
        runtime.assert_not_called();bank.assert_not_called()

    def test_equal_recipe_field_replacements_do_not_preserve_original_objects(self):
        for key in ("config","runtime"):
            recipe=NativeAfterRecipe(config=dict(value=1),runtime=dict(value=2))
            guard=_freeze_after_recipe(recipe);before=digest(_state_payload(recipe))
            object.__setattr__(recipe,key,deepcopy(getattr(recipe,key)))
            self.assertEqual(digest(_state_payload(recipe)),before)
            with self.assertRaisesRegex(ValueError,"class/field references"):guard()
        recipe=NativeAfterRecipe(unqualified_reason="unavailable");guard=_freeze_after_recipe(recipe)
        object.__setattr__(recipe,"extra",True)
        with self.assertRaisesRegex(ValueError,"class/field references"):guard()

    def test_first_ledger_callback_cannot_replace_equal_recipe_before_preflight(self):
        recipe=NativeAfterRecipe(config=dict(value=1),runtime=dict(value=2))
        ledger=SimpleNamespace(verify=lambda:object.__setattr__(recipe,"runtime",deepcopy(recipe.runtime)))
        with patch("pokezero.mcts_eval.search_over_raw_after_population.verify_outcome_runtime") as runtime:
            with self.assertRaisesRegex(ValueError,"class/field references"):
                preflight_after_population(ledger,recipe,checkpoint_contract=None)
        runtime.assert_not_called()

    def test_producer_freezes_recipe_before_its_first_preflight_callback(self):
        recipe=NativeAfterRecipe(config=dict(value=1),runtime=dict(value=2))
        def replace_config(*args,**kwargs):object.__setattr__(recipe,"config",deepcopy(recipe.config))
        with patch("pokezero.mcts_eval.search_over_raw_after_population.preflight_after_population",side_effect=replace_config), \
                patch("pokezero.mcts_eval.search_over_raw_after_population._enclosing") as scope:
            with self.assertRaisesRegex(ValueError,"class/field references"):
                produce_owned_after_population(None,manifest=None,chain_rows=None,private_chains=None,
                    recipe=recipe,checkpoint_contract=None,ownership=None,native_extract=None)
        scope.assert_not_called()
