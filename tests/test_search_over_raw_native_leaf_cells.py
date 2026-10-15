"""Authored atomic/export inventory, no scientific continuations or models."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_native_leaf_cells import (
    leaf_cell_law, plan_native_leaf_cells, NativeLeafCellPlan,
)
from test_search_over_raw_supervision import selection_inputs, INPUT_BINDING
from test_search_over_raw_ancestry import CONTRACT, export_fixture, row_from_payload
import json


RUN = dict(directory="/authored/native", job_sha256="a"*64, binding_sha256="b"*64,
    statistics_root_id="original-root", selection_root_id="original-selection", selection_seed=7,
    runtime_sha256="c"*64)


class NativeLeafCellPlanTests(unittest.TestCase):
    def fixture(self, split=False):
        manifest = selection_inputs("incumbent")["consumed_world_inputs"]
        if split:
            second = deepcopy(manifest["calls"][0])
            manifest["calls"][0]["construction_ordinals"] = [1, 0]
            second.update(call_ordinal=1, construction_ordinals=[3, 2])
            manifest["calls"].append(second)
            for world in manifest["rows"]:
                world["call_ordinal"] = world["ordinal"]//2
        exports = [export_fixture(call, i) for i, call in enumerate(manifest["calls"])]
        return dict(evidence=dict(consumed_world_inputs=manifest, native_searched_ancestry=exports))

    def plan(self, selected, run=RUN):
        return plan_native_leaf_cells(selected, run_binding=run, consumed_binding=INPUT_BINDING,
            ancestry_contract=CONTRACT)

    def test_four_aliases_do_not_multiply_two_events_or_sixteen_cells(self):
        selected = self.fixture(); before = digest(selected)
        plan = self.plan(selected); row = plan.receipt()
        self.assertIs(type(plan), NativeLeafCellPlan)
        self.assertEqual(digest(selected), before)
        self.assertEqual((row["actual_native_calls"], row["actual_model_evaluations"],
            row["retained_unique_evaluation_events"], row["construction_alias_event_records"],
            row["required_label_cells"]), (1, 2, 2, 8, 16))
        for event in row["events"]:
            self.assertEqual(event["construction_aliases"], [0, 1, 2, 3])
            self.assertEqual(event["canonical_construction_ordinal"], 0)
            self.assertEqual(event["required_after_snapshot_index"], 1)
            self.assertEqual([c["replicate"] for c in event["labels"]], list(range(8)))
            self.assertTrue(all(c["status"] == "UNSTARTED_UNCERTAIN" and c["signed_outcome"] is None
                for c in event["labels"]))
        cells = [c for e in row["events"] for c in e["labels"]]
        self.assertEqual(len({c["cell_id"] for c in cells}), 16)
        self.assertEqual(len({c["chance_stream_key"] for c in cells}), 16)
        for key in ("completed_label_cells", "label_process_launches"):
            self.assertEqual(row[key], 0)
        for key in ("independent_terminal_labels", "native_evaluation_input_parity", "runtime_qualified",
                "scientific_collection_authorized"):
            self.assertFalse(row[key])

    def test_identical_frontiers_and_capture_hashes_in_two_calls_are_four_events(self):
        selected = self.fixture(split=True)
        events = self.plan(selected).receipt()["events"]
        self.assertEqual(len(events), 4)
        self.assertEqual(len({e["event_id"] for e in events}), 4)
        self.assertEqual(events[0]["native_capture_sha256"], events[2]["native_capture_sha256"])
        self.assertEqual(events[0]["frontier_utf8_sha256"], events[2]["frontier_utf8_sha256"])
        self.assertNotEqual(events[0]["event_id"], events[2]["event_id"])
        self.assertEqual([e["canonical_construction_ordinal"] for e in events], [0, 0, 2, 2])
        self.assertEqual(self.plan(selected).receipt()["required_label_cells"], 32)

    def test_alias_rule_has_no_constructor_eligibility_or_outcome_input(self):
        selected = self.fixture()
        selected["evidence"]["consumed_world_inputs"]["calls"][0]["construction_ordinals"] = [3, 2, 1, 0]
        row = self.plan(selected).receipt()
        self.assertTrue(all(e["canonical_construction_ordinal"] == 0 for e in row["events"]))
        self.assertEqual(row["law"], leaf_cell_law())
        self.assertEqual(row["law"]["policy"], "raw_argmax_both_seats_from_after_step_zero")
        with self.assertRaises(TypeError):
            plan_native_leaf_cells(selected, run_binding=RUN, consumed_binding=INPUT_BINDING,
                ancestry_contract=CONTRACT, eligible_aliases=[3])

    def test_missing_duplicate_or_numeric_alias_membership_refuses_not_drops(self):
        for aliases in ([1, 2, 3], [0, 0, 2, 3], [False, 1, 2, 3], [0, 1, 2, 4]):
            selected = self.fixture()
            selected["evidence"]["consumed_world_inputs"]["calls"][0]["construction_ordinals"] = aliases
            with self.subTest(aliases=aliases), self.assertRaises(ValueError): self.plan(selected)
        selected = self.fixture(); selected["evidence"]["native_searched_ancestry"].clear()
        with self.assertRaises(ValueError): self.plan(selected)

    def test_missing_retained_evaluation_cannot_shrink_the_population(self):
        selected = self.fixture()
        call = selected["evidence"]["consumed_world_inputs"]["calls"][0]
        payload = json.loads(selected["evidence"]["native_searched_ancestry"][0]["export_utf8"])
        payload["evaluations"].pop()
        selected["evidence"]["native_searched_ancestry"][0] = row_from_payload(payload, call)
        with self.assertRaisesRegex(ValueError, "retained sample omitted"):
            self.plan(selected)

    def test_original_run_identity_changes_event_and_all_chance_streams(self):
        selected = self.fixture(); original = self.plan(selected).receipt()["events"]
        for field, value in (("directory", "/other/native"), ("job_sha256", "d"*64),
                ("binding_sha256", "e"*64), ("runtime_sha256", "f"*64), ("selection_seed", 8)):
            changed = self.plan(selected, dict(RUN, **{field:value})).receipt()["events"]
            self.assertTrue(all(a["event_id"] != b["event_id"] for a,b in zip(original, changed)))
            self.assertTrue(all(a["labels"][0]["chance_stream_key"] != b["labels"][0]["chance_stream_key"]
                for a,b in zip(original, changed)))

    def test_actual_float32_frame_and_prediction_are_retained_not_labels(self):
        selected = self.fixture(); events = self.plan(selected).receipt()["events"]
        exports = json.loads(selected["evidence"]["native_searched_ancestry"][0]["export_utf8"])
        original = {e["capture"]["evaluation_ordinal"]:e for e in exports["evaluations"]}
        for event in events:
            actual = original[event["evaluation_ordinal"]]
            self.assertEqual(event["model_self_value01_bits"], actual["model_self_value01_bits"])
            self.assertEqual(event["model_signed_value"], actual["capture"]["model_signed_value"])
            self.assertTrue(event["model_self_side_one"])
            self.assertEqual(event["model_subject"], "p1")
            self.assertTrue(all(c["signed_outcome"] is None for c in event["labels"]))

    def test_receipt_mutation_cannot_change_frozen_plan(self):
        plan = self.plan(self.fixture()); before = plan.sha256
        plan.receipt()["events"][0]["labels"][0]["signed_outcome"] = 1
        self.assertEqual(plan.sha256, before)
        self.assertIsNone(plan.receipt()["events"][0]["labels"][0]["signed_outcome"])
        with self.assertRaises(FrozenInstanceError): plan.payload_utf8 = "{}"
