"""Loaded-model ancestry seam, never a scientific root or physics claim.

Use a committed engineering fixture and temporary random weights. Compare
capture-on against capture-off at identical fixed work, then feed the exact
native exports into the independent Python readers. No Showdown games,
checkpoint installation, panel admission or terminal labels occur here.
"""

from copy import deepcopy
import hashlib
import json
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search
from test_rollout_model_priors import TIMING_FIELDS
from pokezero.mcts_eval.search_over_raw_ancestry import (
    NativeAncestryContract, NativeAncestrySession, read_native_ancestry_export,
)
from pokezero.mcts_eval.search_over_raw_native_atomic_work import validate_native_report
from pokezero.mcts_eval.search_over_raw_native_boundary import extract_native_capture_boundaries
from pokezero.mcts_eval.search_over_raw_native_edge import (
    extract_native_edge_support, extract_native_successor_support,
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


_ready = _crate_ready and all(hasattr(pokezero_search, name) for name in (
    "NativeVisitedValueBank", "native_scoped_boundary", "native_edge_support",
    "native_scoped_successor_support",
))


@unittest.skipUnless(_ready, "requires the prospective model-enabled native build")
class NativeSearchedAncestryTest(_EncodedSearchFixture, unittest.TestCase):
    root_key = "aa"
    root_binding = sha("excluded-authored-native-root")

    def search(self, position, *, handle=None, **changes):
        options = dict(max_depth=2, seed=5, model_priors=True,
            use_opponent_priors=False, record_atomic_work=True,
            visited_value_bank=handle)
        if handle is not None:
            # The private frontier includes retained seat bridges even when
            # the only permitted operation is export. Their required callback
            # surfaces must never be invoked by capture/model selection.
            ctx = json.loads(position["ctx"])
            orders = []
            for index, slot in enumerate(("p1", "p2")):
                order = list(ctx[slot])
                active = position["actives"][index]
                order[0], order[active] = order[active], order[0]
                orders.append(order)
            def forbidden_provider(_raw):
                raise AssertionError("ancestry export invoked a diagnostic label provider")
            options.update(raw_policy_callbacks=[forbidden_provider, forbidden_provider],
                raw_policy_request_orders=orders)
        options.update(changes)
        # Keep original UTF-8 report bytes: reserializing parsed JSON would
        # not test the selection/export byte binding.
        return self.native.search_batched_multi_encoded(
            position["state_str"], 8, 4, self.tables_json,
            position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]), **options)

    def captured(self, position, *, edges=True):
        contract = NativeAncestryContract(17, capture_native_edges=edges)
        session = NativeAncestrySession(contract, root_key=self.root_key,
            native_module=pokezero_search)
        self.addCleanup(session.close)
        handle = session.begin(dict(state_str=position["state_str"], seed=5))
        raw = self.search(position, handle=handle)
        session.complete(handle, raw)
        row, = session.exports()
        call = dict(native_call_parameters=dict(state_sha256=sha(position["state_str"]),
            requested_iterations=8, batch_size=4, chance_seed=5),
            native_report=json.loads(raw), native_report_utf8=raw,
            report_utf8_sha256=sha(raw))
        return row, call, contract, handle

    def read(self, row, call, contract, position):
        return read_native_ancestry_export(row, call, contract=contract,
            root_key=self.root_key, worker=0, root_binding=self.root_binding,
            searching_side_one=position["self_side"] == "side_one")

    def test_actual_exports_join_atomic_rows_and_supported_paths_for_both_seats(self):
        self.assertEqual(set(self.perspective_positions), {"side_one", "side_two"})
        for seat, position in self.perspective_positions.items():
            for edges in (False, True):
                with self.subTest(seat=seat, edges=edges):
                    baseline = json.loads(self.search(position))
                    row, call, contract, _ = self.captured(position, edges=edges)
                    actual = call["native_report"]
                    self.assertEqual(
                        {k: v for k, v in actual.items() if k not in TIMING_FIELDS},
                        {k: v for k, v in baseline.items() if k not in TIMING_FIELDS})
                    atomic = validate_native_report(actual, **call["native_call_parameters"])
                    self.assertEqual(atomic["completed_iterations"], 8)
                    checked = self.read(row, call, contract, position)
                    self.assertGreater(checked["actual_model_evaluations"], 0)
                    self.assertEqual(len(checked["retained_evaluations"]), 2)
                    self.assertFalse(checked["full_world_equivalence"])
                    self.assertFalse(checked["runtime_qualified"])
                    payload = json.loads(row["export_utf8"])
                    self.assertIs(payload["model_self_side_one"], seat == "side_one")
                    for event in payload["evaluations"]:
                        capture = event["capture"]
                        boundaries = extract_native_capture_boundaries(capture,
                            root_binding=self.root_binding,
                            selection_report_sha256=call["report_utf8_sha256"],
                            native_extract=pokezero_search.native_scoped_boundary)
                        support = extract_native_edge_support(capture,
                            root_binding=self.root_binding,
                            selection_report_sha256=call["report_utf8_sha256"],
                            native_support=pokezero_search.native_edge_support)
                        successors = extract_native_successor_support(capture,
                            root_binding=self.root_binding,
                            selection_report_sha256=call["report_utf8_sha256"],
                            native_successor_support=pokezero_search.native_scoped_successor_support)
                        self.assertEqual(len(boundaries), len(capture["steps"]) + 1)
                        self.assertEqual(len(support), len(capture["steps"]))
                        self.assertEqual(len(successors), len(capture["steps"]))
                        self.assertTrue(all(r["status"] ==
                            ("SUPPORTED" if edges else "UNCHECKABLE") for r in support))
                        for view in boundaries:
                            self.assertFalse(view["runtime_qualified"])
                            self.assertFalse(view["scientific_collection_authorized"])
                        if edges:
                            self.assertTrue(all(r["support_report"]["captured_support"]
                                ["recorded_edge_supported"] for r in successors))
                            self.assertTrue(all(r["support_report"]["positive_successors"]
                                for r in successors))
                        else:
                            self.assertTrue(all(r["support_report"] is None for r in successors))

    def test_real_export_rejects_rehashed_wrong_report_sample_and_atomic_values(self):
        position = self.position
        row, call, contract, _ = self.captured(position)
        self.read(row, call, contract, position)
        for mode in ("report_bytes", "seed", "priority", "value_bits", "event", "path"):
            with self.subTest(mode=mode):
                changed_call = deepcopy(call)
                payload = json.loads(row["export_utf8"])
                first = payload["evaluations"][0]
                if mode == "report_bytes":
                    changed_call["native_report_utf8"] += " "
                    changed_call["report_utf8_sha256"] = sha(changed_call["native_report_utf8"])
                elif mode == "seed": payload["sample_seed"] += 1
                elif mode == "priority": first["priority"] += 1
                elif mode == "value_bits": first["model_self_value01_bits"] ^= 1
                elif mode == "event": first["capture"]["evaluation_ordinal"] += 1
                elif mode == "path": first["capture"]["steps"][-1]["key"][1] += 1
                raw = json.dumps(payload)
                changed = dict(row, export_utf8=raw, export_utf8_sha256=sha(raw))
                with self.assertRaises(ValueError):
                    self.read(changed, changed_call, contract, position)

    def test_fixed_capture_is_one_shot_and_rejects_deadlines_or_evaluator_override(self):
        _, _, _, handle = self.captured(self.position)
        with self.assertRaisesRegex(ValueError, "successful selection required"):
            handle.export_searched_ancestry()
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.search(self.position, handle=handle)
        for options in (dict(time_budget_ms=10), dict(rollout_leaf_mode="hp_fraction")):
            with self.subTest(options=options):
                session = NativeAncestrySession(NativeAncestryContract(17),
                    root_key=self.root_key, native_module=pokezero_search)
                self.addCleanup(session.close)
                handle = session.begin(dict(state_str=self.position["state_str"], seed=5))
                with self.assertRaisesRegex(ValueError, "fixed-work model capture forbids"):
                    self.search(self.position, handle=handle, **options)
                with self.assertRaisesRegex(ValueError, "successful selection required"):
                    handle.export_searched_ancestry()
                with self.assertRaisesRegex(ValueError, "already consumed"):
                    self.search(self.position, handle=handle)


if __name__ == "__main__":
    unittest.main()
