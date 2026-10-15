"""Temporary random models and native fixtures, not fresh source games."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search
import test_raw_policy_leaf_native as raw_fixture
from test_rollout_model_priors import TIMING_FIELDS
from pokezero.mcts_eval.search_over_raw_native_fixed_values import (
    SCHEMA, NativeFixedWorkValueContract, NativeFixedWorkValueSession, read_fixed_native_values,
)


@unittest.skipUnless(_crate_ready, "requires freshly built isolated model runtime")
class NativeFixedWorkValueTests(_EncodedSearchFixture, unittest.TestCase):
    def session(self):
        value = NativeFixedWorkValueSession(NativeFixedWorkValueContract(17),
            root_key="aa", native_module=pokezero_search)
        self.addCleanup(value.close)
        return value

    def search(self, position, handle=None, callbacks=None, **changes):
        options = dict(max_depth=2, seed=5, model_priors=True, use_opponent_priors=False,
            record_atomic_work=True, visited_value_bank=handle)
        if handle is not None:
            ctx = json.loads(position["ctx"])
            orders = []
            for i, slot in enumerate(("p1", "p2")):
                order = list(ctx[slot]); active = position["actives"][i]
                order[0], order[active] = order[active], order[0]
                orders.append(order)
            options.update(raw_policy_callbacks=callbacks or [raw_fixture.RawPolicyLeafNativeTest.certain]*2,
                raw_policy_request_orders=orders)
        options.update(changes)
        iterations = options.pop("iterations", 16)
        batch = options.pop("batch", 16)
        return self.native.search_batched_multi_encoded(position["state_str"], iterations, batch,
            self.tables_json, position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]), **options)

    def selected(self, position, callbacks=None, aliases=None):
        session = self.session()
        aliases = [0] if aliases is None else aliases
        handle = session.begin(dict(state_str=position["state_str"], seed=5, construction_ordinals=aliases))
        raw = self.search(position, handle, callbacks, iterations=16*len(aliases))
        session.complete(handle, raw)
        return session, handle, json.loads(raw)

    def test_actual_fixed_work_labels_preserve_selection_and_never_require_source_p2_history(self):
        self.assertEqual(set(self.perspective_positions), {"side_one", "side_two"})
        for side, position in self.perspective_positions.items():
            with self.subTest(side=side):
                # A deliberately small, authored terminal-capability fixture.
                # Its encoder context is not a production input-parity claim.
                position = self.small_native_position(position)
                calls = []
                def own_policy(raw):
                    value = json.loads(raw); calls.append(value)
                    return raw_fixture.RawPolicyLeafNativeTest.certain(raw)
                baseline = json.loads(self.search(position))
                session, handle, actual = self.selected(position, [own_policy]*2)
                self.assertFalse(calls, "post-selection labels must not price search leaves")
                self.assertEqual({k:v for k,v in baseline.items() if k not in TIMING_FIELDS},
                    {k:v for k,v in actual.items() if k not in TIMING_FIELDS})
                row, = session.label()
                value = json.loads(row["raw_native_export"])
                checked = row["readout"]
                self.assertEqual(value["schema"], SCHEMA)
                self.assertIsNone(value["label_deadline"])
                self.assertIsNone(value["search_decision_ceiling"])
                self.assertNotIn("original_deadline_at", value)
                self.assertEqual(checked["retained_events"], 2)
                self.assertEqual(checked["required_labels"], 16)
                self.assertEqual(checked["unknown_labels"], 0)
                self.assertEqual(checked["label_lossy_subcases"], value["label_lossy_subcases"])
                self.assertGreater(len(calls), 0)
                self.assertTrue(all(set(call) == {"native_request_bundle", "public_branch_lines", "opponent_slot"}
                    for call in calls))
                self.assertFalse(checked["independent_showdown_labels"])
                self.assertFalse(checked["scientific_calibration_evidence"])
                self.assertFalse(checked["runtime_qualified"])
                self.assertFalse(checked["owning_process_validated"])
                for leaf in value["leaves"]:
                    self.assertEqual(len(leaf["labels"]), 8)
                    self.assertNotIn("DEADLINE_UNCERTAIN", [r["status"] for r in leaf["labels"]])
                for mutation in ("oldschema", "deadline", "truth", "drop", "seed", "frame", "bits", "numeric", "lossy"):
                    changed = deepcopy(value)
                    if mutation == "oldschema": changed["schema"] = "pokezero.native-visited-value.engineering.v1"
                    if mutation == "deadline": changed["label_deadline"] = 1.
                    if mutation == "truth": changed["independent_showdown_labels"] = True
                    if mutation == "drop": changed["leaves"][0]["labels"].pop()
                    if mutation == "seed": changed["leaves"][0]["labels"][0]["seed"] += 1
                    if mutation == "frame": changed["model_self_side_one"] = not changed["model_self_side_one"]
                    # Reflection and signed-f32 rounding can erase a least-bit
                    # change, so this negative changes an exponent bit instead.
                    if mutation == "bits": changed["leaves"][0]["model_self_value01_bits"] ^= 0x00800000
                    if mutation == "numeric": changed["leaves"][0]["model_signed_value"] = True
                    if mutation == "lossy": changed["label_lossy_subcases"] = {"authored-negative": True}
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        read_fixed_native_values(json.dumps(changed), binding=row["binding"],
                            contract=session.contract, root_key="aa", worker=0)
                # Reader-only authored cap: do not claim the native fixture
                # encountered this cap. Its original denominator stays fixed.
                capped = deepcopy(value)
                capped["leaves"][0]["labels"][0].update(status="CAPPED_UNCERTAIN",
                    signed_outcome=None, boundaries=250)
                uncertain = read_fixed_native_values(json.dumps(capped), binding=row["binding"],
                    contract=session.contract, root_key="aa", worker=0)
                self.assertEqual(uncertain["required_labels"], 16)
                self.assertEqual(uncertain["unknown_labels"], 1)
                with self.assertRaises(ValueError): session.label()
                with self.assertRaisesRegex(ValueError, "already consumed"):
                    self.search(position, handle)

    def small_native_position(self, position):
        position = deepcopy(position)
        parts = position["state_str"].split("/")
        for side in range(2):
            fields = parts[side].split("=")
            for slot in range(6):
                mon = fields[slot].split(",")
                mon[6] = "150" if slot == position["actives"][side] else "0"
                fields[slot] = ",".join(mon)
            parts[side] = "=".join(fields)
        position["state_str"] = "/".join(parts)
        return position

    def test_provider_failure_and_bad_completion_consume_the_original_session(self):
        def broken(_raw): raise ValueError("authored native label provider failure")
        session, _, _ = self.selected(self.position, [broken]*2)
        with self.assertRaisesRegex(ValueError, "authored native label provider failure"): session.label()
        with self.assertRaises(ValueError): session.label()
        self.assertFalse(session.rows)
        session = self.session()
        handle = session.begin(dict(state_str=self.position["state_str"], seed=5, construction_ordinals=[0]))
        raw = self.search(self.position, handle)
        with self.assertRaises(ValueError): session.complete(handle, "{}")
        with self.assertRaises(ValueError): session.complete(handle, raw)
        with self.assertRaises(ValueError): session.label()

    def test_new_mode_cannot_adopt_deadlines_ancestry_or_other_evaluator(self):
        for options in (dict(time_budget_ms=1), dict(rollout_leaf_mode="hp_fraction")):
            session = self.session()
            handle = session.begin(dict(state_str=self.position["state_str"], seed=5, construction_ordinals=[0]))
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "native fixed-work value probe"):
                self.search(self.position, handle, **options)
            with self.assertRaises(ValueError): handle.label()
        for options in (dict(original_deadline_at=1.), dict(capture_searched_lines=True),
                dict(fixed_work_capture=True, capture_searched_lines=True), dict(capture_native_edges=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pokezero_search.NativeVisitedValueBank(**dict(dict(sample_seed=17, worker=0,
                    root_information_key="aa", original_deadline_at=0., fixed_work_native_labels=True), **options))

    def test_collapsed_allocations_preserve_work_without_multiplying_samples(self):
        position = self.small_native_position(self.position)
        session, _, report = self.selected(position, aliases=[0, 1, 2, 3])
        self.assertEqual(report["iterations"], 64)
        row, = session.label()
        self.assertEqual(row["readout"]["construction_ordinals"], [0, 1, 2, 3])
        self.assertEqual(row["readout"]["retained_events"], 2)
        self.assertEqual(row["readout"]["required_labels"], 16)
        self.assertFalse(row["readout"]["aliases_counted_as_samples"])

    def test_actual_smaller_batch_is_not_relabelled_as_the_declared_allocation(self):
        position = self.small_native_position(self.position)
        handle = pokezero_search.NativeVisitedValueBank(sample_seed=17, worker=0,
            root_information_key="aa", original_deadline_at=0., fixed_work_native_labels=True)
        self.addCleanup(handle.close)
        raw = self.search(position, handle, batch=8)
        binding = dict(native_call_parameters=dict(
            state_sha256=hashlib.sha256(position["state_str"].encode()).hexdigest(),
            requested_iterations=16, batch_size=8, chance_seed=5), construction_ordinals=[0],
            native_report=json.loads(raw), native_report_utf8=raw,
            report_utf8_sha256=hashlib.sha256(raw.encode()).hexdigest())
        export = handle.label()
        with self.assertRaisesRegex(ValueError, "actual allocation"):
            read_fixed_native_values(export, binding=binding,
                contract=NativeFixedWorkValueContract(17), root_key="aa", worker=0)
        for worker in (-1, 4, True):
            with self.subTest(worker=worker), self.assertRaises(ValueError):
                read_fixed_native_values(export, binding=binding,
                    contract=NativeFixedWorkValueContract(17), root_key="aa", worker=worker)

    def test_actual_search_without_session_completion_cannot_adopt_inserted_binding(self):
        position = self.small_native_position(self.position)
        session = self.session()
        handle = session.begin(dict(state_str=position["state_str"], seed=5, construction_ordinals=[0]))
        raw = self.search(position, handle)
        session.rows[0]["binding"] = dict(native_report=json.loads(raw))
        session.rows[0]["completion_attempted"] = True
        with self.assertRaisesRegex(ValueError, "original completed selections"):
            session.label()
        self.assertFalse(session.rows)
        with self.assertRaises(ValueError): handle.label()

    def test_actual_dispatch_wrapper_owns_collapsed_call_and_preserves_both_seat_selection(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import NativeInputHandle, NativeWorldInputs
        import test_search_over_raw_world_inputs as inputs
        from pokezero.mcts_eval.search_over_raw import digest
        from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import (
            RESULT_SCHEMA, runtime_receipt, read_native_fixed_value_result)
        for seat, source_position in self.perspective_positions.items():
            with self.subTest(seat=seat):
                position = self.small_native_position(source_position)
                ctx = json.loads(position["ctx"])
                binding = dict(root_binding="a"*64, subject="p1" if seat == "side_one" else "p2",
                    set_source_hash="b"*64, information_key="aa")
                recorder = NativeWorldInputs(binding, 16, deadline_limited=False, record_atomic_work=True)
                authored_world = inputs.FakeWorld(dict(authored=True),
                    {s:tuple(ctx[s]) for s in ("p1", "p2")}, dict(p1="side_one", p2="side_two"))
                context = SimpleNamespace(player_id=binding["subject"])
                for i in range(4):
                    recorder.attempt(context, i, inputs.FakeOverride(dict(p1="authored", p2="authored")))
                    recorder.constructed(context, authored_world,
                        SimpleNamespace(to_string=lambda:position["state_str"]))
                session = self.session()
                wrapper = NativeInputHandle(self.native, recorder, value_session=session)
                args = [position["state_str"], 64, 16, self.tables_json, position["row_inputs"], position["ctx"],
                    pokezero_search.FoldState.from_payload(position["fold_state"]), 2, 1.4, 5, True, True]
                baseline = json.loads(self.native.search_batched_multi_encoded(*args, record_atomic_work=True))
                orders = []
                for i, slot in enumerate(("p1", "p2")):
                    order = list(ctx[slot]); active = position["actives"][i]
                    order[0], order[active] = order[active], order[0]
                    orders.append(order)
                raw = wrapper.search_batched_multi_encoded(*args,
                    raw_policy_callbacks=[raw_fixture.RawPolicyLeafNativeTest.certain]*2,
                    raw_policy_request_orders=orders)
                actual = json.loads(raw)
                self.assertEqual({k:v for k,v in actual.items() if k not in TIMING_FIELDS},
                    {k:v for k,v in baseline.items() if k not in TIMING_FIELDS})
                self.assertEqual(session.bindings()[0]["construction_ordinals"], [0, 1, 2, 3])
                manifest = recorder.finish(dict(total_iterations=64), attempted=4, constructed=4, searched=4)
                selected = dict(status="SELECTED", scientific_strength_evidence=False, elapsed_seconds=0.,
                    evidence=dict(consumed_world_inputs=manifest))
                result = dict(schema=RESULT_SCHEMA, selected=selected, native_values=dict(schema=SCHEMA,
                    selected_receipt_sha256=digest(selected), native_invocations=session.label()),
                    label_wall_seconds=0., runtime_scope=runtime_receipt(session.contract))
                checked = read_native_fixed_value_result(result, contract=session.contract, root_key="aa")
                self.assertEqual(checked["required_labels"], 16)
                self.assertEqual(checked["unknown_labels"], 0)
                self.assertFalse(checked["owning_process_validated"])


class FixedWorkValueSessionGuardTests(unittest.TestCase):
    """Always-runnable ownership guards with authored FFI substitutes only."""
    def session(self):
        self.handles = []
        def factory(**_options):
            handle = SimpleNamespace(label_calls=0, close_calls=0, hook=lambda: None)
            def label():
                handle.label_calls += 1
                handle.hook()
                return "{}"
            def close(): handle.close_calls += 1
            handle.label, handle.close = label, close
            self.handles.append(handle)
            return handle
        session = NativeFixedWorkValueSession(NativeFixedWorkValueContract(17), root_key="aa",
            native_module=SimpleNamespace(NativeVisitedValueBank=factory))
        self.addCleanup(session.close)
        return session

    def completed(self):
        session = self.session()
        handle = session.begin(dict(state_str="authored-state", seed=5, construction_ordinals=[0]))
        with patch("pokezero.mcts_eval.search_over_raw_native_fixed_values.validate_native_report"):
            session.complete(handle, '{"iterations":16}')
        return session, handle

    def test_prelabel_same_value_replacements_are_rejected_and_original_handle_closed(self):
        for field in ("row", "parameters", "binding", "handle", "aliases"):
            with self.subTest(field=field):
                session, handle = self.completed()
                row = session.rows[0]
                if field == "row": session.rows[0] = dict(row)
                if field == "parameters": row[field] = dict(row[field])
                if field == "binding": row[field] = deepcopy(row[field])
                if field == "handle": row[field] = SimpleNamespace(label=lambda: self.fail("replacement labelled"))
                if field == "aliases": row["construction_ordinals"][0] = 1
                with self.assertRaises(ValueError): session.label()
                self.assertEqual(handle.label_calls, 0)
                self.assertEqual(handle.close_calls, 1)
                self.assertFalse(session.rows)

    def test_label_callback_cannot_replace_prior_binding(self):
        session, handle = self.completed()
        handle.hook = lambda: session.rows[0].update(binding=deepcopy(session.rows[0]["binding"]))
        with self.assertRaises(ValueError): session.label()
        self.assertEqual(handle.label_calls, 1)
        self.assertEqual(handle.close_calls, 1)
        self.assertFalse(session.rows)

    def test_inserted_binding_cannot_bypass_completion_or_start_next_invocation(self):
        session = self.session()
        handle = session.begin(dict(state_str="authored-state", seed=5, construction_ordinals=[0]))
        session.rows[0].update(binding={}, completion_attempted=True)
        with self.assertRaises(ValueError):
            session.begin(dict(state_str="second-state", seed=6, construction_ordinals=[1]))
        with self.assertRaises(ValueError): session.label()
        self.assertEqual(handle.label_calls, 0)
        self.assertEqual(handle.close_calls, 1)

    def test_disjoint_original_aliases_are_not_replaced(self):
        session, handle = self.completed()
        for aliases in ([0], [1, 1], [2, 1], [True], [-1], [4], []):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                session.begin(dict(state_str="other-state", seed=6, construction_ordinals=aliases))
        self.assertEqual(len(self.handles), 1)
        self.assertEqual(handle.label_calls, 0)
        other = session.begin(dict(state_str="other-state", seed=6, construction_ordinals=[1, 2, 3]))
        self.assertIsNot(other, handle)
        self.assertEqual(session.rows[-1]["parameters"]["requested_iterations"], 48)


if __name__ == "__main__":
    unittest.main()
