"""Observational callback phase timing; no search or scientific collection."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.neural_policy import TransformerInferenceTimingAccumulator
from pokezero.policy_opponent import make_policy_opponent_callback, policy_opponent_distribution
from pokezero.policy_opponent_diagnostics import PHASES, PolicyOpponentDiagnostics
from pokezero.policy_opponent_view import PolicyOpponentViewError
from test_engine_world import _dex
from test_policy_opponent import config
from test_policy_opponent_request import bundle
from test_policy_opponent_view import LINES, VOCAB, view
from test_showdown import FakeSetSource


ROW = (.8, .1, 0, 0, .1, 0, 0, 0, 0)
CLOCK = "pokezero.policy_opponent_diagnostics.perf_counter_ns"
FORWARD = "pokezero.neural_policy.evaluate_transformer_action_priors"


def arguments(**kwargs):
    cfg = config()
    return dict(public_lines=LINES, hp_visibility={"p1": "exact", "p2": "exact"},
        opponent_slot="p2", battle_id="authored", battle_seed=20,
        format_id="gen3randombattle", set_source=FakeSetSource(),
        model=SimpleNamespace(config=cfg),
        result=SimpleNamespace(model_config=cfg, belief_set_source_hash=None),
        category_vocab=VOCAB, dex=_dex(), **kwargs)


def payload(supplied=None, suffix=()):
    return json.dumps(dict(native_request_bundle=bundle() if supplied is None else supplied,
        public_branch_lines=list(suffix), opponent_slot="p2"))


class DiagnosticSinkTests(unittest.TestCase):
    def test_exact_nanosecond_accounting_and_inclusive_parent(self):
        sink = PolicyOpponentDiagnostics()
        with patch(CLOCK, side_effect=(10, 20, 50, 90)):
            with sink.phase("callback_total"):
                with sink.phase("payload_binding"):
                    pass
        rows = sink.snapshot()["phases"]
        self.assertEqual(rows["callback_total"], dict(calls=1, failed_calls=0,
            timed_calls=1, elapsed_seconds=80 / 1e9))
        self.assertEqual(rows["payload_binding"]["elapsed_seconds"], 30 / 1e9)
        self.assertEqual(rows["model_evaluation"]["calls"], 0)

    def test_exception_is_same_object_and_only_failed_counts_retained(self):
        sink = PolicyOpponentDiagnostics()
        error = RuntimeError("PRIVATE_PAYLOAD_MUST_NOT_BE_RETAINED")
        with self.assertRaises(RuntimeError) as raised:
            with sink.phase("callback_total"):
                raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(sink.snapshot()["phases"]["callback_total"]["failed_calls"], 1)
        self.assertNotIn(str(error), json.dumps(sink.snapshot()))

    def test_clock_failure_invalidates_without_replacing_success(self):
        for stamps in ((RuntimeError("clock"),), (10, RuntimeError("clock")),
                (-1,), (True,), (1.0,), (10, 9), (10, None)):
            with self.subTest(stamps=stamps):
                sink = PolicyOpponentDiagnostics()
                with patch(CLOCK, side_effect=stamps):
                    with sink.phase("payload_binding"):
                        result = (1., 0.)
                self.assertEqual(result, (1., 0.))
                snap = sink.snapshot()
                self.assertFalse(snap["valid"])
                self.assertEqual(snap["timing_faults"], 1)
                self.assertEqual(snap["phases"]["payload_binding"]["calls"], 1)
                self.assertEqual(snap["phases"]["payload_binding"]["timed_calls"], 0)

    def test_clock_failure_does_not_mask_policy_failure(self):
        sink = PolicyOpponentDiagnostics()
        error = PolicyOpponentViewError("original policy error")
        with patch(CLOCK, side_effect=RuntimeError("broken clock")):
            with self.assertRaises(PolicyOpponentViewError) as raised:
                with sink.phase("model_evaluation"):
                    raise error
        self.assertIs(raised.exception, error)
        self.assertFalse(sink.snapshot()["valid"])

    def test_snapshot_is_detached_bounded_and_explicit_about_limits(self):
        sink = PolicyOpponentDiagnostics()
        first = sink.snapshot()
        first["phases"]["callback_total"]["calls"] = 123
        snap = sink.snapshot()
        self.assertEqual(snap["phases"]["callback_total"]["calls"], 0)
        self.assertEqual(tuple(snap["phases"]), PHASES)
        self.assertTrue(snap["callback_total_contains_components_do_not_sum"])
        self.assertTrue(snap["model_evaluation_includes_lock_wait_and_canonical_setup"])
        self.assertTrue(snap["native_request_and_bridge_outside_python_spans"])
        self.assertTrue(snap["instrumentation_can_change_deadlines"])
        self.assertFalse(snap["qualifies_uninstrumented_runtime"])
        json.dumps(snap, allow_nan=False)
        with self.assertRaises(ValueError):
            sink.phase("arbitrary private phase")
        self.assertEqual(tuple(sink.snapshot()["phases"]), PHASES)

    def test_parallel_spans_have_invocation_owned_starts_and_exact_counts(self):
        sink = PolicyOpponentDiagnostics()
        def run(_):
            for index in range(100):
                with sink.phase("payload_binding"):
                    pass
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, range(4)))
        row = sink.snapshot()["phases"]["payload_binding"]
        self.assertEqual((row["calls"], row["failed_calls"], row["timed_calls"]), (400, 0, 400))
        self.assertGreaterEqual(row["elapsed_seconds"], 0)
        self.assertEqual(len(sink._rows), len(PHASES))


class CallbackDiagnosticTests(unittest.TestCase):
    def test_disabled_diagnostics_read_no_clock(self):
        args = arguments()
        with patch(CLOCK, side_effect=AssertionError("disabled timer was read")) as clock, \
                patch(FORWARD, return_value=ROW):
            for extra in ({}, {"diagnostics": None}):
                self.assertEqual(make_policy_opponent_callback(**args, **extra)(payload()), (.8, .1, .1))
            clock.assert_not_called()

    def test_exact_output_and_forward_argument_parity_both_modes(self):
        for raw_argmax in (False, True):
            sink = PolicyOpponentDiagnostics()
            args = arguments(raw_argmax=raw_argmax)
            with patch(FORWARD, return_value=ROW) as forward:
                plain = make_policy_opponent_callback(**args)(payload())
                first = forward.call_args.kwargs
                measured = make_policy_opponent_callback(**args, diagnostics=sink)(payload())
                second = forward.call_args.kwargs
            self.assertEqual(plain, measured)
            self.assertEqual(first, second)
            self.assertEqual(forward.call_count, 2)
            for phase in PHASES:
                self.assertEqual(sink.snapshot()["phases"][phase]["calls"], 1)
            self.assertTrue(sink.snapshot()["valid"])

    def test_payload_exception_taxonomy_and_cause_parity(self):
        invalid = ("not-json", "{}", payload(suffix=("|turn|2",)).replace('"p2"}', '"p1"}'),
            json.dumps(dict(native_request_bundle=bundle(), public_branch_lines=[1], opponent_slot="p2")))
        args = arguments()
        for value in invalid:
            errors = []
            for sink in (None, PolicyOpponentDiagnostics()):
                with self.assertRaises(PolicyOpponentViewError) as raised:
                    make_policy_opponent_callback(**args, diagnostics=sink)(value)
                errors.append((type(raised.exception), str(raised.exception), type(raised.exception.__cause__)))
            self.assertEqual(*errors)
            rows = sink.snapshot()["phases"]
            self.assertEqual(rows["payload_binding"]["failed_calls"], 1)
            self.assertEqual(rows["callback_total"]["failed_calls"], 1)
            self.assertEqual(rows["view_reconstruction"]["calls"], 0)

    def test_view_failure_retains_original_error_and_no_forward(self):
        supplied = bundle()
        supplied.pop("self_move_states")
        args = arguments()
        with patch(FORWARD) as forward:
            for sink in (None, PolicyOpponentDiagnostics()):
                with self.assertRaises(PolicyOpponentViewError) as raised:
                    make_policy_opponent_callback(**args, diagnostics=sink)(payload(supplied))
                if sink is None:
                    plain_error = str(raised.exception)
                else:
                    self.assertEqual(str(raised.exception), plain_error)
                    self.assertEqual(sink.snapshot()["phases"]["view_reconstruction"]["failed_calls"], 1)
            forward.assert_not_called()

    def test_output_failures_and_model_exceptions_are_unchanged(self):
        args = arguments()
        for row in ((0,) * 9, (.8, .1, .1, 0, 0, 0, 0, 0, 0)):
            errors = []
            for sink in (None, PolicyOpponentDiagnostics()):
                with patch(FORWARD, return_value=row), self.assertRaises(PolicyOpponentViewError) as raised:
                    make_policy_opponent_callback(**args, diagnostics=sink)(payload())
                errors.append(str(raised.exception))
            self.assertEqual(*errors)
            self.assertEqual(sink.snapshot()["phases"]["output_certification"]["failed_calls"], 1)
        sink = PolicyOpponentDiagnostics()
        error = RuntimeError("forward failed")
        with patch(FORWARD, side_effect=error), self.assertRaises(RuntimeError) as raised:
            make_policy_opponent_callback(**args, diagnostics=sink)(payload())
        self.assertIs(raised.exception, error)
        self.assertEqual(sink.snapshot()["phases"]["model_evaluation"]["failed_calls"], 1)
        self.assertEqual(sink.snapshot()["phases"]["output_certification"]["calls"], 0)

    def test_singleton_and_wait_still_certify_without_model_forward(self):
        single = bundle()
        single["request"]["side"]["pokemon"][1]["condition"] = "0 fnt"
        single["request"]["active"][0]["moves"][1]["disabled"] = True
        single["self_move_states"]["snorlax"][1]["disabled"] = True
        single["native_action_indices"] = [0]
        wait = deepcopy(bundle())
        wait["request"].pop("active")
        wait["request"]["wait"] = True
        wait["native_action_indices"] = [None]
        for supplied in (single, wait):
            sink = PolicyOpponentDiagnostics()
            with patch(FORWARD) as forward:
                self.assertEqual(make_policy_opponent_callback(**arguments(raw_argmax=True),
                    diagnostics=sink)(payload(supplied)), (1.,))
                forward.assert_not_called()
            self.assertEqual(sink.snapshot()["phases"]["observation_and_surface"]["calls"], 1)
            self.assertEqual(sink.snapshot()["phases"]["model_evaluation"]["calls"], 0)

    def test_lock_and_existing_neural_timing_are_forwarded_unchanged(self):
        lock, sink, timing = Lock(), PolicyOpponentDiagnostics(), TransformerInferenceTimingAccumulator()
        def forward(**kwargs):
            self.assertTrue(lock.locked())
            self.assertIs(kwargs["timing"], timing)
            timing.add_observation_encoding(.001)
            timing.add_neural_forward(.002, role="action_prior")
            return ROW
        with patch(FORWARD, side_effect=forward):
            make_policy_opponent_callback(**arguments(inference_lock=lock, timing=timing),
                diagnostics=sink)(payload())
        self.assertFalse(lock.locked())
        self.assertEqual(timing.snapshot().neural_forward_count, 1)
        self.assertEqual(sink.snapshot()["phases"]["model_evaluation"]["calls"], 1)

    def test_parent_then_sibling_then_parent_rebuilds_without_input_mutation(self):
        sink = PolicyOpponentDiagnostics()
        args = arguments()
        plain, measured = make_policy_opponent_callback(**args), make_policy_opponent_callback(**args, diagnostics=sink)
        values = [payload(), payload(suffix=("|turn|2",)), payload(suffix=("|turn|3",)), payload()]
        before = list(values)
        with patch(FORWARD, return_value=ROW) as forward:
            for value in values:
                self.assertEqual(plain(value), measured(value))
        self.assertEqual(values, before)
        self.assertEqual(forward.call_count, 8)
        self.assertEqual(sink.snapshot()["phases"]["view_reconstruction"]["calls"], 4)

    def test_arbitrary_or_subclass_collector_is_rejected_before_use(self):
        class Custom(PolicyOpponentDiagnostics):
            def phase(self, phase):
                raise AssertionError("custom diagnostic hook was invoked")
        args = arguments()
        for sink in (object(), Custom(), False):
            with self.subTest(sink=type(sink)), self.assertRaisesRegex(PolicyOpponentViewError, "aggregate"):
                make_policy_opponent_callback(**args, diagnostics=sink)
            with self.assertRaisesRegex(PolicyOpponentViewError, "aggregate"):
                policy_opponent_distribution(view(), native_action_indices=(4, 1, 0),
                    model=args["model"], result=args["result"], category_vocab=VOCAB,
                    dex=_dex(), diagnostics=sink)

    def test_real_canonical_forward_has_identical_output_and_call_count(self):
        import torch
        from pokezero.neural_policy import TransformerPolicyOutput
        class FixedHeads(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.config, self.calls = cfg, 0
            def forward(self, **inputs):
                self.calls += 1
                batch = inputs["categorical_ids"].shape[0]
                logits = torch.tensor([2., 1., 99., 99., 2., 99., 99., 99., 99.])
                return TransformerPolicyOutput(logits.repeat(batch, 1), torch.zeros(batch),
                    -logits.repeat(batch, 1))
        for raw_argmax in (False, True):
            args = arguments(raw_argmax=raw_argmax)
            args["model"] = FixedHeads(args["result"].model_config)
            timing, sink = TransformerInferenceTimingAccumulator(), PolicyOpponentDiagnostics()
            plain = make_policy_opponent_callback(**args)(payload())
            measured = make_policy_opponent_callback(**args, timing=timing, diagnostics=sink)(payload())
            self.assertEqual(plain, measured)
            self.assertEqual(args["model"].calls, 2)
            self.assertEqual(timing.snapshot().neural_forward_count, 1)
            self.assertEqual(sink.snapshot()["phases"]["model_evaluation"]["calls"], 1)
            if raw_argmax:
                self.assertEqual(measured, (1., 0., 0.))


if __name__ == "__main__":
    unittest.main()
