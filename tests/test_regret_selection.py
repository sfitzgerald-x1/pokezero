"""Engineering contracts only; authored weights and states, not strength data."""
import json
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import unittest

from pokezero.engine_search import (
    EngineMctsConfig, EngineSearchWitnessError, native_search_args,
    native_selection_kwargs, validate_native_selection,
)
import test_model_priors_search as fixture
import test_engine_search as python_fixture


MODEL = dict(leaf_eval="model", model_path="m.pt", checkpoint_path="c.pt", tables_path="t.json")


class RegretSelectionConfigTests(unittest.TestCase):
    def test_defaults_and_legacy_native_call_are_unchanged(self):
        old, explicit = EngineMctsConfig(), EngineMctsConfig(selection="puct")
        self.assertEqual(old.selection, "puct")
        self.assertEqual(native_selection_kwargs(old), {})
        self.assertEqual(native_selection_kwargs(SimpleNamespace()), {})
        record = dict(state_str="state", ctx_json="ctx", seed=3, side_key="side_one")
        def args(config):
            return native_search_args(config, record, tables_json="t", root_inputs="r",
                                      rust_fold=None, early_stop_min_sims=0)
        self.assertEqual(args(old), args(explicit))
        self.assertEqual(len(args(old)), 12)

    def test_flag_is_model_only_and_rejects_ambiguous_compositions(self):
        for options in (dict(selection="unknown"), dict(selection=True), dict(selection="regret"),
                        dict(MODEL, selection="regret", fpu_reduction=0.0),
                        dict(MODEL, selection="regret", rollout_leaf_shadow=True),
                        dict(MODEL, selection="regret", policy_opponent=True),
                        dict(MODEL, selection="regret", c_puct=float("nan")),
                        dict(MODEL, selection="regret", c_puct=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EngineMctsConfig(**options)
        config = EngineMctsConfig(**MODEL, selection="regret", use_opponent_priors=True)
        self.assertEqual(native_selection_kwargs(config), {"selection": "regret"})

    def test_missing_or_false_native_opt_in_cannot_masquerade_as_the_arm(self):
        config = EngineMctsConfig(**MODEL, selection="regret")
        for row in ({}, dict(selection="puct", selection_draws=10, iterations=5),
                    dict(selection="regret", selection_draws=True, iterations=0),
                    dict(selection="regret", selection_draws=9, iterations=5)):
            with self.subTest(row=row), self.assertRaises(EngineSearchWitnessError):
                validate_native_selection(config, row)
        validate_native_selection(config, dict(selection="regret", selection_draws=10, iterations=5))
        validate_native_selection(config, dict(selection="regret", selection_draws=0, iterations=0))
        validate_native_selection(EngineMctsConfig(), {})

    def test_real_python_serial_and_parallel_call_seams_forward_the_opt_in(self):
        helper = python_fixture.ModelWorldParallelismTests()
        class Native:
            def __init__(self, omit_for=None):
                self.calls, self.omit_for = [], omit_for
            def search_batched_multi_encoded(self, *args, **kwargs):
                self.calls.append(kwargs)
                report = helper._report(70, 30)
                if len(args) > 24:
                    report = helper._timed_report(70, 30, time_budget_ms=args[-1])
                if args[0] != self.omit_for:
                    report.update(selection="regret", selection_draws=200)
                return json.dumps(report)
        for workers, deadline in ((1, None), (2, None), (2, 10000)):
            with self.subTest(workers=workers, deadline=deadline):
                policy = helper._policy(workers=workers, deadline_ms=deadline)
                policy._config = replace(policy._config, selection="regret")
                handles = [Native() for _ in range(workers)]
                decision = helper._run(policy, handles[0], handles,
                    [helper._world("world-a"), helper._world("world-b")])
                calls = [call for handle in handles for call in handle.calls]
                self.assertEqual(calls, [{"selection": "regret"}] * 2)
                self.assertEqual(decision.metadata["engine_mcts"]["selection"], "regret")
        policy = helper._policy(workers=2)
        policy._config = replace(policy._config, selection="regret")
        handles = [Native("world-a"), Native("world-a")]
        with self.assertRaisesRegex(EngineSearchWitnessError, "selection witness"):
            helper._run(policy, handles[0], handles,
                [helper._world("world-a"), helper._world("world-b")])


@unittest.skipUnless(fixture._crate_ready, "requires freshly built model feature")
class NativeRegretSelectionTests(fixture._EncodedSearchFixture, unittest.TestCase):
    def run_search(self, *, mode=None, position=None, **kwargs):
        p = self.position if position is None else position
        options = dict(max_depth=3, seed=17, model_priors=True, arm_priors=True)
        options.update(kwargs)
        if mode is not None:
            options["selection"] = mode
        ctx = (self._ctx_with_opponent_order(position=p) if options.get("use_opponent_priors")
               else p["ctx"])
        return json.loads(self.native.search_batched_multi_encoded(p["state_str"], 96, 8,
            self.tables_json, p["row_inputs"], ctx,
            fixture.pokezero_search.FoldState.from_payload(p["fold_state"]), **options))

    def test_explicit_puct_reproduces_implicit_path_without_new_fields(self):
        old, explicit = self.run_search(), self.run_search(mode="puct")
        self.assertNotIn("selection", old)
        self.assertEqual(set(old), set(explicit))
        for key in ("side_one", "side_two", "root_value", "iterations", "root_priors",
                    "decision_nodes", "chance_nodes", "leaf_evals", "max_depth_reached"):
            self.assertEqual(old[key], explicit[key], key)

    def test_regret_reaches_both_seats_and_children_with_unchanged_priors(self):
        for position, priced in product(self.perspective_positions.values(), (False, True)):
            with self.subTest(side=position["self_side"], priced=priced):
                old = self.run_search(position=position, use_opponent_priors=priced)
                on = self.run_search(mode="regret", position=position, use_opponent_priors=priced)
                repeated = self.run_search(mode="regret", position=position, use_opponent_priors=priced)
                self.assertEqual(on["selection"], "regret")
                self.assertGreater(on["selection_draws"], 2 * on["iterations"])
                for side in ("side_one", "side_two"):
                    self.assertEqual(sum(s["visits"] for s in on[side]), 96)
                    self.assertEqual({s["move"]: s["prior"] for s in on[side]},
                                     {s["move"]: s["prior"] for s in old[side]})
                    self.assertEqual(on[side], repeated[side])
                self.assertNotEqual((old["side_one"], old["side_two"]),
                                    (on["side_one"], on["side_two"]))
                if priced:
                    opponent = "side_two" if position["self_side"] == "side_one" else "side_one"
                    self.assertGreater(len({s["prior"] for s in on[opponent]}), 1)

    def test_native_boundary_rejects_unsupported_compositions(self):
        for options in (dict(mode="bad"), dict(mode="regret", fpu_reduction=0.0),
                        dict(mode="regret", rollout_leaf_mode="model_value"),
                        dict(mode="regret", c_puct=float("inf")),
                        dict(mode="regret", policy_opponent_callback=lambda _: [],
                             policy_opponent_seed=1, policy_opponent_request_order=[])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_search(**options)


if __name__ == "__main__":
    unittest.main()
