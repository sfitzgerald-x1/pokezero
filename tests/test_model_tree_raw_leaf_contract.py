"""Incumbent raw-terminal integration contracts; synthetic work, not strength."""
from copy import deepcopy
from dataclasses import replace
import json
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.engine_search import (
    EngineMctsPolicy, EngineSearchFallbackError, EngineSearchWitnessError,
    OpponentRequestOrderResolution, RAW_LEAF_COUNTERS, model_raw_leaf_witness,
    raw_policy_leaf_seed, require_model_leaf_witness, validate_native_raw_leaf_witness,
    native_search_args,
)
from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider, refusal_diagnostic
from pokezero.mcts_eval.search_over_raw_adapters import _incumbent_runtime
from test_model_tree_hp_leaf_contract import config as base_config
from test_policy_opponent_engine import Fixtures


def config(**changes):
    return base_config(**{**dict(model_leaf_override="raw_policy_terminal", rollout_policy="raw_argmax",
        rollout_count=1, rollout_threads=1, rollout_max_plies=250, rollout_seed=71,
        rollout_branch_on_damage=True), **changes})


def report(seed=17, **changes):
    result = {key: 0 for key in RAW_LEAF_COUNTERS}
    result.update(model_leaf_override="raw_policy_terminal", raw_leaf_policy="both_seats_own_raw_masked_argmax",
        raw_leaf_request_protocol="private_choice_attempt_redecision_v1",
        raw_leaf_value_frame="side_one_absolute", raw_leaf_model_forwards_retained=True,
        raw_leaf_tree_counters_include_cancelled_work=True, raw_leaf_max_plies=250,
        raw_leaf_seed=raw_policy_leaf_seed(config(), {"seed": seed}), raw_leaf_branch_on_damage=True,
        iterations=100, model_evals=25, raw_leaf_started=12, raw_leaf_terminal=12,
        raw_leaf_plies=30, raw_leaf_provider_calls=60, raw_leaf_policy_evals=40,
        raw_leaf_choice_attempts=60,
        raw_leaf_policy_s=.02, time_budget_exhausted=False)
    result.update(changes)
    return result


class RawLeafContractTests(unittest.TestCase):
    def test_raw_runtime_uses_the_same_checkpoint_and_source_loading_fences(self):
        from test_policy_opponent_engine import PolicyOpponentConfigTests
        # Re-exercise the complete initialization boundary with policy-opponent
        # sampling OFF and terminal raw leaves ON, including fresh-head refusal.
        with patch("test_policy_opponent_engine.config", side_effect=config):
            PolicyOpponentConfigTests("test_initialization_requires_checkpoint_bound_source_and_real_value_head").test_initialization_requires_checkpoint_bound_source_and_real_value_head()

    def test_only_explicit_fixed_strict_raw_policy_mode_is_admitted(self):
        config()
        for changes in (dict(rollout_policy="uniform"), dict(rollout_count=2), dict(rollout_threads=2),
                dict(model_world_workers=2), dict(model_world_workers=True),
                dict(use_opponent_priors=True), dict(rollout_seed=True),
                dict(rollout_seed=-1), dict(rollout_seed=2**64), dict(rollout_max_plies=0),
                dict(model_priors=False), dict(strict_fallbacks=False), dict(early_stop=True),
                dict(policy_opponent=True, policy_opponent_seed=1), dict(rollout_leaf_eval=True)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "model_leaf_override"):
                config(**changes)

    def test_native_abi_preserves_tree_slots_and_splits_leaf_seed(self):
        row = dict(state_str="state", ctx_json="{}", seed=17, side_key="side_two")
        kw = dict(tables_json="tables", root_inputs="root", rust_fold=object(), early_stop_min_sims=0)
        baseline = native_search_args(base_config(), row, **kw)
        raw = native_search_args(config(), row, **kw)
        self.assertEqual(raw[:12], baseline)
        self.assertEqual(raw[12:24], [0, False, False, None, False, "raw_policy_terminal",
            1, 250, "raw_argmax", raw_policy_leaf_seed(config(), row), 1, True])
        self.assertEqual(native_search_args(config(), row, time_budget_ms=800, **kw), [*raw, 800])
        self.assertNotEqual(raw[21], row["seed"])
        self.assertNotEqual(raw[21], raw_policy_leaf_seed(config(), {"seed": 18}))

    def test_native_completed_cancelled_and_discarded_rows_have_distinct_counts(self):
        for r in (report(), report(iterations=1),
                report(iterations=0, raw_leaf_started=1, raw_leaf_terminal=0,
                raw_leaf_cancelled_rows=15, raw_leaf_cancelled_traversals=1, time_budget_exhausted=True),
                report(iterations=0, raw_leaf_started=2, raw_leaf_terminal=2,
                raw_leaf_discarded_terminal_rows=2, raw_leaf_cancelled_rows=4,
                raw_leaf_cancelled_traversals=4, time_budget_exhausted=True)):
            validate_native_raw_leaf_witness(r, cap=250, seed=r["raw_leaf_seed"], branch_on_damage=True)

    def test_native_missing_relabelled_imputed_or_nonconserving_work_refuses(self):
        good = report()
        for key in good:
            changed = dict(good)
            del changed[key]
            # An exhausted flag is needed only when cancellation occurred.
            if key == "time_budget_exhausted":
                continue
            with self.subTest(missing=key), self.assertRaises(EngineSearchWitnessError):
                validate_native_raw_leaf_witness(changed, cap=250, seed=good["raw_leaf_seed"], branch_on_damage=True)
        for change in (dict(raw_leaf_terminal=13), dict(raw_leaf_started=26),
                dict(raw_leaf_policy_evals=61), dict(raw_leaf_cap_fallbacks=1),
                dict(raw_leaf_discarded_terminal_rows=1), dict(raw_leaf_policy_s=float("nan")),
                dict(raw_leaf_seed=True), dict(raw_leaf_value_frame="self_relative"),
                dict(raw_leaf_model_forwards_retained=False), dict(rollout_leaf_mode="rollout"),
                dict(raw_leaf_cancelled_rows=1), dict(raw_leaf_started=True)):
            with self.subTest(change=change), self.assertRaises(EngineSearchWitnessError):
                validate_native_raw_leaf_witness({**good, **change}, cap=250,
                    seed=good["raw_leaf_seed"], branch_on_damage=True)

    def test_private_redecision_work_is_versioned_and_cannot_be_erased(self):
        terminal = report(iterations=1, model_evals=1, raw_leaf_started=1,
            raw_leaf_terminal=1, raw_leaf_plies=0, raw_leaf_provider_calls=0,
            raw_leaf_policy_evals=0, raw_leaf_choice_attempts=0)
        validate_native_raw_leaf_witness(terminal, cap=250,
            seed=terminal["raw_leaf_seed"], branch_on_damage=True)
        phantom = {**terminal, "raw_leaf_choice_attempts": 2,
            "raw_leaf_provider_calls": 2, "raw_leaf_trapped_switch_rejections": 2,
            "raw_leaf_private_redecisions": 2}
        with self.assertRaises(EngineSearchWitnessError):
            validate_native_raw_leaf_witness(phantom, cap=250,
                seed=phantom["raw_leaf_seed"], branch_on_damage=True)
        good = report(raw_leaf_trapped_switch_rejections=2, raw_leaf_private_redecisions=2,
            raw_leaf_choice_attempts=62, raw_leaf_provider_calls=62)
        validate_native_raw_leaf_witness(good, cap=250, seed=good["raw_leaf_seed"], branch_on_damage=True)
        for change in (dict(raw_leaf_choice_attempts=59), dict(raw_leaf_private_redecisions=3),
                dict(raw_leaf_choice_attempts=60, raw_leaf_provider_calls=60),
                dict(raw_leaf_choice_attempts=63, raw_leaf_provider_calls=63),
                dict(raw_leaf_private_redecisions=1), dict(raw_leaf_trapped_switch_rejections=1000),
                dict(raw_leaf_request_protocol="native_truth_mask"), dict(raw_leaf_private_redecisions=True)):
            with self.subTest(change=change), self.assertRaises(EngineSearchWitnessError):
                validate_native_raw_leaf_witness({**good, **change}, cap=250,
                    seed=good["raw_leaf_seed"], branch_on_damage=True)
        old = model_raw_leaf_witness(config(), [dict(world_seed=17, belief_multiplicity=1,
            completed_iterations=100, model_evals=25, report=good)])
        old["schema"] = "pokezero.model-tree-raw-leaf.v1"
        with self.assertRaises(EngineSearchWitnessError):
            require_model_leaf_witness({"engine_mcts": {"model_leaf_override": old}},
                model_leaf_override="raw_policy_terminal")

    def test_aggregate_counts_compute_once_not_belief_multiplicity(self):
        r = report()
        row = dict(world_seed=17, belief_multiplicity=4, completed_iterations=100, model_evals=25, report=r)
        w = model_raw_leaf_witness(config(), [row])
        require_model_leaf_witness({"engine_mcts": {"model_leaf_override": w}}, model_leaf_override="raw_policy_terminal")
        self.assertEqual(w["raw_leaf_started"], 12)
        for mutate in (lambda w: w.update(raw_leaf_started=48),
                lambda w: w.update(raw_leaf_started=12.0),
                lambda w: w.update(raw_leaf_cap_fallbacks=False),
                lambda w: w.update(raw_leaf_policy_s=True),
                lambda w: w["native_invocations"][0].update(completed_iterations=99),
                lambda w: w["native_invocations"][0]["report"].update(raw_leaf_seed=7),
                lambda w: w.update(native_invocations=[])):
            changed = deepcopy(w)
            mutate(changed)
            with self.assertRaises(EngineSearchWitnessError):
                require_model_leaf_witness({"engine_mcts": {"model_leaf_override": changed}}, model_leaf_override="raw_policy_terminal")
        with self.assertRaises(EngineSearchWitnessError):
            require_model_leaf_witness({"engine_mcts": {"model_leaf_override": w}}, model_leaf_override=None)

    def test_each_seat_gets_an_independent_raw_callback_and_order(self):
        policy = object.__new__(EngineMctsPolicy)
        policy._config = config()
        callback = Mock(side_effect=["seat-one", "seat-two"])
        orders = [["rattata"], ["chansey"]]
        with patch.object(policy, "_checkpoint_policy_callback", callback):
            kwargs = policy._policy_opponent_native_kwargs("public-context", {"_raw_policy_request_orders": orders})
        self.assertEqual(kwargs["raw_policy_callbacks"], ["seat-one", "seat-two"])
        self.assertEqual([call.args for call in callback.call_args_list], [("public-context", "p1"), ("public-context", "p2")])
        self.assertTrue(all(call.kwargs == {"raw_argmax": True} for call in callback.call_args_list))
        self.assertIsNot(kwargs["raw_policy_request_orders"][0], orders[0])
        with self.assertRaises(EngineSearchWitnessError):
            policy._policy_opponent_native_kwargs("public-context", {})

    def test_context_free_raw_policy_never_falls_back_to_uniform_action(self):
        policy = object.__new__(EngineMctsPolicy)
        policy._config = config()
        with self.assertRaises(EngineSearchFallbackError):
            policy.select_action(object(), rng=random.Random(7))

    def test_factory_binds_terminal_policy_without_switching_the_incumbent_tree(self):
        with patch("pokezero.mcts_eval.policy_opponent_profile._LiveEngineTimingDecider") as decider:
            make_profile_decider(None, "showdown", arm="incumbent_mcts", mode="matched_deadline",
                opponent_seed=71, deadline_ms=1000, native_batch_guard_ms=64, model_leaf_override="raw_policy_terminal")
        kw = decider.call_args.kwargs
        self.assertEqual((kw["rollout_count"], kw["rollout_max_plies"], kw["rollout_policy"], kw["rollout_seed"]), (1, 250, "raw_argmax", 71))
        self.assertFalse(kw["policy_opponent"])
        self.assertTrue(kw["model_priors"])
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider", return_value=Mock()) as make:
            _incumbent_runtime(None, "showdown", 3., leaf="raw_rollout")
        self.assertEqual(make.call_args.kwargs["model_leaf_override"], "raw_policy_terminal")

    def test_actual_timing_adapter_admits_and_forwards_the_declared_raw_leaf(self):
        from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider
        from pokezero.mcts_eval.manifest import SearchConfig
        from tests.test_mcts_eval_lattice import C
        decider = object.__new__(_LiveEngineTimingDecider)
        # Stop at artifact loading: this exercises actual constructor admission
        # without opening any scientific runtime or checkpoint.
        with patch("pokezero.mcts_eval.lattice.materialize_search_artifacts",
                side_effect=RuntimeError("synthetic artifact loading boundary")):
            with self.assertRaisesRegex(RuntimeError, "synthetic artifact loading boundary"):
                decider.__init__(C, showdown_root="showdown", model_priors=True,
                    use_opponent_priors=False, record_joint_actions=True,
                    model_leaf_override="raw_policy_terminal", model_world_workers=1,
                    rollout_count=1, rollout_max_plies=250, rollout_policy="raw_argmax",
                    rollout_seed=71, rollout_threads=1, rollout_branch_on_damage=True)
        decider._policies = {}
        decider._artifacts = {"model_path": "/model.ts", "tables_path": "/tables.json"}
        decider._dex = decider._set_source = decider._annotation_source = object()
        with patch("pokezero.engine_search.EngineMctsPolicy") as policy:
            decider._policy_for(SearchConfig(depth=6, sims=4096, batch=16, worlds=4))
        actual = policy.call_args.kwargs["config"]
        self.assertEqual(actual.model_leaf_override, "raw_policy_terminal")
        self.assertEqual((actual.rollout_count, actual.rollout_threads, actual.model_world_workers), (1, 1, 1))
        self.assertEqual((actual.rollout_policy, actual.rollout_max_plies, actual.rollout_seed), ("raw_argmax", 250, 71))
        self.assertTrue(actual.rollout_branch_on_damage)
        self.assertTrue(actual.strict_fallbacks)
        self.assertTrue(actual.record_joint_actions)
        self.assertTrue(actual.model_priors)
        self.assertFalse(actual.use_opponent_priors)
        self.assertEqual((actual.search_sims, actual.search_batch, actual.search_depth, actual.worlds), (4096, 16, 6, 4))

    def test_refusal_diagnostic_survives_wrapper_without_reclassifying_outcome(self):
        native = ValueError("synthetic cap refusal")
        diagnostic = dict(schema="raw-policy-terminal-refusal-v1", diagnostic_only_not_policy_input=True,
            unresolved_no_fallback=True, terminal_leaves=0, cap_refusals=1)
        native.raw_policy_leaf_diagnostic = json.dumps(diagnostic)
        wrapper = EngineSearchWitnessError("world refused")
        wrapper.__cause__ = native
        self.assertEqual(refusal_diagnostic(wrapper), diagnostic)


class RawLeafPolicyIntegrationTests(unittest.TestCase):
    def run_policy(self, bad=None, order_missing=False):
        policy = Fixtures._policy(workers=1)
        policy._config = config(worlds=2, search_sims=100, search_batch=10)
        ctx = Fixtures._context()
        ctx.public_materialization_state.self_request = {"side": {"pokemon": [{"details": "Rattata, L100"}]}}
        calls = []
        class Native:
            def search_batched_multi_encoded(_self, *args, **kwargs):
                calls.append((args, kwargs))
                if bad == "refused":
                    raise ValueError("synthetic nonterminal cap")
                r = {**Fixtures._report(60, 40), **report(args[9])}
                if bad == "missing":
                    del r["raw_leaf_terminal"]
                return json.dumps(r)
        module = SimpleNamespace(FoldState=SimpleNamespace(from_payload=lambda payload: object()))
        with patch.dict(sys.modules, {"pokezero_search": module}), \
                patch.object(policy, "_native", return_value=Native()), \
                patch.object(policy, "_validate_model_root_observation"), \
                patch.object(policy, "_root_inputs_json", return_value="{}"), \
                patch.object(policy, "_checkpoint_policy_callback", return_value=lambda raw: (1.,)), \
                patch("pokezero.engine_search.opponent_request_order_resolution", return_value=
                    OpponentRequestOrderResolution(None if order_missing else ("chansey",), "resolved")):
            result = policy._search_model(ctx, [Fixtures._world("a"), Fixtures._world("b")],
                SimpleNamespace(to_payload=lambda: {}), random.Random(7))
        return result, calls

    def test_full_policy_passes_both_providers_and_preserves_tree_random_stream(self):
        decision, calls = self.run_policy()
        self.assertEqual(len(calls), 2)
        rng = random.Random(7)
        self.assertEqual([args[9] for args, _ in calls], [rng.getrandbits(63), rng.getrandbits(63)])
        for args, kwargs in calls:
            self.assertEqual(kwargs["raw_policy_request_orders"], [["rattata"], ["chansey"]])
            self.assertEqual(len(kwargs["raw_policy_callbacks"]), 2)
        w = decision.metadata["engine_mcts"]["model_leaf_override"]
        self.assertEqual((len(w["native_invocations"]), w["raw_leaf_started"]), (2, 24))
        require_model_leaf_witness(decision.metadata, model_leaf_override="raw_policy_terminal")

    def test_native_refusal_or_missing_work_cannot_be_hidden_by_a_healthy_sibling(self):
        for bad in ("refused", "missing"):
            with self.subTest(bad=bad), self.assertRaises(EngineSearchWitnessError):
                self.run_policy(bad=bad)

    def test_unresolved_request_order_refuses_before_search(self):
        with self.assertRaisesRegex(EngineSearchWitnessError, "request order refused"):
            self.run_policy(order_missing=True)


if __name__ == "__main__":
    unittest.main()
