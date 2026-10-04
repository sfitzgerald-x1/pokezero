"""Full-policy contract gates for the isolated own-policy opponent mode."""

from dataclasses import replace
import json
import random
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.engine_search import (
    EngineMctsConfig, EngineMctsPolicy, EngineSearchFallbackError,
    EngineSearchWitnessError, OpponentRequestOrderResolution, policy_opponent_world_seed,
)
import test_engine_search as existing

Fixtures = SimpleNamespace(**{name: getattr(existing.ModelWorldParallelismTests, name)
    for name in ("_policy", "_context", "_report", "_timed_report", "_world")})


def config(**overrides):
    args = dict(leaf_eval="model", model_path="model.pt", checkpoint_path="checkpoint.pt",
                tables_path="tables.json", strict_fallbacks=True,
                policy_opponent=True, policy_opponent_seed=13)
    args.update(overrides)
    return EngineMctsConfig(**args)


class PolicyOpponentConfigTests(unittest.TestCase):
    def test_initialization_requires_checkpoint_bound_source_and_real_value_head(self):
        from pokezero.neural_policy import FreshValueHeadWarning
        import warnings
        model_config = SimpleNamespace(window_size=1)
        model = SimpleNamespace(eval=Mock())
        result = SimpleNamespace(model_config=model_config, belief_set_source_hash="registered")
        def initialize(source_hash, *, fresh=False):
            def load(*args, **kwargs):
                if fresh:
                    warnings.warn("untrained value head", FreshValueHeadWarning)
                return model, result
            with (patch("pathlib.Path.exists", return_value=True),
                  patch("pathlib.Path.read_text", return_value="{}"),
                  patch("pokezero.neural_policy.load_transformer_checkpoint_payload", return_value={}),
                  patch("pokezero.neural_policy.parse_transformer_model_config", return_value=model_config),
                  patch("pokezero.engine_search._fence_calibration_seam"),
                  patch("pokezero.engine_search._latch_encoder_tables_to_model_config", return_value="{}"),
                  patch("pokezero.neural_policy.load_transformer_checkpoint", side_effect=load)):
                return EngineMctsPolicy(dex=None, module=object(), config=config(),
                    set_source=SimpleNamespace(metadata=SimpleNamespace(source_hash=source_hash)))
        for source in (None, "different"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "belief source"):
                initialize(source)
        with self.assertRaises(FreshValueHeadWarning):
            initialize("registered", fresh=True)
        loaded = initialize("registered")
        self.assertIs(loaded._policy_opponent_model, model)
        self.assertIs(loaded._policy_opponent_result, result)
        model.eval.assert_called_once()

    def test_requires_explicit_seed_and_strict_unconfounded_mode(self):
        config()
        for overrides in (
            {"policy_opponent_seed": None}, {"policy_opponent_seed": True},
            {"policy_opponent_seed": -1}, {"policy_opponent_seed": 2**64},
            {"policy_opponent": "yes"}, {"policy_opponent": False},
            {"strict_fallbacks": False}, {"model_priors": False},
            {"use_opponent_priors": True}, {"rollout_leaf_shadow": True},
            {"rollout_leaf_eval": True}, {"early_stop": True},
            {"depth_min": 1}, {"worlds_min": 1}, {"leaf_eval": "hp_fraction"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides)

    def test_seed_split_is_reproducible_and_identity_sensitive(self):
        ctx = Fixtures._context()
        record = {"seed": 123}
        first = policy_opponent_world_seed(config(), ctx, record)
        self.assertEqual(first, policy_opponent_world_seed(config(), ctx, record))
        variants = [
            policy_opponent_world_seed(config(policy_opponent_seed=14), ctx, record),
            policy_opponent_world_seed(config(), ctx, {"seed": 124}),
            policy_opponent_world_seed(config(), SimpleNamespace(**{**vars(ctx), "player_id": "p2"}), record),
            policy_opponent_world_seed(config(), SimpleNamespace(**{**vars(ctx), "decision_round_index": 1}), record),
        ]
        self.assertEqual(len(set([first, *variants, 123])), 6)


class PolicyOpponentEngineTests(unittest.TestCase):
    def test_unsupported_world_is_not_replaced_by_a_new_draw(self):
        from pokezero.engine_world import EngineWorldUnsupported
        from contextlib import ExitStack
        policy = Fixtures._policy(workers=1)
        policy._config = replace(policy._config, strict_fallbacks=True,
                                 policy_opponent=True, policy_opponent_seed=13)
        policy._fixed_override = object()
        policy._dex = None
        with ExitStack() as stack:
            stack.enter_context(patch.object(policy, "_advance_live_fold", return_value=object()))
            stack.enter_context(patch.object(policy, "_public_effect_signals", return_value=({}, {}, {}, {}, {})))
            stack.enter_context(patch.object(policy, "_recharging_slots", return_value=set()))
            stack.enter_context(patch.object(policy, "_truant_loaf_slots", return_value=set()))
            builder = stack.enter_context(patch("pokezero.engine_search.world_battle_spec",
                side_effect=EngineWorldUnsupported("observer-gap", "registered unsupported state")))
            with self.assertRaisesRegex(EngineSearchWitnessError, "construction refused"):
                policy.select_action_with_context(Fixtures._context(), rng=random.Random(7))
        builder.assert_called_once()
        self.assertEqual(policy.stats.worlds_attempted, 1)

    def run_policy(self, *, workers=1, deadline=None, bad=None, provider_error=False,
                   report_override=None, order_missing=False):
        policy = Fixtures._policy(workers=workers, deadline_ms=deadline)
        policy._config = replace(policy._config, strict_fallbacks=True,
                                 policy_opponent=True, policy_opponent_seed=13)
        ctx = Fixtures._context()
        calls = []

        class Native:
            def search_batched_multi_encoded(_self, *args, **kwargs):
                calls.append((args, kwargs))
                if args[0] == bad:
                    raise ValueError("registered provider refused this world")
                report = (Fixtures._timed_report(60, 40, time_budget_ms=args[-1])
                          if deadline is not None else Fixtures._report(60, 40))
                report.update(policy_opponent_mode="own_policy_callback",
                              policy_opponent_seed=kwargs["policy_opponent_seed"],
                              policy_opponent_evals=3, policy_opponent_provider_calls=3, policy_opponent_samples=100,
                              policy_opponent_s=.02, model_priors=True)
                report.update(report_override or {})
                return json.dumps(report)

        def kwargs(_policy, context, record):
            if provider_error:
                raise ValueError("public prefix is incomplete")
            return dict(policy_opponent_callback=lambda raw: (),
                        policy_opponent_seed=policy_opponent_world_seed(_policy._config, context, record),
                        policy_opponent_request_order=record["_policy_opponent_request_order"])

        fake_module = SimpleNamespace(FoldState=SimpleNamespace(from_payload=lambda payload: object()))
        native = Native()
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(EngineMctsPolicy, "_native_handles_for_model_world_workers", return_value=(Native(), Native())),
            patch.object(EngineMctsPolicy, "_validate_model_root_observation", return_value=None),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
            patch.object(EngineMctsPolicy, "_policy_opponent_native_kwargs", kwargs),
            patch("pokezero.engine_search.opponent_request_order_resolution",
                  return_value=OpponentRequestOrderResolution(
                      None if order_missing else ("chansey",),
                      "lost_active_permutation" if order_missing else "resolved")),
        ):
            result = policy._search_model(ctx, [Fixtures._world("a"), Fixtures._world("b")],
                                          SimpleNamespace(to_payload=lambda: {}), random.Random(7))
        return result, calls

    def test_serial_parallel_and_deadline_pass_provider_and_preserve_streams(self):
        cases = [(1, None), (2, None), (1, 1000), (2, 1000)]
        seeds = []
        for workers, deadline in cases:
            with self.subTest(workers=workers, deadline=deadline):
                decision, calls = self.run_policy(workers=workers, deadline=deadline)
                self.assertEqual(len(calls), 2)
                witness = decision.metadata["engine_mcts"]["policy_opponent"]
                self.assertEqual(len(witness["native_invocations"]), 2)
                self.assertEqual(sum(row["evaluations"] for row in witness["native_invocations"]), 6)
                self.assertEqual(sum(row["provider_calls"] for row in witness["native_invocations"]), 6)
                ordered = sorted((args[0], args[9], kwargs["policy_opponent_seed"])
                                 for args, kwargs in calls)
                seeds.append(ordered)
                expected_rng = random.Random(7)
                self.assertEqual([row[1] for row in ordered],
                                 [expected_rng.getrandbits(63) for _ in range(2)])
                self.assertTrue(all(callable(kwargs["policy_opponent_callback"]) for _, kwargs in calls))
        self.assertTrue(all(value == seeds[0] for value in seeds))

    def test_failed_world_cannot_be_hidden_by_healthy_sibling(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaisesRegex(EngineSearchWitnessError, "world refused"):
                self.run_policy(workers=workers, bad="b")

    def test_provider_setup_failure_cannot_fall_back(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaises(EngineSearchWitnessError):
                self.run_policy(workers=workers, provider_error=True)

    def test_missing_or_corrupt_native_witness_refuses_the_decision(self):
        for override in (
            {"policy_opponent_mode": None}, {"policy_opponent_seed": True},
            {"policy_opponent_seed": 7}, {"policy_opponent_evals": -1},
            {"policy_opponent_provider_calls": -1}, {"policy_opponent_provider_calls": 2},
            {"policy_opponent_samples": False}, {"policy_opponent_s": float("nan")},
            {"policy_opponent_s": -1}, {"prior_fallbacks": 1},
            {"model_priors": False}, {"iterations": 99},
            {"remaining_iterations": 1}, {"requested_iterations": 99},
        ):
            with self.subTest(override=override), self.assertRaises(EngineSearchWitnessError):
                self.run_policy(report_override=override)

    def test_unknown_sampled_request_order_refuses_instead_of_guessing(self):
        with self.assertRaisesRegex(EngineSearchWitnessError, "lost_active_permutation"):
            self.run_policy(order_missing=True)

    def test_context_free_call_refuses_and_failed_call_records_whole_wall(self):
        policy = Fixtures._policy(workers=1)
        policy._config = replace(policy._config, strict_fallbacks=True,
                                 policy_opponent=True, policy_opponent_seed=13)
        with self.assertRaises(EngineSearchFallbackError):
            policy.select_action(None, rng=random.Random(1))
        def fail(*args, **kwargs):
            time.sleep(.02)
            raise EngineSearchWitnessError("refused")
        with patch.object(policy, "_search", side_effect=fail), self.assertRaises(EngineSearchWitnessError):
            policy.select_action_with_context(Fixtures._context(), rng=random.Random(1))
        self.assertEqual(policy.stats.decisions, 1)
        self.assertGreaterEqual(policy.stats.decision_wall_seconds, .02)


if __name__ == "__main__":
    unittest.main()
