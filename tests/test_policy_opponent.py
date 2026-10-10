"""Own-head binding, sparse identity, and checkpoint-contract reference tests."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.neural_policy import TransformerPolicyConfig, torch_available
from pokezero.policy_opponent import policy_opponent_distribution
from pokezero.policy_opponent_view import PolicyOpponentViewError
from test_engine_world import _dex
from test_policy_opponent_view import VOCAB, view


def config(**overrides):
    arguments = dict(category_vocab=VOCAB.tokens, category_oov_buckets=VOCAB.oov_buckets,
        observation_schema_version="pokezero.observation.v4", transition_token_budget=0,
        embedding_dim=16, attention_heads=2, transformer_layers=1, feedforward_dim=32, dropout=0.0)
    arguments.update(overrides)
    return TransformerPolicyConfig.compact_category(**arguments)


class PolicyOpponentProviderTest(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.model = SimpleNamespace(config=self.config)
        self.result = SimpleNamespace(model_config=self.config, belief_set_source_hash=None)
        self.view = view()

    def distribution(self, indices=(4, 1, 0), **overrides):
        arguments = dict(native_action_indices=indices, model=self.model, result=self.result,
            category_vocab=VOCAB, dex=_dex())
        arguments.update(overrides)
        return policy_opponent_distribution(self.view, **arguments)

    def test_full_distribution_gather_includes_rare_switch_in_native_order(self):
        with patch("pokezero.neural_policy.evaluate_transformer_action_priors", return_value=(.89, .10, 0, 0, .01, 0, 0, 0, 0)) as forward:
            self.assertEqual(self.distribution(), (.01, .10, .89))
        arguments = forward.call_args.kwargs
        self.assertEqual(arguments["temperature"], 1.0)
        self.assertIs(arguments["model"], self.model)
        obs, = arguments["observations"]
        self.assertEqual(obs.metadata["showdown_slot"], "p2")
        self.assertEqual(obs.metadata["opponent_team"][0]["condition"], "67/100")

    def test_raw_argmax_is_certain_and_ties_use_policy_slot_not_native_order(self):
        for row, expected in (
            ((.89, .10, 0, 0, .01, 0, 0, 0, 0), (0., 0., 1.)),
            ((.25, .50, 0, 0, .25, 0, 0, 0, 0), (0., 1., 0.)),
            ((.5, 0, 0, 0, .5, 0, 0, 0, 0), (0., 0., 1.)),
            ((0, .5, 0, 0, .5, 0, 0, 0, 0), (0., 1., 0.)),
        ):
            with self.subTest(row=row), patch("pokezero.neural_policy.evaluate_transformer_action_priors", return_value=row):
                self.assertEqual(self.distribution(raw_argmax=True), expected)
                self.assertEqual(self.distribution(), tuple(row[i] for i in (4, 1, 0)))

    def test_raw_argmax_preserves_strict_surface_and_probability_refusals(self):
        with patch("pokezero.neural_policy.evaluate_transformer_action_priors", return_value=(0,) * 9) as forward:
            with self.assertRaisesRegex(PolicyOpponentViewError, "no mass"):
                self.distribution(raw_argmax=True)
            forward.reset_mock()
            with self.assertRaises(PolicyOpponentViewError):
                self.distribution((0, 1), raw_argmax=True)
            forward.assert_not_called()
            for flag in (1, "true", None):
                with self.subTest(flag=flag), self.assertRaisesRegex(PolicyOpponentViewError, "boolean"):
                    self.distribution(raw_argmax=flag)

    def test_shared_module_forward_is_protected_by_the_inference_lock(self):
        from threading import Lock
        lock = Lock()
        def evaluate(**kwargs):
            self.assertTrue(lock.locked())
            return (.89, .10, 0, 0, .01, 0, 0, 0, 0)
        with patch("pokezero.neural_policy.evaluate_transformer_action_priors", side_effect=evaluate):
            self.assertEqual(self.distribution(inference_lock=lock), (.01, .10, .89))
        self.assertFalse(lock.locked())

    def test_legal_map_must_match_request_exactly_before_any_forward(self):
        for indices in ((), (0, 1), (0, 1, 4, 5), (0, 0, 4), (None, 1, 4), (-1, 1, 4), (True, 1, 4)):
            with self.subTest(indices=indices), patch("pokezero.neural_policy.evaluate_transformer_action_priors") as forward:
                with self.assertRaises(PolicyOpponentViewError):
                    self.distribution(indices)
                forward.assert_not_called()

    def test_invalid_or_illegal_probability_mass_never_becomes_uniform(self):
        for row in ((0,) * 9, (math.nan, .1, 0, 0, .9, 0, 0, 0, 0),
                    (-.1, .2, 0, 0, .9, 0, 0, 0, 0), (.8, .1, .01, 0, .09, 0, 0, 0, 0), (.1,)):
            with self.subTest(row=row), patch("pokezero.neural_policy.evaluate_transformer_action_priors", return_value=row):
                with self.assertRaises(PolicyOpponentViewError):
                    self.distribution()

    def test_singleton_and_wait_require_certified_surface_but_no_network(self):
        from copy import deepcopy
        from test_policy_opponent_request import bundle, arguments
        from pokezero.policy_opponent_view import build_policy_opponent_view_from_native_bundle
        supplied = bundle()
        supplied["request"]["side"]["pokemon"][1]["condition"] = "0 fnt"
        supplied["request"]["active"][0]["moves"][1]["disabled"] = True
        supplied["self_move_states"]["snorlax"][1]["disabled"] = True
        supplied["native_action_indices"] = [0]
        self.view = build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied, **arguments())
        with patch("pokezero.neural_policy.evaluate_transformer_action_priors") as forward:
            self.assertEqual(self.distribution((0,)), (1.0,))
            self.assertEqual(self.distribution((0,), raw_argmax=True), (1.0,))
            with self.assertRaises(PolicyOpponentViewError):
                self.distribution((1,))
            waiting = deepcopy(bundle())
            waiting["request"].pop("active")
            waiting["request"]["wait"] = True
            waiting["native_action_indices"] = [None]
            self.view = build_policy_opponent_view_from_native_bundle(native_request_bundle=waiting, **arguments())
            self.assertEqual(self.distribution((None,)), (1.0,))
            self.assertEqual(self.distribution((None,), raw_argmax=True), (1.0,))
            forward.assert_not_called()

    def test_model_schema_masks_vocab_and_source_binding_fail_before_forward(self):
        cases = (
            dict(model=SimpleNamespace(config=config(window_size=2))),
            dict(model=SimpleNamespace(config=config(window_size=2)), result=SimpleNamespace(model_config=config(window_size=2))),
            dict(model=SimpleNamespace(config=config(exact_state_enabled=False)), result=SimpleNamespace(model_config=config(exact_state_enabled=False))),
            dict(model=SimpleNamespace(config=config(observation_schema_version="pokezero.observation.v3")), result=SimpleNamespace(model_config=config(observation_schema_version="pokezero.observation.v3"))),
            dict(category_vocab=replace(VOCAB, oov_buckets=1)),
            dict(result=SimpleNamespace(model_config=self.config, belief_set_source_hash="different-source")),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), patch("pokezero.neural_policy.evaluate_transformer_action_priors") as forward:
                with self.assertRaises(PolicyOpponentViewError):
                    self.distribution(**arguments)
                forward.assert_not_called()

    @unittest.skipUnless(torch_available(), "requires Python PyTorch for real own-head forward")
    def test_raw_argmax_matches_canonical_raw_policy_with_real_forward_and_ties(self):
        import random
        import torch
        from pokezero.neural_policy import TransformerPolicyOutput, TransformerSoftmaxPolicy

        class FixedHeads(torch.nn.Module):
            def __init__(self, config, row):
                super().__init__()
                self.config, self.row = config, row
            def forward(self, **inputs):
                batch = inputs["categorical_ids"].shape[0]
                # Deliberately contradictory auxiliary head and huge illegal
                # own logits: neither is allowed to influence the raw action.
                return TransformerPolicyOutput(torch.tensor(self.row).repeat(batch, 1),
                    torch.zeros(batch), torch.tensor([99., -99., -99., -99., -99., 99., 99., 99., 99.]).repeat(batch, 1))

        observation = self.view.observation(category_vocab=VOCAB, dex=_dex())
        for row in ((0., 1., 99., 99., 2., 99., 99., 99., 99.),
                    (2., 1., 99., 99., 2., 99., 99., 99., 99.),
                    (0., 2., 99., 99., 2., 99., 99., 99., 99.)):
            model = FixedHeads(self.config, row)
            policy = TransformerSoftmaxPolicy(model=model, result=self.result, deterministic=True)
            action = policy.select_action(observation, rng=random.Random(7)).action_index
            actual = self.distribution(model=model, raw_argmax=True)
            self.assertEqual(actual, tuple(float(index == action) for index in (4, 1, 0)))

    @unittest.skipUnless(torch_available(), "requires Python PyTorch for real own-head forward")
    def test_real_forward_uses_own_head_not_auxiliary_opponent_head(self):
        import torch
        from pokezero.neural_policy import TransformerPolicyOutput

        class ContradictoryHeads(torch.nn.Module):
            def __init__(self, checkpoint_config):
                super().__init__()
                self.config = checkpoint_config
                self.calls = 0

            def forward(self, **inputs):
                self.calls += 1
                batch = inputs["categorical_ids"].shape[0]
                own = torch.tensor([math.log(.89), math.log(.1), 99, 99, math.log(.01), 99, 99, 99, 99])
                auxiliary = torch.tensor([-99., -99., -99., -99., 99., -99., -99., -99., -99.])
                return TransformerPolicyOutput(own.repeat(batch, 1), torch.zeros(batch), auxiliary.repeat(batch, 1))

        model = ContradictoryHeads(self.config)
        actual = self.distribution(model=model)
        self.assertEqual(model.calls, 1)
        for value, expected in zip(actual, (.01, .1, .89)):
            self.assertAlmostEqual(value, expected, places=6)


if __name__ == "__main__":
    unittest.main()
