"""Phase 3 V1: a configurable value-head width, and a load path scoped to exactly that change.

The incumbent value head is **one linear functional of a 512-dim embedding** — 513 parameters,
0.01% of a 10.01M model — while the policy side gets the whole transformer. V1 tests whether
that is the binding constraint on sibling ordering by widening it, trunk untouched.

The risk this suite exists for is the LOAD PATH, not the head. A strict `load_state_dict`
refuses a changed head outright; `strict=False` would accept it *and* silently absorb a renamed
trunk tensor or a changed embedding width. So the tolerance is scoped to value-head keys and
every test below checks the scoping, not just the happy path.
"""
from __future__ import annotations

import unittest

from pokezero.neural_policy import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    EntityTokenTransformerPolicy,
    TransformerPolicyConfig,
    load_state_dict_allowing_fresh_value_head,
    torch_available,
)


def _cfg(hidden=None):
    return TransformerPolicyConfig.compact_category(
        observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_2,
        category_vocab=tuple(range(1, 17)), category_oov_buckets=4,
        policy_id="v1-capacity", window_size=2, token_type_vocab_size=8,
        categorical_feature_count=1, numeric_feature_count=1, embedding_dim=16,
        transformer_layers=1, attention_heads=4, feedforward_dim=32, dropout=0.0,
        value_head_hidden=hidden,
    )


class ValueHeadShapeTest(unittest.TestCase):
    def setUp(self):
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

    def test_default_is_the_incumbent_single_linear(self):
        """The default must not change any existing lineage."""
        import torch.nn as nn

        head = EntityTokenTransformerPolicy(_cfg()).value_head
        self.assertIsInstance(head, nn.Linear)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 16 + 1)

    def test_hidden_width_builds_an_mlp_and_adds_capacity(self):
        import torch.nn as nn

        head = EntityTokenTransformerPolicy(_cfg(32)).value_head
        self.assertIsInstance(head, nn.Sequential)
        self.assertGreater(sum(p.numel() for p in head.parameters()), 500)

    def test_zero_and_none_both_mean_incumbent(self):
        """`0` must not build a degenerate `Linear(dim, 0)`."""
        import torch.nn as nn

        for value in (None, 0):
            with self.subTest(value_head_hidden=value):
                self.assertIsInstance(
                    EntityTokenTransformerPolicy(_cfg(value)).value_head, nn.Linear)


class LoadPathScopingTest(unittest.TestCase):
    """The tolerance must cover the value head and NOTHING else."""

    def setUp(self):
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        self.incumbent = EntityTokenTransformerPolicy(_cfg())
        self.state = self.incumbent.state_dict()

    def test_widened_head_loads_and_reports_what_it_reinitialised(self):
        reinit = load_state_dict_allowing_fresh_value_head(
            EntityTokenTransformerPolicy(_cfg(32)), self.state)
        self.assertTrue(reinit, "the widened head must be reported as reinitialised")
        self.assertTrue(all(n.startswith("value_head") for n in reinit), reinit)

    def test_an_unchanged_head_reinitialises_NOTHING(self):
        """Otherwise a plain continuation could silently reset a trained head."""
        self.assertEqual(
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), self.state),
            [])

    def test_the_trunk_is_actually_restored_not_just_accepted(self):
        """A load that returns cleanly but leaves the trunk fresh would be worse than a raise."""
        import torch

        widened = EntityTokenTransformerPolicy(_cfg(32))
        load_state_dict_allowing_fresh_value_head(widened, self.state)
        after = widened.state_dict()
        checked = 0
        for name, tensor in self.state.items():
            if name.startswith("value_head"):
                continue
            self.assertTrue(torch.equal(after[name], tensor), f"{name} was not restored")
            checked += 1
        self.assertGreater(checked, 5, "too few trunk tensors compared to be meaningful")

    def test_a_trunk_shape_mismatch_still_RAISES(self):
        import torch

        bad = dict(self.state)
        name = next(n for n in self.state if n.startswith("token_type"))
        bad[name] = torch.zeros(1, 1)
        with self.assertRaises(ValueError) as ctx:
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), bad)
        self.assertIn("checkpoint-contract break", str(ctx.exception))

    def test_an_unexpected_non_value_key_still_RAISES(self):
        import torch

        bad = dict(self.state)
        bad["totally_new_tensor"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), bad)

    def test_a_missing_non_value_key_still_RAISES(self):
        bad = dict(self.state)
        del bad[next(n for n in self.state if n.startswith("token_type"))]
        with self.assertRaises(ValueError):
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), bad)


if __name__ == "__main__":
    unittest.main()
