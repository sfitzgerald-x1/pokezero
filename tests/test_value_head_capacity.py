"""Phase 3 V1: a configurable value-head width, and a load path scoped to exactly that change.

The incumbent value head is **one linear functional of a 512-dim embedding** — 513 parameters,
measured at **0.0050% of the 10,182,667** in `checkpoints/pz-v2-2-1m.pt` — while the policy side
gets the whole transformer. V1 tests whether that is the binding constraint on sibling ordering
by widening it, trunk untouched.

The risk this suite exists for is the LOAD PATH, not the head. A strict `load_state_dict`
refuses a changed head outright; `strict=False` raises on a reshape too, but it *does* absorb a
RENAMED tensor. So the tolerance is scoped to value-head keys and every test below checks the
scoping, not just the happy path.

Two things this suite learned the hard way and now pins: the arm must be **nonlinear** (a
`Sequential` with the activation swapped for `Identity` is a factored linear map, no more
expressive than the incumbent, and it passed an `isinstance` + parameter-count test), and the
untrained-head **warning** must actually fire (three mutations of it left the whole suite
green).
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

    def test_the_widened_head_is_actually_NONLINEAR(self):
        """Otherwise the arm is silently its own null control.

        `nn.GELU()` -> `nn.Identity()` left all 431 tests passing, because the shape test
        asserted `isinstance(Sequential)` and `> 500 params` -- both still true. But
        `Linear -> Identity -> Linear` is a factored linear map: provably no more expressive
        than the incumbent single `Linear`, so V1 would measure its own control and report
        "capacity does not help". Additivity is the property that separates them.
        """
        import torch

        head = EntityTokenTransformerPolicy(_cfg(32)).value_head
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        with torch.no_grad():
            lhs = head(a) + head(b)
            rhs = head(a + b) + head(torch.zeros(1, 16))
        self.assertGreater(
            float((lhs - rhs).abs().max()), 1e-4,
            "the widened head is ADDITIVE, i.e. an affine map -- the activation is missing or "
            "is Identity, and this arm cannot test capacity because it has none to add",
        )

    def test_zero_and_none_both_mean_incumbent(self):
        """`0` must not build a degenerate `Linear(dim, 0)`."""
        import torch.nn as nn

        for value in (None, 0):
            with self.subTest(value_head_hidden=value):
                self.assertIsInstance(
                    EntityTokenTransformerPolicy(_cfg(value)).value_head, nn.Linear)


class CheckpointRoundTripTest(unittest.TestCase):
    """End to end through the real writer and reader, which no test previously touched.

    F1's fatal bug -- `from_dict` dropping the width so a V1 checkpoint reloaded as a random
    incumbent head -- was only ever covered at the `to_dict`/`from_dict` layer. And the warning
    that is the sole signal of an untrained head had no test at all: `if False:`,
    `DeprecationWarning`, and `len(reinit) < 3` all left the suite green.
    """

    def setUp(self):
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

    def _write(self, tmp, model, model_config):
        from pokezero.neural_policy import (
            TransformerEpochMetrics, TransformerTrainingConfig, TransformerTrainingResult,
            save_transformer_checkpoint,
        )

        result = TransformerTrainingResult(
            model_config=model_config,
            training_config=TransformerTrainingConfig(objective="ppo", window_size=2),
            epochs=(TransformerEpochMetrics(epoch=1, examples=10, loss=0.5, policy_loss=-0.1,
                                            policy_accuracy=0.4, value_loss=0.25),))
        path = tmp / "ckpt.pt"
        save_transformer_checkpoint(path, model, result=result)
        return path

    def test_a_widened_checkpoint_round_trips_and_warns_about_NOTHING(self):
        import tempfile
        import warnings
        from pathlib import Path

        import torch
        from pokezero.neural_policy import load_transformer_checkpoint

        cfg = _cfg(32)
        model = EntityTokenTransformerPolicy(cfg)
        with torch.no_grad():
            for p in model.value_head.parameters():
                p.fill_(7.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), model, cfg)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded, result = load_transformer_checkpoint(path)
        self.assertEqual(result.model_config.value_head_hidden, 32)
        self.assertEqual(type(loaded.value_head).__name__, "Sequential")
        self.assertTrue(all(float(p.flatten()[0]) == 7.0
                            for p in loaded.value_head.parameters()),
                        "the trained head was not restored")
        self.assertEqual(
            sum(p.numel() for p in loaded.parameters()),
            sum(p.numel() for p in model.parameters()))
        self.assertEqual(
            [w for w in caught if issubclass(w.category, RuntimeWarning)], [],
            "a clean round trip must not warn about an untrained head")

    def test_a_head_mismatch_WARNS_and_names_the_tensors(self):
        """The mitigation round 1 demanded. Three mutations of it previously went unnoticed."""
        import tempfile
        import warnings
        from pathlib import Path

        import torch
        from pokezero.neural_policy import FreshValueHeadWarning, load_transformer_checkpoint

        # Write an INCUMBENT head under a config that claims a widened one, which is exactly
        # the shape a converter or a hand-edited payload produces.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), EntityTokenTransformerPolicy(_cfg()), _cfg(32))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_transformer_checkpoint(path)
        fresh = [w for w in caught if issubclass(w.category, FreshValueHeadWarning)]
        self.assertEqual(len(fresh), 1, [str(w.message) for w in caught])
        text = str(fresh[0].message)
        for name in ("value_head.0.weight", "value_head.0.bias",
                     "value_head.2.weight", "value_head.2.bias"):
            self.assertIn(name, text)
        self.assertIn("UNTRAINED", text)

    def test_the_warning_is_promotable_to_an_error(self):
        """A bare RuntimeWarning could not be promoted without catching unrelated ones."""
        import tempfile
        import warnings
        from pathlib import Path

        from pokezero.neural_policy import FreshValueHeadWarning, load_transformer_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), EntityTokenTransformerPolicy(_cfg()), _cfg(32))
            with warnings.catch_warnings():
                warnings.simplefilter("error", FreshValueHeadWarning)
                with self.assertRaises(FreshValueHeadWarning):
                    load_transformer_checkpoint(path)


class LoadPathScopingTest(unittest.TestCase):
    """The tolerance must cover the value head and NOTHING else."""

    def setUp(self):
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        self.incumbent = EntityTokenTransformerPolicy(_cfg())
        self.state = self.incumbent.state_dict()

    def test_widened_head_loads_and_reports_EXACTLY_what_it_reinitialised(self):
        """The exact set, not just truthiness.

        Review's uncaught mutation was `return ["value_head.THIS_NAME_IS_A_LIE"]` — the suite
        passed. Since this list is the only signal that a head is untrained, its contents are
        the thing that has to be right.
        """
        reinit = load_state_dict_allowing_fresh_value_head(
            EntityTokenTransformerPolicy(_cfg(32)), self.state)
        self.assertEqual(
            set(reinit),
            {"value_head.0.weight", "value_head.0.bias",
             "value_head.2.weight", "value_head.2.bias"})

    def test_a_MISSING_value_head_still_raises(self):
        """`main`'s safety net, which prefix-only scoping had removed.

        A checkpoint that simply lacks `value_head.*` — truncated write, partial converter,
        hand-built payload — must not load with a random head. The exemption applies only when
        the checkpoint carries a value-head tensor at all, so here there is nothing to excuse.
        """
        bad = {k: v for k, v in self.state.items() if not k.startswith("value_head.")}
        with self.assertRaises(ValueError) as ctx:
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), bad)
        self.assertIn("carries value-head tensors: False", str(ctx.exception))

    def test_a_junk_key_merely_beginning_with_value_head_is_refused(self):
        """`startswith("value_head")` swallowed `value_head_backdoor_trunk.weight`."""
        import torch

        bad = dict(self.state)
        bad["value_head_backdoor_trunk.weight"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            load_state_dict_allowing_fresh_value_head(
                EntityTokenTransformerPolicy(_cfg()), bad)

    def test_config_round_trip_preserves_the_head_width(self):
        """The fatal one: without this, a V1 checkpoint reloads as a random incumbent head."""
        cfg = _cfg(32)
        self.assertEqual(cfg.to_dict()["value_head_hidden"], 32)
        self.assertEqual(
            TransformerPolicyConfig.from_dict(cfg.to_dict()).value_head_hidden, 32)
        self.assertEqual(TransformerPolicyConfig.from_dict(cfg.to_dict()), cfg)
        plain = _cfg()
        self.assertIsNone(
            TransformerPolicyConfig.from_dict(plain.to_dict()).value_head_hidden)

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
