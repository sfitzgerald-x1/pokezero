from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from pokezero.neural_policy import (
    TransformerPolicyConfig,
    torch_available,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_model", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config() -> TransformerPolicyConfig:
    # Real token layout (token_count must cover the action-candidate block)
    # but a tiny network: the tests exercise export mechanics, not weights.
    return TransformerPolicyConfig.compact_category(
        category_vocab=("alpha", "beta", "gamma"),
        category_oov_buckets=2,
        categorical_feature_count=3,
        numeric_feature_count=5,
        embedding_dim=16,
        transformer_layers=1,
        attention_heads=2,
        feedforward_dim=32,
        dropout=0.0,
    )


@unittest.skipUnless(torch_available(), "requires torch")
class ExportableShimTest(unittest.TestCase):
    def test_shim_matches_eager_kwargs_forward(self) -> None:
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        module = _load_module()
        config = _tiny_config()
        model = EntityTokenTransformerPolicy(config).eval()
        shim = module.build_exportable_module(model)
        inputs = module.make_random_inputs(config, 3, seed=11)
        with torch.no_grad():
            reference = model(
                categorical_ids=inputs[0],
                numeric_features=inputs[1],
                token_type_ids=inputs[2],
                attention_mask=inputs[3],
                history_mask=inputs[4],
            )
            policy_logits, value, opponent_logits = shim(*inputs)
        torch.testing.assert_close(policy_logits, reference.policy_logits)
        torch.testing.assert_close(value, reference.value)
        torch.testing.assert_close(opponent_logits, reference.opponent_action_logits)

    def test_random_inputs_satisfy_config_shapes(self) -> None:
        module = _load_module()
        config = _tiny_config()
        inputs = module.make_random_inputs(config, 2, seed=5)
        categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask = inputs
        self.assertEqual(
            tuple(categorical_ids.shape),
            (2, config.window_size, config.token_count, config.categorical_feature_count),
        )
        self.assertEqual(
            tuple(numeric_features.shape),
            (2, config.window_size, config.token_count, config.numeric_feature_count),
        )
        self.assertEqual(tuple(token_type_ids.shape), (2, config.window_size, config.token_count))
        self.assertEqual(tuple(attention_mask.shape), (2, config.window_size, config.token_count))
        self.assertEqual(tuple(history_mask.shape), (2, config.window_size))
        self.assertTrue(bool(attention_mask[..., 0].all()), "first token must stay unmasked")
        self.assertLess(int(categorical_ids.max()), config.categorical_vocab_size)
        self.assertLess(int(token_type_ids.max()), config.token_type_vocab_size)


@unittest.skipUnless(torch_available(), "requires torch")
class TorchScriptRoundTripTest(unittest.TestCase):
    def test_trace_save_load_parity_at_unseen_batch_size(self) -> None:
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        module = _load_module()
        config = _tiny_config()
        model = EntityTokenTransformerPolicy(config).eval()
        shim = module.build_exportable_module(model)
        example = module.make_random_inputs(config, module.TRACE_BATCH, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            ts_path = Path(tmp) / "tiny_ts.pt"
            module.export_torchscript(shim, example, ts_path)
            self.assertTrue(ts_path.exists())
            loaded = torch.jit.load(str(ts_path))
        # A batch size never seen at trace time: catches a trace that baked
        # the batch dimension in as a constant.
        probe = module.make_random_inputs(config, 6, seed=13)
        with torch.no_grad():
            reference = shim(*probe)
            produced = loaded(*probe)
        for expected, actual in zip(reference, produced):
            torch.testing.assert_close(actual, expected)

    def test_validate_torchscript_reports_all_heads_and_batches(self) -> None:
        from pokezero.neural_policy import EntityTokenTransformerPolicy

        module = _load_module()
        config = _tiny_config()
        model = EntityTokenTransformerPolicy(config).eval()
        shim = module.build_exportable_module(model)
        example = module.make_random_inputs(config, module.TRACE_BATCH, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            traced = module.export_torchscript(shim, example, Path(tmp) / "tiny_ts.pt")
        diffs = module.validate_torchscript(shim, traced, config, seed=3, batch_size=4)
        expected_keys = {
            f"{name}@batch{batch}" for name in module.OUTPUT_NAMES for batch in (4, 1)
        }
        self.assertEqual(set(diffs), expected_keys)
        worst, passed = module._parity_verdict(diffs, module.DEFAULT_TOLERANCE)
        self.assertTrue(passed, f"self-parity must pass, worst diff {worst}")


@unittest.skipUnless(torch_available(), "requires torch")
class MaskDynamismTests(unittest.TestCase):
    """Lock in that masking is DYNAMIC in the traced graph, not baked.

    Trace with an all-ones attention mask, then require bit-exact parity on a
    sparse mixed mask — the exact class of silent divergence a future encoder
    change could introduce.
    """

    def test_trace_with_uniform_mask_matches_eager_on_mixed_mask(self) -> None:
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        module = _load_module()
        config = _tiny_config()
        model = EntityTokenTransformerPolicy(config).eval()
        shim = module.build_exportable_module(model)

        with tempfile.TemporaryDirectory() as tmp:
            traced = module.export_torchscript(
                shim, module.make_random_inputs(config, 2, seed=3), Path(tmp) / "tiny_ts.pt"
            )

        probes = []
        ones_inputs = list(module.make_random_inputs(config, 3, seed=17))
        ones_inputs[3] = torch.ones_like(ones_inputs[3])
        probes.append(tuple(ones_inputs))
        sparse_inputs = list(module.make_random_inputs(config, 3, seed=23))
        sparse = torch.rand(sparse_inputs[3].shape) > 0.8
        sparse[..., 0] = True  # token-0 invariant mirrors the real encoder
        probes.append(tuple(sparse_inputs[:3]) + (sparse,) + tuple(sparse_inputs[4:]))
        for probe in probes:
            with torch.no_grad():
                eager_out = shim(*probe)
                traced_out = traced(*probe)
            for eager_t, traced_t in zip(eager_out, traced_out):
                self.assertEqual((eager_t - traced_t).abs().max().item(), 0.0)
                self.assertFalse(torch.isnan(traced_t).any().item())


if __name__ == "__main__":
    unittest.main()