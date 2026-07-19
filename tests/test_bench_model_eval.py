from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from pokezero.neural_policy import TransformerPolicyConfig, torch_available

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_model_eval.py"


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location("bench_model_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass field-type resolution looks the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_config() -> TransformerPolicyConfig:
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
class PadObsBatchTest(unittest.TestCase):
    def _inputs(self, module, batch: int):
        return module.make_random_inputs(_tiny_config(), batch, seed=7)

    def test_shapes_and_filler_rows(self) -> None:
        module = _load_module()
        inputs = self._inputs(module, 3)
        padded, real_rows = module.pad_obs_batch(inputs, 8)
        self.assertEqual(real_rows, 3)
        for original, grown in zip(inputs, padded):
            self.assertEqual(int(grown.shape[0]), 8)
            self.assertEqual(grown.shape[1:], original.shape[1:])
            self.assertEqual(grown.dtype, original.dtype)
            # Real rows preserved bit-for-bit; padding repeats the last row.
            self.assertTrue(bool((grown[:3] == original).all()))
            for row in range(3, 8):
                self.assertTrue(bool((grown[row] == original[-1]).all()))

    def test_noop_at_target_size(self) -> None:
        module = _load_module()
        inputs = self._inputs(module, 4)
        padded, real_rows = module.pad_obs_batch(inputs, 4)
        self.assertEqual(real_rows, 4)
        for original, same in zip(inputs, padded):
            self.assertIs(original, same)

    def test_rejects_oversized_batch(self) -> None:
        module = _load_module()
        inputs = self._inputs(module, 5)
        with self.assertRaises(ValueError):
            module.pad_obs_batch(inputs, 4)

    def test_padded_forward_matches_direct_forward(self) -> None:
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        module = _load_module()
        config = _tiny_config()
        torch.manual_seed(3)
        model = EntityTokenTransformerPolicy(config).eval()
        shim = module.build_exportable_module(model)
        inputs = self._inputs(module, 3)
        padded, real_rows = module.pad_obs_batch(inputs, 8)
        with torch.no_grad():
            direct = shim(*inputs)
            via_pad = shim(*padded)
        for reference, candidate in zip(direct, via_pad):
            torch.testing.assert_close(candidate[:real_rows], reference)


if __name__ == "__main__":
    unittest.main()
