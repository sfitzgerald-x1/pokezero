"""Parity gate for in-crate TorchScript leaf evaluation (track D).

The machine-checkable claim: feeding IDENTICAL pre-encoded observations to
(a) the venv's torch running a TorchScript artifact and (b) the crate's
TorchScriptLeafEval (tch-rs) through the `NativeLeafModel` debug entrypoint
produces the same outputs — expected bit-exact (same libtorch runtime under
both, per scripts/build_search_crate_model.sh), tolerated to 1e-6 fp32.

Skips cleanly unless the crate was built with the `model` feature
(scripts/build_search_crate_model.sh) and torch is importable.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - exercised only when the native crate is built
    import pokezero_search
except ImportError:  # pragma: no cover
    pokezero_search = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
PARITY_TOLERANCE = 1e-6

_crate_has_model = bool(
    pokezero_search is not None and getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
)


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_model", REPO / "scripts" / "export_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config():
    from pokezero.neural_policy import TransformerPolicyConfig

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


def _flatten_inputs(inputs):
    categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask = inputs
    return (
        categorical_ids.flatten().tolist(),
        numeric_features.flatten().tolist(),
        token_type_ids.flatten().tolist(),
        attention_mask.flatten().tolist(),
        history_mask.flatten().tolist(),
    )


@unittest.skipUnless(_crate_has_model, "pokezero_search built without the model feature")
@unittest.skipIf(torch is None, "requires torch")
class CrateTorchScriptParityTest(unittest.TestCase):
    """Crate (tch-rs) vs venv torch on the SAME TorchScript artifact."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.export = _load_export_module()
        cls.config = _tiny_config()
        from pokezero.neural_policy import EntityTokenTransformerPolicy

        torch.manual_seed(31)
        model = EntityTokenTransformerPolicy(cls.config).eval()
        cls.shim = cls.export.build_exportable_module(model)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.ts_path = Path(cls.tmpdir.name) / "tiny_ts.pt"
        cls.export.export_torchscript(
            cls.shim, cls.export.make_random_inputs(cls.config, cls.export.TRACE_BATCH, seed=7), cls.ts_path
        )
        cls.native = pokezero_search.NativeLeafModel(
            str(cls.ts_path),
            device="cpu",
            window=cls.config.window_size,
            tokens=cls.config.token_count,
            categorical_features=cls.config.categorical_feature_count,
            numeric_features=cls.config.numeric_feature_count,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def _parity_at_batch(self, batch: int, seed: int) -> float:
        inputs = self.export.make_random_inputs(self.config, batch, seed=seed)
        loaded = torch.jit.load(str(self.ts_path))
        with torch.no_grad():
            py_logits, py_value, _ = loaded(*inputs)
        values_tanh, logits_flat, priors_flat, action_count = self.native.eval_obs_flat(
            batch, *_flatten_inputs(inputs)
        )
        self.assertEqual(action_count, int(py_logits.shape[-1]))
        crate_logits = torch.tensor(logits_flat).reshape(py_logits.shape)
        crate_value = torch.tensor(values_tanh).reshape(py_value.shape)
        diff = max(
            (crate_logits - py_logits).abs().max().item(),
            (crate_value - py_value).abs().max().item(),
        )
        # Priors: crate masked-softmax (mask omitted -> plain softmax) vs torch.
        py_priors = torch.softmax(py_logits, dim=-1)
        crate_priors = torch.tensor(priors_flat).reshape(py_priors.shape)
        prior_diff = (crate_priors - py_priors).abs().max().item()
        return max(diff, prior_diff)

    def test_parity_batches(self) -> None:
        worst = 0.0
        for batch, seed in ((1, 101), (4, 202), (64, 303)):
            diff = self._parity_at_batch(batch, seed)
            worst = max(worst, diff)
            self.assertLessEqual(
                diff,
                PARITY_TOLERANCE,
                f"crate/torch divergence {diff:.3e} at batch {batch} exceeds {PARITY_TOLERANCE}",
            )
        print(f"\n[parity] crate vs venv torch, max abs diff across batches: {worst:.3e}")

    def test_legal_mask_priors_match_masked_softmax(self) -> None:
        batch = 4
        inputs = self.export.make_random_inputs(self.config, batch, seed=404)
        loaded = torch.jit.load(str(self.ts_path))
        with torch.no_grad():
            py_logits, _, _ = loaded(*inputs)
        action_count = int(py_logits.shape[-1])
        gen = torch.Generator().manual_seed(505)
        legal = torch.rand((batch, action_count), generator=gen) > 0.4
        legal[:, 0] = True  # keep every row at least one legal action
        _, _, priors_flat, _ = self.native.eval_obs_flat(
            batch, *_flatten_inputs(inputs), legal_mask=legal.flatten().tolist()
        )
        crate_priors = torch.tensor(priors_flat).reshape(batch, action_count)
        py_priors = torch.softmax(
            py_logits.masked_fill(~legal, float("-inf")), dim=-1
        )
        self.assertLessEqual((crate_priors - py_priors).abs().max().item(), PARITY_TOLERANCE)
        # Illegal actions carry exactly zero prior mass.
        self.assertEqual(crate_priors[~legal].abs().max().item(), 0.0)

    def test_full_size_artifact_parity_when_present(self) -> None:
        """Same gate against the real-size exported artifact, if one exists."""
        ts_path = REPO / "exports" / "model_ts.pt"
        manifest_path = REPO / "exports" / "export_manifest.json"
        if not ts_path.exists() or not manifest_path.exists():
            self.skipTest("no exports/model_ts.pt artifact on this checkout")
        shapes = json.loads(manifest_path.read_text())["input_shapes"]["categorical_ids"]
        _, window, tokens, cat = shapes
        num = json.loads(manifest_path.read_text())["input_shapes"]["numeric_features"][-1]
        native = pokezero_search.NativeLeafModel(
            str(ts_path),
            device="cpu",
            window=window,
            tokens=tokens,
            categorical_features=cat,
            numeric_features=num,
        )
        from pokezero.neural_policy import TransformerPolicyConfig

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("a", "b"),
            category_oov_buckets=1,
            categorical_feature_count=cat,
            numeric_feature_count=num,
            embedding_dim=16,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
        )
        batch = 8
        inputs = self.export.make_random_inputs(config, batch, seed=606)
        loaded = torch.jit.load(str(ts_path))
        with torch.no_grad():
            py_logits, py_value, _ = loaded(*inputs)
        values_tanh, logits_flat, _, _ = native.eval_obs_flat(batch, *_flatten_inputs(inputs))
        diff = max(
            (torch.tensor(logits_flat).reshape(py_logits.shape) - py_logits).abs().max().item(),
            (torch.tensor(values_tanh).reshape(py_value.shape) - py_value).abs().max().item(),
        )
        print(f"\n[parity] full-size artifact, max abs diff at batch {batch}: {diff:.3e}")
        self.assertLessEqual(diff, PARITY_TOLERANCE)


@unittest.skipUnless(_crate_has_model, "pokezero_search built without the model feature")
@unittest.skipIf(torch is None, "requires torch")
class BatchedModelSearchTest(unittest.TestCase):
    """Mechanics smoke for the virtual-loss batched model-in-the-loop search."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.export = _load_export_module()
        cls.config = _tiny_config()
        from pokezero.neural_policy import EntityTokenTransformerPolicy

        torch.manual_seed(47)
        model = EntityTokenTransformerPolicy(cls.config).eval()
        shim = cls.export.build_exportable_module(model)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.ts_path = Path(cls.tmpdir.name) / "tiny_ts.pt"
        cls.export.export_torchscript(
            shim, cls.export.make_random_inputs(cls.config, cls.export.TRACE_BATCH, seed=7), cls.ts_path
        )
        cls.native = pokezero_search.NativeLeafModel(
            str(cls.ts_path),
            device="cpu",
            window=cls.config.window_size,
            tokens=cls.config.token_count,
            categorical_features=cls.config.categorical_feature_count,
            numeric_features=cls.config.numeric_feature_count,
        )
        cls.template = _flatten_inputs(cls.export.make_random_inputs(cls.config, 1, seed=808))
        try:
            from pokezero.poke_engine_adapter import (
                build_poke_engine_state,
                minimal_gen3_fixture,
            )

            cls.state_str = build_poke_engine_state(minimal_gen3_fixture()).to_string()
        except Exception as exc:  # engine binding missing/broken
            cls.state_str = None
            cls.fixture_error = exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def setUp(self) -> None:
        if self.state_str is None:
            self.skipTest(f"poke_engine fixture unavailable: {self.fixture_error}")

    def _search(self, iterations: int, batch_size: int, seed: int = 0) -> dict:
        report = self.native.search_batched(
            self.state_str, iterations, batch_size, *self.template, seed=seed
        )
        return json.loads(report)

    def test_visit_conservation_and_shape(self) -> None:
        iterations, batch_size = 96, 16
        report = self._search(iterations, batch_size)
        self.assertEqual(report["iterations"], iterations)
        self.assertEqual(report["evaluator"], "torchscript")
        self.assertEqual(report["batch_size"], batch_size)
        self.assertEqual(report["rounds"], iterations // batch_size)
        self.assertEqual(
            report["model_evals"] + report["terminal_leaves"], iterations
        )
        for side in ("side_one", "side_two"):
            visits = sum(entry["visits"] for entry in report[side])
            self.assertEqual(visits, iterations, f"{side} visit conservation")
            for entry in report[side]:
                self.assertGreaterEqual(entry["q"], 0.0)
                self.assertLessEqual(entry["q"], 1.0)

    def test_seed_determinism(self) -> None:
        first = self._search(64, 8, seed=9)
        second = self._search(64, 8, seed=9)
        self.assertEqual(first["side_one"], second["side_one"])
        self.assertEqual(first["side_two"], second["side_two"])

    def test_batch_one_matches_sequential_regime(self) -> None:
        # batch=1 is sequential PUCT with a model leaf: the virtual loss is
        # applied and immediately replaced within the same round.
        report = self._search(32, 1)
        self.assertEqual(report["rounds"], 32)
        self.assertEqual(report["model_evals"] + report["terminal_leaves"], 32)

    def test_rejects_zero_iterations(self) -> None:
        with self.assertRaises(ValueError):
            self.native.search_batched(self.state_str, 0, 8, *self.template)


if __name__ == "__main__":
    unittest.main()
