"""The Phase 2 calibration seam fence.

The crate applies NO value calibration (`model.rs` maps the raw tanh through `0.5*(v+1.0)`),
so a checkpoint carrying a `value_calibration_transform` has a Python value axis and a crate
value axis that differ with nothing reporting it. This suite fences that seam.

**Every checkpoint here is written by `save_transformer_checkpoint`.** The first version of
this fence passed 8 tests that `torch.save`d a live `ValueCalibrationTransform` instance -- a
shape production never writes -- and so never exercised the dict that `to_dict()` actually
persists. Against a real artifact its `getattr` reads returned `None` for every field and the
identity branch was unreachable. Building the fixture through the production writer is the
whole point; `test_production_persists_a_dict_not_a_dataclass` pins that premise directly.
"""
from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pokezero.engine_search import (
    EngineMctsConfig,
    EngineMctsPolicy,
    _fence_calibration_seam,
    _is_identity_calibration,
)
from pokezero.neural_policy import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    EntityTokenTransformerPolicy,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    ValueCalibrationTransform,
    save_transformer_checkpoint,
    torch_available,
)


def _write_checkpoint(directory: Path, transform: ValueCalibrationTransform | None) -> Path:
    """Write a real checkpoint through the production writer."""
    model_config = TransformerPolicyConfig.compact_category(
        observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_2,
        category_vocab=tuple(range(1, 17)),
        category_oov_buckets=4,
        policy_id="calibration-fence",
        window_size=2,
        token_type_vocab_size=8,
        categorical_feature_count=1,
        numeric_feature_count=1,
        embedding_dim=16,
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    result = TransformerTrainingResult(
        model_config=model_config,
        training_config=TransformerTrainingConfig(objective="ppo", window_size=2),
        value_calibration_transform=transform,
        epochs=(
            TransformerEpochMetrics(
                epoch=1, examples=10, loss=0.5, policy_loss=-0.1,
                policy_accuracy=0.4, value_loss=0.25,
            ),
        ),
    )
    path = directory / "transformer.pt"
    save_transformer_checkpoint(path, EntityTokenTransformerPolicy(model_config), result=result)
    return path


def _load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


class ProductionShapeTests(unittest.TestCase):
    """The premise the first fence got wrong, pinned so it cannot silently change."""

    def test_production_persists_a_dict_not_a_dataclass(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(
                Path(tmp), ValueCalibrationTransform(scale=1.5, bias=-0.2)
            )
            stored = _load(path)["value_calibration_transform"]
        self.assertIsInstance(
            stored, dict,
            "save_transformer_checkpoint persists to_dict(). A fence that reads this with "
            "getattr() sees None for every field.",
        )
        # And the reason the old identity branch was unreachable, stated as an assertion.
        self.assertIsNone(getattr(stored, "method", None))
        self.assertEqual(stored["method"], "affine")


class IdentityDetectionTests(unittest.TestCase):
    def test_default_dataclass_is_identity(self) -> None:
        self.assertTrue(_is_identity_calibration(ValueCalibrationTransform()))

    def test_default_dict_is_identity(self) -> None:
        self.assertTrue(_is_identity_calibration(ValueCalibrationTransform().to_dict()))

    def test_affine_is_not_identity(self) -> None:
        self.assertFalse(
            _is_identity_calibration(
                ValueCalibrationTransform(scale=1.5, bias=-0.2).to_dict()
            )
        )

    def test_clip_narrowing_alone_is_not_identity(self) -> None:
        """The hole in the 3-of-6-fields version: scale 1, bias 0, but 0.9 -> 0.2."""
        narrowed = ValueCalibrationTransform(clip_min=-0.2, clip_max=0.2)
        self.assertEqual((narrowed.scale, narrowed.bias), (1.0, 0.0))
        self.assertAlmostEqual(narrowed.apply(0.9), 0.2)
        self.assertFalse(_is_identity_calibration(narrowed.to_dict()))

    def test_isotonic_is_not_identity(self) -> None:
        iso = ValueCalibrationTransform(
            method="isotonic", points=((-1.0, -0.5), (1.0, 0.5))
        )
        self.assertFalse(_is_identity_calibration(iso.to_dict()))

    def test_unparseable_is_refused_not_waved_through(self) -> None:
        self.assertFalse(_is_identity_calibration({"scale": "not-a-number"}))
        self.assertFalse(_is_identity_calibration("a bare string"))


class FenceTests(unittest.TestCase):
    def test_none_transform_passes(self) -> None:
        _fence_calibration_seam({"value_calibration_transform": None}, "x")

    def test_absent_key_passes(self) -> None:
        """Pre-provenance checkpoints predate the field; absence is not a calibration."""
        _fence_calibration_seam({"schema_version": 1}, "x")

    def test_identity_passes(self) -> None:
        _fence_calibration_seam(
            {"value_calibration_transform": ValueCalibrationTransform().to_dict()}, "x"
        )

    def test_affine_refused_with_an_actionable_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _fence_calibration_seam(
                {
                    "value_calibration_transform": ValueCalibrationTransform(
                        scale=1.5, bias=-0.2
                    ).to_dict()
                },
                "checkpoint /tmp/c.pt",
            )
        message = str(ctx.exception)
        self.assertIn("REFUSING model-leaf search", message)
        self.assertIn("/tmp/c.pt", message)
        self.assertIn("hp_fraction_crate", message)  # names a way forward

    def test_refuses_on_a_real_production_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(
                Path(tmp), ValueCalibrationTransform(scale=1.5, bias=-0.2)
            )
            with self.assertRaises(ValueError):
                _fence_calibration_seam(_load(path), f"checkpoint {path}")

    def test_passes_on_a_real_uncalibrated_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(Path(tmp), None)
            _fence_calibration_seam(_load(path), f"checkpoint {path}")


class TheFenceIsActuallyWiredTests(unittest.TestCase):
    """The finding that killed the first revision: the helper existed and nothing called it.

    A unit test of a guard proves the guard's arithmetic, never its reachability. Both tests
    here fail if the call site is deleted or moved off the model-leaf path.
    """

    def _config(self, tmp: Path, checkpoint: Path, leaf_eval: str) -> EngineMctsConfig:
        model = tmp / "model.pt"
        model.write_bytes(b"not a real trace")
        tables = tmp / "tables.json"
        tables.write_text("{}", encoding="utf-8")
        return EngineMctsConfig(
            leaf_eval=leaf_eval,
            model_path=str(model),
            checkpoint_path=str(checkpoint),
            tables_path=str(tables),
        )

    def test_constructing_a_model_leaf_policy_refuses_a_calibrated_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            checkpoint = _write_checkpoint(
                tmp, ValueCalibrationTransform(scale=1.5, bias=-0.2)
            )
            with self.assertRaises(ValueError) as ctx:
                EngineMctsPolicy(
                    dex=None, set_source=None, module=object(),
                    config=self._config(tmp, checkpoint, "model"),
                )
        self.assertIn("REFUSING model-leaf search", str(ctx.exception))

    def test_the_call_site_is_inside_the_model_leaf_branch(self) -> None:
        """Structural, so a refactor that hoists the call out of the guard fails here."""
        source = inspect.getsource(EngineMctsPolicy.__init__)
        tree = ast.parse(inspect.cleandoc(source).replace("def __init__", "def __init__", 1))
        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = ast.dump(node.test)
            if "leaf_eval" not in test or "'model'" not in test.replace('"', "'"):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_fence_calibration_seam"
                ):
                    guarded.append(inner)
        self.assertEqual(
            len(guarded), 1,
            "_fence_calibration_seam must be called exactly once, inside the "
            "`leaf_eval == \"model\"` branch of EngineMctsPolicy.__init__. The first "
            "revision of this fence was never called at all.",
        )


if __name__ == "__main__":
    unittest.main()
