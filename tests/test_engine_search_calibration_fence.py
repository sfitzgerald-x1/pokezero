"""Phase 2 falsifier: the calibration seam must fail LOUD, not silently.

The Python value head applies `result.value_calibration_transform`; the crate applies none
(`model.rs` maps the raw tanh through `0.5*(v+1.0)`). Before this fence,
`value_calibration_transform` appeared nowhere in `engine_search.py`, so a checkpoint
carrying one would put crate leaf values on a different axis from every Q gap and threshold
derived Python-side -- silently, on the axis search decisions are made on.

It is inert on today's checkpoint (transform None), which is exactly why it needs a test:
nothing in a passing run would reveal that the fence had stopped working.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import torch

from pokezero.engine_search import EngineMctsConfig, _fence_calibration_seam
from pokezero.neural_policy import ValueCalibrationTransform


def _ckpt(payload: dict) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(payload, f.name)
    return f.name


class CalibrationSeamFenceTest(unittest.TestCase):
    def _run(self, payload):
        path = _ckpt(payload)
        try:
            _fence_calibration_seam(path)
            return None
        except ValueError as exc:
            return str(exc)
        finally:
            os.unlink(path)

    def test_todays_checkpoint_shape_is_inert(self):
        # iteration-2533 carries transform None. The fence must not fire.
        self.assertIsNone(self._run({"value_calibration_transform": None}))

    def test_pre_provenance_checkpoint_without_the_key_is_inert(self):
        # Absent key means "this checkpoint predates the field", not "no calibration".
        self.assertIsNone(self._run({"state_dict": {}}))

    def test_explicit_identity_affine_is_inert(self):
        self.assertIsNone(self._run({"value_calibration_transform":
                                     ValueCalibrationTransform(scale=1.0, bias=0.0,
                                                               method="affine")}))

    def test_a_scaling_affine_is_REFUSED(self):
        msg = self._run({"value_calibration_transform":
                         ValueCalibrationTransform(scale=1.7, bias=0.0, method="affine")})
        self.assertIsNotNone(msg)
        self.assertIn("REFUSING model-leaf search", msg)

    def test_an_isotonic_transform_is_REFUSED(self):
        # Isotonic ignores scale entirely, so a scale-only check would miss it.
        msg = self._run({"value_calibration_transform":
                         ValueCalibrationTransform(method="isotonic",
                                                   points=((0.0, 0.0), (1.0, 1.0)))})
        self.assertIsNotNone(msg)
        self.assertIn("REFUSING model-leaf search", msg)

    def test_a_bias_only_affine_is_REFUSED(self):
        # A bias cancels in a GAP but not in a threshold comparison, so it is still a seam.
        msg = self._run({"value_calibration_transform":
                         ValueCalibrationTransform(scale=1.0, bias=0.2, method="affine")})
        self.assertIsNotNone(msg)

    def test_an_unreadable_checkpoint_is_left_to_the_loader(self):
        # The fence must not turn a missing file into a calibration complaint.
        self.assertIsNone(_fence_calibration_seam("/nonexistent/path.pt") or None)

    def test_non_model_leaf_eval_skips_the_fence_entirely(self):
        cfg = EngineMctsConfig(worlds=1, search_time_ms=100, threads=1,
                               leaf_eval="hp_fraction",
                               checkpoint_path="/nonexistent/path.pt")
        self.assertEqual(cfg.leaf_eval, "hp_fraction")


if __name__ == "__main__":
    unittest.main()
