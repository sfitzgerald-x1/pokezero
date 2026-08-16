"""The Phase 3 V1 conversion path: rewrite a checkpoint's value-head width.

This exists because PR #1263's approach did not work — `--value-head-hidden` on `train` is a
no-op on a warm start, since the config is derived wholly from the checkpoint. Rewriting the
config stamp and letting #1262's rename tolerance reinitialise the head is the path that does.

Every refusal below was a real failure mode in this series, and one of them is a collision
between two guards I wrote: stripping the stale head tensors makes a checkpoint
indistinguishable from a truncated write, which the load path correctly refuses.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pokezero.neural_policy import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    EntityTokenTransformerPolicy,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    save_transformer_checkpoint,
    torch_available,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_value_head.py"


def _cfg(hidden=None):
    return TransformerPolicyConfig.compact_category(
        observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_2,
        category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id="conv",
        window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
        numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
        attention_heads=4, feedforward_dim=32, dropout=0.0, value_head_hidden=hidden)


class ConvertValueHeadTest(unittest.TestCase):
    def setUp(self):
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.incumbent = self._write("inc.pt", None)
        self.widened = self._write("wide.pt", 32)

    def _write(self, name, hidden):
        cfg = _cfg(hidden)
        result = TransformerTrainingResult(
            model_config=cfg,
            training_config=TransformerTrainingConfig(objective="ppo", window_size=2),
            epochs=(TransformerEpochMetrics(epoch=1, examples=10, loss=0.5, policy_loss=-0.1,
                                            policy_accuracy=0.4, value_loss=0.25),))
        path = self.dir / name
        save_transformer_checkpoint(path, EntityTokenTransformerPolicy(cfg), result=result)
        return path

    def _run(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True)

    def test_widening_produces_a_checkpoint_that_reloads_as_claimed(self):
        out = self.dir / "v1.pt"
        proc = self._run("--checkpoint", str(self.incumbent), "--output", str(out),
                         "--value-head-hidden", "32")
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        self.assertIn("verified: reloads as Sequential", proc.stdout)
        self.assertIn("UNTRAINED", proc.stdout)
        self.assertTrue(out.exists())
        # And independently, not just from the tool's own say-so.
        from pokezero.neural_policy import load_transformer_checkpoint

        model, result = load_transformer_checkpoint(out)
        self.assertEqual(type(model.value_head).__name__, "Sequential")
        self.assertEqual(result.model_config.value_head_hidden, 32)

    def test_the_trunk_is_carried_over_byte_identically(self):
        """If the trunk moved, this would not be a value-head arm at all."""
        import torch

        out = self.dir / "v1.pt"
        self.assertEqual(
            self._run("--checkpoint", str(self.incumbent), "--output", str(out),
                      "--value-head-hidden", "32").returncode, 0)
        before = torch.load(self.incumbent, map_location="cpu", weights_only=False)["state_dict"]
        after = torch.load(out, map_location="cpu", weights_only=False)["state_dict"]
        checked = 0
        for name, tensor in before.items():
            if name.startswith("value_head."):
                continue
            self.assertTrue(torch.equal(after[name], tensor), f"{name} moved")
            checked += 1
        self.assertGreater(checked, 5)

    def test_narrowing_is_refused_without_force(self):
        proc = self._run("--checkpoint", str(self.widened), "--output", str(self.dir / "n.pt"),
                         "--value-head-hidden", "0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("NARROWS", proc.stdout + proc.stderr)

    def test_overwriting_the_input_is_refused(self):
        proc = self._run("--checkpoint", str(self.incumbent), "--output", str(self.incumbent),
                         "--value-head-hidden", "32")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("same file", proc.stdout + proc.stderr)

    def test_a_degenerate_width_is_refused(self):
        proc = self._run("--checkpoint", str(self.incumbent), "--output", str(self.dir / "d.pt"),
                         "--value-head-hidden", "1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("degenerate", proc.stdout + proc.stderr)

    def test_a_no_op_conversion_is_refused(self):
        proc = self._run("--checkpoint", str(self.widened), "--output", str(self.dir / "s.pt"),
                         "--value-head-hidden", "32")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("NOTHING TO DO", proc.stdout + proc.stderr)

    def test_the_source_checkpoint_is_never_modified(self):
        import hashlib

        before = hashlib.sha256(self.incumbent.read_bytes()).hexdigest()
        self._run("--checkpoint", str(self.incumbent), "--output", str(self.dir / "v1.pt"),
                  "--value-head-hidden", "32")
        self.assertEqual(hashlib.sha256(self.incumbent.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
