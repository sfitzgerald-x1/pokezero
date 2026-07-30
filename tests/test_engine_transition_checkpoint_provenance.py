"""Checkpoint provenance pins for resumable certification shards."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "engine_transition_differential_under_test",
    ROOT / "scripts" / "engine_transition_differential.py",
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class CheckpointProvenanceTests(unittest.TestCase):
    provenance = {
        "source_commit": "a" * 40,
        "engine_fingerprint": "b" * 64,
        "image_commit": "c" * 40,
    }

    def test_completed_record_carries_resume_identity(self) -> None:
        record = runner.checkpoint_record(
            seed=1000,
            counts={},
            repros=[],
            seconds=0.1,
            build_check="gated",
            provenance=self.provenance,
        )
        self.assertEqual(record["provenance"], self.provenance)

    def test_resume_rejects_a_mixed_engine_identity(self) -> None:
        mixed = dict(self.provenance)
        mixed["engine_fingerprint"] = "d" * 64
        failures = runner._resume_provenance_failures(
            [{"provenance": mixed}], self.provenance
        )
        self.assertEqual(
            failures,
            ["checkpoint record 1 provenance differs from this resume"],
        )

    def test_resume_repairs_torn_final_line_before_appending(self) -> None:
        first = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        second = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1001}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            checkpoint.write_bytes(
                json.dumps(first, separators=(",", ":")).encode("utf-8") + b"\n" + b'{"schema":'
            )
            self.assertEqual(runner.load_checkpoint(checkpoint), [first])
            self.assertEqual(
                checkpoint.read_bytes(),
                json.dumps(first, separators=(",", ":")).encode("utf-8") + b"\n",
            )
            with checkpoint.open("a", encoding="utf-8") as handle:
                runner.append_checkpoint(handle, second)
            self.assertEqual(runner.load_checkpoint(checkpoint), [first, second])

    def test_resume_rejects_mid_file_corruption_without_truncating(self) -> None:
        first = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        second = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1001}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            original = (
                json.dumps(first, separators=(",", ":")).encode("utf-8")
                + b"\n{bad-json}\n"
                + json.dumps(second, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            checkpoint.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "unparseable line 2"):
                runner.load_checkpoint(checkpoint)
            self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
