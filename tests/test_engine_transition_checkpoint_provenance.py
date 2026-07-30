"""Checkpoint provenance pins for resumable certification shards."""

from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
