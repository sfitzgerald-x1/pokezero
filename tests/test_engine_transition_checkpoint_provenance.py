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

    def test_resume_terminates_complete_final_record_before_appending(self) -> None:
        first = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        second = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1001}
        first_bytes = json.dumps(first, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            checkpoint.write_bytes(first_bytes)
            self.assertEqual(runner.load_checkpoint(checkpoint), [first])
            self.assertEqual(checkpoint.read_bytes(), first_bytes + b"\n")
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

    def test_resume_rejects_parseable_non_records_without_rewriting(self) -> None:
        valid = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        for malformed in ({}, 42, []):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "checkpoint.jsonl"
                original = (
                    json.dumps(valid, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                    + json.dumps(malformed, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
                checkpoint.write_bytes(original)
                with self.assertRaisesRegex(ValueError, "invalid checkpoint record at line 2"):
                    runner.load_checkpoint(checkpoint)
                self.assertEqual(checkpoint.read_bytes(), original)

    def test_resume_rejects_parseable_mid_file_non_record(self) -> None:
        first = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        second = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1001}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            original = (
                json.dumps(first, separators=(",", ":")).encode("utf-8")
                + b"\n{}\n"
                + json.dumps(second, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            checkpoint.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "invalid checkpoint record at line 2"):
                runner.load_checkpoint(checkpoint)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_checkpoint_binding_includes_divergence_classes(self) -> None:
        record = {
            "counters": {
                "boundaries_full_round": 1,
                "boundaries_measured": 1,
                "divergence_class:branch_event": 7,
                "engine_error": 0,
                "transition:diverged": 7,
                "transition:matched": 0,
            },
            "repros": [],
        }
        report = runner.build_report(
            [record],
            elapsed=1.0,
            approximate_sleep=False,
            matcher="strict",
            keep_repro=0,
        )
        self.assertEqual(report["divergence_classes"], {"branch_event": 7})
        report["divergence_classes"] = {}
        self.assertIn(
            "report divergence_classes does not match the checkpoint aggregate",
            runner.checkpoint_report_binding_failures([record], report),
        )


if __name__ == "__main__":
    unittest.main()
