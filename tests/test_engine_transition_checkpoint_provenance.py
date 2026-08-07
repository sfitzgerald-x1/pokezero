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
            self.assertEqual(
                runner.load_checkpoint(checkpoint, repair_torn_tail=True),
                [first],
            )
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
            self.assertEqual(
                runner.load_checkpoint(checkpoint, repair_torn_tail=True),
                [first],
            )
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

    def test_read_only_load_preserves_complete_record_without_newline(self) -> None:
        record = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        original = json.dumps(record, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            checkpoint.write_bytes(original)
            self.assertEqual(runner.load_checkpoint(checkpoint), [record])
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_read_only_load_rejects_torn_tail_without_rewriting(self) -> None:
        record = {"schema": runner.CHECKPOINT_SCHEMA, "seed": 1000}
        original = (
            json.dumps(record, separators=(",", ":")).encode("utf-8")
            + b"\n"
            + b'{"schema":'
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            checkpoint.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "refusing to repair checkpoint evidence"):
                runner.load_checkpoint(checkpoint)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_checkpoint_binding_includes_divergence_classes(self) -> None:
        record = {
            "build_check": "gated",
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
        report = runner.build_report(
            [record],
            elapsed=1.0,
            approximate_sleep=False,
            matcher="strict",
            keep_repro=0,
        )
        report["build_check"] = "NOT-GATED: skipped"
        report["acceptance_eligible"] = False
        failures = runner.checkpoint_report_binding_failures([record], report)
        self.assertIn(
            "report build_check does not match the checkpoint aggregate",
            failures,
        )
        self.assertIn(
            "report acceptance_eligible does not match the checkpoint aggregate",
            failures,
        )


class RollPathProvenanceTests(unittest.TestCase):
    """C116 Phase 2: which roll path a shard ran on, and who may merge with whom.

    One BUILD serves the collapsed cascade and the enumerated oracle, selected at
    runtime. So ``source_commit``, ``engine_fingerprint`` and ``image_commit`` are
    identical between the two configurations by design, and this field is the only
    thing that can tell a collapsed shard from an enumerated one.
    """

    provenance = {
        "source_commit": "a" * 40,
        "engine_fingerprint": "b" * 64,
        "image_commit": "c" * 40,
    }

    @staticmethod
    def _record(enumerate_rolls: bool | None) -> dict:
        provenance = {
            "source_commit": "a" * 40,
            "engine_fingerprint": "b" * 64,
            "image_commit": "c" * 40,
        }
        if enumerate_rolls is not None:
            provenance["enumerate_rolls"] = enumerate_rolls
        return {"provenance": provenance}

    def test_import_does_not_select_a_roll_path(self) -> None:
        """Importing the differential must leave the process configuration alone.

        The module used to run ``os.environ.setdefault`` at import, which made every
        importer -- this test module included -- an enabler for its whole process and
        every child of it. The behavioural proof lives in
        ``tests/test_roll_enumeration_scope.py``; this is the cheap unit-level echo.
        """
        self.assertFalse(runner.ENUMERATE_ROLLS)

    def test_merge_refuses_shards_from_different_roll_paths(self) -> None:
        with self.assertRaises(ValueError) as caught:
            runner._merged_roll_path([self._record(False), self._record(True)])
        self.assertIn("DIFFERENT roll paths", str(caught.exception))

    def test_merge_derives_the_roll_path_from_the_records(self) -> None:
        self.assertTrue(
            runner._merged_roll_path([self._record(True), self._record(True)])
        )
        self.assertFalse(
            runner._merged_roll_path([self._record(False), self._record(False)])
        )

    def test_a_record_predating_the_field_counts_as_collapsed(self) -> None:
        """Absent is collapsed as a matter of HISTORY, not as a lenient default.

        Records written before the field existed came out of builds with no enumerated
        path compiled into them at all.
        """
        self.assertFalse(runner._merged_roll_path([self._record(None)]))
        with self.assertRaises(ValueError):
            runner._merged_roll_path([self._record(None), self._record(True)])

    def test_source_tree_is_recorded_but_does_not_gate_a_resume(self) -> None:
        """The field's comment says "recorded, not enforced". This is that, enforced.

        ``_resume_provenance_failures`` compares provenance with ``!=``, so adding a key
        to the dict silently added it to resume IDENTITY. Review demonstrated the
        consequence: sweep 200 games on a clean tree, touch any tracked file, resume ->
        exit 1, "checkpoint record 1 provenance differs from this resume". A crash-safe
        sweep that cannot survive an unrelated edit is not crash-safe, and it was the
        exact recorded-vs-enforced contradiction this branch fixed two docstrings for.
        """
        clean = dict(self.provenance, source_tree="clean")
        dirty = dict(self.provenance, source_tree="dirty")
        self.assertEqual(runner._resume_provenance_failures([{"provenance": clean}], dirty), [])
        self.assertEqual(runner._resume_provenance_failures([{"provenance": dirty}], clean), [])

    def test_every_other_provenance_field_still_gates_a_resume(self) -> None:
        """The negative control: excluding one key must not have excluded the rest."""
        base = dict(self.provenance, source_tree="clean", enumerate_rolls=False)
        for field, other in (
            ("source_commit", "f" * 40),
            ("engine_fingerprint", "e" * 64),
            ("image_commit", "d" * 40),
            ("enumerate_rolls", True),
        ):
            with self.subTest(field=field):
                mixed = dict(base)
                mixed[field] = other
                self.assertEqual(
                    runner._resume_provenance_failures([{"provenance": mixed}], base),
                    ["checkpoint record 1 provenance differs from this resume"],
                    f"{field} stopped being part of resume identity",
                )

    def test_source_tree_reports_the_checkout_state(self) -> None:
        """It has to actually measure something, not just always say "clean"."""
        self.assertIn(runner._source_tree_state(), {"clean", "dirty", "unknown"})

    def test_a_clean_dirty_flip_stays_visible_in_the_report(self) -> None:
        """Excluded from identity, NOT dropped: the artifact still shows both states.

        Otherwise "recorded, not enforced" would have quietly become "not recorded".
        """
        records = []
        for index, state in enumerate(("clean", "dirty")):
            records.append({
                "schema": runner.CHECKPOINT_SCHEMA,
                "build_check": "gated",
                "seed": 1000 + index,
                "seconds": 1.0,
                "counters": {"boundaries_measured": 1, "transition:matched": 1},
                "repros": [],
                "provenance": dict(self.provenance, enumerate_rolls=False, source_tree=state),
            })
        report = runner.build_report(
            records, elapsed=None, approximate_sleep=None, matcher=None, keep_repro=0
        )
        distinct = report["checkpoint_provenance"]["distinct"]
        self.assertEqual(len(distinct), 2, distinct)
        self.assertTrue(any('"source_tree": "dirty"' in blob for blob in distinct))
        self.assertTrue(any('"source_tree": "clean"' in blob for blob in distinct))

    def test_the_merged_report_labels_itself_from_the_shards(self) -> None:
        """Not from the merging PROCESS, which ran no games and made no engine call."""
        record = {
            "schema": runner.CHECKPOINT_SCHEMA,
            "build_check": "gated",
            "seed": 1000,
            "seconds": 1.0,
            "counters": {"boundaries_measured": 1, "transition:matched": 1},
            "repros": [],
            "provenance": self._record(True)["provenance"],
        }
        merged = runner.build_report(
            [record],
            elapsed=None,
            approximate_sleep=None,
            matcher=None,
            keep_repro=0,
            enumerate_rolls=runner._merged_roll_path([record]),
        )
        self.assertTrue(merged["enumerate_rolls"])
        self.assertFalse(
            runner.ENUMERATE_ROLLS,
            "the merging process itself stayed on the collapsed default, so this label "
            "could only have come from the records",
        )


if __name__ == "__main__":
    unittest.main()
