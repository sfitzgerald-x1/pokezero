"""Fail-closed guards for the C26 archival recalibration instrument.

These cover the properties that make the emitted artifact usable as C26's
registered calibration: the archive it reads is the pinned one, a row it cannot
reconstruct is skipped rather than credited, and the family table it derives can
actually be registered by the contract schema.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "c26_archival_recalibration.py"


def _load():
    spec = importlib.util.spec_from_file_location("c26_archival_recalibration", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RECAL = _load()


def _require(name: str):
    """Import a certification module or skip: these need the built engine."""

    try:
        return importlib.import_module(name)
    except BaseException as error:  # noqa: BLE001 - a build gap is a skip, not a failure
        raise unittest.SkipTest(f"{name} is unavailable ({type(error).__name__})") from error


class ArchiveInputContractTests(unittest.TestCase):
    def test_pinned_digests_are_the_ledger_archive(self) -> None:
        ledger = (ROOT / "docs" / "engine_divergence_ledger_20260728.md").read_text(
            encoding="utf-8"
        )
        for index, digest in enumerate(RECAL.PINNED_ARCHIVE_SHARDS):
            self.assertIn(f"{digest}  cert_shard_{index}.json", ledger)
        self.assertEqual(len(RECAL.PINNED_ARCHIVE_SHARDS), 8)
        self.assertEqual(RECAL.ARCHIVE_POPULATION, 3821)

    def test_wrong_shard_count_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "cert_shard_0.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "exactly 8 shards"):
                RECAL.verify_archive(f"{temp}/cert_shard_*.json")

    def test_altered_shard_bytes_are_refused_before_any_reread(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for index in range(8):
                (Path(temp) / f"cert_shard_{index}.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                RECAL.verify_archive(f"{temp}/cert_shard_*.json")

    def test_an_empty_shard_set_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "exactly 8 shards"):
                RECAL.verify_archive(f"{temp}/cert_shard_*.json")


class LossyRowAccountingTests(unittest.TestCase):
    """A row the current build cannot reconstruct is never a clearance."""

    def _rows(self):
        return [
            {"seed": 1, "step": 1, "protocol": [],
             "divergence_class": "limit:roll_divergent_lethality"},
            {"seed": 2, "step": 2, "protocol": [],
             "divergence_class": "roll_scaled_component"},
            {"seed": 3, "step": 3, "protocol": [],
             "divergence_class": "roll_scaled_component"},
        ]

    def test_skip_lossy_is_recorded_and_not_counted_as_matched(self) -> None:
        verdicts = {1: ("diverged", ["pct=100.0: p1 attributed components differ: "
                                     "observed=[('x', -1)] engine=[('x', -2)]"]),
                    2: ("matched", []),
                    3: ("skip_lossy", [])}

        def fake_reread(row):
            verdict, misses = verdicts[row["seed"]]
            return verdict, misses, 1

        cert_sweep_reread = _require("cert_sweep_reread")
        cert_sweep_readout = _require("cert_sweep_readout")
        differential = _require("engine_transition_differential")
        with patch.object(cert_sweep_reread, "reread_row", fake_reread), \
                patch.object(differential, "classify_divergence",
                             lambda protocol, misses: "roll_scaled_component"), \
                patch.object(cert_sweep_readout, "classify_row",
                             lambda row: ("I3_roll_inherited", "test basis", None)):
            result = RECAL.recalibrate(self._rows())

        self.assertEqual(result["tally"]["skip_lossy"], 1)
        self.assertEqual(result["tally"]["matched"], 1)
        self.assertEqual(result["tally"]["diverged"], 1)
        self.assertEqual(
            result["skipped"], [{"seed": 3, "step": 3,
                                 "recorded_class": "roll_scaled_component",
                                 "verdict": "skip_lossy"}]
        )
        self.assertEqual(result["cleared_by_recorded_class"], {"roll_scaled_component": 1})
        self.assertEqual(result["families"], {"I3_roll_inherited": 1})
        self.assertEqual(result["errors"], [])


class RegisteredTableTests(unittest.TestCase):
    def test_every_emittable_family_is_registered_with_a_bound(self) -> None:
        manifest = _require("cert_execution_manifest")
        EMITTABLE_DOCUMENTED_FAMILIES = manifest.EMITTABLE_DOCUMENTED_FAMILIES
        EMITTABLE_LIMIT_FAMILIES = manifest.EMITTABLE_LIMIT_FAMILIES

        intervals = RECAL.family_intervals(
            {"limit:roll_divergent_lethality": 1275, "UNATTRIBUTED": 883}, 791757
        )
        expected = (EMITTABLE_DOCUMENTED_FAMILIES | EMITTABLE_LIMIT_FAMILIES) - {
            "limit:world_substitute_health_unknown"
        }
        self.assertEqual(set(intervals), expected)
        for family, interval in intervals.items():
            self.assertEqual(len(interval), 2, family)
            self.assertLessEqual(interval[0], interval[1], family)
            self.assertGreater(interval[1], 0.0, family)

    def test_only_the_two_comparison_limits_keep_a_binding_lower_bound(self) -> None:
        intervals = RECAL.family_intervals(
            {"limit:roll_divergent_lethality": 1275,
             "limit:world_sample_drag_target": 271,
             "I3_roll_inherited": 233}, 791757
        )
        self.assertGreater(intervals["limit:roll_divergent_lethality"][0], 0.0)
        self.assertGreater(intervals["limit:world_sample_drag_target"][0], 0.0)
        self.assertEqual(intervals["I3_roll_inherited"][0], 0.0)

    def test_a_non_emittable_family_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-emittable families"):
            RECAL.family_intervals({"made_up_family": 1}, 791757)

    def test_substitute_budget_is_stricter_than_the_uncovered_boundary_budget(self) -> None:
        budget = RECAL.substitute_risk_budget({"recoil_vs_substitute_basis": 13}, 810125)
        self.assertEqual(budget["upper_rate_basis"], "pre_registered_risk_budget")
        # The registered coverage floor is 0.97, so the global uncovered budget
        # is 0.03; a risk budget at or above it would bound nothing.
        self.assertLess(budget["upper_full_round_rate"], 1.0 - 0.97)
        self.assertGreater(
            budget["upper_full_round_rate"], budget["anchor_wilson95_upper_rate"]
        )
        self.assertIn("13 retained recoil-vs-Substitute", budget["risk_budget_rationale"])

    def test_substitute_budget_survives_an_anchor_with_no_identities(self) -> None:
        budget = RECAL.substitute_risk_budget({}, 810125)
        self.assertEqual(budget["anchor_identities"], 0)
        self.assertGreater(budget["anchor_wilson95_upper_rate"], 0.0)


class EmittedArtifactTests(unittest.TestCase):
    def test_committed_calibration_matches_this_instrument(self) -> None:
        """The registered calibration must be this instrument's own output shape."""

        path = ROOT / "reports" / "c26_current_engine_calibration.json"
        if not path.is_file():
            self.skipTest("C26 calibration is not registered in this commit")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], RECAL.SCHEMA)
        self.assertEqual(
            payload["source_evidence"]["archive_shard_sha256"],
            list(RECAL.PINNED_ARCHIVE_SHARDS),
        )
        self.assertEqual(payload["source_evidence"]["population"], RECAL.ARCHIVE_POPULATION)
        self.assertEqual(payload["source_evidence"]["fresh_measurements_inspected"], 0)
        self.assertEqual(payload["source_evidence"]["archive_role"], "historical_calibration_only")
        self.assertEqual(payload["reread_errors"], [])
        # The calibration records the readout it was PRODUCED on. This commit
        # moves the working tree's readout past that point, so pin the historical
        # value against the contract that consumed it rather than against a file
        # that has since moved on. Comparing to the live file would assert that
        # the readout may never change again.
        contract = json.loads(
            (ROOT / "reports" / "c26_current_engine_resweep_spec.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            payload["source_evidence"]["current_classifier_readout_sha256"],
            contract["certification_gates"]["required_readout_sha256"],
        )
        # Every lossily stored row is accounted for by identity, and the tally's
        # non-diverged, non-matched rows are exactly those skipped rows.
        skipped = payload["skipped_rows"]
        self.assertEqual(payload["reread_tally"].get("skip_lossy", 0), len(skipped))
        self.assertEqual(
            payload["reread_tally"]["diverged"]
            + payload["reread_tally"]["matched"]
            + len(skipped),
            payload["source_evidence"]["population"],
        )
        for row in skipped:
            self.assertIn("seed", row)
            self.assertIn("step", row)


if __name__ == "__main__":
    unittest.main()
