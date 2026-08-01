"""Integrity checks for the current-engine C15 re-read harness."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_c15_script():
    script = REPO_ROOT / "scripts" / "c15_why_adjudication.py"
    spec = importlib.util.spec_from_file_location("c15_why_adjudication_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C15 = _load_c15_script()


def _targets() -> dict[tuple[int, int], str]:
    reports = REPO_ROOT / "reports"
    prediction = json.loads(
        (reports / "c15_why_magnitude_statgap_predictions.json").read_text()
    )
    remainder = json.loads(
        (reports / "c15_why_magnitude_statgap_remainder_predictions.json").read_text()
    )
    targets, _, _ = C15._target_plan(prediction, remainder)
    return targets


class C15WhyAdjudicationTest(unittest.TestCase):
    def test_historical_repair_lane_sources_are_hashed_and_disjoint(self) -> None:
        sources, provenance = C15._overlap_sources()

        self.assertEqual(
            {name: len(identities) for name, identities in sources.items()},
            {"patches_42_43": 158, "patch_44": 5, "bench_rest_world": 1},
        )
        self.assertEqual(
            provenance["patches_42_43"]["verification_path"],
            "reports/c15_engine_patch_verification.json",
        )
        C15._assert_no_active_lane_overlap(_targets(), sources)

    def test_overlap_fails_closed_before_replay(self) -> None:
        sources, _ = C15._overlap_sources()
        mutated = {name: set(identities) for name, identities in sources.items()}
        mutated["fixture"] = {(2001162, 120)}

        with self.assertRaisesRegex(ValueError, r"fixture=2001162/120"):
            C15._assert_no_active_lane_overlap(_targets(), mutated)

    def test_current_reread_preserves_the_comparator_payload(self) -> None:
        row = {"seed": 1, "step": 2}
        with mock.patch.object(
            C15, "reread_row", return_value=("diverged", ["first", "second"], 7)
        ):
            self.assertEqual(
                C15._current_reread(row),
                {"verdict": "diverged", "misses": ["first", "second"], "branch_count": 7},
            )

    def test_current_reread_rejects_unmodeled_slot_side_mappings(self) -> None:
        with self.assertRaisesRegex(ValueError, "without slot_sides"):
            C15._current_reread(
                {"seed": 1, "step": 2, "slot_sides": {"p1": "side_two"}}
            )


if __name__ == "__main__":
    unittest.main()
