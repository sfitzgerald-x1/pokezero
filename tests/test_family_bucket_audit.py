"""Pins for the audit's signature and bucket logic.

This script decides whether a divergence family is engine-gap,
instrument-artifact or comparison-limit, and its output is the evidence
artifact the ledger publishes. It shipped with no tests, and re-review found
three ways it could emit a confident `instrument-artifact` verdict from
non-evidence. Those three are pinned here.
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cert_sweep_readout as readout  # noqa: E402
import family_bucket_audit as audit  # noqa: E402


def _sig(miss: str) -> dict:
    return audit.signatures(readout, {"branch_misses": [miss]})


class NetIdenticalTests(unittest.TestCase):
    """`net_identical` is the ONLY signature treated as decisive."""

    def test_an_empty_observation_is_not_an_agreeing_end_state(self) -> None:
        """sum([]) == 0, so cancelling engine components scored as agreement.

        Sandstorm and Leftovers are both max_hp/16 in gen3, so exact
        cancellation is common. Showdown reported NO components and the engine
        reported two -- an I5_boundary_truncation shape, not an artifact.
        """

        signature = _sig(
            "pct=100.0: p1 attributed components differ: observed_only=[] "
            "engine_only=[('sandstorm', -6), ('itemleftovers', 6)]"
        )
        self.assertFalse(signature["net_identical"])

    def test_a_genuine_agreeing_decomposition_still_scores(self) -> None:
        signature = _sig(
            "pct=100.0: p1 attributed components differ: "
            "observed_only=[('', -30), ('heal', 10)] "
            "engine_only=[('', -25), ('heal', 5)]"
        )
        self.assertTrue(signature["net_identical"])


class ToleranceLabelTests(unittest.TestCase):
    def test_transposed_labels_do_not_pass_the_roll_window(self) -> None:
        """zip pairs by POSITION. Swapped sources are an attribution tie."""

        signature = _sig(
            "pct=100.0: p1 roll-scaled components differ: "
            "observed_only=[('recoil', -20), ('', -100)] "
            "engine_only=[('', -20), ('recoil', -100)]"
        )
        self.assertFalse(signature["rolls_inside_tolerance"])

    def test_an_ordinary_roll_pair_still_passes(self) -> None:
        signature = _sig(
            "pct=100.0: p1 roll-scaled components differ: "
            "observed_only=[('', -100)] engine_only=[('', -104)]"
        )
        self.assertTrue(signature["rolls_inside_tolerance"])


class SampleFloorTests(unittest.TestCase):
    def test_one_row_does_not_buy_a_bucket_verdict(self) -> None:
        bucket, basis = audit.bucket_from_signatures(
            Counter({"measured": 1, "net_identical": 1}), 1
        )
        self.assertEqual(bucket, "candidate-not-finding")
        self.assertIn("floor", basis)

    def test_enough_rows_do(self) -> None:
        bucket, _ = audit.bucket_from_signatures(
            Counter({"measured": 6, "net_identical": 6}), 6
        )
        self.assertEqual(bucket, "instrument-artifact")

    def test_an_empty_population_is_no_rows(self) -> None:
        bucket, _ = audit.bucket_from_signatures(Counter(), 0)
        self.assertEqual(bucket, "no-rows")


if __name__ == "__main__":
    unittest.main()
