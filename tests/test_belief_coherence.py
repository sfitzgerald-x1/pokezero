"""V1 — whole-game belief coherence, as a live integration test.

The plan's §3 of the readiness doc asked for this and it was never built: nothing anywhere asserted
that the TRUE variant stays in the candidate set at any point of a real game (grepped `tests/` for
containment/coherence/omniscient assertions, 2026-08-04: none reach it). Containment is the property
whose violation is maximally harmful — it poisons `CANDIDATE_SET_COUNT`, `UNCERTAINTY`, every
`possible-*` count and every sampled search world at once, and it fails SILENTLY.

This runs a SHORT sweep of the same harness the fleet gate runs
(`scripts/belief_coherence_gate.py`), rather than re-implementing its assertions here: two copies of
a coherence check drifting apart is the very defect class the harness exists to catch. The long
sweep (≥20k games) is a fleet job; this is the always-on regression guard.

Skips cleanly without a built Showdown checkout + node.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests.test_tier2_live_env import _integration_root

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class BeliefCoherenceSweepTest(unittest.TestCase):
    """A few real games, all seven assertion families, zero tolerated violations."""

    @classmethod
    def setUpClass(cls) -> None:
        from belief_coherence_gate import run_sweep

        cls.summary = run_sweep(
            showdown_root=_integration_root(),
            games=3,
            seed=7,
            clone_equivalence_every=5,
        )

    def test_no_coherence_violations(self) -> None:
        """Zero violations of families 1/2/4/5/6/7 — the plan's exit criterion, in miniature."""
        self.assertEqual(
            self.summary["violation_counts"],
            {name: 0 for name in self.summary["violation_counts"]},
            msg=f"coherence violations: {self.summary.get('violations')}",
        )
        self.assertEqual(self.summary["verdict"], "PASS")

    def test_sweep_actually_reached_the_properties_it_asserts(self) -> None:
        """The vacuous-pass guard.

        A containment sweep over games where no opponent mon was ever recognized passes trivially,
        which is the bug and not the fix (plan §3). Each of these counters was chosen because a
        zero would make the corresponding assertion meaningless, so the sweep's own verdict is FAIL
        unless all of them are positive — this test pins that the run really did reach them.
        """
        reach = self.summary["reachability"]
        self.assertGreater(reach["mon_observations"], 100, "too few belief observations")
        self.assertGreater(reach["distinct_species"], 5, "too few species reached")
        self.assertGreater(reach["narrowing_steps"], 0, "no set ever narrowed; containment is idle")
        self.assertGreater(reach["pinned_and_correct"], 0, "no set was ever pinned to one variant")
        self.assertGreater(reach["stat_legality_checks"], 0, "assertion 6 never ran")
        self.assertGreater(reach["pin_conflict_checks"], 0, "assertion 4 never ran")
        self.assertGreater(reach["clone_equivalence_checks"], 0, "assertion 7 never ran")
        self.assertTrue(self.summary["reached"])

    def test_every_monotonicity_growth_is_an_attributed_fallback(self) -> None:
        """Growth is allowed ONLY through the documented inconsistent-fallback, and is counted.

        The plan requires every fallback be "attributed to a known cause, not silently absorbed".
        A monotonicity violation is recorded whenever a set grows WITHOUT matching the fallback
        signature (full species pool at uncertainty 1.0), so an empty violation list plus a
        published fallback count is the attribution.
        """
        self.assertEqual(self.summary["violation_counts"]["monotonicity"], 0)
        self.assertIn("inconsistent_fallbacks", self.summary["counts"])

    def test_containment_holds_for_every_observation_not_merely_on_average(self) -> None:
        """containment_ok must equal the observation count — no silent partial credit."""
        counts = self.summary["counts"]
        self.assertEqual(
            counts.get("containment_ok", 0),
            counts.get("mon_observations", -1),
            "some observations were not containment-checked or not contained",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
