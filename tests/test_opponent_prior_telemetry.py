"""Seat attribution for opponent-prior application, pinned at the Python boundary.

The crate resolves priors for two seats and, until this telemetry existed,
reported only `prior_branches` / `prior_fallbacks` -- both-seat sums. A run in
which the opponent map refused EVERY branch is indistinguishable, in those two
numbers, from one in which it applied cleanly. That is not a cosmetic gap: it
makes a paired `use_opponent_priors` delta unreadable, because a flat result
could equally mean "opponent priors do not help" or "opponent priors never ran",
and the second reads as the first.

These tests pin the part that must not regress: the seat-attributed counters
reach the shard SEPARATELY, the applied rate is computed from the opponent's own
denominator, and a digest of "nothing gathered" is never recorded as if it were
a real one.

The gather/apply arithmetic itself is pinned Rust-side in `priors.rs`
(`the_opponent_digest_separates_a_reordering_from_the_same_vectors` and the
seat-attributed count assertions); this file covers the plumbing between there
and a shard record.
"""

from __future__ import annotations

import unittest

from pokezero.engine_search import EngineMctsStats


class OpponentPriorTelemetryTest(unittest.TestCase):
    def test_seat_counters_are_reported_separately_from_the_both_seat_sums(self) -> None:
        """The opponent's numbers must not be recoverable only by subtraction."""
        stats = EngineMctsStats()
        stats.prior_fallbacks = 7
        stats.opponent_priors_applied = 3
        stats.opponent_priors_refused = 4
        stats.acting_priors_applied = 11
        stats.acting_priors_refused = 3

        payload = stats.to_dict()

        self.assertEqual(payload["opponent_priors_applied"], 3)
        self.assertEqual(payload["opponent_priors_refused"], 4)
        self.assertEqual(payload["acting_priors_applied"], 11)
        self.assertEqual(payload["acting_priors_refused"], 3)
        # The aggregate is untouched -- existing consumers keep reading it.
        self.assertEqual(payload["prior_fallbacks"], 7)

    def test_applied_rate_uses_the_opponents_own_denominator(self) -> None:
        """Not the both-seat total, which would flatter a refusing opponent.

        With the acting seat applying 11 of 14 and the opponent 3 of 7, a rate
        computed over both seats reads 14/21 = 67% and hides that the opponent
        half ran a third of the time.
        """
        stats = EngineMctsStats()
        stats.opponent_priors_applied = 3
        stats.opponent_priors_refused = 4
        stats.acting_priors_applied = 11
        stats.acting_priors_refused = 3

        self.assertAlmostEqual(stats.to_dict()["opponent_priors_applied_rate"], 3 / 7)

    def test_applied_rate_is_none_rather_than_zero_when_the_opponent_never_ran(self) -> None:
        """Zero would read as "ran and always refused", which is a different fact.

        `use_opponent_priors=False` produces no opponent resolutions at all. A
        0.0 there would be indistinguishable from a flag that was on and refused
        every time -- the exact conflation this telemetry exists to prevent.
        """
        stats = EngineMctsStats()
        self.assertIsNone(stats.to_dict()["opponent_priors_applied_rate"])

    def test_a_total_refusal_is_visible_and_not_confusable_with_success(self) -> None:
        stats = EngineMctsStats()
        stats.opponent_priors_applied = 0
        stats.opponent_priors_refused = 250

        payload = stats.to_dict()
        self.assertEqual(payload["opponent_priors_applied_rate"], 0.0)
        self.assertEqual(payload["opponent_priors_applied"], 0)
        self.assertEqual(payload["opponent_prior_digests"], [])


class OpponentPriorDigestPlumbingTest(unittest.TestCase):
    """The digest is the M9 observable; the plumbing must not blunt it."""

    def test_digests_are_deduplicated_but_order_independent(self) -> None:
        stats = EngineMctsStats()
        stats.opponent_prior_digests = ["00000000deadbeef", "00000000feedface", "00000000deadbeef"]
        self.assertEqual(
            stats.to_dict()["opponent_prior_digests"],
            ["00000000deadbeef", "00000000feedface"],
        )

    def test_more_than_one_digest_is_not_by_itself_a_defect(self) -> None:
        """Different positions legitimately gather different vectors.

        Recorded so a reader does not treat digest multiplicity as a signal.
        The digest discriminates WITHIN a fixture held constant, which is how
        the M9 fixture uses it -- not across a whole campaign.
        """
        stats = EngineMctsStats()
        stats.opponent_prior_digests = ["1", "2", "3"]
        self.assertEqual(len(stats.to_dict()["opponent_prior_digests"]), 3)


if __name__ == "__main__":
    unittest.main()
