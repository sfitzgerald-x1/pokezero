from __future__ import annotations

import unittest

from pokezero.capstone_analysis import CapstoneGameOutcome, paired_capstone_delta


def outcome(
    *,
    band: str,
    seed: int,
    seat: str,
    score: float,
    tied: bool = False,
    capped: bool = False,
) -> CapstoneGameOutcome:
    return CapstoneGameOutcome(
        band=band,
        seed=seed,
        seat=seat,
        score=score,
        tied=tied,
        capped=capped,
    )


class CapstoneAnalysisTest(unittest.TestCase):
    def test_mirrored_seed_bootstrap_uses_half_point_ties_and_caps(self) -> None:
        baseline = (
            outcome(band="a", seed=1, seat="p1", score=0.0),
            outcome(band="a", seed=1, seat="p2", score=0.0),
            outcome(band="a", seed=2, seat="p1", score=0.5, tied=True),
            outcome(band="a", seed=2, seat="p2", score=0.5, tied=True),
            outcome(band="b", seed=3, seat="p1", score=1.0),
            outcome(band="b", seed=3, seat="p2", score=1.0),
            outcome(band="b", seed=4, seat="p1", score=0.0),
            outcome(band="b", seed=4, seat="p2", score=0.0),
        )
        candidate = (
            outcome(band="a", seed=1, seat="p1", score=1.0),
            outcome(band="a", seed=1, seat="p2", score=1.0),
            outcome(band="a", seed=2, seat="p1", score=0.5, capped=True),
            outcome(band="a", seed=2, seat="p2", score=0.5, capped=True),
            outcome(band="b", seed=3, seat="p1", score=0.0),
            outcome(band="b", seed=3, seat="p2", score=0.0),
            outcome(band="b", seed=4, seat="p1", score=0.0),
            outcome(band="b", seed=4, seat="p2", score=0.0),
        )

        first = paired_capstone_delta(baseline, candidate, bootstrap_replicates=500, bootstrap_seed=7)
        second = paired_capstone_delta(baseline, candidate, bootstrap_replicates=500, bootstrap_seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first["overall"]["paired_games"], 8)
        self.assertEqual(first["overall"]["paired_seed_groups"], 4)
        self.assertEqual(first["overall"]["baseline"]["score_rate"], 0.375)
        self.assertEqual(first["overall"]["candidate"]["score_rate"], 0.375)
        self.assertEqual(first["overall"]["baseline"]["ties"], 2)
        self.assertEqual(first["overall"]["candidate"]["capped_games"], 2)
        self.assertEqual(first["overall"]["candidate_minus_baseline_score_rate"], 0.0)
        self.assertEqual(
            first["overall"]["score_flip_counts"],
            {"candidate_better": 2, "baseline_better": 2, "equal": 4},
        )
        self.assertEqual(
            first["overall"]["win_flip_counts"],
            {"both_won": 0, "baseline_only_won": 2, "candidate_only_won": 2, "neither_won": 4},
        )
        self.assertEqual(first["by_band"]["a"]["candidate_minus_baseline_score_rate"], 0.5)
        self.assertEqual(first["by_band"]["b"]["candidate_minus_baseline_score_rate"], -0.5)

    def test_requires_complete_mirrored_seat_pairs_by_default(self) -> None:
        baseline = (outcome(band="a", seed=1, seat="p1", score=0.0),)
        candidate = (outcome(band="a", seed=1, seat="p1", score=1.0),)

        with self.assertRaisesRegex(ValueError, "mirrored p1 and p2"):
            paired_capstone_delta(baseline, candidate, bootstrap_replicates=10)

    def test_rejects_missing_or_duplicate_pair_keys(self) -> None:
        baseline = (
            outcome(band="a", seed=1, seat="p1", score=0.0),
            outcome(band="a", seed=1, seat="p1", score=0.0),
        )
        candidate = (outcome(band="a", seed=1, seat="p1", score=1.0),)

        with self.assertRaisesRegex(ValueError, "duplicate baseline"):
            paired_capstone_delta(baseline, candidate, bootstrap_replicates=10)

    def test_rejects_inconsistent_half_point_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "ties and capped games"):
            outcome(band="a", seed=1, seat="p1", score=0.5)
        with self.assertRaisesRegex(ValueError, "ties and capped games"):
            outcome(band="a", seed=1, seat="p1", score=1.0, tied=True)


if __name__ == "__main__":
    unittest.main()
