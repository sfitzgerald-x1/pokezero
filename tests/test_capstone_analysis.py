from __future__ import annotations

import unittest

from pokezero.capstone_analysis import (
    CapstoneArmEvidence,
    CapstoneGameOutcome,
    CapstonePairedArmEvidence,
    analyze_paired_capstone_arms,
    analyze_primary_capstone,
    paired_capstone_delta,
)


def outcome(
    *,
    band: str,
    seed: int,
    seat: str,
    score: float,
    tied: bool = False,
    capped: bool = False,
    root_puct_fallbacks: int = 0,
    privileged_fallbacks: int = 0,
) -> CapstoneGameOutcome:
    return CapstoneGameOutcome(
        band=band,
        seed=seed,
        seat=seat,
        score=score,
        tied=tied,
        capped=capped,
        root_puct_fallbacks=root_puct_fallbacks,
        privileged_fallbacks=privileged_fallbacks,
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
            {"candidate_better": 1, "baseline_better": 1, "equal": 2},
        )
        self.assertEqual(
            first["overall"]["win_flip_counts"],
            {"both_won": 0, "baseline_only_won": 1, "candidate_only_won": 1, "neither_won": 2},
        )
        self.assertEqual(first["by_band"]["a"]["candidate_minus_baseline_score_rate"], 0.5)
        self.assertEqual(first["by_band"]["b"]["candidate_minus_baseline_score_rate"], -0.5)

    def test_overall_bootstrap_stratifies_by_band(self) -> None:
        baseline = (
            outcome(band="a", seed=1, seat="p1", score=0.0),
            outcome(band="a", seed=1, seat="p2", score=0.0),
            outcome(band="b", seed=2, seat="p1", score=0.0),
            outcome(band="b", seed=2, seat="p2", score=0.0),
            outcome(band="b", seed=3, seat="p1", score=0.0),
            outcome(band="b", seed=3, seat="p2", score=0.0),
            outcome(band="b", seed=4, seat="p1", score=0.0),
            outcome(band="b", seed=4, seat="p2", score=0.0),
        )
        candidate = tuple(
            outcome(
                band=item.band,
                seed=item.seed,
                seat=item.seat,
                score=1.0 if item.band == "a" else 0.0,
            )
            for item in baseline
        )

        report = paired_capstone_delta(baseline, candidate, bootstrap_replicates=100, bootstrap_seed=7)

        # Each resample retains one A seed and three B seeds, so every bootstrap mean is 1/4.
        self.assertEqual(report["overall"]["candidate_minus_baseline_score_rate"], 0.25)
        self.assertEqual(report["overall"]["paired_bootstrap_95"], {"lower": 0.25, "upper": 0.25})
        self.assertTrue(report["bootstrap"]["stratified_by_band"])

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

    def test_primary_capstone_requires_shared_roster_zero_fallbacks_and_calibration_label(self) -> None:
        baseline_outcomes = (
            outcome(band="a", seed=1, seat="p1", score=0.0),
            outcome(band="a", seed=1, seat="p2", score=0.0),
        )
        candidate_outcomes = (
            outcome(band="a", seed=1, seat="p1", score=1.0),
            outcome(band="a", seed=1, seat="p2", score=1.0),
        )
        baseline = CapstoneArmEvidence(
            arm_id="raw",
            outcomes=baseline_outcomes,
            uses_value_leaves=False,
        )
        candidate = CapstoneArmEvidence(
            arm_id="value-leaf",
            outcomes=candidate_outcomes,
            uses_value_leaves=True,
            calibrated_value_copy="isotonic-frozen",
        )
        expected_keys = tuple(item.key for item in baseline_outcomes)

        report = analyze_primary_capstone(
            baseline,
            (candidate,),
            expected_keys=expected_keys,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )

        self.assertEqual(report["shared_seed_roster"]["outcomes"], 2)
        self.assertEqual(report["primary_arms"][0]["calibrated_value_copy"], "isotonic-frozen")
        self.assertEqual(
            report["primary_arms"][0]["paired_delta_vs_baseline"]["overall"][
                "candidate_minus_baseline_score_rate"
            ],
            1.0,
        )
        fallback_candidate = CapstoneArmEvidence(
            arm_id="fallback-arm",
            outcomes=(
                outcome(band="a", seed=1, seat="p1", score=1.0, root_puct_fallbacks=1),
                outcome(band="a", seed=1, seat="p2", score=1.0),
            ),
            uses_value_leaves=True,
            calibrated_value_copy="isotonic-frozen",
        )
        with self.assertRaisesRegex(ValueError, "zero search fallbacks"):
            analyze_primary_capstone(
                baseline,
                (fallback_candidate,),
                expected_keys=expected_keys,
                bootstrap_replicates=100,
            )
        missing_roster_candidate = CapstoneArmEvidence(
            arm_id="missing-roster-arm",
            outcomes=(candidate_outcomes[0],),
            uses_value_leaves=True,
            calibrated_value_copy="isotonic-frozen",
        )
        with self.assertRaisesRegex(ValueError, "shared capstone seed roster"):
            analyze_primary_capstone(
                baseline,
                (missing_roster_candidate,),
                expected_keys=expected_keys,
                bootstrap_replicates=100,
            )
        privileged_candidate = CapstoneArmEvidence(
            arm_id="privileged-fallback-arm",
            outcomes=(
                outcome(band="a", seed=1, seat="p1", score=1.0, privileged_fallbacks=1),
                outcome(band="a", seed=1, seat="p2", score=1.0),
            ),
            uses_value_leaves=True,
            calibrated_value_copy="isotonic-frozen",
        )
        with self.assertRaisesRegex(ValueError, "zero privileged fallbacks"):
            analyze_primary_capstone(
                baseline,
                (privileged_candidate,),
                expected_keys=expected_keys,
                bootstrap_replicates=100,
            )
        with self.assertRaisesRegex(ValueError, "calibrated_value_copy"):
            CapstoneArmEvidence(
                arm_id="unlabeled-value-leaf",
                outcomes=candidate_outcomes,
                uses_value_leaves=True,
            )

    def test_per_arm_raw_controls_keep_external_opponent_pairs_honest(self) -> None:
        expected = (
            outcome(band="a", seed=1, seat="p1", score=0.0),
            outcome(band="a", seed=1, seat="p2", score=0.0),
        )
        pair = CapstonePairedArmEvidence(
            arm_id="value-24",
            baseline=CapstoneArmEvidence(arm_id="raw", outcomes=expected, uses_value_leaves=False),
            candidate=CapstoneArmEvidence(
                arm_id="value-24",
                outcomes=tuple(
                    outcome(band=item.band, seed=item.seed, seat=item.seat, score=1.0)
                    for item in expected
                ),
                uses_value_leaves=True,
                calibrated_value_copy="frozen-isotonic",
            ),
        )

        report = analyze_paired_capstone_arms(
            (pair,),
            expected_keys=tuple(item.key for item in expected),
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )

        self.assertEqual(report["analysis_method"], "primary_capstone_per_arm_raw_controls_paired_bootstrap")
        self.assertEqual(report["primary_arms"][0]["arm_id"], "value-24")
        self.assertEqual(
            report["primary_arms"][0]["paired_delta_vs_baseline"]["overall"][
                "candidate_minus_baseline_score_rate"
            ],
            1.0,
        )

    def test_per_arm_raw_control_rejects_mislabeled_or_partial_pairs(self) -> None:
        baseline = CapstoneArmEvidence(
            arm_id="not-raw",
            outcomes=(outcome(band="a", seed=1, seat="p1", score=0.0),),
            uses_value_leaves=False,
        )
        candidate = CapstoneArmEvidence(
            arm_id="value-0",
            outcomes=(outcome(band="a", seed=1, seat="p1", score=1.0),),
            uses_value_leaves=True,
            calibrated_value_copy="frozen-isotonic",
        )
        with self.assertRaisesRegex(ValueError, "baseline arm_id"):
            CapstonePairedArmEvidence(arm_id="value-0", baseline=baseline, candidate=candidate)


if __name__ == "__main__":
    unittest.main()
