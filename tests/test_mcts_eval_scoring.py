"""Scoring/merge contract (plan section 3 + B3)."""

from __future__ import annotations

import unittest

from pokezero.mcts_eval.scoring import (
    GameResult,
    Interval,
    MergeError,
    bootstrap_indices,
    bootstrap_mean,
    bootstrap_paired_delta,
    merge_game_results,
    outcome_record,
    pair_scores,
    parity_label,
    promote_spare_pairs,
)

PROV = "p" * 64


def _game(seed: int, seat: str, outcome: str, *, config_id: str = "d8-s4096-b16-w4-local", **kw):
    values = dict(
        config_id=config_id, seed=seed, seat=seat, outcome=outcome, turns=30, provenance_sha256=PROV
    )
    values.update(kw)
    return GameResult(**values)


class ScoringConventionTest(unittest.TestCase):
    def test_cap_and_tie_score_half(self) -> None:
        self.assertEqual(_game(1, "p1", "win").score, 1.0)
        self.assertEqual(_game(1, "p1", "tie").score, 0.5)
        self.assertEqual(_game(1, "p1", "cap").score, 0.5)
        self.assertEqual(_game(1, "p1", "loss").score, 0.0)

    def test_invalid_outcome_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outcome"):
            _game(1, "p1", "forfeit")

    def test_pair_averages_both_seats(self) -> None:
        results = [_game(1, "p1", "win"), _game(1, "p2", "loss")]
        self.assertEqual(pair_scores(results, seeds=[1], config_id=results[0].config_id), [0.5])

    def test_missing_seat_fails_closed(self) -> None:
        with self.assertRaisesRegex(MergeError, "missing seat"):
            pair_scores([_game(1, "p1", "win")], seeds=[1], config_id="d8-s4096-b16-w4-local")

    def test_record_retains_raw_counts(self) -> None:
        results = [
            _game(1, "p1", "win"), _game(1, "p2", "loss"),
            _game(2, "p1", "cap"), _game(2, "p2", "tie", opponent_crashed=True),
        ]
        record = outcome_record(results, config_id=results[0].config_id)
        self.assertEqual((record["win"], record["loss"], record["cap"], record["tie"]), (1, 1, 1, 1))
        self.assertEqual(record["opponent_crashes"], 1)


class DuplicateIdempotencyTest(unittest.TestCase):
    def test_canonically_matching_duplicate_is_idempotent(self) -> None:
        a = _game(1, "p1", "win", decision_walls_s=(1.0, 2.0))
        b = _game(1, "p1", "win", decision_walls_s=(9.0, 9.0))  # retry: timing differs only
        merged = merge_game_results([a, b])
        self.assertEqual(len(merged), 1)

    def test_conflicting_duplicate_is_terminal(self) -> None:
        a = _game(1, "p1", "win")
        b = _game(1, "p1", "loss")
        with self.assertRaisesRegex(MergeError, "canonical outcome conflict"):
            merge_game_results([a, b])

    def test_provenance_drift_is_a_conflict(self) -> None:
        a = _game(1, "p1", "win")
        b = _game(1, "p1", "win", provenance_sha256="q" * 64)
        with self.assertRaisesRegex(MergeError, "canonical outcome conflict"):
            merge_game_results([a, b])


class BootstrapTest(unittest.TestCase):
    def test_indices_are_deterministic_and_shared(self) -> None:
        first = bootstrap_indices(sample_size=50, resamples=100, seed=7)
        second = bootstrap_indices(sample_size=50, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, bootstrap_indices(sample_size=50, resamples=100, seed=8))

    def test_mean_interval_brackets_point(self) -> None:
        values = [0.5] * 25 + [1.0] * 25
        indices = bootstrap_indices(sample_size=50, resamples=500, seed=3)
        interval = bootstrap_mean(values, indices)
        self.assertAlmostEqual(interval.point, 0.75)
        self.assertLessEqual(interval.low, interval.point)
        self.assertGreaterEqual(interval.high, interval.point)

    def test_identical_vectors_give_zero_paired_delta(self) -> None:
        values = [0.4, 0.6, 0.5, 1.0, 0.0] * 10
        indices = bootstrap_indices(sample_size=50, resamples=200, seed=11)
        delta = bootstrap_paired_delta(values, values, indices)
        self.assertEqual(delta.point, 0.0)
        self.assertEqual((delta.low, delta.high), (0.0, 0.0))

    def test_paired_delta_detects_uniform_improvement(self) -> None:
        baseline = [0.5] * 50
        treatment = [0.7] * 50
        indices = bootstrap_indices(sample_size=50, resamples=200, seed=5)
        delta = bootstrap_paired_delta(treatment, baseline, indices)
        self.assertAlmostEqual(delta.point, 0.2)
        self.assertGreater(delta.low, 0.0)

    def test_length_mismatch_rejected(self) -> None:
        indices = bootstrap_indices(sample_size=2, resamples=10, seed=1)
        with self.assertRaisesRegex(ValueError, "equal-length"):
            bootstrap_paired_delta([0.5, 0.5], [0.5], indices)


class ParityLanguageTest(unittest.TestCase):
    def test_labels_follow_section_3(self) -> None:
        self.assertEqual(parity_label(Interval(0.40, 0.35, 0.45)), "clearly below parity")
        self.assertEqual(parity_label(Interval(0.52, 0.45, 0.60)), "parity-compatible")
        self.assertEqual(parity_label(Interval(0.62, 0.55, 0.70)), "directionally above parity")

    def test_never_claims_parity_achieved(self) -> None:
        labels = {
            parity_label(Interval(p / 100, p / 100 - 0.05, p / 100 + 0.05)) for p in range(0, 101)
        }
        self.assertNotIn("parity achieved", labels)


class SparePromotionTest(unittest.TestCase):
    def test_excluded_pair_is_replaced_in_fixed_order(self) -> None:
        remaining, subs = promote_spare_pairs(
            primary_seeds=[1, 2, 3], spare_seeds=[90, 91], excluded=[2]
        )
        self.assertEqual(subs, {2: 90})
        self.assertEqual(sorted(remaining), [1, 3, 90])
        self.assertEqual(len(remaining), 3)  # entries still share a full set

    def test_exhausted_spare_band_is_terminal(self) -> None:
        with self.assertRaisesRegex(MergeError, "spare seed band exhausted"):
            promote_spare_pairs(primary_seeds=[1, 2], spare_seeds=[90], excluded=[1, 2])


if __name__ == "__main__":
    unittest.main()
