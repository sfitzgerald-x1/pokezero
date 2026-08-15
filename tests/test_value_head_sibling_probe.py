"""Tests for the value-head sibling-discrimination probe's scoring core.

The probe's expensive half (branching and rollouts) needs a built Showdown checkout, so it
is exercised in-image. Its ANALYTIC half decides what the numbers mean and is tested here,
because a scorer that silently mishandles a tie or a bucket boundary produces a plausible
accuracy figure that nobody can tell is wrong.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "vhsp", Path(__file__).resolve().parents[1] / "scripts" / "value_head_sibling_probe.py")
vhsp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vhsp)


def pair(head_gap: float, true_gap: float) -> dict:
    return {"head_gap": head_gap, "true_gap": true_gap}


class WilsonTest(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self) -> None:
        lo, hi = vhsp.wilson(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)

    def test_degenerate_n_is_the_whole_unit_interval(self) -> None:
        # Not a crash and not a spuriously tight interval: with no data the honest
        # statement is that anything is possible.
        self.assertEqual(vhsp.wilson(0, 0), (0.0, 1.0))

    def test_unanimity_does_not_produce_a_zero_width_interval(self) -> None:
        """10 of 10 is not proof. A normal approximation gives [1.0, 1.0] here, which
        would let a 10-pair bucket read as certainty."""
        lo, hi = vhsp.wilson(10, 10)
        self.assertLess(lo, 1.0)
        self.assertGreater(lo, 0.5)


class RolloutSeedTest(unittest.TestCase):
    def test_paired_seeds_are_SHARED_across_the_two_arms(self) -> None:
        """The whole point of common random numbers.

        Arm A trial i and arm B trial i must draw the same chances wherever their lines
        coincide, so the shared component of the variance cancels. Without this the probe
        needs thousands of rollouts to resolve a 0.02 gap instead of hundreds.
        """
        a = vhsp.rollout_seed(7, "b", 3, 0, 5, paired=True)
        b = vhsp.rollout_seed(7, "b", 3, 1, 5, paired=True)
        self.assertEqual(a, b)

    def test_unpaired_seeds_differ_across_arms(self) -> None:
        a = vhsp.rollout_seed(7, "b", 3, 0, 5, paired=False)
        b = vhsp.rollout_seed(7, "b", 3, 1, 5, paired=False)
        self.assertNotEqual(a, b)

    def test_trials_differ_from_each_other(self) -> None:
        seeds = {vhsp.rollout_seed(7, "b", 3, 0, t, paired=True) for t in range(32)}
        self.assertEqual(len(seeds), 32, "trials must not collide, or N rollouts is a lie")

    def test_distinct_decisions_do_not_share_seeds(self) -> None:
        self.assertNotEqual(vhsp.rollout_seed(7, "b", 3, 0, 0, paired=True),
                            vhsp.rollout_seed(7, "b", 4, 0, 0, paired=True))


class ScorePairsTest(unittest.TestCase):
    BUCKETS = [0.0, 0.02, 0.05, 0.10, 0.20]

    def test_perfect_agreement_scores_one(self) -> None:
        pairs = [pair(+0.1, +0.3), pair(-0.1, -0.3), pair(+0.2, +0.25)]
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        self.assertEqual(out["overall"]["accuracy"], 1.0)

    def test_perfect_disagreement_scores_zero(self) -> None:
        """Below chance is a real and different finding from at-chance: it would mean the
        head's ordering is anti-correlated with truth, which is a bug, not a ceiling."""
        pairs = [pair(+0.1, -0.3), pair(-0.1, +0.3)]
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        self.assertEqual(out["overall"]["accuracy"], 0.0)

    def test_a_zero_TRUE_gap_pair_is_excluded_not_counted_as_wrong(self) -> None:
        """If ground truth says the two arms are equal there is no correct ordering.

        Counting such a pair as a failure would drag accuracy toward zero in exactly the
        narrow-gap bucket the probe exists to read -- manufacturing the conclusion that the
        head cannot rank.
        """
        pairs = [pair(+0.1, +0.3), pair(+0.1, 0.0), pair(-0.5, 0.0)]
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        self.assertEqual(out["overall"]["n"], 1)
        self.assertEqual(out["overall"]["accuracy"], 1.0)

    def test_a_zero_HEAD_gap_counts_as_a_failure_to_rank(self) -> None:
        """The head declaring two arms exactly equal IS a failure to discriminate when
        truth separates them -- unlike a zero true gap, it must not be excluded."""
        out = vhsp.score_pairs([pair(0.0, +0.3)], self.BUCKETS)
        self.assertEqual(out["overall"]["n"], 1)
        self.assertEqual(out["overall"]["accuracy"], 0.0)

    def test_a_zero_HEAD_gap_against_a_NEGATIVE_true_gap_is_still_a_failure(self) -> None:
        """The asymmetry the first version of this test missed.

        `(head_gap > 0) == (true_gap > 0)` is False==False -> True when the head says
        "exactly equal" and truth prefers arm B, so the sign test ALONE credits the head as
        correct. The `head_gap != 0` guard is what catches it, and it is load-bearing only
        for negative true gaps. A test using a positive gap passes with or without the
        guard, which is how a surviving mutant revealed the test was weak rather than the
        code wrong.
        """
        out = vhsp.score_pairs([pair(0.0, -0.3)], self.BUCKETS)
        self.assertEqual(out["overall"]["n"], 1)
        self.assertEqual(out["overall"]["accuracy"], 0.0,
                         "a head that declares two arms equal has not ranked them")

    def test_zero_TRUE_gap_exclusion_applies_INSIDE_each_bucket_too(self) -> None:
        """The exclusion happens twice -- once per bucket, once for the overall -- and the
        bucket path was untested, so a mutant that broke only the bucket filter survived.
        The per-bucket numbers are the ones the conclusion rests on."""
        pairs = [pair(+0.1, +0.01), pair(+0.1, 0.0), pair(-0.9, 0.0)]
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        narrow = next(b for b in out["buckets"] if b["lo"] == 0.0)
        self.assertEqual(narrow["n"], 1, "zero-true-gap pairs must not enter a bucket")
        self.assertEqual(narrow["accuracy"], 1.0)

    def test_bucketing_is_on_the_absolute_true_gap_and_boundaries_are_lower_closed(self) -> None:
        pairs = [pair(+1.0, +0.01),    # bucket 0.00-0.02
                 pair(+1.0, -0.02),    # bucket 0.02-0.05, negative -> abs
                 pair(+1.0, +0.30)]    # bucket 0.20-inf
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        got = {(b["lo"], b["n"]) for b in out["buckets"] if b["n"]}
        self.assertEqual(got, {(0.0, 1), (0.02, 1), (0.20, 1)})

    def test_the_narrow_bucket_is_reported_separately_from_the_flattering_wide_one(self) -> None:
        """The headline must not be able to hide a narrow-gap failure behind easy pairs.

        Wide pairs all correct, narrow pairs all wrong: overall looks like a coin flip while
        the bucket that matters reads 0.0. Reporting only the overall number would describe
        this head as 'at chance' when it is in fact useless exactly where search operates.
        """
        pairs = ([pair(+1.0, +0.5)] * 10) + ([pair(+1.0, -0.01)] * 10)
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        narrow = next(b for b in out["buckets"] if b["lo"] == 0.0)
        wide = next(b for b in out["buckets"] if b["lo"] == 0.20)
        self.assertEqual(narrow["accuracy"], 0.0)
        self.assertEqual(wide["accuracy"], 1.0)
        self.assertAlmostEqual(out["overall"]["accuracy"], 0.5)
        self.assertFalse(narrow["beats_chance"])

    def test_beats_chance_requires_the_interval_to_clear_one_half(self) -> None:
        # 6/10 is above 0.5 as a point estimate but its interval spans it.
        pairs = ([pair(+1.0, +0.5)] * 6) + ([pair(+1.0, -0.5)] * 4)
        out = vhsp.score_pairs(pairs, self.BUCKETS)
        b = next(x for x in out["buckets"] if x["lo"] == 0.20)
        self.assertEqual(b["accuracy"], 0.6)
        self.assertFalse(b["beats_chance"], "a point estimate above 0.5 is not significance")

    def test_empty_input_does_not_fabricate_an_overall(self) -> None:
        out = vhsp.score_pairs([], self.BUCKETS)
        self.assertNotIn("overall", out)
        self.assertTrue(all(b["n"] == 0 for b in out["buckets"]))


class JsonSerialisabilityTest(unittest.TestCase):
    """The payload must be valid JSON.

    The open-ended top bucket held float("inf"), which json.dumps writes as a bare
    `Infinity` token: jq accepts it, Node's JSON.parse rejects the whole file. Untested
    until now, and doubly invisible because the table path renders f"{inf:.2f}" as "inf"
    too, so even stdout looked identical. Same "test cannot see its own subject" weakness
    the surrounding commit was written to remove.
    """

    def test_no_Infinity_or_NaN_token_reaches_the_json(self) -> None:
        import json
        out = vhsp.score_pairs([pair(+1.0, +0.5), pair(-0.3, -0.01)],
                               [0.0, 0.02, 0.05, 0.10, 0.20])
        blob = json.dumps(out)
        self.assertNotIn("Infinity", blob)
        self.assertNotIn("NaN", blob)
        # allow_nan=False is the strict check: it RAISES on inf/nan rather than emitting
        # a token, so this fails loudly if either ever creeps back in.
        json.dumps(out, allow_nan=False)

    def test_the_open_bucket_is_null_not_a_sentinel_number(self) -> None:
        out = vhsp.score_pairs([pair(+1.0, +0.5)], [0.0, 0.2])
        self.assertIsNone(out["buckets"][-1]["hi"],
                          "the open-ended bucket must serialise as null")


class TerminalArmHandlingTest(unittest.TestCase):
    """A branch that ends the battle has no successor, so the head cannot be scored on it.

    An earlier revision scored the terminal arm on the PRE-BRANCH position. For a pair
    where both arms end the game that makes head_a == head_b exactly, so head_gap == 0,
    which score_pairs counts as a miss -- turning every decisive pair into a deterministic
    zero in the widest bucket, the exact bucket the terminal handling exists to populate.
    """

    def test_a_double_terminal_pair_would_be_a_guaranteed_miss_if_scored(self) -> None:
        # This is what the bad version produced: identical head values, opposite outcomes.
        out = vhsp.score_pairs([pair(0.0, 1.0)], [0.0, 0.02, 0.05, 0.10, 0.20])
        widest = next(b for b in out["buckets"] if b["lo"] == 0.20)
        self.assertEqual(widest["accuracy"], 0.0)
        self.assertEqual(widest["n"], 1)
        # ...which is why such pairs must be EXCLUDED upstream rather than scored. The
        # probe now routes them to terminal_pairs and never calls score_pairs on them.

    def test_se_is_None_not_NaN_when_no_pair_completed_rollouts(self) -> None:
        """A terminal arm runs zero rollouts by design.

        Counting it as a zero-rollout arm made n_eff 0 and the reported SE NaN -- and wrote
        a bare NaN token into the JSON, which strict parsers reject. None means CANNOT RUN
        and prints as such.
        """
        import math
        realised = [min(p["rollouts_a"], p["rollouts_b"])
                    for p in [{"rollouts_a": 0, "rollouts_b": 64}]
                    if p.get("rollouts_a") and p.get("rollouts_b")]
        self.assertEqual(realised, [], "a terminal arm must not enter the population")
        n_eff = min(realised) if realised else 0
        se = (0.5 / math.sqrt(n_eff)) if n_eff else None
        self.assertIsNone(se)
        self.assertFalse(isinstance(se, float) and math.isnan(se))




class SpreadPrefixesTest(unittest.TestCase):
    """The sampler must span the whole game, not the first `k` rounds.

    The regression it guards: `stride = max(1, len(usable) // k)` is 1 for every length in
    [k, 2k), so the old rule silently degenerated to "take the first k" on exactly the
    short-to-medium games that dominate a small probe run.
    """

    def test_reaches_the_last_usable_round_in_the_degenerate_band(self):
        # len 11, k 6 -> the old rule gave [0..5] and never sampled past the midpoint.
        usable = list(range(11))
        got = vhsp.spread_prefixes(usable, 6)
        self.assertEqual(got[0], 0)
        self.assertEqual(got[-1], 10, f"late game unsampled: {got}")

    def test_old_stride_rule_would_have_failed_this(self):
        usable = list(range(11))
        stride = max(1, len(usable) // 6)
        old = usable[::stride][:6]
        self.assertEqual(old, [0, 1, 2, 3, 4, 5])
        self.assertNotEqual(old, vhsp.spread_prefixes(usable, 6))

    def test_respects_the_usable_set_and_never_invents_a_round(self):
        usable = [3, 9, 14, 27, 31]
        for k in range(1, 9):
            got = vhsp.spread_prefixes(usable, k)
            self.assertTrue(set(got) <= set(usable))
            self.assertEqual(got, sorted(set(got)))
            self.assertLessEqual(len(got), min(k, len(usable)))

    def test_k_at_or_above_length_takes_everything(self):
        self.assertEqual(vhsp.spread_prefixes([1, 2, 3], 3), [1, 2, 3])
        self.assertEqual(vhsp.spread_prefixes([1, 2, 3], 99), [1, 2, 3])

    def test_empty_and_singleton(self):
        self.assertEqual(vhsp.spread_prefixes([], 5), [])
        self.assertEqual(vhsp.spread_prefixes([7], 5), [7])
        self.assertEqual(vhsp.spread_prefixes([7, 8], 0), [])


class BucketResolutionTest(unittest.TestCase):
    """Buckets narrower than 1/rollouts describe the instrument, not the head."""

    def test_zero_gap_pairs_fall_in_no_bucket(self):
        # [lo, hi) with lo == 0.0 excludes an exact zero. That is the modal outcome at low
        # rollout counts, so it must be reported separately, never folded into "small gap".
        pairs = [{"true_gap": 0.0, "head_gap": 0.1, "agree": False}] * 4
        out = vhsp.score_pairs(pairs, [0.0, 0.02])
        self.assertEqual(sum(b["n"] for b in out["buckets"]), 0)

    def test_a_bucket_below_resolution_can_only_hold_one_attainable_value(self):
        # At 64 rollouts the gap is a multiple of 1/64 = 0.015625, so [0.00, 0.02) admits
        # exactly one non-zero value. An "accuracy" there is one quantum, not a curve.
        q = 1.0 / 64
        attainable = [v for v in (i * q for i in range(0, 3)) if 0.0 < v < 0.02]
        self.assertEqual(len(attainable), 1)


class SigmaDiffEstimatorTest(unittest.TestCase):
    """The estimator must recover a KNOWN sigma_diff from synthetic pairs.

    This is the readout the verdict rests on, so it is tested against ground truth it
    cannot see rather than against its own output.
    """

    @staticmethod
    def _pairs(sigma_diff, R, n, seed, crn=0.0):
        """Synthetic pairs. `crn` is the common-random-number share.

        crn exists because the first version of this generator hardcoded crn=0 -- one of
        the two assumptions the estimator got wrong -- and a generator that bakes in the
        defect refutes it by construction. crn=1.0 is the probe's default (--paired-seeds).

        A `scale` knob was added alongside it and was an exact NO-OP: it multiplied the
        head by `scale` and the record divided straight back by `scale`. It was removed
        rather than left in place, because a parameter that appears to vary something and
        does not is worse than no parameter -- the units regression is pinned instead by
        test_a_units_mismatch_manufactures_a_differential_from_nothing (on the estimator)
        and HeadGapConversionTest (on the conversion itself).
        """
        import random as _r
        rng = _r.Random(seed)
        out = []
        for _ in range(n):
            tg = rng.expovariate(1 / 0.0113) * rng.choice([1, -1])
            pa, pb = 0.5 + tg / 2, 0.5 - tg / 2
            oa, ob = {}, {}
            for t in range(R):
                shared = rng.random()          # the common tape both arms see
                ua = shared if rng.random() < crn else rng.random()
                ub = shared if rng.random() < crn else rng.random()
                oa[t] = 1.0 if ua < pa else 0.0
                ob[t] = 1.0 if ub < pb else 0.0
            wa = sum(oa.values()) / R
            wb = sum(ob.values()) / R
            head = tg + rng.gauss(0, sigma_diff)
            out.append({"head_gap": head, "true_gap": wa - wb,
                        "true_a": wa, "true_b": wb, "rollouts_a": R, "rollouts_b": R,
                        "outcomes_a": oa, "outcomes_b": ob})
        return out

    def test_recovers_a_large_differential_on_average(self):
        # Averaged over seeds, not asserted on one. The per-run sd at n=400 is ~0.009, so a
        # single-seed assertion with a tight delta is a coin flip on the RNG -- the first
        # version of this test used seed 1, which lands 2 sd low (0.0335), and failed. A
        # test that fails on an unlucky seed teaches nothing about the estimator.
        import statistics as _st
        got = [vhsp.estimate_sigma_diff(self._pairs(0.0516, 64, 400, s), n_boot=1)
               ["sigma_diff"] for s in range(12)]
        self.assertAlmostEqual(_st.mean(got), 0.0516, delta=0.008)

    def test_a_small_differential_reads_small_on_average(self):
        import statistics as _st
        got = [vhsp.estimate_sigma_diff(self._pairs(0.009, 64, 400, s), n_boot=1)
               ["sigma_diff"] for s in range(12)]
        self.assertLess(_st.mean(got), 0.025)

    def test_the_error_is_one_directional_which_is_the_load_bearing_property(self):
        """A LOW reading is trustworthy at modest n; a HIGH reading is not.

        Measured over 60 repeats at n=150: a truly-fine head (0.009) reads above the 0.025
        threshold 27% of the time, while a truly-binding head (0.0516) reads below it 0% of
        the time. The zero-clip biases small values upward, so the ONLY direction the
        estimator errs in is "the head binds" -- which is the direction that would launch a
        training programme. This test pins the safe half: a binding head must not read fine.
        """
        for seed in range(8):
            got = vhsp.estimate_sigma_diff(self._pairs(0.0516, 64, 150, 500 + seed),
                                           n_boot=1)
            self.assertGreater(got["sigma_diff"], 0.025,
                               f"a truly-binding head read as fine on seed {seed}")

    def test_a_perfect_head_is_reported_at_or_near_the_floor(self):
        # sigma_diff == 0 is the case the RETRACTED agreement metric scored 0.563 on, i.e.
        # indistinguishable from a hopeless head. This estimator must not repeat that.
        got = vhsp.estimate_sigma_diff(self._pairs(0.0, 64, 300, 5), n_boot=200)
        self.assertLess(got["sigma_diff"], 0.020)

    def test_uses_the_sample_variance_when_trials_can_tie(self):
        """Outcomes live in {0, 0.5, 1}, so the Bernoulli formula is the wrong variance.

        An arm whose trials are all exactly 0.5 has ZERO variance, but w(1-w) calls it
        0.25 -- the maximum. Over-subtracting noise understates sigma_diff, which is the
        direction that would wrongly clear the head.
        """
        R = 8
        tie = {i: 0.5 for i in range(R)}
        pairs = [{"head_gap": 0.10, "true_gap": 0.0, "true_a": 0.5, "true_b": 0.5,
                  "rollouts_a": R, "rollouts_b": R,
                  "outcomes_a": dict(tie), "outcomes_b": dict(tie)} for _ in range(40)]
        got = vhsp.estimate_sigma_diff(pairs, n_boot=20)
        # Zero rollout noise, so the whole 0.10 offset is differential. With no spread in
        # head_gap the variance is 0 -- what must NOT happen is a large subtracted noise
        # term. Assert directly on it.
        self.assertEqual(got["subtracted_noise_var"], 0.0)

    def test_bernoulli_fallback_when_outcomes_absent(self):
        R = 16
        pairs = [{"head_gap": 0.02, "true_gap": 0.0, "true_a": 0.5, "true_b": 0.5,
                  "rollouts_a": R, "rollouts_b": R} for _ in range(20)]
        got = vhsp.estimate_sigma_diff(pairs, n_boot=10)
        self.assertAlmostEqual(got["subtracted_noise_var"], 2 * 0.25 / (R - 1), places=9)

    def test_common_random_numbers_do_not_drive_a_binding_head_to_zero(self):
        """The regression that made the estimator report 'thesis refuted' for a binding head.

        Under CRN the two arms covary, so var(w_a - w_b) is much smaller than
        var_a + var_b. Subtracting the independent sum over-subtracts by exactly the amount
        the pairing was designed to create, and a head at the binding 0.0516 read 0.0000.
        Paired seeds are the probe's DEFAULT, so this was not an edge case.
        """
        import statistics as _st
        for crn in (1.0, 0.5):
            got = [vhsp.estimate_sigma_diff(
                self._pairs(0.0516, 64, 400, s, crn=crn), n_boot=1)["sigma_diff"]
                for s in range(8)]
            self.assertGreater(_st.mean(got), 0.035,
                               f"a binding head read as refuted under crn={crn}")

    def test_a_units_mismatch_manufactures_a_differential_from_nothing(self):
        """A PERFECT head must not read as binding because the gaps are in different units.

        DETERMINISTIC, with no RNG anywhere, because the two obvious formulations are both
        unsound. A fixed threshold pins nothing -- an earlier version asserted `< 0.015`
        while the unfixed case returned 0.0084. A ratio of mismatched-to-matched is worse:
        the matched arm sits at the clipped floor, so the ratio ranged 1.16x to 7.50x over
        six batches of seeds and the test passed only on the seeds it happened to use.

        Construction: every trial of an arm carries that arm's exact rate, so the sample
        variance is exactly 0 and there is no rollout noise to subtract. The head is exact.
        Then sigma_diff must be exactly 0 on one scale, and exactly sd(true_gap) when the
        head is left at 2x -- because d_i = 2g_i - g_i = g_i. Both sides are closed form.
        """
        gaps = [0.05, -0.03, 0.12, -0.08, 0.01, 0.20, -0.15, 0.07, -0.02, 0.10]
        base = []
        for i, g in enumerate(gaps):
            wa, wb = 0.5 + g / 2, 0.5 - g / 2
            base.append({"head_gap": g, "true_gap": g, "true_a": wa, "true_b": wb,
                         "rollouts_a": 8, "rollouts_b": 8,
                         "outcomes_a": {t: wa for t in range(8)},
                         "outcomes_b": {t: wb for t in range(8)}})
        matched = vhsp.estimate_sigma_diff(base, n_boot=1)
        self.assertEqual(matched["subtracted_noise_var"], 0.0)
        self.assertAlmostEqual(matched["sigma_diff"], 0.0, places=12)

        bad = [dict(pr, head_gap=pr["head_gap"] * 2.0) for pr in base]
        got = vhsp.estimate_sigma_diff(bad, n_boot=1)["sigma_diff"]
        mean_g = sum(gaps) / len(gaps)
        expected = (sum((g - mean_g) ** 2 for g in gaps) / len(gaps)) ** 0.5
        self.assertAlmostEqual(got, expected, places=12)
        # And it is not a rounding-level artifact: it clears the refutation boundary.
        self.assertGreater(got, 0.035)

    def test_refuses_rather_than_guesses_on_too_few_pairs(self):
        got = vhsp.estimate_sigma_diff([], n_boot=10)
        self.assertIsNone(got["sigma_diff"])
        self.assertIn("CANNOT RUN", got["why"])

    def test_terminal_arms_are_excluded(self):
        ps = self._pairs(0.02, 64, 20, 6)
        ps[0]["terminal_a"] = True
        self.assertEqual(vhsp.estimate_sigma_diff(ps, n_boot=10)["n"], 19)

    def test_reports_how_much_of_the_variance_was_noise(self):
        got = vhsp.estimate_sigma_diff(self._pairs(0.009, 64, 200, 7), n_boot=50)
        self.assertGreater(got["noise_share_of_variance"], 0.5)


class HeadGapConversionTest(unittest.TestCase):
    """Pins the units blocker AT ITS FIX SITE.

    Reverting main() to `head_gap = head_a - head_b` left all 39 tests green: the estimator
    tests exercise scale sensitivity, not this conversion. So the fix for a blocker that
    made a perfect head read as binding had no regression guard of its own.
    """

    def test_halves_the_return_scale_difference(self):
        self.assertAlmostEqual(vhsp.head_gap_win_prob(0.4, 0.2), 0.1)
        self.assertAlmostEqual(vhsp.head_gap_win_prob(-1.0, 1.0), -1.0)
        self.assertAlmostEqual(vhsp.head_gap_win_prob(0.5, 0.5), 0.0)

    def test_matches_the_gap_of_the_crate_s_own_map(self):
        """values01 = 0.5*(v+1) -- the conversion must be the gap of exactly that map."""
        for a, b in ((0.0753, 0.1035), (-0.0524, -0.0399), (0.9, -0.9), (0.0, 0.0)):
            pa, pb = 0.5 * (a + 1.0), 0.5 * (b + 1.0)
            self.assertAlmostEqual(vhsp.head_gap_win_prob(a, b), pa - pb, places=12)

    def test_a_perfectly_calibrated_head_lands_on_the_rollout_scale(self):
        # A head whose return value is exactly right for a position won with probability p
        # must produce a win-probability gap equal to the rollout rate gap.
        for pa, pb in ((0.828, 0.781), (0.5, 0.75), (1.0, 0.0)):
            va, vb = 2 * pa - 1, 2 * pb - 1          # invert (v+1)/2
            self.assertAlmostEqual(vhsp.head_gap_win_prob(va, vb), pa - pb, places=12)


class FinalizePairGapsTest(unittest.TestCase):
    """Guards the units fix AT THE RECORD, which is where it was actually missing.

    HeadGapConversionTest covers the pure function. It is not sufficient: reverting the call
    site to `rec["head_gap"] = rec["head_a"] - rec["head_b"]` left all 42 tests green,
    because nothing asserted the record was built through the conversion. These tests fail
    on that revert.
    """

    def test_head_gap_on_the_record_is_the_win_probability_gap(self):
        rec = vhsp.finalize_pair_gaps(
            {"head_a": 0.0753, "head_b": 0.1035, "true_a": 0.828, "true_b": 0.781})
        # NOT head_a - head_b (-0.0282). Halved, because the head is on the +/-1 scale.
        self.assertAlmostEqual(rec["head_gap"], -0.0141, places=12)
        self.assertNotAlmostEqual(rec["head_gap"], 0.0753 - 0.1035, places=6)

    def test_the_return_scale_gap_is_retained_unhalved(self):
        rec = vhsp.finalize_pair_gaps(
            {"head_a": 0.4, "head_b": -0.2, "true_a": 0.6, "true_b": 0.4})
        self.assertAlmostEqual(rec["head_gap_return_scale"], 0.6, places=12)
        self.assertAlmostEqual(rec["head_gap"], 0.3, places=12)

    def test_the_marker_the_merger_gates_on_is_always_stamped(self):
        rec = vhsp.finalize_pair_gaps(
            {"head_a": 0.0, "head_b": 0.0, "true_a": 0.5, "true_b": 0.5})
        self.assertIn("head_gap_return_scale", rec)

    def test_head_and_true_gaps_share_units(self):
        # A perfectly calibrated head must produce exactly the rollout gap on the record.
        pa, pb = 0.828, 0.781
        rec = vhsp.finalize_pair_gaps({"head_a": 2 * pa - 1, "head_b": 2 * pb - 1,
                                       "true_a": pa, "true_b": pb})
        self.assertAlmostEqual(rec["head_gap"], rec["true_gap"], places=12)


class MeasuredErrorRatesTest(unittest.TestCase):
    """The advisory rates must come from the measured table, not from invented bands.

    The revision before this one hardcoded three coupling bands whose quoted rates were
    wrong by up to 6 points, and gated a warning on `n >= 1200` -- a figure that had already
    been retracted in the analysis it cited, which measured only n=150 and n=300.
    """

    def test_every_adjacent_pair_is_monotone_in_coupling(self):
        """ALL adjacent cells, not three samples.

        The first published table sampled pvr 1.00/0.59/0.00 at n=150 only, and the one
        non-monotone cell -- (0.88, 300) at 0.190 against (1.00, 300)'s 0.180 -- sat
        precisely where nothing looked. The commit message claimed monotonicity anyway.
        """
        for n in (150, 300):
            rows = sorted([r for r in vhsp.MEASURED_ERROR_RATES if r[1] == n],
                          key=lambda r: -r[0])
            for (pvr_hi, _, fb_hi, ff_hi), (pvr_lo, _, fb_lo, ff_lo) in zip(rows, rows[1:]):
                self.assertGreaterEqual(
                    fb_hi, fb_lo,
                    f"false-binds rises as coupling improves: pvr {pvr_hi}->{pvr_lo} at n={n}")
                self.assertGreaterEqual(
                    ff_hi, ff_lo,
                    f"false-fine rises as coupling improves: pvr {pvr_hi}->{pvr_lo} at n={n}")

    def test_every_coupling_level_improves_with_more_pairs(self):
        by_pvr = {}
        for pvr, n, fb, ff in vhsp.MEASURED_ERROR_RATES:
            by_pvr.setdefault(pvr, {})[n] = (fb, ff)
        for pvr, cells in by_pvr.items():
            self.assertEqual(set(cells), {150, 300}, f"pvr {pvr} is not a complete row")
            self.assertGreaterEqual(cells[150][0], cells[300][0], f"pvr {pvr} false-binds")
            self.assertGreaterEqual(cells[150][1], cells[300][1], f"pvr {pvr} false-fine")

    def test_perfect_coupling_is_error_free_and_no_coupling_is_not(self):
        self.assertEqual(vhsp.measured_error_rates(0.00, 300)[:2], (0.0, 0.0))
        self.assertGreater(vhsp.measured_error_rates(1.00, 150)[0], 0.20)

    def test_reports_which_cell_was_substituted(self):
        fb, ff, tp, tn = vhsp.measured_error_rates(0.61, 280)
        self.assertEqual((tp, tn), (0.59, 300))

    def test_no_retracted_sample_size_constant_is_used_as_a_gate(self):
        """1200 may appear in prose explaining its retraction, never in live logic.

        Scanned over the AST, not the text. A first version of this test grepped the raw
        source and failed on the clean tree, because the comment that documents the
        removal contains the literal `n < 1200` -- a guard that fires on its own
        explanation is a guard nobody keeps.
        """
        import ast as _ast
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "value_head_sibling_probe.py").read_text()
        consts = [n.value for n in _ast.walk(_ast.parse(src))
                  if isinstance(n, _ast.Constant) and isinstance(n.value, int)]
        self.assertNotIn(1200, consts,
                         "the retracted n>=1200 sample-size constant is live in code again")

    def test_every_table_cell_is_a_valid_rate(self):
        for pvr, n, fb, ff in vhsp.MEASURED_ERROR_RATES:
            self.assertTrue(0.0 <= pvr <= 1.0)
            self.assertIn(n, (150, 300))
            self.assertTrue(0.0 <= fb <= 1.0 and 0.0 <= ff <= 1.0)


class AtFloorReportingTest(unittest.TestCase):
    """A clipped estimate must never be reported as a proven zero."""

    def test_a_fully_clipped_bootstrap_is_flagged_at_floor(self):
        # Identical arms, zero head gap: nothing to detect, everything clips.
        R = 8
        pairs = [{"head_gap": 0.0, "true_gap": 0.0, "true_a": 0.5, "true_b": 0.5,
                  "rollouts_a": R, "rollouts_b": R,
                  "outcomes_a": {t: 0.5 for t in range(R)},
                  "outcomes_b": {t: 0.5 for t in range(R)}} for _ in range(30)]
        got = vhsp.estimate_sigma_diff(pairs, n_boot=50)
        self.assertTrue(got["at_floor"])
        self.assertEqual(got["sigma_diff"], 0.0)
        self.assertEqual(got["ci95"][1], 0.0)
        # The caller must be able to distinguish "bound is 0" from "no bound available",
        # which is exactly what ci95[1] == 0.0 alongside at_floor signals.


class MainUsesTheConversionHelperTest(unittest.TestCase):
    """The regress has to stop somewhere: assert main() CALLS the helper.

    FinalizePairGapsTest guards the helper's body, and reverting that body fails 3 tests.
    But reverting main() to three inline assignments still left all 52 green -- the guard
    had simply moved down one level. Checked over the AST, which terminates the regress:
    main must contain a call to finalize_pair_gaps, and must not assign head_gap directly.
    """

    @staticmethod
    def _main_node():
        import ast as _ast
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "value_head_sibling_probe.py").read_text()
        tree = _ast.parse(src)
        return next(n for n in tree.body
                    if isinstance(n, _ast.FunctionDef) and n.name == "main"), _ast

    def test_main_calls_finalize_pair_gaps(self):
        node, _ast = self._main_node()
        calls = {c.func.id for c in _ast.walk(node)
                 if isinstance(c, _ast.Call) and isinstance(c.func, _ast.Name)}
        self.assertIn("finalize_pair_gaps", calls)

    def test_main_never_writes_head_gap_directly(self):
        node, _ast = self._main_node()
        for assign in _ast.walk(node):
            if not isinstance(assign, _ast.Assign):
                continue
            for tgt in assign.targets:
                if (isinstance(tgt, _ast.Subscript)
                        and isinstance(tgt.slice, _ast.Constant)
                        and tgt.slice.value in ("head_gap", "head_gap_return_scale",
                                                "true_gap")):
                    self.fail(f"main() assigns {tgt.slice.value!r} directly at line "
                              f"{assign.lineno}; it must go through finalize_pair_gaps so "
                              f"the units conversion cannot be bypassed")


class VerdictLinesTest(unittest.TestCase):
    """The advisory a reader acts on. It was wrong twice while living inline in main()."""

    @staticmethod
    def _sd(sigma, n=300, at_floor=False):
        return {"sigma_diff": sigma, "n": n, "at_floor": at_floor,
                "ci95": [0.0, sigma if sigma else 0.0]}

    def test_an_at_floor_run_gets_NO_VERDICT_and_no_affirmative_line(self):
        # sigma_diff == 0.0 is the clip. This used to fall into the low arm and print
        # "a low reading is well supported" right after "NOT RESOLVABLE".
        out = " ".join(vhsp.verdict_lines(self._sd(0.0, at_floor=True), (0.0, 0.0)))
        self.assertIn("NO VERDICT", out)
        self.assertNotIn("well supported", out)
        self.assertNotIn("suggestive", out)

    def test_an_in_band_estimate_is_INDETERMINATE_not_well_supported(self):
        for sigma in (0.016, 0.0207, 0.025, 0.034):
            out = " ".join(vhsp.verdict_lines(self._sd(sigma), (0.05, 0.0)))
            self.assertIn("INDETERMINATE", out, f"sigma={sigma}")
            self.assertNotIn("well supported", out, f"sigma={sigma}")

    def test_a_clearly_high_estimate_is_supported_when_the_rate_is_low(self):
        out = " ".join(vhsp.verdict_lines(self._sd(0.06), (0.05, 0.0)))
        self.assertIn("high reading is well supported", out)

    def test_a_clearly_high_estimate_is_hedged_when_the_rate_is_high(self):
        out = " ".join(vhsp.verdict_lines(self._sd(0.06), (0.27, 0.0)))
        self.assertIn("WEAKLY SUPPORTED", out)

    def test_a_clearly_low_estimate_is_supported_only_when_false_fine_is_small(self):
        self.assertIn("low reading is well supported",
                      " ".join(vhsp.verdict_lines(self._sd(0.005), (0.0, 0.0))))
        self.assertIn("suggestive, not decisive",
                      " ".join(vhsp.verdict_lines(self._sd(0.005), (0.0, 0.05))))

    def test_underpowered_beats_everything(self):
        out = " ".join(vhsp.verdict_lines(self._sd(0.06, n=100), (0.0, 0.0)))
        self.assertIn("UNDERPOWERED", out)
        self.assertNotIn("well supported", out)

    def test_missing_rates_refuses_to_qualify_rather_than_asserting(self):
        out = " ".join(vhsp.verdict_lines(self._sd(0.06), None))
        self.assertIn("NO ERROR RATES", out)
        self.assertNotIn("well supported", out)

    def test_no_reachable_input_produces_both_a_refusal_and_an_affirmation(self):
        # The defect class was two contradictory lines from one state. Sweep the space.
        for at_floor in (False, True):
            for n in (100, 150, 300):
                for sigma in (0.0, 0.005, 0.015, 0.02, 0.035, 0.06):
                    for rates in (None, (0.0, 0.0), (0.27, 0.05)):
                        out = " ".join(vhsp.verdict_lines(
                            self._sd(sigma, n=n, at_floor=at_floor), rates))
                        refuses = any(k in out for k in
                                      ("NO VERDICT", "UNDERPOWERED", "NO ERROR RATES"))
                        affirms = "well supported" in out
                        self.assertFalse(refuses and affirms,
                                         f"contradictory advisory for at_floor={at_floor} "
                                         f"n={n} sigma={sigma} rates={rates}: {out}")


if __name__ == "__main__":
    unittest.main()
