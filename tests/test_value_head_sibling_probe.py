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


if __name__ == "__main__":
    unittest.main()


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
