"""The CROSS-ARM contention gate: was the time-budgeted opponent equally strong against both?

`compare_foulplay_think` (the within-arm gate) answers whether two readings MAY be compared.
It reports a ratio and deliberately never judges it, and its only shipping caller runs it on
p1 against p2 of ONE arm -- where both seats are equally starved, so within-arm symmetry
cannot see the confound that matters. This file pins the caller that compares BETWEEN arms and
refuses: the oracle-leaf arm spends measurably more CPU per decision than the raw arm (1.8x at
R=8, 4.3x at R=32), and a time-budgeted opponent given less effective compute against the
hungrier arm hands that arm a weaker opponent in the flattering direction.

Every check here ships with the input on which it reads False, per the program-wide rule, and
the fixtures are built from RATES AND COUNTS so nothing under test is pre-supplied: `think()`
hands over no `status`, no ratio and no coverage -- the gate derives all three, so deleting the
derivation cannot leave this file green.

Pure fixtures plus two measured constants, both named where they are used. No cluster, no
checkpoint, no crate.
"""

from __future__ import annotations

import json
import math
import unittest

from pokezero.foulplay_bridge import (
    FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS,
    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
    FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE,
    FOULPLAY_THINK_MEASURED_DECISION_LOG_SD,
    FOULPLAY_THINK_MEASURED_DECISION_LOG_SD_DF,
    FOULPLAY_THINK_MEASURED_POSITION_LOG_SD,
    FOULPLAY_THINK_MEASURED_RUN_LOG_SD,
    FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF,
    FOULPLAY_THINK_MIN_MEASURED_DECISIONS,
    FOULPLAY_THINK_MIN_STRATUM_DECISIONS,
    FOULPLAY_THINK_SCHEMA_VERSION,
    cross_arm_foulplay_contention,
    foulplay_think_nominal_fold_resolution,
    pool_foulplay_think,
)

# --- the measurement this gate's threshold was derived from -----------------------------
# `crossarm-contention-dispersion.py` (kept outside the repo), driving foul-play's own
# `get_result_from_mcts` over poke_engine under `fork`, foul-play's own `init_logging`,
# captured off fd 1 and parsed by the SHIPPING parser. SIX passes x 24 shared positions,
# 144/144 decisions measured, zero miss reasons, all in `2x1000ms`. These are the six passes'
# ARITHMETIC per-stratum means -- the gate's own statistic -- in run order. Passes 3 and 4 ran
# ~10% slower for their whole duration while the box was busier, which is the effect a
# cross-arm gate must not read as contention.
MEASURED_PASS_MEANS = (385_187.5, 382_166.7, 387_916.7, 347_291.7, 349_229.2, 382_666.7)
MEASURED_PASS_N = 24
#: A campaign's per-stratum n, for the controls that need the gate to actually certify. 24 is
#: the n the MEASUREMENT had, and the resolution gate correctly says 24 decisions cannot resolve
#: a 1.25 threshold (the crossover is ~27). The measured quantity being reused at campaign n is
#: the RATE, which is what was measured; n is a property of the run being judged. Eras 61-64
#: measured 48.7 foul-play decisions per game, so a few hundred games puts thousands in each
#: stratum and 1,000 is conservative.
CAMPAIGN_N = 1_000
# The largest of the 15 pairwise folds among those six matched "arms".
MEASURED_MAX_MATCHED_FOLD = 1.1170
# The starvation the instrument caught behind a thin stratum, per the instrument's own PR.
MEASURED_STARVATION_FOLD = 3.8


def think(
    rate=380_000.0,
    decisions=120,
    *,
    strata=None,
    record_failures=0,
    observable=True,
    start_method="fork",
    schema_version=None,
):
    """One seat's `foulplay_think` run header, in the producer's shape.

    Inputs only. `iterations_coverage`, `mean_iterations_per_budget_second` and the totals are
    computed here the way the producer computes them, from `strata` and `decisions`, so a
    fixture cannot hand the gate the answer it is supposed to derive.
    """
    # `is None`, not truthiness: `strata={}` is the "measured nothing" fixture and must not
    # fall back to the healthy default.
    strata = {"2x1000ms": (rate, decisions)} if strata is None else strata
    measured = sum(n for _, n in strata.values())
    attempted = decisions + record_failures
    return {
        "schema_version": schema_version or FOULPLAY_THINK_SCHEMA_VERSION,
        "decisions": decisions,
        "decisions_attempted": attempted,
        "record_failures": record_failures,
        "iterations_measured_decisions": measured,
        # `int()` only when the arithmetic is finite: the non-finite-rate fixtures below are
        # deliberately impossible shards, and the fixture must be able to BUILD one.
        "total_iterations": (
            int(total)
            if math.isfinite(total := sum(r * 2.0 * n for r, n in strata.values()))
            else total
        ),
        "mean_iterations_per_budget_second": (
            sum(r * n for r, n in strata.values()) / measured if measured else None
        ),
        "iterations_coverage": (measured / attempted) if attempted else None,
        "iterations_observable": observable,
        "opponent_start_method": start_method,
        "by_stratum": {
            name: {
                "iterations_measured_decisions": n,
                "mean_iterations_per_budget_second": r,
            }
            for name, (r, n) in sorted(strata.items())
        },
        "miss_decisions": max(0, decisions - measured),
    }


def verdict(search_headers, raw_headers, **kw):
    return cross_arm_foulplay_contention(
        pool_foulplay_think(search_headers, label="search"),
        pool_foulplay_think(raw_headers, label="raw"),
        **kw,
    )


def numbers_in(payload):
    """Every float in the payload, so a withheld ratio can be shown to be absent."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            found.append(node)

    walk(payload)
    return found


# --- t and chi-square tails, so the df claims are COMPUTED and not quoted ------------------
# Stdlib only, same rule as the rest of this file: no scipy in this project's dependencies, and
# a hardcoded quantile is exactly the kind of number these tests exist to stop being asserted.
# Both are checked against textbook values in `RunTermDegreesOfFreedomTest` before being used.


def _betacf(a, b, x):
    tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((a - 1.0 + m2) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2)),
        ):
            d = 1.0 + aa * d
            d = d if abs(d) > tiny else tiny
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        lbeta + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_tail(t, df):
    """P(|T_df| > t)."""
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def student_t_two_sided_quantile(p, df):
    """The t with `student_t_two_sided_tail(t, df) == p`, by bisection."""
    lo, hi = 0.0, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_tail(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gammainc_lower(s, x):
    """Regularized lower incomplete gamma P(s, x)."""
    if x <= 0.0:
        return 0.0
    if x < s + 1.0:
        term = total = 1.0 / s
        for k in range(1, 1000):
            term *= x / (s + k)
            total += term
            if abs(term) < abs(total) * 1e-17:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    tiny, b, c = 1e-300, x + 1.0 - s, 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def chi_square_quantile(p, df):
    """The x with `P(X^2_df <= x) == p`, by bisection."""
    lo, hi = 0.0, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc_lower(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fold_log_sd(n_a, n_b):
    """The SD `foulplay_think_nominal_fold_resolution` exponentiates three of."""
    return math.sqrt(
        2.0 * FOULPLAY_THINK_MEASURED_RUN_LOG_SD**2
        + FOULPLAY_THINK_MEASURED_DECISION_LOG_SD**2 * (1.0 / n_a + 1.0 / n_b)
    )


def satterthwaite_df(n_a, n_b):
    """Effective df of that SD: a 5-df term plus a 115-df term, in known proportions."""
    v_run = 2.0 * FOULPLAY_THINK_MEASURED_RUN_LOG_SD**2
    v_dec = FOULPLAY_THINK_MEASURED_DECISION_LOG_SD**2 * (1.0 / n_a + 1.0 / n_b)
    return (v_run + v_dec) ** 2 / (
        v_run**2 / FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF
        + v_dec**2 / FOULPLAY_THINK_MEASURED_DECISION_LOG_SD_DF
    )


class MatchedArmsTest(unittest.TestCase):
    """The negative control, and it is what decides the threshold.

    A gate like this one can MANUFACTURE the defect it detects -- the instrument's own give-up
    heuristic did exactly that once, producing a contention finding out of a healthy opponent.
    So the first obligation is measured pairs of arms that were NOT contending, and there are
    fifteen of them here rather than one.
    """

    def _pairs(self, n, **kw):
        """Every one of the 15 pairs of the six measured passes, as two arms."""
        out = []
        for i, a in enumerate(MEASURED_PASS_MEANS):
            for b in MEASURED_PASS_MEANS[i + 1 :]:
                out.append((a, b, verdict([think(a, n)], [think(b, n)], **kw)))
        return out

    def test_every_measured_matched_pair_of_real_foulplay_passes(self) -> None:
        """All 15 pairs of six uncontended real passes, including the two slow ones.

        This is the test a single lucky pair could not be: passes 3 and 4 ran ~10% slower for
        their whole duration, so eight of the fifteen pairs straddle that shift and are the
        realistic matched-arm case rather than the best case. Run at campaign n, because at the
        measurement's own n=24 the gate correctly refuses to certify anything at all (below).
        """
        folds = []
        for a, b, result in self._pairs(CAMPAIGN_N):
            with self.subTest(a=a, b=b):
                self.assertEqual(result["refusal_reasons"], [])
                self.assertEqual(result["status"], "ok")
            folds.append(result["worst_stratum"]["fold_ratio"])
        self.assertEqual(len(folds), 15)
        self.assertAlmostEqual(max(folds), MEASURED_MAX_MATCHED_FOLD, places=3)
        # Bimodal, and both modes matter: seven pairs inside one regime, eight across the shift.
        self.assertEqual(sum(1 for f in folds if f > 1.05), 8)

    def test_the_measurements_own_n_is_too_thin_to_certify_anything(self) -> None:
        """The gate's honesty about its own evidence, and it is not a formality.

        The threshold was derived from passes of 24 decisions each, and 24 decisions cannot
        resolve a 1.25 fold: the nominal bound there is 1.2508. So the gate certifies nothing at
        that n -- including on the matched pairs it was derived from -- and the reason is
        THINNESS, never contention. A gate that certified at the n of its own calibration would
        be certifying noise.

        The MECHANISM is exclusion, not refusal (see the dead-band test below): the only stratum
        is dropped from the compared set, which takes the compared share to 0.0, and the share
        floor is what refuses. Both halves are asserted, because "refused" alone would also be
        satisfied by the version of this gate that voided matched arms over an 8-decision
        stratum.

        AND THE REASON SAYS WHICH SHORTFALL IT WAS. 100% of this arm was excluded for resolution
        and 0% is a coverage gap, so the reason is the resolution one and the plain coverage one
        must NOT fire -- otherwise "this stratum cannot resolve the threshold" is reported as
        "the two arms visited different schedules", which is the objection the refusing branch of
        this fix raised against a bare exclusion.
        """
        for a, b, result in self._pairs(MEASURED_PASS_N):
            with self.subTest(a=a, b=b):
                self.assertEqual(result["status"], "refused")
                self.assertEqual(
                    result["strata_excluded_for_resolution"]["2x1000ms"][
                        "search_iterations_measured_decisions"
                    ],
                    MEASURED_PASS_N,
                )
                self.assertGreater(
                    result["strata_excluded_for_resolution"]["2x1000ms"][
                        "nominal_fold_resolution_point_estimate"
                    ],
                    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
                )
                self.assertEqual(result["cross_arm_compared_share"], {"search": 0.0, "raw": 0.0})
                self.assertEqual(
                    result["cross_arm_share_excluded_for_resolution"],
                    {"search": 1.0, "raw": 1.0},
                )
                self.assertIn(
                    "search:cross_arm_strata_excluded_for_resolution_cover_too_little",
                    result["refusal_reasons"],
                )
                self.assertNotIn(
                    "search:cross_arm_compared_strata_cover_too_little",
                    result["refusal_reasons"],
                )
                self.assertNotIn(
                    "cross_arm_rate_ratio_exceeds_threshold", result["refusal_reasons"]
                )
                self.assertIn("rates_withheld_because", result)

    def test_a_threshold_below_the_nominal_asymptote_certifies_nothing_at_any_n(self) -> None:
        """The behaviour, WITHOUT the retracted claim that used to be attached to it.

        The whole-run term never shrinks, so the nominal z=3 resolution never goes below
        exp(3*sqrt(2)*run_sd) = 1.2447 however large n gets. A threshold under that certifies
        nothing at ANY denominator, and the gate behaves accordingly: at 1.24 every one of the 15
        matched pairs ends up with its only stratum excluded and refuses -- for RESOLUTION, which
        is the reason that is true, and not for a coverage gap -- even at 10^6 decisions per arm.

        WHAT IS NOT CLAIMED, and used to be: that 1.2448 is therefore "the tightest threshold
        this instrument can support". 1.2447 is a point estimate of a quantity with 5 degrees of
        freedom quoted to five digits -- `test_the_run_terms_five_df_does_not_support_the_floor`
        below puts the same floor at 1.47-1.49 through t and near 1.58 at the run SD's own 95%
        upper bound. This test pins the gate's OPERATING criterion, which is a nominal z=3 rule
        and has to be some fixed rule; it does not pin a physical constant.
        """
        floor = math.exp(3.0 * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD)
        self.assertAlmostEqual(floor, 1.24473, places=5)
        self.assertGreater(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, floor)
        for a, b, result in self._pairs(10**6, max_fold_ratio=1.24):
            with self.subTest(a=a, b=b):
                self.assertEqual(result["status"], "refused")
                self.assertIn("2x1000ms", result["strata_excluded_for_resolution"])
                self.assertIn(
                    "search:cross_arm_strata_excluded_for_resolution_cover_too_little",
                    result["refusal_reasons"],
                )
                self.assertNotIn(
                    "search:cross_arm_compared_strata_cover_too_little",
                    result["refusal_reasons"],
                )
        # And at the shipped threshold the same arms certify.
        for a, b, result in self._pairs(10**6):
            with self.subTest(a=a, b=b):
                self.assertEqual(result["refusal_reasons"], [])
                self.assertEqual(result["strata_excluded_for_resolution"], {})

    def test_a_threshold_below_the_measured_spread_would_call_matched_arms_contended(self) -> None:
        """The failing input for the CHOICE of threshold, separated from the floor above.

        Holding the resolution gate out of the way (by asking only whether the CONTENTION
        refusal fires), a limit at 1.10 calls six of fifteen pairs of uncontended real searches
        contended and 1.05 calls eight. Those are the numbers a "more vigilant" threshold buys.
        """
        for limit, expected in ((1.10, 6), (1.05, 8)):
            refused = sum(
                1
                for a, b, _ in self._pairs(CAMPAIGN_N)
                if max(a / b, b / a) > limit
            )
            with self.subTest(limit=limit):
                self.assertEqual(refused, expected)
                # And the gate agrees, once its own floor is not the binding reason.
                folds = [
                    result["worst_stratum"]["fold_ratio"]
                    for _, _, result in self._pairs(CAMPAIGN_N)
                ]
                self.assertEqual(sum(1 for f in folds if f > limit), expected)

    def test_the_threshold_clears_the_measured_matched_spread_and_catches_the_real_thing(self) -> None:
        """The constant, pinned between the two numbers it sits between.

        The margin is quoted as LOG-margin, because the gated quantity is a fold ratio and its
        noise is multiplicative -- and because the linear form is the shape of a claim this
        constant's docstring already had to strike once ("1 + 3.1x the observed excess"). What it
        says is `ln(1.25)/ln(1.117) = 2.02`: a margin on ONE observed maximum over six passes. It
        is not a sigma and not a probability; see `RunTermDegreesOfFreedomTest`.
        """
        self.assertGreater(
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, MEASURED_MAX_MATCHED_FOLD
        )
        self.assertLess(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, MEASURED_STARVATION_FOLD)
        self.assertGreater(
            math.log(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO)
            / math.log(MEASURED_MAX_MATCHED_FOLD),
            2.0,
        )

    def test_the_bound_is_nearly_flat_in_n_which_is_what_a_fixed_threshold_rests_on(self) -> None:
        """And this is the claim the fixed threshold rests on -- flatness, not a sigma count.

        If the bound moved materially with n, a fixed threshold would be wrong -- tight at small
        n, loose at large n. The whole-run term dominates, so it does not: 1.2723 at the
        per-stratum floor against 1.2454 at n=200, a 2.7-point range over a 40x change in n.

        The sigma multiples below are recorded because they are what the bound is BUILT from, and
        they are NOMINAL z's on a variance with 5 degrees of freedom -- not a statement that this
        is a 3-sigma test. `RunTermDegreesOfFreedomTest` is where that distinction is pinned.
        """
        floor = FOULPLAY_THINK_MIN_STRATUM_DECISIONS
        self.assertAlmostEqual(
            foulplay_think_nominal_fold_resolution(floor, floor), 1.2723, places=4
        )
        self.assertAlmostEqual(foulplay_think_nominal_fold_resolution(20, 20), 1.2518, places=4)
        self.assertAlmostEqual(foulplay_think_nominal_fold_resolution(200, 200), 1.2454, places=4)
        for n in (floor, 20, 24, 200, 2000):
            with self.subTest(n=n):
                sd = math.log(foulplay_think_nominal_fold_resolution(n, n)) / 3.0
                sigmas = math.log(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO) / sd
                self.assertGreater(sigmas, 2.7)
                self.assertLess(sigmas, 3.2)

    def test_the_detectable_bound_includes_the_whole_run_term(self) -> None:
        """The flattering error a two-pass predecessor of this measurement shipped.

        Sampling alone puts the n=200 floor at 1.035 and reads as "this stratum could have
        resolved a 3.5% starvation". Measured across six passes the whole-run term alone puts it
        at 1.246. Dropping the run term from the bound is the mutation this test catches, and it
        is the one that matters: the field's entire job is to say how strongly a stratum passed.
        """
        sampling_only = math.exp(
            3.0 * FOULPLAY_THINK_MEASURED_DECISION_LOG_SD * (2.0 / 200) ** 0.5
        )
        self.assertAlmostEqual(sampling_only, 1.0160, places=4)
        self.assertGreater(foulplay_think_nominal_fold_resolution(200, 200), 1.2)
        self.assertGreater(FOULPLAY_THINK_MEASURED_RUN_LOG_SD, 0.0)
        # And at large n the bound converges on the run term alone, not on 1.0.
        self.assertAlmostEqual(
            foulplay_think_nominal_fold_resolution(10**7, 10**7),
            math.exp(3.0 * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD),
            places=4,
        )

    def _mixed_pairs(self, major, minor, **kw):
        """The same 15 measured pairs, but with a REALISTIC stratum mix on each arm.

        Every other matched-arms fixture in this file gives each arm one fat stratum, which is
        the shape that hid the dead band: `by_stratum` is per-DECISION -- foul-play recomputes
        `(num_battles, search_time)` every decision -- so a real arm has one dominant schedule
        and a tail of rare ones. The minor stratum's rate is half the major's, which is what the
        `8x500ms`/`2x1000ms` pair actually looks like (60,000 visits/sample at 500 ms against
        240,000 at 1000 ms).
        """
        out = []
        for i, a in enumerate(MEASURED_PASS_MEANS):
            for b in MEASURED_PASS_MEANS[i + 1 :]:
                arms = [
                    think(
                        decisions=major + minor,
                        strata={"2x1000ms": (rate, major), "8x500ms": (rate / 2.0, minor)},
                    )
                    for rate in (a, b)
                ]
                out.append((a, b, verdict([arms[0]], [arms[1]], **kw)))
        return out

    def test_matched_arms_with_a_realistic_stratum_mix_still_certify(self) -> None:
        """THE DEAD BAND, and the fixture whose absence hid it.

        Two arms that are matched in every stratum, differing from the single-fat-stratum
        fixtures above only by carrying one rare schedule: 1,000 decisions at `2x1000ms` and 10
        at `8x500ms`, i.e. 1.0% of each arm. 10 decisions cannot resolve 1.25 (nominal bound
        1.2588, against a crossover of 27), so that stratum certifies nothing -- and while an
        unresolvable stratum REFUSED, all 15 of these pairs came back `refused` with a reason
        that reads as an instrument fault, `winner=None`, and no adoption for the campaign. The
        flaw was 1% of the arm.

        Excluded instead, the share floor decides and reads 0.990: the arm is certified on the
        99% of it that can carry a verdict, the 1% is named in
        `strata_excluded_for_resolution`, and the uncovered remainder stays inside the 5% the
        floor allows.
        """
        self.assertGreater(
            foulplay_think_nominal_fold_resolution(10, 10),
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
        )
        for a, b, result in self._mixed_pairs(CAMPAIGN_N, 10):
            with self.subTest(a=a, b=b):
                self.assertEqual(result["refusal_reasons"], [])
                self.assertEqual(result["status"], "ok")
                # The minor stratum is excluded, named, and subtracted from the share.
                self.assertEqual(
                    sorted(result["strata_excluded_for_resolution"]), ["8x500ms"]
                )
                self.assertEqual(sorted(result["by_stratum"]), ["2x1000ms"])
                self.assertAlmostEqual(
                    result["cross_arm_compared_share"]["search"], 1000 / 1010, places=6
                )
                self.assertEqual(result["worst_stratum"]["stratum"], "2x1000ms")

    def test_the_share_floor_and_not_the_resolution_decides_how_much_may_be_excluded(
        self,
    ) -> None:
        """Exclusion is not leniency: the excluded slice still has to be small.

        The same shape at three sizes, on the WORST measured matched pair. 0.5% and 1.0% of the
        arm pass; 10% does not, because what was excluded counts as uncovered and 0.90 is under
        `FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE`. So nothing is ever certified ON a stratum
        that cannot resolve the threshold, and the unbounded remainder stays capped at a
        twentieth of each arm -- both guarantees the refusal was there for.
        """
        for major, minor, share, expected in (
            (995, 5, 0.995, "ok"),
            (1000, 10, 1000 / 1010, "ok"),
            (234, 26, 0.9, "refused"),
        ):
            with self.subTest(minor=minor):
                self.assertGreater(
                    foulplay_think_nominal_fold_resolution(minor, minor),
                    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
                )
                self.assertLessEqual(
                    foulplay_think_nominal_fold_resolution(major, major),
                    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
                )
                worst = self._mixed_pairs(major, minor)[-1]
                result = worst[2]
                self.assertEqual(result["status"], expected)
                self.assertAlmostEqual(
                    result["cross_arm_compared_share"]["search"], share, places=6
                )
                self.assertIn("8x500ms", result["strata_excluded_for_resolution"])
                self.assertAlmostEqual(
                    result["cross_arm_share_excluded_for_resolution"]["search"],
                    minor / (major + minor),
                    places=6,
                )
                if expected == "refused":
                    # THE RESOLUTION REASON, not the plain coverage one: every decision outside
                    # the compared set here is inside an excluded stratum, so a bare
                    # "cover_too_little" would be the wrong diagnosis with the wrong remedy.
                    self.assertIn(
                        "search:cross_arm_strata_excluded_for_resolution_cover_too_little",
                        result["refusal_reasons"],
                    )
                    self.assertNotIn(
                        "search:cross_arm_compared_strata_cover_too_little",
                        result["refusal_reasons"],
                    )
                    # And NOT for contention: these arms are matched.
                    self.assertNotIn(
                        "cross_arm_rate_ratio_exceeds_threshold", result["refusal_reasons"]
                    )

    def test_an_excluded_stratum_cannot_hide_a_starvation_inside_the_allowance(self) -> None:
        """The direction the exclusion must NOT open up.

        A 1%-of-the-arm stratum where the opponent realized 3.8x less work against the hungry arm
        is excluded (10 decisions cannot resolve 1.25) and therefore not refused for contention.
        That is the same guarantee the share floor gives everywhere else -- the excluded slice is
        UNBOUNDED, capped only in size -- and it is pinned here so the allowance is a known
        remainder rather than a discovery. What must hold: the cell still passes, and the payload
        names the excluded stratum, so the remainder is legible in the artifact.
        """
        search = think(
            decisions=1010,
            strata={"2x1000ms": (380_000.0, 1000), "8x500ms": (190_000.0 / 3.8, 10)},
        )
        raw = think(
            decisions=1010,
            strata={"2x1000ms": (380_000.0, 1000), "8x500ms": (190_000.0, 10)},
        )
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "ok")
        self.assertIn("8x500ms", result["strata_excluded_for_resolution"])
        # No rate escapes with it: only denominators and the resolution.
        self.assertEqual(
            sorted(result["strata_excluded_for_resolution"]["8x500ms"]),
            [
                "nominal_fold_resolution_point_estimate",
                "raw_iterations_measured_decisions",
                "search_iterations_measured_decisions",
            ],
        )
        # Which is why 1% is the most that can hide there: at 10% the share floor refuses.
        big = verdict(
            [
                think(
                    decisions=260,
                    strata={"2x1000ms": (380_000.0, 234), "8x500ms": (50_000.0, 26)},
                )
            ],
            [
                think(
                    decisions=260,
                    strata={"2x1000ms": (380_000.0, 234), "8x500ms": (190_000.0, 26)},
                )
            ],
        )
        self.assertEqual(big["status"], "refused")
        self.assertIn(
            "search:cross_arm_strata_excluded_for_resolution_cover_too_little",
            big["refusal_reasons"],
        )

    def test_a_pure_decision_mix_difference_is_not_refused(self) -> None:
        """The strata exist because an unstratified comparison INVENTS an effect.

        Identical per-stratum rates, different mixes. The unstratified ratio is far from 1
        and the gate must still pass: gating on it would refuse every pair of arms whose
        games ran to different lengths.
        """
        early, late = 120_000.0, 240_000.0
        search = think(decisions=200, strata={"8x500ms": (early, 160), "2x1000ms": (late, 40)})
        raw = think(decisions=200, strata={"8x500ms": (early, 40), "2x1000ms": (late, 160)})
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "ok")
        for block in result["by_stratum"].values():
            self.assertAlmostEqual(block["fold_ratio"], 1.0)
        self.assertGreater(result["unstratified_ratio_lean_over_hungry"], 1.4)


class RunTermDegreesOfFreedomTest(unittest.TestCase):
    """The run term has FIVE degrees of freedom, and this class is what that costs.

    Six passes is enough evidence for a fixed threshold of about 1.25. It is not enough for the
    two claims a previous version of the constant's docstring made from it -- a floor quoted to
    five digits, and a stated false-refusal rate -- and neither of those was wrong by arithmetic:
    the decomposition reproduces exactly. They were wrong by INFERENCE, because both read a 5-df
    variance as if the number were known. Each retraction here ships with the computation that
    refutes it, per this file's own rule about failing inputs, and the quantiles are computed
    rather than quoted.
    """

    def test_the_t_and_chi_square_helpers_agree_with_textbook_values(self) -> None:
        """The instrument that measures the other tests, measured first.

        A hand-rolled incomplete beta that is subtly wrong would make every retraction below
        look proven. Four published quantiles, two per distribution.
        """
        self.assertAlmostEqual(student_t_two_sided_quantile(0.01, 5), 4.0321, places=4)
        self.assertAlmostEqual(student_t_two_sided_quantile(0.05, 10), 2.2281, places=4)
        self.assertAlmostEqual(chi_square_quantile(0.05, 5), 1.1455, places=4)
        self.assertAlmostEqual(chi_square_quantile(0.95, 5), 11.0705, places=4)
        # And the two-sided tail inverts its own quantile.
        self.assertAlmostEqual(student_t_two_sided_tail(4.0321, 5), 0.01, places=6)

    def test_the_effective_df_of_the_gates_own_sd_is_about_five(self) -> None:
        """Not 115, and not 120: the term that dominates is the one with 5 df.

        Satterthwaite on `2*run^2 + decision^2*(1/n_a + 1/n_b)`. At the n's the gate actually
        certifies at the run term is ~96% of the variance, so the effective df sits just above 5
        and falls toward exactly 5 as n grows -- which is the opposite of the intuition that a
        thousand decisions per stratum buys precision on the threshold.
        """
        self.assertEqual(FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF, 5)
        self.assertEqual(FOULPLAY_THINK_MEASURED_DECISION_LOG_SD_DF, 115)
        self.assertAlmostEqual(
            satterthwaite_df(
                FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS,
                FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS,
            ),
            5.396,
            places=3,
        )
        self.assertAlmostEqual(satterthwaite_df(200, 200), 5.053, places=3)
        self.assertAlmostEqual(satterthwaite_df(10**6, 10**6), 5.000, places=3)
        # Monotone toward 5 from above, never approaching the 115.
        for n in (27, 200, 1000, 10**6):
            with self.subTest(n=n):
                self.assertGreater(satterthwaite_df(n, n), 5.0)
                self.assertLess(satterthwaite_df(n, n), 5.5)

    def test_the_run_terms_five_df_does_not_support_the_floor_that_was_claimed(self) -> None:
        """RETRACTED: "no n makes the resolution better than 1.2448, so 1.25 is the tightest
        threshold this instrument can support".

        The arithmetic is right -- exp(3*sqrt(2)*0.0516) = 1.24473 -- and the inference is not.
        `3` is a z, applied to a variance with 5 df. Matching the same two-sided tail (0.0027)
        through t needs 5.23 to 5.51 across the n's that certify, which puts the floor at
        1.47-1.49; and the run SD's own 95% chi-square upper bound is 0.1078, which puts it near
        1.58. 1.25 is BELOW all of those, so the honest statement is that the threshold sits
        inside the instrument's uncertainty about its own resolution, not that it is that
        resolution rounded up.
        """
        nominal = math.exp(3.0 * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD)
        self.assertAlmostEqual(nominal, 1.24473, places=5)

        three_sigma_tail = math.erfc(3.0 / (2**0.5))
        self.assertAlmostEqual(three_sigma_tail, 0.0026998, places=7)

        floors = []
        for n in (FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS, 200, 10**6):
            t = student_t_two_sided_quantile(three_sigma_tail, satterthwaite_df(n, n))
            with self.subTest(n=n):
                self.assertGreater(t, 5.2)
                self.assertLess(t, 5.6)
            floors.append(math.exp(t * fold_log_sd(n, n)))
        self.assertAlmostEqual(min(floors), 1.475, places=3)
        self.assertAlmostEqual(max(floors), 1.495, places=3)

        # The variance's own 95% upper bound, which is the second reason the five digits are not
        # available: chi-square on 5 df puts the run SD above 0.107, more than double.
        upper_run_sd = FOULPLAY_THINK_MEASURED_RUN_LOG_SD * math.sqrt(
            FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF
            / chi_square_quantile(0.05, FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF)
        )
        self.assertAlmostEqual(upper_run_sd, 0.1078, places=4)
        self.assertAlmostEqual(math.exp(3.0 * (2**0.5) * upper_run_sd), 1.580, places=3)

        # AND THE THRESHOLD IS STILL 1.25. Every one of those floors is above it, so no
        # correction here argues for a different constant -- only for a different claim.
        self.assertLess(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, min(floors))

    def test_no_false_refusal_rate_is_available_from_six_passes(self) -> None:
        """RETRACTED: "refuses a genuinely matched pair on the order of once in 300 strata".

        That is a Gaussian tail computed from a 5-df variance: 1.25 is 3.06 nominal sigma out and
        the normal two-sided tail there is 1 in 449, which the previous text rounded to "on the
        order of 300". Carried through t at the same df it is 1 in 35.5 at the asymptote and 1 in
        36.7 at the calibration's n=24, both two-sided -- an order of
        magnitude worse, in the direction that matters -- and the DATA bound it only at 18.1%,
        the exact one-sided 95% upper bound for 0 refusals in 15 matched pairs. All three numbers
        are computed here, because the point is the spread between them.
        """
        sigmas = math.log(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO) / fold_log_sd(10**6, 10**6)
        self.assertAlmostEqual(sigmas, 3.058, places=3)

        gaussian = math.erfc(sigmas / (2**0.5))
        self.assertAlmostEqual(1.0 / gaussian, 449.0, delta=1.0)

        with_t = student_t_two_sided_tail(sigmas, satterthwaite_df(10**6, 10**6))
        self.assertAlmostEqual(1.0 / with_t, 35.5, delta=0.5)
        # An order of magnitude, which is what makes the retraction load-bearing rather than
        # pedantic: 1-in-300 reads as negligible and 1-in-36 does not.
        self.assertGreater(with_t / gaussian, 10.0)

        pairs = 15
        rule_of_three = 1.0 - 0.05 ** (1.0 / pairs)
        self.assertAlmostEqual(rule_of_three, 0.18104, places=5)
        # Wider than either model, which is the honest bound: 0 of 15 is compatible with 18%.
        self.assertGreater(rule_of_three, with_t)
        self.assertEqual(len(MEASURED_PASS_MEANS) * (len(MEASURED_PASS_MEANS) - 1) // 2, pairs)

        # WHAT WOULD SUPPLY THE RATE: twenty passes, for 19 df and 190 pairs.
        passes = 20
        self.assertEqual(passes - 1, 19)
        self.assertEqual(passes * (passes - 1) // 2, 190)
        self.assertGreater(satterthwaite_df(10**6, 10**6), 5.0)

    def test_the_matched_arm_spread_is_two_states_and_not_a_gaussian(self) -> None:
        """Why a sigma is the wrong summary of these six passes at all.

        The 15 folds are bimodal with nothing between the modes, and the six pass means split the
        same way: two passes in a slow host state and four in a fast one. So the tail any
        Gaussian implies is an extrapolation across a gap that ONE observation produced -- the
        transition into the slow state -- and the within-state spread is under 2%.

        And the reframing has to be named too: `ln(1.25)/ln(1.1170) = 2.02` is the same shape as
        the struck "1 + 3.1x the observed excess" claim (0.25/0.117 = 2.14). Quoting it as
        log-margin is more nearly right than quoting it linearly, but it is a MARGIN on one
        observed maximum, not a probability.
        """
        folds = sorted(
            max(a / b, b / a)
            for i, a in enumerate(MEASURED_PASS_MEANS)
            for b in MEASURED_PASS_MEANS[i + 1 :]
        )
        low = [f for f in folds if f < 1.05]
        high = [f for f in folds if f >= 1.05]
        self.assertEqual((len(low), len(high)), (7, 8))
        self.assertAlmostEqual(max(low), 1.0150, places=4)
        self.assertAlmostEqual(min(high), 1.0943, places=4)
        # A gap of 0.079 with nothing in it, on a range whose total width is 0.116.
        self.assertGreater(min(high) - max(low), 0.6 * (max(folds) - min(folds)))

        slow = [m for m in MEASURED_PASS_MEANS if m < 360_000.0]
        fast = [m for m in MEASURED_PASS_MEANS if m >= 360_000.0]
        self.assertEqual((len(slow), len(fast)), (2, 4))
        # The gap between the two states rests on TWO passes landing in the slow one.
        self.assertGreater(min(fast) / max(slow), 1.09)
        for group in (slow, fast):
            with self.subTest(group=group):
                self.assertLess(max(group) / min(group), 1.02)

        self.assertAlmostEqual(
            math.log(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO) / math.log(max(folds)),
            2.02,
            places=2,
        )
        self.assertAlmostEqual(
            (FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO - 1.0) / (max(folds) - 1.0),
            2.14,
            places=2,
        )

    def test_the_position_term_is_the_largest_one_and_it_is_left_out(self) -> None:
        """N3's constant, and the crossover it implies if it does not cancel.

        `FOULPLAY_THINK_MEASURED_POSITION_LOG_SD` is 0.1003 -- 1.90x the per-decision residual
        and 1.94x the run term, the two that ARE in `foulplay_think_nominal_fold_resolution` -- and
        it is omitted because two paired arms replay the same battle seeds. Real arms diverge
        after their first differing choice, so for fully diverged arms it does not cancel, and
        then the crossover against 1.25 moves from 27 to 124. Strata of 27..123 are certified able
        to resolve the threshold on the cancelling assumption and cannot under the other one.

        Given that an unresolvable stratum is EXCLUDED rather than refused, the whole consequence
        is confined to the share floor's 5% remainder -- which is why this is stated at the use
        site rather than corrected into the constant.
        """
        self.assertAlmostEqual(
            FOULPLAY_THINK_MEASURED_POSITION_LOG_SD / FOULPLAY_THINK_MEASURED_DECISION_LOG_SD,
            1.90,
            places=2,
        )
        self.assertAlmostEqual(
            FOULPLAY_THINK_MEASURED_POSITION_LOG_SD / FOULPLAY_THINK_MEASURED_RUN_LOG_SD,
            1.94,
            places=2,
        )
        combined = math.hypot(
            FOULPLAY_THINK_MEASURED_DECISION_LOG_SD, FOULPLAY_THINK_MEASURED_POSITION_LOG_SD
        )
        self.assertAlmostEqual(combined, 0.113395, places=6)

        def no_cancellation(n):
            return math.exp(
                3.0
                * math.sqrt(
                    2.0 * FOULPLAY_THINK_MEASURED_RUN_LOG_SD**2 + combined**2 * (2.0 / n)
                )
            )

        crossover = next(
            n
            for n in range(1, 100_000)
            if no_cancellation(n) <= FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
        )
        self.assertEqual(crossover, 124)
        self.assertGreater(crossover, FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS)
        # The shipped floor's own stratum, under the other assumption: 1.2683, not 1.2500.
        n = FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS
        self.assertAlmostEqual(no_cancellation(n), 1.2683, places=4)
        self.assertAlmostEqual(foulplay_think_nominal_fold_resolution(n, n), 1.2500, places=4)

    def test_the_position_terms_magnitude_on_the_sd_and_on_the_empirical_anchor(self) -> None:
        """The mechanism was already conceded; this is the size of it, which is the finding.

        A scope note saying "position variance may not cancel for real arms" is compatible with
        the effect being 0.1% or 100%. Measured on the decomposition it is **+7.3%** on SD(log
        fold) at the calibration's own n=24 and **+0.9%** at n=200 -- it bites where the gate is
        weakest and washes out where it is strong, because the run term dominates everywhere. So
        it is a scope qualifier, not an argument for moving 1.25.

        WHAT IT DOES CHANGE is the STATUS of the 1.1170 anchor, and this is the one direction in
        which that anchor is optimistic rather than conservative: the six passes replayed
        identical positions, so 1.1170 is the matched-arm spread with the largest term removed by
        construction. Inflated by the n=24 factor it is ~1.126, which leaves 1.25 with 1.88x
        log-margin rather than 2.02x. Still ample; no longer the number it was quoted as.
        """
        combined = math.hypot(
            FOULPLAY_THINK_MEASURED_DECISION_LOG_SD, FOULPLAY_THINK_MEASURED_POSITION_LOG_SD
        )

        def diverging_sd(n):
            return math.sqrt(
                2.0 * FOULPLAY_THINK_MEASURED_RUN_LOG_SD**2 + combined**2 * (2.0 / n)
            )

        expected = {24: (0.0746, 0.0800, 0.073), 200: (0.0732, 0.0738, 0.009)}
        for n, (shared, diverged, inflation) in expected.items():
            with self.subTest(n=n):
                self.assertAlmostEqual(fold_log_sd(n, n), shared, places=4)
                self.assertAlmostEqual(diverging_sd(n), diverged, places=4)
                self.assertAlmostEqual(
                    diverging_sd(n) / fold_log_sd(n, n) - 1.0, inflation, places=3
                )
        # Monotone: the correction shrinks as n grows, because the run term is what is left.
        self.assertGreater(
            diverging_sd(24) / fold_log_sd(24, 24), diverging_sd(1000) / fold_log_sd(1000, 1000)
        )

        # The anchor, corrected at its own n. The fold is multiplicative, so the SD ratio is an
        # exponent on it, not a multiplier.
        corrected = MEASURED_MAX_MATCHED_FOLD ** (diverging_sd(24) / fold_log_sd(24, 24))
        self.assertAlmostEqual(corrected, 1.126, places=3)
        self.assertGreater(corrected, MEASURED_MAX_MATCHED_FOLD)
        log_margin = math.log(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO) / math.log(corrected)
        self.assertAlmostEqual(log_margin, 1.88, places=2)
        # And 1.25 still clears it, which is why this is a qualifier and not a threshold change.
        self.assertGreater(log_margin, 1.5)
        self.assertGreater(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, corrected)

    def test_the_two_quoted_conventions_are_labelled_because_a_sign_flips(self) -> None:
        """Which tail, and which n. Both have been stated ambiguously in this file's history.

        WHICH TAIL: every t here is matched to the two-sided tail of z=3, p=0.0027 exactly. The
        neighbouring conventions are visibly different numbers -- p=0.002 gives t=5.893 and a
        floor of 1.537, p=0.001 gives t=6.869 and 1.651 -- so a floor quoted without its p is
        unreadable, and all three are far above 1.25 either way.

        WHICH n: 1.25 sits on OPPOSITE SIDES of the nominal point estimate at the two n's this
        file quotes. At the measurement's own n=24 the bound is 1.2506 and 1.25 is 0.05% BELOW it;
        at the asymptote it is 1.2447 and 1.25 is 0.42% ABOVE it. The retraction holds on both
        conventions -- 1.25 is inside the interval either way -- but the direction is not a fact
        until the n is attached.
        """
        p3 = math.erfc(3.0 / (2**0.5))
        self.assertAlmostEqual(p3, 0.0026998, places=7)
        asymptotic_sd = (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD
        labelled = {0.0027: (5.507, 1.4946), 0.002: (5.893, 1.5374), 0.001: (6.869, 1.6508)}
        for p, (t_expected, floor_expected) in labelled.items():
            with self.subTest(p=p):
                t = student_t_two_sided_quantile(p, FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF)
                self.assertAlmostEqual(t, t_expected, places=3)
                self.assertAlmostEqual(math.exp(t * asymptotic_sd), floor_expected, places=4)
                # Whichever convention: far above the threshold, so the retraction is robust.
                self.assertGreater(
                    math.exp(t * asymptotic_sd), FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
                )
        # The one this file uses is the 3-sigma-matched one, and it is the tightest of the three.
        self.assertAlmostEqual(
            student_t_two_sided_quantile(p3, FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF), 5.507, places=3
        )

        at_24 = math.exp(3.0 * fold_log_sd(MEASURED_PASS_N, MEASURED_PASS_N))
        asymptote = math.exp(3.0 * asymptotic_sd)
        self.assertAlmostEqual(at_24, 1.2506, places=4)
        self.assertAlmostEqual(asymptote, 1.2447, places=4)
        # The sign flip, which is the whole reason the n has to be quoted.
        self.assertLess(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, at_24)
        self.assertGreater(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, asymptote)
        self.assertAlmostEqual(
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO / at_24 - 1.0, -0.0005, places=4
        )
        self.assertAlmostEqual(
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO / asymptote - 1.0, 0.0042, places=4
        )

    def test_the_arm_level_floor_is_the_crossover_and_not_the_twenty(self) -> None:
        """The dead constant, reconciled.

        `FOULPLAY_THINK_MIN_MEASURED_DECISIONS` is 20 and the resolving crossover is 27, so 20
        can never be what admits an arm to this gate: an arm whose decisions all sit in one
        20-decision stratum clears the within-arm floor, has that stratum excluded here, and
        refuses on coverage at share 0.0. The two constants are kept distinct rather than
        collapsed -- 20 also serves callers that read one arm and never compare a fold ratio --
        and this test is what stops them being read as one number.
        """
        self.assertLess(
            FOULPLAY_THINK_MIN_MEASURED_DECISIONS,
            FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS,
        )
        at_the_within_arm_floor = verdict(
            [think(380_000.0, FOULPLAY_THINK_MIN_MEASURED_DECISIONS)],
            [think(380_000.0, FOULPLAY_THINK_MIN_MEASURED_DECISIONS)],
        )
        self.assertEqual(at_the_within_arm_floor["status"], "refused")
        self.assertEqual(
            at_the_within_arm_floor["cross_arm_compared_share"], {"search": 0.0, "raw": 0.0}
        )
        self.assertIn("2x1000ms", at_the_within_arm_floor["strata_excluded_for_resolution"])
        # And the reason says resolution, not coverage: a 20-decision arm did not visit a
        # narrower set of schedules, it visited one that cannot resolve the threshold.
        self.assertIn(
            "search:cross_arm_strata_excluded_for_resolution_cover_too_little",
            at_the_within_arm_floor["refusal_reasons"],
        )
        self.assertNotIn(
            "search:cross_arm_compared_strata_cover_too_little",
            at_the_within_arm_floor["refusal_reasons"],
        )
        # And at the crossover the same arms certify, so 27 is the operative floor.
        at_the_crossover = verdict(
            [think(380_000.0, FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS)],
            [think(380_000.0, FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS)],
        )
        self.assertEqual(at_the_crossover["status"], "ok")


class StarvedArmTest(unittest.TestCase):
    def test_a_starved_search_arm_is_refused(self) -> None:
        """The confound the whole instrument exists for, at the size already measured."""
        raw_rate = 224_200.0
        search_rate = raw_rate / MEASURED_STARVATION_FOLD
        result = verdict(
            [think(search_rate, 200)], [think(raw_rate, 200)]
        )
        self.assertIn("cross_arm_rate_ratio_exceeds_threshold", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")

    def test_the_direction_of_the_confound_is_named_not_left_to_a_sign(self) -> None:
        """Argument order is a real bug class here, so the convention is asserted.

        `ratio_lean_over_hungry > 1` MUST mean the opponent realized less work against the
        hungry arm. Swapping the two arguments inverts it, and a 0.26 reads as innocuous to
        anyone scanning for "above 1".
        """
        result = verdict([think(100_000.0, 200)], [think(380_000.0, 200)])
        worst = result["worst_stratum"]
        self.assertGreater(worst["ratio_lean_over_hungry"], 1.0)
        self.assertEqual(result["hungry_arm"], "search")
        self.assertEqual(result["lean_arm"], "raw")
        self.assertGreater(
            worst["raw_mean_iterations_per_budget_second"],
            worst["search_mean_iterations_per_budget_second"],
        )

    def test_a_starved_raw_arm_is_refused_too(self) -> None:
        """Either direction. A paired delta is contaminated whichever arm was flattered."""
        result = verdict([think(380_000.0, 200)], [think(100_000.0, 200)])
        self.assertIn("cross_arm_rate_ratio_exceeds_threshold", result["refusal_reasons"])
        self.assertLess(result["worst_stratum"]["ratio_lean_over_hungry"], 1.0)
        self.assertGreater(result["worst_stratum"]["fold_ratio"], 1.0)

    def test_a_threshold_refusal_KEEPS_its_evidence(self) -> None:
        """The one refusal that must NOT withhold, and the distinction is the point.

        Withholding exists because an INADMISSIBLE comparison produces a reassuring number
        nobody should be able to reconstruct. A threshold refusal is the opposite case: the
        comparison was admissible and it found something, and the ratio IS the finding. A
        blanket "withhold on any refusal" would delete the evidence for the confound.
        """
        result = verdict([think(224_200.0 / 3.8, 200)], [think(224_200.0, 200)])
        self.assertEqual(
            result["refusal_reasons"], ["cross_arm_rate_ratio_exceeds_threshold"]
        )
        self.assertNotIn("rates_withheld_because", result)
        self.assertAlmostEqual(result["worst_stratum"]["fold_ratio"], 3.8, places=6)
        self.assertIsNotNone(result["unstratified_ratio_lean_over_hungry"])

    def test_the_boundary_is_where_the_constant_says_it_is(self) -> None:
        """Just over refuses, just under passes -- so the constant is load-bearing."""
        base = 380_000.0
        over = verdict([think(base / 1.26, 200)], [think(base, 200)])
        under = verdict([think(base / 1.24, 200)], [think(base, 200)])
        self.assertIn("cross_arm_rate_ratio_exceeds_threshold", over["refusal_reasons"])
        self.assertEqual(under["status"], "ok")

    def test_one_bad_stratum_refuses_even_when_the_others_are_clean(self) -> None:
        """Refusing on the WORST stratum, not on an average of them.

        A starvation that only bites in the early game -- where foul-play's own schedule
        gives it 8 sampled battles and our search has the most to do -- would be diluted to
        nothing by averaging across strata.
        """
        search = think(
            decisions=200, strata={"8x500ms": (40_000.0, 100), "2x1000ms": (240_000.0, 100)}
        )
        raw = think(
            decisions=200, strata={"8x500ms": (120_000.0, 100), "2x1000ms": (240_000.0, 100)}
        )
        result = verdict([search], [raw])
        self.assertIn("cross_arm_rate_ratio_exceeds_threshold", result["refusal_reasons"])
        self.assertEqual(result["worst_stratum"]["stratum"], "8x500ms")
        self.assertAlmostEqual(result["by_stratum"]["2x1000ms"]["fold_ratio"], 1.0)


class WithheldRatesTest(unittest.TestCase):
    """A refused comparison must not leave a reconstructible reassuring number behind.

    The instrument hit this exact hazard at row level: the withheld rate was recoverable by
    arithmetic from the row's own inputs, so a non-attributable row now omits those inputs
    too. Same rule one level up -- and here the number at risk is specifically 1.0, which is
    what an inadmissible comparison produces.
    """

    @staticmethod
    def _sliver(search_private_rate, raw_private_rate):
        """A well-powered SHARED stratum that holds 2% of each arm, and nothing else shared.

        The shared sliver clears `FOULPLAY_THINK_MIN_STRATUM_DECISIONS`, so it is not excluded
        as thin -- it is reported, at a perfectly flat ratio -- while 98% of each arm sits in
        a stratum the other never visited and is never compared like-for-like.
        """
        search = think(
            decisions=510,
            strata={"8x500ms": (search_private_rate, 500), "2x1000ms": (380_000.0, 10)},
        )
        raw = think(
            decisions=510,
            strata={"4x750ms": (raw_private_rate, 500), "2x1000ms": (380_000.0, 10)},
        )
        return search, raw

    def test_a_sliver_of_shared_stratum_cannot_report_1_0(self) -> None:
        """The case that read `ok` with a ratio of 1.0 on 2% of each arm.

        The within-arm gate refuses it on `compared_strata_cover_too_little`. What is pinned
        here is that no ratio survives into this layer's payload -- because a 1.0 in an
        artifact outlives the sentence next to it.
        """
        search, raw = self._sliver(380_000.0, 380_000.0)
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertIn("search:compared_strata_cover_too_little", result["refusal_reasons"])
        self.assertIn("rates_withheld_because", result)
        self.assertNotIn(1.0, numbers_in(result["by_stratum"]))
        self.assertNotIn("unstratified_ratio_lean_over_hungry", result)
        self.assertNotIn("worst_stratum", result)

    def test_a_real_starvation_hidden_behind_a_sliver_is_refused(self) -> None:
        """The sliver is dangerous because it HIDES something: 3.8x, on 500 of 510."""
        search, raw = self._sliver(59_000.0, 224_200.0)
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertNotIn("worst_stratum", result)

    def test_the_denominators_survive_the_withholding(self) -> None:
        """Withholding the ratio, not the diagnosis: an analyst still sees the shape.

        Taken from each arm's OWN strata rather than from the compared subset, which is where
        the sliver's whole story lives: 500 against 0 in each private stratum.
        """
        search, raw = self._sliver(59_000.0, 224_200.0)
        result = verdict([search], [raw])
        self.assertEqual(
            result["by_stratum"],
            {
                "2x1000ms": {
                    "search_iterations_measured_decisions": 10,
                    "raw_iterations_measured_decisions": 10,
                },
                "4x750ms": {
                    "search_iterations_measured_decisions": 0,
                    "raw_iterations_measured_decisions": 500,
                },
                "8x500ms": {
                    "search_iterations_measured_decisions": 500,
                    "raw_iterations_measured_decisions": 0,
                },
            },
        )
        self.assertTrue(result["refusal_reasons"])


class EveryRatioCarriesItsDenominatorsTest(unittest.TestCase):
    def test_both_arms_n_travel_with_every_reported_ratio(self) -> None:
        search = think(decisions=200, strata={"8x500ms": (120_000.0, 90), "2x1000ms": (240_000.0, 110)})
        raw = think(decisions=200, strata={"8x500ms": (121_000.0, 80), "2x1000ms": (239_000.0, 120)})
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            {
                name: (
                    block["search_iterations_measured_decisions"],
                    block["raw_iterations_measured_decisions"],
                )
                for name, block in result["by_stratum"].items()
            },
            {"8x500ms": (90, 80), "2x1000ms": (110, 120)},
        )

    def test_each_stratum_reports_what_it_could_have_resolved(self) -> None:
        """A stratum that passed at n=30 must not read like one that passed at n=2000."""
        thin = verdict([think(380_000.0, 30)], [think(380_000.0, 30)])
        thick = verdict([think(380_000.0, 2000)], [think(380_000.0, 2000)])
        self.assertEqual(thin["status"], "ok")
        self.assertEqual(thick["status"], "ok")
        self.assertGreater(
            thin["by_stratum"]["2x1000ms"]["nominal_fold_resolution_point_estimate"],
            thick["by_stratum"]["2x1000ms"]["nominal_fold_resolution_point_estimate"],
        )
        self.assertAlmostEqual(
            thick["by_stratum"]["2x1000ms"]["nominal_fold_resolution_point_estimate"],
            1.24480,
            places=5,
        )

    def test_a_stratum_that_cannot_resolve_the_threshold_is_excluded_from_the_compared_set(
        self,
    ) -> None:
        """The guard the "reported, not gated" comment used to stand in for.

        The first version said a gate on `nominal_fold_resolution_point_estimate` "could
        never read False,
        and would certify nothing". Wrong on its own numbers: at the within-arm floor of 5 the
        bound is 1.2723 -- WIDER than the 1.25 it is compared against -- so such a stratum
        returned `ok` while being unable to tell a matched pair from a starved one.

        The crossover is `FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS` = 27 on both
        arms, and the consequence is EXCLUSION from the compared set. Here that is the whole arm,
        so the share goes to 0.0 and the gate still refuses; the point of exclusion is what
        happens when the unresolvable stratum is a slice instead (below).
        """
        for n, excluded in ((26, True), (27, False)):
            result = verdict([think(380_000.0, n)], [think(380_000.0, n)])
            with self.subTest(n=n):
                self.assertEqual(
                    "2x1000ms" in result["strata_excluded_for_resolution"], excluded
                )
                self.assertEqual(
                    "2x1000ms" not in result["by_stratum"]
                    or "fold_ratio" not in result["by_stratum"]["2x1000ms"],
                    excluded,
                )
                self.assertEqual(result["status"], "refused" if excluded else "ok")
        self.assertEqual(FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS, 27)

    def test_the_resolving_floor_constant_is_recomputed_from_the_two_sds(self) -> None:
        """A derived constant that is only typed in has already drifted once.

        27 is not chosen: it is the smallest n at which `foulplay_think_nominal_fold_resolution`
        comes in at or under the threshold. Recomputed here from the two measured SDs so that
        moving either one and leaving the constant alone fails.
        """
        crossover = next(
            n
            for n in range(1, 100_000)
            if foulplay_think_nominal_fold_resolution(n, n)
            <= FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
        )
        self.assertEqual(crossover, FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS)
        self.assertAlmostEqual(foulplay_think_nominal_fold_resolution(26, 26), 1.250197, places=6)
        self.assertAlmostEqual(foulplay_think_nominal_fold_resolution(27, 27), 1.249996, places=6)

    def test_the_resolution_comparison_is_strict_at_its_own_boundary(self) -> None:
        """`>` and not `>=`, pinned where the two differ.

        A mutation from `>` to `>=` on the resolution comparison survived every other test in
        this file, because no fixture put a stratum's resolution EXACTLY on the threshold: the
        value is an `exp()` and never lands on 1.25. Handing the gate its own computed resolution
        as `max_fold_ratio` is the one input where the two operators disagree -- strict keeps the
        stratum, non-strict excludes it and takes the share to 0.0 -- so this is the failing
        input for that mutation and nothing else.
        """
        n = FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS
        exact = foulplay_think_nominal_fold_resolution(n, n)
        result = verdict(
            [think(380_000.0, n)], [think(380_000.0, n)], max_fold_ratio=exact
        )
        self.assertEqual(result["strata_excluded_for_resolution"], {})
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(
            result["by_stratum"]["2x1000ms"]["nominal_fold_resolution_point_estimate"], exact
        )
        # And one representable step tighter DOES exclude it, so the assertion above is a
        # boundary and not a region.
        tighter = verdict(
            [think(380_000.0, n)],
            [think(380_000.0, n)],
            max_fold_ratio=math.nextafter(exact, 0.0),
        )
        self.assertIn("2x1000ms", tighter["strata_excluded_for_resolution"])


class ExclusionKeepsItsOwnDiagnosisTest(unittest.TestCase):
    """The objection to a bare exclusion, honoured rather than overridden.

    Two independent fixes of the dead band disagreed about the disposition of a stratum that
    cannot resolve the threshold. One REFUSED and defended it in terms: "an exclusion would land
    in the uncompared remainder and be reported as a coverage problem, which is a different
    diagnosis." That is a real cost of a bare exclusion -- "this stratum could not resolve the
    threshold" and "the two arms visited different schedules" have different remedies. The other
    EXCLUDED, which is what ships, because a refusal returns `winner=None` and adopts nothing over
    a fraction of a percent of an arm.

    So the exclusion carries the distinction the refusal was protecting: named strata, a per-arm
    magnitude, and a refusal reason that says which shortfall it was. This class is the failing
    input for collapsing the two reasons back into one.
    """

    RESOLUTION = "search:cross_arm_strata_excluded_for_resolution_cover_too_little"
    COVERAGE = "search:cross_arm_compared_strata_cover_too_little"

    def test_a_pure_resolution_shortfall_is_not_reported_as_a_coverage_gap(self) -> None:
        """One perturbation: a 10% slice too thin to resolve, and NOTHING else uncovered.

        Both arms visit exactly the same two schedules with the same counts, so there is no
        coverage gap of any kind -- every decision is either compared or excluded for resolution.
        The share is 0.90 and refuses. Before the reasons were split, the only thing the artifact
        said was `cross_arm_compared_strata_cover_too_little`, which points a reader at decision
        mix when the cause is a denominator.
        """
        strata = {"2x1000ms": (380_000.0, 234), "8x500ms": (190_000.0, 26)}
        arms = [think(decisions=260, strata=strata) for _ in range(2)]
        result = verdict([arms[0]], [arms[1]])
        self.assertEqual(result["status"], "refused")
        self.assertIn("8x500ms", result["strata_excluded_for_resolution"])
        self.assertAlmostEqual(result["cross_arm_compared_share"]["search"], 0.9, places=9)
        self.assertAlmostEqual(
            result["cross_arm_share_excluded_for_resolution"]["search"], 0.1, places=9
        )
        self.assertIn(self.RESOLUTION, result["refusal_reasons"])
        self.assertNotIn(self.COVERAGE, result["refusal_reasons"])

    def test_a_pure_coverage_gap_is_not_reported_as_a_resolution_exclusion(self) -> None:
        """The mirror, one perturbation: a 10% slice the LEAN arm never visited.

        Thick enough to resolve the threshold on its own -- 26 is under the crossover, so this
        fixture uses 60 -- but present on one arm only, so it is never compared. Nothing is
        excluded for resolution, and the resolution reason must stay silent or it becomes noise
        that a reader learns to ignore.
        """
        search = think(
            decisions=600, strata={"2x1000ms": (380_000.0, 540), "8x500ms": (190_000.0, 60)}
        )
        raw = think(decisions=600, strata={"2x1000ms": (380_000.0, 600)})
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["strata_excluded_for_resolution"], {})
        self.assertAlmostEqual(result["cross_arm_compared_share"]["search"], 0.9, places=9)
        self.assertEqual(result["cross_arm_share_excluded_for_resolution"]["search"], 0.0)
        self.assertIn(self.COVERAGE, result["refusal_reasons"])
        self.assertNotIn(self.RESOLUTION, result["refusal_reasons"])

    def test_both_causes_at_once_report_both(self) -> None:
        """Because the honest answer to "which was it" is sometimes "both".

        20% excluded for resolution and 20% never shared, so coverage is short even counting every
        excluded stratum as covered: 0.60 compared, 0.80 resolvable, both under 0.95. A rule that
        reported only the first cause would understate what has to be fixed.
        """
        search = think(
            decisions=100,
            strata={
                "2x1000ms": (380_000.0, 60),
                "8x500ms": (190_000.0, 20),
                "16x250ms": (95_000.0, 20),
            },
        )
        raw = think(
            decisions=100,
            strata={"2x1000ms": (380_000.0, 80), "8x500ms": (190_000.0, 20)},
        )
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertEqual(sorted(result["strata_excluded_for_resolution"]), ["8x500ms"])
        self.assertAlmostEqual(result["cross_arm_compared_share"]["search"], 0.6, places=9)
        self.assertAlmostEqual(
            result["cross_arm_share_excluded_for_resolution"]["search"], 0.2, places=9
        )
        self.assertIn(self.RESOLUTION, result["refusal_reasons"])
        self.assertIn(self.COVERAGE, result["refusal_reasons"])

    def test_the_share_floor_is_inclusive_exactly_at_the_floor(self) -> None:
        """`share < floor` and not `<=`, at the one input where the two operators disagree.

        No fixture in this file used to put a compared share EXACTLY on 0.95 -- the shares in play
        were 0.995, 0.990, 0.900 and 0.0 -- so a mutation from `<` to `<=` survived the whole
        suite while turning the floor from "at least 95% of the arm" into "more than 95%". Here
        190 of 200 decisions are compared and the other 10 are excluded for resolution, which is
        0.95 exactly in binary floating point (190/200 and the literal 0.95 are the same double).
        Inclusive: the cell certifies.
        """
        strata = {"2x1000ms": (380_000.0, 190), "8x500ms": (190_000.0, 10)}
        arms = [think(decisions=200, strata=strata) for _ in range(2)]
        result = verdict([arms[0]], [arms[1]])
        self.assertEqual(
            result["cross_arm_compared_share"]["search"],
            FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE,
        )
        self.assertEqual(result["refusal_reasons"], [])
        self.assertEqual(result["status"], "ok")
        self.assertIn("8x500ms", result["strata_excluded_for_resolution"])
        # And one decision further out DOES refuse, so the assertion above is a boundary.
        over = {"2x1000ms": (380_000.0, 189), "8x500ms": (190_000.0, 11)}
        over_arms = [think(decisions=200, strata=over) for _ in range(2)]
        refused = verdict([over_arms[0]], [over_arms[1]])
        self.assertEqual(refused["status"], "refused")
        self.assertIn(self.RESOLUTION, refused["refusal_reasons"])

    def test_the_resolvable_share_comparison_is_strict_at_the_floor_too(self) -> None:
        """The SECOND `<`, on `(covered + excluded) / measured`, at its own exact boundary.

        The coverage reason fires only when coverage would be short even if every excluded
        stratum had been resolvable. Here 180 of 200 are compared (0.90, refused), 10 are excluded
        for resolution and 10 sit in a schedule the lean arm never visited -- so the resolvable
        share is 190/200 = 0.95 EXACTLY. At the floor, coverage was sufficient, so the shortfall
        is attributable to the exclusion and the coverage reason must not fire. `<=` there would
        report a coverage gap that the arm does not have.

        This is also why the implementation divides `(covered + excluded)` once rather than adding
        two shares: 180/200 + 10/200 is not guaranteed to be the same double as 190/200.
        """
        search = think(
            decisions=200,
            strata={
                "2x1000ms": (380_000.0, 180),
                "8x500ms": (190_000.0, 10),
                "16x250ms": (95_000.0, 10),
            },
        )
        raw = think(
            decisions=200,
            strata={"2x1000ms": (380_000.0, 190), "8x500ms": (190_000.0, 10)},
        )
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertAlmostEqual(result["cross_arm_compared_share"]["search"], 0.9, places=9)
        covered = 180
        excluded = 10
        self.assertEqual(
            (covered + excluded) / 200, FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE
        )
        self.assertIn(self.RESOLUTION, result["refusal_reasons"])
        self.assertNotIn(self.COVERAGE, result["refusal_reasons"])

    def test_an_impossible_header_is_not_laundered_into_a_resolution_exclusion(self) -> None:
        """The forged-header check has to see the EXCLUDED strata too.

        `stratum_counts_exceed_measured_decisions` was computed from `covered / measured` alone.
        Put the impossible excess inside a stratum this layer excludes and that ratio comes back
        under 1.0: 90 compared and 10 excluded against a claimed 95 measured reads share 0.947 and
        resolvable share 1.053. It still refused -- for resolution -- so nothing was certified,
        but the reported cause was a thin denominator when the real cause is a shard whose own
        arithmetic does not close. That is the same laundering the malformed-ratio check is
        ordered first to prevent, one level up.
        """
        forged = think(
            decisions=100,
            strata={"2x1000ms": (380_000.0, 90), "8x500ms": (190_000.0, 10)},
        )
        forged["iterations_measured_decisions"] = 95
        honest = think(
            decisions=100,
            strata={"2x1000ms": (380_000.0, 90), "8x500ms": (190_000.0, 10)},
        )
        result = verdict([forged], [honest])
        self.assertEqual(result["status"], "refused")
        self.assertIn(
            "search:stratum_counts_exceed_measured_decisions", result["refusal_reasons"]
        )
        # And NOT the resolution reason, which is what it used to say: the excluded stratum is
        # real, but it is not why this shard is inadmissible.
        self.assertNotIn(self.RESOLUTION, result["refusal_reasons"])
        self.assertNotIn(self.COVERAGE, result["refusal_reasons"])
        # The two shares that straddle 1.0, so the fixture is pinned where the two sums differ.
        self.assertLess(result["cross_arm_compared_share"]["search"], 1.0)
        self.assertGreater(
            result["cross_arm_compared_share"]["search"]
            + result["cross_arm_share_excluded_for_resolution"]["search"],
            1.0,
        )
        # The honest arm is unaffected and reports the truth about itself.
        self.assertAlmostEqual(result["cross_arm_compared_share"]["raw"], 0.9, places=9)

    def test_the_verdict_note_explains_the_two_reasons_in_words(self) -> None:
        """A reason key is only legible to someone who has read this file.

        `verdict_note` is what a reader of the artifact sees, so it has to carry the distinction
        too -- otherwise the split reasons are two opaque strings and the diagnosis is back where
        it was. Both keys, both remedies, and that neither is contention.
        """
        note = verdict([think(380_000.0, 200)], [think(380_000.0, 200)])["verdict_note"]
        self.assertIn("NAMES WHICH", note)
        self.assertIn("cross_arm_strata_excluded_for_resolution_cover_too_little", note)
        self.assertIn("cross_arm_compared_strata_cover_too_little", note)
        self.assertIn("different remedies", note)
        self.assertIn("Neither is contention", note)

    def test_the_compared_share_numerator_is_the_actually_compared_set(self) -> None:
        """The dead band's other half: the share once counted a stratum the gate had refused.

        `per_stratum[name] = entry` ran before the refusal and was never undone, so an
        unresolvable stratum was simultaneously a hard refusal AND part of the compared-share
        numerator -- `cross_arm_compared_share` read 1.0 while the refusal said the same stratum
        could not be compared. The numerator is `per_stratum`, which an excluded stratum is not
        in, so the share here must be 1000/1010 and not 1.0.
        """
        strata = {"2x1000ms": (380_000.0, 1000), "8x500ms": (190_000.0, 10)}
        result = verdict(
            [think(decisions=1010, strata=strata)], [think(decisions=1010, strata=strata)]
        )
        self.assertEqual(result["status"], "ok")
        for label in ("search", "raw"):
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    result["cross_arm_compared_share"][label], 1000 / 1010, places=9
                )
                self.assertNotEqual(result["cross_arm_compared_share"][label], 1.0)
                self.assertAlmostEqual(
                    result["cross_arm_share_excluded_for_resolution"][label],
                    10 / 1010,
                    places=9,
                )
                # The two account for the whole arm here, because nothing else is uncovered.
                self.assertAlmostEqual(
                    result["cross_arm_compared_share"][label]
                    + result["cross_arm_share_excluded_for_resolution"][label],
                    1.0,
                    places=9,
                )

    def test_the_reported_resolution_field_does_not_assert_a_confidence_level(self) -> None:
        """The rename, pinned so the old key cannot come back into an artifact.

        `detectable_fold_ratio_3sigma` asserted a confidence level a 5-df point estimate cannot
        carry, in a field whose whole job is to say how strongly a stratum passed -- and a field
        name outlives the paragraph next to it. The key is now
        `nominal_fold_resolution_point_estimate`, in both places it appears.
        """
        passing = verdict([think(380_000.0, 200)], [think(380_000.0, 200)])
        excluded = verdict([think(380_000.0, 10)], [think(380_000.0, 10)])
        blocks = [
            passing["by_stratum"]["2x1000ms"],
            excluded["strata_excluded_for_resolution"]["2x1000ms"],
        ]
        for block in blocks:
            with self.subTest(block=sorted(block)):
                self.assertIn("nominal_fold_resolution_point_estimate", block)
                self.assertNotIn("detectable_fold_ratio_3sigma", block)
        for payload in (passing, excluded):
            self.assertNotIn("3sigma", json.dumps(payload))
            self.assertNotIn("detectable", json.dumps(payload))


class TheResolvingFloorIsTwentySevenTest(unittest.TestCase):
    """27 against 24, reconciled to one number with the reason the other one existed.

    Two branches derived two floors from the same six passes. This class is what makes the
    disagreement checkable rather than a matter of whose commit landed second.
    """

    def test_twenty_four_is_the_calibrations_n_and_not_a_crossover(self) -> None:
        """The one-line reason 27 wins: 24 does not clear the threshold on the shipped SDs.

        The nominal bound at the calibration's own n=24 is 1.2506, ABOVE 1.25 -- which is exactly
        the fact the alternative branch used to argue that a derived floor was untenable, and it
        argues just as well that 24 is not the floor. 24 is a crossover only for a run SD in
        [0.051426, 0.051475] (to six decimals), a window 0.0516 sits above.
        """
        self.assertEqual(FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS, 27)
        self.assertGreater(
            foulplay_think_nominal_fold_resolution(MEASURED_PASS_N, MEASURED_PASS_N),
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
        )
        self.assertAlmostEqual(
            foulplay_think_nominal_fold_resolution(MEASURED_PASS_N, MEASURED_PASS_N),
            1.250649,
            places=6,
        )

        def crossover(run_sd):
            for n in range(1, 100_000):
                sd = math.sqrt(
                    2.0 * run_sd**2
                    + FOULPLAY_THINK_MEASURED_DECISION_LOG_SD**2 * (2.0 / n)
                )
                if math.exp(3.0 * sd) <= FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO:
                    return n
            return None

        self.assertEqual(crossover(FOULPLAY_THINK_MEASURED_RUN_LOG_SD), 27)
        # The window in which 24 WOULD be the crossover, closed at both quoted endpoints, with
        # the neighbours on either side to show it is a window and not a region.
        self.assertEqual(crossover(0.051426), 24)
        self.assertEqual(crossover(0.051475), 24)
        self.assertEqual(crossover(0.051425), 23)
        self.assertEqual(crossover(0.051476), 25)
        self.assertGreater(FOULPLAY_THINK_MEASURED_RUN_LOG_SD, 0.051475)

    def test_the_two_branches_disagreed_about_which_sd_not_about_deriving(self) -> None:
        """Where 24 came from: the raw pass-mean SD, not the run component.

        `sqrt(0.002782) = 0.052745` is the SD of the six pass MEANS, which still contains the
        sampling term. The run component removes it: `sqrt(0.002782 - 0.0529**2/24) = 0.05163`,
        rounded to the shipped 0.0516. At the pass-mean SD the asymptote is 1.2508 -- ABOVE the
        threshold -- so no n at any denominator would ever certify, which is what made a derived
        floor look like a gate that could never open. On the component the module actually uses
        the asymptote is 1.2447 and the crossover exists.

        Every figure the two branches disagreed on falls out of that one substitution, and each
        is labelled with its p because two of them differ only by convention.
        """
        pass_mean_sd = math.sqrt(0.002782)
        self.assertAlmostEqual(pass_mean_sd, 0.052745, places=6)
        component = math.sqrt(
            0.002782 - FOULPLAY_THINK_MEASURED_DECISION_LOG_SD**2 / MEASURED_PASS_N
        )
        self.assertAlmostEqual(component, 0.05163, places=5)
        self.assertAlmostEqual(component, FOULPLAY_THINK_MEASURED_RUN_LOG_SD, places=4)
        self.assertGreater(pass_mean_sd, component)

        self.assertAlmostEqual(math.exp(3.0 * (2**0.5) * pass_mean_sd), 1.25079, places=5)
        self.assertGreater(
            math.exp(3.0 * (2**0.5) * pass_mean_sd), FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
        )
        self.assertAlmostEqual(
            math.exp(3.0 * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD), 1.24473, places=5
        )
        self.assertLess(
            math.exp(3.0 * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD),
            FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
        )

        # THE FLOORS, each with its p and its SD. The pass-mean column is where 1.5521, 1.6692
        # and 1.7312 come from; none of them is a run-term number.
        df = FOULPLAY_THINK_MEASURED_RUN_LOG_SD_DF
        three_sigma = math.erfc(3.0 / (2**0.5))
        for p, on_component, on_pass_mean in (
            (three_sigma, 1.4946, 1.5080),
            (0.002, 1.5374, 1.5521),
            (0.001, 1.6508, 1.6692),
        ):
            t = student_t_two_sided_quantile(p, df)
            with self.subTest(p=p):
                self.assertAlmostEqual(
                    math.exp(t * (2**0.5) * FOULPLAY_THINK_MEASURED_RUN_LOG_SD),
                    on_component,
                    places=4,
                )
                self.assertAlmostEqual(
                    math.exp(t * (2**0.5) * pass_mean_sd), on_pass_mean, places=4
                )

        # And the chi-square interval, one-sided against two-sided, on both SDs.
        def upper_sd(sd, tail):
            return sd * math.sqrt(df / chi_square_quantile(tail, df))

        self.assertAlmostEqual(upper_sd(FOULPLAY_THINK_MEASURED_RUN_LOG_SD, 0.05), 0.1078, places=4)
        self.assertAlmostEqual(
            math.exp(3.0 * (2**0.5) * upper_sd(FOULPLAY_THINK_MEASURED_RUN_LOG_SD, 0.05)),
            1.580,
            places=3,
        )
        self.assertAlmostEqual(
            math.exp(3.0 * (2**0.5) * upper_sd(FOULPLAY_THINK_MEASURED_RUN_LOG_SD, 0.025)),
            1.711,
            places=3,
        )
        self.assertAlmostEqual(
            math.exp(3.0 * (2**0.5) * upper_sd(pass_mean_sd, 0.025)), 1.7312, places=4
        )

    def test_moving_the_constant_moves_a_share_and_not_a_verdict(self) -> None:
        """Why a DERIVED floor is admissible here and was not admissible as a refusal.

        The objection that retired the derived floor was "a gate whose behaviour flips on the
        third decimal place of a 5-df estimate is not a gate". Against a refusal that was true: a
        stratum crossing the floor voided the whole comparison. Against an exclusion it is not:
        crossing the floor moves that stratum's decisions from the compared share into the
        excluded share, and the verdict follows the share floor. Swept across the whole plausible
        range of the floor -- 5 to 200 -- these matched arms certify at every value, because the
        stratum that moves is 1% of the arm.
        """
        strata = {"2x1000ms": (380_000.0, 1000), "8x500ms": (190_000.0, 10)}
        arms = [think(decisions=1010, strata=strata) for _ in range(2)]
        for floor_n in (5, 10, 11, 26, 27, 28, 124, 200):
            # `max_fold_ratio` is the only knob the gate exposes, so the floor is moved the way
            # the arithmetic moves it: the threshold at which a stratum of `floor_n` resolves.
            limit = foulplay_think_nominal_fold_resolution(floor_n, floor_n)
            result = verdict([arms[0]], [arms[1]], max_fold_ratio=limit)
            with self.subTest(floor_n=floor_n):
                self.assertEqual(result["status"], "ok")
                excluded = result["cross_arm_share_excluded_for_resolution"]["search"]
                self.assertLessEqual(excluded, 10 / 1010)
                self.assertGreaterEqual(
                    result["cross_arm_compared_share"]["search"],
                    FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE,
                )

    def test_the_other_two_floors_are_inert_at_this_layer(self) -> None:
        """5 and 20 are both below 27, so neither can admit anything to a cross-arm verdict.

        Stated as a test and not only as a comment, because three floors that do not agree get
        read as one number. What 5 still does: decide whether a too-thin stratum is named in
        `thin_strata` or in `strata_excluded_for_resolution`. What 20 still does: govern callers
        that read one arm and never compare a fold ratio.
        """
        for inert in (
            FOULPLAY_THINK_MIN_STRATUM_DECISIONS,
            FOULPLAY_THINK_MIN_MEASURED_DECISIONS,
        ):
            with self.subTest(inert=inert):
                self.assertLess(inert, FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS)
                self.assertGreater(
                    foulplay_think_nominal_fold_resolution(inert, inert),
                    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
                )
        # A stratum at 5 is reportable by the within-arm gate and excluded by this one: the two
        # exclusions are distinguishable, which is 5's whole remaining job here.
        def with_minor(n):
            return think(
                decisions=200 + n,
                strata={"2x1000ms": (380_000.0, 200), "8x500ms": (190_000.0, n)},
            )

        below_five = verdict([with_minor(5)], [with_minor(4)])
        self.assertIn("8x500ms", below_five["thin_strata"])
        self.assertNotIn("8x500ms", below_five["strata_excluded_for_resolution"])
        at_five = verdict([with_minor(5)], [with_minor(5)])
        self.assertNotIn("8x500ms", at_five["thin_strata"] or [])
        self.assertIn("8x500ms", at_five["strata_excluded_for_resolution"])


class UnmeasuredIsNotFlatTest(unittest.TestCase):
    """A shard that measured nothing must refuse, not read as "no contention detected"."""

    def test_zero_coverage_on_one_arm_is_refused(self) -> None:
        blind = think(decisions=200, strata={})
        self.assertIsNone(blind["mean_iterations_per_budget_second"])
        result = verdict([blind], [think(380_000.0, 200)])
        self.assertEqual(result["status"], "refused")
        self.assertIn("search:no_rate_measured", result["refusal_reasons"])
        self.assertIn("search:zero_measured_decisions", result["refusal_reasons"])
        self.assertIn("rates_withheld_because", result)

    def test_null_on_both_arms_is_refused_not_read_as_flat(self) -> None:
        blind = think(decisions=200, strata={})
        result = verdict([blind], [blind])
        self.assertEqual(result["status"], "refused")
        self.assertIn("raw:no_rate_measured", result["refusal_reasons"])

    def test_unequal_coverage_is_refused_at_near_identical_rates(self) -> None:
        """Coverage is treatment-dependent, so unequal coverage compares two subsamples.

        The surviving decisions on the CPU-heavy arm are the fastest-draining, i.e. the least
        contended, which pulls its rate UP toward the reassuring answer -- so this refusal
        has to fire even though the two rates look fine.
        """
        search = think(decisions=200, strata={"2x1000ms": (380_000.0, 80)})
        raw = think(decisions=200, strata={"2x1000ms": (381_000.0, 200)})
        self.assertAlmostEqual(search["iterations_coverage"], 0.4)
        result = verdict([search], [raw])
        self.assertIn("coverage_gap_exceeds_limit", result["refusal_reasons"])
        self.assertIn("rates_withheld_because", result)

    def test_a_handful_of_measured_decisions_is_refused(self) -> None:
        result = verdict([think(380_000.0, 6)], [think(380_000.0, 6)])
        self.assertIn("search:below_minimum_measured_decisions", result["refusal_reasons"])

    def test_no_shared_stratum_is_refused(self) -> None:
        search = think(decisions=200, strata={"8x500ms": (120_000.0, 200)})
        raw = think(decisions=200, strata={"2x1000ms": (240_000.0, 200)})
        result = verdict([search], [raw])
        self.assertIn("no_shared_stratum_with_measured_rate", result["refusal_reasons"])
        # The unstratified ratio it WOULD have reported is 2.0, from the schedule alone.
        self.assertNotIn(2.0, numbers_in(result))


class StartMethodTest(unittest.TestCase):
    """`iterations` is only observable under `fork`. Re-measured on CPython 3.12.13:
    `fork` emits the per-sample line, `spawn` and `forkserver` emit nothing. So a shard whose
    header names a non-emitting method has NO coverage and must refuse rather than read as
    "no contention detected".
    """

    def test_a_spawn_shard_is_refused(self) -> None:
        result = verdict(
            [think(380_000.0, 200, observable=False, start_method="spawn")],
            [think(380_000.0, 200)],
        )
        self.assertIn("search:start_method_cannot_emit_iterations", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")

    def test_a_forkserver_shard_is_refused(self) -> None:
        result = verdict(
            [think(380_000.0, 200)],
            [think(380_000.0, 200, observable=False, start_method="forkserver")],
        )
        self.assertIn("raw:start_method_cannot_emit_iterations", result["refusal_reasons"])

    def test_a_failed_probe_is_unknown_not_observable(self) -> None:
        """`iterations_observable: null` is what a failed probe records.

        The run-level reading status refuses only the explicit `False`, so without this the
        UNKNOWN case pools into a clean comparison: the header still carries rates, still
        carries strata, and nothing says the one fact the measurement rests on was never
        established.
        """
        unknown = think(380_000.0, 200, observable=None, start_method=None)
        result = verdict([unknown], [think(380_000.0, 200)])
        self.assertIn("search:start_method_unknown", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")

    def test_shards_that_disagree_about_the_start_method_are_refused(self) -> None:
        """The `fork` shards supply the numerator and the others supply the denominator.

        Both headers here claim `iterations_observable: True`, so the observability refusals
        cannot cover for this one: what is wrong is that the arm was not one measurement
        regime, and a pooled coverage across two regimes is a blend of "measured" and
        "unmeasurable".
        """
        result = verdict(
            [
                think(380_000.0, 200, start_method="fork"),
                think(380_000.0, 200, start_method="forkserver"),
            ],
            [think(380_000.0, 400)],
        )
        self.assertIn("search:start_method_not_uniform", result["refusal_reasons"])


class PoolingTest(unittest.TestCase):
    """An arm is not one shard: a cell is a seed band per shard and two seats per shard."""

    def test_pooling_weights_by_measured_decisions_not_by_shard(self) -> None:
        """An average of averages is a different number, and it is the wrong one."""
        pooled = pool_foulplay_think(
            [think(300_000.0, 10), think(400_000.0, 90)], label="search"
        )
        self.assertEqual(pooled["iterations_measured_decisions"], 100)
        self.assertAlmostEqual(pooled["mean_iterations_per_budget_second"], 390_000.0)
        self.assertAlmostEqual(
            pooled["by_stratum"]["2x1000ms"]["mean_iterations_per_budget_second"], 390_000.0
        )
        self.assertEqual(pooled["by_stratum"]["2x1000ms"]["iterations_measured_decisions"], 100)
        # The unweighted answer, for contrast: it is 350,000 and it is 10% off.
        self.assertNotAlmostEqual(pooled["mean_iterations_per_budget_second"], 350_000.0)

    def test_a_seat_with_no_think_block_is_refused_not_skipped(self) -> None:
        """`seat_block` emits `foulplay_think: None` for a producer too old to measure.

        Skipping those would compute coverage over the shards that DID carry a block, which
        reads clean while a quarter of the arm was never measured at all.
        """
        pooled = pool_foulplay_think([think(380_000.0, 200), None], label="search")
        self.assertEqual(pooled["headers_without_block"], 1)
        result = cross_arm_foulplay_contention(
            pooled, pool_foulplay_think([think(380_000.0, 200)], label="raw")
        )
        self.assertIn("search:think_block_absent_in_pool", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")

    def test_a_header_from_a_different_think_schema_is_refused(self) -> None:
        """A v1 header has no `by_stratum` and no `decisions_attempted`.

        Pooling it silently produces an arm-level reading over a subset of its own shards.
        """
        result = verdict(
            [think(380_000.0, 200), think(380_000.0, 200, schema_version="pokezero.foulplay-think.v1")],
            [think(380_000.0, 400)],
        )
        self.assertIn("search:think_schema_mismatch", result["refusal_reasons"])

    def test_an_arm_with_no_headers_at_all_is_refused(self) -> None:
        pooled = pool_foulplay_think([], label="search")
        self.assertEqual(pooled["pool_refusals"], ["search:no_think_headers"])
        result = cross_arm_foulplay_contention(
            pooled, pool_foulplay_think([think(380_000.0, 200)], label="raw")
        )
        self.assertEqual(result["status"], "refused")
        self.assertIn("search:no_think_headers", result["refusal_reasons"])

    def test_pooled_coverage_is_over_decisions_attempted(self) -> None:
        """A decision the telemetry LOST is a decision that was not measured.

        Coverage over rows produced reads 1.0 while 900 of 1000 decisions went unrecorded --
        and the coverage gate, whose whole purpose is catching treatment-correlated coverage,
        then sees a gap of 0.0.
        """
        pooled = pool_foulplay_think(
            [think(380_000.0, 100, record_failures=900)], label="search"
        )
        self.assertAlmostEqual(pooled["iterations_coverage"], 0.1)
        self.assertEqual(pooled["decisions_attempted"], 1000)

    def test_observability_folds_conservatively(self) -> None:
        """True only when EVERY constituent proved observable."""
        self.assertIs(
            pool_foulplay_think([think(1.0, 20), think(1.0, 20)], label="a")[
                "iterations_observable"
            ],
            True,
        )
        self.assertIs(
            pool_foulplay_think(
                [think(1.0, 20), think(1.0, 20, observable=False)], label="a"
            )["iterations_observable"],
            False,
        )
        self.assertIsNone(
            pool_foulplay_think(
                [think(1.0, 20), think(1.0, 20, observable=None)], label="a"
            )["iterations_observable"]
        )


class NothingVanishesQuietlyTest(unittest.TestCase):
    """A stratum that leaves the comparison without a stated reason fails toward "comparable".

    Found by reading the within-arm gate's own skips rather than its refusals: it declines to
    divide by a zero hungry-arm rate and moves on, which is correct arithmetic and a silent
    exclusion. Zero visits per granted second is TOTAL starvation -- the strongest possible
    reading of the confound -- and it is exactly the value that drops out.
    """

    def test_a_totally_starved_stratum_cannot_drop_out_of_the_comparison(self) -> None:
        """The failing input, sized so no other guard covers for this one.

        The starved stratum holds 30 of 200 decisions on each arm: enough to be well-powered,
        too little for the compared-share floor to notice its absence. Coverage is 1.0 on both
        arms and the surviving stratum's ratio is exactly 1.0, so before this guard the verdict
        was `ok` while the opponent realized NOTHING on 30 of the search arm's decisions.
        """
        search = think(
            decisions=200, strata={"8x500ms": (0.0, 30), "2x1000ms": (380_000.0, 170)}
        )
        raw = think(
            decisions=200, strata={"8x500ms": (120_000.0, 30), "2x1000ms": (380_000.0, 170)}
        )
        # The conditions no other guard trips on: full coverage, and the compared stratum
        # carries 85% of each arm, over the 80% floor.
        self.assertAlmostEqual(search["iterations_coverage"], 1.0)
        self.assertAlmostEqual(raw["iterations_coverage"], 1.0)
        result = verdict([search], [raw])
        self.assertIn("stratum_dropped_without_reason", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")
        # And the cross-arm coverage floor catches the same input from the other side: the
        # compared set now holds 170 of 200 on each arm, which is 0.85 and under 0.95.
        self.assertIn(
            "search:cross_arm_compared_strata_cover_too_little", result["refusal_reasons"]
        )

    def test_a_zero_rate_on_the_lean_arm_cannot_be_skipped_when_picking_the_worst(self) -> None:
        """The mirror case, which stays INSIDE the compared set.

        The hungry arm's rate is fine so the stratum is compared, but the ratio is 0.0 and a
        fold ratio is undefined -- so the most extreme stratum in the payload was the one
        unable to raise a refusal.
        """
        search = think(
            decisions=200, strata={"8x500ms": (120_000.0, 30), "2x1000ms": (380_000.0, 170)}
        )
        raw = think(
            decisions=200, strata={"8x500ms": (0.0, 30), "2x1000ms": (380_000.0, 170)}
        )
        result = verdict([search], [raw])
        self.assertIn("stratum_rate_not_a_ratio", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")
        # And the rates are WITHHELD, unlike a threshold refusal: `worst_stratum` here would
        # be the largest fold over an INCOMPLETE set of strata, which is a reassuring number
        # computed by leaving the degenerate one out.
        self.assertIn("rates_withheld_because", result)
        self.assertNotIn("worst_stratum", result)

    def test_a_stratum_named_as_thin_is_not_double_refused(self) -> None:
        """The guard must not fire on the exclusions that ARE stated.

        A stratum below the per-stratum floor is excluded and named in `thin_strata`, which is
        a stated exclusion. Refusing it as unexplained too would make the new reason fire on
        every run that has a short stratum, i.e. always.
        """
        search = think(
            decisions=203, strata={"2x1000ms": (380_000.0, 200), "8x500ms": (120_000.0, 3)}
        )
        raw = think(
            decisions=203, strata={"2x1000ms": (381_000.0, 200), "8x500ms": (121_000.0, 3)}
        )
        result = verdict([search], [raw])
        self.assertEqual(result["thin_strata"], ["8x500ms"])
        self.assertNotIn("stratum_dropped_without_reason", result["refusal_reasons"])
        self.assertEqual(result["status"], "ok")


class ReviewFoundHolesTest(unittest.TestCase):
    """The holes an independent adversarial review demonstrated, each with its input.

    Every one of these returned `status: "ok"` before the fix, and every one is a
    flattering-direction failure: a starved opponent reading as a matched one.
    """

    def test_a_starvation_hidden_by_a_thin_LEAN_stratum_is_refused(self) -> None:
        """The first version of `stratum_dropped_without_reason` required n>=5 on BOTH arms.

        A stratum's decision count follows each arm's OWN game lengths, so `n_lean = 3` while
        `n_hungry = 40` is ordinary rather than adversarial -- and with the hungry rate at 0 the
        stratum is neither compared nor named thin, so an ARBITRARILY large starvation on a fifth
        of the hungry arm published `worst_stratum.fold_ratio: 1.0000` and was adopted.
        """
        search = think(
            decisions=200, strata={"8x500ms": (0.0, 40), "2x1000ms": (380_000.0, 160)}
        )
        raw = think(
            decisions=203, strata={"8x500ms": (120_000.0, 3), "2x1000ms": (380_000.0, 200)}
        )
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertIn("stratum_dropped_without_reason", result["refusal_reasons"])
        self.assertNotIn("worst_stratum", result)
        self.assertNotIn(1.0, numbers_in(result["by_stratum"]))

    def test_the_uncompared_fifth_the_within_arm_floor_permits_is_refused(self) -> None:
        """The composed bound, which was the review's headline.

        `FOULPLAY_THINK_MIN_COMPARED_SHARE` is 0.8, so a fifth of each arm may sit outside the
        compared strata -- unbounded. Composed with every compared stratum reading just inside
        1.25, the review measured a TRUE arm-level fold of 1.5581 on a cell this gate called
        `ok`. Here the hungry arm's uncompared fifth is a stratum the lean arm never visited.
        """
        search = think(
            decisions=200,
            strata={"8x500ms": (10_000.0, 40), "2x1000ms": (380_000.0 / 1.2499, 160)},
        )
        raw = think(decisions=200, strata={"2x1000ms": (380_000.0, 200)})
        result = verdict([search], [raw])
        self.assertEqual(result["status"], "refused")
        self.assertIn(
            "search:cross_arm_compared_strata_cover_too_little", result["refusal_reasons"]
        )
        self.assertNotIn("worst_stratum", result)

    def test_the_composed_worst_case_is_the_number_the_docstring_states(self) -> None:
        """f/s, and it has to be checked rather than asserted.

        With compared share s and every compared stratum within f, the hungry arm's own mean is at
        least s times its compared rate -- worst case, its uncompared decisions realized zero
        work -- so the arm-level fold is at most f/s. At the within-arm gate's s=0.8 that is
        1.5625, and an independent review's composition measured 1.5581 on a fixture: the formula
        reproduces the measurement, so 1.3158 at s=0.95 is a checked bound and not arithmetic
        nobody tried.

        AND THE HEADLINE IS THE SHORTFALL, NOT THE FOLD: `1 - s/f` = 24.0%. Readers subtract, so
        a fold of 1.3158 in the headline invites "31.6% shortfall" (wrong, by 7.6 points in the
        unsafe direction) and the bare per-stratum 1.25 invites "25%" (wrong twice over -- 20%
        before coverage is composed in, 24% after). Only the shortfall is directly comparable to
        the number a reader carries away, so it is what the docstring and `verdict_note` state.
        """
        f = FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
        s = FOULPLAY_THINK_MIN_CROSS_ARM_COMPARED_SHARE
        self.assertAlmostEqual(f / 0.8, 1.5625, places=4)
        self.assertAlmostEqual(f / s, 1.3158, places=4)
        self.assertAlmostEqual(1.0 - s / f, 0.240, places=4)
        # The two misreadings the phrasing exists to prevent, pinned as numbers.
        self.assertAlmostEqual(1.0 - 1.0 / (f / s), 0.240, places=4)
        self.assertAlmostEqual(1.0 - 1.0 / f, 0.200, places=4)
        self.assertGreater((f / s - 1.0) - (1.0 - s / f), 0.07)
        # And the artifact says the shortfall, in words, with the fold in parentheses.
        note = verdict([think(380_000.0, 200)], [think(380_000.0, 200)])["verdict_note"]
        self.assertIn("SHORTFALL is at most 24.0%", note)
        self.assertIn("1.3158", note)
        # The measured composition at s=0.8, reproduced through the gate: 40 of 200 hungry-arm
        # decisions in a stratum the lean arm never visited, the rest at exactly f.
        search = think(
            decisions=200,
            strata={"8x500ms": (1.0, 40), "2x1000ms": (380_000.0 / f, 160)},
        )
        raw = think(decisions=200, strata={"2x1000ms": (380_000.0, 200)})
        arm_fold = (
            raw["mean_iterations_per_budget_second"]
            / search["mean_iterations_per_budget_second"]
        )
        self.assertGreater(arm_fold, 1.55)
        self.assertLess(arm_fold, f / 0.8)
        # And it is refused now, which is the point of the 0.95 floor.
        self.assertEqual(verdict([search], [raw])["status"], "refused")

    def test_strata_counts_that_exceed_the_arm_are_refused(self) -> None:
        """A header whose `by_stratum` counts sum above its own measured total.

        Not reachable from the shipping producer, which derives both from the same rows. It
        produced a compared share of 10.0 -- an impossible number that passed a `>= 0.8` floor
        and a `>= 0.95` one alike, because nothing checked the other side.
        """
        forged = think(380_000.0, 200)
        forged["iterations_measured_decisions"] = 20
        result = verdict([forged], [think(380_000.0, 200)])
        self.assertIn(
            "search:stratum_counts_exceed_measured_decisions", result["refusal_reasons"]
        )
        self.assertEqual(result["status"], "refused")

    def test_a_rate_that_is_not_a_positive_finite_number_is_refused(self) -> None:
        """`max(r, 1/r)` returns -1.0 for a negative rate and NaN propagates as False.

        Both read `ok`, and the NaN also writes non-strict JSON into the report artifact. Not
        reachable from the shipping producer, which refuses a non-positive budget -- but this
        function is the gate for shards it did not produce.
        """
        for bad in (-380_000.0, float("nan"), float("inf")):
            result = verdict([think(bad, 200)], [think(380_000.0, 200)])
            with self.subTest(rate=bad):
                self.assertEqual(result["status"], "refused")
                self.assertNotIn("worst_stratum", result)
                json.dumps(result, allow_nan=False)

    def test_the_boundary_is_exclusive_exactly_at_the_threshold(self) -> None:
        """`>` not `>=`, pinned at the exact value rather than at 1.24 and 1.26."""
        base = 380_000.0
        exact = verdict(
            [think(base / FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, 200)], [think(base, 200)]
        )
        self.assertAlmostEqual(
            exact["worst_stratum"]["fold_ratio"], FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO
        )
        self.assertEqual(exact["status"], "ok")


class MustBePooledTest(unittest.TestCase):
    def test_an_unpooled_shard_header_is_refused(self) -> None:
        """The pooling guards are not optional, so the pooled shape is not optional.

        A caller handing a single shard's run header straight in gets the within-arm checks
        and none of the pooling ones -- absent blocks, a mixed think schema, an UNKNOWN start
        method, disagreeing start methods -- and every one of those bypasses fails toward
        "comparable".
        """
        bare = think(380_000.0, 200)
        self.assertNotIn("pool_refusals", bare)
        result = cross_arm_foulplay_contention(bare, bare)
        self.assertIn("search:not_pooled", result["refusal_reasons"])
        self.assertIn("raw:not_pooled", result["refusal_reasons"])
        self.assertEqual(result["status"], "refused")

    def test_a_missing_arm_is_refused(self) -> None:
        result = cross_arm_foulplay_contention(
            None, pool_foulplay_think([think(380_000.0, 200)], label="raw")
        )
        self.assertEqual(result["status"], "refused")
        self.assertIn("search:not_pooled", result["refusal_reasons"])
        self.assertIn("missing_arm", result["refusal_reasons"])


class SerializableTest(unittest.TestCase):
    def test_the_verdict_is_json_serializable(self) -> None:
        """It goes into a report artifact, so this is not a formality."""
        json.dumps(verdict([think(380_000.0, 200)], [think(380_000.0, 200)]))
        json.dumps(verdict([think(100_000.0, 200)], [think(380_000.0, 200)]))
        json.dumps(verdict([], [None]))


if __name__ == "__main__":
    unittest.main()
