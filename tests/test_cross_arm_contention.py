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
import unittest

from pokezero.foulplay_bridge import (
    FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
    FOULPLAY_THINK_MEASURED_RATE_CV,
    FOULPLAY_THINK_MIN_STRATUM_DECISIONS,
    FOULPLAY_THINK_SCHEMA_VERSION,
    cross_arm_foulplay_contention,
    pool_foulplay_think,
)

# --- the two measured anchors this gate's threshold was derived from --------------------
# Both from `crossarm-contention-dispersion.py` (kept outside the repo), driving foul-play's
# own `get_result_from_mcts` over poke_engine under `fork`, foul-play's own `init_logging`,
# captured off fd 1 and parsed by the SHIPPING parser. 2 passes x 24 positions, 48/48
# decisions measured, zero miss reasons, all in `2x1000ms`.
MEASURED_MATCHED_PAIR = (367_062.5, 396_750.0, 24)  # pass 0 mean, pass 1 mean, n per pass
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
        "total_iterations": int(sum(r * 2.0 * n for r, n in strata.values())),
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


class MatchedArmsTest(unittest.TestCase):
    """The negative control, and it is the one that decides the threshold.

    A gate like this one can MANUFACTURE the defect it detects -- the instrument's own give-up
    heuristic did exactly that once, producing a contention finding out of a healthy opponent.
    So the first obligation is a measured pair of arms that were NOT contending.
    """

    def test_a_measured_matched_pair_of_real_foulplay_passes(self) -> None:
        a_rate, b_rate, n = MEASURED_MATCHED_PAIR
        result = verdict([think(a_rate, n)], [think(b_rate, n)])
        self.assertEqual(result["refusal_reasons"], [])
        self.assertEqual(result["status"], "ok")
        worst = result["worst_stratum"]
        # 396,750 / 367,062.5 = 1.0809. Two passes over the SAME 24 positions with nothing
        # contending that was not contending in both.
        self.assertAlmostEqual(worst["fold_ratio"], 1.0809, places=4)
        self.assertLess(worst["fold_ratio"], FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO)

    def test_a_threshold_near_the_measured_noise_would_refuse_that_matched_pair(self) -> None:
        """Why the threshold is 1.25 and not something that looks more vigilant.

        This is the failing input for the CHOICE of threshold: at 1.05 the gate refuses a
        measured pair of uncontended real searches, i.e. it invents the confound. If the
        constant is ever tightened toward the sampling SE, this test says what breaks.
        """
        a_rate, b_rate, n = MEASURED_MATCHED_PAIR
        strict = verdict([think(a_rate, n)], [think(b_rate, n)], max_fold_ratio=1.05)
        self.assertEqual(
            strict["refusal_reasons"], ["cross_arm_rate_ratio_exceeds_threshold"]
        )

    def test_the_threshold_sits_between_the_two_measured_anchors(self) -> None:
        """The constant itself, pinned against both numbers it was derived from."""
        matched_fold = MEASURED_MATCHED_PAIR[1] / MEASURED_MATCHED_PAIR[0]
        self.assertGreater(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, matched_fold)
        self.assertLess(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, MEASURED_STARVATION_FOLD)
        # And 3 sigma at the within-arm gate's own per-stratum floor, the second derivation.
        floor = FOULPLAY_THINK_MIN_STRATUM_DECISIONS
        three_sigma = 1.0 + 3.0 * FOULPLAY_THINK_MEASURED_RATE_CV * (2.0 / floor) ** 0.5
        self.assertAlmostEqual(three_sigma, 1.2183, places=3)
        self.assertGreater(FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO, three_sigma)

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
        """A stratum that passed at n=5 must not read like one that passed at n=200.

        Reported, not gated: the within-arm floor already holds n at 5 or more, where the
        3-sigma figure sits just under the threshold, so a gate on it could never read False.
        Saying so here rather than shipping a check that certifies nothing.
        """
        thin = verdict(
            [think(decisions=200, strata={"a": (380_000.0, 195), "2x1000ms": (380_000.0, 5)})],
            [think(decisions=200, strata={"a": (380_000.0, 195), "2x1000ms": (380_000.0, 5)})],
        )
        thick = verdict([think(380_000.0, 200)], [think(380_000.0, 200)])
        self.assertGreater(
            thin["by_stratum"]["2x1000ms"]["detectable_fold_ratio_3sigma"],
            thick["by_stratum"]["2x1000ms"]["detectable_fold_ratio_3sigma"],
        )
        self.assertAlmostEqual(
            thick["by_stratum"]["2x1000ms"]["detectable_fold_ratio_3sigma"], 1.0345, places=4
        )


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
        self.assertEqual(result["refusal_reasons"], ["stratum_dropped_without_reason"])
        self.assertEqual(result["status"], "refused")

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
