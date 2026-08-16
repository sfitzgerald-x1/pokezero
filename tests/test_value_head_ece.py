"""Tests for `scripts/value_head_ece.py`, which had none.

The script's whole claim is that the ECE it recomputes IS the one in the published artifacts:
it says so in its docstring, it bootstraps that number, and every paired delta a Phase 3
advancement decision reads is a difference of it. That claim rests on one line of arithmetic
being a faithful transcription of `pokezero.value_calibration`, and it was not: the script
divided by a precomputed bin width (`int((p + 1) / (2 / bins))`) where the library multiplies
by the bin count (`int(((p + 1) / 2) * bins)`). `2 / 10` is not representable in binary
floating point, so the two disagree at 3 of the 9 interior bin edges of the default grid and
put those points in the bin BELOW -- silently, on a metric whose entire purpose is
comparability with numbers computed elsewhere.

So the tests here are agreement tests against the library, driven through the library's own
accumulator rather than through a model, plus hand-computed known answers for the aggregation
and the one formatting path that could not render the value its own metric is defined to
return.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BINS = 10


def _load():
    spec = importlib.util.spec_from_file_location(
        "value_head_ece", REPO / "scripts" / "value_head_ece.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vhe = _load()


def _library_totals(bins=BINS):
    from pokezero.value_calibration import _ValueCalibrationTotals
    return _ValueCalibrationTotals(bin_count=bins)


def _library_bin_index(prediction, bins=BINS):
    return _library_totals(bins)._bin_index(prediction)


def _edges(bins=BINS):
    return [-1.0 + 2.0 * k / bins for k in range(bins + 1)]


def test_bin_index_at_the_three_edges_the_shipped_arithmetic_got_wrong():
    """Hardcoded, not mirrored off the library: these are the answers, and they are the ones
    the script disagreed with.

    Bins are half-open `[lower, upper)` (`_BinTotals.to_bin` reports them that way), so an
    edge value belongs to the bin ABOVE. The shipped `int((p + 1) / (2 / bins))` returned
    2, 5 and 6 here.
    """
    assert vhe.bin_index(-0.40, BINS) == 3
    assert vhe.bin_index(0.20, BINS) == 6
    assert vhe.bin_index(0.40, BINS) == 7
    assert [_library_bin_index(p) for p in (-0.40, 0.20, 0.40)] == [3, 6, 7]


def test_bin_index_agrees_with_the_library_at_every_bin_edge():
    for bins in (2, 4, 10, 16, 20):
        for p in _edges(bins):
            assert vhe.bin_index(p, bins) == _library_bin_index(p, bins), (bins, p)


def test_bin_index_agrees_with_the_library_over_a_dense_sweep():
    """Edges are where the two disagreed, so the sweep is deliberately edge-heavy: a uniform
    random sample over [-1, 1] never lands on one and reports zero mismatches."""
    rng = random.Random(20260816)
    probes = [k / 1000.0 for k in range(-1000, 1001)]
    probes += [k / 20.0 for k in range(-20, 21)]
    probes += [rng.uniform(-1.0, 1.0) for _ in range(4000)]
    for bins in (2, 10, 20):
        for p in probes:
            assert vhe.bin_index(p, bins) == _library_bin_index(p, bins), (bins, p)


def test_bin_index_closes_the_top_bin_and_clips_out_of_range_predictions():
    """The library's two special cases, both of which keep a saturated head in range."""
    assert vhe.bin_index(1.0, BINS) == BINS - 1 == _library_bin_index(1.0)
    assert vhe.bin_index(1.5, BINS) == BINS - 1 == _library_bin_index(1.5)
    assert vhe.bin_index(-1.0, BINS) == 0 == _library_bin_index(-1.0)
    assert vhe.bin_index(-1.5, BINS) == 0 == _library_bin_index(-1.5)


def test_ece_from_reproduces_the_librarys_ece_bias_and_mae():
    """The cross-check the script performs at runtime, performed here without a checkpoint.

    `_ValueCalibrationTotals` is the same accumulator `evaluate_value_calibration` feeds, so
    driving it with a synthetic (prediction, return) column is the library's own number. The
    column includes every bin edge many times over, because that is the only place the two
    implementations ever differed.
    """
    rng = random.Random(4242)
    preds, rets = [], []
    for _ in range(2000):
        preds.append(rng.uniform(-1.0, 1.0))
        rets.append(rng.choice((-1.0, 0.0, 1.0)))
    for _ in range(40):
        for edge in _edges():
            preds.append(edge)
            rets.append(rng.choice((-1.0, 0.0, 1.0)))

    got = vhe.ece_from(preds, rets, BINS)
    totals = _library_totals()
    totals.add(predictions=tuple(preds), returns=tuple(rets))
    report = totals.to_report()

    assert got["n"] == report.examples
    assert got["ece"] == pytest.approx(report.expected_calibration_error, abs=1e-12)
    assert got["bias"] == pytest.approx(report.bias, abs=1e-9)
    assert got["mae"] == pytest.approx(report.mae, abs=1e-9)
    # The disagreement this test exists to catch is ~1e-4 on this column, not 1e-12: the
    # tolerance is tight because the two are meant to be the same arithmetic.
    assert got["ece"] > 0.05


def test_ece_from_is_a_hand_computed_known_answer():
    """Three examples, two bins, weighted by occupancy.

    -0.9 lands in bin 0 twice (mean prediction -0.9, mean return -0.9, so that bin contributes
    nothing however wrong the individual examples are -- which is the property ECE has and MAE
    does not), and 0.5 lands alone in bin 7 with an error of 0.4 at weight 1/3.
    """
    preds = [-0.9, -0.9, 0.5]
    rets = [-1.0, -0.8, 0.1]
    got = vhe.ece_from(preds, rets, BINS)
    assert [vhe.bin_index(p, BINS) for p in preds] == [0, 0, 7]
    assert got["ece"] == pytest.approx(0.4 / 3.0)
    assert got["bias"] == pytest.approx(0.4 / 3.0)
    assert got["mae"] == pytest.approx(0.6 / 3.0)
    assert got["abs_bias"] == pytest.approx(0.4 / 3.0)
    assert got["bias_share_of_ece"] == pytest.approx(1.0)
    assert got["n"] == 3
    # MAE is 0.2 while ECE is 0.133: the two examples that cancel inside bin 0 are exactly the
    # difference, so a test asserting only one of them would not see a binning change.
    assert got["mae"] > got["ece"]


def test_bias_share_of_ece_is_absent_when_ece_is_zero_and_still_renders():
    """`abs(bias)/ece` is undefined at ECE 0 -- a head perfectly calibrated on this set, which
    is the one head nobody would want the tool to crash on. Both readouts formatted it with
    `:.4f`, which raises TypeError on None.
    """
    got = vhe.ece_from([0.25, -0.25], [0.25, -0.25], BINS)
    assert got["ece"] == 0.0
    assert got["bias"] == 0.0
    assert got["bias_share_of_ece"] is None
    with pytest.raises(TypeError):
        f"{got['bias_share_of_ece']:.4f}"          # the formatting that shipped
    assert vhe.fmt_share(got["bias_share_of_ece"]) == f"{'n/a':>11s}"
    assert vhe.fmt_share(None, 0) == "n/a"
    assert vhe.fmt_share(0.5) == f"{0.5:11.4f}"
    assert vhe.fmt_share(0.5, 0) == "0.5000"


def test_the_two_ece_tolerances_are_distinct_and_the_help_says_which_is_unconditional():
    """`--expect-ece`'s help claimed its 1e-6 comparison was what protected comparability with
    the artifacts. It is not: the recomputed-vs-library cross-check runs on every head whether
    or not the flag is passed, and at 1e-9. Two different checks with different subjects were
    described as one, so a reader who skipped the optional flag believed nothing was verifying
    that the bootstrapped metric is the published metric.
    """
    assert vhe.LIBRARY_CROSSCHECK_TOL == 1e-9
    assert vhe.PUBLISHED_ECE_TOL == 1e-6
    assert vhe.LIBRARY_CROSSCHECK_TOL < vhe.PUBLISHED_ECE_TOL
    # The help text is BUILT from the constants, so it cannot drift from what runs.
    assert f"{vhe.LIBRARY_CROSSCHECK_TOL:g}" in vhe.EXPECT_ECE_HELP
    assert f"{vhe.PUBLISHED_ECE_TOL:g}" in vhe.EXPECT_ECE_HELP
    assert "unconditional" in vhe.EXPECT_ECE_HELP
    assert "SEPARATE check" in vhe.EXPECT_ECE_HELP


def test_pct_indexes_the_sorted_draws_and_cannot_run_off_the_end():
    draws = [0.5, 0.1, 0.9, 0.3, 0.7]
    assert vhe.pct(draws, 0.0) == 0.1
    assert vhe.pct(draws, 0.5) == 0.5
    assert vhe.pct(draws, 1.0) == 0.9
