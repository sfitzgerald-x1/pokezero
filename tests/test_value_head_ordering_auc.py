"""Known-answer tests for OI-1, `scripts/value_head_ordering_auc.py`.

This file is not a coverage exercise. OI-1 is a GATE, and this programme's standing rule is
that every gate ships with a demonstrated failing input: a check that cannot read False
certifies nothing (the campaign found three such guards, all green for months). So every
refusal below is shown refusing, the power guard is shown BOTH refusing and returning a
verdict, and the statistic is fed the four constructions an ordering gate has to separate:

  * a pure output rescale and a monotone recalibration -- the two vectors that gamed the
    beta+ECE pair, which must move the statistic by EXACTLY ZERO,
  * a saturating recalibration, which is monotone but collapses orderings into ties and must
    NOT be able to launder that into a pass,
  * a deliberately ordering-corrupted head, which must go clearly negative,
  * a head genuinely shrunk toward the labels, which must go clearly positive.

The synthetic fixtures here are self-contained; the banked-data numbers live in the PR body
and in the artifact the CLI writes, because tests must not depend on a bank that is not in
this repo.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "value_head_ordering_auc", REPO / "scripts" / "value_head_ordering_auc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oi = _load()


# ----------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------
def make_pairs(n=465, seed=11, beta=0.30, noise=0.03, level_sd=1.0, n_games=80,
               zero_frac=0.20, structural_tie_frac=0.05):
    """A bank shaped like the real one, INCLUDING the coupling that makes the tie rule matter.

    465 pairs over 80 games, a fifth with true_gap == 0, R=64 label noise on the truth column,
    and a head that is an attenuated, noisy view of the latent sibling gap.

    The load-bearing ingredient -- corrected after a review measured it, because the first
    version of this docstring named the wrong one -- is `level_sd`: the head's OUTPUT LEVEL must
    reach saturation, since that is what a saturating recalibration collapses into ties. At
    level_sd=1.0 the drop-ties laundering reproduces (+0.077 with the coupling below, +0.041
    without it); at level_sd=0.2 it does not reproduce at all (+0.001, 5 ties), which
    `test_the_laundering_demo_needs_a_saturating_head_and_says_so` pins so the fixture cannot
    quietly stop containing its own subject.

    `room` adds the measured COMPRESSION signature on top: the head's sibling gap shrinks where
    its output level is extreme (a bounded head has no room left) while its idiosyncratic error
    does not, so near-saturated pairs carry less signal at the same noise. That makes the fixture
    faithful to the banked head and roughly doubles the laundering effect. It is not what makes
    the demo work.

    `structural_tie_frac` reproduces the banked head's 36 exact head-side ties (sibling actions
    whose successor observations are identical), so the baseline cell exercises the tie path too.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        game = 24010000 + (i % n_games)
        latent = rng.gauss(0, 0.10)
        true_gap = 0.0 if rng.random() < zero_frac else latent + rng.gauss(0, 0.065)
        level = 0.95 * math.tanh(rng.gauss(0, level_sd))
        room = 1.0 - level * level
        half_gap = 0.0 if rng.random() < structural_tie_frac else \
            beta * latent * room + rng.gauss(0, noise)
        # head_a/head_b are the transform surface; the gap is their difference over GAP_SCALE.
        head_a = level + half_gap
        head_b = level - half_gap
        true_a = 0.5 + true_gap / 2
        rows.append({
            "seed": game, "prefix": i // n_games, "seat": "p1",
            "head_a": head_a, "head_b": head_b,
            "head_gap": oi.head_gap_from_values(head_a, head_b),
            "true_gap": true_gap, "true_a": true_a, "true_b": true_a - true_gap,
            "noise_var": 0.065 ** 2, "rollouts_a": 64, "rollouts_b": 64,
        })
    return rows


def write(tmp_path, name, rows, **extra):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"pairs": rows, "n_pairs": len(rows), **extra}))
    return p


@pytest.fixture(scope="module")
def bank():
    return make_pairs()


@pytest.fixture(scope="module")
def cell(bank):
    rows = {oi.pair_key(p): p for p in bank}
    return rows, sorted(rows)


def demo(cell, name, seed=20260816):
    rows, keys = cell
    return oi.build_demo_cell(rows, keys, name, seed=seed)


def cmp_at(cell, arm_rows, tau=oi.TAU_PRIMARY, **kw):
    rows, keys = cell
    kw.setdefault("bootstrap_reps", 200)
    return oi.compare(rows, arm_rows, keys, tau, **kw)


# ----------------------------------------------------------------------------------------
# the statistic itself
# ----------------------------------------------------------------------------------------
def test_a_head_side_tie_scores_exactly_one_half_and_is_not_dropped():
    assert oi.pair_score(0.0, 0.3) == 0.5
    assert oi.pair_score(0.0, -0.3) == 0.5
    assert oi.pair_score(0.1, 0.3) == 1.0
    assert oi.pair_score(-0.1, 0.3) == 0.0


def test_a_zero_true_gap_pair_is_refused_rather_than_scored():
    # Scoring it either way is a defect: as a failure it penalises a perfect head, as a
    # success it rewards a coin flip. The only correct behaviour is to never reach here.
    with pytest.raises(ValueError):
        oi.pair_score(0.2, 0.0)


def test_zero_true_gap_pairs_are_excluded_from_every_eligible_set(cell):
    rows, keys = cell
    n_zero = sum(1 for k in keys if rows[k]["true_gap"] == 0.0)
    assert n_zero > 0, "fixture must contain the case the exclusion exists for"
    for tau in (0.0, 0.05, 0.10, 0.15):
        sel = oi.eligible_keys(rows, keys, tau)
        assert all(rows[k]["true_gap"] != 0.0 for k in sel)
    # tau=0 is the case where the `!= 0` clause is NOT implied by the threshold, so it is the
    # one that would silently regress if the clause were deleted.
    assert len(oi.eligible_keys(rows, keys, 0.0)) == len(keys) - n_zero


def test_eligibility_tightens_monotonically_with_tau(cell):
    rows, keys = cell
    sizes = [len(oi.eligible_keys(rows, keys, t)) for t in (0.0, 0.05, 0.10, 0.15)]
    assert sizes == sorted(sizes, reverse=True)


def test_c_is_an_auc_not_an_accuracy_when_ties_are_present(cell):
    """With head ties present, C must sit strictly between the drop-ties accuracy and the
    count-ties-as-wrong accuracy. This is what pins the 0.5 credit as a real half."""
    rows, keys = cell
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    tied = {oi.pair_key(rows[k]) for k in sel[:7]}
    forced = {k: (dict(rows[k], head_a=0.0, head_b=0.0, head_gap=0.0) if k in tied else rows[k])
              for k in keys}
    scores = oi.score_cell(forced, sel)
    c = oi.concordance(scores)
    as_wrong = sum(1.0 if s == 1.0 else 0.0 for s in scores) / len(scores)
    kept = [s for s in scores if s != 0.5]
    as_dropped = sum(kept) / len(kept)
    assert as_wrong < c < as_dropped


# ----------------------------------------------------------------------------------------
# the demonstrated failing inputs (the programme-wide rule)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["affine_rescale_k3.2764", "monotone_tanh2",
                                  "monotone_sigmoid6", "additive_offset_plus0.2"])
def test_scale_offset_and_monotone_gaming_move_the_gate_by_exactly_zero(cell, name):
    for tau in (oi.TAU_PRIMARY, *oi.TAU_CHECKS):
        r = cmp_at(cell, demo(cell, name), tau)
        assert r["delta_c"] == 0.0, (name, tau, r["delta_c"])
        assert r["discordant_total"] == 0
        assert r["verdict"] == "NO_ADVANCE_INVARIANT"
        assert r["advanced"] is False


def test_the_gaming_transforms_really_do_change_the_head_they_leave_the_gate_blind_to(cell):
    """Guard against a vacuous demo: if the fixture did not actually alter the head, the four
    exact zeros above would prove nothing at all."""
    rows, keys = cell
    for name in ("affine_rescale_k3.2764", "monotone_tanh2", "monotone_sigmoid6",
                 "additive_offset_plus0.2"):
        arm = demo(cell, name)
        assert any(arm[k]["head_a"] != rows[k]["head_a"] for k in keys), name
        assert any(abs(arm[k]["head_gap"]) != abs(rows[k]["head_gap"]) for k in keys) or \
            name == "additive_offset_plus0.2", name
    # the offset case is the one whose GAP is genuinely untouched -- that is the whole reason
    # it games ECE and cannot game this gate.
    off = demo(cell, "additive_offset_plus0.2")
    assert all(abs(off[k]["head_gap"] - rows[k]["head_gap"]) < 1e-12 for k in keys)


@pytest.mark.parametrize("name", ["saturating_tanh50", "saturating_tanh50_on_winprob"])
def test_a_saturating_recalibration_reads_negative_and_manufactures_ties(cell, name):
    r = cmp_at(cell, demo(cell, name))
    assert r["head_ties_created_by_arm"] > 0
    assert r["delta_c"] < 0
    assert not r["verdict"].startswith("PASS")


def test_the_saturating_head_would_pass_under_a_drop_ties_rule(cell):
    """THE reason head-side ties score 0.5 instead of being dropped.

    A saturating transform breaks orderings into ties. Dropping ties deletes exactly the pairs
    it broke, so the surviving subset is enriched for pairs the head got right and the naive
    statistic goes UP. The shipped rule and the refused rule must disagree in SIGN here, or the
    0.5 credit is decoration.
    """
    rows, keys = cell
    arm = demo(cell, "saturating_tanh50")
    shipped = cmp_at(cell, arm)
    naive = oi.drop_ties_delta(rows, arm, keys, oi.TAU_PRIMARY)
    assert shipped["delta_c"] < 0 < naive["delta_c_drop_ties"]
    assert naive["n_arm_kept"] < naive["n_baseline_kept"]


def test_the_laundering_demo_needs_a_saturating_head_and_says_so():
    """A fixture that does not contain its own subject proves nothing. The laundering demo needs
    head LEVELS that saturate; with levels pinned near zero, tanh(50 v) collapses almost nothing
    and the drop-ties number is ~0. Pinned so the fixture's realism is a measured property."""
    flat = {oi.pair_key(p): p for p in make_pairs(level_sd=0.2)}
    keys = sorted(flat)
    arm = oi.build_demo_cell(flat, keys, "saturating_tanh50")
    r = oi.compare(flat, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=20)
    assert r["head_ties_created_by_arm"] < 10
    assert abs(oi.drop_ties_delta(flat, arm, keys, oi.TAU_PRIMARY)["delta_c_drop_ties"]) < 0.01
    # the shipped fixture, by contrast, saturates and launders
    sat = {oi.pair_key(p): p for p in make_pairs()}
    k2 = sorted(sat)
    arm2 = oi.build_demo_cell(sat, k2, "saturating_tanh50")
    assert oi.compare(sat, arm2, k2, oi.TAU_PRIMARY,
                      bootstrap_reps=20)["head_ties_created_by_arm"] > 50
    assert oi.drop_ties_delta(sat, arm2, k2, oi.TAU_PRIMARY)["delta_c_drop_ties"] > 0.03


@pytest.mark.parametrize("name", ["ordering_corrupted_15pct", "ordering_corrupted_25pct"])
def test_a_deliberately_ordering_corrupted_head_reads_clearly_negative(cell, name):
    r = cmp_at(cell, demo(cell, name))
    assert r["delta_c"] <= -0.02, r["delta_c"]
    assert r["discordant_baseline_better"] > r["discordant_arm_better"]
    assert not r["verdict"].startswith("PASS")


def test_corruption_severity_orders_the_statistic(cell):
    mild = cmp_at(cell, demo(cell, "ordering_corrupted_15pct"))["delta_c"]
    worse = cmp_at(cell, demo(cell, "ordering_corrupted_25pct"))["delta_c"]
    assert worse < mild < 0


@pytest.mark.parametrize("name", ["positive_control_blend_truth_0.10",
                                  "positive_control_blend_truth_0.15"])
def test_the_positive_control_is_detected_in_the_right_direction(cell, name):
    r = cmp_at(cell, demo(cell, name))
    assert r["delta_c"] >= 0.02, r["delta_c"]
    assert r["exact_signflip_p_two_sided"] < 0.05
    assert r["verdict"] in ("PASS", "PASS_PENDING_REMEASURE")


def test_the_positive_control_beats_every_gaming_and_corruption_input(cell):
    """Compared on `delta_c_gate`, which is what the verdict reads: the label-aimed abstention
    scores large and positive on the NAIVE statistic (that is the attack) and <= 0 on the gate."""
    pos = min(cmp_at(cell, demo(cell, n))["delta_c_gate"] for n in oi.DEMOS
              if n.startswith("positive_control"))
    bad = max(cmp_at(cell, demo(cell, n))["delta_c_gate"] for n in oi.DEMOS
              if not n.startswith("positive_control"))
    assert pos > 0 >= bad


def test_every_demo_in_the_registry_is_exercised_by_this_file():
    """If a demo is added to the registry and not asserted anywhere, the failing-input rule is
    satisfied on paper only."""
    text = Path(__file__).read_text()
    missing = [n for n in oi.DEMOS if text.count(f'"{n}"') == 0]
    assert not missing, f"demos never asserted in this file: {missing}"


def test_transforming_the_gap_instead_of_the_values_would_make_two_demos_vacuous(cell):
    """The fixtures must transform head_a/head_b and RECOMPUTE the gap. Applying the same
    transform to head_gap directly is equivalent only for linear maps -- and for the
    saturating demo it silently turns a real failure into a clean pass, because tanh is
    sign-preserving on the gap."""
    rows, keys = cell
    g = lambda v: math.tanh(50 * v)  # noqa: E731
    wrong = {k: dict(rows[k], head_gap=g(rows[k]["head_gap"])) for k in keys}
    right = demo(cell, "saturating_tanh50")
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    d_wrong = oi.concordance(oi.score_cell(wrong, sel)) - oi.concordance(
        oi.score_cell(rows, sel))
    d_right = oi.concordance(oi.score_cell(right, sel)) - oi.concordance(
        oi.score_cell(rows, sel))
    assert d_wrong == 0.0
    assert d_right < 0.0


# ----------------------------------------------------------------------------------------
# the decision rule itself. An adversarial review mutated `delta >= target` to `delta > 0`,
# deleted the `p < ALPHA` conjunct, mutated `delta <= -target` to `delta < 0`, and widened
# `advanced` to any PASS* verdict -- and all four mutants left the suite green. The gate's own
# rule had no demonstrated failing input, which is precisely what the programme-wide rule
# forbids. These are that demonstration.
# ----------------------------------------------------------------------------------------
def _rigged(n_eligible, n_up, n_down, tie_up=0, target=oi.ADVANCE_DELTA_C):
    """A synthetic pair of cells with an exactly known dC: `n_up` orderings the arm fixes,
    `n_down` it breaks, `tie_up` wrong orderings it converts to head-side ties (+0.5 each)."""
    rows, arm = {}, {}
    for i in range(n_eligible):
        k = (i, 0, "p1")
        if i < n_up:                     ha, hb, aa, ab = -0.2, 0.2, 0.2, -0.2
        elif i < n_up + n_down:          ha, hb, aa, ab = 0.2, -0.2, -0.2, 0.2
        elif i < n_up + n_down + tie_up: ha, hb, aa, ab = -0.2, 0.2, 0.1, 0.1
        else:                            ha, hb, aa, ab = 0.2, -0.2, 0.2, -0.2
        rows[k] = {"seed": i, "prefix": 0, "seat": "p1", "head_a": ha, "head_b": hb,
                   "head_gap": oi.head_gap_from_values(ha, hb), "true_gap": 0.3,
                   "true_a": 0.65, "true_b": 0.35, "noise_var": 0.004,
                   "rollouts_a": 64, "rollouts_b": 64}
        arm[k] = dict(rows[k], head_a=aa, head_b=ab,
                      head_gap=oi.head_gap_from_values(aa, ab))
    keys = sorted(rows)
    return oi.compare(rows, arm, keys, oi.TAU_PRIMARY, target=target, bootstrap_reps=20)


def test_a_gain_below_the_preregistered_threshold_is_not_a_pass():
    """Kills `delta >= target` -> `delta > 0`. 30 fixes in 6000 pairs is dC = +0.005: highly
    significant (nothing broke) and FAR below the 2 pp threshold. Significance is not size."""
    r = _rigged(6000, n_up=30, n_down=0)
    assert r["delta_c"] == pytest.approx(0.005)
    assert r["exact_signflip_p_two_sided"] < 1e-6
    assert r["powered_for_target"] is True
    assert r["verdict"] == "NO_ADVANCE" and r["advanced"] is False


def test_a_threshold_sized_gain_that_is_not_significant_is_not_a_pass():
    """Kills the deletion of the `p_exact < ALPHA` conjunct: dC clears 2 pp on a coin-flip
    split of the discordances, which is exactly what noise looks like."""
    r = _rigged(400, n_up=70, n_down=62)
    assert r["delta_c"] >= oi.ADVANCE_DELTA_C
    assert r["exact_signflip_p_two_sided"] > oi.ALPHA
    assert not r["verdict"].startswith("PASS") and r["advanced"] is False


def test_a_regression_smaller_than_the_threshold_is_not_convicted():
    """Kills `delta <= -target` -> `delta < 0`: a significant but sub-threshold loss must not
    be reported as REGRESSED, for the same reason a sub-threshold gain is not a PASS."""
    r = _rigged(6000, n_up=0, n_down=30)
    assert r["delta_c"] == pytest.approx(-0.005)
    assert r["exact_signflip_p_two_sided"] < 1e-6
    assert r["verdict"] == "NO_ADVANCE"


def test_advanced_is_true_for_pass_alone_and_not_for_pass_pending_remeasure():
    """Kills `advanced = verdict.startswith("PASS")`. PASS_PENDING_REMEASURE exists precisely
    to NOT advance an arm."""
    powered = _rigged(6000, n_up=200, n_down=20)
    assert powered["verdict"] == "PASS" and powered["advanced"] is True
    small = _rigged(129, n_up=8, n_down=0)
    assert small["delta_c"] >= oi.ADVANCE_DELTA_C
    assert small["exact_signflip_p_two_sided"] < oi.ALPHA
    assert small["powered_for_target"] is False
    assert small["verdict"] == "PASS_PENDING_REMEASURE"
    assert small["advanced"] is False


def test_the_verdict_is_one_of_the_declared_set():
    for r in (_rigged(6000, 200, 20), _rigged(400, 70, 62), _rigged(129, 8, 0),
              _rigged(6000, 0, 400), _rigged(6000, 0, 0)):
        assert r["verdict"] in oi.VERDICTS


# ----------------------------------------------------------------------------------------
# the abstention hole half credit opens, and the guard that closes it
# ----------------------------------------------------------------------------------------
def test_manufacturing_ties_where_the_head_is_wrong_cannot_buy_advancement(cell):
    """The mirror image of the drop-ties laundering, found by adversarial review: half credit
    pays +0.5 a pair for ABSTAINING on pairs the head got wrong. Naive dC is large and
    positive; the gate reads the version that scores every manufactured tie as wrong, where
    the attack is worth nothing."""
    r = cmp_at(cell, demo(cell, "abstention_gaming_label_aimed_ties"))
    assert r["delta_c"] > 0.05, "the attack must really work against naive scoring"
    assert r["head_ties_created_by_arm"] > 0
    assert r["discordant_full_swings"] == 0, "it reorders nothing"
    assert r["delta_c_new_ties_scored_wrong"] <= 0.0
    assert r["delta_c_gate"] == r["delta_c_new_ties_scored_wrong"]
    assert not r["verdict"].startswith("PASS") and r["advanced"] is False


def test_the_p_reported_with_the_gated_statistic_is_that_statistic_s_p(cell):
    """The count-test defect in miniature, applied to this guard: if the gate reads the
    tie-guarded dC it must report the tie-guarded p, not the naive one. A mutation that reverted
    only the p half survived an earlier suite because the two were computed independently."""
    r = cmp_at(cell, demo(cell, "abstention_gaming_label_aimed_ties"))
    assert r["delta_c_gate"] == r["delta_c_new_ties_scored_wrong"] != r["delta_c"]
    assert r["p_gate"] == r["exact_signflip_p_new_ties_scored_wrong"]
    assert r["p_gate"] != r["exact_signflip_p_two_sided"]
    # and the naive p is the one that looked like overwhelming evidence
    assert r["exact_signflip_p_two_sided"] < 1e-6 < r["p_gate"]


def test_the_signflip_test_is_invariant_to_a_common_rescale_of_the_differences():
    """The statistic is a randomization test on signs, so scaling every difference must not move
    the p -- including across the exact/approximate branch choice, which normalises by the
    smallest magnitude for exactly this reason."""
    diffs = [1.0, 1.0, -0.5, 1.0, -0.5, -0.5, 0.5]
    base = oi.exact_signflip_p(diffs)
    for factor in (0.5, 2.0, 100.0, 0.013):
        assert oi.exact_signflip_p([d * factor for d in diffs]) == pytest.approx(base, rel=1e-12)


def test_the_abstention_guard_is_inert_for_an_arm_that_creates_no_ties(cell):
    """The other branch: a guard that always fires would just be a second threshold. A real
    ordering improvement creates no new ties, so the gate reads its plain dC untouched."""
    r = cmp_at(cell, demo(cell, "positive_control_blend_truth_0.15"))
    assert r["head_ties_created_by_arm"] == 0
    assert r["delta_c_gate"] == r["delta_c"]
    assert r["p_gate"] == r["exact_signflip_p_two_sided"]


def test_a_powered_abstention_attack_is_refused_rather_than_advanced():
    """At a bank size where the attack WOULD be powered, the guard -- not the power rule -- is
    what stops it."""
    rows, arm = {}, {}
    for i in range(6000):
        k = (i, 0, "p1")
        wrong = i < 1800
        ha, hb = (-0.2, 0.2) if wrong else (0.2, -0.2)
        rows[k] = {"seed": i, "prefix": 0, "seat": "p1", "head_a": ha, "head_b": hb,
                   "head_gap": oi.head_gap_from_values(ha, hb), "true_gap": 0.3,
                   "true_a": 0.65, "true_b": 0.35, "noise_var": 0.004,
                   "rollouts_a": 64, "rollouts_b": 64}
        arm[k] = (dict(rows[k], head_a=0.0, head_b=0.0, head_gap=0.0) if wrong
                  else dict(rows[k]))
    keys = sorted(rows)
    r = oi.compare(rows, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=20)
    assert r["delta_c"] == pytest.approx(0.15)
    assert r["powered_for_target"] is True
    assert r["delta_c_gate"] == 0.0
    assert r["verdict"] == "NO_ADVANCE" and r["advanced"] is False


# ----------------------------------------------------------------------------------------
# the paired test
# ----------------------------------------------------------------------------------------
def test_the_count_test_points_the_wrong_way_when_swings_are_mixed():
    """THE input that broke the first version of this gate, kept as a regression test.

    900 pairs where the arm fixes an ordering (+1.0 each) and 1500 where it turns a right
    ordering into a tie (-0.5 each): dC = +0.0250, but the discordance COUNT test sees 1500 vs
    900 and returns p ~ 1e-34 in the BASELINE's favour. The first version read PASS off that.
    The sign-flip test is computed on the statistic being reported, so it cannot disagree with
    it in direction.
    """
    r = _rigged_mixed()
    assert r["delta_c"] == pytest.approx(0.025)
    assert r["head_ties_created_by_arm"] == 0, "keep the abstention guard out of this test"
    assert r["mcnemar_count_test_p_sensitivity_only"] < 1e-20   # the broken headline
    assert r["discordant_baseline_better"] > r["discordant_arm_better"]  # counts say "worse"
    # The shipped test is computed on the reported statistic, so it agrees with it in direction:
    # here the +2.5 pp really is significant (z ~ 4.2 on sum-of-differences), and that -- not a
    # count of discordances pointing the other way -- is what the PASS rests on.
    assert r["exact_signflip_p_two_sided"] < oi.ALPHA
    assert r["exact_signflip_p_two_sided"] > 1e-8, (
        "the count test's 1e-34 was an artefact of the wrong null, not real evidence")
    assert r["verdict"] == "PASS"


def _rigged_mixed():
    """Mixed swing sizes with NO tie manufacturing, so the abstention guard is inert and the
    only thing under test is the p-value. 900 pairs the arm fixes outright (+1.0 each) and 1500
    where the BASELINE was tied and the arm resolves them wrongly (-0.5 each): dC = +0.0250
    while the discordance counts are 1500 to 900 against the arm."""
    rows, arm = {}, {}
    for i in range(6000):
        k = (i, 0, "p1")
        if i < 900:
            ha, hb, aa, ab = -0.2, 0.2, 0.2, -0.2        # base wrong -> arm right (+1.0)
        elif i < 2400:
            ha, hb, aa, ab = 0.1, 0.1, -0.2, 0.2         # base TIED -> arm wrong (-0.5)
        else:
            ha, hb, aa, ab = 0.2, -0.2, 0.2, -0.2
        rows[k] = {"seed": i, "prefix": 0, "seat": "p1", "head_a": ha, "head_b": hb,
                   "head_gap": oi.head_gap_from_values(ha, hb), "true_gap": 0.3,
                   "true_a": 0.65, "true_b": 0.35, "noise_var": 0.004,
                   "rollouts_a": 64, "rollouts_b": 64}
        arm[k] = dict(rows[k], head_a=aa, head_b=ab,
                      head_gap=oi.head_gap_from_values(aa, ab))
    return oi.compare(rows, arm, keys=sorted(rows), tau=oi.TAU_PRIMARY, bootstrap_reps=20)


def test_the_signflip_test_reduces_exactly_to_mcnemar_when_every_swing_is_full():
    """It must be a generalisation, not a different test: on full swings the two agree to the
    last bit, so the doc's 'exact McNemar' requirement is met and only extended."""
    for b, c in ((0, 5), (1, 9), (3, 7), (5, 5), (12, 8), (20, 0), (0, 0)):
        diffs = [-1.0] * b + [1.0] * c
        assert oi.exact_signflip_p(diffs) == pytest.approx(oi.exact_mcnemar_p(b, c), rel=1e-12)


def test_the_signflip_null_distribution_is_exact_for_half_swings():
    """Hand-computable case: two half swings in the same direction. Sum = +1.0; under sign
    flips the reachable sums are -1, 0, 0, +1, so P(|S| >= 1) = 2/4."""
    assert oi.exact_signflip_p([0.5, 0.5]) == pytest.approx(0.5)
    # one full and one half in the same direction: sums +-1.5, +-0.5 -> P(|S|>=1.5) = 2/4
    assert oi.exact_signflip_p([1.0, 0.5]) == pytest.approx(0.5)
    # three halves all one way: sums +-1.5, +-0.5 with multiplicities 1,3 -> 2/8
    assert oi.exact_signflip_p([0.5, 0.5, 0.5]) == pytest.approx(0.25)


def test_the_signflip_large_m_fallback_tracks_the_exact_value():
    """The approximation branch had zero coverage. Compare it to the exact enumeration just
    below the switchover, on the same data."""
    diffs = [1.0] * 220 + [-1.0] * 180
    exact = oi.exact_signflip_p(diffs)                       # m = 400, still exact
    approx = oi.exact_signflip_p(diffs + [0.0] * 5)          # zeros are dropped: same data
    assert exact == pytest.approx(approx, rel=1e-12)
    big = [1.0] * 620 + [-1.0] * 580                          # m = 1200 -> normal branch
    z = 40 / math.sqrt(1200)
    assert oi.exact_signflip_p(big) == pytest.approx(math.erfc(z / math.sqrt(2)), rel=1e-9)


def test_the_mcnemar_normal_branch_tracks_the_exact_value_at_the_cutoff():
    b, c = 1100, 900
    exact = oi.exact_mcnemar_p(b, c)                # n = 2000, exact branch
    approx = oi.exact_mcnemar_p(b + 1, c)          # n = 2001, normal branch
    assert exact == pytest.approx(approx, abs=2e-3)
    assert 0.0 < approx < 1.0


def test_exact_mcnemar_matches_hand_computed_binomial_tails():
    assert oi.exact_mcnemar_p(0, 0) == 1.0
    assert oi.exact_mcnemar_p(0, 5) == pytest.approx(2 * (1 / 32))
    assert oi.exact_mcnemar_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)
    assert oi.exact_mcnemar_p(5, 5) == 1.0
    assert oi.exact_mcnemar_p(3, 7) == oi.exact_mcnemar_p(7, 3)          # symmetric
    assert oi.exact_mcnemar_p(20, 0) < oi.exact_mcnemar_p(12, 8)


def test_the_test_is_paired_which_is_the_whole_point_at_this_n(cell):
    """A five-ordering change out of 129 pairs is invisible to an unpaired comparison of two
    C's and perfectly visible to the paired one. This programme has applied unpaired tests to
    paired designs before; the asymmetry below is why that mattered."""
    rows, keys = cell
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    flip = [k for k in sel if rows[k]["head_gap"] * rows[k]["true_gap"] > 0][:9]
    arm = {k: (dict(rows[k], head_a=rows[k]["head_b"], head_b=rows[k]["head_a"],
                    head_gap=-rows[k]["head_gap"]) if k in flip else rows[k]) for k in keys}
    r = oi.compare(rows, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=200)
    assert r["discordant_total"] == 9 and r["discordant_baseline_better"] == 9
    assert r["exact_signflip_p_two_sided"] == pytest.approx(2 / 512)
    # the unpaired two-proportion z on the same data cannot see it
    n = r["n_eligible"]
    p1, p2 = r["c_baseline"], r["c_arm"]
    pooled = (p1 + p2) / 2
    z = (p1 - p2) / math.sqrt(2 * pooled * (1 - pooled) / n)
    assert abs(z) < 1.96, "the unpaired test should MISS what the paired test finds"


def test_half_swings_from_tie_creation_are_counted_as_discordances(cell):
    r = cmp_at(cell, demo(cell, "saturating_tanh50"))
    assert r["discordant_half_swings_tie_transitions"] > 0
    assert r["mean_swing"] < 1.0
    assert (r["discordant_full_swings"] + r["discordant_half_swings_tie_transitions"]
            == r["discordant_total"])


def test_the_full_swing_only_p_is_reported_as_a_sensitivity_not_the_headline(cell):
    r = cmp_at(cell, demo(cell, "saturating_tanh50"))
    # all of this demo's discordances are tie transitions, so restricting to full swings
    # throws the entire signal away -- which is exactly why it is the sensitivity and not
    # the primary.
    assert r["exact_mcnemar_p_full_swings_only"] == 1.0
    assert r["exact_signflip_p_two_sided"] < 1.0


def test_delta_c_equals_the_mean_per_pair_swing(cell):
    rows, keys = cell
    arm = demo(cell, "ordering_corrupted_25pct")
    r = cmp_at(cell, arm)
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    by_hand = sum(oi.pair_score(arm[k]["head_gap"], rows[k]["true_gap"])
                  - oi.pair_score(rows[k]["head_gap"], rows[k]["true_gap"])
                  for k in sel) / len(sel)
    assert r["delta_c"] == pytest.approx(by_hand)


# ----------------------------------------------------------------------------------------
# power: the guard must be able to read BOTH ways
# ----------------------------------------------------------------------------------------
def test_the_gate_refuses_a_verdict_at_the_banked_n(cell):
    """A 465-pair bank cannot see a 2 pp ordering change. The gate must say so instead of
    emitting 'no advancement', which reads like a finding and is a power failure."""
    rows, keys = cell
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    arm = {k: rows[k] for k in keys}
    # a genuine but small improvement: flip 3 wrong orderings right, ~2 pp on 129 pairs
    fix = [k for k in sel if rows[k]["head_gap"] * rows[k]["true_gap"] < 0][:3]
    arm.update({k: dict(rows[k], head_a=rows[k]["head_b"], head_b=rows[k]["head_a"],
                        head_gap=-rows[k]["head_gap"]) for k in fix})
    r = oi.compare(rows, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=200)
    assert 0 < r["delta_c"] < 0.03
    assert r["powered_for_target"] is False
    assert r["verdict"] in ("UNDERPOWERED_NO_VERDICT", "PASS_PENDING_REMEASURE")
    assert r["advanced"] is False
    # and it must say what it would take, in the units the bank is measured in
    assert r["required_n_eligible_for_target"] > r["n_eligible"]
    assert r["required_n_banked_pairs_for_target"] > len(keys)


def test_the_power_guard_returns_a_real_pass_when_the_bank_is_big_enough():
    """The other branch. Without this, 'UNDERPOWERED_NO_VERDICT' could be unconditional and
    the guard would certify nothing -- the exact defect the failing-input rule exists for."""
    rows_l = make_pairs(n=12000, seed=5, n_games=2000, zero_frac=0.20)
    rows = {oi.pair_key(p): p for p in rows_l}
    keys = sorted(rows)
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)
    target_flips = int(0.03 * len(sel))
    fix = [k for k in sel if rows[k]["head_gap"] * rows[k]["true_gap"] < 0][:target_flips]
    arm = dict(rows)
    arm.update({k: dict(rows[k], head_a=rows[k]["head_b"], head_b=rows[k]["head_a"],
                        head_gap=-rows[k]["head_gap"]) for k in fix})
    r = oi.compare(rows, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=200)
    assert r["n_eligible"] > 2000
    assert r["powered_for_target"] is True
    assert r["verdict"] == "PASS"
    assert r["advanced"] is True


def test_the_powered_bank_still_refuses_the_gaming_transforms():
    """Power is not a laundry: a bigger bank must not turn a rescale into an advancement."""
    rows_l = make_pairs(n=6000, seed=6, n_games=1000)
    rows = {oi.pair_key(p): p for p in rows_l}
    keys = sorted(rows)
    for name in ("affine_rescale_k3.2764", "monotone_tanh2", "additive_offset_plus0.2"):
        arm = oi.build_demo_cell(rows, keys, name)
        r = oi.compare(rows, arm, keys, oi.TAU_PRIMARY, bootstrap_reps=100)
        assert r["delta_c"] == 0.0 and r["verdict"] == "NO_ADVANCE_INVARIANT", name


def test_mde_is_the_effect_a_simulation_detects_80_percent_of_the_time():
    """The power formula, simulated rather than asserted. A metric nobody fed a known answer
    to has failed to measure its own name three times in this programme."""
    n_elig, rate = 4000, 0.20
    mde = oi.mde_delta_c(n_elig, int(rate * n_elig))
    n_d = int(rate * n_elig)
    q = 0.5 + mde * n_elig / (2 * n_d)          # invert dC = rate*(2q-1)
    rng = random.Random(7)
    hits = 0
    reps = 400
    for _ in range(reps):
        c = sum(1 for _ in range(n_d) if rng.random() < q)
        if oi.exact_mcnemar_p(n_d - c, c) < oi.ALPHA:
            hits += 1
    assert 0.70 <= hits / reps <= 0.90, hits / reps


def test_required_n_inverts_the_mde_formula():
    for rate in (0.05, 0.2, 0.5):
        n = oi.required_n_eligible(0.02, rate)
        assert oi.mde_delta_c(n, round(rate * n)) == pytest.approx(0.02, rel=0.02)


def test_fewer_discordances_make_a_given_delta_easier_not_harder_to_see():
    """The counter-intuitive direction of the formula, pinned so a future 'fix' cannot quietly
    invert it: the same dC concentrated in fewer changed orderings is a more lopsided majority
    and therefore MORE detectable."""
    assert oi.mde_delta_c(1000, 50) < oi.mde_delta_c(1000, 400)


def test_the_clustered_bootstrap_widens_when_pairs_share_a_game():
    """The exact test assumes independent pairs; the bank has ~6 per game. If the clustered
    interval were not wider under strong clustering, it would not be measuring clustering."""
    n = 600
    rng = random.Random(3)
    per_game = {g: (1 if rng.random() < 0.5 else 0) for g in range(n // 6 + 1)}
    base, arm, tight, loose = [], [], [], []
    for i in range(n):
        game = i // 6
        base.append(1.0)
        arm.append(1.0 - per_game[game])   # the effect is COMMON to a whole game
        tight.append(i)                    # every pair its own cluster
        loose.append(game)
    lo_t, hi_t = oi.clustered_bootstrap_ci(base, arm, tight, reps=400)
    lo_l, hi_l = oi.clustered_bootstrap_ci(base, arm, loose, reps=400)
    assert (hi_l - lo_l) > (hi_t - lo_t)


# ----------------------------------------------------------------------------------------
# selection on a noisy label: the scope of the label-SE rule, measured
# ----------------------------------------------------------------------------------------
def test_noisy_selection_attenuates_and_cannot_manufacture_a_delta():
    """The programme's spec asks for label SE << tau, and at R=64 that condition is NOT met.
    This measures what the violation actually does to the PAIRED statistic -- which is the
    whole basis for reporting absolute levels while gating only on dC.

    (a) A monotone-transformed arm scores dC = 0 EXACTLY on the noisily-selected set: a noisy
        label cannot MANUFACTURE a paired difference, because the selection is COMMON to both
        cells and the statistic is a difference on identical pairs.
    (b) On that same set, a genuinely better arm's dC scored against the NOISY label is a
        SHRUNK version of its dC against the latent label. The shrinkage has a closed form:
        the label's sign is wrong on a fraction f of the selected pairs, and a wrong label
        reverses the credit, so the expected factor is (1 - 2f). Both the measured ratio and
        (1 - 2f) are asserted, so this is a known-answer check and not a vibe.

    The eligible set is held FIXED between the two scorings on purpose. Comparing
    "selected on noisy" against "selected on latent" would compare two different populations --
    the latent-selected set is easier (bigger true gaps, so the head is right more often and
    there is less room to improve), and that ceiling effect can make the noisy number look
    LARGER, which is a selection artefact and not an inflation of the estimator.
    """
    rng = random.Random(19)
    n = 40000
    rows = {}
    for i in range(n):
        latent = rng.gauss(0, 0.10)
        noisy = latent + rng.gauss(0, 0.0884)      # R=64 worst-case gap SE
        hg = 0.30 * latent + rng.gauss(0, 0.02)
        rows[(i, 0, "p1")] = {
            "seed": i, "prefix": 0, "seat": "p1",
            "head_a": hg, "head_b": -hg, "head_gap": hg,
            "true_gap": noisy, "_latent": latent,
            "true_a": 0.5, "true_b": 0.5 - noisy,
            "noise_var": 0.0884 ** 2, "rollouts_a": 64, "rollouts_b": 64}
    keys = sorted(rows)

    # (a) cannot manufacture
    arm_mono = oi.build_demo_cell(rows, keys, "monotone_tanh2")
    assert oi.compare(rows, arm_mono, keys, oi.TAU_PRIMARY,
                      bootstrap_reps=50)["delta_c"] == 0.0

    # (b) a real improvement: halve the head's idiosyncratic error
    arm = {}
    for k in keys:
        p = rows[k]
        hg = 0.30 * p["_latent"] + 0.5 * (p["head_gap"] - 0.30 * p["_latent"])
        arm[k] = dict(p, head_a=hg, head_b=-hg, head_gap=hg)
    sel = oi.eligible_keys(rows, keys, oi.TAU_PRIMARY)        # selected ONCE, on the noisy gap
    latent_rows = {k: dict(rows[k], true_gap=rows[k]["_latent"]) for k in keys}
    latent_arm = {k: dict(arm[k], true_gap=rows[k]["_latent"]) for k in keys}

    def d(base_rows, arm_rows):
        return (oi.concordance(oi.score_cell(arm_rows, sel))
                - oi.concordance(oi.score_cell(base_rows, sel)))

    d_noisy = d(rows, arm)
    d_latent = d(latent_rows, latent_arm)
    flipped = sum(1 for k in sel
                  if (rows[k]["true_gap"] > 0) != (rows[k]["_latent"] > 0)) / len(sel)
    assert 0.05 < flipped < 0.30, flipped
    assert d_latent > 0 and d_noisy > 0
    attenuation = d_noisy / d_latent
    assert 0.4 < attenuation < 1.0, attenuation
    assert attenuation == pytest.approx(1 - 2 * flipped, abs=0.12), (attenuation, flipped)


def test_the_label_se_note_reports_the_condition_as_unmet_rather_than_waiving_it(cell):
    rows, keys = cell
    note = oi.label_se_note(rows, keys, oi.TAU_PRIMARY)
    assert note["condition_label_se_much_less_than_tau_met"] is False
    assert note["label_gap_se_banked_mean"] == pytest.approx(0.065, abs=1e-6)
    assert note["label_gap_se_worst_case"] == pytest.approx(0.5 * math.sqrt(2 / 64))
    assert note["se_over_tau_banked"] == pytest.approx(0.65, abs=0.01)
    assert "IS NOT MET" in note["text"]


def test_the_label_se_condition_reads_true_on_a_bank_that_actually_meets_it():
    """The other branch. A review found this field hardcoded to False with a test that asserted
    False -- a check that cannot read True certifies nothing about any future bank, and the
    whole point of the field is that a larger-R bank would flip it."""
    rows = {oi.pair_key(p): dict(p, noise_var=0.005 ** 2) for p in make_pairs(n=60)}
    keys = sorted(rows)
    note = oi.label_se_note(rows, keys, oi.TAU_PRIMARY)
    assert note["se_over_tau_banked"] == pytest.approx(0.05)
    assert note["condition_label_se_much_less_than_tau_met"] is True
    assert "IS MET" in note["text"] and "IS NOT MET" not in note["text"]


# ----------------------------------------------------------------------------------------
# refusals, each shown refusing
# ----------------------------------------------------------------------------------------
def test_it_refuses_a_cell_whose_head_gap_is_not_the_difference_of_its_head_values(tmp_path):
    """The transform surface guard. If head_gap is not (head_a-head_b)/2, then transforming
    the values and recomputing the gap scores a head nobody ran -- and the two non-linear
    demos would be measuring an artefact."""
    rows = make_pairs(n=40)
    rows[7]["head_gap"] = rows[7]["head_gap"] + 0.05
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "bad", rows), "bad")
    assert "head_gap" in str(e.value) and "REFUSING" in str(e.value)


def test_the_gap_convention_is_pinned_to_a_literal_from_the_banked_schema(tmp_path):
    """A mutation check found this hole: every other test derives head_gap through
    `head_gap_from_values`, so changing GAP_SCALE changed the fixtures too and no test noticed.
    The banked schema is head_gap = (head_a - head_b)/2 -- head values are return-scale in
    [-1,1] and true_gap is a win-probability difference -- so the convention is pinned here
    against a LITERAL, not against the function under test.
    """
    assert oi.head_gap_from_values(0.4, 0.1) == pytest.approx(0.15)
    rows = [{"seed": 1, "prefix": 0, "seat": "p1", "head_a": 0.4, "head_b": 0.1,
             "head_gap": 0.15, "true_gap": 0.2, "true_a": 0.6, "true_b": 0.4,
             "noise_var": 0.004, "rollouts_a": 64, "rollouts_b": 64}]
    assert len(oi.load_cell(write(tmp_path, "literal", rows), "literal")) == 1
    rows[0]["head_gap"] = 0.30                      # the return-scale difference, not the gap
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "wrongscale", rows), "wrongscale")
    assert "REFUSING" in str(e.value)


def test_it_accepts_the_same_file_once_the_gap_is_consistent(tmp_path):
    rows = make_pairs(n=40)
    assert len(oi.load_cell(write(tmp_path, "ok", rows), "ok")) == 40


def test_it_refuses_cells_that_disagree_on_the_reused_ground_truth(tmp_path):
    a = make_pairs(n=60, seed=1)
    b = [dict(p) for p in a]
    b[3]["true_gap"] = b[3]["true_gap"] + 0.01
    cells = {"ref": oi.load_cell(write(tmp_path, "a", a), "ref"),
             "arm": oi.load_cell(write(tmp_path, "b", b), "arm")}
    with pytest.raises(SystemExit) as e:
        oi.align(cells, "ref")
    assert "true_gap column differs" in str(e.value)


def test_it_refuses_a_cell_scored_on_a_different_pair_set(tmp_path):
    a = make_pairs(n=60, seed=1)
    cells = {"ref": oi.load_cell(write(tmp_path, "a", a), "ref"),
             "arm": oi.load_cell(write(tmp_path, "b", a[:-2]), "arm")}
    with pytest.raises(SystemExit) as e:
        oi.align(cells, "ref")
    assert "same pairs" in str(e.value)


def test_it_refuses_a_mixed_rollout_budget(tmp_path):
    rows = make_pairs(n=40)
    rows[5]["rollouts_b"] = 16
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "mixed", rows), "mixed")
    assert "R is pinned" in str(e.value)


def test_it_refuses_a_file_carrying_only_head_gap(tmp_path):
    rows = make_pairs(n=40)
    for p in rows:
        p.pop("head_a")
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "gaponly", rows), "gaponly")
    assert "head_a" in str(e.value)


def test_it_refuses_a_non_finite_head_value(tmp_path):
    rows = make_pairs(n=40)
    rows[2]["head_a"] = float("nan")
    rows[2]["head_gap"] = float("nan")
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "nan", rows), "nan")
    assert "non-finite" in str(e.value)


def test_it_refuses_duplicate_pair_identities(tmp_path):
    rows = make_pairs(n=40)
    rows.append(dict(rows[0]))
    with pytest.raises(SystemExit) as e:
        oi.load_cell(write(tmp_path, "dupe", rows), "dupe")
    assert "duplicate pair identity" in str(e.value)


def test_it_refuses_an_empty_pairs_array(tmp_path):
    with pytest.raises(SystemExit):
        oi.load_cell(write(tmp_path, "empty", []), "empty")


def test_it_refuses_a_tau_no_pair_can_satisfy(cell):
    rows, keys = cell
    with pytest.raises(SystemExit) as e:
        oi.compare(rows, rows, keys, 99.0, bootstrap_reps=10)
    assert "eligible" in str(e.value)


def test_it_refuses_a_negative_tau(cell):
    rows, keys = cell
    with pytest.raises(ValueError):
        oi.eligible_keys(rows, keys, -0.1)


# ----------------------------------------------------------------------------------------
# public-repo hygiene: the artifact must not carry the bank's filesystem layout
# ----------------------------------------------------------------------------------------
def test_provenance_records_the_basename_and_hash_and_never_the_directory(tmp_path):
    nested = tmp_path / "shared" / "scott-experiment" / "phase3-valuetune-20260816"
    nested.mkdir(parents=True)
    path = nested / "pairs-v1.json"
    path.write_text(json.dumps({"pairs": make_pairs(n=10)}))
    prov = oi.provenance(path)
    assert prov["file"] == "pairs-v1.json"
    assert len(prov["sha256"]) == 64
    blob = json.dumps(prov)
    assert "/" not in blob and "scott-experiment" not in blob


def test_the_cli_writes_an_artifact_with_no_absolute_paths_in_it(tmp_path, monkeypatch):
    nested = tmp_path / "shared" / "bank"
    nested.mkdir(parents=True)
    rows = make_pairs(n=200, seed=4)
    (nested / "base.json").write_text(json.dumps({"pairs": rows}))
    arm = [dict(p) for p in rows]
    (nested / "arm.json").write_text(json.dumps({"pairs": arm}))
    out = tmp_path / "board.json"
    monkeypatch.setattr("sys.argv", [
        "value_head_ordering_auc.py",
        "--ref", f"base={nested / 'base.json'}",
        "--cell", f"arm={nested / 'arm.json'}",
        "--bootstrap-reps", "50", "--json", str(out)])
    assert oi.main() == 0
    text = out.read_text()
    assert str(tmp_path) not in text and "/shared" not in text
    doc = json.loads(text)
    assert doc["cells"]["arm"][str(oi.TAU_PRIMARY)]["delta_c"] == 0.0


def test_the_cli_states_its_three_standing_caveats_in_the_artifact(tmp_path, monkeypatch):
    rows = make_pairs(n=200, seed=4)
    (tmp_path / "base.json").write_text(json.dumps({"pairs": rows}))
    out = tmp_path / "board.json"
    monkeypatch.setattr("sys.argv", [
        "value_head_ordering_auc.py", "--ref", f"base={tmp_path / 'base.json'}",
        "--bootstrap-reps", "10", "--json", str(out)])
    assert oi.main() == 0
    doc = json.loads(out.read_text())
    cav = doc["caveats"]
    assert "REALIZED REPLY" in cav["conditioned_on_realized_reply"]
    assert "UPPER BOUND" in cav["conditioned_on_realized_reply"]
    assert "NEVER GATED" in cav["absolute_levels_never_gated"]
    assert cav["label_se_vs_tau"]["condition_label_se_much_less_than_tau_met"] is False
    assert "NECESSARY, NOT SUFFICIENT" in cav["advancement_requires_supporting_beta_ece"]
    assert doc["preregistered"]["tau_primary"] == 0.10
    assert doc["preregistered"]["head_side_ties"].startswith("score 0.5")


def test_the_cli_demo_mode_reports_that_no_gaming_input_passed(tmp_path, monkeypatch):
    rows = make_pairs(n=465, seed=11)
    (tmp_path / "base.json").write_text(json.dumps({"pairs": rows}))
    out = tmp_path / "demos.json"
    monkeypatch.setattr("sys.argv", [
        "value_head_ordering_auc.py", "--ref", f"base={tmp_path / 'base.json'}",
        "--demos", "--bootstrap-reps", "10", "--json", str(out)])
    assert oi.main() == 0
    doc = json.loads(out.read_text())
    assert doc["demo_summary"]["gaming_or_corruption_inputs_that_PASSED"] == []
    assert all(v > 0 for v in
               doc["demo_summary"]["positive_controls_delta_c_at_tau_primary"].values())
    assert set(doc["demos"]) == set(oi.DEMOS)


def test_the_cli_refuses_a_malformed_cell_spec(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["value_head_ordering_auc.py", "--ref", "justaname"])
    with pytest.raises(SystemExit) as e:
        oi.main()
    assert "NAME=FILE" in str(e.value)


def test_the_cli_refuses_two_cells_with_the_same_name(tmp_path, monkeypatch):
    rows = make_pairs(n=40)
    (tmp_path / "a.json").write_text(json.dumps({"pairs": rows}))
    monkeypatch.setattr("sys.argv", [
        "value_head_ordering_auc.py", "--ref", f"x={tmp_path / 'a.json'}",
        "--cell", f"x={tmp_path / 'a.json'}"])
    with pytest.raises(SystemExit) as e:
        oi.main()
    assert "duplicate cell name" in str(e.value)


# ----------------------------------------------------------------------------------------
# the preregistered spec must be in the file, not in a command line
# ----------------------------------------------------------------------------------------
def test_the_pinned_spec_is_not_overridable_from_the_command_line():
    """tau, R and the pass threshold are the gate. If they were flags with defaults, the gate
    would be whatever the last invocation chose."""
    src = (REPO / "scripts" / "value_head_ordering_auc.py").read_text()
    for forbidden in ("--tau", "--threshold", "--rollouts", "--drop-ties", "--alpha",
                      "--target"):
        # the quoted form is how argparse would receive it; the bare string also appears in
        # prose explaining why `--drop-ties` does NOT exist, and that prose is the point.
        assert f'"{forbidden}"' not in src, f"{forbidden} must not be a CLI flag"
    assert oi.TAU_PRIMARY == 0.10
    assert oi.TAU_CHECKS == (0.05, 0.15)
    assert oi.PINNED_ROLLOUTS == 64
    assert oi.ADVANCE_DELTA_C == 0.02


def test_the_tau_checks_carry_no_independent_p_claim(tmp_path, monkeypatch):
    rows = make_pairs(n=400, seed=4)
    (tmp_path / "base.json").write_text(json.dumps({"pairs": rows}))
    arm = [dict(p, head_a=p["head_b"], head_b=p["head_a"],
                head_gap=-p["head_gap"]) if i % 9 == 0 else dict(p)
           for i, p in enumerate(rows)]
    (tmp_path / "arm.json").write_text(json.dumps({"pairs": arm}))
    out = tmp_path / "b.json"
    monkeypatch.setattr("sys.argv", [
        "value_head_ordering_auc.py", "--ref", f"base={tmp_path / 'base.json'}",
        "--cell", f"arm={tmp_path / 'arm.json'}",
        "--bootstrap-reps", "10", "--json", str(out)])
    oi.main()
    doc = json.loads(out.read_text())
    assert doc["preregistered"]["tau_checks"] == [0.05, 0.15]
    assert "no independent p" in (REPO / "scripts" / "value_head_ordering_auc.py").read_text()
    cell = doc["cells"]["arm"]
    assert cell["0.1"]["is_primary_tau"] is True
    assert "p_is_a_sign_consistency_check_not_an_independent_test" not in cell["0.1"]
    for t in ("0.05", "0.15"):
        assert cell[t]["p_is_a_sign_consistency_check_not_an_independent_test"] is True
    sc = cell["sign_consistency"]
    assert sc["primary_sign"] == -1                     # the corruption is a regression
    assert sc["all_nonzero_signs_agree"] is True


def test_sign_consistency_flags_a_direction_that_flips_with_tau():
    disagree = oi.sign_consistency({"0.1": {"delta_c": 0.03}, "0.05": {"delta_c": -0.01},
                                    "0.15": {"delta_c": 0.02}})
    assert disagree["all_nonzero_signs_agree"] is False
    agree = oi.sign_consistency({"0.1": {"delta_c": 0.03}, "0.05": {"delta_c": 0.0},
                                 "0.15": {"delta_c": 0.02}})
    assert agree["all_nonzero_signs_agree"] is True     # an exact zero is not a disagreement
