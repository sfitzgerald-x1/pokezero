#!/usr/bin/env python3
"""OI-1 — the ordering instrument. The Phase 3 advancement gate on the ONE property
search consumes from the value head: the order it puts counterfactual siblings in.

WHY THIS EXISTS, stated as the failure it replaces. The first Phase 3 arms were scored on
beta (sibling-gap compression) plus ECE (calibration), and the campaign proved the pair
JOINTLY GAMEABLE with zero ordering improvement:

  * beta by output scale. A 513-parameter control with NO mechanism change moved beta by
    -0.0202 -- larger than the candidate arm's signal -- purely by contracting its output
    spread 7.5% with R^2 flat. `sibling_gap_compare.py`'s rescale test was built to name
    exactly that, and it did.
  * ECE by monotone recalibration. A constant offset shrinks ECE and CANCELS EXACTLY in a
    sibling comparison: it is subtracted away in every gap.

Neither statistic reads ordering. Search does not consume the head's level, its spread, or
its calibration -- it consumes argmax over siblings. So the gate is a pairwise AUC, which is
invariant to every strictly monotone reparameterisation of the head's output and therefore
defeats both named gaming vectors BY CONSTRUCTION rather than by threshold-tuning.

THE STATISTIC (preregistered, see the constants below):

    C(tau) = (1/n) * sum_i [ 1{sign(head_gap_i) == sign(true_gap_i)} + 0.5 * 1{head_gap_i == 0} ]

over pairs eligible at tau. This is the tie-aware AUC of head-gap against the sign label.

    HEAD-SIDE TIES SCORE 0.5 AND ARE NEVER DROPPED. That rule is load-bearing, not tidiness.
    A SATURATING recalibration -- tanh(50 v) -- is monotone-but-collapsing: it maps 197 of
    the bank's 465 pairs to an exact head-side tie. Under the shipped rule it scores
    dC = -0.0349 at tau=0.10. Under a drop-ties rule it scores +0.0554 and PASSES, having
    "improved" by DELETING the pairs it broke. The gate would then be gameable by a
    one-line output transform -- the same class of defect it was built to close. There is no
    `--drop-ties` flag in this tool on purpose; `tests/test_value_head_ordering_auc.py`
    computes the drop-ties number to prove the rule is what stops the laundering.

    AND THE MIRROR-IMAGE HOLE THAT HALF CREDIT OPENS IS CLOSED SEPARATELY. Half credit means
    an arm that goes to an exact tie precisely where it used to be WRONG collects +0.5 a pair
    for abstaining: measured +0.1357 (p = 5.8e-11, 35 manufactured ties) on the banked pairs
    by the review that found it, with zero orderings improved. So advancement must ALSO survive
    scoring every newly manufactured tie as WRONG (`delta_c_new_ties_scored_wrong`), and the
    verdict is read off the worse of the two. A real ordering improvement creates no new ties
    and is untouched; an abstention buys nothing. Aiming that attack needs the labels, but "the
    attacker would need the labels" is not a property of the statistic.

    PAIRS WITH true_gap == 0 ARE EXCLUDED, NEVER SCORED (95 of the banked 465). There is no
    ordering to be right about, so counting them as failures would penalise a perfect head
    and counting them as successes would reward a coin flip. `value_head_sibling_probe.py`
    already excludes them; this tool keeps the convention.

THE ADVANCEMENT STATISTIC IS PAIRED. dC = C_arm - C_baseline on the IDENTICAL eligible set,
tested by an EXACT SIGN-FLIP (randomization) test over the orderings that actually CHANGED --
which IS the exact McNemar test when every changed ordering moves C by the same amount, and is
the correct generalisation when some of them are tie transitions worth half a swing. The
discordance-COUNT version is reported as a sensitivity only; using it as the headline is what
an independent review broke (see `exact_signflip_p` for the input that made it certify PASS off
a p pointing at the baseline). Arms are rescores of one banked pair set: same states, same
reused ground truth. This programme has a documented history of applying unpaired tests to
paired designs (`sibling_gap_compare.py`'s header is the previous instance -- two overlapping
marginal CIs eyeballed as a difference test), so the paired test is the default here and the
unpaired one is not implemented. `grep -rn mcnemar` over this repo returned nothing before this
file: the paired tests quoted in earlier reports were computed ad hoc and are not reproducible
from the tree.

WHAT THIS INSTRUMENT DOES NOT MEASURE -- printed in its own output, every run, because it
bounds every number it produces:

  1. ORDERING GIVEN THE REALIZED REPLY. Both the label (true_a/true_b) and the head's input
     are conditioned on the opponent's single realized reply at that decision. Search
     integrates over the reply DISTRIBUTION, so C is an UPPER BOUND on the ordering quality
     search can cash. More rollouts per arm shrink the label's noise; they do not touch this
     conditioning. Only a label marginalised over replies would (the plan's backup-dilution
     item).
  2. ABSOLUTE LEVELS ARE DESCRIPTIVE, NEVER GATED. Eligibility selects on a NOISY measured
     gap (per-pair gap SE 0.0653 banked mean, 0.0884 worst case at R=64 -- both comparable to
     tau=0.10, NOT far below it). The ceiling programme's spec asks for label SE << tau; at
     the banked R that condition IS NOT MET at any tau this bank can support, and the tool
     says so on every run rather than quietly reporting C as if it were unbiased. What the
     rule protects is absolute-level claims, and those are reported and never gated.
     What transfers to the paired dC, stated at the strength it was actually measured at and
     no further: (i) two cells that differ by a strictly monotone reparameterisation score
     dC = 0 under ANY label, noisy or not -- an invariance, and trivially true, recorded so it
     is not mistaken for something stronger; (ii) for a genuinely better arm the measured dC is
     ATTENUATED toward the null, factor 0.78 at tau=0.10 (0.62 at 0.05, 0.88 at 0.15) against
     the closed form 1-2f at label-flip rate f = 9.0%. That is a bias toward "no advancement",
     i.e. conservative for a GATE. It is NOT a theorem that noise can never inflate a
     particular arm's point estimate: a single noisy draw can, and what bounds that is the
     exact test's alpha, not the attenuation.
  3. PAIRS CLUSTER BY GAME. 465 pairs come from 80 games at ~6 sampled decisions each, so
     the exact test's independence assumption is optimistic. A game-clustered bootstrap runs
     alongside every comparison; if its interval is materially wider than the exact test
     suggests, the exact p is anti-conservative and the clustered interval is the one to
     read.

THE GATE IS INERT BELOW ITS POWERED n, AND SAYS SO. At the banked 465 pairs -- 129 eligible
at tau=0.10 -- the achieved MDE across the nine banked arms is 3.1-9.0 pp (up to 15 pp on the
heavily-discordant demos) against arms that produce 1-5 pp. An instrument
that returned "no advancement" there would be reporting a power failure in the vocabulary of a
finding. So no arm can reach PASS unless the achieved MDE clears the preregistered target, the
required n is printed, and a significant-but-underpowered gain is labelled
PASS_PENDING_REMEASURE rather than promoted (its direction is established; its magnitude is
winner's-curse inflated, and the programme's own rule is that a favourable result triggers
re-measurement, not reporting). Per the failing-input rule the power guard ships with BOTH
branches demonstrated: it refuses at the banked n and returns a real PASS on a synthetic bank
large enough to earn one -- a guard that can only ever say "underpowered" certifies nothing
either.

Usage (the nine banked Phase 3 arms against the banked baseline):

    value_head_ordering_auc.py --ref base=vhprobe-pairs-20260815.json \\
        --cell v1=pairs-v1.json --cell v2=pairs-v2.json ... --json scoreboard.json

    value_head_ordering_auc.py --demos --ref base=vhprobe-pairs-20260815.json \\
        --demo-cell ctlL=pairs-ctlL.json --json demos.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# ----------------------------------------------------------------------------------------
# PREREGISTERED SPEC. The ceiling programme requires this instrument to "arrive pinned": R,
# the |true-gap| threshold tau, the pair-sampling distribution, and a preregistered pass
# threshold. Changing any value below changes the gate and belongs in a reviewed commit, not
# in a command line -- which is why none of them is a default-carrying CLI flag.
# ----------------------------------------------------------------------------------------
TAU_PRIMARY = 0.10
TAU_CHECKS = (0.05, 0.15)          # sign-consistency checks; NO independent p is claimed
PINNED_ROLLOUTS = 64               # R. Every banked pair must carry rollouts_a=rollouts_b=R
ADVANCE_DELTA_C = 0.02             # the preregistered pass threshold on dC
ALPHA = 0.05                       # two-sided, exact
POWER = 0.80
ASSUMED_DISCORDANCE_RATE = 0.20    # only for the "required n" projection when none observed
GAP_SCALE = 2.0                    # head_gap == (head_a - head_b) / GAP_SCALE (win-prob units)
MUCH_LESS_THAN_RATIO = 0.2         # the reading of "label SE << tau" this instrument commits to

# 0.5 * sqrt(2/R): the gap SE of the R-rollout label near p=0.5, i.e. the worst case. The
# banked mean is computed per run from noise_var and is smaller.
LABEL_GAP_SE_WORST_CASE = 0.5 * math.sqrt(2.0 / PINNED_ROLLOUTS)

PAIR_SAMPLING_DISTRIBUTION = (
    "top-2 sibling action pairs at 6 sampled decision prefixes per game, 80 self-play games "
    "(8 shards x 10), seat p1 only, one checkpoint, paired seeds (common random numbers) "
    "across the two arms of a pair; labels are R=64 policy-continuation rollouts to terminal "
    "and are win probabilities for the acting seat"
)

VERDICTS = (
    "PASS",                      # dC >= ADVANCE_DELTA_C, exact p < ALPHA, and powered
    "PASS_PENDING_REMEASURE",    # significant gain, but the design was not powered for the
                                 # target: the DIRECTION is established, the MAGNITUDE is
                                 # winner's-curse inflated. NOT an advancement -- the
                                 # programme's own rule is that a result moving in the
                                 # hoped-for direction triggers re-measurement, not reporting.
    "REGRESSED",                 # dC <= -ADVANCE_DELTA_C and exact p < ALPHA
    "NO_ADVANCE",                # powered for the target and no significant gain
    "NO_ADVANCE_INVARIANT",      # zero orderings changed: dC is exactly 0 by construction
    "UNDERPOWERED_NO_VERDICT",   # achieved MDE > target: the gate refuses to read
)
# The power rule is DELIBERATELY ASYMMETRIC, and the asymmetry is the conservative direction:
# certifying an advancement requires power for the preregistered target, while a significant
# REGRESSION is allowed to stand because it advances nothing. An underpowered rejection risks
# an inflated magnitude, not a wrong sign beyond alpha.


# ----------------------------------------------------------------------------------------
# statistics, stdlib only
# ----------------------------------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval. `k` may be fractional -- tie half-credit makes it so."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def exact_signflip_p(diffs: Sequence[float]) -> float:
    """THE primary test: exact paired randomization (sign-flip) test of H0: dC = 0.

    WHY NOT THE COUNT TEST. An independent adversarial review broke the first version of this
    gate here, and the break is worth recording in full. Counting discordances -- b = orderings
    the baseline got right and the arm got wrong, c = the reverse -- and testing b vs c under
    Bin(b+c, 1/2) is exact McNemar, and it tests H0: dC = 0 ONLY when every discordance moves C
    by the same amount. This statistic has TWO step sizes: a flipped ordering moves C by 1/n, a
    tie transition by 0.5/n. So the count test tests the wrong null the moment ties are in play
    -- which is the case the tie rule exists for. The demonstration, on the shipped code:

        n = 6000, 900 pairs where the arm flips a wrong ordering right (+1.0 each) and 1500
        where it turns a right ordering into a tie (-0.5 each): dC = +0.0250 and the count test
        returned p = 2.2e-34 IN THE BASELINE'S FAVOUR -- and the gate read PASS off it.

    A p-value pointing the opposite way from the effect it is attached to is not a conservative
    bug; it certifies whichever direction the counts happen to favour. So the primary test is a
    sign-flip randomization test on the per-pair differences themselves: under H0 each discordant
    pair's difference is equally likely to have either sign, and the null distribution of their
    SUM (which is n * dC) is enumerated exactly. The statistic tested is the statistic reported.

    It is a strict generalisation, not a replacement: when every discordance is a full swing this
    returns EXACTLY the two-sided exact McNemar p (pinned by
    `test_the_signflip_test_reduces_exactly_to_mcnemar_when_every_swing_is_full`).
    """
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    s_obs = abs(sum(d))
    # Scores live on the 0 / 0.5 / 1 lattice, so the magnitudes are integer multiples of the
    # smallest one and the null distribution is an exact convolution. Normalising by the
    # smallest magnitude rather than by a hardcoded 2 keeps the branch choice SCALE-INVARIANT:
    # a mutation that halved every difference (a no-op for a scale-invariant test) otherwise
    # slipped the mixed-swing case into the approximation branch.
    unit = min(abs(x) for x in d)
    w = [abs(x) / unit for x in d]
    if len(d) <= 400 and all(abs(x - round(x)) < 1e-9 for x in w):
        wi = [round(x) for x in w]
        total = sum(wi)
        counts = [0] * (total + 1)
        counts[0] = 1
        for weight in wi:
            nxt = [0] * (total + 1)
            for j, c in enumerate(counts):
                if c:
                    nxt[j] += c
                    nxt[j + weight] += c
            counts = nxt
        # with integer weights u_i, a sign assignment whose positive weights sum to j has
        # signed total (2j - total) in units of `unit`; the observed one is s_obs / unit.
        thresh = s_obs / unit - 1e-9
        hits = sum(c for j, c in enumerate(counts) if abs(2 * j - total) >= thresh)
        return min(1.0, hits / (1 << len(wi)))
    # Large m: the exact enumeration is O(m * sum(w)) in big integers. Var(S) = sum(w_i^2/4)
    # for independent +-w_i/2, and the normal approximation is accurate to well past the
    # fourth decimal at this m.
    sd = math.sqrt(sum(x * x for x in d))
    if sd == 0:
        return 1.0
    return min(1.0, math.erfc(s_obs / sd / math.sqrt(2)))


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar / sign test: 2 * P(X <= min(b,c)) under Bin(b+c, 1/2).

    Retained as a SENSITIVITY restricted to full swings, where it is exactly right (and equals
    `exact_signflip_p` on the same data). It is NOT the headline: see that function for the
    input on which using it as the headline read PASS off a p pointing the other way.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    if n > 2000:
        # Exact stays correct at any n but the tail sum walks lo+1 big-integer terms, and
        # 2.0**n overflows a float past n=1023 (the first version of this function did exactly
        # that and raised OverflowError mid-comparison). At this n the normal approximation
        # with a continuity correction is accurate to well past the fourth decimal, and no
        # gate decision turns on the fifth.
        z = (abs(n / 2 - lo) - 0.5) / (0.5 * math.sqrt(n))
        return min(1.0, math.erfc(z / math.sqrt(2)))
    # Integer numerator over integer denominator: exact, and immune to the float overflow.
    tail = sum(math.comb(n, k) for k in range(lo + 1))
    return min(1.0, 2 * tail / (1 << n))


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam-style rational approximation, |err| < 1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile out of range: {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z_ALPHA_2 = _z(1 - ALPHA / 2)
Z_POWER = _z(POWER)


def mde_delta_c(n_eligible: int, n_discordant: int, mean_swing: float = 1.0) -> float:
    """Smallest |dC| the paired sign test can see at ALPHA/POWER, given the discordance seen.

    Derivation, so it can be checked rather than believed. Discordances land
    arm-better with probability q; the sign test rejects at |q - 1/2| >= (z_a + z_b) /
    (2 sqrt(n_d)). A discordance moves C by `mean_swing`/n_eligible (1.0 for a flipped
    ordering, 0.5 for one that became or stopped being a head-side tie), so

        MDE = 2 * mean_swing * (z_a + z_b) / (2 sqrt(n_d)) * (n_d / n_eligible)
            = mean_swing * (z_a + z_b) * sqrt(n_d) / n_eligible

    Note the direction that surprises people: at fixed n_eligible, FEWER discordances make a
    given dC EASIER to detect, because the same movement then rests on a more lopsided
    majority. `test_mde_is_the_effect_a_simulation_detects_80_percent_of_the_time` feeds a
    known dC through the actual test and checks the rejection rate, because a power formula
    nobody simulated is a comment, not a guarantee.
    """
    if n_eligible <= 0 or n_discordant <= 0:
        return float("nan")
    return mean_swing * (Z_ALPHA_2 + Z_POWER) * math.sqrt(n_discordant) / n_eligible


def required_n_eligible(target: float = ADVANCE_DELTA_C,
                        discordance_rate: float = ASSUMED_DISCORDANCE_RATE,
                        mean_swing: float = 1.0) -> int:
    """Eligible pairs needed for `target` to be detectable at ALPHA/POWER. Inverts the above:
    MDE = mean_swing * (z_a+z_b) * sqrt(rate/n) => n = rate * ((z_a+z_b)*mean_swing/target)^2.
    """
    if target <= 0 or not 0 < discordance_rate <= 1:
        raise ValueError("target must be > 0 and discordance_rate in (0, 1]")
    return math.ceil(discordance_rate * ((Z_ALPHA_2 + Z_POWER) * mean_swing / target) ** 2)


# ----------------------------------------------------------------------------------------
# loading, with the refusals the design rests on
# ----------------------------------------------------------------------------------------
def pair_key(p: Mapping[str, Any]) -> tuple:
    return (p["seed"], p["prefix"], p["seat"])


def head_gap_from_values(head_a: float, head_b: float) -> float:
    """The ONLY place a head gap is derived. Fixtures and arms go through the same function.

    A transform must be applied to head_a and head_b and the gap RECOMPUTED. Transforming
    head_gap directly is equivalent only for LINEAR maps, so a fixture that does it makes the
    monotone-recalibration and saturating demos vacuous -- tanh(50 * gap) preserves the sign
    of the gap and would read a triumphant 0.0000 while tanh(50 v) applied to the values
    collapses 197 pairs to ties.
    """
    return (head_a - head_b) / GAP_SCALE


def load_cell(path: Path, name: str, rollouts: int = PINNED_ROLLOUTS) -> dict[tuple, dict]:
    """Read a pairs file into {pair_key: row}, refusing anything the gate cannot score."""
    doc = json.loads(Path(path).read_text())
    pairs = doc.get("pairs")
    if not pairs:
        raise SystemExit(f"REFUSING: {name} ({path.name}) has no `pairs` array.")
    rows: dict[tuple, dict] = {}
    for p in pairs:
        for field in ("head_a", "head_b", "head_gap", "true_gap", "seed", "prefix", "seat"):
            if p.get(field) is None:
                raise SystemExit(
                    f"REFUSING: {name} ({path.name}) has a pair missing {field!r}. This gate "
                    f"recomputes gaps from head_a/head_b, so a file carrying only head_gap "
                    f"cannot be scored under a non-linear transform and is refused outright.")
        for field in ("head_a", "head_b", "head_gap", "true_gap"):
            if not math.isfinite(float(p[field])):
                raise SystemExit(f"REFUSING: {name} ({path.name}) has a non-finite {field}.")
        # noise_var is optional, but if present it must be usable: a NaN here propagated silently
        # into `label_gap_se_banked_mean` and `se_over_tau_banked` in a committed artifact, which
        # is how a caveat about label noise turns into a NaN nobody reads.
        nv = p.get("noise_var")
        if nv is not None and not (math.isfinite(float(nv)) and float(nv) >= 0.0):
            raise SystemExit(
                f"REFUSING: {name} ({path.name}) pair {pair_key(p)} has noise_var={nv!r}. The "
                f"label-SE caveat is computed from this column; a NaN or negative there is "
                f"reported as a NaN caveat, which is worse than no caveat.")
        # THE TRANSFORM SURFACE MUST BE THE VALUES, NOT THE GAP. Every fixture and every
        # rescore in this programme derives head_gap as (head_a - head_b)/2; if a file
        # disagrees, then head_a/head_b are not the quantities its head_gap came from and
        # applying a non-linear transform to them would score a head nobody ran.
        derived = head_gap_from_values(float(p["head_a"]), float(p["head_b"]))
        if abs(derived - float(p["head_gap"])) > 1e-9:
            raise SystemExit(
                f"REFUSING: {name} ({path.name}) pair {pair_key(p)} has head_gap "
                f"{p['head_gap']!r} but (head_a - head_b)/{GAP_SCALE:g} = {derived!r}. The "
                f"gate's fixtures transform the VALUES and recompute the gap; a file whose "
                f"gap is not that function of its values cannot be transformed consistently.")
        for field in ("rollouts_a", "rollouts_b"):
            got = p.get(field)
            if got is not None and int(got) != rollouts:
                raise SystemExit(
                    f"REFUSING: {name} ({path.name}) pair {pair_key(p)} has {field}={got}, "
                    f"but R is pinned at {rollouts} in this instrument's spec. A bank mixing "
                    f"label budgets mixes label noise levels across the eligible set.")
        key = pair_key(p)
        if key in rows:
            raise SystemExit(
                f"REFUSING: {name} ({path.name}) has duplicate pair identity {key}. Pair "
                f"identity is what makes the comparison paired.")
        rows[key] = p
    return rows


def align(cells: Mapping[str, Mapping[tuple, dict]], ref: str) -> list[tuple]:
    """Refuse anything but an identical pair set, then pin the shared ground-truth column.

    `sibling_gap_compare.py` intersects and prints a NOTE. For an ADVANCEMENT statistic that
    is too soft: "dC on the identical eligible set" is the contract, and a cell that dropped
    pairs is a cell whose C is computed somewhere else.
    """
    ref_keys = set(cells[ref])
    for name, rows in cells.items():
        if set(rows) != ref_keys:
            missing = sorted(ref_keys - set(rows))[:3]
            extra = sorted(set(rows) - ref_keys)[:3]
            raise SystemExit(
                f"REFUSING: cell {name!r} is not scored on the same pairs as {ref!r} "
                f"({len(rows)} vs {len(ref_keys)}). missing e.g. {missing}, extra e.g. "
                f"{extra}. dC is only defined on an identical eligible set.")
    keys = sorted(ref_keys)
    ref_truth = [float(cells[ref][k]["true_gap"]) for k in keys]
    for name, rows in cells.items():
        if [float(rows[k]["true_gap"]) for k in keys] != ref_truth:
            raise SystemExit(
                f"REFUSING: {name}'s true_gap column differs from {ref}'s. The cells were "
                f"not scored against one ground truth and no dC below would mean anything.")
    return keys


# ----------------------------------------------------------------------------------------
# the statistic
# ----------------------------------------------------------------------------------------
def eligible_keys(rows: Mapping[tuple, dict], keys: Sequence[tuple], tau: float) -> list[tuple]:
    """|true_gap| >= tau AND true_gap != 0. The second clause is not redundant at tau=0."""
    if tau < 0:
        raise ValueError("tau must be >= 0")
    return [k for k in keys
            if float(rows[k]["true_gap"]) != 0.0 and abs(float(rows[k]["true_gap"])) >= tau]


def pair_score(head_gap: float, true_gap: float) -> float:
    """1 correct, 0.5 head-side tie, 0 wrong. Ties are never dropped -- see the module head."""
    if true_gap == 0.0:
        raise ValueError("true_gap == 0 pairs are excluded, never scored")
    if head_gap == 0.0:
        return 0.5
    return 1.0 if (head_gap > 0) == (true_gap > 0) else 0.0


def score_cell(rows: Mapping[tuple, dict], keys: Sequence[tuple],
               head_gap: Callable[[Mapping[str, Any]], float] | None = None) -> list[float]:
    hg = head_gap or (lambda p: float(p["head_gap"]))
    return [pair_score(hg(rows[k]), float(rows[k]["true_gap"])) for k in keys]


def concordance(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores)


def clustered_bootstrap_ci(base: Sequence[float], arm: Sequence[float],
                           clusters: Sequence[Any], reps: int = 4000,
                           seeds: Sequence[int] = (20260816, 3, 42)) -> tuple[float, float]:
    """Game-clustered bootstrap of dC. Resamples GAMES, not pairs: ~6 pairs share a game
    trajectory, so pair-level resampling (and the exact test) both understate the variance."""
    by_cluster: dict[Any, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    blocks = list(by_cluster.values())
    draws: list[float] = []
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(reps):
            tot = 0.0
            cnt = 0
            for _ in range(len(blocks)):
                for i in blocks[rng.randrange(len(blocks))]:
                    tot += arm[i] - base[i]
                    cnt += 1
            draws.append(tot / cnt if cnt else 0.0)
    draws.sort()
    if not draws:
        return (float("nan"), float("nan"))
    return (percentile(draws, 0.025), percentile(draws, 0.975))


def percentile(sorted_draws: Sequence[float], q: float) -> float:
    """The q-th order statistic, indexed on len-1.

    Extracted from the bootstrap so it can be fed a known answer: `int(q * len)` picks the
    (q*len + 1)-th order statistic -- the 2.508th percentile of 12,000 draws rather than the
    2.5th -- and inline in a resampler that defect is unreachable by any test, which is how a
    mutation of it survived a sweep.
    """
    if not sorted_draws:
        return float("nan")
    last = len(sorted_draws) - 1
    return sorted_draws[min(last, max(0, int(round(q * last))))]


def compare(ref_rows: Mapping[tuple, dict], arm_rows: Mapping[tuple, dict],
            keys: Sequence[tuple], tau: float, *, target: float = ADVANCE_DELTA_C,
            arm_head_gap: Callable[[Mapping[str, Any]], float] | None = None,
            bootstrap_reps: int = 2000) -> dict:
    """The whole comparison at one tau: absolute levels (descriptive), dC, exact paired p,
    achieved power, clustered sensitivity, and a verdict that may be a refusal to read."""
    sel = eligible_keys(ref_rows, keys, tau)
    n = len(sel)
    if n == 0:
        raise SystemExit(f"REFUSING: no pair in the bank is eligible at tau={tau}.")
    base = score_cell(ref_rows, sel)
    arm = score_cell(arm_rows, sel, head_gap=arm_head_gap)
    hg_arm = arm_head_gap or (lambda p: float(p["head_gap"]))

    c_base, c_arm = concordance(base), concordance(arm)
    delta = c_arm - c_base
    diffs = [(k, a - b) for k, a, b in zip(sel, arm, base) if a != b]
    up = [d for _, d in diffs if d > 0]
    down = [d for _, d in diffs if d < 0]
    b_cnt, c_cnt = len(down), len(up)
    p_exact = exact_signflip_p([d for _, d in diffs])
    p_counts = exact_mcnemar_p(b_cnt, c_cnt)
    full = [d for _, d in diffs if abs(d) == 1.0]
    half = [d for _, d in diffs if abs(d) == 0.5]
    p_full_only = exact_mcnemar_p(sum(1 for d in full if d < 0), sum(1 for d in full if d > 0))
    mean_swing = st.mean(abs(d) for _, d in diffs) if diffs else 1.0
    mde = mde_delta_c(n, len(diffs), mean_swing)
    powered = bool(diffs) and math.isfinite(mde) and mde <= target

    ties_base = sum(1 for k in sel if float(ref_rows[k]["head_gap"]) == 0.0)
    ties_arm = sum(1 for k in sel if hg_arm(arm_rows[k]) == 0.0)
    new_ties_keys = {k for k in sel
                     if hg_arm(arm_rows[k]) == 0.0 and float(ref_rows[k]["head_gap"]) != 0.0}
    new_ties = len(new_ties_keys)

    # THE ABSTENTION GUARD. Half credit for a head-side tie is what stops a saturating
    # transform from laundering broken orderings (see the module head), but it also opens the
    # mirror-image door: an arm that goes to an exact tie precisely where it used to be WRONG
    # collects +0.5 per abstention with no reordering whatsoever. Measured on the banked pairs
    # by the review that found it: dC = +0.1357, p = 5.8e-11, 35 manufactured ties, zero
    # orderings improved. It needs label access to aim, so it is not something training
    # stumbles into -- but "the attacker would need the labels" is not a property of the
    # statistic, and this gate is not allowed to rest on one.
    # So advancement must also survive scoring EVERY NEWLY MANUFACTURED TIE AS WRONG. The price
    # is exact and worth stating rather than glossing: delta - delta_adv = 0.5 * (new ties) / n,
    # because each new tie loses its half credit. An arm that creates no new ties pays nothing
    # (delta_adv == delta, and the guard is inert -- true of all nine banked arms at every tau).
    # An arm that ties on genuinely ambiguous siblings DOES pay, at that rate: the guard is a
    # deliberate conservative bias against buying advancement with abstentions, not a claim that
    # every tie is an abstention.
    arm_adv = [0.0 if k in new_ties_keys else s for k, s in zip(sel, arm)]
    delta_adv = concordance(arm_adv) - c_base
    diffs_adv = [a - b for a, b in zip(arm_adv, base) if a != b]
    p_adv = exact_signflip_p(diffs_adv)
    # The p must belong to the statistic the gate READS -- that is the whole lesson of the
    # count-test defect above, applied to this guard as well. So when the arm manufactured ties
    # and the guard therefore bites, the p reported alongside is the guarded statistic's p, not
    # the naive one. (A first version took min(delta) and max(p) independently; a mutation that
    # reverted the p half survived the suite, because the two were not tied together.)
    if delta_adv < delta:
        delta_gate, p_gate = delta_adv, p_adv
    else:
        delta_gate, p_gate = delta, p_exact

    if not diffs:
        verdict = "NO_ADVANCE_INVARIANT"
    elif delta_gate >= target and p_gate < ALPHA:
        verdict = "PASS" if powered else "PASS_PENDING_REMEASURE"
    elif delta <= -target and p_exact < ALPHA:
        verdict = "REGRESSED"
    elif powered:
        verdict = "NO_ADVANCE"
    else:
        verdict = "UNDERPOWERED_NO_VERDICT"

    lo, hi = clustered_bootstrap_ci(base, arm, [k[0] for k in sel], reps=bootstrap_reps)
    return {
        "tau": tau,
        "n_eligible": n,
        "n_excluded_true_gap_zero": sum(1 for k in keys if float(ref_rows[k]["true_gap"]) == 0.0),
        "n_excluded_below_tau": len(keys) - n
        - sum(1 for k in keys if float(ref_rows[k]["true_gap"]) == 0.0),
        "c_baseline": c_base,
        "c_arm": c_arm,
        "c_baseline_ci95_descriptive_only": list(wilson(sum(base), n)),
        "c_arm_ci95_descriptive_only": list(wilson(sum(arm), n)),
        "delta_c": delta,
        # the abstention guard's numbers, and the pair the verdict is actually read off
        "delta_c_new_ties_scored_wrong": delta_adv,
        "delta_c_gate": delta_gate,
        "exact_signflip_p_new_ties_scored_wrong": p_adv,
        "p_gate": p_gate,
        "head_ties_baseline": ties_base,
        "head_ties_arm": ties_arm,
        "head_ties_created_by_arm": new_ties,
        "discordant_total": len(diffs),
        "discordant_baseline_better": b_cnt,
        "discordant_arm_better": c_cnt,
        "discordant_full_swings": len(full),
        "discordant_half_swings_tie_transitions": len(half),
        "mean_swing": mean_swing,
        # PRIMARY: the sign-flip randomization test on the per-pair differences, i.e. on the
        # statistic actually reported. The two below are sensitivities and are NOT the headline:
        # the discordance-count test is exact only when every swing is full, and using it as the
        # headline read PASS off a p pointing the other way (see `exact_signflip_p`).
        "exact_signflip_p_two_sided": p_exact,
        "mcnemar_count_test_p_sensitivity_only": p_counts,
        "exact_mcnemar_p_full_swings_only": p_full_only,
        "mde_delta_c_at_80pct_power": mde,
        "powered_for_target": powered,
        # Whether the bank was big enough for the effect it actually saw, which is a different
        # question from whether it was big enough for the preregistered target.
        "powered_for_observed_effect": bool(diffs) and math.isfinite(mde) and abs(delta) >= mde,
        "target_delta_c": target,
        "required_n_eligible_for_target": (
            required_n_eligible(target, len(diffs) / n, mean_swing) if diffs else None),
        # What that means for the bank: eligibility at this tau keeps n/len(keys) of the pairs,
        # so this is the number of PAIRS that would have to be banked to earn a verdict on this
        # arm at this discordance rate. It is the sizing number, and it is the honest answer to
        # "why is the gate inert" -- not "the arm did nothing".
        "required_n_banked_pairs_for_target": (
            math.ceil(required_n_eligible(target, len(diffs) / n, mean_swing) * len(keys) / n)
            if diffs else None),
        "clustered_bootstrap_ci95_delta_c": [lo, hi],
        "verdict": verdict,
        "advanced": verdict == "PASS",
    }


# ----------------------------------------------------------------------------------------
# the demonstrated failing inputs. Every gate this programme lands ships with them.
# ----------------------------------------------------------------------------------------
def _blend_toward_truth(rho: float) -> Callable[[Mapping[str, Any], random.Random], tuple]:
    def f(p, _rng):
        ta, tb = 2 * float(p["true_a"]) - 1, 2 * float(p["true_b"]) - 1
        return ((1 - rho) * float(p["head_a"]) + rho * ta,
                (1 - rho) * float(p["head_b"]) + rho * tb)
    return f


def _corrupt(fraction: float) -> Callable[[Mapping[str, Any], random.Random], tuple]:
    def f(p, rng):
        a, b = float(p["head_a"]), float(p["head_b"])
        return (b, a) if rng.random() < fraction else (a, b)
    return f


def _pointwise(g: Callable[[float], float]) -> Callable[[Mapping[str, Any], random.Random], tuple]:
    return lambda p, _rng: (g(float(p["head_a"])), g(float(p["head_b"])))


def _abstain_where_wrong(p: Mapping[str, Any], _rng: random.Random) -> tuple:
    """LABEL-AIMED ABSTENTION: go to an exact head-side tie exactly where the head is wrong.

    The attack half credit invites, and the reason `delta_c_new_ties_scored_wrong` exists. It
    reorders nothing; it converts 0s into 0.5s. It needs the label to aim, so no training run
    finds it by accident -- but the gate must still refuse it, and the demo table shows both the
    naive number it would collect and the zero the shipped rule pays it.
    """
    hg, tg = float(p["head_gap"]), float(p["true_gap"])
    wrong = tg != 0.0 and hg != 0.0 and (hg > 0) != (tg > 0)
    return (0.0, 0.0) if wrong else (float(p["head_a"]), float(p["head_b"]))


# name -> (construction, what it is, what the gate MUST do with it)
DEMOS: dict[str, tuple[Callable, str, str]] = {
    "affine_rescale_k3.2764": (
        _pointwise(lambda v: 3.2764 * v),
        "the banked beta-gaming factor applied as a pure output rescale",
        "dC exactly 0 / NO_ADVANCE_INVARIANT"),
    "monotone_tanh2": (
        _pointwise(lambda v: math.tanh(2 * v)),
        "a strictly monotone recalibration",
        "dC exactly 0 / NO_ADVANCE_INVARIANT"),
    "monotone_sigmoid6": (
        _pointwise(lambda v: 1.0 / (1.0 + math.exp(-6 * v))),
        "a strictly monotone recalibration onto (0,1)",
        "dC exactly 0 / NO_ADVANCE_INVARIANT"),
    "additive_offset_plus0.2": (
        _pointwise(lambda v: v + 0.2),
        "the ECE-gaming vector: a constant offset, which cancels in every sibling gap",
        "dC exactly 0 / NO_ADVANCE_INVARIANT"),
    "saturating_tanh50": (
        _pointwise(lambda v: math.tanh(50 * v)),
        "MONOTONE BUT COLLAPSING: saturates the head and manufactures head-side ties",
        "dC negative, and NOT launderable by dropping the ties it created"),
    "saturating_tanh50_on_winprob": (
        _pointwise(lambda v: math.tanh(50 * ((v + 1) / 2))),
        "the same saturation applied after a win-prob reparameterisation (harsher: almost "
        "every pair collapses)",
        "dC strongly negative"),
    "abstention_gaming_label_aimed_ties": (
        _abstain_where_wrong,
        "LABEL-AIMED ABSTENTION: ties manufactured exactly where the head is wrong, collecting "
        "half credit for zero reordering (the mirror-image of the drop-ties hole)",
        "must NOT pass: dC is large and positive under naive scoring, and the gate reads it off "
        "delta_c_new_ties_scored_wrong, where it is <= 0"),
    "ordering_corrupted_15pct": (
        _corrupt(0.15),
        "ordering deliberately corrupted: 15% of pairs have their two head values swapped",
        "dC clearly negative"),
    "ordering_corrupted_25pct": (
        _corrupt(0.25),
        "ordering deliberately corrupted at 25%",
        "dC clearly negative"),
    "positive_control_blend_truth_0.10": (
        _blend_toward_truth(0.10),
        "POSITIVE CONTROL: head values shrunk 10% toward the (return-scaled) labels, a real "
        "ordering improvement with no scale, offset or calibration change available to it",
        "dC clearly positive -- if this does not move, the instrument is dead"),
    "positive_control_blend_truth_0.15": (
        _blend_toward_truth(0.15),
        "POSITIVE CONTROL at 15%",
        "dC clearly positive"),
}


def build_demo_cell(rows: Mapping[tuple, dict], keys: Sequence[tuple], name: str,
                    seed: int = 20260816) -> dict[tuple, dict]:
    """Materialise a demo cell. The transform is applied to head_a/head_b and head_gap is
    RECOMPUTED through `head_gap_from_values` -- the same function the loader validates real
    rescores against."""
    construction = DEMOS[name][0]
    rng = random.Random(seed)
    out: dict[tuple, dict] = {}
    for k in keys:
        p = rows[k]
        a, b = construction(p, rng)
        out[k] = dict(p, head_a=a, head_b=b, head_gap=head_gap_from_values(a, b))
    return out


def drop_ties_delta(ref_rows: Mapping[tuple, dict], arm_rows: Mapping[tuple, dict],
                    keys: Sequence[tuple], tau: float) -> dict:
    """The statistic this gate REFUSES to use, computed only to show what it would certify.

    Each cell scored on its own non-tied subset -- the natural naive implementation. A
    saturating head passes here by deleting the pairs it broke.
    """
    sel = eligible_keys(ref_rows, keys, tau)

    def c_drop(rows):
        kept = [k for k in sel if float(rows[k]["head_gap"]) != 0.0]
        if not kept:
            return float("nan"), 0
        hit = sum(1 for k in kept
                  if (float(rows[k]["head_gap"]) > 0) == (float(ref_rows[k]["true_gap"]) > 0))
        return hit / len(kept), len(kept)

    cb, nb = c_drop(ref_rows)
    ca, na = c_drop(arm_rows)
    return {"tau": tau, "c_baseline_drop_ties": cb, "n_baseline_kept": nb,
            "c_arm_drop_ties": ca, "n_arm_kept": na, "delta_c_drop_ties": ca - cb}


# ----------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------
CONDITIONING_CAVEAT = (
    "OI-1 measures ordering GIVEN THE OPPONENT'S SINGLE REALIZED REPLY: both the label "
    "(true_a/true_b) and the head's input are conditioned on the reply that actually "
    "happened at that decision, while search integrates over the reply DISTRIBUTION. C is "
    "therefore an UPPER BOUND on the ordering quality search can cash, and more rollouts "
    "cannot fix it -- only a reply-marginalised label would."
)
ABSOLUTE_LEVEL_CAVEAT = (
    "Absolute C is DESCRIPTIVE AND NEVER GATED: eligibility selects on a noisy measured gap, "
    "so C conditions on selection and is biased upward as a statement about the latent gap. "
    "The gate reads only the paired dC, where the selection is common to both cells."
)
LABEL_SE_CAVEAT = (
    "The programme's spec asks for label SE << tau. At R={R} the per-pair gap SE is {banked:.4f} "
    "(banked mean) / {worst:.4f} (worst case near p=0.5) against tau={tau}, a ratio of "
    "{ratio:.2f} -- the << condition {met} at the ratio this instrument commits to. This is "
    "reported, not waived: it is why absolute levels are never gated. What it means for the "
    "PAIRED dC, at the strength measured and no further: two cells differing by a monotone "
    "reparameterisation score exactly 0 under any label (an invariance), and a real difference "
    "is ATTENUATED toward the null (0.78 at tau=0.10), which is conservative for a gate. It is "
    "not a guarantee that noise cannot inflate one arm's point estimate; alpha bounds that."
)
ADVANCEMENT_NOTE = (
    "A PASS here is NECESSARY, NOT SUFFICIENT for arm advancement: the ceiling programme "
    "requires the ordering metric to move WITH beta/ECE as supporting evidence, never the "
    "pair alone and never ordering alone."
)


def label_se_note(rows: Mapping[tuple, dict], keys: Sequence[tuple], tau: float) -> dict:
    """Compute the spec's label-SE-vs-tau condition from the bank. DERIVED, not asserted: a
    review found this field hardcoded to False and witnessed only by a test asserting False,
    which certifies nothing about any bank. `MUCH_LESS_THAN_RATIO` is the reading of "<<" this
    instrument commits to, and a bank with small enough label noise makes the field read True
    (`test_the_label_se_condition_reads_true_on_a_bank_that_actually_meets_it`)."""
    var = [float(rows[k]["noise_var"]) for k in keys if rows[k].get("noise_var") is not None]
    banked = math.sqrt(st.mean(var)) if var else float("nan")
    ratio = banked / tau if tau else float("inf")
    met = bool(math.isfinite(ratio) and ratio <= MUCH_LESS_THAN_RATIO)
    return {
        "rollouts_R": PINNED_ROLLOUTS,
        "label_gap_se_banked_mean": banked,
        "label_gap_se_worst_case": LABEL_GAP_SE_WORST_CASE,
        "tau": tau,
        "se_over_tau_banked": ratio,
        "much_less_than_ratio_used": MUCH_LESS_THAN_RATIO,
        "condition_label_se_much_less_than_tau_met": met,
        "text": LABEL_SE_CAVEAT.format(
            R=PINNED_ROLLOUTS, banked=banked, worst=LABEL_GAP_SE_WORST_CASE, tau=tau,
            ratio=ratio, met=("IS MET" if met else "IS NOT MET")),
    }


def provenance(path: Path) -> dict:
    """Basename plus content hash -- NEVER the directory. Banked pairs files live under
    cluster-internal paths; this repo is public, and a scoreboard is not a place to leak a
    filesystem layout. The hash is what makes the artifact reproducible anyway."""
    data = Path(path).read_bytes()
    return {"file": Path(path).name, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def _fmt(row: dict) -> str:
    mde = row["mde_delta_c_at_80pct_power"]
    half = row["discordant_half_swings_tie_transitions"]
    adv = ("" if row["delta_c_gate"] == row["delta_c"]
           else f" (gate reads {row['delta_c_gate']:+.4f} with new ties scored wrong)")
    return (f"  tau={row['tau']:<5.2f} n={row['n_eligible']:<4d} C {row['c_baseline']:.4f} -> "
            f"{row['c_arm']:.4f}  dC {row['delta_c']:+.4f}{adv}  "
            f"disc {row['discordant_total']:<3d} (arm+{row['discordant_arm_better']}/"
            f"base+{row['discordant_baseline_better']}, {half} tie-swings)  "
            f"p={row['exact_signflip_p_two_sided']:.4f}  MDE "
            f"{'n/a' if not math.isfinite(mde) else f'{mde:.4f}'}  {row['verdict']}")


def sign_consistency(by_tau: Mapping[str, Mapping[str, Any]]) -> dict:
    """tau=0.05 and 0.15 are SIGN-CONSISTENCY CHECKS with no independent p claimed. Three
    thresholds tested as three hypotheses would be three bites at the same 465 pairs; what
    they are for is whether the direction survives moving the threshold."""
    signs = {t: (0 if r["delta_c"] == 0 else (1 if r["delta_c"] > 0 else -1))
             for t, r in by_tau.items()}
    nonzero = [s for s in signs.values() if s != 0]
    return {
        "signs_by_tau": signs,
        "primary_sign": signs[str(TAU_PRIMARY)],
        "all_nonzero_signs_agree": len(set(nonzero)) <= 1,
        "note": "tau checks carry NO independent p; only tau=%.2f is the test." % TAU_PRIMARY,
    }


def run_scoreboard(ref_name: str, ref_rows, cells, keys, taus, bootstrap_reps) -> dict:
    out: dict[str, Any] = {"cells": {}}
    for name, rows in cells.items():
        if name == ref_name:
            continue
        by_tau = {str(t): compare(ref_rows, rows, keys, t, bootstrap_reps=bootstrap_reps)
                  for t in taus}
        for t, row in by_tau.items():
            row["is_primary_tau"] = float(t) == TAU_PRIMARY
            if not row["is_primary_tau"]:
                row["p_is_a_sign_consistency_check_not_an_independent_test"] = True
        by_tau["sign_consistency"] = sign_consistency(
            {k: v for k, v in by_tau.items() if k != "sign_consistency"})
        out["cells"][name] = by_tau
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, metavar="NAME=FILE",
                    help="the baseline cell dC is taken against")
    ap.add_argument("--cell", action="append", default=[], metavar="NAME=FILE",
                    help="an arm to score (repeatable)")
    ap.add_argument("--demos", action="store_true",
                    help="run the demonstrated failing inputs and the positive control")
    ap.add_argument("--demo-cell", action="append", default=[], metavar="NAME=FILE",
                    help="a REAL arm to include in the demo table (e.g. the 513-param control "
                         "that gamed the old gate)")
    ap.add_argument("--demo-seed", type=int, default=20260816)
    ap.add_argument("--bootstrap-reps", type=int, default=2000,
                    help="per seed, three seeds; the clustered sensitivity interval")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    def parse(spec: str) -> tuple[str, Path]:
        name, _, p = spec.partition("=")
        if not name or not p:
            raise SystemExit(f"REFUSING: --cell/--ref want NAME=FILE, got {spec!r}")
        return name, Path(p)

    ref_name, ref_path = parse(args.ref)
    cells = {ref_name: load_cell(ref_path, ref_name)}
    prov = {ref_name: provenance(ref_path)}
    for spec in args.cell + args.demo_cell:
        name, path = parse(spec)
        if name in cells:
            raise SystemExit(f"REFUSING: duplicate cell name {name!r}")
        cells[name] = load_cell(path, name)
        prov[name] = provenance(path)
    keys = align(cells, ref_name)
    ref_rows = cells[ref_name]
    taus = [TAU_PRIMARY, *TAU_CHECKS]

    print("OI-1 ORDERING INSTRUMENT (pairwise tie-aware AUC of head gap vs the sign label)")
    print(f"  pair sampling: {PAIR_SAMPLING_DISTRIBUTION}")
    print(f"  pinned: R={PINNED_ROLLOUTS}  tau_primary={TAU_PRIMARY} "
          f"checks={list(TAU_CHECKS)} (no independent p)  pass threshold dC>="
          f"{ADVANCE_DELTA_C} at exact two-sided p<{ALPHA}, {POWER:.0%} power")
    print(f"  aligned on {len(keys)} pairs across {len(cells)} cells; baseline = {ref_name}")
    note = label_se_note(ref_rows, keys, TAU_PRIMARY)
    for text in (CONDITIONING_CAVEAT, ABSOLUTE_LEVEL_CAVEAT, note["text"], ADVANCEMENT_NOTE):
        print("  NOTE: " + text.replace("\n", " "))

    result: dict[str, Any] = {
        "schema": "pokezero.phase3.ordering-instrument.v1",
        "instrument": "OI-1",
        "baseline_cell": ref_name,
        "n_pairs_aligned": len(keys),
        "preregistered": {
            "statistic": "C(tau) = mean[ 1{sign(head_gap)==sign(true_gap)} + 0.5*1{head_gap==0} ]",
            "head_side_ties": "score 0.5, never dropped",
            "true_gap_zero_pairs": "excluded, never scored",
            "tau_primary": TAU_PRIMARY, "tau_checks": list(TAU_CHECKS),
            "rollouts_R": PINNED_ROLLOUTS,
            "pair_sampling_distribution": PAIR_SAMPLING_DISTRIBUTION,
            "advancement_statistic": "dC = C_arm - C_baseline on the identical eligible set",
            "test": "exact two-sided McNemar/sign test over discordant orderings (paired)",
            "pass_threshold_delta_c": ADVANCE_DELTA_C, "alpha": ALPHA, "power": POWER,
            "verdicts": list(VERDICTS),
        },
        "caveats": {
            "conditioned_on_realized_reply": CONDITIONING_CAVEAT,
            "absolute_levels_never_gated": ABSOLUTE_LEVEL_CAVEAT,
            "label_se_vs_tau": note,
            "advancement_requires_supporting_beta_ece": ADVANCEMENT_NOTE,
            "pairs_cluster_by_game": (
                "465 pairs from 80 games at ~6 sampled decisions each; the exact test assumes "
                "independence, so a game-clustered bootstrap interval accompanies every dC and "
                "is the one to read when the two disagree."),
            "half_credit_assumes_a_random_tie_break": (
                "The 0.5 credit for a head-side tie assumes SEARCH BREAKS AN EXACT TIE UNIFORMLY "
                "AT RANDOM. Nothing in this instrument verifies that: if the crate's tie-break is "
                "deterministic (by action index, say), then a tie is worth whatever that "
                "deterministic choice is worth on the pair, and 0.5 flatters an abstaining head. "
                "OPEN ITEM -- verify the crate's sibling tie-break. Mitigations already in place: "
                "the abstention guard re-reads advancement with every newly manufactured tie "
                "scored WRONG (the pessimistic end of that range), and on this bank the question "
                "is moot because zero baseline head ties survive eligibility at tau >= 0.05 "
                "(34 of the bank's 36 ties sit on true_gap == 0 pairs, which are never scored)."),
        },
        "provenance": prov,
    }

    arm_names = [n for n, _ in map(parse, args.cell)]
    if arm_names:
        board = run_scoreboard(ref_name, ref_rows, {n: cells[n] for n in arm_names},
                              keys, taus, args.bootstrap_reps)
        result.update(board)
        print("\n=== ARM SCOREBOARD (dC vs baseline; paired exact test) ===")
        for name in arm_names:
            print(f"{name}:")
            for t in taus:
                print(_fmt(result["cells"][name][str(t)]))

    if args.demos:
        demo_out: dict[str, Any] = {}
        print("\n=== DEMONSTRATED FAILING INPUTS + POSITIVE CONTROL ===")
        demo_specs = [(n, None) for n in DEMOS] + [(n, cells[n]) for n, _ in
                                                   map(parse, args.demo_cell)]
        for name, rows in demo_specs:
            arm_rows = rows if rows is not None else build_demo_cell(
                ref_rows, keys, name, seed=args.demo_seed)
            entry: dict[str, Any] = {
                "what": DEMOS[name][1] if rows is None else "a REAL rescored arm",
                "must": DEMOS[name][2] if rows is None else
                        "must not PASS: it gamed the beta/ECE pair with no mechanism change",
                "by_tau": {str(t): compare(ref_rows, arm_rows, keys, t,
                                           bootstrap_reps=args.bootstrap_reps) for t in taus},
                "drop_ties_counterfactual": {
                    str(t): drop_ties_delta(ref_rows, arm_rows, keys, t) for t in taus},
            }
            demo_out[name] = entry
            print(f"{name}: {entry['what']}")
            print(f"  MUST: {entry['must']}")
            for t in taus:
                print(_fmt(entry["by_tau"][str(t)]))
            dt = entry["drop_ties_counterfactual"][str(TAU_PRIMARY)]
            print(f"  drop-ties counterfactual (the rule this gate refuses) at "
                  f"tau={TAU_PRIMARY}: dC {dt['delta_c_drop_ties']:+.4f} on "
                  f"{dt['n_arm_kept']}/{dt['n_baseline_kept']} kept pairs")
        result["demos"] = demo_out
        passed = [n for n, e in demo_out.items()
                  if e["by_tau"][str(TAU_PRIMARY)]["verdict"].startswith("PASS")]
        gaming = [n for n in demo_out if not n.startswith("positive_control")]
        result["demo_summary"] = {
            "gaming_or_corruption_inputs_that_PASSED": [n for n in passed if n in gaming],
            "positive_controls_delta_c_at_tau_primary": {
                n: demo_out[n]["by_tau"][str(TAU_PRIMARY)]["delta_c"]
                for n in demo_out if n.startswith("positive_control")},
        }
        print(f"\ngaming/corruption inputs that PASSED: "
              f"{result['demo_summary']['gaming_or_corruption_inputs_that_PASSED'] or 'none'}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, sort_keys=True))
        print(f"\nwrote {Path(args.json).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
