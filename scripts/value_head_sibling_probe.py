#!/usr/bin/env python3
"""Can the value head rank the two moves search is choosing between?

THE QUESTION, and why the numbers we already have cannot answer it.

The value-gap investigation reported a top-1/top-2 root Q gap of 0.0192 against a
calibration error of 0.0516 and concluded the search cannot resolve its own arms. Both of
those are properties of SEARCH OUTPUT: `top_arms[].q` is the crate's `MoveStats::mean()`
-- `total_value / visits`, the backed-up subtree mean over every simulation through an arm
(`rust/pokezero-search/src/lib.rs:205`) -- and the ECE was computed from that same
backed-up Q against the realized outcome
(`deployment/mcts/analyze_value_gap.py:618-624`). The raw value head appears in no banked
shard at all.

So the head's own ability to rank siblings has never been measured. This measures it.

WHY NOT ECE. Global calibration cannot answer the question even in principle. Two heads
can share an ECE of 0.05: one off by a constant +0.05 on every position, which cancels
exactly in a comparison and ranks siblings perfectly; one unbiased on average but
scattering +/-0.05 independently per position, which destroys ranking whenever the true gap
is ~0.02. Only the second breaks search, and recalibration -- the obvious remedy -- fixes
only the first. Getting this distinction wrong costs a training programme.

WHAT IS MEASURED. For each sampled decision, with the opponent's reply held fixed:

    v_head(A), v_head(B)   the head's value at each arm's successor state
    w_true(A), w_true(B)   empirical win rate from N rollouts to terminal from each

and then the only quantity that matters:

    does sign(v_head(A) - v_head(B)) agree with sign(w_true(A) - w_true(B)) ?

reported BY TRUE-GAP BUCKET, because a head that ranks correctly when the truth is wide and
coin-flips when it is narrow has a different problem from one that is wrong everywhere --
and search lives in the narrow bucket.

THE MEASUREMENT TRAP, one level down. Ground truth is itself a noisy estimate. At N
rollouts the standard error of the GAP between two arms near 0.5 is 0.5*sqrt(2/N) -- 0.0707
at N=100, still 3.7x the 0.019 gap it must resolve. (Per-ARM it is 0.5/sqrt(N) = 0.05; this
file previously quoted the per-arm figure against a gap, understating the noise by 41%.) Paired rollouts (common random numbers across the two
arms) cancel much of the shared variance, which is why `--paired-seeds` is on by default.
Even so, the honest output is accuracy per bucket with the resolvable floor stated, not a
single number pretending to resolve 0.019.

SEARCH STACK. This uses the Python replay/rollout machinery (`replay_branching.py`,
`rollout.py`), not the Rust crate MCTS the banked numbers came from. That is deliberate and
sound: the quantity is a property of the HEAD and of successor states produced by the env,
not of any search. The successors here come from `replay_trajectory_branch`, so no search
is involved on either side of the comparison.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rollout_seed(base: int, battle_id: str, round_index: int, arm: int, trial: int,
                 *, paired: bool) -> int:
    """Seed for one rollout.

    With `paired=True` the arm index is EXCLUDED from the hash, so arm A's trial i and
    arm B's trial i share a seed and therefore share their chance draws wherever the two
    lines coincide. That is common random numbers: it cancels the shared component of the
    variance, which is most of it, and is the difference between resolving a 0.02 gap at a
    few hundred rollouts and needing several thousand.
    """
    parts = [str(base), battle_id, str(round_index), str(trial)]
    if not paired:
        parts.append(str(arm))
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:12], 16)


def _json_hi(hi: float):
    """None for the open-ended bucket, so the payload stays valid JSON."""
    return None if hi == float("inf") else hi


# Measured error rates of the sigma_diff verdict, as a function of the coupling the run
# actually achieves and the number of pairs. 600 repeats per cell, R=64, boundary 0.025
# (the midpoint of the 0.015/0.035 verdict band -- see the HOW TO READ note below).
#
# Indexed by paired_variance_ratio, i.e. by the quantity the probe MEASURES, not by a
# simulated crn knob nobody can read off a run. An earlier revision hardcoded three bands
# with rates that were wrong by up to 6 points and a gate at "n >= 1200" -- a figure that had
# already been RETRACTED in the analysis it cited, which measured only n=150 and n=300. The
# fix is to carry the measurements rather than prose about them.
#
# Caveat that no amount of repeats removes: the coupling here is a monotone-coupling model,
# while the real run shares one Showdown tape across every trial of a pair. So read these as
# the shape of the dependence, and the measured ratio as the input to it.
MEASURED_ERROR_RATES = (
    # (paired_variance_ratio, n_pairs, false "head binds", false "head is fine")
    # 600 repeats per cell. The first published version used 200 and was NOT monotone --
    # (0.88, 300) read 0.190 against (1.00, 300)'s 0.180, i.e. the error rate appeared to
    # RISE as coupling improved. That was Monte Carlo noise in a tail estimate, confirmed by
    # an independent replication, but the table had been committed with a claim of
    # monotonicity that no test checked at the failing cell. At 600 repeats both axes are
    # monotone and the test below verifies every adjacent pair rather than three samples.
    (1.00, 150, 0.275, 0.052),
    (1.00, 300, 0.212, 0.012),
    (0.88, 150, 0.232, 0.033),
    (0.88, 300, 0.167, 0.007),
    (0.75, 150, 0.202, 0.015),
    (0.75, 300, 0.132, 0.002),
    (0.59, 150, 0.160, 0.000),
    (0.59, 300, 0.078, 0.000),
    (0.37, 150, 0.052, 0.000),
    (0.37, 300, 0.012, 0.000),
    (0.00, 150, 0.000, 0.000),
    (0.00, 300, 0.000, 0.000),
)


def measured_error_rates(pvr: float, n: int) -> tuple[float, float, float, int]:
    """Nearest measured cell for this run's coupling and sample size.

    Returns (false_binds, false_fine, table_pvr, table_n). Nearest rather than interpolated:
    the cells are 600-repeat tail estimates, so interpolating between them would imply more
    precision than they carry. The cell used is reported so the reader can see the
    substitution.

    Two deliberate properties, both PESSIMISTIC, so a substitution never flatters the run:

    - n above 300 clamps to the n=300 row, because nothing larger was measured. A run at
      n=1000 is therefore quoted 18% where its true rate is lower. Conservative, but a
      reader must not take the quoted figure as measured AT their n.
    - a pvr between two cells picks the nearer one, and where that is ambiguous the weights
      favour the worse neighbour: pvr=0.50 maps to the 0.59 row (16.5% at n=150), not the
      0.37 row (6.5%).
    """
    return min(
        ((fb, ff, tp, tn) for tp, tn, fb, ff in MEASURED_ERROR_RATES),
        key=lambda r: (abs(r[2] - pvr) * 4.0 + abs(r[3] - n) / 300.0),
    )


# The verdict band. 0.025 -- the midpoint -- is where the ERROR RATES were measured, and is
# deliberately not a verdict boundary: an estimate of 0.02 must not decide a training
# programme in either direction.
VERDICT_LOW, VERDICT_HIGH = 0.015, 0.035


def verdict_lines(sd: Mapping[str, Any],
                  rates: tuple[float, float] | None) -> list[str]:
    """The advisory the reader acts on. Extracted so it can be TESTED.

    It lived inline in main() and was wrong twice in a row there, both times in ways a
    two-line test would have caught:

    1. It branched on `sigma_diff`, which is 0.0 whenever the estimate CLIPS, so an
       at-floor run printed "a low reading is well supported" ten lines below "NOT
       RESOLVABLE ... no upper bound can be quoted from this run". The affirmative line is
       the one that gets acted on, so the run asserted exactly the measured zero this
       module promises never to assert.
    2. It partitioned on 0.025 alone, so every estimate in the 0.015-0.035 band was called
       "well supported" -- contradicting the HOW TO READ line printed immediately above.

    This PR has now had to move a guard up a level five times because the thing being
    guarded was one call away from the assertion. Returning lines instead of printing them
    is what makes that unnecessary here.
    """
    n = sd.get("n") or 0
    if n < 150:
        return [f"  *** UNDERPOWERED: {n} pairs. 150 is the smallest size any error rate "
                f"has been measured at. ***"]
    if sd.get("at_floor"):
        return ["  NO VERDICT: the estimate is at the clip, so there is no point value to "
                "compare against the thresholds. The error rates above do not apply -- they "
                "were measured on runs that produced an estimate. Raise the pair count or "
                "the rollouts per arm."]
    if rates is None:
        return ["  NO ERROR RATES: the coupling could not be measured, so this run's "
                "verdict cannot be qualified. Treat it as indicative only."]
    pt = sd["sigma_diff"]
    fb, ff = rates
    if VERDICT_LOW < pt < VERDICT_HIGH:
        return [f"  INDETERMINATE: {pt:.4f} falls between {VERDICT_LOW} and {VERDICT_HIGH}. "
                f"That is a real answer, not a rounding problem -- this run does not decide "
                f"whether the head is the binding constraint, in either direction."]
    if pt >= VERDICT_HIGH:
        if fb >= 0.10:
            return [f"  *** A HIGH READING IS WEAKLY SUPPORTED: at this run's coupling and "
                    f"n={n}, {fb:.0%} of genuinely-fine heads read above the boundary. "
                    f"Raise the pair count, or improve the coupling, before spending "
                    f"training compute on this. ***"]
        return [f"  A high reading is well supported here: only {fb:.0%} of genuinely-fine "
                f"heads reach this side at this coupling and n."]
    if ff >= 0.02:
        return [f"  NOTE: {ff:.0%} of genuinely-BINDING heads read below the boundary at "
                f"this coupling and n. A low reading is suggestive, not decisive."]
    return [f"  A low reading is well supported here: only {ff:.0%} of genuinely-binding "
            f"heads read this low at this coupling and n."]


def head_gap_win_prob(head_a: float, head_b: float) -> float:
    """Convert a +/-1 return-scale head DIFFERENCE into a win-probability difference.

    E[return] = P(win) - P(loss), so (E+1)/2 = P(win) + P(draw)/2 -- exactly this probe's
    outcome coding (win 1.0, draw/cap 0.5, loss 0.0) and exactly the crate's own map
    (`values01 = 0.5*(v+1.0)`, rust/pokezero-search/src/model.rs:373). The gap of that
    affine map is the raw gap halved, exact including draws.

    Extracted from main() so it can be tested. Reverting the conversion at the call site
    left all 39 tests green -- the estimator tests exercise scale sensitivity, not this
    conversion -- so the fix for the units blocker had no regression guard of its own.
    """
    return (head_a - head_b) / 2.0


def finalize_pair_gaps(rec: dict) -> dict:
    """Stamp head_gap / head_gap_return_scale / true_gap onto a pair record, in one place.

    Exists because a pure `head_gap_win_prob` helper was NOT enough of a guard. Reverting
    the CALL SITE to `rec["head_gap"] = rec["head_a"] - rec["head_b"]` left the whole suite
    green: the conversion test exercised the function, and nothing asserted that main()
    actually used it. That is the units blocker -- the one that made a perfect head read
    0.0225 against a 0.015 boundary -- sitting unguarded behind a test that looked like it
    covered it.

    `head_gap_return_scale` is also the marker the shard merger uses to reject pre-fix
    records, so this function is the single point where a post-fix pair is defined.
    """
    rec["head_gap_return_scale"] = rec["head_a"] - rec["head_b"]
    rec["head_gap"] = head_gap_win_prob(rec["head_a"], rec["head_b"])
    rec["true_gap"] = rec["true_a"] - rec["true_b"]
    return rec


def estimate_sigma_diff(pairs: Sequence[Mapping[str, Any]],
                        n_boot: int = 2000, seed: int = 20260815) -> dict:
    """Estimate the SIBLING-DIFFERENTIAL head error by subtracting the known rollout noise.

    Why this exists, and why sign-agreement is not enough. The agreement metric compares the
    head against an ordering estimated from R rollouts. At R=64 that estimate has SE
    sqrt(2*0.25/64) = 0.088, while the median gap it must resolve is 0.0078 -- 11x smaller.
    Simulated over 6,000 pairs, a PERFECT head (sigma_diff = 0) scores 0.563 agreement and a
    useless one (0.20) scores 0.492: seven points of range, and the ground truth, not the
    head, is what the number describes. Reporting agreement as the verdict would certify a
    training programme off a figure a flawless head also produces.

    The fix is not more rollouts -- resolving 0.0078 by Monte Carlo needs ~33,000 per arm.
    It is that the noise variance is MEASURABLE from the retained per-trial outcomes, so it can be removed
    analytically:

        head_gap     = true_gap + differential_head_error     var: s_true^2 + s_diff^2
        measured_gap = true_gap + rollout_noise               var: s_true^2 + s_noise^2
        =>  s_diff^2 = var(head_gap - measured_gap) - s_noise^2

    The true_gap term cancels in the difference, which is what makes this work without ever
    knowing the true gap. s_noise^2 is the plug-in per-pair SAMPLE variance of the
    per-trial DIFFERENCES -- not Bernoulli: outcomes live in {0, 0.5, 1} and their
    differences in {-1, -0.5, 0, 0.5, 1}. Taking it over differences absorbs whatever
    covariance the paired seeds create rather than assuming the arms are independent;
    they are not, by design.

    Clipped at zero: a negative estimate means the differential is BELOW the noise floor, so
    the reading is an upper bound and is reported as such, never as a measured zero.
    """
    usable = [p for p in pairs
              if p.get("head_gap") is not None and p.get("true_gap") is not None
              and (p.get("rollouts_a") or 0) > 1 and (p.get("rollouts_b") or 0) > 1
              and not p.get("terminal_a") and not p.get("terminal_b")]
    if len(usable) < 2:
        return {"sigma_diff": None, "n": len(usable),
                "why": "CANNOT RUN: fewer than 2 pairs with rollouts on both arms"}

    def paired_noise_var(p):
        """var(w_a - w_b) for arms that share a seed per trial. NOT var_a + var_b.

        The design uses common random numbers: `rollout_seed(..., paired=True)` gives arm A
        and arm B the same seed on trial i precisely so their outcomes covary, and the
        module docstring calls that "the difference between resolving a 0.02 gap at a few
        hundred rollouts and needing several thousand". Summing two independent arm
        variances therefore subtracts MORE noise than exists, by exactly the amount the
        design was built to create.

        The consequence was not subtle. Simulated through this function at n=400, R=64, a
        head at the binding 0.0516 read 0.0000 under perfect CRN -- "below the refutation
        threshold, thesis refuted, stop the programme" -- for a head that in fact binds.
        That is the one direction the operating-characteristic table claimed was safe at
        every sample size.

        var(w_a - w_b) = var(mean of the per-trial differences d_i = o_a,i - o_b,i), which
        absorbs the covariance whatever it happens to be, so this is right for paired AND
        unpaired runs rather than assuming which one is configured.
        """
        oa, ob = p.get("outcomes_a") or {}, p.get("outcomes_b") or {}
        shared = sorted(set(oa) & set(ob))
        if len(shared) >= 2:
            d = [oa[t] - ob[t] for t in shared]
            m = sum(d) / len(d)
            # Sample variance (n-1) here; point() below uses the population form (/n).
            # An earlier comment claimed both used n-1 and warned against exactly the mix
            # the code then had. Measured difference at n=400: 0.01517 vs 0.01520 --
            # negligible, so the comment is corrected rather than the convention changed.
            return (sum((x - m) ** 2 for x in d) / (len(d) - 1)) / len(d)
        # No shared trials: fall back to the independent sum, which OVER-subtracts if the
        # run was paired. Unreachable in practice -- `usable` requires >1 rollout on both
        # arms and `pairing_intact` equalises the trial sets -- but it is counted, because
        # an earlier comment claimed it was "flagged in the payload" and no flag existed.
        return (_arm_var(oa, p["true_a"], p["rollouts_a"])
                + _arm_var(ob, p["true_b"], p["rollouts_b"]))

    def _arm_var(outcomes, w, r):
        """Variance of one arm's mean.

        NOT the Bernoulli w(1-w)/(r-1) the first version used. A trial can score 0.5 -- a
        draw, or a decision-round cap -- so outcomes live in {0, 0.5, 1} and the Bernoulli
        formula OVERSTATES the variance of any arm that drew ties (a half is closer to the
        mean than a 0/1 mix with the same average). Overstating the subtracted noise
        understates sigma_diff, which is the direction that would wrongly clear the head.
        The retained per-trial outcomes make the sample variance available, so use it and
        assume nothing.
        """
        vals = list(outcomes.values()) if outcomes else []
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) / len(vals)
        return w * (1 - w) / (r - 1) if r > 1 else 0.0   # fallback only

    def noise_var(p):
        # A precomputed value wins, so a SHARDED run can be merged from JSON without the
        # per-trial outcome dumps: the outcomes are what the paired variance needs, and
        # they are deliberately not serialized.
        if p.get("noise_var") is not None:
            return p["noise_var"]
        return paired_noise_var(p)

    def point(sample):
        d = [q["head_gap"] - q["true_gap"] for q in sample]
        m = sum(d) / len(d)
        var_d = sum((x - m) ** 2 for x in d) / len(d)
        nv = sum(noise_var(q) for q in sample) / len(sample)
        return var_d - nv, var_d, nv

    # MEASURE THE COUPLING, do not assume it. The estimator's precision depends almost
    # entirely on how strongly the paired seeds actually correlate the two arms, and that
    # is a property of the run, not of the design intent. Simulated at R=64, n=150: with no
    # coupling the false-"the head binds" rate is 22% and false-"fine" is 8%; with full
    # coupling both are 0% and the spread is 8x tighter. So the operating characteristics
    # cannot be quoted without knowing which regime the run is in.
    #
    # The ratio paired/independent is exactly that: 1.0 means the pairing bought nothing,
    # and it falls toward 0 as coupling rises. Cheap, and it comes free from data already
    # retained.
    couplings = []
    for q in usable:
        q["noise_var"] = noise_var(q)
        # At MERGE time the per-trial outcomes are not serialized, so _arm_var falls back
        # to w(1-w)/(R-1). That is not a loss: for outcomes in {0,1} the sample variance of
        # the mean is ALGEBRAICALLY IDENTICAL to that expression (verified to 1e-15 at
        # R=8/64/256), and a single 0.5 among 64 trials moves it by 1.7%. Draws and caps
        # are rare -- the comparable 200-game head-to-head had 1 tie and 0 caps -- so the
        # merged coupling figure is exact to well within its own use. Written down because
        # the alternative was re-running 8 GPUs to recover a difference that is zero.
        #
        # DIRECTION, when it is not zero: with a 0.5 present the fallback OVERSTATES the
        # independent sum, so the ratio reads lower, i.e. coupling looks stronger and the
        # error rates look better than they are. That is the permissive direction, which is
        # why the magnitude is bounded above rather than merely mentioned.
        indep = (_arm_var(q.get("outcomes_a"), q["true_a"], q["rollouts_a"])
                 + _arm_var(q.get("outcomes_b"), q["true_b"], q["rollouts_b"]))
        if indep > 0:
            couplings.append(q["noise_var"] / indep)
    raw, var_d, nv = point(usable)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [usable[rng.randrange(len(usable))] for _ in range(len(usable))]
        boot.append(math.sqrt(max(point(sample)[0], 0.0)))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    return {
        "sigma_diff": math.sqrt(max(raw, 0.0)),
        "ci95": [lo, hi],
        "at_floor": raw <= 0.0,
        "n": len(usable),
        "var_of_difference": var_d,
        "subtracted_noise_var": nv,
        # If the noise term is most of the observed variance the estimate is a small
        # difference of two larger numbers, and the CI, not the point, is the result.
        # None when var_d == 0, and this used to be formatted with {:.0%} -- a TypeError
        # at the very end of a multi-hour run, AFTER the table printed and BEFORE the JSON
        # was written, so the artifact was lost. A ratio above 1.0 is not a decoration
        # either: it IS the at-floor condition, so it is reported as such.
        "noise_share_of_variance": (nv / var_d) if var_d > 0 else None,
        # median over pairs of var(w_a - w_b) / (var_a + var_b). 1.0 = the paired seeds
        # bought nothing; lower = stronger common random numbers = a sharper estimator.
        "paired_variance_ratio": (statistics.median(couplings) if couplings else None),
        "paired_variance_ratio_n": len(couplings),
    }


def spread_prefixes(usable: Sequence[int], k: int) -> list[int]:
    """Sample k prefixes spanning the WHOLE game, endpoints included.

    The previous rule was `usable[::max(1, len(usable)//k)][:k]`, whose stride collapses to
    1 whenever `k <= len(usable) < 2k` -- 11 usable rounds and k=6 then took rounds 0..5 and
    nothing after, so 45% of the game supplied every sample and the late game supplied none.
    Value-head error is not stationary across a game (an early position is nearly a coin
    flip; a late one is often decided), so an early-game sample answers a different question
    than the one asked and would bias the headline in an unknown direction.
    """
    u = list(usable)
    if not u or k <= 0:
        return []
    k = min(k, len(u))
    if k == 1:
        return [u[0]]
    return sorted({u[round(i * (len(u) - 1) / (k - 1))] for i in range(k)})


def score_pairs(pairs: Sequence[Mapping[str, Any]], buckets: Sequence[float]) -> dict:
    """Sign-agreement between the head's ordering and ground truth, per true-gap bucket."""
    # The open-ended top bucket. float("inf") serialises as a bare `Infinity` token, which
    # is not valid JSON -- jq accepts it, Node's JSON.parse rejects the whole file. Kept as
    # inf for the comparison and rendered as None on the way out (see below).
    edges = list(buckets) + [float("inf")]
    out: dict[str, Any] = {"buckets": [], "n_pairs": len(pairs)}
    for lo, hi in zip(edges, edges[1:]):
        sel = [p for p in pairs if lo <= abs(p["true_gap"]) < hi]
        # A pair whose ground-truth gap is zero has no correct ordering; excluded rather
        # than counted as a failure of the head.
        sel = [p for p in sel if p["true_gap"] != 0.0]
        if not sel:
            out["buckets"].append({"lo": lo, "hi": _json_hi(hi), "n": 0, "accuracy": None})
            continue
        agree = sum(1 for p in sel
                    if (p["head_gap"] > 0) == (p["true_gap"] > 0) and p["head_gap"] != 0)
        n = len(sel)
        lo_ci, hi_ci = wilson(agree, n)
        out["buckets"].append({
            "lo": lo, "hi": _json_hi(hi), "n": n, "agree": agree, "accuracy": agree / n,
            "ci95": [lo_ci, hi_ci],
            "beats_chance": lo_ci > 0.5,
            "mean_true_gap": statistics.mean(abs(p["true_gap"]) for p in sel),
            "mean_head_gap": statistics.mean(abs(p["head_gap"]) for p in sel),
        })
    allsel = [p for p in pairs if p["true_gap"] != 0.0]
    if allsel:
        agree = sum(1 for p in allsel
                    if (p["head_gap"] > 0) == (p["true_gap"] > 0) and p["head_gap"] != 0)
        out["overall"] = {"n": len(allsel), "agree": agree,
                          "accuracy": agree / len(allsel),
                          "ci95": list(wilson(agree, len(allsel)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--showdown-root", required=True, type=Path)
    ap.add_argument("--games", type=int, default=4,
                    help="source games; decisions are sampled from their trajectories")
    ap.add_argument("--seed-start", type=int, default=24000000)
    ap.add_argument("--decisions-per-game", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=64,
                    help="rollouts per arm. the GAP SE near 0.5 is 0.5*sqrt(2/N) -- 0.0884 at 64 -- "
                         "so read the per-bucket floor, not a single headline number")
    ap.add_argument("--paired-seeds", action="store_true", default=True,
                    help="common random numbers across the two arms (default on)")
    ap.add_argument("--no-paired-seeds", dest="paired_seeds", action="store_false")
    ap.add_argument("--max-decision-rounds", type=int, default=250,
                    help="rollout cap; 250 is effectively 'play to terminal'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--buckets", default="0.0,0.02,0.05,0.10,0.20")
    ap.add_argument("--allow-unstamped-belief", action="store_true",
                    help="run against a checkpoint with no belief_set_source_hash. The "
                         "env is pinned belief-ON regardless, so this asserts you know "
                         "the checkpoint was trained that way. Record it in the write-up.")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from pokezero.local_showdown import (
        LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (
        category_vocab_from_model_config, evaluate_transformer_action_priors,
        evaluate_transformer_observation_value, feature_masks_from_model_config,
        load_transformer_checkpoint, observation_spec_from_model_config,
    )
    from pokezero.replay_branching import replay_trajectory_branch
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.search import player_observation_history

    buckets = [float(x) for x in args.buckets.split(",")]
    print(f"loading checkpoint {args.checkpoint}", flush=True)
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)

    # THE ENV MUST BE BUILT UNDER THE CHECKPOINT'S OWN SCHEMA, not the repo default.
    # feature_masks_from_model_config's docstring calls itself "THE single derivation point
    # from stamped provenance to env behavior. Every harness that builds an env for a
    # loaded checkpoint must route through this" -- and an earlier revision of this probe
    # did not, so the env emitted observations under the current default spec (rotated to
    # v4 on main) while this checkpoint was trained on an older one. The model rejected
    # them outright:
    #     ValueError: categorical_ids shape does not match TransformerPolicyConfig.
    # A loud failure, fortunately. The same mismatch in a shape-compatible direction would
    # have silently scored the head on observations it was never trained on -- which is
    # precisely the error class this probe exists to avoid making about itself.
    model_config = result.model_config
    env_spec = observation_spec_from_model_config(model_config)
    env_masks = feature_masks_from_model_config(model_config)
    env_vocab = category_vocab_from_model_config(model_config, args.showdown_root)
    print(f"env bound to the checkpoint's schema: spec={getattr(env_spec, 'schema_version', env_spec)}",
          flush=True)

    _belief_reported: set[int] = set()   # print the verification once, not once per rollout

    def make_env():
        """Env bound to the checkpoint on ALL FOUR axes, through the mandated entry point.

        Three axes (spec, vocab, masks) were latched by hand in the previous revision. The
        fourth -- BELIEF SET SOURCE -- was left to `belief_set_source_env_enabled()`, which
        defaults to "0" = DISABLED, while this checkpoint was trained belief-ON. That fails
        silently and shape-compatibly: candidate-set columns go unpopulated and
        `tier2_residuals_active()` returns False even with the checkpoint's
        `tier2_residuals=True` mask latched, so transition tokens lose their Tier-2
        residuals. Nothing raises, both arms are affected equally, and the probe reports a
        confident "the head cannot rank siblings" about observations the head never saw --
        the exact error class this file exists to avoid making about itself.

        Hand-setting the fields also skipped `env_config_from_checkpoint_provenance`, the
        repo's fail-closed single entry point -- the very function the comment above quotes
        as "every harness must route through this". Routed through it now, so a conflict
        raises instead of being silently resolved.
        """
        base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
        cfg_env = env_config_from_checkpoint_provenance(
            base, env_masks, context="value_head_sibling_probe",
            required_specs=env_spec, required_vocabs=env_vocab)
        env = LocalShowdownEnv(cfg_env)
        # The datum exists, so assert on it rather than trusting the wiring.
        want = getattr(result, "belief_set_source_hash", None)
        got = getattr(env, "belief_set_source_hash", None)
        # `want is None` is the DANGEROUS case, not the safe one. Per neural_policy.py
        # 915-918 it means the checkpoint had belief source disabled, or mixed provenance,
        # or predates the stamp -- so pinning the env ON is the exact mirror of B1, silent
        # and shape-compatible in the other direction. An earlier revision skipped the
        # check when either side was None, i.e. it was inert in precisely the case it was
        # written for.
        if want is None and not args.allow_unstamped_belief:
            raise SystemExit(
                "CANNOT RUN: this checkpoint carries no belief_set_source_hash, which means "
                "belief source was DISABLED, provenance was mixed, or it predates the stamp "
                "(neural_policy.py:915-918). The env here is pinned belief-ON, so running "
                "anyway would score the head on observations it was not trained on -- the "
                "mirror image of the fault this pin exists to prevent. Pass "
                "--allow-unstamped-belief to override, and say so in the write-up.")
        if want is not None and got != want:
            raise SystemExit(
                f"CANNOT RUN: belief-set-source mismatch. checkpoint {want!r} vs env {got!r}. "
                "Scoring the head on observations trained under a different belief "
                "condition would produce a confident wrong verdict.")
        # Printed, not just checked: a reader of the log must be able to tell "verified"
        # from "skipped", and only one of those two is worth staking a conclusion on.
        if _belief_reported:
            return env
        _belief_reported.add(1)
        if want is None:
            print("belief set source: PINNED ON, but the checkpoint carries NO HASH and "
                  "--allow-unstamped-belief was passed. UNVERIFIED -- state this in any "
                  "write-up that quotes this run.")
        else:
            print(f"belief set source: PINNED ON, hash VERIFIED equal to the checkpoint's "
                  f"({want[:16]}...)")
        return env

    def head_value(observations) -> float:
        """The raw value head on an observation history. NOT a backed-up Q.

        This is the quantity that appears in no banked shard, and the whole point of the
        probe. Calibration transform included, so it is the number search would see.
        """
        return evaluate_transformer_observation_value(
            model=model, result=result, observations=observations, device=args.device)

    def make_policy():
        """Rollout policy. Kwargs taken from the bridge's own construction
        (`foulplay_bridge.py:3688`) rather than guessed: the real knobs are
        `deterministic` and `sampling_temperature`, not `sample` and `temperature`.

        `deterministic=False` on purpose. Ground truth is a win RATE over N rollouts, so
        the continuation has to vary between trials; a deterministic policy would replay
        one identical line N times and report a rate of exactly 0 or 1 with a spuriously
        tiny interval.
        """
        from pokezero.neural_policy import TransformerSoftmaxPolicy
        return TransformerSoftmaxPolicy(
            model=model, result=result, device=args.device,
            deterministic=False, sampling_temperature=1.0)

    # VALUE CALIBRATION. The Python head applies result.value_calibration_transform; the
    # crate does NOT -- it maps the raw tanh straight through 0.5*(v+1). A transform with
    # scale != 1 would therefore put head_gap on a different axis from the Q gaps that set
    # the 0.015/0.035 boundaries, a multiplicative error of the same class as the units
    # blocker (a bias cancels in a gap; a scale does not). Recorded and checked, not assumed.
    _vct = getattr(result, "value_calibration_transform", None)
    _vct_method = getattr(_vct, "method", None) if _vct is not None else None
    # `.scale` is MEANINGLESS for an isotonic transform -- apply() ignores it entirely
    # (neural_policy.py:862-867) and warps through fitted points instead. Reading .scale
    # there would record `value_calibration_scale: 1.0` for a transform that is not
    # identity, which is worse than recording nothing: it is an affirmative false
    # reassurance on the axis the verdict thresholds live on.
    _vct_scale = (getattr(_vct, "scale", 1.0)
                  if _vct is not None and _vct_method == "affine" else 1.0)
    print(f"value calibration transform: {'NONE (identity)' if _vct is None else _vct}")
    if _vct is not None and _vct_method != "affine":
        print(f"  WARNING: method={_vct_method!r} is not affine, so no single scale factor "
              f"describes it and the crate applies none of it. head_gap may sit on a "
              f"different axis from the Q gaps behind the 0.015/0.035 thresholds by an "
              f"amount this run cannot summarise. Re-derive the thresholds before quoting "
              f"a verdict.")
    elif _vct is not None and abs(_vct_scale - 1.0) > 1e-9:
        print(f"  WARNING: scale={_vct_scale} != 1. The crate applies no calibration, so "
              f"head_gap is on a {_vct_scale}x axis relative to the Q gaps behind the "
              f"verdict thresholds. Treat the sigma_diff boundaries as rescaled by that "
              f"factor, or re-derive them.")

    cfg = RolloutConfig(max_decision_rounds=args.max_decision_rounds)
    pairs: list[dict] = []
    terminal_pairs: list[dict] = []
    skipped = collections.Counter()

    for gi in range(args.games):
        seed = args.seed_start + gi
        env = make_env()
        env.reset(seed=seed)
        policies = {"p1": make_policy(), "p2": make_policy()}
        source = continue_rollout_from_current_state(
            env=env, policies=policies, config=cfg, seed=seed,
            battle_id=f"probe-{seed}", starting_decision_round_index=0)
        traj = source.trajectory
        rounds = source.decision_round_count
        print(f"game {gi} seed={seed}: {rounds} decision rounds, "
              f"terminal winner={source.terminal.winner}", flush=True)

        # Sample decisions spread across the game rather than clustered at the opening:
        # the banked row cap already biases toward early-game and repeating that bias here
        # would make the probe unrepresentative of the decisions search actually faces.
        seat = "p1"
        # Only rounds where THIS SEAT has a step, and where the OPPONENT also has one --
        # both are needed, since a successor is only defined by a joint action. An evenly
        # spaced prefix ignores that: a seat does not act every decision round (waits,
        # forced switches), so most sampled rounds raised LookupError.
        seat_rounds = {st.turn_index for st in traj.steps if st.player_id == seat}
        opp_rounds = {st.turn_index for st in traj.steps if st.player_id != seat}
        usable = sorted(seat_rounds & opp_rounds)
        # Drop the last: branching AT the final round has no successor to evaluate.
        usable = [r for r in usable if r < rounds - 1]
        if len(usable) < 2:
            skipped["no_usable_joint_rounds"] += 1
            continue
        prefixes = spread_prefixes(usable, args.decisions_per_game)
        print(f"  {len(usable)} rounds have BOTH seats acting; sampling {prefixes}", flush=True)

        for prefix in prefixes:
            # The two arms to compare. Taken from the POLICY's top-2 priors rather than from
            # a search, so the probe measures the head on the pair a searcher would be
            # deciding between without needing a search in the loop.
            try:
                # The SAME builder search uses (search.py:3846, called with
                # through_decision_round=prefix at search.py:2021). An earlier revision
                # filtered turn_index < prefix, deleting the observation AT the decision.
                # The head consumes a sliding window, so that is a non-contiguous input it
                # was never trained on -- and because both arms shared the hole, nothing
                # would have looked wrong. It would have produced a confident
                # "the head cannot rank siblings" from an input search never produces.
                obs_hist = player_observation_history(
                    traj, player_id=seat, through_decision_round=prefix)
                arm_a, arm_b, opp_action = _top_two_and_opponent(
                    traj, seat, prefix, model, result, args.device,
                    evaluate_transformer_action_priors, obs_hist)
            except Exception as exc:                      # noqa: BLE001
                skipped[f"arm_selection:{type(exc).__name__}: {str(exc)[:80]}"] += 1
                continue
            if arm_a is None or arm_b is None:
                skipped["fewer_than_two_legal_arms"] += 1
                continue

            rec: dict[str, Any] = {"seed": seed, "prefix": prefix, "seat": seat,
                                   "arm_a": arm_a, "arm_b": arm_b}
            ok = True
            for label, arm in (("a", arm_a), ("b", arm_b)):
                branch_actions = {seat: arm, ("p2" if seat == "p1" else "p1"): opp_action}
                # HEAD value at this arm's successor -- one branch, no rollout.
                try:
                    benv = make_env()
                    benv.reset(seed=seed)
                    # check_prefix_observations ON here: this branch runs ONCE per pair,
                    # and it is the only proof the replayed prefix reproduced the sampled
                    # decision. Without it a divergent prefix silently yields head and true
                    # values for a DIFFERENT position. Left off for the rollout branches
                    # below only because those run N times per pair, and this one has
                    # already pinned the prefix they share.
                    br = replay_trajectory_branch(
                        benv, traj, prefix_decision_round_count=prefix,
                        branch_actions=branch_actions, check_prefix_observations=True)
                    hist, branch_terminal = _post_branch_history(br, seat, obs_hist)
                    if hist is None:
                        if branch_terminal is not None:
                            # The branch ENDED the battle, so there is no successor state
                            # and the head cannot be asked about one. An earlier revision
                            # scored it on the PRE-BRANCH position: for a pair where both
                            # arms end the game that makes head_a == head_b exactly, so
                            # head_gap == 0, which score_pairs counts as a miss -- turning
                            # every decisive pair into a deterministic zero in the widest
                            # bucket, for a reason that has nothing to do with the head.
                            #
                            # Recorded with its exact ground truth and EXCLUDED from the
                            # ranking metric. The exclusion is a real limitation of the
                            # measurement, not a defect: the wide-gap bucket is
                            # under-sampled by construction, and the count is printed so a
                            # reader can see that rather than infer a head failure.
                            rec[f"true_{label}"] = (
                                1.0 if branch_terminal.winner == seat
                                else (0.5 if branch_terminal.winner is None else 0.0))
                            rec[f"terminal_{label}"] = True
                            continue
                        # DO WHAT SEARCH DOES. `search.py:3743-3745` falls back to
                        # `env.observe(player_id)` when the branch step returns no
                        # observation for the seat -- so a state the head is asked to
                        # evaluate in production was being dropped here as unmeasurable.
                        # Dropping it is not neutral: it is exactly the "seat does not act
                        # next" states (the opponent forced, this seat locked in) that get
                        # filtered, and those are systematically different positions.
                        try:
                            fallback_obs = benv.observe(seat)
                        except Exception as exc:          # noqa: BLE001
                            skipped[f"observe_fallback:{type(exc).__name__}"] += 1
                            ok = False
                            break
                        if fallback_obs is None:
                            # Unreachable today -- LocalShowdownEnv.observe always returns
                            # an observation -- but a drop must never be silent if that
                            # changes, and it must not reuse the old label, which now means
                            # something else.
                            skipped["observe_fallback_returned_none"] += 1
                            ok = False
                            break
                        rec[f"observe_fallback_{label}"] = True
                        hist = (*obs_hist, fallback_obs)
                    rec[f"head_{label}"] = head_value(hist)
                except Exception as exc:                  # noqa: BLE001
                    skipped[f"branch:{type(exc).__name__}: {str(exc)[:120]}"] += 1
                    ok = False
                    break
                # GROUND TRUTH: N rollouts to terminal from that same successor.
                wins = 0.0
                done = 0
                failed_trials: set[int] = set()
                trial_outcomes: list[tuple[int, float]] = []
                for trial in range(args.rollouts):
                    rseed = rollout_seed(seed, f"probe-{seed}", prefix,
                                         0 if label == "a" else 1, trial,
                                         paired=args.paired_seeds)
                    try:
                        renv = make_env()
                        renv.reset(seed=seed)
                        # NOT replay_trajectory_branch_rollout: it hardcodes
                        # seed=trajectory.seed (replay_branching.py:337), so every trial
                        # replays IDENTICALLY and N rollouts are one sample repeated N
                        # times. The smoke run showed it -- 8/8 trials returned the same
                        # winner and every true rate came out exactly 0.000, with a
                        # reported SE of 0.177 that described nothing. Driving the two
                        # primitives it wraps lets the per-trial seed actually reach the
                        # continuation's player RNGs, which is what makes the win rate a
                        # rate. The PREFIX still replays under the source seed, as it must.
                        br2 = replay_trajectory_branch(
                            renv, traj, prefix_decision_round_count=prefix,
                            branch_actions=branch_actions, check_prefix_observations=False)
                        # M2 -- WHAT THE PAIRED SEED ACTUALLY PAIRS. `rollout_seed` varies
                        # per (pair, trial) so the two ARMS share a tape within a trial and
                        # differ across trials. But the env was reset from the source
                        # trajectory's seed, so the SHOWDOWN PRNG tape (damage rolls, crits,
                        # speed ties, secondary effects) is identical across all trials of a
                        # pair; only policy sampling varies. Trials are therefore NOT
                        # independent draws over battle randomness, and the paired SE below
                        # is a variance over policy sampling alone. It understates the total
                        # uncertainty in the true gap. Stated because a reader would
                        # otherwise read "64 rollouts" as 64 independent games.
                        cont = continue_rollout_from_current_state(
                            env=renv,
                            policies={"p1": make_policy(), "p2": make_policy()},
                            config=cfg, seed=rseed,
                            battle_id=f"probe-roll-{seed}-{prefix}-{label}-{trial}",
                            starting_decision_round_index=prefix + 1,
                            available_observations=br2.step_result.observations,
                            reset_policies=True)
                        rr = cont
                    except Exception:                     # noqa: BLE001
                        # Counted, never silent. A dropped trial also BREAKS the paired-seed
                        # design: if arm A's trial 7 fails and arm B's does not, the two
                        # arms no longer share their common random numbers and the variance
                        # cancellation the whole design rests on is gone for that pair.
                        failed_trials.add(trial)
                        skipped["rollout_failed"] += 1
                        continue
                    term = rr.terminal
                    # A capped game has no winner. Counted as a half, and counted
                    # SEPARATELY, because silently treating it as a loss would bias
                    # exactly the long grindy lines that stall.
                    if term.winner is None and term.capped:
                        rec[f"capped_{label}"] = rec.get(f"capped_{label}", 0) + 1
                        outcome = 0.5
                    elif term.winner is None:
                        # A genuine DRAW, not a cap. Scored 0.5 here and 0.5 in the
                        # terminal-branch path; an earlier revision scored it 0.0 here and
                        # 0.5 there, so the same outcome had two values in one file and a
                        # tie-forcing arm was systematically undervalued.
                        rec[f"drawn_{label}"] = rec.get(f"drawn_{label}", 0) + 1
                        outcome = 0.5
                    else:
                        outcome = 1.0 if term.winner == seat else 0.0
                    wins += outcome
                    trial_outcomes.append((trial, outcome))
                    done += 1
                if done == 0:
                    skipped["no_rollouts_completed"] += 1
                    ok = False
                    break
                rec[f"true_{label}"] = wins / done
                rec[f"rollouts_{label}"] = done
                rec[f"failed_{label}"] = sorted(failed_trials)
                # Retained for the PAIRED standard error. The design is common random
                # numbers, so the quantity score_pairs consumes is the per-trial DIFFERENCE
                # d_i = w_a,i - w_b,i, whose SE is sd(d)/sqrt(N). Reporting the unpaired
                # per-arm 0.5/sqrt(N) describes a different estimator and understates the
                # design -- and these outcomes were being computed and thrown away.
                rec[f"outcomes_{label}"] = dict(trial_outcomes)
            if not ok:
                continue
            # PAIRING RECONCILIATION. Common random numbers only cancel variance if the two
            # arms ran the SAME trials. If either arm lost trials, the shared component is
            # gone for the trials the other kept, so the pair's true_gap is no longer a
            # paired estimate. Recorded rather than silently accepted; a pair whose arms
            # disagree on which trials survived is marked and excluded from the paired
            # claim, because using it would quietly reintroduce the variance the design
            # exists to remove.
            if rec.get("terminal_a") or rec.get("terminal_b"):
                # Exact ground truth, but no head estimate for the terminal arm, so the
                # pair cannot test the head's ORDERING. Kept in the record and counted.
                skipped["terminal_branch_no_head_estimate"] += 1
                terminal_pairs.append(rec)
                continue
            fa, fb = set(rec.get("failed_a", ())), set(rec.get("failed_b", ()))
            rec["pairing_intact"] = (fa == fb)
            if not rec["pairing_intact"]:
                skipped["pairing_broken_by_failed_trials"] += 1
                continue
            # ONE SCALE. The head is on the +/-1 RETURN scale -- ValueCalibrationTransform
            # clips to [-1, 1] (neural_policy.py:841-842) and is fitted against returns of
            # win +1 / draw 0 / loss -1 (dataset.py:2146-2156). The rollout ground truth is
            # a win RATE in [0, 1]. So head_gap was ~2x true_gap, the true_gap term did NOT
            # cancel in their difference, and the estimator returned
            # sqrt(sigma_true^2 + sigma_diff^2) -- a PERFECT head read 0.0225, over the
            # 0.015 refutation boundary on an artifact of units alone.
            #
            # Confirmed on the artifact, not just the source: the smoke run printed
            # "head -0.0524/-0.0399" beside "true 0.500/0.750" -- the head sits near 0 for
            # an even position where the rate sits near 0.5.
            #
            # Converted to WIN-PROBABILITY units, because that is what the thresholds in
            # the required-head-error analysis are in (a 0.0078 median Q gap, a 0.0516 ECE
            # against a 0/1 outcome). Sign-agreement is scale-invariant, which is why this
            # defect arrived with the sigma_diff readout and was invisible before it.
            finalize_pair_gaps(rec)
            pairs.append(rec)
            # Per-arm values printed in WIN-PROBABILITY units, the same units as the gap
            # beside them and as the rollout truth. Printing the raw +/-1 head values next
            # to a halved gap made the line fail its own arithmetic -- a reader subtracting
            # 0.1035 - 0.0753 got 0.0282 against a printed gap of 0.0141 and would
            # reasonably conclude one of the two was wrong.
            pa_ = (rec["head_a"] + 1.0) / 2.0
            pb_ = (rec["head_b"] + 1.0) / 2.0
            print(f"  prefix {prefix}: head p {pa_:.4f}/{pb_:.4f} "
                  f"(gap {rec['head_gap']:+.4f})  true {rec['true_a']:.3f}/"
                  f"{rec['true_b']:.3f} (gap {rec['true_gap']:+.3f})  "
                  f"[head raw +/-1: {rec['head_a']:+.4f}/{rec['head_b']:+.4f}]", flush=True)

    if not pairs:
        print(f"CANNOT RUN: no scorable pairs. skipped={dict(skipped)}")
        return 2

    # Resolution from what actually RAN. Printing 0.5/sqrt(--rollouts) would overstate the
    # precision of every pair that lost a trial, and the whole point of the probe is that
    # the ground truth's own noise is the thing most likely to fool it.
    realised = [min(p["rollouts_a"], p["rollouts_b"]) for p in pairs
                if p.get("rollouts_a") and p.get("rollouts_b")]
    n_eff = min(realised) if realised else 0
    median_realised = statistics.median(realised) if realised else 0
    if realised and min(realised) != max(realised):
        print(f"  realized rollouts/arm vary: min {min(realised)}, median "
              f"{median_realised}, max {max(realised)} over {len(realised)} pairs -- "
              f"the resolution below uses the median, so individual pairs are coarser.")
    # sqrt(2) matters here: 0.5/sqrt(n) is the SE of ONE ARM's rate, but the quantity being
    # resolved is the GAP between two arms, whose SE is 0.5*sqrt(2/n). Comparing a gap
    # against a per-arm SE understates the noise by 41% and made the pairing look worse than
    # it is. The module docstring repeated the same conflation.
    se = (0.5 * math.sqrt(2.0 / n_eff)) if n_eff else None
    # The PAIRED SE, which is the one that describes this design. Per pair, over the trials
    # both arms completed: d_i = w_a,i - w_b,i, SE = sd(d)/sqrt(N). Reported beside the
    # unpaired figure so the reader can see how much the common random numbers actually
    # bought, rather than taking "(paired seeds reduce this)" on faith.
    paired_ses = []
    for pr in pairs:
        oa, ob = pr.get("outcomes_a") or {}, pr.get("outcomes_b") or {}
        shared = sorted(set(oa) & set(ob))
        if len(shared) < 2:
            continue
        d = [oa[t] - ob[t] for t in shared]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        paired_ses.append(sd / math.sqrt(len(d)))
    paired_se = statistics.median(paired_ses) if paired_ses else None

    # M4: THE BUCKETS MUST NOT BE FINER THAN THE GROUND TRUTH CAN RESOLVE. With R
    # completed rollouts per arm the true gap is quantised to multiples of 1/R, so a bucket
    # narrower than 1/R either is empty or contains a single attainable value -- and since
    # a bucket [lo,hi) with lo=0 excludes true_gap==0, the modal small-gap outcome, the
    # narrowest bucket can report an accuracy computed on a handful of pairs that happened
    # to land on one quantum. That reads as "the head is 55% accurate on small gaps" when
    # the instrument cannot see a small gap at all. Coarsen, and say so.
    # TWO corrections over the first attempt, both of which changed the reported table.
    #
    # 1. The quantum is 0.5/R, not 1/R. A trial can score 0.5 (a cap, or a draw), so an
    #    arm's rate moves in half-win steps and the gap's attainable spacing is half what
    #    the earlier version claimed. It over-merged by 2x while stating the wrong quantum
    #    as a fact.
    # 2. MEDIAN, not min. `min(realised)` lets a single degenerate pair -- one that
    #    completed 2 trials -- set resolution 0.25, drop every bucket edge below it, and
    #    collapse the whole per-bucket table, the probe's headline, to one [0.0, inf) row
    #    for the entire run. One bad pair must not silently redefine the output for all the
    #    others.
    resolution = (0.5 / median_realised) if median_realised else None
    dropped_edges = []
    if resolution is not None:
        keep = [b for b in buckets if b == 0.0 or b >= resolution]
        dropped_edges = [b for b in buckets if b not in keep]
        if dropped_edges:
            print(f"\nBUCKET FLOOR: a median {median_realised} completed rollouts per "
                  f"arm, with half-win outcomes possible, quantises the true gap to "
                  f"multiples of {resolution:.4f}, so edges {dropped_edges} are "
                  f"below the resolution of the instrument and are MERGED, not reported. "
                  f"An accuracy on a bucket narrower than one quantum is an artefact.")
        buckets = keep
    scored = score_pairs(pairs, buckets)
    scored["gap_resolution"] = resolution
    scored["dropped_bucket_edges"] = dropped_edges
    # The zero bucket is the modal outcome and is reported on its own, never folded into a
    # "small gap" bucket where it would masquerade as measured discrimination.
    exact_zero = sum(1 for pr in pairs if pr.get("true_gap") == 0.0)
    scored["n_true_gap_exactly_zero"] = exact_zero
    if exact_zero:
        print(f"  {exact_zero} of {len(pairs)} pairs have true_gap EXACTLY 0.0 -- the two "
              f"siblings were indistinguishable at this rollout count, so there is no "
              f"ordering for the head to get right or wrong. They are excluded from every "
              f"bucket by construction ([lo,hi) with lo=0), which is stated here because "
              f"silently excluding the modal outcome would inflate the headline.")
    exact = len(terminal_pairs)
    print(f"\n=== sibling discrimination, {len(pairs)} pairs ===")
    if se is None:
        print("ground-truth resolution: CANNOT RUN -- no pair completed rollouts on both "
              "arms, so no rollout-based gap is resolvable")
    else:
        print(f"ground-truth resolution: worst-case {n_eff} rollouts/arm actually completed "
              f"-> SE ~{se:.4f} near 0.5"
              f"{' (paired seeds reduce this)' if args.paired_seeds else ''}")
    if paired_se is not None:
        zero_sd = sum(1 for x in paired_ses if x == 0.0)
        print(f"  PAIRED SE on the true gap: median {paired_se:.4f} across "
              f"{len(paired_ses)} pairs. CAVEAT, and it is not a small one: every trial of "
              f"a pair resets from the SAME source seed, so the Showdown PRNG tape is "
              f"shared and only policy sampling varies. These are correlated replays, not "
              f"independent games, so this SE covers policy-sampling variance ALONE and "
              f"understates the true uncertainty by an unmeasured amount.")
        if zero_sd:
            print(f"    {zero_sd} pairs have SE exactly 0.0 -- every trial returned the "
                  f"same result, which is degeneracy, not precision.")
    else:
        print("  PAIRED SE: unavailable (no pair had >=2 shared completed trials)")
    if exact:
        print(f"  {exact} pairs had an arm END the battle: exact ground truth, but NO head "
              f"estimate exists for a state that does not exist, so they are excluded from "
              f"the ranking metric. The widest bucket is under-sampled by that much.")
    print(f"{'true-gap bucket':>18s} {'n':>5s} {'accuracy':>9s} {'95% CI':>16s} {'>chance':>8s}")
    for b in scored["buckets"]:
        if not b["n"]:
            continue
        hi = "inf" if b["hi"] is None else f"{b['hi']:.2f}"
        print(f"{b['lo']:.2f}-{hi:>6s}{'':>5s} {b['n']:5d} {b['accuracy']:9.3f} "
              f"[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] {str(b['beats_chance']):>8s}")
    sd = estimate_sigma_diff(pairs)
    scored["sigma_diff"] = sd
    print("\n=== SIBLING-DIFFERENTIAL HEAD ERROR (the quantity that decides this) ===")
    if sd.get("sigma_diff") is None:
        print(f"  {sd.get('why')}")
    else:
        if sd["at_floor"]:
            # Printing "UPPER BOUND: 0.0000" was the measured zero the docstring promises
            # never to print. The actual bound is the top of the interval.
            # ci95[1] is itself 0.0 whenever >=97.5% of bootstrap resamples clip, so
            # printing it unguarded still asserts a PROVEN exact zero -- the thing the
            # docstring forbids, just one step further along. Measured at 3 in 100 runs
            # for a perfect head at n=400. Report unresolvable rather than zero.
            if sd["ci95"][1] <= 0.0:
                print(f"  sigma_diff is NOT RESOLVABLE at n={sd['n']} with these rollouts: "
                      f"the estimate and the whole bootstrap interval sit at the clip. "
                      f"That is an absence of resolution, NOT a demonstration of zero "
                      f"differential error -- no upper bound can be quoted from this run.")
            else:
                print(f"  sigma_diff is BELOW THE NOISE FLOOR of this run. The point "
                      f"estimate clips to 0, which is not a measurement -- the result is "
                      f"the UPPER BOUND: sigma_diff <= {sd['ci95'][1]:.4f} "
                      f"(n={sd['n']} pairs)")
        else:
            print(f"  sigma_diff estimate: {sd['sigma_diff']:.4f}  95% CI "
                  f"[{sd['ci95'][0]:.4f}, {sd['ci95'][1]:.4f}]  from n={sd['n']} pairs")
        share = sd.get("noise_share_of_variance")
        share_txt = ("undefined (the observed variance is exactly zero)" if share is None
                     else f"{share:.0%} of it" + (" -- ABOVE 100%, which IS the at-floor "
                                                  "condition" if share > 1 else ""))
        print(f"  var(head_gap - measured_gap) {sd['var_of_difference']:.6f} minus paired "
              f"rollout-noise var {sd['subtracted_noise_var']:.6f} ({share_txt} was noise)")
        pvr = sd.get("paired_variance_ratio")
        rates = None
        if pvr is not None:
            fb, ff, tp, tn = measured_error_rates(pvr, sd["n"])
            rates = (fb, ff)
            print(f"  MEASURED COUPLING: var(w_a-w_b) is {pvr:.2f}x the independent sum, "
                  f"over {sd['paired_variance_ratio_n']} pairs. 1.0 means the paired seeds "
                  f"bought nothing; 0.0 means the arms move together perfectly.")
            print(f"  ERROR RATES AT THIS COUPLING AND n, measured (600 repeats, nearest "
                  f"cell pvr={tp:.2f} n={tn}): false 'the head binds' {fb:.1%}, "
                  f"false 'the head is fine' {ff:.1%}. These are selected by the measured "
                  f"ratio, not assumed.")
        print("  DIRECTION OF THE REMAINING BIAS. Do not read a confident sign into this "
              "-- an earlier revision asserted 'both known biases inflate sigma_diff' and "
              "one of the two could not be reproduced (holding the head perfect and adding "
              "a shared per-pair tape shift moved the reading DOWN, not up). What IS "
              "established: the zero-clip lifts small values, so a reading at the floor is "
              "an upper bound; and trials of a pair share the post-branch PRNG tape, so "
              "this covers within-tape variation only and the tape-to-tape component is "
              "unmeasured in BOTH directions.")
        # Thresholds inlined, not cited by path: the analysis lives in a gitignored private
        # tree, so a public-repo reader cannot resolve a reference to it.
        print("  HOW TO READ: sigma_diff <= 0.015 -> the head is NOT the binding "
              "constraint. >= 0.035 -> it is, and no quantity of sims fixes it (more sims "
              "reduce variance; this is bias). Between the two -> INDETERMINATE, which is "
              "a real answer and not a rounding problem. The error rates quoted below use "
              "a single 0.025 boundary -- the midpoint of that band -- because a "
              "false-positive rate needs one line; they are therefore the rates for the "
              "most permissive reading, and the band edges are strictly better.")
        for line in verdict_lines(sd, rates):
            print(line)
    o = scored.get("overall")
    if o:
        print(f"{'OVERALL':>18s} {o['n']:5d} {o['accuracy']:9.3f} "
              f"[{o['ci95'][0]:.3f},{o['ci95'][1]:.3f}]")
    print("\nA bucket whose CI includes 0.500 has NOT shown the head can rank at that gap.")
    # This line used to say "the bucket that matters is the SMALLEST one", printed even
    # after that bucket had been merged away -- inviting a reader to read a merged
    # [0.00, 0.20) row as the narrow one. It also pointed at the wrong readout entirely.
    print("DO NOT read the verdict off this table. Sign-agreement is measured against an "
          "ordering estimated from R rollouts, whose SE at R=64 is 0.088 against a 0.0078 "
          "median gap, so at the gaps search actually lives at a PERFECT head scores ~0.563 "
          "(simulated, 6,000 pairs) and a useless one ~0.492. The table is a coarse "
          "sanity check on wide-gap pairs only. The verdict is sigma_diff above.")
    if skipped:
        print(f"skipped: {dict(skipped)}")
    if args.json:
        args.json.write_text(json.dumps(
            {"belief_set_source_hash": getattr(result, "belief_set_source_hash", None),
             "value_calibration_transform": (None if _vct is None else str(_vct)),
             # None, not 1.0, when the method ignores scale -- writing 1.0 there is an
             # affirmative false reassurance on the axis the thresholds live on.
             "value_calibration_scale": (_vct_scale if _vct_method == "affine" else None),
             "value_calibration_method": _vct_method,
             "config": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "ground_truth_se": se,
             "ground_truth_se_note": "gap SE = 0.5*sqrt(2/n), not the per-arm 0.5/sqrt(n)",
             "paired_se_median": paired_se,
             "paired_se_caveat": (
                 "Trials of a pair share the source reset seed, so the Showdown PRNG tape "
                 "is common and only policy sampling varies. Correlated replays, not "
                 "independent games: this SE is policy-sampling variance alone and "
                 "understates total uncertainty by an unmeasured amount."),
             "scored": scored,
             # outcomes_a/outcomes_b are 2xR entries per pair and exist only to compute the
             # paired SE, which is retained. Dropped from the payload so the artifact stays
             # readable rather than being mostly raw trial dumps.
             "pairs": [{k: val for k, val in pr.items()
                        if not k.startswith("outcomes_")} for pr in pairs],
             "terminal_pairs_excluded": terminal_pairs,
             "skipped": dict(skipped)}, indent=1, default=str))
        print(f"wrote {args.json}")
    return 0


def _post_branch_history(branch_result, seat, prefix_history):
    """Prefix history plus the observation at the branched successor.

    ReplayBranchResult carries (prefix, branch_round, step_result); the post-branch
    observations live on step_result.observations (env.StepResult). An earlier revision read
    a non-existent `.observations` off the branch result itself, which would have appended
    nothing and silently scored the head on the PREFIX instead of the successor -- the two
    arms would then have produced identical values and every pair would have tied.
    """
    step_result = getattr(branch_result, "step_result", None)
    terminal = getattr(step_result, "terminal", None)
    obs = dict(getattr(step_result, "observations", {}) or {})
    nxt = obs.get(seat)
    if nxt is None:
        # Two legitimate reasons the seat has no observation: the branch ENDED the battle
        # (LocalShowdownEnv.step returns observations only for next_requested, and nothing
        # on terminal), or the seat is simply not requested next. A terminal branch is the
        # most informative pair in the sample -- arm A a winning KO against arm B a whiff
        # is |true_gap| ~ 1.0 -- so discarding it would systematically restrict the sample
        # to non-decisive positions and empty the wide-gap bucket for a reason that has
        # nothing to do with the head. Signalled, not raised.
        return None, terminal
    return tuple(prefix_history) + (nxt,), terminal


def _top_two_and_opponent(traj, seat, prefix, model, result, device,
                          evaluate_transformer_action_priors, history):
    """The two arms to compare, plus the opponent's reply to hold fixed.

    The opponent's action is FIXED across the two arms on purpose. This is a
    simultaneous-move game, so a successor is only defined given both actions; varying the
    opponent between arms would compare two different subgames and attribute the difference
    to the head.
    """
    step = next((s for s in traj.steps
                 if s.player_id == seat and s.turn_index == prefix), None)
    if step is None:
        raise LookupError(f"no step for {seat} at round {prefix}")
    mask = list(step.observation.legal_action_mask)
    legal = [i for i, ok in enumerate(mask) if ok]
    if len(legal) < 2:
        return None, None, 0
    # The FULL seat history, not a single observation. TransformerSoftmaxPolicy tensorises
    # history[-window_size:] (neural_policy.py:1725-1741), so a length-1 call returns a
    # different prior vector and the "top two" would not be the pair the model would
    # actually be deciding between -- worst exactly at the mid-game positions this probe
    # deliberately samples, where the history is longest.
    priors = evaluate_transformer_action_priors(
        model=model, result=result, observations=tuple(history), device=device)
    ranked = sorted(legal, key=lambda i: -priors[i])
    opp = "p2" if seat == "p1" else "p1"
    ostep = next((s for s in traj.steps
                  if s.player_id == opp and s.turn_index == prefix), None)
    opp_action = int(ostep.action_index) if ostep is not None else 0
    return ranked[0], ranked[1], opp_action


if __name__ == "__main__":
    raise SystemExit(main())
