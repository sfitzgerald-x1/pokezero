"""UNCALIBRATED fixed-128 Student-t candidates for synthetic comparison only.

This separate entry does not add t-tests to JointDesign or grant admission.
A sample-skewness gate is not a theorem of normality or finite-sample validity.
Unknown contrasts use a conservative moment envelope, not ordinary t variance
after lower-endpoint imputation; the skew gate abstains on uncertain data.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

from pokezero.mcts_eval.search_over_raw_statistics import (
    ARMS, primary_cluster_intervals, require, validate_intervals,
)


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate NIST DLMF 8.17.22/23 with scaled fraction updates.

    https://dlmf.nist.gov/8.17#v gives the coefficients and complementary branch.
    Numerical convergence is required; exhaustion is an error, not a p-value.
    """
    tiny = 1e-300
    c = 1.0
    denominator = 1 - (a + b) * x / (a + 1)
    if abs(denominator) < tiny:
        denominator = math.copysign(tiny, denominator)
    d = 1 / denominator
    value = d
    for m in range(1, 1001):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        for coefficient in (even, odd):
            denominator = 1 + coefficient * d
            numerator = 1 + coefficient / c
            if abs(denominator) < tiny:
                denominator = math.copysign(tiny, denominator)
            if abs(numerator) < tiny:
                numerator = math.copysign(tiny, numerator)
            d = 1 / denominator
            c = numerator
            update = c * d
            value *= update
        if abs(update - 1) < 2e-14:
            return value
    raise ArithmeticError("incomplete beta fraction failed to converge")


def _beta_regularized(a: float, b: float, x: float) -> float:
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _beta_regularized(b, a, 1 - x)
    factor = math.exp(a * math.log(x) + b * math.log1p(-x)
        - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
    return factor * _beta_fraction(a, b, x) / a


def student_survival(t: float, degrees: int) -> float:
    require(type(degrees) is int and 1 <= degrees <= 255, "degrees must be integer 1..255")
    require(type(t) in (float, int) and math.isfinite(t), "finite t statistic required")
    # Integrating the Student density after z=nu/(nu+t²) yields this beta form.
    positive_tail = .5 * _beta_regularized(degrees / 2, .5, degrees / (degrees + t * t))
    return positive_tail if t >= 0 else 1 - positive_tail


def student_critical(degrees: int, alpha: float = .025) -> float:
    require(type(alpha) in (float, int) and math.isfinite(alpha) and 0 < alpha < .5,
        "one-sided alpha must be finite in (0,.5)")
    student_survival(0., degrees)
    return _student_critical_cached(degrees, float(alpha))


@lru_cache(maxsize=32)
def _student_critical_cached(degrees: int, alpha: float) -> float:
    lower, upper = 0., 1.
    while student_survival(upper, degrees) > alpha:
        upper *= 2
    for _ in range(70):
        mid = (lower + upper) / 2
        if student_survival(mid, degrees) > alpha:
            lower = mid
        else:
            upper = mid
    return upper


@dataclass(frozen=True)
class TCandidateDesign:
    seeds: tuple[int, ...]
    skew_limit: float | None = .5

    def __post_init__(self):
        require(type(self.seeds) is tuple and len(self.seeds) == 128
            and all(type(s) is int and 0 <= s < 2**32 for s in self.seeds)
            and len(set(self.seeds)) == 128, "exact 128 distinct uint32 seed roster required")
        require(self.skew_limit is None or (type(self.skew_limit) is float and self.skew_limit == .5),
            "only the prospectively frozen .5 gate or plain comparator is supported")


def candidate_arm(intervals: Sequence[Sequence[float]], skew_limit: float | None) -> dict:
    validate_intervals(intervals)
    require(len(intervals) == 128, "no interim t looks")
    require(skew_limit is None or (type(skew_limit) is float and skew_limit == .5),
        "unregistered skew threshold")
    n = len(intervals)
    lower = [float(row[0]) for row in intervals]
    mean = math.fsum(lower) / n
    uncertain = any(lo != hi for lo, hi in intervals)
    centered = [x - mean for x in lower]
    second = math.fsum(x * x for x in centered)
    if uncertain:
        # For any positive-mean completion: mu>=L>0 and sum(x²)<=Qmax,
        # so sample variance <= (Qmax-n*L²)/(n-1). This is an outer bound,
        # not an exact completion optimizer or a finite-sample t theorem.
        max_squares = math.fsum(max(lo * lo, hi * hi) for lo, hi in intervals)
        variance_upper = max(0., max_squares - n * max(mean, 0.) ** 2) / (n - 1)
    else:
        variance_upper = second / (n - 1)
    statistic = math.sqrt(n) * max(mean, 0.) / math.sqrt(variance_upper) if variance_upper > 0 else 0.
    skew = None
    if not uncertain and second > 0:
        m2 = second / n
        m3 = math.fsum(x ** 3 for x in centered) / n
        skew = math.sqrt(n * (n - 1)) / (n - 2) * m3 / m2 ** 1.5
    gate_abstains = skew_limit is not None and (skew is None or abs(skew) > skew_limit)
    # Zero variance abstains even for positive constants. Rejecting an interval
    # roster compatible with that constant would violate completion dominance.
    constant_completion_possible = max(lo for lo, _ in intervals) <= min(hi for _, hi in intervals)
    zero_variance = variance_upper <= 0 or constant_completion_possible
    return dict(lower_mean=mean, uncertain=uncertain, variance_upper=variance_upper,
        completion_statistic_lower=statistic, adjusted_sample_skewness=skew,
        skew_gate_abstains=gate_abstains, zero_variance_abstains=zero_variance,
        constant_completion_possible=constant_completion_possible,
        nominal_crossing=bool(mean > 0 and not zero_variance and not gate_abstains
            and statistic > student_critical(n - 1)),
        finite_sample_validity_established=False)


def adjudicate_t_candidate(design: TCandidateDesign, rows: Sequence[dict]) -> dict:
    require(isinstance(design, TCandidateDesign), "separate synthetic t design required")
    require(len(rows) <= 128, "roster overflow")
    intervals = {arm: [] for arm in ARMS}
    refused = 0
    terminal = None
    for i, row in enumerate(rows):
        require(terminal is None, "rows after terminal stop")
        require(set(row) == {"seed", "contrasts", "refused", "provenance_ok"}, "closed t record required")
        require(type(row["seed"]) is int and row["seed"] == design.seeds[i], "changed seed prefix")
        require(type(row["refused"]) is bool and type(row["provenance_ok"]) is bool, "explicit status flags")
        require(set(row["contrasts"]) == set(ARMS), "both contrasts required")
        for arm in ARMS:
            pair = row["contrasts"][arm]
            validate_intervals([pair])
            intervals[arm].append(pair)
        refused += int(row["refused"] or any(row["contrasts"][arm][0] != row["contrasts"][arm][1] for arm in ARMS))
        if not row["provenance_ok"]:
            terminal = "PROVENANCE_FAILURE"
        elif refused > 2:
            terminal = "REFUSAL_CAP_EXCEEDED"
    evaluations = {arm: None for arm in ARMS}
    if terminal is None and len(rows) == 128:
        terminal = "FIXED_CAP_REACHED"
        evaluations = {arm: candidate_arm(intervals[arm], design.skew_limit) for arm in ARMS}
    return dict(status=terminal or "NONTERMINAL_PREFIX", observed_clusters=len(rows), refused_clusters=refused,
        candidate_evaluations=evaluations,
        nominal_primary_crossings={arm: bool(evaluations[arm] and evaluations[arm]["nominal_crossing"]) for arm in ARMS},
        full_roster_identification={arm: [
            (math.fsum(pair[0] for pair in intervals[arm]) - (128 - len(rows))) / 128,
            (math.fsum(pair[1] for pair in intervals[arm]) + (128 - len(rows))) / 128] for arm in ARMS},
        scientific_admission=False, phase_b_authorized=False, method_adopted=False,
        finite_sample_validity_established=False)


def adjudicate_t_primary_cells(design: TCandidateDesign, rows: Sequence[dict]) -> dict:
    converted = []
    for row in rows:
        require(set(row) == {"seed", "primary_cells", "refused", "provenance_ok"}, "closed six-cell t record required")
        converted.append(dict(seed=row["seed"], contrasts=primary_cluster_intervals(row["primary_cells"]),
            refused=row["refused"], provenance_ok=row["provenance_ok"]))
    return adjudicate_t_candidate(design, converted)
