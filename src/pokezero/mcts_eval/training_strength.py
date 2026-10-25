"""Prospectively registered-N training comparison; no collection authority.

Both seats stay in their seed cluster. The interval is an approximate two-sided
95% paired-t interval, extended over every feasible completion of round caps.
It does not change the frozen 128-seed Phase 0 reader or its advancement rules.
"""
from math import isfinite, sqrt

from .fullgame_strength import SEATS, _index, summarize_pairs


def _registered_count(count):
    if type(count) is not int or count < 2:
        raise ValueError("prospectively registered seed count of at least two required")
    return count


def _least_endpoint(endpoints, count, critical):
    # At a fixed sum, the largest sum of squares minimizes mean - t * SE.
    # Fixed coordinates are accumulated once; only uncertain coordinates enter
    # the dynamic program. No per-completion vectors or unknown-count cutoff.
    total = sum(lo for lo, hi in endpoints if lo == hi)
    squares = sum(lo * lo for lo, hi in endpoints if lo == hi)
    states = {total: squares}
    for lo, hi in endpoints:
        if lo == hi:
            continue
        following = {}
        for subtotal, squared_sum in states.items():
            for value in (lo, hi):
                new_total, new_squares = subtotal + value, squared_sum + value * value
                following[new_total] = max(following.get(new_total, -1), new_squares)
        states = following
    candidates = []
    for total, squares in states.items():
        se = sqrt((count * squares - total * total) / (count - 1)) / (4 * count)
        candidates.append((total / (4 * count) - critical * se, total, squares))
    endpoint, total, squares = min(candidates)
    return endpoint, dict(sum_quarter_units=total, sum_squared_quarter_units=squares,
                          retained_sum_states=len(states))


def registered_paired_interval(lower, upper, *, registered_seed_count):
    """Envelope of paired-t intervals over a fixed quarter-grid completion box.

    The lower endpoint is concave, so its minimum is at a box vertex. Reflecting
    the contrasts gives the upper endpoint. Exact refers to this optimization,
    not to finite-sample statistical coverage. Constant possible completions
    retain the legacy reader's undefined-t convention.
    """
    count = _registered_count(registered_seed_count)
    if len(lower) != count or len(upper) != count:
        raise ValueError("all registered seed contrast intervals required")
    endpoints = []
    for lo, hi in zip(lower, upper):
        if any(type(v) not in (int, float) or not isfinite(v)
               or not -1 <= v <= 1 or 4 * v != int(4 * v) for v in (lo, hi)):
            raise ValueError("finite quarter-grid contrast endpoints required")
        if lo > hi:
            raise ValueError("ordered contrast interval required")
        endpoints.append((int(4 * lo), int(4 * hi)))
    unknown = sum(lo != hi for lo, hi in endpoints)
    common = dict(seed_clusters=count, degrees_of_freedom=count - 1,
                  confidence_level=0.95, finite_sample_certified=False,
                  unknown_seed_clusters=unknown,
                  completion_optimization="EXACT_QUARTER_GRID")
    if max(lower) <= min(upper):
        return dict(common, status=("DEGENERATE_COMPLETION_POSSIBLE" if unknown
                                    else "DEGENERATE_VARIANCE"),
                    lower=None, upper=None, improvement_established=False)
    from scipy.stats import t

    critical = float(t.isf(0.025, count - 1))
    lo, lo_totals = _least_endpoint(endpoints, count, critical)
    reflected, reflected_totals = _least_endpoint([(-hi, -lo) for lo, hi in endpoints],
                                                 count, critical)
    # A numerically marginal boundary is not evidence of a strict improvement.
    numerical_guard = 1e-12
    return dict(common, status="APPROXIMATE_REGISTERED_N_PAIRED_T_ALL_COMPLETIONS",
                lower=lo, upper=-reflected, lower_completion_totals=lo_totals,
                reflected_upper_completion_totals=reflected_totals,
                numerical_guard=numerical_guard,
                improvement_established=lo > numerical_guard)


def summarize_training_pairs(seeds, treatment, control, *, registered_seed_count,
                             normal_complete_attempt=False):
    """Account for the full roster; infer only after independent final validation.

    The caller must establish normal completion, provenance and public outcomes.
    Round caps remain unknown [0,1]. Failed, refused, unstarted or unvalidated
    cells cannot be turned into an ordinary fixed-sample inference by imputation.
    No five-point rule or deployment qualification applies to this comparison.
    """
    count = _registered_count(registered_seed_count)
    if type(normal_complete_attempt) is not bool:
        raise ValueError("explicit normal-complete-attempt boolean required")
    treatment, control = tuple(treatment), tuple(control)
    result = summarize_pairs(seeds, treatment, control)
    if len(seeds) != count:
        raise ValueError("seed roster differs from prospectively registered count")
    # Keep descriptive arithmetic, not legacy Phase 0 decision fields.
    for key in ("primary_test", "screening_rule_passed", "deployment_point_rule_passed",
                "deployability_qualified", "interpretation"):
        result.pop(key)
    arms = [_index(treatment, seeds), _index(control, seeds)]
    scored = [[row.score for row in arm.values() if row.validated] for arm in arms]
    paired = [arms[0][seed, seat].score - arms[1][seed, seat].score
              for seed in seeds for seat in SEATS
              if all((seed, seat) in arm and arm[seed, seat].validated for arm in arms)]
    result.update(treatment_scored_point=sum(scored[0]) / len(scored[0]) if scored[0] else None,
                  control_scored_point=sum(scored[1]) / len(scored[1]) if scored[1] else None,
                  paired_scored_cells=len(paired),
                  paired_scored_point=sum(paired) / len(paired) if paired else None,
                  primary_interval=None, improvement_established=False,
                  statistical_unit="seed, with both seats averaged within seed",
                  interpretation="Approximate two-sided 95% mean-effect interval; every unknown retained. No collection or deployment authority.")
    reason = "NORMAL_FULL_ATTEMPT_NOT_ATTESTED"
    if normal_complete_attempt:
        accounted = all((row := arm.get((seed, seat))) is not None
                        and (row.validated or row.status == "CAPPED")
                        for arm in arms for seed in seeds for seat in SEATS)
        reason = None if accounted else "UNVALIDATED_OR_NON_CAP_CELL"
    if reason is None:
        interval = registered_paired_interval([r["lower"] for r in result["per_seed"]],
                                              [r["upper"] for r in result["per_seed"]],
                                              registered_seed_count=count)
        result.update(primary_interval=interval,
                      improvement_established=interval["improvement_established"])
    result["inference_unavailable_reason"] = reason
    return result
