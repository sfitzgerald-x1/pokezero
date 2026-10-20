"""Fixed-roster Phase 0 arithmetic, not collection or deployment authority.

Both seats stay inside each seed cluster. Missing/capped/refused games are
bounded, never scored as draws or removed. The fixed-N paired t procedure is
approximate mean-effect inference, not finite-sample certification.
"""
from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean, stdev


SEATS = ("p1", "p2")
STATUSES = {"COMPLETE", "CAPPED", "REFUSED", "FAILED", "UNSTARTED"}
PRIMARY_SEEDS = 128
ALPHA = 0.025


@dataclass(frozen=True)
class Outcome:
    seed: int
    seat: str
    status: str
    score: float | None = None
    validated: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("unsigned 32-bit seed required")
        if self.seat not in SEATS or self.status not in STATUSES:
            raise ValueError("known seat and outcome status required")
        if type(self.validated) is not bool:
            raise ValueError("explicit boolean validation required")
        if self.score is not None and (
            type(self.score) not in (int, float) or self.score not in (0, 0.5, 1)
        ):
            raise ValueError("win/loss/genuine-draw score required")
        if self.validated and (self.status != "COMPLETE" or self.score is None):
            raise ValueError("only a scored COMPLETE outcome can validate")

    def bounds(self):
        # Legacy wire scores (including a capped .5) are not observations.
        return (self.score, self.score) if self.validated else (0.0, 1.0)


def _index(rows, seeds):
    result = {}
    for row in rows:
        if not isinstance(row, Outcome) or row.seed not in seeds:
            raise ValueError("only registered Outcome rows accepted")
        key = (row.seed, row.seat)
        if key in result:
            raise ValueError("duplicate seed-seat outcome")
        result[key] = row
    return result


def paired_t(values):
    """Frozen 128-seed, one-sided nominal .025 test on pessimistic contrasts."""
    if len(values) != PRIMARY_SEEDS:
        raise ValueError("primary inference requires all 128 registered seed contrasts")
    if any(type(v) not in (int, float) or not isfinite(v) or not -1 <= v <= 1 for v in values):
        raise ValueError("finite bounded seed contrasts required")
    mean = fmean(values)
    standard_error = stdev(values) / sqrt(PRIMARY_SEEDS)
    common = dict(mean=mean, standard_error=standard_error, degrees_of_freedom=127,
                  alpha=ALPHA, finite_sample_certified=False)
    if standard_error == 0:
        return dict(common, status="DEGENERATE_VARIANCE", lower=None,
                    nominal_one_sided_p=None, screening_rule_passed=False)
    from scipy.stats import t

    lower = mean - float(t.isf(ALPHA, 127)) * standard_error
    return dict(common, status="APPROXIMATE_FIXED_N_PAIRED_T", lower=lower,
                nominal_one_sided_p=float(t.sf(mean / standard_error, 127)),
                screening_rule_passed=lower > 0)


def summarize_pairs(seeds, candidate, control, *, terminal=False):
    """Join all registered cells; caller must establish durable run termination.

    This pure reader does not attest process/provenance or authorize advancement.
    A false terminal flag leaves inference and screening unavailable.
    """
    if type(terminal) is not bool:
        raise ValueError("explicit terminal boolean required")
    if (type(seeds) is not list or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("fixed unique unsigned 32-bit seed roster required")
    arms = [_index(candidate, seeds), _index(control, seeds)]
    per_seed = []
    known = [0, 0]
    for seed in seeds:
        bounds = []
        for index, arm in enumerate(arms):
            seat_bounds = []
            for seat in SEATS:
                row = arm.get((seed, seat), Outcome(seed, seat, "UNSTARTED"))
                known[index] += row.validated
                seat_bounds.append(row.bounds())
            bounds.append(tuple(fmean(b[i] for b in seat_bounds) for i in (0, 1)))
        per_seed.append(dict(seed=seed, lower=bounds[0][0] - bounds[1][1],
                             upper=bounds[0][1] - bounds[1][0]))
    lower = [r["lower"] for r in per_seed]
    upper = [r["upper"] for r in per_seed]
    identification_bounds = [fmean(lower), fmean(upper)]
    inference = paired_t(lower) if terminal and len(seeds) == PRIMARY_SEEDS else None
    return dict(
        seed_clusters=len(seeds), planned_games_per_arm=2 * len(seeds),
        candidate_validated_games=known[0], control_validated_games=known[1],
        identification_bounds=identification_bounds,
        identified_point_estimate=(identification_bounds[0]
                                   if identification_bounds[0] == identification_bounds[1] else None),
        per_seed=per_seed, primary_test=inference,
        screening_rule_passed=bool(inference and inference["screening_rule_passed"]),
        deployment_point_rule_passed=bool(inference and inference["screening_rule_passed"]
                                          and identification_bounds[0] >= 0.05),
        pessimistic_histogram=dict(sorted(Counter(lower).items())),
        nonzero_pessimistic_clusters=sum(v != 0 for v in lower),
        largest_absolute_pessimistic_cluster=max(abs(v) for v in lower),
        deployability_qualified=False,
        interpretation="score units; multiply by100 for percentage points; nominal .025 per candidate, not familywise",
    )
