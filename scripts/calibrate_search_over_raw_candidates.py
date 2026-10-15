#!/usr/bin/env python3
"""Synthetic fixed-look Phase B candidates on the original amendment laws.

No games, historical outcomes, method adoption, interim looks or holdout access.
Studentization does not change a one-sample sign-flip ordering: conditional on
the magnitudes, Q=sum(x*x) is fixed and t(s)=sqrt(n-1)*s/sqrt(n*Q-s*s)
is increasing in the signed sum s. Thus its exact p-value equals the plain
mean sign-flip p-value, including skewed null failures. FFT computes that
finite randomization distribution on the original quarter-score grid.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

NULLS = (
    ("null_all_ties", (0.,), (1.,)),
    ("null_low_variance", (-.25, 0., .25), (.2, .6, .2)),
    ("null_extreme_seat_dependence", (-1., 1.), (.5, .5)),
    ("null_skew_positive_frequent", (-1., .25), (.2, .8)),
    ("null_skew_negative_frequent", (-.25, 1.), (.8, .2)),
    ("null_negative_mean", (-.5, .25), (.4, .6)),
)


@lru_cache(maxsize=4096)
def randomization_cdf(counts):
    total = sum(weight * count for weight, count in enumerate(counts))
    size = 1 << max(0, total.bit_length())
    frequency = np.arange(size)
    transform = np.ones(size, complex)
    for weight, count in enumerate(counts):
        if weight and count:
            transform *= ((1 + np.exp(-2j * np.pi * frequency * weight / size)) / 2) ** count
    probability = np.maximum(np.fft.ifft(transform).real[:total + 1], 0.)
    probability /= probability.sum()
    return np.cumsum(probability)


def studentized_signflip_pvalue(values):
    x = np.asarray(values)
    integers = np.rint(x * 4).astype(int)
    if not np.all(np.isfinite(x)) or not np.all(integers / 4 == x) or np.any(np.abs(integers) > 4):
        raise ValueError("finite original quarter-score contrasts in [-1,1] required")
    counts = tuple(np.bincount(np.abs(integers), minlength=5).tolist())
    total = sum(i * count for i, count in enumerate(counts))
    cutoff = (total - int(integers.sum())) // 2
    return float(np.clip(randomization_cdf(counts)[cutoff], 0., 1.))


def empirical_bernstein_reject(values, alpha=.025):
    """Existing range-two Maurer-Pontil bound, fixed look, no uncertain cells."""
    x = np.asarray(values)
    n = x.shape[-1]
    logarithm = math.log(2 / alpha)
    return x.mean(axis=-1) > (np.sqrt(2 * x.var(axis=-1, ddof=1) * logarithm / n)
        + 14 * logarithm / (3 * (n - 1)))


def four_games(rng, delta, reps, n):
    # Exact original October9/10 generator, not a new shared-control law.
    u = rng.random((reps, n, 4))
    win = np.array([.45 + delta / 2] * 2 + [.45 - delta / 2] * 2)
    score = np.where(u < win, 1., np.where(u < win + .1, .5, 0.))
    return (score[:, :, 0] + score[:, :, 1] - score[:, :, 2] - score[:, :, 3]) / 2


def rate(flags):
    n = len(flags)
    k = int(np.count_nonzero(flags))
    p, z = k / n, 1.959963984540054
    den = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(count=k, replicates=n, rate=p,
        monte_carlo_interval95=[max(0., middle - radius), min(1., middle + radius)])


def evaluate(values, unknown, alpha):
    # Lower imputation for sign-flip is tested, NOT asserted variance-monotone.
    lower = np.where(unknown, -1., values)
    flip = np.array([studentized_signflip_pvalue(row) <= alpha for row in lower])
    bernstein = empirical_bernstein_reject(values, alpha) & ~unknown.any(axis=1)
    return dict(studentized_signflip=rate(flip), empirical_bernstein=rate(bernstein),
        empirical_bernstein_abstentions=int(unknown.any(axis=1).sum()),
        signflip_missing_rule="experimental lower endpoint imputation; empirical calibration required",
        bernstein_missing_rule="abstain on any uncertain seed; no complete-case deletion")


def run(replicates=10000, sizes=(64, 128, 256), alpha=.025, seed=2026101502):
    rng = np.random.default_rng(seed)
    results = []
    for n in sizes:
        cases = [(name, rng.choice(values, (replicates, n), p=probs), "false_positive")
            for name, values, probs in NULLS]
        cases += [(f"four_games_delta_{delta:.2f}", four_games(rng, delta, replicates, n),
            "false_positive" if delta == 0 else "power") for delta in (0., .05, .10, .15, .20)]
        for name, values, metric in cases:
            missingness = ("none", "random05", "positive25") if name in (
                "null_skew_positive_frequent", "four_games_delta_0.10") else ("none",)
            for missing in missingness:
                unknown = np.zeros(values.shape, bool)
                if missing == "random05":unknown = rng.random(values.shape) < .05
                if missing == "positive25":unknown = (values > 0) & (rng.random(values.shape) < .25)
                results.append(dict(family=name, seed_clusters=n, metric=metric,
                    missingness=missing, **evaluate(values, unknown, alpha)))
    return dict(schema="pokezero.phase-b-candidate-synthetic-comparison.v1", results=results,
        replicates=replicates, rng_seed=seed, alpha_per_primary=alpha,
        two_primary_rule="Bonferroni .025 per arm; no joint FWER estimate from this single-contrast generator",
        looks="one fixed final look per declared n; not an interim-monitoring design",
        source_families="Original October9 amendment six discrete nulls and October10 four_games generator",
        studentization_result="exact same one-sided ordering and p-values as ordinary one-sample mean sign-flips; studentization cannot repair skewness here",
        randomization_numerics="finite sign distribution evaluated by floating-point FFT, checked against exhaustive small fixtures",
        bernstein_source="https://arxiv.org/abs/0907.3740",
        scientific_calls=0, method_adopted=False, holdout_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("at least100 replicates required")
    result = run(args.replicates)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(args.output), conditions=len(result["results"]))))


if __name__ == "__main__":
    main()
