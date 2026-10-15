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
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

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


def joint_candidate(lower, upper, refused, method):
    """Original t-comparison stopping law, replacing only its terminal test."""
    if lower.shape != upper.shape or lower.shape[1:] != (256, 2) or refused.shape != lower.shape[:2]:
        raise ValueError("original 256-by-two generator batch required")
    lo, hi = lower[:, :128], upper[:, :128]
    missing = (lo != hi).any(axis=2) | refused[:, :128]
    hit = np.cumsum(missing, axis=1) > 2
    failed = hit.any(axis=1)
    stop = np.where(failed, hit.argmax(axis=1) + 1, 128)
    crossed = np.zeros((len(lo), 2), bool)
    abstains = np.zeros_like(crossed)
    accepted = np.flatnonzero(~failed)
    if method == "empirical_bernstein":
        abstains[accepted] = (lo[accepted] != hi[accepted]).any(axis=1)
        crossed[accepted] = empirical_bernstein_reject(lo[accepted].transpose(0, 2, 1)) & ~abstains[accepted]
    elif method == "studentized_signflip":
        for r in accepted:
            for arm in range(2):
                crossed[r, arm] = studentized_signflip_pvalue(lo[r, :, arm]) <= .025
    else:
        raise ValueError("unknown candidate")
    return dict(crossed=crossed, stop=stop, failed=failed, abstains=abstains)


def run_joint(generator_path, *, replicates=20000, seed=2026101301):
    """Same 19 families, cohorts and six-cell join as the published t runner.

    Load its unchanged generator and scalar cell join from its own checkout.
    This is a synthetic comparison, not a change to the live scientific runtime.
    """
    if type(replicates) is not int or replicates < 1000:
        raise ValueError("at least 1000 replicates required")
    path = Path(generator_path).resolve()
    source = path.parent.parent / "src"
    if any(name == "pokezero" or name.startswith("pokezero.") for name in sys.modules):
        raise RuntimeError("run the joint comparison in a fresh process with its generator checkout")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("published_joint_generator", path)
    joint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joint)
    from pokezero.mcts_eval.search_over_raw_statistics import primary_cluster_intervals
    stats_path = source / "pokezero/mcts_eval/search_over_raw_statistics.py"
    inputs = (Path(__file__).resolve(), path, stats_path,
        path.parent / "calibrate_search_over_raw_t_candidate.py")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    methods = ("empirical_bernstein", "studentized_signflip")
    results = []
    wall, cpu = time.perf_counter(), time.process_time()
    for scenario, child in zip(joint.SCENARIOS, np.random.SeedSequence(seed).spawn(len(joint.SCENARIOS))):
        rng = np.random.default_rng(child)
        totals = {m: dict(false=0, detected=np.zeros(2, int), both=0, refusals=0,
            abstains=np.zeros(2, int), clusters=0, squared=0) for m in methods}
        for offset in range(0, replicates, 1000):
            faithful = scenario[1] == "games" and scenario[3] in {"none", "random"}
            data = joint.generate(rng, scenario, min(1000, replicates - offset), original_cells=faithful)
            lower, upper, refused = data[:3]
            if offset == 0 and faithful:
                scores, known = data[3:]
                for r in range(min(3, len(lower))):
                    for i in range(128):
                        cells = {(arm, seat): dict(status="COMPLETE" if known[r, i, a, s] else "REFUSED",
                            validated=bool(known[r, i, a, s]),
                            score=float(scores[r, i, a, s]) if known[r, i, a, s] else None)
                            for a, arm in enumerate(("raw", *joint.ARMS)) for s, seat in enumerate(("p1", "p2"))}
                        joined = primary_cluster_intervals(cells)
                        if any(tuple(joined[arm]) != (lower[r, i, a], upper[r, i, a])
                                for a, arm in enumerate(joint.ARMS)):
                            raise AssertionError("original six-cell contrast parity failed")
            for method, counts in totals.items():
                vector = joint_candidate(lower, upper, refused, method)
                crossed, stop = vector["crossed"], vector["stop"]
                counts["false"] += int((crossed & (np.asarray(scenario[2]) <= 0)).any(axis=1).sum())
                counts["detected"] += crossed.sum(axis=0)
                counts["both"] += int(crossed.all(axis=1).sum())
                counts["refusals"] += int(vector["failed"].sum())
                counts["abstains"] += vector["abstains"].sum(axis=0)
                counts["clusters"] += int(stop.sum())
                counts["squared"] += int((stop.astype(float) ** 2).sum())
        for method, counts in totals.items():
            mean = counts["clusters"] / replicates
            variance = max(0., (counts["squared"] - replicates * mean ** 2) / (replicates - 1))
            results.append(dict(scenario=scenario[0], outside_assumptions=scenario[1] == "dependent",
                true_advantages=dict(zip(joint.ARMS, scenario[2])), method=method,
                familywise_false_positive=joint.rate(counts["false"], replicates),
                primary_detection={arm: joint.rate(int(counts["detected"][a]), replicates) for a, arm in enumerate(joint.ARMS)},
                both_detected=joint.rate(counts["both"], replicates), refusal_cap_stop=joint.rate(counts["refusals"], replicates),
                uncertainty_abstention={arm: joint.rate(int(counts["abstains"][a]), replicates) for a, arm in enumerate(joint.ARMS)},
                expected_clusters=mean, clusters_standard_error=math.sqrt(variance / replicates),
                expected_full_cluster_game_budget=12 * mean, projected_sequential_hours=None))
        print(scenario[0] + " completed", flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("synthetic comparison input drift")
    return dict(schema="pokezero.phase-b-candidate-joint-comparison.v1", results=results,
        replicates=replicates, rng_seed=seed, chunk_replicates=1000, scenarios=joint.SCENARIOS,
        input_sha256=hashes, alpha_per_primary=.025, family_alpha=.05, terminal_look=128,
        refusal_rule="third refused or uncertain cluster stops both methods without rejection",
        original_cell_parity_checked=True, source_hashes_unchanged=True,
        signflip_missing_rule="experimental lower endpoints; no general validity proof",
        bernstein_missing_rule="abstain on any uncertain arm; no complete-case deletion",
        assumptions="IID bounded seed clusters, unrestricted shared control within seed; dependent family is a negative control",
        computation_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
        runtime_scope="synthetic CPU/wall and expected 12-game cluster budgets only; real game hours unmeasured",
        scientific_calls=0, method_adopted=False, holdout_opened=False, calibration_certified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--joint-generator", type=Path,
        help="published 19-family joint driver; uses its fixed-128 t-comparison cohorts and stopping law")
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("at least100 replicates required")
    result = run_joint(args.joint_generator, replicates=args.replicates) if args.joint_generator else run(args.replicates)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(args.output), conditions=len(result["results"]))))


if __name__ == "__main__":
    main()
