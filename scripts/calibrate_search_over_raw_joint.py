"""Create-only SYNTHETIC joint calibration; never launch/adopt a battle study.

Compares frozen-stake betting, Hoeffding, empirical Bernstein, three caps,
shared controls, partial nulls, uncertain endpoints and a two-refused-cluster
cap. Independent-seed assumptions are explicit; cross-cluster dependence is a
negative control, not admissible calibration evidence. No historical outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from pokezero.mcts_eval.search_over_raw_statistics import ARMS, JointDesign, adjudicate

CAPS = (64, 128, 256)
METHODS = (("fixed_stake_betting", .2), ("fixed_stake_betting", .4),
    ("fixed_stake_betting", .6), ("hoeffding", .4), ("empirical_bernstein", .4))
SCENARIOS = (
    ("null_all_ties", "ties", (0., 0.), "none"),
    ("null_low_variance", "low", (0., 0.), "none"),
    ("null_extreme_seat_dependence", "extreme", (0., 0.), "none"),
    ("null_skew_positive_frequent", "skew_positive", (0., 0.), "none"),
    ("null_skew_negative_frequent", "skew_negative", (0., 0.), "none"),
    ("null_negative_mean", "negative", (-.05, -.05), "none"),
    ("null_shared_raw_control", "games", (0., 0.), "none"),
    ("partial_null_incumbent", "games", (0., .15), "none"),
    ("partial_null_reference", "games", (.15, 0.), "none"),
    ("partial_null_skew", "partial_skew", (0., .1), "none"),
    ("both_plus05_shared_raw", "games", (.05, .05), "none"),
    ("both_plus10_shared_raw", "games", (.1, .1), "none"),
    ("both_plus15_shared_raw", "games", (.15, .15), "none"),
    ("both_plus20_shared_raw", "games", (.2, .2), "none"),
    ("null_shared_raw_missing05", "games", (0., 0.), "random"),
    ("null_positive_outcomes_missing25", "skew_positive", (0., 0.), "positive"),
    ("both_plus10_random_missing05", "games", (.1, .1), "random"),
    ("both_plus10_positive_missing25", "games", (.1, .1), "positive"),
    ("INVALID_cross_cluster_dependence", "dependent", (0., 0.), "none"),
)


def write_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(payload)


def rate(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return dict(count=k, replicates=n, estimate=p,
        monte_carlo_95pct_wilson=[max(0., center - half), min(1., center + half)])


def generate(rng, scenario, reps):
    _, law, deltas, missing_law = scenario
    shape = (reps, 256, 2)
    if law == "games":
        # A SINGLE raw control is shared by both contrasts. Two independent
        # seats per policy, losses0/draws.5/wins1; not the old four-game study.
        uniforms = rng.random((reps, 256, 3, 2))
        pwin = np.asarray((.45, .45 + deltas[0], .45 + deltas[1]))[None, None, :, None]
        score = np.where(uniforms < pwin, 1., np.where(uniforms < pwin + .1, .5, 0.))
        means = score.mean(axis=3)
        candidate_means = means[:, :, 1:]
        d = means[:, :, 1:] - means[:, :, :1]
    elif law == "ties":
        d = np.zeros(shape)
    elif law == "low":
        d = rng.choice((-.25, 0., .25), shape, p=(.2, .6, .2))
    elif law == "extreme":
        d = rng.choice((-1., 1.), shape)
    elif law == "skew_positive":
        d = rng.choice((-1., .25), shape, p=(.2, .8))
    elif law == "skew_negative":
        d = rng.choice((-.25, 1.), shape, p=(.8, .2))
    elif law == "negative":
        d = rng.choice((-.5, .25), shape, p=(.4, .6))
    elif law == "partial_skew":
        d = np.stack((rng.choice((-1., .25), shape[:2], p=(.2, .8)),
            rng.choice((-.25, .25), shape[:2], p=(.3, .7))), axis=2)
    else:
        assert law == "dependent"
        d = np.broadcast_to(rng.choice((-1., 1.), (reps, 1, 2)), shape).copy()
    if missing_law == "random":
        # Both raw-vs-FoulPlay seats refuse together in5% of clusters. Known
        # candidate scores are retained: c minus an unknown raw mean is[c-1,c].
        assert law == "games"
        missing = np.broadcast_to((rng.random(shape[:2]) < .05)[:, :, None], shape)
    elif missing_law == "positive":
        missing = (d > 0) & (rng.random(shape) < .25)
    else:
        missing = np.zeros(shape, dtype=bool)
    if missing_law == "random":
        return (np.where(missing, candidate_means - 1, d),
            np.where(missing, candidate_means, d), missing.any(axis=2))
    return np.where(missing, -1., d), np.where(missing, 1., d), missing.any(axis=2)


def vector_adjudicate(lower, upper, refused, design):
    """Fast batch equivalent, checked against the production scalar kernel."""
    reps = len(lower)
    cap = len(design.seeds)
    crossings = np.zeros((reps, 2), dtype=int)
    stop = np.full(reps, cap, dtype=int)
    active = np.ones(reps, dtype=bool)
    refusal_count = np.zeros(reps, dtype=int)
    wealth = np.zeros((reps, 2))
    cap_failure = np.zeros(reps, dtype=bool)
    for n in range(1, cap + 1):
        refusal_count += refused[:, n - 1] & active
        new_cap_failure = active & (refusal_count > design.refusal_cap)
        stop[new_cap_failure] = n
        cap_failure |= new_cap_failure
        active &= ~new_cap_failure
        wealth += np.log1p(lower[:, n - 1] * np.asarray(design.stakes))
        if n not in design.looks:
            continue
        if design.method == "fixed_stake_betting":
            crossed = wealth >= math.log(2 / design.alpha)
        else:
            alpha = design.alpha / (2 * len(design.looks))
            values = lower[:, :n]
            if design.method == "hoeffding":
                lcb = values.mean(axis=1) - math.sqrt(2 * math.log(1 / alpha) / n)
            else:
                logarithm = math.log(2 / alpha)
                lcb = values.mean(axis=1) - np.sqrt(2 * values.var(axis=1, ddof=1) * logarithm / n)
                lcb -= 14 * logarithm / (3 * (n - 1))
                lcb = np.where((values != upper[:, :n]).any(axis=1), -np.inf, lcb)
            crossed = lcb > 0
        crossings = np.where(active[:, None] & (crossings == 0) & crossed, n, crossings)
        both = active & (crossings > 0).all(axis=1)
        stop[both] = n
        active &= ~both
    return crossings, stop, cap_failure


def check_scalar_parity(lower, upper, refused, design, vector):
    crossings, stop, cap_failure = vector
    for replicate in range(min(3, len(lower))):
        rows = [dict(seed=design.seeds[i], refused=bool(refused[replicate, i]), provenance_ok=True,
            contrasts={arm: (float(lower[replicate, i, j]), float(upper[replicate, i, j]))
                for j, arm in enumerate(ARMS)}) for i in range(int(stop[replicate]))]
        result = adjudicate(design, rows)
        actual = [result["primary_crossings"][arm] or 0 for arm in ARMS]
        expected_status = ("REFUSAL_CAP_EXCEEDED" if cap_failure[replicate] else
            "BOTH_PRIMARY_THRESHOLDS_CROSSED" if all(crossings[replicate]) else "CAP_REACHED")
        if (actual != crossings[replicate].tolist() or result["status"] != expected_status
                or result["observed_clusters"] != int(stop[replicate])):
            raise AssertionError("batch/scalar adjudication mismatch")


def run(output, *, replicates=20000, seed=2026101015, assumed_seconds_per_cluster=None):
    if type(replicates) is not int or replicates < 1000:
        raise ValueError("at least1000 synthetic replicates required")
    if assumed_seconds_per_cluster is not None and (not math.isfinite(assumed_seconds_per_cluster)
            or assumed_seconds_per_cluster <= 0):
        raise ValueError("positive explicitly hypothetical cluster cost required")
    root = Path(__file__).resolve().parent.parent
    inputs = [Path(__file__).resolve(), root / "src/pokezero/mcts_eval/search_over_raw_statistics.py",
        root / "src/pokezero/mcts_eval/search_over_raw.py", root / "tests/test_search_over_raw_statistics.py"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    designs = [JointDesign(tuple(range(cap)), tuple(range(16, cap + 1, 16)), method, (stake, stake))
        for cap in CAPS for method, stake in METHODS]
    contract = dict(kind="pokezero.synthetic-joint-calibration.v1", status="REGISTERED_NOT_ADOPTED",
        seed=seed, replicates=replicates, chunk_replicates=1000, designs=[asdict(d) for d in designs],
        scenarios=SCENARIOS, input_sha256=hashes, scientific_calls=0, historical_outcomes_read=False,
        assumptions=["independent identically distributed seed clusters in a fixed prospective order",
            "both arms/control frozen before collection; within-cluster dependence unrestricted",
            "fixed Bonferroni alpha/2, NOT Holm; fixed stakes chosen before study outcomes",
            "registered looks; stop after both crossings or third refused cluster; no retries",
            "empirical Bernstein abstains on uncertain prefixes; lower imputation is not variance-monotone"],
        statistical_extension="Joint shared-raw-control six-primary-cell generator rather than independent old four-game contrasts; six secondary cells add cost, not primary outcomes",
        missingness_laws={"random": "5% of seed clusters lose both sharedraw primary seats; preserve known candidate scores with[c-1,c] bounds",
            "positive": "25% of positive underlying arm contrasts become[-1,1]; conservative generic whole-contrast erasure, not an exact six-cell failure law"},
        discrete_laws="General bounded-contrast stress, not necessarily jointly realizable six-game outcomes",
        effect_target_used_for_stake_sensitivity=.1, no_live_stake_selection=True,
        assumed_seconds_per_cluster=assumed_seconds_per_cluster,
        runtime_scope="12games per full PhaseB seed; optional cost is a hypothetical planning assumption, NOT measured feasibility",
        theorem_sources=["https://arxiv.org/abs/2010.09686", "https://arxiv.org/abs/0907.3740"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "contract.json", contract)
    started = time.perf_counter()
    results = []
    children = np.random.SeedSequence(seed).spawn(len(SCENARIOS))
    for scenario, child in zip(SCENARIOS, children):
        rng = np.random.default_rng(child)
        totals = [dict(false_any=0, detected=[0, 0], both=0, refusals=0, clusters=0, clusters_squared=0)
            for _ in designs]
        for offset in range(0, replicates, 1000):
            batch = min(1000, replicates - offset)
            lower, upper, refused = generate(rng, scenario, batch)
            for design, counts in zip(designs, totals):
                vector = vector_adjudicate(lower, upper, refused, design)
                if offset == 0:
                    check_scalar_parity(lower, upper, refused, design, vector)
                crossings, stop, cap_failure = vector
                # A threshold crossed before a later operational refusal-cap
                # failure does not count as an admissible study result.
                detected = (crossings > 0) & ~cap_failure[:, None]
                true_null = np.asarray(scenario[2]) <= 0
                counts["false_any"] += int((detected & true_null).any(axis=1).sum())
                counts["detected"] = [old + int(new) for old, new in zip(counts["detected"], detected.sum(axis=0))]
                counts["both"] += int(detected.all(axis=1).sum())
                counts["refusals"] += int(cap_failure.sum())
                counts["clusters"] += int(stop.sum())
                counts["clusters_squared"] += int((stop.astype(float) ** 2).sum())
        for design, counts in zip(designs, totals):
            mean = counts["clusters"] / replicates
            variance = max(0., (counts["clusters_squared"] - replicates * mean ** 2) / (replicates - 1))
            results.append(dict(scenario=scenario[0], outside_assumptions=scenario[1] == "dependent",
                true_advantages=dict(zip(ARMS, scenario[2])), cap=len(design.seeds), method=design.method,
                stakes=design.stakes, familywise_false_positive=rate(counts["false_any"], replicates),
                primary_detection={arm: rate(counts["detected"][i], replicates) for i, arm in enumerate(ARMS)},
                both_detected=rate(counts["both"], replicates), refusal_cap_stop=rate(counts["refusals"], replicates),
                expected_clusters=mean, clusters_standard_error=math.sqrt(variance / replicates),
                expected_full_cluster_game_budget=12 * mean,
                projected_sequential_hours=None if assumed_seconds_per_cluster is None else mean * assumed_seconds_per_cluster / 3600))
        print(scenario[0] + " completed", flush=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in hashes.items()):
        raise RuntimeError("source drift during synthetic calibration")
    result = dict(kind="pokezero.synthetic-joint-calibration-result.v1", status="COMPLETE_NOT_ADOPTED",
        contract_sha256=hashlib.sha256((output / "contract.json").read_bytes()).hexdigest(),
        input_sha256=hashes, numpy_version=np.__version__, computation_seconds=time.perf_counter() - started,
        scalar_vector_parity_checked=True, source_hashes_unchanged=True, results=results,
        simulation_not_a_proof=True, no_optimized_stake_adopted=True, scientific_calls=0,
        detection_definition="registered threshold crossing with no later refusal-cap failure before terminal; not scientific admission",
        phase_b_authorized=False, no_representative_runtime_claim=True)
    write_new(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026101015)
    parser.add_argument("--assumed-seconds-per-cluster", type=float)
    args = parser.parse_args()
    run(args.output, replicates=args.replicates, seed=args.seed,
        assumed_seconds_per_cluster=args.assumed_seconds_per_cluster)
