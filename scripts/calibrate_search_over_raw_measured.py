#!/usr/bin/env python3
"""Measured Phase A variance and variance-matched screening sensitivity.

Read completed exploration output only. Never open a holdout, allocate seeds,
change a screen, or treat missing contrasts as complete cases. The fitted
Gaussian random-effects family is an analytical extension, not a validity
guarantee or a replacement for the predeclared non-Gaussian null families.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("measured_screen_kernel",
    ROOT / "src/pokezero/mcts_eval/search_over_raw_screening.py")
kernel = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kernel
_spec.loader.exec_module(kernel)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_panel(readout_path):
    path = Path(readout_path)
    readout = json.loads(path.read_text())
    entry = json.loads((path.parent / "entry.json").read_text())
    contract = entry["contract"]
    panel = contract["panels"]["exploration"]
    seeds, slots = panel["seeds"], panel["root_slots"]
    if len(seeds) != 32 or len(set(seeds)) != 32 or len(slots) != 200:
        raise ValueError("the fixed 32-seed/200-root panel is required")
    configs = [c for c in contract["configurations"]
        if c["arm"] != "raw" and c["belief"] == "public"]
    if not configs:
        raise ValueError("no measured deployable configurations")
    rejected = {str(Path(row["path"]).resolve()) for row in readout["rejected_root_receipts"]}
    rows = [json.loads(p.read_text()) for p in sorted((path.parent / "completed-roots").glob("*.json"))
        if str(p.resolve()) not in rejected]
    by_id = {row["root_id"]: row for row in rows}
    roster = {row["root_id"]: row for row in slots}
    if len(by_id) != len(rows) or not set(by_id) <= set(roster):
        raise ValueError("duplicate or foreign completed roots")
    if len(rows) != readout["recorded_roots"]:
        raise ValueError("completed rows differ from the final owning readout")
    groups, missing = [], 0
    for seed in seeds:
        values = []
        for slot in (r for r in slots if r["source_seed"] == seed):
            row = by_id.get(slot["root_id"])
            if row is not None and (row["source_seed"] != seed
                    or row["source_pipeline_validated"] is not True):
                raise ValueError("source/seed mismatch")
            vector = []
            for config in configs:
                interval = [-1., 1.] if row is None else row["root_intervals"][digest(config)]
                if (len(interval) != 2 or not all(np.isfinite(interval))
                        or not -1 <= interval[0] <= interval[1] <= 1):
                    raise ValueError("invalid bounded root contrast")
                if interval[0] != interval[1]:
                    missing += 1
                vector.append(interval[0])
            values.append(vector)
        groups.append(np.asarray(values, dtype=float))
    metadata = dict(readout=str(path.resolve()), configurations=configs,
        planned_roots=200, recorded_roots=len(rows), uncertain_config_root_contrasts=missing,
        fixed_denominator=True, readout_identification_bounds={
            name: data["fixed_seed_weighted_mean_bounds"]
            for name, data in readout.get("leaves", readout.get("beliefs", {})).items()})
    return metadata, groups if not missing and len(rows) == 200 else None


def positive_semidefinite(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0)) @ vectors.T


def variance_components(groups):
    counts = np.asarray([len(g) for g in groups])
    if len(groups) != 32 or np.any(counts < 2):
        raise ValueError("32 groups with repeated roots required")
    means = np.stack([g.mean(axis=0) for g in groups])
    within = sum((g - g.mean(axis=0)).T @ (g - g.mean(axis=0)) for g in groups)
    within /= counts.sum() - len(groups)
    observed_between = np.atleast_2d(np.cov(means, rowvar=False, ddof=1))
    between_raw = observed_between - within * np.mean(1 / counts)
    between = positive_semidefinite(between_raw)
    return dict(seed_root_counts=counts.tolist(), mean=means.mean(axis=0).tolist(),
        pooled_within_seed_covariance=within.tolist(),
        observed_seed_mean_covariance=observed_between.tolist(),
        between_seed_covariance_mom_unprojected=between_raw.tolist(),
        between_seed_covariance_mom_psd=between.tolist(),
        equal_seed_mean_sampling_covariance=(observed_between / 32).tolist(),
        decomposition_assumption="independent random intercept plus homoscedastic root residual; negative between eigenvalues projected to zero",
        continuation_noise="already included in measured root variance; eight continuations are not eight independent roots")


def bootstrap_bank(namespace, draws=10000):
    rng = random.Random(kernel.ScreeningDesign(namespace, tuple(range(32))).random_seed)
    bank = np.zeros((32, draws))
    for i in range(draws):
        bank[:, i] = np.bincount(rng.choices(range(32), k=32), minlength=32) / 32
    return bank


def simulate(components, configs, *, replicates=2000, draws=10000, seed=2026101501):
    """Independent synthetic exploration/holdout; shared bootstrap indices.

    Selection is over the measured PUBLIC configurations only, within arm.
    This cannot certify unmeasured reference arms, ceilings or full A1/A3.
    """
    rng = np.random.default_rng(seed)
    counts = np.asarray(components["seed_root_counts"])
    within = np.asarray(components["pooled_within_seed_covariance"])
    between = np.asarray(components["between_seed_covariance_mom_psd"])
    q = len(configs)
    arms = sorted({c["arm"] for c in configs})
    bank = bootstrap_bank("b641e390-7b8e-4294-b8c3-49af3d386ba4", draws)
    order = min(draws - 1, int(.05 * draws))
    results = []
    for delta in (0., .05, .10, .15, .20):
        hits = {arm: 0 for arm in arms}
        any_hits, realized = 0, np.zeros(q)
        for _ in range(replicates):
            panels = []
            for _panel in range(2):
                intercepts = rng.multivariate_normal(np.zeros(q), between, size=32)
                means = []
                for i, count in enumerate(counts):
                    residuals = rng.multivariate_normal(np.zeros(q), within, size=int(count))
                    roots = np.clip(delta + intercepts[i] + residuals, -1., 1.)
                    means.append(roots.mean(axis=0))
                panels.append(np.stack(means))
            exploration, holdout = panels
            realized += holdout.mean(axis=0)
            selected = [max((i for i, c in enumerate(configs) if c["arm"] == arm),
                key=lambda i: (exploration[:, i].mean(), -i)) for arm in arms]
            # Shared resampling bank preserves measured cross-config covariance.
            endpoints = np.partition(holdout[:, selected].T @ bank, order, axis=1)[:, order]
            flags = endpoints > 0
            any_hits += bool(np.any(flags))
            for arm, flag in zip(arms, flags):
                hits[arm] += bool(flag)
        results.append(dict(nominal_shift=delta, realized_clipped_mean=(realized / replicates).tolist(),
            metric="false_positive_rate" if delta == 0 else "detection_power",
            per_arm={arm: dict(count=hits[arm], replicates=replicates, rate=hits[arm] / replicates,
                mc_standard_error=float(np.sqrt(hits[arm] / replicates * (1 - hits[arm] / replicates) / replicates)))
                for arm in arms}, any_arm_rate=any_hits / replicates))
    return dict(results=results, seed_clusters=32, bootstrap_draws=draws,
        alpha_per_arm=.05, bonferroni=False, independent_exploration_and_holdout=True,
        measured_arms=arms, both_plan_arms_measured=set(arms) == set(kernel.ARMS),
        analytical_extension="clipped Gaussian random-effects family fitted to measured root covariance; target shifts are not exact effects after clipping",
        scope="measured public configurations only, not full staged A1/A2/A3 or distribution-free screening validity",
        holdout_opened=False, confirmatory_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("at least 100 synthetic replicates required")
    metadata, groups = load_panel(args.readout)
    result = dict(metadata, status="INSUFFICIENT_IDENTIFIED_PANEL", variances=None, calibration=None)
    if groups is not None:
        result["status"] = "MEASURED_VARIANCE_SENSITIVITY"
        result["variances"] = variance_components(groups)
        result["calibration"] = simulate(result["variances"], metadata["configurations"], replicates=args.replicates)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
