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


def interval_variance_bounds(intervals, weights):
    """Finite-roster population dispersion, not a future-population CI.

    Minimize squared distance to the interval boxes over the mean. The second
    moment bound gives a conservative maximum; cross-arm covariance is not
    inferred from marginal boxes. Unknown outcomes are never imputed.
    """
    boxes=np.asarray(intervals,dtype=float);w=np.asarray(weights,dtype=float)
    if boxes.shape!=(len(w),2) or not len(w) or not np.all(np.isfinite(boxes)) \
            or not np.all(np.isfinite(w)) or np.any(w<=0) or abs(w.sum()-1)>1e-12 \
            or np.any(boxes[:,0]<-1) or np.any(boxes[:,1]>1) or np.any(boxes[:,0]>boxes[:,1]):
        raise ValueError("valid bounded contrast boxes and fixed positive weights required")
    low,high=boxes.T
    if np.all(low==high):
        exact=float(w@((low-float(w@low))**2))
        return [exact,exact]
    left,right=-1.,1.
    for _ in range(80):
        middle=(left+right)/2
        if float(w@(middle-np.clip(middle,low,high)))>0:right=middle
        else:left=middle
    middle=(left+right)/2
    minimum=max(0.,float(w@((np.clip(middle,low,high)-middle)**2))-4*(right-left)-1e-12)
    mean_low,mean_high=float(w@low),float(w@high)
    mean_square_min=0. if mean_low<=0<=mean_high else min(mean_low**2,mean_high**2)
    maximum=min(1.,max(0.,float(w@np.maximum(low**2,high**2))-mean_square_min)+1e-12)
    return [minimum,maximum]


def load_panel(readout_path):
    path = Path(readout_path)
    readout = json.loads(path.read_text())
    entry = json.loads((path.parent / "entry.json").read_text())
    contract = entry["contract"]
    panel = contract["panels"]["exploration"]
    seeds, slots = panel["seeds"], panel["root_slots"]
    if len(seeds) != 32 or len(set(seeds)) != 32 or len(slots) != 200:
        raise ValueError("the fixed 32-seed/200-root panel is required")
    schema=contract.get("schema")
    fixed=(schema=="pokezero.search-over-raw.fixed-total-exploration.v1"
        or schema=="pokezero.search-over-raw.fixed-total-exploration.compact-v2")
    if fixed:
        if readout.get("schema")!=schema+".readout":
            raise ValueError("fixed-work readout and original entry schema differ")
        total=contract["accepted_native_iterations_total"]
        expected_leaves=["model","hp_fraction"] if total==320 else ["model"]
        if type(total) is not int or total not in (320,1024,4096) or contract["leaves"]!=expected_leaves \
                or contract["native_worlds"]!=4 or contract["native_iterations_per_world"]!=total//4 \
                or contract["wall_clock_search_budget"] is not None:
            raise ValueError("fixed-work configuration scope drift")
        if schema.endswith("compact-v2") and (total!=4096
                or contract.get("native_atomic_encoding")!="native-atomic-backups-compact-v2"
                or readout.get("native_atomic_encoding")!=contract["native_atomic_encoding"]):
            raise ValueError("compact fixed-work encoding scope drift")
        if schema.endswith(".v1") and ("native_atomic_encoding" in contract or "native_atomic_encoding" in readout):
            raise ValueError("compact encoding cannot enter the legacy fixed-work lane")
        terminal=json.loads((path.parent/"auditor/terminal.json").read_bytes())
        status=terminal.get("status");supervisor=terminal.get("supervisor_exit_code");worker=terminal.get("worker_exit_code")
        recognized=(status=="STAGED_EXPLORATION_RUN_COMPLETE" and supervisor==0 and worker==0
            or status=="STAGED_EXPLORATION_RUN_INCOMPLETE" and supervisor==124 and type(worker) is int
            or status=="OPERATIONAL_STOP" and supervisor==1
                and (type(worker) is int or worker is None and readout.get("recorded_roots")==0))
        if type(supervisor) is not int or not recognized:
            raise ValueError("recognized compatible closed exploration terminal required")
        for key in ("supervisor_exit_code","worker_exit_code"):
            if type(readout.get(key)) is not type(terminal.get(key)) or readout.get(key)!=terminal[key] \
                    or key=="worker_exit_code" and type(worker) not in (int,type(None)):
                raise ValueError("closed owning terminal/readout mismatch")
        if readout.get("terminal_status")!=terminal.get("status"):
            raise ValueError("closed owning terminal status mismatch")
        configs=[dict(arm="incumbent",belief="public",leaf=leaf,
            accepted_native_iterations_total=contract["accepted_native_iterations_total"],worlds=4)
            for leaf in contract["leaves"]]
    else:
        configs = [c for c in contract["configurations"]
            if c["arm"] != "raw" and c["belief"] == "public"]
    if not configs:
        raise ValueError("no measured deployable configurations")
    rejected = set() if fixed else {str(Path(row["path"]).resolve()) for row in readout["rejected_root_receipts"]}
    rows = [json.loads(p.read_text()) for p in sorted((path.parent / "completed-roots").glob("*.json"))
        if str(p.resolve()) not in rejected]
    by_id = {row["root_id"]: row for row in rows}
    roster = {row["root_id"]: row for row in slots}
    if len(roster)!=200 or {r["source_seed"] for r in slots}!=set(seeds) \
            or len(by_id) != len(rows) or not set(by_id) <= set(roster):
        raise ValueError("duplicate or foreign completed roots")
    if type(readout["recorded_roots"]) is not int or len(rows) != readout["recorded_roots"]:
        raise ValueError("completed rows differ from the final owning readout")
    groups, box_groups, missing = [], [], 0
    for seed in seeds:
        values,boxes = [],[]
        for slot in (r for r in slots if r["source_seed"] == seed):
            row = by_id.get(slot["root_id"])
            if row is not None:
                if row["source_seed"]!=seed:
                    raise ValueError("source/seed mismatch")
                source_missing=(fixed and row.get("actions")=={}
                    and row.get("input_unavailable")=={"source":"missing_fixed_slot"})
                if row.get("source_pipeline_validated") is not True and not source_missing:
                    raise ValueError("source pipeline validation absent")
            vector,vector_boxes = [],[]
            for config in configs:
                interval = [-1., 1.] if row is None else (
                    row["intervals"][config["leaf"]] if fixed else row["root_intervals"][digest(config)])
                if (type(interval) is not list or len(interval) != 2
                        or not all(type(v) in (int,float) and np.isfinite(v) for v in interval)
                        or not -1 <= interval[0] <= interval[1] <= 1):
                    raise ValueError("invalid bounded root contrast")
                if row is not None and row.get("source_pipeline_validated") is not True and interval!=[-1.,1.]:
                    raise ValueError("missing source cannot acquire an identified contrast")
                if interval[0] != interval[1]:
                    missing += 1
                vector.append(interval[0])
                vector_boxes.append(interval)
            values.append(vector)
            boxes.append(vector_boxes)
        groups.append(np.asarray(values, dtype=float))
        box_groups.append(np.asarray(boxes,dtype=float))
    identification={name:data["fixed_seed_weighted_bounds"] for name,data in readout["effects"].items()} if fixed else {
        name:data["fixed_seed_weighted_mean_bounds"]
        for name,data in readout.get("leaves",readout.get("beliefs",{})).items()}
    dispersion=[]
    for index,config in enumerate(configs):
        root_boxes=np.concatenate([g[:,index,:] for g in box_groups])
        weights=np.concatenate([np.full(len(g),1/(32*len(g))) for g in box_groups])
        seed_boxes=np.stack([g[:,index,:].mean(axis=0) for g in box_groups])
        if fixed and not np.allclose(seed_boxes.mean(axis=0),identification[config["leaf"]],atol=1e-12,rtol=0):
            raise ValueError("fixed readout bounds differ from original fixed-roster rows")
        if fixed:
            known=int(np.sum(root_boxes[:,0]==root_boxes[:,1]))
            effect=readout["effects"][config["leaf"]]
            if type(effect["identified_roots"]) is not int or type(effect["uncertain_fixed_slots"]) is not int \
                    or effect["identified_roots"]!=known or effect["uncertain_fixed_slots"]!=200-known:
                raise ValueError("fixed readout identification counts differ from scheduled rows")
        dispersion.append(dict(configuration=config,
            root_population_dispersion_bounds=interval_variance_bounds(root_boxes,weights),
            seed_mean_population_dispersion_bounds=interval_variance_bounds(seed_boxes,np.full(32,1/32))))
    metadata = dict(readout=str(path.resolve()), configurations=configs,
        planned_roots=200, recorded_roots=len(rows), uncertain_config_root_contrasts=missing,
        fixed_denominator=True,readout_identification_bounds=identification,
        finite_roster_dispersion_bounds=dispersion,
        dispersion_scope="equal-seed finite scheduled roster, not a confidence bound for future population variance; joint covariance remains unidentified with missing contrasts",
        validation_scope="closed owning readout and scheduled-root arithmetic; not a rerun of scientific provenance validators")
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
        scope="measured public configurations only; does not calibrate the revised fixed-work selection procedure, full staged A1/A2/A3 or distribution-free screening validity",
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
