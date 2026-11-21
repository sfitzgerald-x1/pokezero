"""Offline transfer scoring from raw terminal witnesses and the run manifest."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .s0_transfer_roster import FP, H2H, SCHEMA, cell_key, digest, expected_cells, read_terminals, write_json


CONTRASTS = {
    "D": (FP, (("showdown", "T", 1), ("showdown", "B", -1))),
    "G": (H2H, (("engine", "T", 1), ("showdown", "T", -1))),
    "J": (H2H, (("engine", "T", 1), ("showdown", "T", -1),
                 ("engine", "C", -1), ("showdown", "C", 1))),
    "Q_FP": (FP, (("showdown", "C", 1), ("showdown", "B", -1))),
    "Q_H2H": (H2H, (("engine", "C", 1), ("showdown", "C", -1))),
}
ANCHOR = (FP, (("showdown", "T", 1), ("showdown", "H", -1)))
UNKNOWN = {"NOT_ATTEMPTED", "CAPPED", "REFUSED", "FAILED"}


def terminal_score(row):
    """Attribute the simulator terminal to the evaluated actor's physical seat."""
    evidence = row["evidence"]
    if row["status"] in UNKNOWN:
        if not evidence.get("reason"):
            raise ValueError("unknown cell requires a reason")
        return None
    if row["status"] != "COMPLETE":
        raise ValueError("unsupported disposition")
    if row["environment"] == "engine":
        winner = evidence["winner"]
        if evidence["status"] != "COMPLETE" or type(winner) is not int or winner not in (0, 1, 2):
            raise ValueError("invalid engine terminal")
        return .5 if winner == 2 else float(winner == row["seat"])
    marker, names = evidence["marker"], evidence["players"]
    if (not isinstance(names, list) or len(names) != 2
            or any(not isinstance(name, str) or not name for name in names) or names[0] == names[1]):
        raise ValueError("two distinct simulator player identities required")
    if marker in ("|tie", "|tie|"):
        return .5
    if marker.startswith("|win|") and marker[5:] in names:
        return float(names.index(marker[5:]) == row["seat"])
    raise ValueError("invalid literal Showdown terminal")


def collect_scores(manifest, rows):
    if manifest["schema"] != SCHEMA or type(manifest["dry_run"]) is not bool:
        raise ValueError("explicit transfer schema and run class required")
    expected = expected_cells(manifest["banks"], dry_run=manifest["dry_run"])
    scores = {}
    for row in rows:
        key = cell_key(row)
        spec = expected.get(key)
        if key in scores or spec is None or any(row[k] != v for k, v in spec.items()):
            raise ValueError("duplicate, extra or changed roster cell")
        if row["status"] == "COMPLETE":
            start = manifest["starts"][str(row["cluster"])]
            field = "engine_birth" if row["environment"] == "engine" else "showdown_birth"
            if row["evidence"].get("birth") != start[field]:
                raise ValueError("completed cell lacks the declared physical start")
            if row["assay"] == FP:
                seeds = row["evidence"]["seed"], row["evidence"]["fp_seed"]
                if seeds != (row["chance"], row["action"]):
                    raise ValueError("Foul Play ran with different seeds")
        scores[key] = terminal_score(row)
    return {key: scores.get(key) for key in expected}


def contrast_replica(scores, replica, bank, terms):
    values = np.asarray([
        [[scores[replica, bank[0], cluster, env, role, seat]
          for env, role, _ in terms] for seat in (0, 1)]
        for cluster in bank[1]
    ], dtype=np.float64)
    signs = np.asarray([sign for _, _, sign in terms], dtype=np.float64)
    known = np.isfinite(values)
    pairs = known.all(axis=2)
    complete = pairs.all(axis=1)
    contributions = values * signs
    lo = np.where(known, contributions, np.minimum(signs, 0)).sum(axis=2).mean(axis=1)
    hi = np.where(known, contributions, np.maximum(signs, 0)).sum(axis=2).mean(axis=1)
    count = len(values)
    result = dict(
        known_seat_tuples=int(pairs.sum()), seat_tuples=2 * count,
        known_clusters=int(complete.sum()), clusters=count,
        identification_bounds_pp=[100 * float(lo.mean()), 100 * float(hi.mean())],
        complete_only_pp=100 * float(lo[complete].mean()) if complete.any() else None,
        complete_cluster_variance_pp2=10_000 * float(lo[complete].var(ddof=1)) if complete.sum() > 1 else None,
    )
    return lo, hi, lo[complete], result


def bootstrap(cluster_bounds, *, draws, seed, quantile):
    """A fixed PCG64 index stream per replica, shared across contrasts."""
    index_generators = [np.random.Generator(np.random.PCG64(s))
                        for s in np.random.SeedSequence(seed).spawn(len(cluster_bounds))]
    sample = np.zeros((draws, 2), dtype=np.float64)
    for bounds, rng in zip(cluster_bounds, index_generators):
        values = np.stack(bounds, axis=-1)
        for start in range(0, draws, 256):
            stop = min(start + 256, draws)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)), dtype=np.int64)
            sample[start:stop] += values[indices].mean(axis=1)
    sample *= 100 / len(cluster_bounds)
    return np.quantile(sample, [quantile, 1 - quantile], axis=0, method="linear")


def rejection(name, endpoints):
    (lo_min, lo_max), (hi_min, hi_max) = endpoints
    boundary = 3 if name in ("G", "J") else 2
    robust = hi_max < -boundary or (name != "D" and lo_min > boundary)
    possible = hi_min < -boundary or (name != "D" and lo_max > boundary)
    return bool(robust), bool(possible)


def recompute(manifest, rows):
    rows = list(rows)
    scores = collect_scores(manifest, rows)
    dispositions = Counter(row["status"] for row in rows)
    if missing := len(scores) - len(rows):
        dispositions["NOT_ATTEMPTED"] += missing
    matches = {}
    for (_, assay, _, environment, role, _), score in scores.items():
        counts = matches.setdefault(f"{assay}/{environment}/{role}", dict(wins=0, ties=0, losses=0, unknown=0))
        outcome = {1.: "wins", .5: "ties", 0.: "losses", None: "unknown"}[score]
        counts[outcome] += 1
    banks = manifest["banks"]
    dry_run = manifest["dry_run"]
    draws = 2048 if dry_run else 1_000_000
    q = .05 / (2 * 90)
    if not dry_run and (len(banks) != 2 or any(len(bank) != 5000 for b in banks.values() for bank in b.values())):
        raise ValueError("registered route requires two replicas and 5,000 clusters per bank")
    definitions = CONTRASTS if dry_run else {**CONTRASTS, "T_H": ANCHOR}
    results = {}
    complete_floor = True
    for name, (assay, terms) in definitions.items():
        values = [contrast_replica(scores, r, (assay, bank[assay]), terms) for r, bank in sorted(banks.items())]
        by_replica = dict(zip(sorted(banks), (report for _, _, _, report in values)))
        complete_floor &= all(
            report["known_seat_tuples"] * 10 >= report["seat_tuples"] * 9
            and report["known_clusters"] * 10 >= report["clusters"] * 9
            for report in by_replica.values()
        )
        intervals = bootstrap([(lo, hi) for lo, hi, _, _ in values], draws=draws,
                              seed=manifest["bootstrap_seed"], quantile=q)
        results[name] = dict(
            by_replica=by_replica,
            identification_bounds_pp=np.mean([v["identification_bounds_pp"] for v in by_replica.values()],
                                             axis=0).tolist(),
            confidence_envelope_pp=[float(intervals[0, 0]), float(intervals[1, 1])],
            endpoint_bounds_pp=intervals.tolist(),
        )
        if name == "G":
            completed = [known for _, _, known, _ in values]
            report_draws = 2048 if dry_run else 20_000
            envelope = bootstrap([(lo, hi) for lo, hi, _, _ in values], draws=report_draws,
                                 seed=manifest["bootstrap_seed"], quantile=.05)
            known_interval = known_estimate = None
            if all(len(known) for known in completed):
                known_ci = bootstrap([(known, known) for known in completed], draws=report_draws,
                                     seed=manifest["bootstrap_seed"], quantile=.05)
                known_interval = known_ci[:, 0].tolist()
                known_estimate = 100 * float(np.mean([known.mean() for known in completed]))
            worst_case_ci = [float(envelope[0, 0]), float(envelope[1, 1])]
            results[name].update(
                gap_pp=known_estimate, paired_ci90_pp=known_interval,
                missingness_bounded_ci90_pp=worst_case_ci, threshold_pp=3,
                bounded_ci90_within_threshold=-3 <= worst_case_ci[0] <= worst_case_ci[1] <= 3,
            )
        if name != "T_H" and not dry_run:
            robust, possible = rejection(name, intervals)
            results[name].update(robust_rejection=robust, possible_rejection=possible)
    verdict = None
    if not dry_run:
        if not complete_floor:
            verdict = "VOID"
        elif any(results[name]["robust_rejection"] for name in CONTRASTS):
            verdict = "FAIL"
        elif any(results[name]["possible_rejection"] for name in CONTRASTS):
            verdict = "VOID"
        else:
            verdict = "NOT-FAILED"
    return dict(
        schema="pokezero.s0-transfer-scores.r2-simple", dry_run=dry_run,
        status="UNSCORED" if dry_run else "OFFLINE_RECOMPUTE",
        verdict=verdict, primitive_cells=len(scores), complete_cells=sum(v is not None for v in scores.values()),
        missing_cells=sum(v is None for v in scores.values()), dispositions=dict(dispositions),
        matches=matches, known_floor=complete_floor, contrasts=results,
        resampler=dict(algorithm="numpy.PCG64/SeedSequence/integers-int64", numpy=np.__version__,
                       seed=manifest["bootstrap_seed"], draws=draws, quantile=q, interpolation="linear"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest_sha256 = digest(manifest)
    rows = [row for side in ("engine", "showdown")
            for row in read_terminals(root / f"{side}.jsonl", manifest_sha256=manifest_sha256)]
    write_json(root / "scores.json", recompute(manifest, rows))


if __name__ == "__main__":
    main()
