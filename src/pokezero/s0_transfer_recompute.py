"""Offline transfer scoring from raw terminal witnesses and the run manifest."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .s0_transfer_roster import FP, H2H, SCHEMA, cell_key, digest, expected_cells, panel_lock, raw_paths, read_terminals, write_json


CONTRASTS = {
    "D": (FP, (("showdown", "T", 1), ("showdown", "B", -1))),
    "G": (H2H, (("engine", "T", 1), ("showdown", "T", -1))),
    "J": (H2H, (("engine", "T", 1), ("showdown", "T", -1),
                 ("engine", "C", -1), ("showdown", "C", 1))),
    "Q_FP": (FP, (("showdown", "C", 1), ("showdown", "B", -1))),
    "Q_H2H": (H2H, (("engine", "C", 1), ("showdown", "C", -1))),
}
ANCHOR = (FP, (("showdown", "T", 1), ("showdown", "H", -1)))
CLUSTERS_PER_REPLICA = 5_000
REGISTERED_DRAWS = 1_000_000
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
    if not isinstance(marker, str):
        raise ValueError("literal Showdown terminal must be text")
    if marker in ("|tie", "|tie|"):
        return .5
    if marker.startswith("|win|") and marker[5:] in names:
        return float(names.index(marker[5:]) == row["seat"])
    raise ValueError("invalid literal Showdown terminal")


def collect_scores(manifest, rows, *, counts=None):
    if manifest["schema"] != SCHEMA or type(manifest["dry_run"]) is not bool:
        raise ValueError("explicit transfer schema and run class required")
    expected = expected_cells(manifest["banks"], dry_run=manifest["dry_run"])
    scores = {}
    statuses = {}
    invalid = integrity = 0
    for row in rows:
        key = None
        try:
            key = cell_key(row)
            spec = expected.get(key)
            if spec is None:
                integrity += 1
                raise ValueError("extra roster cell")
            if key in scores:
                integrity += 1
                raise ValueError("duplicate roster cell")
            if any(type(row[k]) != type(v) or row[k] != v for k, v in spec.items()):
                raise ValueError("changed roster cell")
            if row["status"] == "COMPLETE":
                start = manifest["starts"][str(row["cluster"])]
                field = "engine_birth" if row["environment"] == "engine" else "showdown_birth"
                if row["evidence"].get("birth") != start[field]:
                    raise ValueError("completed cell lacks the declared physical start")
                if row["assay"] == FP:
                    if (row["evidence"]["seed"], row["evidence"]["fp_seed"]) != (row["chance"], row["action"]):
                        raise ValueError("Foul Play ran with different seeds")
            scores[key] = terminal_score(row)
            statuses[key] = row["status"]
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            if counts is None:
                raise ValueError(str(error)) from error
            invalid += 1
            if key is not None and all(isinstance(x, (int, str)) for x in key) and key in expected:
                scores[key] = None
                statuses[key] = "REFUSED"
    if counts is not None:
        dispositions = Counter(statuses.values())
        absent = len(expected) - len(statuses)
        if absent:
            dispositions["NOT_ATTEMPTED"] += absent
        counts.update(invalid_rows=invalid, integrity_errors=integrity, dispositions=dict(dispositions))
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


def joint_bootstrap(matrices, draws, seed, q):
    sample = np.zeros((draws, matrices[0].shape[1]))
    for matrix, child in zip(matrices, np.random.SeedSequence(seed).spawn(len(matrices))):
        rng = np.random.Generator(np.random.PCG64(child))
        n = len(matrix)
        for start in range(0, draws, 64):
            out = sample[start:start + 64]
            indices = rng.integers(n, size=(len(out), n), dtype=np.int64)
            indices += n * np.arange(len(out))[:, None]
            weights = np.bincount(indices.ravel(), minlength=len(out)*n).reshape(-1, n)
            out += (weights.astype(np.float64) @ matrix) / n
    sample *= 100 / len(matrices)
    return np.quantile(sample, [q, 1-q], axis=0, method="linear")


def rejection(name, endpoints):
    (lo_min, lo_max), (hi_min, hi_max) = endpoints
    boundary = 3 if name in ("G", "J") else 2
    robust = hi_max < -boundary or (name != "D" and lo_min > boundary)
    possible = hi_min < -boundary or (name != "D" and lo_max > boundary)
    return bool(robust), bool(possible)


def recompute(manifest, rows):
    accounting = {}
    scores = collect_scores(manifest, rows, counts=accounting)
    matches = {}
    for (_, assay, _, environment, role, _), score in scores.items():
        counts = matches.setdefault(f"{assay}/{environment}/{role}", dict(wins=0, ties=0, losses=0, unknown=0))
        outcome = {1.: "wins", .5: "ties", 0.: "losses", None: "unknown"}[score]
        counts[outcome] += 1
    banks = manifest["banks"]
    dry_run = manifest["dry_run"]
    draws = 2048 if dry_run else REGISTERED_DRAWS
    q = .05 / (2 * 90)
    if not dry_run and (len(banks) != 2 or any(len(bank) != CLUSTERS_PER_REPLICA for b in banks.values() for bank in b.values())):
        raise ValueError("registered route requires two replicas and 5,000 clusters per bank")
    definitions = CONTRASTS if dry_run else {**CONTRASTS, "T_H": ANCHOR}
    results = {}
    complete_floor = True
    inputs = {name: [contrast_replica(scores, r, (assay, bank[assay]), terms)
                     for r, bank in sorted(banks.items())]
              for name, (assay, terms) in definitions.items()}
    # Reuse the exact index stream for all five tests and the descriptive anchor.
    # A batched histogram per draw avoids repeatedly gathering billions of tuples.
    matrices = [np.column_stack([col for values in inputs.values() for col in values[r][:2]])
                for r in range(len(banks))]
    intervals_all = joint_bootstrap(matrices, draws, manifest["bootstrap_seed"], q)
    for i, (name, values) in enumerate(inputs.items()):
        by_replica = dict(zip(sorted(banks), (report for _, _, _, report in values)))
        complete_floor &= all(
            report["known_seat_tuples"] * 10 >= report["seat_tuples"] * 9
            and report["known_clusters"] * 10 >= report["clusters"] * 9
            for report in by_replica.values()
        )
        intervals = intervals_all[:, 2*i:2*i+2]
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
        if not complete_floor or accounting["integrity_errors"]:
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
        missing_cells=sum(v is None for v in scores.values()), **accounting,
        matches=matches, known_floor=complete_floor, contrasts=results,
        resampler=dict(algorithm="numpy.PCG64/SeedSequence/integers-int64", numpy=np.__version__,
                       seed=manifest["bootstrap_seed"], draws=draws, quantile=q, interpolation="linear"),
    )


def route_verdict(scores, manifest, raw_sha, primary_sha):
    """Translate independently scored, dispatch-bound rows to the trainer stop contract."""
    from .v5_deferred_evaluation import read_owned_milestone
    from .v5_evaluation import Artifact
    if manifest["dry_run"] or scores["verdict"] not in {"FAIL", "VOID", "NOT-FAILED"}:
        raise ValueError("a registered transfer look is required")
    refs = manifest["dispatches"]
    if len(refs) != 2 or refs.keys() != manifest["banks"].keys():
        raise ValueError("original dispatch required for both replicas")
    paths, digests, protocols = set(), [], set()
    route = manifest["route_declaration_sha256"]
    for replica, record in sorted(refs.items()):
        ref = Artifact(**record)
        request = read_owned_milestone(ref)
        target = manifest["actors"][replica]["T"]
        if (not route or ref.document()["route_declaration_sha256"] != route
                or request.run_directory in paths
                or request.snapshot.archive_sha256 != target["archive_sha256"]
                or request.snapshot.checkpoint_sha256 != target["checkpoint_sha256"]
                or request.declared_games != 20_000 * manifest["config"]["control_cycles"]):
            raise ValueError("scored actors or route differ from original dispatches")
        paths.add(request.run_directory)
        protocols.add(request.protocol_sha256)
        digests.append(ref.sha256)
    if len(protocols) != 1:
        raise ValueError("replica evaluation protocols differ")
    endpoints = {}
    for name in CONTRASTS:
        lo, hi = scores["contrasts"][name]["endpoint_bounds_pp"]
        endpoints[name] = dict(L_min=lo[0], U_max=hi[1])
    return dict(schema="s0.transfer.r2", route_declaration_sha256=route, verdict=scores["verdict"],
                validity="VALID" if scores["known_floor"] and not scores["integrity_errors"] else "INVALID",
                dispatch_sha256s=digests, raw_cells_sha256=raw_sha, primary_result_sha256=primary_sha,
                independent_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                completion_endpoints_pp=endpoints)


def score_directory(root, *, publish_stop=False, lock_seconds=0):
    with panel_lock(root, scoring=True, wait_seconds=lock_seconds):
        return _score_directory(root, publish_stop=publish_stop)


def _score_directory(root, *, publish_stop):
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest_sha = digest(manifest)
    files = {}
    def rows():
        for side in ("engine", "showdown"):
            for path in raw_paths(root, side, manifest_sha):
                raw = hashlib.sha256()
                yield from read_terminals(path, manifest_sha256=manifest_sha, hasher=raw)
                files[path.name] = raw.hexdigest()
    scores = recompute(manifest, rows())
    path = root / "scores.json"
    write_json(path, scores)
    if not manifest["dry_run"]:
        primary_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        verdict = route_verdict(scores, manifest, digest(files), primary_sha)
        path = root / "route-verdict.json"
        write_json(path, verdict)
        if publish_stop and verdict["verdict"] == "FAIL":
            from .v5_deferred_evaluation import publish_route_failures
            from .v5_evaluation import Artifact
            result = Artifact(str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
            refs = [Artifact(**record) for record in manifest["dispatches"].values()]
            publish_route_failures(refs, result)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--publish-stop", action="store_true", help="sign and deliver a robust FAIL to both trainers")
    parser.add_argument("--lock-seconds", type=int, default=0, help="wait this long for active workers")
    args = parser.parse_args()
    score_directory(args.directory, publish_stop=args.publish_stop, lock_seconds=args.lock_seconds)


if __name__ == "__main__":
    main()
