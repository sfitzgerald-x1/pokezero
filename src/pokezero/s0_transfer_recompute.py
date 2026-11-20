"""Offline transfer scoring from primitive terminal witnesses and a sealed roster."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .s0_transfer_roster import FP, H2H, SCHEMA, actor_bindings, cell_key, digest, expected_cells, write_once


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
    """Decode the actual producer witness without using its score or aggregate."""
    evidence = row["evidence"]
    if row["status"] in UNKNOWN:
        if row["score"] is not None or not evidence.get("reason"):
            raise ValueError("unknown cell requires a reason and no score")
        return None
    if row["status"] != "COMPLETE":
        raise ValueError("unsupported disposition")
    if row["environment"] == "engine":
        if evidence["status"] != "COMPLETE" or type(evidence["winner"]) is not int:
            raise ValueError("engine terminal status and winner required")
        winner = evidence["winner"]
        if winner not in (0, 1, 2):
            raise ValueError("unexpected engine winner")
        return .5 if winner == 2 else float(winner == row["seat"])
    if row["assay"] == FP:
        original = evidence["source"]["terminal_witness"]
        if (original["capped"] or evidence["marker"] != original["public_terminal_line"]
                or evidence["players"] != [original["players"][p] for p in ("p1", "p2")]):
            raise ValueError("normalized external terminal differs from its source")
        winner = original["winner_player"]
        if original["status"] != ("PUBLIC_TIE" if winner is None else "PUBLIC_WIN"):
            raise ValueError("external witness lacks a valid terminal status")
        if winner not in (None, "p1", "p2") or evidence["winner"] != (2 if winner is None else int(winner[1]) - 1):
            raise ValueError("external terminal winner differs from its source")
    marker = evidence["marker"]
    names = evidence["players"]
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 2:
        raise ValueError("two distinct simulator player identities required")
    if marker in ("|tie", "|tie|") and evidence["winner"] == 2:
        return .5
    if marker.startswith("|win|") and marker[5:] in names:
        winner = names.index(marker[5:])
        if type(evidence["winner"]) is int and evidence["winner"] == winner:
            return float(winner == row["seat"])
    raise ValueError("literal Showdown terminal and winner disagree")


def collect_scores(manifest, rows):
    if manifest["schema"] != SCHEMA or type(manifest["dry_run"]) is not bool:
        raise ValueError("explicit r2 cell schema and run class required")
    expected = expected_cells(manifest["banks"], dry_run=manifest["dry_run"])
    if manifest["expected_cells_sha256"] != digest(list(expected.values())):
        raise ValueError("frozen roster cell identities changed")
    scores = {}
    manifest_sha = digest(manifest)
    for row in rows:
        key = cell_key(row)
        spec = expected.get(key)
        if key in scores or spec is None or any(row[k] != v for k, v in spec.items()):
            raise ValueError("duplicate, extra or changed roster cell")
        if row["manifest_sha256"] != manifest_sha:
            raise ValueError("cell belongs to another manifest")
        start = manifest["starts"][str(row["cluster"])]
        if row["start_sha256"] != digest(start):
            raise ValueError("cell roster binding changed")
        if row["actor_bindings"] != actor_bindings(manifest, row):
            raise ValueError("physical actor-role bindings differ")
        if row["evidence_sha256"] != digest(row["evidence"]):
            raise ValueError("terminal evidence changed")
        if row["status"] == "COMPLETE":
            birth_field = "engine_birth" if row["environment"] == "engine" else "showdown_birth"
            if row["evidence"].get("birth") != start[birth_field]:
                raise ValueError("completed cell lacks the actual declared start")
            if row["assay"] == FP and (row["evidence"]["seed"], row["evidence"]["fp_seed"]) != (row["chance"], row["action"]):
                raise ValueError("external opponent seed receipt differs")
        if row["score"] != (score := terminal_score(row)):
            raise ValueError("producer score does not match terminal evidence")
        scores[key] = score
    if scores.keys() != expected.keys():
        raise ValueError("missing expected cell or unknown placeholder")
    return scores


def verify_control(manifest):
    if manifest["banks"].keys() != manifest["actors"].keys() or manifest["banks"].keys() != manifest["control"].keys():
        raise ValueError("each registered replica requires its own control lifecycle")
    for replica, actors in manifest["actors"].items():
        baseline, control = actors["B"], actors["C"]
        declared = manifest["control"][replica]
        cycles = declared["shadow_cycles"]
        if (declared["dry_run"] != manifest["dry_run"] or declared["training_games"] != 0
                or baseline["checkpoint"] == control["checkpoint"] or baseline["archive"] == control["archive"]
                or baseline["tensor_sha256"] != control["tensor_sha256"]
                or baseline["model_sha256"] != control["model_sha256"]
                or baseline["self_moves"] != control["self_moves"]
                or cycles not in ((2,) if manifest["dry_run"] else (50, 100))
                or len(declared["cycle_sha256"]) != cycles):
            raise ValueError("C must retain B policy tensors on an independent no-update lifecycle")
        root = Path(control["checkpoint"]).parent.parent
        iterations = []
        prior = 0
        for i, seal in enumerate(declared["cycle_sha256"], 1):
            cycle = (root / f"{i:03d}" / "cycle.json").read_bytes()
            if hashlib.sha256(cycle).hexdigest() != seal:
                raise ValueError("control lifecycle receipt changed")
            value = json.loads(cycle)
            if (value["schema"] != "pokezero.s0-control-cycle.r2" or value["dry_run"] != manifest["dry_run"]
                    or value["training_games"] != 0 or value["cycle"] != i or value["baseline"] != baseline
                    or value["fixture_sha256"] != declared["last_fixture_sha256"]
                    or value["control"]["tensor_sha256"] != baseline["tensor_sha256"]
                    or value["control"]["model_sha256"] != baseline["model_sha256"]
                    or value["control"]["self_moves"] != baseline["self_moves"]):
                raise ValueError("control fixture, weights or lifecycle changed")
            if manifest["dry_run"]:
                if value["completed_iteration"] is not None:
                    raise ValueError("scratch lifecycle carries training progress")
            else:
                receipt = value["completed_iteration"]
                if receipt["iteration"] != i or receipt["completed_games"] <= prior:
                    raise ValueError("completed iteration receipt differs")
                prior = receipt["completed_games"]
                iterations.append(receipt)
        if digest(iterations) != declared["completed_iterations_sha256"] or value["control"] != control:
            raise ValueError("served control differs from the terminal lifecycle receipt")


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
    return lo, hi, result


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


def recompute(manifest, rows, *, control_verified=False):
    scores = collect_scores(manifest, rows)
    banks = manifest["banks"]
    dry_run = manifest["dry_run"]
    draws = 2048 if dry_run else 1_000_000
    q = .05 / (2 * 90)
    if not dry_run and not control_verified:
        raise ValueError("independently verified control lifecycle is required")
    if not dry_run and (len(banks) != 2 or any(len(bank) != 5000 for b in banks.values() for bank in b.values())):
        raise ValueError("registered route requires two replicas and 5,000 clusters per bank")
    definitions = CONTRASTS if dry_run else {**CONTRASTS, "T_H": ANCHOR}
    results = {}
    complete_floor = True
    for name, (assay, terms) in definitions.items():
        values = [contrast_replica(scores, r, (assay, bank[assay]), terms) for r, bank in sorted(banks.items())]
        by_replica = dict(zip(sorted(banks), (report for _, _, report in values)))
        complete_floor &= all(
            report["known_seat_tuples"] * 10 >= report["seat_tuples"] * 9
            and report["known_clusters"] * 10 >= report["clusters"] * 9
            for report in by_replica.values()
        )
        intervals = bootstrap([(lo, hi) for lo, hi, _ in values], draws=draws,
                              seed=manifest["bootstrap_seed"], quantile=q)
        results[name] = dict(
            by_replica=by_replica,
            identification_bounds_pp=np.mean([v["identification_bounds_pp"] for v in by_replica.values()],
                                             axis=0).tolist(),
            confidence_envelope_pp=[float(intervals[0, 0]), float(intervals[1, 1])],
            endpoint_bounds_pp=intervals.tolist(),
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
        schema="pokezero.s0-transfer-offline-recompute.r2", dry_run=dry_run,
        status="UNSCORED" if dry_run else "OFFLINE_RECOMPUTE",
        verdict=verdict, primitive_cells=len(scores), complete_cells=sum(v is not None for v in scores.values()),
        manifest_sha256=digest(manifest), known_floor=complete_floor, contrasts=results,
        resampler=dict(algorithm="numpy.PCG64/SeedSequence/integers-int64", numpy=np.__version__,
                       seed=manifest["bootstrap_seed"], draws=draws, quantile=q, interpolation="linear"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("seals", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    seals = json.loads(args.seals.read_bytes())
    manifest = json.loads(args.manifest.read_bytes())
    if seals["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError("sealed roster changed")
    rows = []
    for name, expected_hash in sorted(seals["shards"].items()):
        if Path(name).name != name:
            raise ValueError("expected local shard filename")
        raw = (args.seals.parent / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError("sealed shard changed")
        rows.extend(json.loads(raw))
    verify_control(manifest)
    write_once(args.out, recompute(manifest, rows, control_verified=True))


if __name__ == "__main__":
    main()
