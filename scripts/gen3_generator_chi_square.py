#!/usr/bin/env python3
"""Conditional goodness-of-fit gate for real randomTeam sets.

Expected counts are conditioned on the *actual generator call context*, not
equal weighting of contexts or an estimated species-frequency distribution.
Rare cells are pooled deterministically using expected counts only. All
generated samples must be on both the exact support and the variant list.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile


def pooled_cells(expected, observed, minimum=5.0):
    cells = []
    pending_e, pending_o = 0.0, 0
    for key in sorted(expected):
        pending_e += expected[key]
        pending_o += observed.get(key, 0)
        if pending_e >= minimum:
            cells.append([pending_e, pending_o])
            pending_e, pending_o = 0.0, 0
    if pending_e:
        if cells:
            cells[-1][0] += pending_e
            cells[-1][1] += pending_o
        else:
            cells.append([pending_e, pending_o])
    return cells


def validate_enumeration(enumeration, manifest):
    """Use the compiler's strict publication gate, including source-list coverage."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "enumeration.json").write_text(json.dumps(enumeration, ensure_ascii=False))
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        code = """
            const fs=require('fs'), e=require(process.argv[1]);
            const enumeration=JSON.parse(fs.readFileSync(process.argv[2]));
            const manifest=JSON.parse(fs.readFileSync(process.argv[3]));
            console.log(e.mapToVariants(enumeration,manifest).sha256);
        """
        result = subprocess.run(
            ["node", "-e", code, str(Path(__file__).with_name("gen3_generator_weights.cjs")),
             str(root / "enumeration.json"), str(root / "manifest.json")],
            text=True, capture_output=True, timeout=60,
        )
        if result.returncode:
            raise ValueError("enumeration/variant validation failed: " + result.stderr[-2000:])
        return result.stdout.strip()


def evaluate(enumeration, samples, manifest, minimum_sets=1_000_000):
    from scipy.stats import chi2

    mapped_table_hash = validate_enumeration(enumeration, manifest)
    counts = {}
    seeds = set()
    total = 0
    for sample in samples:
        if sample.get("schema") != "gen3-generator-sample-v1":
            raise ValueError("sample schema mismatch")
        if sample["showdown_commit"] != enumeration["showdown_commit"]:
            raise ValueError("Showdown pin mismatch")
        if sample.get("executed_sha256") != enumeration["executed_sha256"]:
            raise ValueError("executed generator mismatch")
        if sample["seed"] in seeds:
            raise ValueError("duplicate sample seed")
        seeds.add(sample["seed"])
        shard_total = 0
        for species, rows in sample["counts"].items():
            target = counts.setdefault(species, {})
            for context, histogram in rows.items():
                target.setdefault(context, Counter()).update(histogram)
                if any(type(n) is not int or n <= 0 for n in histogram.values()):
                    raise ValueError("invalid sampled count")
                shard_total += sum(histogram.values())
        if shard_total != sample["sets"]:
            raise ValueError("sample count mismatch")
        total += shard_total
    if total < minimum_sets:
        raise ValueError(f"need >= {minimum_sets} sets; found {total}")
    rows = []
    for species, contexts in sorted(counts.items()):
        for context, observed in sorted(contexts.items()):
            exact = enumeration["species"][species][context]
            if set(observed) - set(exact["weights"]):
                raise ValueError(f"sample outside exact support: {species}/{context}")
            n = sum(observed.values())
            expected = {key: n * int(weight) / int(exact["denominator"]) for key, weight in exact["weights"].items()}
            cells = pooled_cells(expected, observed)
            df = len(cells) - 1
            statistic = sum((o - e) ** 2 / e for e, o in cells)
            rows.append(dict(species=species, context=int(context), sets=n, cells=len(cells), df=df,
                             chi_square=statistic, p=float(chi2.sf(statistic, df)) if df else 1.0))
    degrees = sum(row["df"] for row in rows)
    statistic = sum(row["chi_square"] for row in rows)
    global_p = float(chi2.sf(statistic, degrees)) if degrees else 1.0
    tested = sum(row["df"] > 0 for row in rows)
    threshold = 0.001 / max(tested, 1)  # Familywise alpha, fixed before observing samples.
    failures = [row for row in rows if row["df"] and row["p"] < threshold]
    return dict(passed=global_p >= 0.001 and not failures, sets=total, species=len(counts),
                enumeration_sha256=enumeration["sha256"], mapped_table_sha256=mapped_table_hash,
                contexts=len(rows), tested_contexts=tested, degrees_of_freedom=degrees,
                chi_square=statistic, p=global_p, global_alpha=0.001, per_context_familywise_alpha=0.001,
                per_context_threshold=threshold, failed_contexts=failures,
                min_context_p=min((row["p"] for row in rows if row["df"]), default=1.0),
                coverage_missing=0, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumeration", required=True, type=Path)
    parser.add_argument("--samples", nargs="+", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.enumeration.read_text()), [json.loads(p.read_text()) for p in args.samples],
                      json.loads(args.variants.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
