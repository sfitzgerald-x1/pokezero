#!/usr/bin/env python
"""Measure the `leaf_batch > 1` fidelity loss on the arbiter arm's sequential path.

WHY THIS SCRIPT EXISTS. The refusal in ``EngineMctsConfig.__post_init__`` for
``leaf_batch > 1`` quotes a magnitude, and the two previous revisions of that
message quoted magnitudes that nobody could regenerate: first a single
six-significant-figure triple from a run whose config was never recorded
(independent review failed to reproduce it in 9792 crate runs), then a range from
a review sweep whose grid lived only in a review comment. A number in a refusal
is evidence, and evidence has to be re-derivable, so the grid is code.

WHAT IT MEASURES. The arm is certified only at ``leaf_batch=1``: batching changes
SELECTION, because a round's virtual losses are not replaced until the round's
rows are priced. So this is not a throughput knob with a rounding error attached
-- it is a different search. The observable is the reported ``root_value``, which
is what any downstream strength number is built out of, and the statistic is
``|root_value(leaf_batch=1) - root_value(leaf_batch=b)|`` in percentage points.

GRID (5 x 12 x 3 x 3 = 540 triples, 1620 crate searches; ~1 minute):
    fixtures      -- all five crate fixtures under
                     rust/pokezero-search/src/test_fixtures/
    search seed   -- 0..11
    rollout_seed  -- 0..2
    max_depth     -- 2, 3, 4
    fixed         -- iterations=400, R=8, rollout_max_plies=200, one thread,
                     rollout_branch_on_damage off, leaf_mode="rollout"
    leaf_batch    -- 1, 8, 64

Pass ``--seed-band 12`` to re-run on a disjoint band of search seeds; the summary
should be stable to a few hundredths of a pp, and that reproduction is the only
reason to trust the digits.

Usage:
    python scripts/sweep_leaf_batch_fidelity_gap.py [--seed-band 0] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

FIXTURES = (
    pathlib.Path(__file__).resolve().parent.parent
    / "rust"
    / "pokezero-search"
    / "src"
    / "test_fixtures"
)

BATCHES = (1, 8, 64)


def _summary(values: list[float]) -> dict[str, float]:
    quartiles = statistics.quantiles(values, n=4)
    return {
        "n": len(values),
        "min": min(values),
        "p25": quartiles[0],
        "median": statistics.median(values),
        "p75": quartiles[2],
        "max": max(values),
        "frac_ge_0.6pp": sum(v >= 0.6 for v in values) / len(values),
        "frac_ge_5pp": sum(v >= 5.0 for v in values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-band",
        type=int,
        default=0,
        help="first search seed; the band is [seed_band, seed_band + 12)",
    )
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args()

    try:
        import pokezero_search
    except ImportError:
        print(
            "pokezero_search is not built; run scripts/build_search_crate_model.sh",
            file=sys.stderr,
        )
        return 2
    if not hasattr(pokezero_search, "puct_search_multi_rollout"):
        print("the installed crate has no rollout seam", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for path in sorted(FIXTURES.glob("*.state")):
        state = path.read_text().strip()
        for seed in range(args.seed_band, args.seed_band + 12):
            for rollout_seed in range(3):
                for depth in (2, 3, 4):
                    root_values = {}
                    for batch in BATCHES:
                        report = json.loads(
                            pokezero_search.puct_search_multi_rollout(
                                state,
                                400,
                                max_depth=depth,
                                c_puct=1.4,
                                seed=seed,
                                rollouts=8,
                                rollout_max_plies=200,
                                rollout_policy="uniform",
                                rollout_seed=rollout_seed,
                                rollout_threads=1,
                                rollout_branch_on_damage=False,
                                leaf_batch=batch,
                                leaf_mode="rollout",
                            )
                        )
                        root_values[batch] = report["root_value"]
                    rows.append(
                        {
                            "fixture": path.name,
                            "seed": seed,
                            "rollout_seed": rollout_seed,
                            "max_depth": depth,
                            "root_value": root_values,
                            "pp_gap_8": abs(root_values[1] - root_values[8]) * 100.0,
                            "pp_gap_64": abs(root_values[1] - root_values[64]) * 100.0,
                        }
                    )

    print(f"triples={len(rows)} crate_runs={len(rows) * len(BATCHES)}")
    print(f"seed band = {args.seed_band}..{args.seed_band + 11}")
    for key, label in (("pp_gap_8", "leaf_batch 1 vs 8 "), ("pp_gap_64", "leaf_batch 1 vs 64")):
        overall = _summary([row[key] for row in rows])
        print(
            f"  {label} (pp)  min {overall['min']:6.2f}  median {overall['median']:6.2f}"
            f"  max {overall['max']:6.2f}   >=0.6pp {overall['frac_ge_0.6pp']:.3f}"
            f"  >=5pp {overall['frac_ge_5pp']:.3f}"
        )
        for fixture in sorted({row["fixture"] for row in rows}):
            per = _summary([row[key] for row in rows if row["fixture"] == fixture])
            print(
                f"      {fixture:24} min {per['min']:6.2f}  median {per['median']:6.2f}"
                f"  max {per['max']:6.2f}"
            )
    # The distribution is fixture-dependent rather than a floor, which is the
    # part the refusal message has to state honestly: the smallest gap in this
    # grid sits BELOW the ~0.6 pp arm of the contrast the arbiter must resolve.
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
