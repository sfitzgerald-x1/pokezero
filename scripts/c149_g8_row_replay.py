#!/usr/bin/env python
"""Replay ONE retained sweep row through the CURRENT build and record the arm set.

Written for ledger G8 / C149: the per-roll split of the residual-kill arm changes
the BRANCH SET, and a verdict-level sweep cannot see that. This records the three
things the C149 prediction pins at row level -- verdict, miss list, branch count --
plus the two the sweep cannot report at all:

  * the branch MASS SUM, which is the conservation check on a change that replaces
    one arm of mass ``n/16`` with ``n`` arms of mass ``1/16``;
  * the census of p2 ``heal`` magnitudes the engine can emit on the boundary, which
    is the quantity c140 section 2 measured as ``{29: 93.9062 %}`` on the collapsed
    path and which the split is supposed to widen.

Nothing here reimplements the comparator: the verdict comes from
``cert_sweep_reread.reread_row``, which calls the shipped
``evaluate_boundary_strict``, and the arm set comes from the shipped
``pokezero_search.branch_events``.

Usage::

    PYTHONPATH=src python scripts/c149_g8_row_replay.py \\
        --sweep reports/artifacts/c149_base_dev_sweep.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _party_display(state_str: str) -> dict[str, list[str]]:
    from cert_sweep_reread import _party_display as shared

    return shared(state_str)


def replay(row: Mapping[str, Any]) -> dict[str, Any]:
    import poke_engine
    import pokezero_search

    from cert_sweep_reread import reread_row
    from engine_transition_differential import damage_component_events

    verdict, misses, branch_count = reread_row(row)

    # The arm set, from the same renderer the differential uses. `slot_sides` is
    # the identity mapping for a retained row, exactly as `reread_row` assumes.
    party = _party_display(row["engine_states"][0])
    ctx = json.dumps({"p1": list(party["p1"]), "p2": list(party["p2"]), "turn": 0})
    pre_hp = {
        "p1": int(row["pre_features"]["p1_hp"]),
        "p2": int(row["pre_features"]["p2_hp"]),
    }
    mass = 0.0
    branches = 0
    heal_mass: Counter = Counter()
    for state_str in row["engine_states"]:
        state = poke_engine.State.from_string(state_str)
        rendered = json.loads(
            pokezero_search.branch_events(
                state.to_string(), row["choices"]["p1"], row["choices"]["p2"], ctx, True, True
            )
        )
        for branch in rendered.get("branches") or []:
            branches += 1
            percentage = float(branch.get("percentage") or 0.0)
            mass += percentage
            # Components through the SHIPPED extractor, not a local parse of the
            # protocol -- the bare silent mirror is exactly the component whose
            # bucketing c140 section 5 is about, and re-deriving it here would be
            # re-deriving the thing under test.
            components = damage_component_events(list(branch.get("events") or []), pre_hp)
            for component in components["p2"]:
                if component.source == "heal":
                    heal_mass[abs(int(component.delta))] += percentage

    return {
        "seed": row["seed"],
        "step": row["step"],
        "verdict": verdict,
        "misses": misses,
        "branch_count": branch_count,
        "rendered_branches": branches,
        "mass_sum_percent": round(mass, 6),
        "p2_heal_magnitude_mass_percent": {
            str(k): round(v, 4) for k, v in sorted(heal_mass.items())
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="sweep artifact holding the retained repro")
    parser.add_argument("--index", type=int, default=0, help="which retained repro to replay")
    parser.add_argument("--json", help="write the result here")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    repros = report.get("repros") or []
    if not repros:
        print(f"error: {args.sweep} retains no repros", file=sys.stderr)
        return 2

    from engine_build_fingerprint import compute_fingerprint

    result = replay(repros[args.index])
    result["provenance"] = {
        "source_sweep": args.sweep,
        "engine_fingerprint": compute_fingerprint()["fingerprint"],
    }
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
