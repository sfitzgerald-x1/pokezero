#!/usr/bin/env python3
"""V2 gate — the ENGINE-anchored arm of the ``EXPECTED_{DEF,SPA,SPD,SPE}`` differential.

``tests/test_expected_stat_differential.py`` proves the encoder agrees with the generator core
(``randbats_spread_details``) over the whole pool and the whole level axis. That is a
Python-vs-Python comparison, and the plan's first standard says such a differential proves
nothing on its own -- both sides sharing the bug is exactly how C1 survived. This script closes
the chain by anchoring the core to the ENGINE:

1. plays omniscient controlled games on the local Showdown BattleStream (both seats
   uniform-random-legal with a move bias), reusing ``scripts/tier2_gate.py``'s plumbing;
2. reads each seat's own opening ``|request|``, whose ``stats`` are SERVER-COMPUTED -- the same
   channel ``scripts/investment_gate.py`` step 4 already cross-checks the core against;
3. for every mon, compares three things against that engine truth:
   - **core**: ``randbats_spread_details`` for the mon's true set;
   - **pinned encoder**: what the encoder emits with candidates pinned to the true variant --
     this must be EXACT, and is the assertion V2's exit criterion rests on;
   - **bound soundness**: with the FULL candidate set for the species, the emitted value must be
     a real upper bound on the engine's value (the column is a max over candidates, so an engine
     value ABOVE it would mean the bound is a lie, not merely loose).

Exit criterion (plan §1, V2): **zero mismatches** on the core and pinned-encoder arms, and zero
bound violations. Completion evidence is this script's own summary line with counts (plan §3,
"completion evidence over file existence"), not the presence of the output file.

Usage:
    uv run python scripts/expected_stat_gate.py --games 200 --seed 11 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/expected-stat-gate-2026-08-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.belief import RevealedPokemonBelief  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.gen3_damage import randbats_spread_details  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import (  # noqa: E402
    canonical_gen3_randbat_species_id,
    load_gen3_randbat_source_cached,
)
from pokezero.showdown import (  # noqa: E402
    NUMERIC_EXPECTED_DEF,
    NUMERIC_EXPECTED_SPA,
    NUMERIC_EXPECTED_SPD,
    NUMERIC_EXPECTED_SPE,
    _ACTUAL_STAT_DIVISOR,
    _encode_expected_stats,
    _V4_NUMERIC_FEATURE_COUNT,
)
from pokezero.tier2 import variant_has_physical_attack  # noqa: E402
from tier2_gate import _first_requests, _play_game, _team_truth  # noqa: E402

COLUMNS: tuple[tuple[str, int], ...] = (
    ("def", NUMERIC_EXPECTED_DEF),
    ("spa", NUMERIC_EXPECTED_SPA),
    ("spd", NUMERIC_EXPECTED_SPD),
    ("spe", NUMERIC_EXPECTED_SPE),
)


def _emitted_stat(num_row: list[float], slot: int) -> int:
    """The integer stat the column encodes, undoing the /714 scale.

    Values at or above the divisor clamp to 1.0 and are reported as the clamp so a caller can
    exclude them rather than read the clamp as a mismatch. No gen3 randbats Def/SpA/SpD/Spe
    reaches 714, so this is a guard, not a live path.
    """
    return round(num_row[slot] * _ACTUAL_STAT_DIVISOR)


def _encode_columns(
    dex, *, species: str, level: int, variants: tuple[Mapping[str, Any], ...]
) -> dict[str, int]:
    num_row = [0.0] * _V4_NUMERIC_FEATURE_COUNT
    belief = RevealedPokemonBelief(
        showdown_slot="p2a", species=species, candidate_variants=variants
    )
    _encode_expected_stats(
        num_row,
        dex,
        base_species=species,
        battle_species=species,
        details=f"{species}, L{level}",
        belief=belief,
        exact_spreads=True,
    )
    return {stat: _emitted_stat(num_row, slot) for stat, slot in COLUMNS}


def _true_variant_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate-variant shape the encoder consumes, built from request-side truth."""
    return {"moves": list(row.get("moves") or ()), "item": row.get("item")}


def run_gate(
    *,
    showdown_root: Path,
    games: int = 200,
    seed: int = 11,
    max_steps: int = 400,
    move_bias: float = 0.75,
) -> dict[str, Any]:
    """Run the engine-anchored sweep and return its summary dict.

    Split out from ``main`` so ``tests/test_expected_stat_differential.py`` can run a short sweep
    in CI. Without that the ENGINE arm lived only in this script, leaving the always-on coverage
    to the encoder-vs-core comparison -- which shares an implementation with the thing under test
    and so cannot, alone, satisfy the plan's "compare against the engine or the sim" standard.
    """

    class _Args:
        pass

    args = _Args()
    args.showdown_root = showdown_root
    args.games = games
    args.seed = seed
    args.max_steps = max_steps
    args.move_bias = move_bias

    dex = load_showdown_dex_cached(args.showdown_root)
    source = load_gen3_randbat_source_cached(args.showdown_root)
    rng = random.Random(args.seed)
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=str(args.showdown_root)))

    core_mismatches: list[dict[str, Any]] = []
    pinned_mismatches: list[dict[str, Any]] = []
    bound_violations: list[dict[str, Any]] = []
    counts = Counter()
    species_seen: set[str] = set()

    try:
        for game in range(args.games):
            lines = _play_game(
                env,
                seed=args.seed + game,
                rng=rng,
                max_steps=args.max_steps,
                move_bias=args.move_bias,
            )
            first = _first_requests(lines)
            if "p1" not in first or "p2" not in first:
                counts["games_without_both_requests"] += 1
                continue
            counts["games"] += 1
            for slot in ("p1", "p2"):
                for species_key, row in _team_truth(first[slot]).items():
                    engine_stats = row.get("stats") or {}
                    if not engine_stats:
                        counts["mons_without_engine_stats"] += 1
                        continue
                    canonical = canonical_gen3_randbat_species_id(species_key) or species_key
                    info = dex.species_info(canonical)
                    if info is None:
                        counts["mons_without_dex_entry"] += 1
                        continue
                    level = int(row.get("level") or 100)
                    counts["mons"] += 1
                    species_seen.add(canonical)

                    variant = _true_variant_payload(row)
                    has_physical = variant_has_physical_attack(variant.get("moves") or (), dex)

                    # (a) core vs engine -- the anchor. If this fails the core has drifted from
                    # the generator and every Python-side assertion built on it is void.
                    core = randbats_spread_details(
                        info.base_stats,
                        level=level,
                        moves=variant.get("moves") or (),
                        item=variant.get("item"),
                        has_physical_attack=has_physical,
                    ).stats
                    for stat, _slot in COLUMNS:
                        truth = engine_stats.get(stat)
                        if truth is None:
                            continue
                        counts["core_comparisons"] += 1
                        if int(core[stat]) != int(truth):
                            core_mismatches.append(
                                {
                                    "species": canonical,
                                    "level": level,
                                    "stat": stat,
                                    "core": int(core[stat]),
                                    "engine": int(truth),
                                    "moves": variant.get("moves"),
                                }
                            )

                    # (b) pinned encoder vs engine -- V2's exit criterion.
                    pinned = _encode_columns(
                        dex, species=canonical, level=level, variants=(variant,)
                    )
                    for stat, _slot in COLUMNS:
                        truth = engine_stats.get(stat)
                        if truth is None or not info.base_stats.get(stat):
                            continue
                        counts["pinned_comparisons"] += 1
                        if pinned[stat] != int(truth):
                            pinned_mismatches.append(
                                {
                                    "species": canonical,
                                    "level": level,
                                    "stat": stat,
                                    "encoder": pinned[stat],
                                    "engine": int(truth),
                                    "moves": variant.get("moves"),
                                }
                            )

                    # (c) bound soundness with the FULL candidate set for the species. The
                    # column is a max over candidates, so the engine's true value must never
                    # exceed it; a loose bound is acceptable, a violated one is a wrong belief.
                    universe = source.universe_for(canonical)
                    candidates = tuple(
                        {"moves": list(entry.moves), "item": entry.item}
                        for entry in (universe.variants if universe is not None else ())
                    )
                    if not candidates:
                        counts["mons_without_candidates"] += 1
                        continue
                    counts["mons_with_candidates"] += 1
                    bounded = _encode_columns(
                        dex, species=canonical, level=level, variants=candidates
                    )
                    for stat, _slot in COLUMNS:
                        truth = engine_stats.get(stat)
                        if truth is None or not info.base_stats.get(stat):
                            continue
                        counts["bound_comparisons"] += 1
                        if int(truth) > bounded[stat]:
                            bound_violations.append(
                                {
                                    "species": canonical,
                                    "level": level,
                                    "stat": stat,
                                    "bound": bounded[stat],
                                    "engine": int(truth),
                                }
                            )
    finally:
        env.close()

    passed = not core_mismatches and not pinned_mismatches and not bound_violations
    # Reachability: a run that compared nothing must not report PASS. This is the vacuous-pass
    # guard the plan requires on every harness assertion.
    reached = (
        counts["core_comparisons"] > 0
        and counts["pinned_comparisons"] > 0
        and counts["bound_comparisons"] > 0
    )
    verdict = "PASS" if (passed and reached) else "FAIL"

    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "reached_comparisons": reached,
        "args": {
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "move_bias": args.move_bias,
            "showdown_root": str(args.showdown_root),
        },
        "counts": dict(counts),
        "distinct_species": len(species_seen),
        "core_mismatches": core_mismatches[:50],
        "core_mismatch_count": len(core_mismatches),
        "pinned_mismatches": pinned_mismatches[:50],
        "pinned_mismatch_count": len(pinned_mismatches),
        "bound_violations": bound_violations[:50],
        "bound_violation_count": len(bound_violations),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--move-bias", type=float, default=0.75)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = run_gate(
        showdown_root=args.showdown_root,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        move_bias=args.move_bias,
    )
    counts = summary["counts"]
    verdict = summary["verdict"]
    reached = summary["reached_comparisons"]
    core_mismatches = summary["core_mismatches"]
    pinned_mismatches = summary["pinned_mismatches"]
    bound_violations = summary["bound_violations"]
    species_seen = summary["distinct_species"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expected-stat-gate.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"[expected-stat-gate] {verdict} games={counts.get('games', 0)} mons={counts.get('mons', 0)} "
        f"species={species_seen} "
        f"core={counts.get('core_comparisons', 0)}/{summary['core_mismatch_count']}-mismatched "
        f"pinned={counts.get('pinned_comparisons', 0)}/{summary['pinned_mismatch_count']}-mismatched "
        f"bounds={counts.get('bound_comparisons', 0)}/{summary['bound_violation_count']}-violated "
        f"reached={reached}"
    )
    for label, rows in (
        ("core", core_mismatches),
        ("pinned", pinned_mismatches),
        ("bound", bound_violations),
    ):
        for row in rows[:10]:
            print(f"  [{label}] {row}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
