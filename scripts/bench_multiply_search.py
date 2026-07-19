#!/usr/bin/env python3
"""Bench the multi-ply decision/chance search core (rust/pokezero-search).

Sweeps `pokezero_search.puct_search_multi` (HpFractionEval leaf pricing — the
model-priced path shares the identical tree core behind the LeafEval seam)
over search depths x sims-per-decision on curated positions, reporting:

- sims/s and ms/decision per (position, depth, sims) cell;
- tree growth (decision/chance nodes, leaf evals) — the branching-factor
  blowup that locates the practical depth wall;
- argmax stability: depth>1 vs depth=1 root argmax across seeds.

Usage:
    python scripts/bench_multiply_search.py --depths 1,2,3,4 --sims 256,1024 \
        --seeds 5 --min-time 1.0 --out bench_multiply.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _positions():
    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
        minimal_gen3_fixture,
    )

    def mon(species, types, moves, *, hp=300, maxhp=300, atk=250, dfn=250, spa=250, spd=250, spe=250):
        return PokemonSpec(
            id=species,
            level=100,
            types=types,
            hp=hp,
            maxhp=maxhp,
            attack=atk,
            defense=dfn,
            special_attack=spa,
            special_defense=spd,
            speed=spe,
            status="none",
            moves=tuple(MoveSpec(id=m, pp=16) for m in moves),
        )

    midgame = BattleSpec(
        side_one=SideSpec(
            pokemon=(
                mon("snorlax", ("normal",), ("bodyslam", "earthquake", "shadowball", "selfdestruct"), spe=110),
                mon("starmie", ("water", "psychic"), ("surf", "psychic", "thunderbolt", "icebeam"), hp=260, maxhp=260, spe=330),
                mon("heracross", ("bug", "fighting"), ("megahorn", "brickbreak", "rockslide", "rest"), spe=280),
            )
        ),
        side_two=SideSpec(
            pokemon=(
                mon("metagross", ("steel", "psychic"), ("meteormash", "earthquake", "explosion", "psychic"), spe=230),
                mon("salamence", ("dragon", "flying"), ("dragonclaw", "earthquake", "rockslide", "fireblast"), hp=280, maxhp=280, spe=300),
                mon("suicune", ("water",), ("surf", "icebeam", "calmmind", "rest"), hp=320, maxhp=320, spe=240),
            )
        ),
    )

    def small(species, moves, *, hp=100, maxhp=100, speed=100):
        return PokemonSpec(
            id=species, level=100, types=("normal",), hp=hp, maxhp=maxhp,
            attack=100, defense=100, special_attack=100, special_defense=100,
            speed=speed, status="none",
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    endgame_straddle = BattleSpec(
        side_one=SideSpec(pokemon=(small("rattata", ("splash", "tackle"), speed=200),)),
        side_two=SideSpec(pokemon=(small("chansey", ("splash", "tackle"), hp=50),)),
    )

    return {
        "minimal_1v1": build_poke_engine_state(minimal_gen3_fixture()).to_string(),
        "midgame_3v3": build_poke_engine_state(midgame).to_string(),
        "endgame_straddle": build_poke_engine_state(endgame_straddle).to_string(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--sims", default="256,1024")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds for the argmax-stability read.")
    parser.add_argument("--min-time", type=float, default=1.0, help="Minimum measured seconds per timing cell.")
    parser.add_argument("--deep-ko-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default=None, help="Optional markdown output path.")
    args = parser.parse_args(argv)

    import pokezero_search

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    sims_list = [int(x) for x in args.sims.split(",") if x.strip()]
    positions = _positions()

    lines = [
        "| position | depth | sims | sims/s | ms/decision | decision nodes | chance nodes | leaf evals | deep-KO triggers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, state in positions.items():
        for depth in depths:
            for sims in sims_list:
                # warm cell
                pokezero_search.puct_search_multi(
                    state, sims, max_depth=depth, seed=0, deep_ko_split=args.deep_ko_split
                )
                runs = 0
                start = time.perf_counter()
                while time.perf_counter() - start < args.min_time:
                    report = json.loads(
                        pokezero_search.puct_search_multi(
                            state, sims, max_depth=depth, seed=runs,
                            deep_ko_split=args.deep_ko_split,
                        )
                    )
                    runs += 1
                elapsed = time.perf_counter() - start
                per_decision = elapsed / runs
                row = (
                    f"| {name} | {depth} | {sims} | {sims / per_decision:,.0f} | "
                    f"{per_decision * 1e3:.2f} | {report['decision_nodes']} | "
                    f"{report['chance_nodes']} | {report['leaf_evals']} | "
                    f"{report['deep_ko_triggers']} |"
                )
                lines.append(row)
                print(row)

    stability = [
        "",
        f"Argmax stability across seeds 0..{args.seeds - 1} (sims=1024):",
        "",
        "| position | depth | argmax per seed | consistent | matches depth-1 |",
        "|---|---|---|---|---|",
    ]
    for name, state in positions.items():
        base_argmax = None
        for depth in depths:
            argmaxes = []
            for seed in range(args.seeds):
                report = json.loads(
                    pokezero_search.puct_search_multi(
                        state, max(sims_list), max_depth=depth, seed=seed,
                        deep_ko_split=args.deep_ko_split,
                    )
                )
                argmaxes.append(report["side_one"][0]["move"])
            consistent = len(set(argmaxes)) == 1
            if depth == 1:
                base_argmax = argmaxes[0] if consistent else None
            matches = (
                "n/a" if depth == 1
                else ("yes" if base_argmax is not None and set(argmaxes) == {base_argmax} else "no")
            )
            row = f"| {name} | {depth} | {', '.join(argmaxes)} | {'yes' if consistent else 'NO'} | {matches} |"
            stability.append(row)
            print(row)

    table = "\n".join(lines + stability)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
