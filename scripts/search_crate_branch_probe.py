#!/usr/bin/env python
"""Behavioral smoke for the native branch-events seam used by certification."""

from __future__ import annotations

import json

import pokezero_search

from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)


def _pokemon(species: str, moves: tuple[str, ...], speed: int) -> PokemonSpec:
    return PokemonSpec(
        id=species,
        level=100,
        types=("normal",),
        hp=100,
        maxhp=100,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=speed,
        moves=tuple(MoveSpec(id=move, pp=32) for move in moves),
    )


def main() -> None:
    state = build_poke_engine_state(
        BattleSpec(
            side_one=SideSpec(
                pokemon=(_pokemon("rattata", ("toxic",), speed=200),)
            ),
            side_two=SideSpec(
                pokemon=(_pokemon("chansey", ("splash",), speed=100),)
            ),
        )
    ).to_string()
    context = json.dumps({"p1": ["Rattata"], "p2": ["Chansey"], "turn": 1})
    report = json.loads(
        pokezero_search.branch_events(
            state,
            "toxic",
            "splash",
            context,
            True,
            False,
        )
    )
    branches = report.get("branches") or []
    percentages = sorted(round(float(branch["percentage"])) for branch in branches)
    if percentages != [15, 85]:
        raise SystemExit(
            f"branch-events probe expected Toxic masses [15, 85], got {percentages}"
        )
    if not all(branch.get("turn_completed") for branch in branches):
        raise SystemExit("branch-events probe returned an incomplete turn")
    print("[search-crate-branch-events] PASS Toxic masses 15/85")


if __name__ == "__main__":
    main()
