#!/usr/bin/env python
"""Behavioral smoke for the native branch-events seam used by certification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pokezero_search

from engine_build_fingerprint import STAMP_SCHEMA, _installed_artifacts, compute_fingerprint

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
    module_path = Path(pokezero_search.__file__ or "").resolve()
    expected_prefix = Path(sys.prefix).resolve()
    if not module_path.is_file() or expected_prefix not in module_path.parents:
        raise SystemExit(
            "branch-events probe imported pokezero_search outside the target venv: "
            f"{module_path}"
        )
    stamp_path = expected_prefix / ".engine-build-fingerprint.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"branch-events probe cannot read engine stamp {stamp_path}: {error}")
    if stamp.get("schema") != STAMP_SCHEMA:
        raise SystemExit("branch-events probe requires a two-consumer engine build stamp")
    current_fingerprint = compute_fingerprint()["fingerprint"]
    if stamp.get("fingerprint") != current_fingerprint:
        raise SystemExit("branch-events probe engine fingerprint does not match this checkout")
    if stamp.get("artifacts") != _installed_artifacts():
        raise SystemExit("branch-events probe imported artifacts that do not match the build stamp")

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
    if report.get("end_of_turn") is not True:
        raise SystemExit("branch-events probe did not report end_of_turn for a resolved Toxic turn")
    for branch in branches:
        events = branch.get("events")
        if not isinstance(events, list) or not events or events[0] != "|":
            raise SystemExit("branch-events probe returned an unmapped event sequence")
        if branch.get("lossy") != []:
            raise SystemExit(f"branch-events probe unexpectedly rendered lossy branch: {branch.get('lossy')!r}")
        if "|upkeep" not in events or "|turn|2" not in events:
            raise SystemExit("branch-events probe omitted mapped end-of-turn events")
    if not any("|-status|p2a: Chansey|tox" in branch["events"] for branch in branches):
        raise SystemExit("branch-events probe omitted the mapped Toxic status event")
    if not any("|-miss|p1a: Rattata|p2a: Chansey" in branch["events"] for branch in branches):
        raise SystemExit("branch-events probe omitted the mapped Toxic miss event")
    print(
        "[search-crate-branch-events] PASS "
        f"Toxic masses 15/85 mapped-events lossy=0 eot=true module={module_path} "
        f"fingerprint={current_fingerprint}"
    )


if __name__ == "__main__":
    main()
