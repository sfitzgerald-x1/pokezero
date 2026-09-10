#!/usr/bin/env python3
"""Source-bound two-seat replay for native forced-replacement prior mapping.

This deliberately exercises a real Showdown boundary rather than constructing a
request literal: a level-5 Magikarp is knocked out by Mewtwo, leaving exactly
one legal replacement.  The fixture runs once with p1 forced to replace and
once with p2 forced to replace.  At each boundary it proves that the current
source carries the request through public materialization and engine-world
construction into the compiled native root option-to-action map.

The script exits nonzero on any discrepancy.  With ``--out`` it atomically
writes either a complete PASS.json or NONPASS.json, so a cluster launcher can
use the result as its terminal durable artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_encoder_tables import build_tables  # noqa: E402
from pokezero.actions import MOVE_ACTION_COUNT  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.engine_world import world_battle_spec  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.golden_corpus import _json_safe  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402
from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


SEED = 109200600
RESULT_SCHEMA_VERSION = "pokezero.force-switch-prior-replay.v1"


@dataclass(frozen=True)
class SeatReplay:
    """Complete evidence for one real forced-replacement boundary."""

    player_id: str
    request_force_switch: bool
    payload_request_kind: str
    acting_side_force_switch: bool
    opposing_side_force_switch: bool
    live_legal_action_indices: tuple[int, ...]
    native_root_action_indices: tuple[int, ...]
    native_root_options: tuple[str, ...]
    state_sha256: str


def _config(showdown_root: Path) -> LocalShowdownConfig:
    return LocalShowdownConfig(
        showdown_root=showdown_root,
        read_timeout_seconds=10.0,
        # The native LeafEncoder accepts the current v2.2+ layout, not the
        # legacy v2.1 fixture layout.  This is a root-map replay, so retain a
        # supported production layout rather than letting the harness itself
        # be the source of a compatibility failure.
        observation_spec=V2_2_REPLAY_OBSERVATION_SPEC,
    )


def _start_override(forced_player: str) -> BattleStartOverride:
    """Return a one-replacement battle where ``forced_player`` must switch.

    Mewtwo moves first and KOs the opposing level-5 Magikarp.  The fainted
    player has exactly one healthy bench Pokemon, which makes a missing or
    move-valued native root option unambiguous.
    """

    magikarp_team = pack_team(
        (
            FixturePokemon(
                species="Magikarp", ability="Swift Swim", moves=("Tackle",), level=5
            ),
            FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Tackle",)),
        )
    )
    mewtwo_team = pack_team(
        (FixturePokemon(species="Mewtwo", ability="Pressure", moves=("Psychic",)),)
    )
    if forced_player == "p1":
        teams = {"p1": magikarp_team, "p2": mewtwo_team}
    elif forced_player == "p2":
        teams = {"p1": mewtwo_team, "p2": magikarp_team}
    else:
        raise ValueError(f"forced_player must be p1 or p2, got {forced_player!r}")
    return BattleStartOverride(player_teams=teams)


def _root_inputs_json(
    *,
    player_id: str,
    observation: Any,
    materialization: Any,
) -> str:
    """Mirror ``EngineMctsPolicy._root_inputs_json`` for this live boundary."""

    row = {
        "battle_id": "force-switch-prior-replay",
        "battle_seed": SEED,
        "format_id": materialization.format_id,
        "player_id": player_id,
        "observation_schema_version": observation.schema_version,
        "observation_metadata": _json_safe(
            dict(observation.metadata), context="observation_metadata"
        ),
        "public_materialization": _json_safe(
            _public_materialization_payload(materialization),
            context="public_materialization",
        ),
    }
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _side_force_switch(world: Any, player_id: str) -> tuple[bool, bool]:
    acting_attr = world.slot_sides[player_id]
    opposing_player = "p2" if player_id == "p1" else "p1"
    opposing_attr = world.slot_sides[opposing_player]
    acting_side = getattr(world.spec, acting_attr)
    opposing_side = getattr(world.spec, opposing_attr)
    return bool(acting_side.force_switch), bool(opposing_side.force_switch)


def replay_seat(*, player_id: str, showdown_root: Path, tables_json: str) -> SeatReplay:
    """Drive and validate one real forced replacement for ``player_id``."""

    override = _start_override(player_id)
    dex = load_showdown_dex_cached(showdown_root)
    with LocalShowdownEnv(_config(showdown_root)) as env:
        env.reset_with_start_override(seed=SEED, start_override=override)
        env.step({"p1": 0, "p2": 0})
        observation = env.observe(player_id)
        materialization = env.public_materialization_state(player_id)

    force_switch = materialization.self_request.get("forceSwitch")
    if not isinstance(force_switch, list) or not force_switch or force_switch[0] is not True:
        raise AssertionError(
            f"{player_id}: fixture did not reach a forceSwitch request: {force_switch!r}"
        )
    payload = _public_materialization_payload(materialization)
    request_kind = str(payload.get("selfRequestKind") or "")
    if request_kind != "force-switch":
        raise AssertionError(
            f"{player_id}: payload lost force-switch request kind: {request_kind!r}"
        )

    world = world_battle_spec(materialization, override, dex=dex)
    acting_force_switch, opposing_force_switch = _side_force_switch(world, player_id)
    if not acting_force_switch or opposing_force_switch:
        raise AssertionError(
            f"{player_id}: engine world force-switch flags were acting={acting_force_switch} "
            f"opposing={opposing_force_switch}"
        )

    live_legal = tuple(
        action_index
        for action_index, legal in enumerate(observation.legal_action_mask)
        if legal
    )
    if not live_legal or any(action_index < MOVE_ACTION_COUNT for action_index in live_legal):
        raise AssertionError(
            f"{player_id}: forced replacement was not switch-only: {live_legal}"
        )

    import pokezero_search

    state = build_poke_engine_state(world.spec)
    state_string = state.to_string()
    context_json = json.dumps(
        {
            "p1": list(world.party_species["p1"]),
            "p2": list(world.party_species["p2"]),
            "turn": int(materialization.replay.turn_number),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    encoder = pokezero_search.LeafEncoder(
        tables_json,
        _root_inputs_json(
            player_id=player_id,
            observation=observation,
            materialization=materialization,
        ),
        context_json,
        state_string,
    )
    root_map = tuple(encoder.self_action_map(state_string, root=True))
    root_options = tuple(str(option) for option, _ in root_map)
    root_indices = tuple(action_index for _, action_index in root_map)
    if not root_indices:
        raise AssertionError(f"{player_id}: native root map is empty")
    if any(action_index is None for action_index in root_indices):
        raise AssertionError(f"{player_id}: native root map contains an unmapped option: {root_map!r}")
    if any(not isinstance(action_index, int) for action_index in root_indices):
        raise AssertionError(f"{player_id}: native root map has non-integer action index: {root_map!r}")
    if any(action_index < MOVE_ACTION_COUNT for action_index in root_indices):
        raise AssertionError(
            f"{player_id}: native root map contains a move at forced replacement: {root_map!r}"
        )
    if len(set(root_indices)) != len(root_indices):
        raise AssertionError(f"{player_id}: native root map is not injective: {root_map!r}")
    if set(root_indices) != set(live_legal):
        raise AssertionError(
            f"{player_id}: native root map {sorted(root_indices)} does not equal "
            f"live switch-only legal mask {list(live_legal)}"
        )

    return SeatReplay(
        player_id=player_id,
        request_force_switch=True,
        payload_request_kind=request_kind,
        acting_side_force_switch=acting_force_switch,
        opposing_side_force_switch=opposing_force_switch,
        live_legal_action_indices=live_legal,
        native_root_action_indices=tuple(int(index) for index in root_indices),
        native_root_options=root_options,
        state_sha256=hashlib.sha256(state_string.encode("utf-8")).hexdigest(),
    )


def run_replay(*, showdown_root: Path) -> dict[str, Any]:
    """Run both seats and return only complete source-bound evidence."""

    tables_json = json.dumps(
        build_tables(
            str(showdown_root),
            observation_schema_version=V2_2_REPLAY_OBSERVATION_SPEC.schema_version,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    seats = tuple(
        replay_seat(player_id=player_id, showdown_root=showdown_root, tables_json=tables_json)
        for player_id in ("p1", "p2")
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result": "PASS",
        "seed": SEED,
        "showdown_root": str(showdown_root.resolve()),
        "encoder_tables_sha256": hashlib.sha256(tables_json.encode("utf-8")).hexdigest(),
        "seats": [asdict(seat) for seat in seats],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", type=Path, default=Path(DEFAULT_SHOWDOWN_ROOT))
    parser.add_argument("--out", type=Path, help="Directory to receive complete terminal JSON.")
    args = parser.parse_args(argv)

    try:
        result = run_replay(showdown_root=args.showdown_root)
    except Exception as error:
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "result": "NONPASS",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if args.out is not None:
            _atomic_write_json(args.out / "NONPASS.json", failure)
        else:
            print(json.dumps(failure, sort_keys=True, indent=2), file=sys.stderr)
        return 1

    if args.out is not None:
        _atomic_write_json(args.out / "PASS.json", result)
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
