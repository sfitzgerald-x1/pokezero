#!/usr/bin/env python3
"""V4 — full-game Python/Rust v4 replay differential (accumulator drift).

Every existing v4 parity check is SINGLE-STATE: one corpus row with hand-set metadata. The k0 pack's
B-columns (hazard credit, items-removed credit, switch propensity) and the last-damage ledgers are
ACCUMULATORS -- the Python env maintains them across turns while the native side rebuilds them from
fold products at search time. Fixture parity proves the WRITE, not the ACCUMULATION, and the gap
between those two is where both Rust defects this generation lived.

So this replays FULL GAMES: every decision point, BOTH seats, through both encoders, byte-comparing
all five observation arrays. The row-input surface is built with the SAME helpers production uses
(`_public_materialization_payload` + `_json_safe`, exactly as `engine_search.py` builds the crate's
input), so this compares the shape the native leaf actually consumes rather than a shape invented
here.

Exit criterion (plan §1, V4): **byte-identical across >=200 full games** (~20k states). Gates
NATIVE-LEAF consumers -- search-enabled evals and their strength numbers -- not collection, which is
Python-side.

Per plan §3, the wheel must be rebuilt and reinstalled before results are read: twice this
generation a stale binary's results were read as real. This script prints the loaded module path and
its build fingerprint so the artifact records WHICH binary produced the verdict.

Usage:
    uv run python scripts/v4_replay_parity_gate.py --games 200 --seed 3 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/v4-replay-parity-2026-08-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy  # noqa: E402
import pokezero_search  # noqa: E402

import golden_encoder_backends as backends  # noqa: E402
import export_encoder_tables as exporter  # noqa: E402
from pokezero.golden_corpus import _json_safe  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from pokezero.observation import (  # noqa: E402
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    OBSERVATION_SCHEMA_VERSION_V4,
)
from pokezero.showdown import MOVE_ACTION_COUNT, observation_spec_for_schema  # noqa: E402

PLAYERS = ("p1", "p2")

# Columns whose whole point is accumulation across turns. Reported separately so a mismatch says
# WHICH kind of drift it is rather than only "numeric_features differ".
ACCUMULATOR_COLUMNS = (
    "NUMERIC_SELF_HAZARD_CREDIT",
    "NUMERIC_OPP_HAZARD_CREDIT",
    "NUMERIC_SELF_HAZARD_EXPECTED",
    "NUMERIC_SELF_ITEMS_REMOVED_CREDIT",
    "NUMERIC_OPP_ITEMS_REMOVED_CREDIT",
    "NUMERIC_MON_SWITCHED_VS_ACTIVE",
    "NUMERIC_MON_STAYED_VS_ACTIVE",
    "NUMERIC_LAST_DAMAGE_DEALT",
    "NUMERIC_LAST_DAMAGE_TAKEN",
)


def _v4_header(spec) -> dict[str, Any]:
    """A minimal v4 corpus header, enough for ``observation_contract_from_header``."""
    masks = DEFAULT_OBSERVATION_FEATURE_MASKS
    return {
        "observation": {
            "schema_version": OBSERVATION_SCHEMA_VERSION_V4,
            "token_count": spec.token_count,
            "categorical_feature_count": spec.categorical_feature_count,
            "numeric_feature_count": spec.numeric_feature_count,
            # Stamped from the SAME defaults the env below is constructed with, so the two cannot
            # disagree about the contract under test.
            "feature_masks": {
                "stats_block": masks.opponent_tendency_stats_block,
                "exact_state": masks.exact_state,
                "transition_token_budget": masks.transition_token_budget,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        }
    }


def _row_inputs(env: LocalShowdownEnv, player: str, observation, *, seed: int) -> dict[str, Any]:
    """The crate's sanctioned input surface for one live decision point.

    Built with production's own helpers rather than re-derived here: `engine_search.py` composes the
    identical dict for the native policy, so a divergence found by this gate is a divergence the
    native leaf would really see.
    """
    state = env.public_materialization_state(player)
    return {
        "battle_id": "v4-replay-parity",
        "battle_seed": int(seed),
        "format_id": "gen3randombattle",
        "player_id": player,
        "observation_schema_version": observation.schema_version,
        "observation_metadata": _json_safe(
            dict(observation.metadata), context="observation_metadata"
        ),
        "public_materialization": _json_safe(
            _public_materialization_payload(state), context="public_materialization"
        ),
    }


def run_gate(
    *,
    showdown_root: Path,
    games: int = 200,
    seed: int = 3,
    max_steps: int = 400,
    move_bias: float = 0.75,
) -> dict[str, Any]:
    spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V4)
    header = _v4_header(spec)
    contract_spec, masks = backends.observation_contract_from_header(header)
    tables = exporter.build_tables(
        str(showdown_root), observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4
    )
    tables_json = json.dumps(tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    numeric_columns = tables["layout"]["numeric_columns"]

    python_backend = backends.PythonReferenceBackend(showdown_root=showdown_root, header=header)
    rust_backend = backends.RustBackend(tables_json=tables_json, header=header)

    env = LocalShowdownEnv(
        LocalShowdownConfig(
            showdown_root=str(showdown_root),
            observation_spec=spec,
            feature_masks=DEFAULT_OBSERVATION_FEATURE_MASKS,
            set_belief_source=True,
        )
    )

    counts: Counter = Counter(
        {"games": 0, "states": 0, "arrays_compared": 0, "accumulator_states_reached": 0}
    )
    mismatches: list[dict[str, Any]] = []

    try:
        for game in range(games):
            rng = random.Random(seed * 1_000_003 + game)
            env.reset(seed=seed + game)
            counts["games"] += 1
            steps = 0
            while steps < max_steps and env.terminal() is None:
                requested = env.requested_players()
                if not requested:
                    break
                # BOTH seats every decision point, not just the requested one: an accumulator that
                # drifts only on the non-acting seat is still a wrong input at the next search.
                for player in PLAYERS:
                    observation = env.observe(player)
                    row_inputs = _row_inputs(env, player, observation, seed=seed + game)
                    want = python_backend.encode(row_inputs)
                    got = rust_backend.encode(row_inputs)
                    counts["states"] += 1
                    if any(
                        numpy.asarray(want["numeric_features"])[
                            :, numeric_columns[name]
                        ].any()
                        for name in ACCUMULATOR_COLUMNS
                        if name in numeric_columns
                    ):
                        counts["accumulator_states_reached"] += 1
                    for name in backends.ARRAY_NAMES:
                        counts["arrays_compared"] += 1
                        left = numpy.ascontiguousarray(want[name]).tobytes()
                        right = numpy.ascontiguousarray(got[name]).tobytes()
                        if left == right:
                            continue
                        detail: dict[str, Any] = {
                            "game": game,
                            "step": steps,
                            "player": player,
                            "array": name,
                        }
                        if name == "numeric_features":
                            lhs = numpy.asarray(want[name])
                            rhs = numpy.asarray(got[name])
                            bad = numpy.argwhere(lhs != rhs)
                            by_name = {v: k for k, v in numeric_columns.items()}
                            detail["columns"] = sorted(
                                {by_name.get(int(col), int(col)) for _row, col in bad[:200]},
                                key=str,
                            )
                            detail["accumulator_columns"] = sorted(
                                c for c in detail["columns"] if c in ACCUMULATOR_COLUMNS
                            )
                            if len(bad):
                                r, c = bad[0]
                                detail["first"] = {
                                    "token": int(r),
                                    "column": by_name.get(int(c), int(c)),
                                    "python": float(lhs[r, c]),
                                    "rust": float(rhs[r, c]),
                                }
                        mismatches.append(detail)
                actions = {}
                for player in requested:
                    mask = env.observe(player).legal_action_mask
                    legal = [index for index, allowed in enumerate(mask) if allowed]
                    if not legal:
                        break
                    moves = [index for index in legal if index < MOVE_ACTION_COUNT]
                    if moves and rng.random() < move_bias:
                        actions[player] = rng.choice(moves)
                    else:
                        actions[player] = rng.choice(legal)
                if len(actions) != len(requested):
                    counts["truncated_games"] += 1
                    break
                env.step(actions)
                steps += 1
    finally:
        env.close()

    # Reachability: byte-parity over states that never populated an accumulator would prove only the
    # WRITE, which fixture parity already proved. The accumulators are the point.
    reached = counts["states"] > 0 and counts["accumulator_states_reached"] > 0
    verdict = "PASS" if (not mismatches and reached) else "FAIL"
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "reached": reached,
        # WHICH binary produced this verdict (plan §3: a stale wheel's results were read as real
        # twice this generation).
        "native_module": getattr(pokezero_search, "__file__", "?"),
        "native_fingerprint": getattr(pokezero_search, "ENGINE_BUILD_FINGERPRINT", None),
        "args": {"games": games, "seed": seed, "max_steps": max_steps, "move_bias": move_bias},
        "counts": dict(counts),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:40],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
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
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v4-replay-parity.json").write_text(json.dumps(summary, indent=2) + "\n")
    counts = summary["counts"]
    print(
        f"[v4-replay-parity] {summary['verdict']} games={counts.get('games', 0)} "
        f"states={counts.get('states', 0)} arrays={counts.get('arrays_compared', 0)} "
        f"accumulator_states={counts.get('accumulator_states_reached', 0)} "
        f"mismatches={summary['mismatch_count']} reached={summary['reached']} "
        f"native={summary['native_module']}"
    )
    for row in summary["mismatches"][:10]:
        print(f"  {row}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
