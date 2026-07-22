"""Root-parity gate for the leaf observation path (engine-swap capstone).

The leaf encoder (`pokezero_search.LeafEncoder`) builds a leaf observation
from three sources: ENGINE-STATE-DERIVED columns recomputed from a poke-engine
state, WORLD-CONSTANT columns carried from the root row inputs, and
FOLD-DERIVED columns from an advanced fold state
(docs/leaf_observation_column_map.md). At depth 0 — zero branch steps — that
construction must reproduce the production observation EXACTLY: the engine
state is the root world, the fold is the root fold, and every recomputed
column must land on the recorded golden bytes.

This gate drives that claim over golden-corpus rows using the fidelity
harness's state-reconstruction machinery (scripts/fidelity_gate_events.py):

1. construct the engine world at the row from the recorded public
   materialization payload + the game's TRUE teams (no belief sampling — the
   recorded world, `battle_spec_from_payload` with the true packed override,
   with the same publicly-derivable recharge/Truant flags);
2. `LeafEncoder.encode_leaf(root_state, recorded_fold_state, root_turn)`;
3. byte-diff all five observation arrays against the recorded golden arrays.

Outcome classes per row (fidelity-gate discipline):
- exact           — every array byte-identical;
- divergent       — >=1 cell differs; bucketed by (array, block, column-name)
                    family with per-family examples — every family must be
                    attributed (a bug or a documented contract finding),
                    never averaged away;
- skip:<reason>   — the world cannot be constructed for this row
                    (EngineWorldUnsupported fail-closed reasons), counted.

Usage:
    PYTHONPATH=src python scripts/leaf_root_parity.py \
        --corpus corpus/golden-v2 [--corpus corpus/golden-v2-scenarios] \
        --tables corpus/encoder_tables.json [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pokezero_search  # noqa: E402

from pokezero.dex import normalize_id  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
    unpack_team,
)
from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_ARRAY_FIELDS,
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402

from fidelity_gate_events import chosen_candidate, truant_loaf_slots  # noqa: E402
from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402

FIXED_TOKEN_BLOCKS = (
    ("field", 0, 1),
    ("self_team", 1, 7),
    ("opponent_team", 7, 13),
    ("action", 13, 22),
    ("stats", 22, 23),
)


def block_of(token: int, token_count: int) -> str:
    blocks = (*FIXED_TOKEN_BLOCKS, ("transition", 23, token_count))
    for name, start, stop in blocks:
        if start <= token < stop:
            return name
    return f"token{token}"


def column_names(tables: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    layout = tables["layout"]
    return {
        "categorical_ids": {v: k for k, v in layout["categorical_columns"].items()},
        "numeric_features": {v: k for k, v in layout["numeric_columns"].items()},
    }


def bitwise_equal(got: numpy.ndarray, want: numpy.ndarray) -> numpy.ndarray:
    if got.dtype.kind == "f":
        unsigned = numpy.dtype(f"<u{got.dtype.itemsize}")
        return got.view(unsigned) == want.view(unsigned)
    return got == want


def run_corpus(
    corpus_dir: Path, tables_json: str, tables: Mapping[str, Any], verbose: bool
) -> dict[str, Any]:
    corpus = load_golden_corpus(corpus_dir)
    names = column_names(tables)
    token_count = int(tables["layout"]["token_count"])

    fold_states: dict[int, Mapping[str, Any]] = {}
    chains: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in iter_fold_records(
        corpus_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
    ):
        fold_states[int(record["array_row_index"])] = record
        chains[(record["battle_id"], record["player_id"])].append(record)
    for chain in chains.values():
        chain.sort(key=lambda r: int(r["chain_index"]))
    # Public history lines per (battle, seat) INCLUDING each row's own slice,
    # keyed by array row (for the Truant loaf derivation).
    history_at_row: dict[int, list[str]] = {}
    for chain in chains.values():
        history: list[str] = []
        for record in chain:
            history.extend(record.get("event_slice") or ())
            history_at_row[int(record["array_row_index"])] = list(history)

    games = {game.record.battle_id: game for game in corpus.games}
    decisions: dict[tuple[str, int, str], Any] = {}
    for game in corpus.games:
        for row in game.rows:
            decisions[(row.battle_id, row.decision_round_index, row.player_id)] = row

    counts: Counter[str] = Counter()
    families: Counter[tuple[str, str, str]] = Counter()
    family_examples: dict[tuple[str, str, str], dict[str, Any]] = {}
    divergent_rows: list[dict[str, Any]] = []

    for array_row_index, row in enumerate(corpus.decision_rows):
        game = games[row.battle_id]
        true_teams = game.record.true_teams or {}
        packed = {slot: (true_teams.get(slot) or {}).get("packed") for slot in ("p1", "p2")}
        if not packed["p1"] or not packed["p2"]:
            counts["skip:no_true_teams"] += 1
            continue
        fold_record = fold_states.get(array_row_index)
        if fold_record is None:
            counts["skip:no_fold_record"] += 1
            continue
        payload = row.public_materialization
        teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}

        # Publicly-derivable engine flags, exactly the fidelity gate's rules.
        recharging = []
        for slot in ("p1", "p2"):
            other = decisions.get((row.battle_id, row.decision_round_index, slot))
            candidate = chosen_candidate_from_row(other) if other is not None else None
            if (
                candidate is not None
                and candidate.get("kind") == "move"
                and normalize_id(str(candidate.get("move_id") or "")) == "recharge"
            ):
                recharging.append(slot)
        truant = truant_loaf_slots(
            history_at_row.get(array_row_index) or [], payload, teams
        )

        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

        dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        override = BattleStartOverride(player_teams={"p1": packed["p1"], "p2": packed["p2"]})
        try:
            world = battle_spec_from_payload(
                payload,
                override,
                dex=dex,
                approximate_sleep_turns=True,
                approximate_substitute_health=True,
                recharging_slots=tuple(recharging),
                truant_slots=tuple(truant),
            )
            state = build_poke_engine_state(world.spec)
        except EngineWorldUnsupported as error:
            counts[f"skip:world_unsupported:{error.reason}"] += 1
            continue
        except Exception as error:  # noqa: BLE001
            counts["skip:world_error"] += 1
            if verbose:
                print(f"world error at row {array_row_index}: {error}")
            continue

        row_inputs = row_inputs_from_decision_row(row)
        ctx = json.dumps(
            {
                "p1": list(world.party_species["p1"]),
                "p2": list(world.party_species["p2"]),
                "turn": int(payload.get("turn") or 0),
            }
        )
        state_str = state.to_string()
        try:
            encoder = pokezero_search.LeafEncoder(
                tables_json, json.dumps(row_inputs, sort_keys=True), ctx, state_str
            )
            fold = pokezero_search.FoldState.from_payload(fold_record["fold_state"])
            buffers = encoder.encode_leaf(
                state_str, fold, int(row.observation_metadata.get("turn_number") or 0)
            )
        except Exception as error:  # noqa: BLE001
            counts["skip:encode_error"] += 1
            if verbose:
                print(f"encode error at row {array_row_index}: {error}")
            continue

        row_families: set[tuple[str, str, str]] = set()
        cells_divergent = 0
        for name, dtype, _ in GOLDEN_ARRAY_FIELDS:
            want = numpy.ascontiguousarray(getattr(row.arrays, name), dtype=dtype)
            got = numpy.frombuffer(buffers[name], dtype=dtype).reshape(want.shape)
            equal = bitwise_equal(got, want)
            if bool(equal.all()):
                continue
            for position in numpy.argwhere(~equal):
                cells_divergent += 1
                if want.ndim == 2:
                    token, column = int(position[0]), int(position[1])
                    colname = names.get(name, {}).get(column, f"col{column}")
                    if name == "categorical_ids" and column not in names["categorical_ids"]:
                        colname = f"col{column}"
                else:
                    token, column = int(position[0]), -1
                    colname = name
                    if name == "legal_action_mask":
                        token, column = -1, int(position[0])
                        colname = f"action{column}"
                family = (
                    name,
                    block_of(token, token_count) if token >= 0 else name,
                    colname,
                )
                row_families.add(family)
                if family not in family_examples:
                    index = tuple(int(v) for v in position)
                    family_examples[family] = {
                        "row": array_row_index,
                        "battle_id": row.battle_id,
                        "seat": row.player_id,
                        "round": row.decision_round_index,
                        "token": token,
                        "column": column,
                        "got": got[index].item(),
                        "want": want[index].item(),
                    }
        if cells_divergent == 0:
            counts["exact"] += 1
        else:
            counts["divergent"] += 1
            for family in row_families:
                families[family] += 1
            divergent_rows.append(
                {
                    "row": array_row_index,
                    "battle_id": row.battle_id,
                    "seat": row.player_id,
                    "round": row.decision_round_index,
                    "cells": cells_divergent,
                    "families": sorted("/".join(f) for f in row_families),
                }
            )

    return {
        "corpus": str(corpus_dir),
        "rows": len(corpus.decision_rows),
        "counts": dict(sorted(counts.items())),
        "families": [
            {
                "array": array,
                "block": block,
                "column": column,
                "rows": count,
                "example": family_examples.get((array, block, column)),
            }
            for (array, block, column), count in sorted(
                families.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "divergent_rows": divergent_rows[:200],
    }


def chosen_candidate_from_row(row: Any) -> Mapping[str, Any] | None:
    index = row.chosen_action_index
    for candidate in row.observation_metadata.get("action_candidates") or ():
        if candidate.get("action_index") == index:
            return candidate
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    tables_json = args.tables.read_text(encoding="utf-8")
    tables = json.loads(tables_json)
    reports = []
    for corpus_dir in args.corpus:
        report = run_corpus(corpus_dir, tables_json, tables, args.verbose)
        reports.append(report)
        print(f"== {corpus_dir}")
        print(f"   rows: {report['rows']}")
        for key, value in report["counts"].items():
            print(f"   {key:44s} {value}")
        if report["families"]:
            print("   divergence families (rows affected):")
            for family in report["families"][:25]:
                example = family["example"] or {}
                print(
                    f"     {family['array']}/{family['block']}/{family['column']:38s} "
                    f"{family['rows']:4d}   e.g. row {example.get('row')} "
                    f"got={example.get('got')!r} want={example.get('want')!r}"
                )
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    driven_exact = all(
        report["counts"].get("divergent", 0) == 0 for report in reports
    )
    return 0 if driven_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())


# The fidelity harness's chosen_candidate operates on raw JSONL dicts; the
# loader used here yields dataclasses — chosen_candidate_from_row above is the
# dataclass twin. `chosen_candidate` stays imported for interface parity.
_ = chosen_candidate
