"""Full-corpus mapping assertion for self-side model priors (engine swap).

For every drivable golden-corpus decision row (identical world construction
to ``scripts/leaf_root_parity.py`` — recorded payload + TRUE teams +
publicly-derivable recharge/Truant flags), the prior wiring's option→action
map (``LeafEncoder.self_action_map``, derived from the leaf encoder's own
``action_surface``) is asserted against the RECORDED request mask:

- interior surface (``get_all_options`` — every interior decision node's
  option order): every option maps, injectively, and the mapped set EQUALS
  the recorded golden ``legal_action_mask``'s legal bits;
- root surface (``root_get_all_options`` — force-trapped / slow-uturn aware,
  the ROOT node's order): every option maps, injectively, and the mapped set
  is CONTAINED in the recorded legal bits (narrower is legitimate — the
  engine refuses options the request could not know are dead — wider never).

Exit 0 iff zero mismatches. Companion unit test (committed sample, always
runs): ``tests/test_prior_action_mapping.py``.

Usage:
    PYTHONPATH=src python scripts/prior_mapping_assert.py \
        --corpus corpus/golden-v2 --corpus corpus/golden-v2-scenarios \
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

from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
    unpack_team,
)
from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT  # noqa: E402
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402

from fidelity_gate_events import (  # noqa: E402
    anchor_observation_metadata,
    production_recharging_slots,
    truant_loaf_slots,
)
from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402
from differential_denominator import check_denominator, gate as denominator_gate


def transformed_slots_from_metadata(metadata, player_id: str) -> dict[str, str]:
    """The publicly-Transformed actives, mirroring production's own derivation.

    `EngineMctsPolicy._public_effect_signals` builds exactly this from the
    observation's belief view and hands it to `battle_spec_from_payload` as
    `transformed_slots`; `engine_world._apply_transform` then replaces the
    transformed active's moveset with the copied one.

    This gate did not pass it, and so constructed a DIFFERENT world than
    production does: a transformed Ditto kept its own two-move surface while
    the recorded request mask carried the copied moves, which surfaces as
    `interior_set_mismatch` and reads as a mapping defect when it is a world
    defect. Recharge and Truant were already threaded here for the same reason;
    Transform was the one left out.
    """
    out: dict[str, str] = {}
    belief_view = (metadata or {}).get("belief_view")
    if not isinstance(belief_view, Mapping):
        return out
    opponent_slot = "p2" if player_id == "p1" else "p1"
    for slot, rows in (
        (opponent_slot, belief_view.get("opponent_pokemon")),
        (player_id, belief_view.get("self_pokemon")),
    ):
        for mon in rows or ():
            if not isinstance(mon, Mapping) or not mon.get("active"):
                continue
            if not mon.get("transformed"):
                continue
            target = str(mon.get("transform_species") or "")
            if target:
                out[slot] = target
    return out


def run_corpus(corpus_dir: Path, tables_json: str, verbose: bool) -> dict[str, Any]:
    corpus = load_golden_corpus(corpus_dir)
    dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)

    chains: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in iter_fold_records(
        corpus_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
    ):
        chains[(record["battle_id"], record["player_id"])].append(record)
    for chain in chains.values():
        chain.sort(key=lambda r: int(r["chain_index"]))
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
    mismatches: list[dict[str, Any]] = []
    root_narrower_rows = 0

    for array_row_index, row in enumerate(corpus.decision_rows):
        game = games[row.battle_id]
        true_teams = game.record.true_teams or {}
        packed = {slot: (true_teams.get(slot) or {}).get("packed") for slot in ("p1", "p2")}
        if not packed["p1"] or not packed["p2"]:
            counts["skip:no_true_teams"] += 1
            continue
        payload = row.public_materialization
        teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}

        # PRODUCTION's derivation, not the recorded candidate's. Deriving `recharging` from the
        # chosen action seeded this gate's world from the thing the gate is checking, so the world
        # could only ever agree -- the "would ratify a symmetric write rather than catch a bad one"
        # warning that leaf.rs and leaf_vs_reality.py both carry. See
        # fidelity_gate_events.production_recharging_slots.
        recharging = production_recharging_slots(
            anchor_observation_metadata(
                decisions.get((row.battle_id, row.decision_round_index, row.player_id))
            ),
            row.player_id,
        )
        truant = truant_loaf_slots(history_at_row.get(array_row_index) or [], payload, teams)
        transformed = transformed_slots_from_metadata(
            anchor_observation_metadata(
                decisions.get((row.battle_id, row.decision_round_index, row.player_id))
            ),
            row.player_id,
        )

        override = BattleStartOverride(player_teams={"p1": packed["p1"], "p2": packed["p2"]})
        try:
            world = battle_spec_from_payload(
                payload,
                override,
                dex=dex,
                # All four approximation flags, matching the PRODUCTION policy's
                # defaults (EngineMctsConfig: every one of these is True). The
                # gate previously passed only the first two and left the other
                # two at battle_spec_from_payload's False default, so it built a
                # world under different fidelity settings than the code it is
                # asserting -- and then attributed the difference to the map.
                #
                # approximate_hidden_duration_volatiles is the one that matters
                # here: it is what withdraws `taunt` from the supported set
                # (engine_world.py ~1309), so a taunted row raises
                # `volatile_unsupported` and SKIPS. That skip is the documented
                # fail-closed boundary -- Taunt's age is not publicly derivable
                # today and no seed is right -- not a row being excused.
                approximate_sleep_turns=True,
                approximate_substitute_health=True,
                approximate_partial_trap_turns=True,
                approximate_hidden_duration_volatiles=True,
                recharging_slots=tuple(recharging),
                truant_slots=tuple(truant),
                transformed_slots=transformed,
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
                tables_json,
                json.dumps(row_inputs_from_decision_row(row), sort_keys=True),
                ctx,
                state_str,
            )
            interior = encoder.self_action_map(state_str)
            root = encoder.self_action_map(state_str, root=True)
        except Exception as error:  # noqa: BLE001
            counts["skip:map_error"] += 1
            if verbose:
                print(f"map error at row {array_row_index}: {error}")
            continue

        recorded = {
            action_index
            for action_index, legal in enumerate(
                numpy.asarray(row.arrays.legal_action_mask).flatten().tolist()
            )
            if legal
        }
        problems: list[str] = []
        interior_indices = [index for _, index in interior]
        if None in interior_indices:
            problems.append("interior_unmapped_option")
        elif len(set(interior_indices)) != len(interior_indices):
            problems.append("interior_not_injective")
        elif set(interior_indices) != recorded:
            problems.append("interior_set_mismatch")
        root_indices = [index for _, index in root]
        if None in root_indices:
            problems.append("root_unmapped_option")
        elif len(set(root_indices)) != len(root_indices):
            problems.append("root_not_injective")
        elif not set(root_indices) <= recorded:
            problems.append("root_wider_than_mask")
        elif set(root_indices) != recorded:
            root_narrower_rows += 1

        # Counted after the skip decision and BEFORE classification, so it is independent
        # of the exact/divergent pair. Deriving it from their sum makes the partition
        # identity unfalsifiable — the tautology `leaf_vs_reality` shipped and then cited
        # as evidence (C112). An attempt that reaches here and never classifies leaves
        # matched + diverged < measured, which is what rule 3 is for.
        counts["attempted"] += 1
        if problems:
            counts["mismatch"] += 1
            for problem in problems:
                counts[f"mismatch:{problem}"] += 1
            if len(mismatches) < 50:
                mismatches.append(
                    {
                        "row": array_row_index,
                        "battle_id": row.battle_id,
                        "seat": row.player_id,
                        "round": row.decision_round_index,
                        "problems": problems,
                        "interior": [[d, i] for d, i in interior],
                        "root": [[d, i] for d, i in root],
                        "recorded_legal": sorted(recorded),
                    }
                )
        else:
            counts["exact"] += 1

    return {
        "corpus": str(corpus_dir),
        "rows": len(corpus.decision_rows),
        "root_narrower_rows": root_narrower_rows,
        "counts": dict(sorted(counts.items())),
        "mismatches": mismatches,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    tables_json = args.tables.read_text(encoding="utf-8")
    reports = []
    for corpus_dir in args.corpus:
        report = run_corpus(corpus_dir, tables_json, args.verbose)
        reports.append(report)
        print(f"== {corpus_dir}")
        print(f"   rows: {report['rows']}  root-narrower rows: {report['root_narrower_rows']}")
        for key, value in report["counts"].items():
            print(f"   {key:44s} {value}")
        for mismatch in report["mismatches"][:10]:
            print(f"   MISMATCH {mismatch}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    clean = all(report["counts"].get("mismatch", 0) == 0 for report in reports)
    # The denominator rule. `clean` is `all(mismatch == 0)`, vacuously true over zero rows.
    denominator = denominator_gate([
        check_denominator(
            str(report["corpus"]),
            measured=report["counts"].get("attempted", 0),
            matched=report["counts"].get("exact", 0),
            diverged=report["counts"].get("mismatch", 0),
            # Subscript, not .get: a renamed report key must CRASH rather than silently
            # disable rule 4, the only load-bearing rule. `measured` above fails safe (a
            # rename makes it 0 and rule 2 fires); this one would fail OPEN.
            contained=report["rows"],
            skipped=sum(v for k, v in report["counts"].items() if k.startswith("skip")),
        )
        for report in reports
    ])
    return 1 if (not clean or denominator) else 0


if __name__ == "__main__":
    raise SystemExit(main())
