"""Fidelity gate for the instruction→event mapping (track B, engine swap).

For every same-seat decision boundary (row n -> row n+1) of a golden corpus
v2 fold sidecar, this harness:

1. constructs the poke-engine state at row n from the recorded PUBLIC
   materialization payload + the game's TRUE teams (the fixed-override /
   scenario pattern: ``engine_world.battle_spec_from_payload`` with the true
   packed teams as the override — no belief sampling, the recorded world);
2. re-plays the decision rounds between the two boundaries: per round, the
   joint actions the players ACTUALLY took (from the recorded decision rows)
   are stepped through ``generate_instructions_from_move_pair`` via
   ``pokezero_search.branch_events``, and the enumerated outcome branch
   CONSISTENT with what actually happened is selected by matching the
   branch's post-state (active identity / HP / status per side) against the
   next round's recorded public payload;
3. renders that branch's instructions as protocol lines (the mapper under
   test), advances the RECORDED row-n fold state over the synthesized lines
   (Rust ``pokezero_search.FoldState``), applies row n+1's recorded
   annotation overlay, and compares fold PRODUCTS (transition tokens +
   tendencies — the encoder-visible surface) against row n+1's recorded
   products.

Outcome classes per boundary:
  (a) products byte-identical (canonical JSON);
  (b) token streams semantically equal, bytes differ only through documented
      equivalences (float rounding within tolerance; known action aliases);
  (c) real divergence (reported with a reason histogram).
Boundaries that cannot be driven are SKIPPED with a counted reason
(world construction fail-closed, unmappable action, no branch matching the
realized outcome, mapper-reported lossy rendering).

Usage:
    PYTHONPATH=src python scripts/fidelity_gate_events.py \
        --corpus /path/to/golden-v2-scenarios [--corpus /path/to/golden-v2] \
        [--json report.json] [--verbose]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pokezero_search  # noqa: E402

from pokezero.dex import load_showdown_dex, normalize_id  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
    hidden_power_engine_id,
    unpack_team,
)
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402

FLOAT_FIELDS = ("damage_fraction", "self_hp_cost", "residual", "investment")
FLOAT_TOL = 0.02
# Damage rolls: the engine collapses to 0.925x max, the realized roll is in
# [0.85, 1.0]x — a relative deviation of up to ~8.1% on damage-derived floats.
ROLL_FIELDS = ("damage_fraction", "self_hp_cost", "residual")
ROLL_REL_TOL = 0.09


def float_equiv(key: str, a: Any, b: Any) -> bool:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    delta = abs(float(a) - float(b))
    if delta <= FLOAT_TOL:
        return True
    magnitude = max(abs(float(a)), abs(float(b)))
    return key in ROLL_FIELDS and magnitude > 0 and delta / magnitude <= ROLL_REL_TOL
STATUS_CODES = {
    "brn": "burn",
    "par": "paralyze",
    "psn": "poison",
    "tox": "toxic",
    "frz": "freeze",
    "slp": "sleep",
    "": "none",
}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus(corpus_dir: Path) -> dict[str, Any]:
    games: dict[str, dict[str, Any]] = {}
    decisions: dict[tuple[str, int, str], dict[str, Any]] = {}
    with (corpus_dir / "rows.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "game":
                games[row["battle_id"]] = row
            elif row.get("record_type") == "decision":
                key = (row["battle_id"], row["decision_round_index"], row["player_id"])
                decisions[key] = row
    fold_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(corpus_dir / "fold.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "fold_row" and "fold_state" not in row:
                continue
            fold_rows[(row["battle_id"], row["player_id"])].append(row)
    for chain in fold_rows.values():
        chain.sort(key=lambda r: r["chain_index"])
    return {"games": games, "decisions": decisions, "fold_chains": fold_rows}


# ---------------------------------------------------------------------------
# Action mapping: recorded decision -> engine move string
# ---------------------------------------------------------------------------


def chosen_candidate(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    index = row.get("chosen_action_index")
    for candidate in row.get("observation_metadata", {}).get("action_candidates") or ():
        if candidate.get("action_index") == index:
            return candidate
    return None


def engine_move_string(candidate: Mapping[str, Any], side_moves: list[str], side_species: list[str]) -> str | None:
    if candidate.get("kind") == "move":
        wanted = normalize_id(str(candidate.get("move_id") or ""))
        if wanted == "recharge":
            # Hyper Beam recharge turn: the engine's forced choice is None
            # with the MUSTRECHARGE volatile (passed via recharging_slots).
            return "none"
        for move_id in side_moves:
            if move_id == wanted:
                return move_id
            if wanted.startswith("hiddenpower") and move_id.startswith("hiddenpower"):
                return move_id
            if wanted == "return" and move_id == "return102":
                return move_id
        return None
    if candidate.get("kind") == "switch":
        species = normalize_id(str((candidate.get("pokemon") or {}).get("species") or ""))
        for party_species in side_species:
            if normalize_id(party_species) == species:
                return normalize_id(party_species)
        return None
    return None


# ---------------------------------------------------------------------------
# Branch <-> reality matching
# ---------------------------------------------------------------------------


def parse_condition(condition: str) -> tuple[int | None, int | None, str]:
    head, _, status = condition.partition(" ")
    status = status.strip()
    if status == "fnt" or head == "0":
        return 0, None, "none"
    current, _, maximum = head.partition("/")
    try:
        return int(current), int(maximum) if maximum else None, STATUS_CODES.get(status, status or "none")
    except ValueError:
        return None, None, "none"


def _action_actors(lines) -> list[str]:
    actors = []
    for line in lines:
        parts = line.split("|")
        if len(parts) > 2 and parts[1] in ("move", "cant"):
            actors.append(parts[2][:2])
    return actors


def realized_action_actors(slice_lines, turn: int) -> list[str]:
    """Actor sequence (p1/p2) of the action lines in the recorded slice's
    block for battle turn `turn` (the block ending at |turn|turn+1, or the
    trailing partial block)."""

    blocks: list[list[str]] = [[]]
    turn_of_block: dict[int, int] = {}
    for line in slice_lines:
        if line.startswith("|turn|"):
            try:
                number = int(line.split("|")[2])
            except (IndexError, ValueError):
                number = -1
            turn_of_block[number - 1] = len(blocks) - 1
            blocks.append([])
        else:
            blocks[-1].append(line)
    index = turn_of_block.get(turn)
    if index is None:
        index = len(blocks) - 1 if blocks[-1] else None
    if index is None:
        return []
    return _action_actors(blocks[index])


def candidate_action_actors(event_lines) -> list[str]:
    return _action_actors(event_lines)


def payload_side_target(payload: Mapping[str, Any], slot: str) -> dict[str, Any] | None:
    side = (payload.get("sides") or {}).get(slot)
    if not isinstance(side, Mapping):
        return None
    for row in side.get("pokemon") or ():
        if row.get("active"):
            hp, _maxhp, status = parse_condition(str(row.get("condition") or ""))
            boosts = {
                str(key): int(value)
                for key, value in (side.get("boosts") or {}).items()
                if int(value)
            }
            return {
                "species": normalize_id(str(row.get("species") or "")),
                "hp": hp,
                "status": status,
                "boosts": boosts,
            }
    return None


def branch_matches_target(
    branch_post: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any] | None],
    party_species: Mapping[str, list[str]],
    pre_hp: Mapping[str, int | None],
) -> bool:
    for slot in ("p1", "p2"):
        target = targets.get(slot)
        if target is None:
            continue
        post = branch_post.get(slot) or {}
        active_index = post.get("active_index", -1)
        species_list = party_species[slot]
        if not 0 <= active_index < len(species_list):
            return False
        if normalize_id(species_list[active_index]) != target["species"]:
            return False
        if target["hp"] is not None:
            engine_hp = post.get("active_hp", -1)
            if target["hp"] == 0:
                if engine_hp > 0:
                    return False
            else:
                # The engine collapses damage rolls to 0.925x the max roll;
                # the realized roll is in [0.85, 1.0]x, so the engine value
                # can be off by up to ~8.1% of the damage it modeled. Scale
                # the tolerance by the engine-implied HP delta.
                previous = pre_hp.get(slot)
                allowed = 2
                if previous is not None:
                    allowed = max(2, int(0.09 * abs(previous - engine_hp)) + 2)
                if abs(engine_hp - target["hp"]) > allowed:
                    return False
        if target["hp"] not in (None, 0):
            if post.get("active_status", "none") != target["status"]:
                return False
            # Boost stages disambiguate invisible secondaries (e.g. Psychic's
            # SpD drop) between otherwise HP-identical branches. Only checked
            # for a surviving active (boosts reset on faint/switch).
            engine_boosts = {
                key: int(value)
                for key, value in (post.get("boosts") or {}).items()
                if int(value)
            }
            if engine_boosts != (target.get("boosts") or {}):
                return False
    return True


# ---------------------------------------------------------------------------
# Product comparison (a/b/c classification)
# ---------------------------------------------------------------------------


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def classify_products(ours: Mapping[str, Any], recorded: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Returns (class, reasons): 'a' byte-identical, 'b' documented
    equivalences only, 'c' real divergence."""

    if canonical(ours) == canonical(recorded):
        return "a", []
    reasons: list[str] = []
    hard = False

    if ours.get("transition_token_total") != recorded.get("transition_token_total"):
        return "c", [
            "token_total: %s != %s"
            % (ours.get("transition_token_total"), recorded.get("transition_token_total"))
        ]
    ours_tokens = ours.get("transition_tokens") or []
    rec_tokens = recorded.get("transition_tokens") or []
    if len(ours_tokens) != len(rec_tokens):
        return "c", ["token_tail_len: %d != %d" % (len(ours_tokens), len(rec_tokens))]

    for position, (mine, real) in enumerate(zip(ours_tokens, rec_tokens)):
        keys = set(mine) | set(real)
        for key in sorted(keys):
            a, b = mine.get(key), real.get(key)
            if a == b:
                continue
            if key in FLOAT_FIELDS and float_equiv(key, a, b):
                reasons.append(f"tok{position}.{key}: float_rounding {a}~{b}")
                continue
            reasons.append(f"tok{position}.{key}: {a!r} != {b!r}")
            hard = True

    for key in ("tendency_stats", "cb_pinned_species", "investment_pinned"):
        if canonical(ours.get(key)) != canonical(recorded.get(key)):
            reasons.append(f"{key} differs")
            hard = True

    # turn_merged divergences are derived from the per-action stream; only
    # flag them when the per-action stream itself was clean (they then carry
    # independent information).
    if not hard and canonical(ours.get("turn_merged_tokens")) != canonical(
        recorded.get("turn_merged_tokens")
    ):
        merged_hard = False
        ours_merged = ours.get("turn_merged_tokens") or []
        rec_merged = recorded.get("turn_merged_tokens") or []
        if len(ours_merged) != len(rec_merged):
            merged_hard = True
            reasons.append("merged_tail_len differs")
        else:
            for position, (mine, real) in enumerate(zip(ours_merged, rec_merged)):
                if canonical(mine) == canonical(real):
                    continue
                flat_a, flat_b = _flatten(mine), _flatten(real)
                for key in sorted(set(flat_a) | set(flat_b)):
                    a, b = flat_a.get(key), flat_b.get(key)
                    if a == b:
                        continue
                    leaf = key.rsplit(".", 1)[-1]
                    if leaf in FLOAT_FIELDS and float_equiv(leaf, a, b):
                        reasons.append(f"merged{position}.{key}: float_rounding")
                        continue
                    reasons.append(f"merged{position}.{key}: {a!r} != {b!r}")
                    merged_hard = True
        hard = hard or merged_hard

    return ("c" if hard else "b"), reasons


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            flat.update(_flatten(value, f"{prefix}{index}."))
    else:
        flat[prefix[:-1] if prefix.endswith(".") else prefix] = payload
    return flat


# ---------------------------------------------------------------------------
# Boundary driver
# ---------------------------------------------------------------------------


class BoundaryResult:
    def __init__(self, status: str, detail: str = "") -> None:
        self.status = status  # 'a' | 'b' | 'c' | skip reason slug
        self.detail = detail


def truant_loaf_slots(
    history_lines: list[str],
    payload: Mapping[str, Any],
    teams: Mapping[str, tuple],
) -> list[str]:
    """Slots whose active Truant mon publicly acted in the last completed
    turn of the recorded history (so it loafs the coming turn)."""

    slots: list[str] = []
    slice_lines = history_lines
    turn_marks = [i for i, line in enumerate(slice_lines) if line.startswith("|turn|")]
    if len(turn_marks) >= 2:
        block = slice_lines[turn_marks[-2] : turn_marks[-1]]
    elif turn_marks:
        block = slice_lines[: turn_marks[-1]]
    else:
        block = slice_lines
    for slot in ("p1", "p2"):
        target = payload_side_target(payload, slot)
        if target is None:
            continue
        team = teams.get(slot) or ()
        active = next(
            (mon for mon in team if normalize_id(mon.species) == target["species"]),
            None,
        )
        if active is None or normalize_id(str(active.ability or "")) != "truant":
            continue
        moved = any(
            line.startswith(f"|move|{slot}a: ") for line in block
        )
        loafed = any(
            line.startswith(f"|cant|{slot}a: ") and "Truant" in line for line in block
        )
        if moved and not loafed:
            slots.append(slot)
    return slots


def drive_boundary(
    *,
    corpus: Mapping[str, Any],
    battle_id: str,
    seat: str,
    row_n: Mapping[str, Any],
    row_next: Mapping[str, Any],
    dex,
    history_lines: list[str],
    verbose: bool = False,
) -> BoundaryResult:
    games = corpus["games"]
    decisions = corpus["decisions"]
    game = games[battle_id]
    true_teams = game.get("true_teams") or {}
    packed = {slot: (true_teams.get(slot) or {}).get("packed") for slot in ("p1", "p2")}
    if not packed["p1"] or not packed["p2"]:
        return BoundaryResult("skip:no_true_teams")

    round_n = row_n["decision_round_index"]
    round_next = row_next["decision_round_index"]
    anchor = decisions.get((battle_id, round_n, seat))
    if anchor is None:
        return BoundaryResult("skip:no_anchor_row")
    payload = anchor.get("public_materialization")
    if not isinstance(payload, Mapping):
        return BoundaryResult("skip:no_payload")

    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    party_display = {slot: [mon.species for mon in teams[slot]] for slot in ("p1", "p2")}

    # Publicly-derivable engine flags (mirrors the live policy's derivation):
    # a "recharge" forced choice marks the slot as MUSTRECHARGE; a Truant mon
    # that publicly ACTED in the last completed turn loafs this turn.
    recharging = []
    for slot in ("p1", "p2"):
        row = decisions.get((battle_id, round_n, slot))
        candidate = chosen_candidate(row) if row is not None else None
        if (
            candidate is not None
            and candidate.get("kind") == "move"
            and normalize_id(str(candidate.get("move_id") or "")) == "recharge"
        ):
            recharging.append(slot)
    truant = truant_loaf_slots(history_lines, payload, teams)

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
        return BoundaryResult(f"skip:world_unsupported:{error.reason}")
    except Exception as error:  # noqa: BLE001
        return BoundaryResult("skip:world_error", str(error))
    # Engine move ids per side, in party order (for action mapping).
    def active_move_ids(row: Mapping[str, Any], slot: str) -> list[str]:
        # ENGINE move ids of the ACTING mon (poke-engine wants its own gen3
        # ids: typed+BP hidden power), resolved from the row's own payload
        # active species.
        active = payload_side_target(row.get("public_materialization") or {}, slot)
        mons = teams[slot]
        if active is not None:
            matching = [
                mon for mon in mons if normalize_id(mon.species) == active["species"]
            ]
            if matching:
                mons = matching
        moves = []
        for mon in mons:
            for move in mon.moves:
                move_id = normalize_id(move)
                if move_id.startswith("hiddenpower"):
                    try:
                        move_id = hidden_power_engine_id(move_id, mon.ivs)
                    except Exception:  # noqa: BLE001
                        pass
                moves.append(move_id)
        return moves

    state_str = state.to_string()
    turn = int(payload.get("turn") or 0)
    synthesized: list[str] = []
    lossy_tags: list[str] = []
    pre_hp: dict[str, int | None] = {
        slot: (payload_side_target(payload, slot) or {}).get("hp")
        for slot in ("p1", "p2")
    }

    for round_index in range(round_n, round_next):
        moves = {}
        for slot in ("p1", "p2"):
            row = decisions.get((battle_id, round_index, slot))
            if row is None:
                moves[slot] = "none"
                continue
            candidate = chosen_candidate(row)
            if candidate is None:
                return BoundaryResult("skip:no_candidate", f"round {round_index} {slot}")
            move = engine_move_string(candidate, active_move_ids(row, slot), party_display[slot])
            if move is None:
                return BoundaryResult(
                    "skip:action_unmapped",
                    f"round {round_index} {slot} {candidate.get('kind')}:{candidate.get('move_id') or (candidate.get('pokemon') or {}).get('species')}",
                )
            moves[slot] = move

        ctx = json.dumps({"p1": party_display["p1"], "p2": party_display["p2"], "turn": turn})
        try:
            report = json.loads(
                pokezero_search.branch_events(state_str, moves["p1"], moves["p2"], ctx, True, True)
            )
        except ValueError as error:
            return BoundaryResult("skip:branch_events_error", f"round {round_index}: {error}")

        # Target: the next round's recorded public payload (the boundary
        # seat's own row at the final round).
        target_payload = None
        for slot in (seat, "p1", "p2"):
            row = decisions.get((battle_id, round_index + 1, slot))
            if row is not None:
                target_payload = row.get("public_materialization")
                break
        if not isinstance(target_payload, Mapping):
            return BoundaryResult("skip:no_target_payload", f"round {round_index + 1}")
        targets = {slot: payload_side_target(target_payload, slot) for slot in ("p1", "p2")}

        matches = [
            branch
            for branch in report["branches"]
            if branch.get("post_state")
            and branch_matches_target(branch.get("post") or {}, targets, party_display, pre_hp)
        ]
        if not matches:
            return BoundaryResult("skip:no_branch_match", f"round {round_index}")
        if len(matches) > 1:
            # Post-state ties (speed ties, invisible secondaries): resolve on
            # the realized ACTION ORDER — the recorded slice's |move|/|cant|
            # actor sequence for this turn is part of the realized outcome.
            realized = realized_action_actors(row_next.get("event_slice") or (), turn)
            if realized:
                ordered = [
                    branch
                    for branch in matches
                    if candidate_action_actors(branch["events"]) == realized
                ]
                if ordered:
                    matches = ordered
        matches.sort(key=lambda b: -float(b.get("percentage") or 0.0))
        branch = matches[0]
        synthesized.extend(branch["events"])
        lossy_tags.extend(branch.get("lossy") or ())
        state_str = branch["post_state"]
        for slot in ("p1", "p2"):
            pre_hp[slot] = (branch.get("post") or {}).get(slot, {}).get("active_hp")
        if branch.get("turn_completed"):
            turn += 1

    if lossy_tags:
        return BoundaryResult("skip:lossy_render", ",".join(sorted(set(lossy_tags))))

    fold = pokezero_search.FoldState.from_payload(row_n["fold_state"])
    try:
        fold.advance_in_place(synthesized)
        overlay = row_next.get("annotation_overlay") or {}
        if overlay:
            fold.apply_annotations_in_place(
                {int(k): tuple(v) for k, v in overlay.items()}
            )
    except Exception as error:  # noqa: BLE001
        return BoundaryResult("c", f"advance_error: {error}")
    products = fold.products_payload()
    klass, reasons = classify_products(products, row_next["products"])
    if verbose and klass != "a":
        print(f"--- {battle_id} {seat} rounds {round_n}->{round_next} class={klass}")
        for reason in reasons[:12]:
            print("   ", reason)
    return BoundaryResult(klass, "; ".join(reasons[:8]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_corpus(corpus_dir: Path, verbose: bool) -> dict[str, Any]:
    corpus = load_corpus(corpus_dir)
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    dex = load_showdown_dex(DEFAULT_SHOWDOWN_ROOT)
    stats: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for (battle_id, seat), chain in sorted(corpus["fold_chains"].items()):
        history: list[str] = list(chain[0].get("event_slice") or ())
        for row_n, row_next in zip(chain, chain[1:]):
            result = drive_boundary(
                corpus=corpus,
                battle_id=battle_id,
                seat=seat,
                row_n=row_n,
                row_next=row_next,
                dex=dex,
                history_lines=list(history),
                verbose=verbose,
            )
            history.extend(row_next.get("event_slice") or ())
            stats[result.status] += 1
            if result.status not in ("a",):
                details.append(
                    {
                        "battle_id": battle_id,
                        "seat": seat,
                        "rounds": [row_n["decision_round_index"], row_next["decision_round_index"]],
                        "status": result.status,
                        "detail": result.detail,
                    }
                )
    total_pairs = sum(len(chain) - 1 for chain in corpus["fold_chains"].values() if len(chain) > 1)
    return {
        "corpus": str(corpus_dir),
        "row_pair_boundaries": total_pairs,
        "counts": dict(sorted(stats.items())),
        "non_a": details,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    reports = []
    for corpus_dir in args.corpus:
        report = run_corpus(corpus_dir, args.verbose)
        reports.append(report)
        print(f"== {corpus_dir}")
        print(f"   row-pair boundaries: {report['row_pair_boundaries']}")
        for key, value in report["counts"].items():
            print(f"   {key:40s} {value}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
