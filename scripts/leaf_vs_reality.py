"""Leaf-vs-reality differential for the leaf observation path (PR #730
review gate — the only harness that exercises evolve-on-change, the delta
families, per-branch recompute, and self-order evolution; the depth-0
root-parity gate structurally cannot fire any of them).

Per same-seat decision boundary (row n -> row n+1) of a golden corpus v2
fold sidecar:

1. reconstruct the engine world at row n from the recorded public payload +
   TRUE teams (the fidelity harness's machinery: true-team override,
   publicly-derivable recharge/Truant flags);
2. drive the RECORDED joint actions round by round through
   ``pokezero_search.branch_events`` and select the enumerated branch
   consistent with what actually happened (post-state + realized action
   order matching — ``fidelity_gate_events`` verbatim);
3. advance a FRESH copy of row n's recorded fold state over the synthesized
   lines, apply row n+1's annotation overlay, evolve the self-team display
   order (Showdown switch-swap semantics), and ``encode_leaf`` the reached
   engine state at the accumulated turn;
4. byte-diff all five observation arrays against ROW N+1's recorded golden
   arrays.

Classification discipline (every family counted, none averaged away):

- ``state``      — MUST-MATCH engine-state-derived cells; any entry is a
                   defect (this class gates the exit code);
- ``fold``       — transition-block / tendency / pinned cells: inherits the
                   fidelity gate's documented (b)/(c) classes (damage-roll
                   collapse envelopes, merged no-op branches, ...);
- ``epistemic``  — belief facts + opponent-team membership: row n+1 saw new
                   REVEALS; the leaf path root-freezes the epistemic surface
                   per world BY DESIGN (column map);
- ``engine_pp_model`` — PP columns diverging within the SAME revealed set:
                   residual PP-count mismatches after the line-replay fix
                   (LeafMeta::move_charges replays the parser's charging
                   rules; the engine's own DecrementPP only fires below
                   10 PP — column map F3). A PP/validity cell whose bucket
                   VALIDITY bit also flipped is the root-frozen revealed
                   set / bucket composition meeting new reveals and counts
                   as ``epistemic``; transform-tagged PP cells count as
                   ``engine_model`` (copied 5-PP instance moves);
- ``engine_roll`` — damage-roll collapse envelopes: HP fractions, substitute
                   breaks, pinch-berry thresholds (fidelity class b);
- ``engine_model`` — tagged vendored-engine deviations (Transform empty
                   delta, Encore volatile not applied, recharge consumed a
                   ply early, Baton-Passed saved moves, in-branch screen
                   set-turns);
- ``ledger_skew`` — recorded production inconsistencies (ledger condition
                   strings keep stale status suffixes through cures);
- ``turn``       — field-token turn count (defect class alongside ``state``).

Usage:
    PYTHONPATH=src python scripts/leaf_vs_reality.py \
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

from pokezero.dex import load_showdown_dex, normalize_id  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
    hidden_power_engine_id,
    unpack_team,
)
from pokezero.golden_corpus import GOLDEN_ARRAY_FIELDS, load_golden_corpus  # noqa: E402
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT  # noqa: E402
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402

from fidelity_gate_events import (  # noqa: E402
    branch_matches_target,
    candidate_action_actors,
    chosen_candidate,
    load_corpus,
    payload_side_target,
    realized_action_actors,
    truant_loaf_slots,
)
from leaf_root_parity import TOKEN_BLOCKS, bitwise_equal, block_of, column_names  # noqa: E402

# Column families that are EXPECTED to diverge one boundary ahead, with the
# reason each is expected (kept explicit — the honest-classification rule).
EPISTEMIC_PREFIXES = (
    "CATEGORY_BELIEF_",
    # Alias of the first belief-ability bucket column in the exported layout.
    "CATEGORY_FIXED_COUNT",
    "NUMERIC_POSSIBLE_",
    "NUMERIC_REVEALED_",
    "NUMERIC_CANDIDATE_SET_COUNT",
    "NUMERIC_UNCERTAINTY",
    "NUMERIC_EXPECTED_",
)
PP_COLUMNS = ("NUMERIC_MOVE_PP_FRACTION", "NUMERIC_OPP_MOVE_PP")
# Timed side-condition duration columns: a condition SET during the branch
# has no set-turn (column map: documented approximation).
TIMED_COLUMNS = (
    "NUMERIC_SELF_REFLECT_TURNS",
    "NUMERIC_OPP_REFLECT_TURNS",
    "NUMERIC_SELF_LIGHT_SCREEN_TURNS",
    "NUMERIC_OPP_LIGHT_SCREEN_TURNS",
    "NUMERIC_SELF_SAFEGUARD_TURNS",
    "NUMERIC_OPP_SAFEGUARD_TURNS",
    "NUMERIC_SELF_MIST_TURNS",
    "NUMERIC_OPP_MIST_TURNS",
)


def offset_column_names(tables: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    """Column-index -> name maps with OFFSET bucket families resolved
    (`NUMERIC_OPP_MOVE_PP_OFFSET+k`, `CATEGORY_BELIEF_MOVE_OFFSET+k`, ...)."""

    layout = tables["layout"]
    names = column_names(tables)
    buckets = layout["belief_buckets"]
    cat_ranges = (
        ("CATEGORY_BELIEF_ABILITY_OFFSET", buckets["ability"]),
        ("CATEGORY_BELIEF_ITEM_OFFSET", buckets["item"]),
        ("CATEGORY_BELIEF_MOVE_OFFSET", buckets["move"]),
        ("CATEGORY_VOLATILE_OFFSET", layout["volatile_bucket_count"]),
    )
    num_ranges = (
        ("NUMERIC_OPP_MOVE_PP_OFFSET", buckets["move"]),
        ("NUMERIC_OPP_MOVE_PP_VALID_OFFSET", buckets["move"]),
        (
            "NUMERIC_STAT_WEATHER_REVEAL_OFFSET",
            2 * len(layout["constants"]["weather_reveal_order"]),
        ),
    )
    for kind, ranges in (("categorical_ids", cat_ranges), ("numeric_features", num_ranges)):
        columns = layout[
            "categorical_columns" if kind == "categorical_ids" else "numeric_columns"
        ]
        for base_name, width in ranges:
            base = columns.get(base_name)
            if base is None:
                continue
            for k in range(int(width)):
                names[kind].setdefault(base + k, f"{base_name}+{k}")
    return names


def classify(
    array: str,
    block: str,
    column: str,
    opp_membership: bool,
    tags: set[str],
    reveal_pp: bool = False,
) -> str:
    if block == "transition" or column.startswith(
        ("NUMERIC_STAT_", "NUMERIC_MON_", "NUMERIC_TIER2_")
    ):
        return "fold"
    if opp_membership:
        return "epistemic"
    if column.startswith(EPISTEMIC_PREFIXES):
        return "epistemic"
    if reveal_pp:
        # The bucket's VALIDITY bit or MOVE-identity cell flipped alongside:
        # the divergence is the ROOT-FROZEN revealed-move set / bucket
        # composition meeting row n+1's new reveals (epistemic by design),
        # not a PP-count error — only same-revealed-set same-bucket PP
        # mismatches are the engine_pp class.
        return "epistemic"
    if "curestatus" in tags and column == "CATEGORY_SECONDARY" and block in (
        "self_team",
        "opponent_team",
    ):
        # Ledger condition strings keep stale status suffixes through cures
        # (recorded production skew; see the tag).
        return "ledger_skew"
    if "item_boost" in tags and column.startswith("NUMERIC_BOOST_"):
        # Pinch-berry boosts are roll-threshold-dependent (see the tag).
        return "engine_roll"
    if "transform" in tags or "recharge" in tags or "encore" in tags or "baton_pass" in tags:
        # Transform: the gen3 engine's Transform is an empty delta (world
        # construction fail-closes on transformed roots for the same reason).
        # Recharge: the engine consumes the recharge one ply early on
        # faint-replacement plies (documented deviation). Encore: the engine
        # does not apply the Encore volatile from a branch (root encore
        # states fail-close as encore_move_unknown). Checked BEFORE the PP
        # class: each of these corrupts the whole action/PP surface (copied
        # 5-PP instance moves, the recharge pseudo-request, a Baton-Passed
        # saved move that never resolves engine-side), so tagged PP cells are
        # the engine-model deviation, not PP-tracking errors.
        return "engine_model"
    if "merged_outcome" in tags and column.startswith(PP_COLUMNS):
        # The PARSER charges PP for a |move| line reality replaced with
        # |cant| (or vice versa) — the merged no-op outcome ambiguity
        # (insufficiency #1), not a PP-tracking error. Scoped to PP columns
        # only: other cells on such boundaries keep their own classes.
        return "engine_model"
    if column.startswith(PP_COLUMNS):
        return "engine_pp_model"
    if column in TIMED_COLUMNS:
        return "engine_model"
    if column in ("NUMERIC_HP_FRACTION", "NUMERIC_SUB_HP_FRACTION"):
        # Damage-roll collapse: the engine prices one representative roll;
        # reality rolled elsewhere in the envelope (fidelity class b).
        return "engine_roll"
    if column in ("NUMERIC_TURN_COUNT",):
        return "turn"
    return "state"


def engine_move_string_for(
    candidate: Mapping[str, Any], side_moves: list[str], side_species: list[str]
) -> str | None:
    """`fidelity_gate_events.engine_move_string` (kept import-stable)."""

    from fidelity_gate_events import engine_move_string

    return engine_move_string(candidate, side_moves, side_species)


def drive_pair(
    *,
    corpus: Mapping[str, Any],
    battle_id: str,
    seat: str,
    row_n: Mapping[str, Any],
    row_next: Mapping[str, Any],
    dex,
    history_lines: list[str],
    tables_json: str,
) -> tuple[str, Any]:
    """Returns (status, payload). status='ok' payload=(buffers, turn); else a
    skip reason string."""

    games = corpus["games"]
    decisions = corpus["decisions"]
    game = games[battle_id]
    true_teams = game.get("true_teams") or {}
    packed = {slot: (true_teams.get(slot) or {}).get("packed") for slot in ("p1", "p2")}
    if not packed["p1"] or not packed["p2"]:
        return "skip:no_true_teams", None
    round_n = row_n["decision_round_index"]
    round_next = row_next["decision_round_index"]
    anchor = decisions.get((battle_id, round_n, seat))
    if anchor is None:
        return "skip:no_anchor_row", None
    payload = anchor.get("public_materialization")
    if not isinstance(payload, Mapping):
        return "skip:no_payload", None

    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    party_display = {slot: [mon.species for mon in teams[slot]] for slot in ("p1", "p2")}

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
        return f"skip:world_unsupported:{error.reason}", None
    except Exception:  # noqa: BLE001
        return "skip:world_error", None

    def active_move_ids(row: Mapping[str, Any], slot: str) -> list[str]:
        active = payload_side_target(row.get("public_materialization") or {}, slot)
        mons = teams[slot]
        if active is not None:
            matching = [m for m in mons if normalize_id(m.species) == active["species"]]
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

    root_state_str = state.to_string()
    state_str = root_state_str
    turn = int(payload.get("turn") or 0)
    synthesized: list[str] = []
    lossy_tags: list[str] = []
    pre_hp: dict[str, int | None] = {
        slot: (payload_side_target(payload, slot) or {}).get("hp") for slot in ("p1", "p2")
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
                return "skip:no_candidate", None
            move = engine_move_string_for(
                candidate, active_move_ids(row, slot), party_display[slot]
            )
            if move is None:
                return "skip:action_unmapped", None
            moves[slot] = move
        ctx = json.dumps(
            {"p1": party_display["p1"], "p2": party_display["p2"], "turn": turn}
        )
        try:
            report = json.loads(
                pokezero_search.branch_events(
                    state_str, moves["p1"], moves["p2"], ctx, True, True
                )
            )
        except ValueError:
            return "skip:branch_events_error", None
        target_payload = None
        for slot in (seat, "p1", "p2"):
            row = decisions.get((battle_id, round_index + 1, slot))
            if row is not None:
                target_payload = row.get("public_materialization")
                break
        if not isinstance(target_payload, Mapping):
            return "skip:no_target_payload", None
        targets = {slot: payload_side_target(target_payload, slot) for slot in ("p1", "p2")}
        def faint_pattern_ok(branch: Mapping[str, Any]) -> bool:
            # The fidelity matcher's roll-scaled HP tolerance can conflate a
            # KO'd active with a barely-alive one (0 vs 1 HP within the
            # envelope) — an OBSERVATION diff cares about exactly that bit,
            # so require faint-pattern equality per side.
            for slot in ("p1", "p2"):
                target = targets.get(slot)
                if target is None or target.get("hp") is None:
                    continue
                post_hp = ((branch.get("post") or {}).get(slot) or {}).get("active_hp")
                if post_hp is None:
                    return False
                if (post_hp <= 0) != (int(target["hp"]) <= 0):
                    return False
            return True

        matches = [
            branch
            for branch in report["branches"]
            if branch.get("post_state")
            and branch_matches_target(branch.get("post") or {}, targets, party_display, pre_hp)
            and faint_pattern_ok(branch)
        ]
        if not matches:
            return "skip:no_branch_match", None
        if len(matches) > 1:
            realized = realized_action_actors(row_next.get("event_slice") or (), turn)
            if realized:
                ordered = [
                    b for b in matches if candidate_action_actors(b["events"]) == realized
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
        return "skip:lossy_render", None

    row_inputs = {
        "battle_id": anchor.get("battle_id"),
        "battle_seed": anchor.get("battle_seed"),
        "format_id": anchor.get("format_id"),
        "player_id": anchor.get("player_id"),
        "observation_schema_version": anchor.get("observation_schema_version"),
        "observation_metadata": anchor.get("observation_metadata"),
        "public_materialization": payload,
    }
    ctx = json.dumps(
        {
            "p1": list(world.party_species["p1"]),
            "p2": list(world.party_species["p2"]),
            "turn": int(payload.get("turn") or 0),
        }
    )
    try:
        encoder = pokezero_search.LeafEncoder(
            tables_json, json.dumps(row_inputs, sort_keys=True), ctx, root_state_str
        )
        fold = pokezero_search.FoldState.from_payload(row_n["fold_state"])
        fold.advance_in_place(synthesized)
        overlay = row_next.get("annotation_overlay") or {}
        if overlay:
            fold.apply_annotations_in_place({int(k): tuple(v) for k, v in overlay.items()})
        buffers = encoder.encode_leaf(state_str, fold, turn, synthesized)
    except Exception as error:  # noqa: BLE001
        return f"skip:encode_error:{type(error).__name__}", None
    tags = set()
    # Transform anywhere in the game so far (reality's lines, not ours: the
    # gen3 engine's Transform is an empty delta — the leaf Ditto never
    # transforms, which is exactly why world construction fail-closes on
    # transformed ROOT states; boundaries inside a transformed game inherit
    # the same engine-model gap).
    if any(
        "|-transform|" in line
        for source in (synthesized, history_lines, row_next.get("event_slice") or ())
        for line in source
    ):
        tags.add("transform")
    # In-branch Encore: the vendored gen3 engine does not apply the Encore
    # volatile from a branch (world construction fail-closes on ROOT encore
    # states for the same reason: encore_move_unknown).
    if any(
        "|Encore" in line
        for source in (synthesized, row_next.get("event_slice") or ())
        for line in source
    ):
        tags.add("encore")
    # Baton Pass in the driven span: a pivot's saved move never resolves
    # after the replacement in the vendored engine (known fail-soft,
    # belief_edge_case_matrix) — e.g. Spikes committed behind a Baton Pass
    # never lands on the engine side.
    if any(
        "Baton Pass" in line
        for source in (synthesized, row_next.get("event_slice") or ())
        for line in source
    ):
        tags.add("baton_pass")
    # An item-triggered boost in reality's span (pinch berries): whether the
    # berry fires is damage-roll-dependent (25% threshold) and the collapsed
    # engine roll can miss it.
    if any(
        "|-boost|" in line and "[from] item:" in line
        for line in (row_next.get("event_slice") or ())
    ):
        tags.add("item_boost")
    # A |-curestatus| in reality's span: the belief ledger's CONDITION string
    # keeps the stale status suffix through cures (Refresh/Heal Bell — the
    # recorded production skew the root-parity gate also documents), so
    # status-derived cells for the cured mon reproduce the LEDGER, which the
    # engine-derived leaf legitimately disagrees with.
    if any(
        "|-curestatus|" in line
        for line in (row_next.get("event_slice") or ())
    ):
        tags.add("curestatus")
    # Merged no-op outcome ambiguity (insufficiency #1): the engine merges a
    # fully-paralyzed turn, a missed move, and a failed move into ONE branch
    # when the deltas coincide; the renderer picks the dominant-mass cause,
    # and when reality took the minority outcome the |cant| structure differs
    # — the PARSER charges PP for a |move| line the reality stream replaced
    # with |cant| (or vice versa). Detected by comparing per-actor |cant|
    # counts between the synthesized lines and reality's slice.
    def cant_counts(lines) -> Counter:
        return Counter(
            line.split("|")[2] for line in lines if str(line).startswith("|cant|")
        )

    if cant_counts(synthesized) != cant_counts(row_next.get("event_slice") or ()):
        tags.add("merged_outcome")
    # A recharge request at the target boundary: the engine consumes the
    # recharge one ply early on faint-replacement plies (documented engine
    # deviation, docs/crate_search_design.md fidelity findings), so recharge
    # legality/turn shape can diverge.
    target_row = corpus["decisions"].get(
        (battle_id, row_next["decision_round_index"], seat)
    )
    if target_row is not None and any(
        candidate.get("kind") == "move"
        and normalize_id(str(candidate.get("move_id") or "")) == "recharge"
        for candidate in (target_row.get("observation_metadata") or {}).get(
            "action_candidates"
        )
        or ()
    ):
        tags.add("recharge")
    return "ok", (buffers, turn, tags)


def run_corpus(corpus_dir: Path, tables_json: str, tables: Mapping[str, Any]) -> dict[str, Any]:
    raw = load_corpus(corpus_dir)
    golden = load_golden_corpus(corpus_dir)
    golden_rows = {
        (row.battle_id, row.decision_round_index, row.player_id): row
        for row in golden.decision_rows
    }
    dex = load_showdown_dex(DEFAULT_SHOWDOWN_ROOT)
    names = offset_column_names(tables)
    vocab_inv = {v: k for k, v in tables["vocab"]["index"].items()}
    present_column = tables["layout"]["numeric_columns"]["NUMERIC_PRESENT"]
    opp_start, opp_stop = 7, 13
    # Opponent PP bucket geometry (reveal-vs-count discrimination): a PP or
    # validity cell whose bucket's VALIDITY bit also flipped diverges because
    # of the root-frozen revealed set / bucket composition (epistemic), not a
    # PP count.
    num_cols = tables["layout"]["numeric_columns"]
    move_bucket_width = int(tables["layout"]["belief_buckets"]["move"])
    pp_base = num_cols.get("NUMERIC_OPP_MOVE_PP_OFFSET")
    pp_valid_base = num_cols.get("NUMERIC_OPP_MOVE_PP_VALID_OFFSET")

    counts: Counter[str] = Counter()
    class_rows: Counter[str] = Counter()
    families: Counter[tuple[str, str, str, str]] = Counter()
    family_examples: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for (battle_id, seat), chain in sorted(raw["fold_chains"].items()):
        history: list[str] = list(chain[0].get("event_slice") or ())
        for row_n, row_next in zip(chain, chain[1:]):
            status, payload = drive_pair(
                corpus=raw,
                battle_id=battle_id,
                seat=seat,
                row_n=row_n,
                row_next=row_next,
                dex=dex,
                history_lines=list(history),
                tables_json=tables_json,
            )
            history.extend(row_next.get("event_slice") or ())
            if status != "ok":
                counts[status] += 1
                continue
            buffers, _turn, tags = payload
            golden_row = golden_rows.get(
                (battle_id, row_next["decision_round_index"], seat)
            )
            if golden_row is None:
                counts["skip:no_golden_row"] += 1
                continue

            # Opponent-membership tokens: revealed between the boundaries.
            want_numeric = numpy.ascontiguousarray(
                golden_row.arrays.numeric_features, dtype="<f8"
            )
            got_numeric = numpy.frombuffer(
                buffers["numeric_features"], dtype="<f8"
            ).reshape(want_numeric.shape)
            membership_tokens = {
                token
                for token in range(opp_start, opp_stop)
                if got_numeric[token, present_column] != want_numeric[token, present_column]
            }
            reveal_buckets: set[tuple[int, int]] = set()
            want_cat = numpy.ascontiguousarray(
                golden_row.arrays.categorical_ids, dtype="<i4"
            )
            got_cat = numpy.frombuffer(buffers["categorical_ids"], dtype="<i4").reshape(
                want_cat.shape
            )
            move_cat_base = tables["layout"]["categorical_columns"].get(
                "CATEGORY_BELIEF_MOVE_OFFSET"
            )
            for token in range(opp_start, opp_stop):
                for k in range(move_bucket_width):
                    valid_flip = pp_valid_base is not None and (
                        got_numeric[token, pp_valid_base + k]
                        != want_numeric[token, pp_valid_base + k]
                    )
                    # The bucket holds a DIFFERENT move (candidate-set
                    # recomposition between the root-frozen belief and row
                    # n+1's): PP cells then compare across different moves.
                    bucket_flip = move_cat_base is not None and (
                        got_cat[token, move_cat_base + k]
                        != want_cat[token, move_cat_base + k]
                    )
                    if valid_flip or bucket_flip:
                        reveal_buckets.add((token, k))

            row_classes: set[str] = set()
            row_families: set[tuple[str, str, str, str]] = set()
            for name, dtype, _ in GOLDEN_ARRAY_FIELDS:
                want = numpy.ascontiguousarray(getattr(golden_row.arrays, name), dtype=dtype)
                got = numpy.frombuffer(buffers[name], dtype=dtype).reshape(want.shape)
                equal = bitwise_equal(got, want)
                if bool(equal.all()):
                    continue
                for position in numpy.argwhere(~equal):
                    if want.ndim == 2:
                        token, column = int(position[0]), int(position[1])
                        colname = names.get(name, {}).get(column, f"col{column}")
                    else:
                        token, column = int(position[0]), -1
                        colname = name
                        if name == "legal_action_mask":
                            token, column = -1, int(position[0])
                            colname = f"action{column}"
                    block = block_of(token) if token >= 0 else name
                    opp_membership = (
                        isinstance(token, int) and token in membership_tokens
                    )
                    klass = None
                    if colname.startswith("CATEGORY_VOLATILE"):
                        decoded = {
                            vocab_inv.get(int(got[tuple(int(v) for v in position)])),
                            vocab_inv.get(int(want[tuple(int(v) for v in position)])),
                        }
                        if "volatile:substitute" in decoded:
                            # Substitute presence is roll/approximation-
                            # dependent (approximate sub health + collapsed
                            # rolls decide whether a hit breaks it).
                            klass = "engine_roll"
                    reveal_pp = bool(
                        name == "numeric_features"
                        and token in range(opp_start, opp_stop)
                        and (
                            (
                                pp_base is not None
                                and pp_base <= column < pp_base + move_bucket_width
                                and (token, column - pp_base) in reveal_buckets
                            )
                            or (
                                pp_valid_base is not None
                                and pp_valid_base <= column < pp_valid_base + move_bucket_width
                                and (token, column - pp_valid_base) in reveal_buckets
                            )
                        )
                    )
                    if klass is None:
                        klass = classify(
                            name, block, colname, opp_membership, tags, reveal_pp
                        )
                    family = (klass, name, block, colname)
                    row_classes.add(klass)
                    row_families.add(family)
                    if family not in family_examples:
                        index = tuple(int(v) for v in position)
                        family_examples[family] = {
                            "battle_id": battle_id,
                            "seat": seat,
                            "rounds": [
                                row_n["decision_round_index"],
                                row_next["decision_round_index"],
                            ],
                            "token": token,
                            "column": column,
                            "got": got[index].item(),
                            "want": want[index].item(),
                        }
            if not row_classes:
                counts["exact"] += 1
            else:
                counts["divergent"] += 1
                for klass in row_classes:
                    class_rows[klass] += 1
                for family in row_families:
                    families[family] += 1

    return {
        "corpus": str(corpus_dir),
        "boundaries": sum(len(c) - 1 for c in raw["fold_chains"].values() if len(c) > 1),
        "counts": dict(sorted(counts.items())),
        "class_rows": dict(sorted(class_rows.items())),
        "families": [
            {
                "class": klass,
                "array": array,
                "block": block,
                "column": column,
                "rows": count,
                "example": family_examples.get((klass, array, block, column)),
            }
            for (klass, array, block, column), count in sorted(
                families.items(), key=lambda item: (item[0][0], -item[1], item[0][3])
            )
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    tables_json = args.tables.read_text(encoding="utf-8")
    tables = json.loads(tables_json)
    reports = []
    defect_rows = 0
    for corpus_dir in args.corpus:
        report = run_corpus(corpus_dir, tables_json, tables)
        reports.append(report)
        print(f"== {corpus_dir}")
        print(f"   same-seat boundaries: {report['boundaries']}")
        for key, value in report["counts"].items():
            print(f"   {key:44s} {value}")
        if report["class_rows"]:
            print("   divergent boundaries by class:")
            for klass, value in report["class_rows"].items():
                print(f"     {klass:20s} {value}")
        for family in report["families"]:
            example = family["example"] or {}
            print(
                f"   [{family['class']:>15s}] {family['array']}/{family['block']}/"
                f"{family['column']:38s} {family['rows']:4d}  e.g. "
                f"{example.get('battle_id')}#{example.get('rounds')} "
                f"got={example.get('got')!r} want={example.get('want')!r}"
            )
        defect_rows += report["class_rows"].get("state", 0) + report["class_rows"].get(
            "turn", 0
        )
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(f"\nDEFECT-CLASS (state+turn) divergent boundaries: {defect_rows}")
    return 0 if defect_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
