"""Extract the fully-decoded v2.2 (turn-merged) observation dump at the |turn|16
boundary of the seed-148 explosion fixture.

Mirrors tests/test_corpus_replay.py's replay loop exactly (line-by-line feed,
incremental belief-engine ingest), belief-on (POKEZERO_BELIEF_SET_SOURCE=1, set source
attached), spec = V2_2_REPLAY_OBSERVATION_SPEC with the turn-merged vocabulary, ALL
feature masks on (stats / exact-state / K=128 / tier2_residuals / tier2_investment).
Unlike the v2 turn-10 doc's game, this fixture carries REAL |request| lines at every
decision, so the boundary request is the committed server request — no synthesis.

The dump is DETERMINISTIC (sorted belief sets, rounded floats, no wall-clock or commit
stamps): tests/test_token_format_doc.py regenerates it and asserts byte-identity with
the committed docs/token-format/turn16-token-dump.json, so the documentation can never
silently drift from the encoder.

Run from anywhere:  uv run python docs/token-format/extract_turn16.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POKEZERO_BELIEF_SET_SOURCE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "turn16-token-dump.json"
SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)
BOUNDARY_TURN = 16

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from test_explosion_fixture import MANIFEST, load_explosion_fixture_lines  # noqa: E402

import pokezero.showdown as sd  # noqa: E402
from pokezero.belief import PublicBattleBeliefEngine  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.observation import (  # noqa: E402
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_COUNT,
    STATS_TOKEN_COUNT,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
)
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.randbat_vocab import gen3_category_vocabulary  # noqa: E402
from pokezero.showdown import (  # noqa: E402
    ACTION_CANDIDATE_TOKEN_OFFSET,
    FIELD_TOKEN_OFFSET,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_OFFSET,
    STATS_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    V2_1_REPLAY_OBSERVATION_SPEC,
    V2_2_REPLAY_OBSERVATION_SPEC,
    V2_REPLAY_OBSERVATION_SPEC,
    _ReplayParser,
    normalize_for_player,
    observation_from_player_state,
)
from pokezero.turn_merged import (  # noqa: E402
    PHASE_REPLACEMENT,
    PHASE_TURN,
    SUB_BLOCK_ACTION,
    SUB_BLOCK_NEGATED,
    SUB_BLOCK_PENDING,
)

spec = V2_2_REPLAY_OBSERVATION_SPEC

# ------------------------------------------------------------ numeric index -> name map
# Built programmatically from the module's NUMERIC_* constants. The three *_OFFSET
# constants are range bases (opp-move PP ledger + the v2.1 PP-validity twin: 16
# belief-move buckets each; stats-token weather reveals: 4 weathers x 2 bits); in-range
# columns get base_name+k.
numeric_consts = {
    name: value
    for name, value in vars(sd).items()
    if name.startswith("NUMERIC_") and isinstance(value, int)
}
RANGE_BASES = {
    "NUMERIC_OPP_MOVE_PP_OFFSET": sd.BELIEF_MOVE_BUCKET_COUNT,  # 76..91
    "NUMERIC_STAT_WEATHER_REVEAL_OFFSET": 8,  # 97..104
    "NUMERIC_OPP_MOVE_PP_VALID_OFFSET": sd.BELIEF_MOVE_BUCKET_COUNT,  # 121..136
}
index_to_name: dict[int, str] = {}
for name, value in numeric_consts.items():
    if name in RANGE_BASES:
        for k in range(RANGE_BASES[name]):
            index_to_name[value + k] = name if k == 0 else f"{name}+{k}"
for name, value in numeric_consts.items():
    if name not in RANGE_BASES:
        index_to_name[value] = name  # plain constants win over range-extension collisions
assert sorted(index_to_name) == list(range(spec.numeric_feature_count)), (
    f"numeric name map must cover 0..{spec.numeric_feature_count - 1} exactly"
)
NUMERIC_NAMES = [index_to_name[i] for i in range(spec.numeric_feature_count)]

# -------------------------------------------------------- categorical slot layout names
_CAT_RANGE_BASES = (
    ("CATEGORY_BELIEF_ABILITY_OFFSET", sd.BELIEF_ABILITY_BUCKET_COUNT),
    ("CATEGORY_BELIEF_ITEM_OFFSET", sd.BELIEF_ITEM_BUCKET_COUNT),
    ("CATEGORY_BELIEF_MOVE_OFFSET", sd.BELIEF_MOVE_BUCKET_COUNT),
    ("CATEGORY_VOLATILE_OFFSET", sd.VOLATILE_BUCKET_COUNT),
)
_CAT_EXCLUDE = {"CATEGORY_ID_BUCKETS", "CATEGORY_FIXED_COUNT"} | {n for n, _ in _CAT_RANGE_BASES}
cat_slot_names: dict[int, str] = {}
for base_name, count in _CAT_RANGE_BASES:
    base = getattr(sd, base_name)
    for k in range(count):
        cat_slot_names[base + k] = base_name if k == 0 else f"{base_name}+{k}"
for name, value in vars(sd).items():
    if name.startswith("CATEGORY_") and isinstance(value, int) and name not in _CAT_EXCLUDE:
        cat_slot_names[value] = name  # plain columns (fixed 9 + the 12 TM columns) win
assert sorted(cat_slot_names) == list(range(spec.categorical_feature_count)), (
    f"cat slot map must cover 0..{spec.categorical_feature_count - 1} exactly"
)
CAT_NAMES = [cat_slot_names[i] for i in range(spec.categorical_feature_count)]

# ------------------------------------------------------------------------------ resources
dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
vocab = gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)
set_source = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
# ALL masks on: stats + exact-state + full K + BOTH tier2 provenance switches. The
# investment switch defaults False; flipping it here documents the full v2.2 surface
# (the columns still read 0.0 on this path — see the masks section note).
masks = ObservationFeatureMasks(tier2_investment=True)

id_to_token = {row: token for row, token in enumerate(vocab.tokens, start=1)}

manifest = json.loads(MANIFEST.read_text())
all_lines = [line for line in load_explosion_fixture_lines() if line]

# --------------------------------------------------------------- replay to the boundary
# Identical loop shape to tests/test_corpus_replay.py: feed one line at a time, ingest
# newly-parsed public events into the persistent belief engine. The boundary is the p1
# |request| line immediately after |turn|16 — this fixture commits the REAL server
# requests at every decision, so nothing is synthesized. (The p2 request line is not
# fed: it is the opponent's private view and the p1 observation never reads it.)
engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=set_source)
parser = _ReplayParser(battle_id="explosion-seed148")
fed = 0
boundary_line_index = None
past_boundary_turn = False
for line_index, line in enumerate(all_lines):
    if past_boundary_turn and line.startswith("|request|"):
        payload = json.loads(line[len("|request|") :])
        if payload.get("side", {}).get("id") != "p1":
            continue  # never feed p2's private request
    parser.feed([line])
    events = parser.public_events
    for event in events[fed:]:
        engine.ingest_event(event)
    fed = len(events)
    if line == f"|turn|{BOUNDARY_TURN}":
        past_boundary_turn = True
    elif past_boundary_turn and line.startswith("|request|"):
        boundary_line_index = line_index
        break
assert boundary_line_index is not None, "p1 request after the boundary turn not found"

replay = parser.snapshot()
assert replay.turn_number == BOUNDARY_TURN
state = normalize_for_player(
    replay,
    player_id="p1",
    configured_showdown_slot="p1",
    format_id="gen3randombattle",
    belief_engine=engine,
    include_turn_merged=True,
)
observation = observation_from_player_state(
    state, category_vocab=vocab, spec=spec, dex=dex, feature_masks=masks
)
observation.validate(spec)
assert vocab.observed_oov_tokens == frozenset(), vocab.observed_oov_tokens


# ------------------------------------------------------------------------- token dumping
def decode_cat(cat_id: int) -> str:
    token = id_to_token.get(cat_id)
    return token if token is not None else f"<oov-bucket:{cat_id}>"


def section_for(index: int) -> str:
    if index < SELF_POKEMON_TOKEN_OFFSET:
        return "field"
    if index < OPPONENT_POKEMON_TOKEN_OFFSET:
        return "self_pokemon"
    if index < ACTION_CANDIDATE_TOKEN_OFFSET:
        return "opponent_pokemon"
    if index < STATS_TOKEN_OFFSET:
        return "action_candidate"
    if index < TRANSITION_TOKEN_OFFSET:
        return "stats"
    return "transition"


def sub_block_dict(sub) -> dict:
    out = {"status": sub.status, "actor_slot": sub.actor_slot}
    if sub.actor_species:
        out["actor_species"] = sub.actor_species
    if sub.status != SUB_BLOCK_ACTION:
        return out  # NEGATED/PENDING/ABSENT carry identity only; all else neutral
    out.update(
        kind=sub.kind,
        action=sub.action,
        damage_fraction=round(sub.damage_fraction, 6),
        damage_outcome=sub.damage_outcome,
        effectiveness=sub.effectiveness,
        side_effect=sub.side_effect,
        n_hits=sub.n_hits,
    )
    if sub.self_hp_cost:  # #519: upfront move price / recoil, fraction of own max HP
        out["self_hp_cost"] = round(sub.self_hp_cost, 6)
    for flag in ("called", "transformed", "crit", "miss", "ko", "pursuit_intercept"):
        if getattr(sub, flag):
            out[flag] = True
    for optional in ("defender_species", "cant_reason", "baton_pass_species"):
        if getattr(sub, optional) is not None:
            out[optional] = getattr(sub, optional)
    if sub.residual_valid:
        out["residual"] = round(sub.residual, 6)
        out["residual_valid"] = True
    if sub.cb_bit:
        out["cb_bit"] = True
    if sub.investment:
        out["investment"] = sub.investment
    return out


def merged_token_dict(token) -> dict:
    return {
        "turn": token.turn,
        "phase": token.phase,
        "first": sub_block_dict(token.first),
        "second": sub_block_dict(token.second),
        "context_trio": {
            "own_spikes_layers": token.own_spikes_layers,
            "opp_spikes_layers": token.opp_spikes_layers,
            "weather": token.weather,
        },
    }


attention = list(observation.attention_mask)
tokens_out = []
for index in range(spec.token_count):
    numeric_row = observation.numeric_features[index]
    cat_row = observation.categorical_ids[index]
    nonzero = {NUMERIC_NAMES[i]: round(float(v), 6) for i, v in enumerate(numeric_row) if v != 0.0}
    cats = {}
    for i, cid in enumerate(cat_row):
        if cid != 0:
            cats[CAT_NAMES[i]] = {"id": int(cid), "token": decode_cat(int(cid))}
    if not attention[index]:
        assert not nonzero and not cats, f"non-attended token {index} carries data"
        continue
    entry = {
        "index": index,
        "section": section_for(index),
        "token_type_id": int(observation.token_type_ids[index]),
        "attended": True,
        "categoricals": cats,
        "numerics": nonzero,
    }
    slot_cat = cats.get("CATEGORY_SLOT")
    if slot_cat:
        entry["slot"] = slot_cat["token"]
    if index >= TRANSITION_TOKEN_OFFSET:
        entry["decoded_turn_merged_token"] = merged_token_dict(
            state.turn_merged_tokens[index - TRANSITION_TOKEN_OFFSET]
        )
    tokens_out.append(entry)

merged_count = len(state.turn_merged_tokens)
assert sum(1 for t in tokens_out if t["section"] == "transition") == merged_count


# --------------------------------------------------------------- line->token examples
def turn_block(turn: int) -> list[str]:
    start = all_lines.index(f"|turn|{turn}")
    try:
        end = all_lines.index(f"|turn|{turn + 1}")
    except ValueError:
        end = len(all_lines)
    return [line for line in all_lines[start:end] if not line.startswith("|request|")]


def merged_slot(turn: int, phase: str) -> int:
    for i, token in enumerate(state.turn_merged_tokens):
        if token.turn == turn and token.phase == phase:
            return i
    raise KeyError((turn, phase))


def encoded_entry(slot: int) -> dict:
    return next(t for t in tokens_out if t["index"] == TRANSITION_TOKEN_OFFSET + slot)


def tm_example(turn: int, phase: str, source_lines: list[str], note: str, extra_slots=()) -> dict:
    slot = merged_slot(turn, phase)
    example = {
        "note": note,
        "protocol_lines": source_lines,
        "transition_slot": slot,
        "token_index": TRANSITION_TOKEN_OFFSET + slot,
        "decoded_fields": merged_token_dict(state.turn_merged_tokens[slot]),
        "encoded_token": encoded_entry(slot),
    }
    companions = []
    for companion_turn, companion_phase in extra_slots:
        companion = merged_slot(companion_turn, companion_phase)
        companions.append(
            {
                "transition_slot": companion,
                "token_index": TRANSITION_TOKEN_OFFSET + companion,
                "decoded_fields": merged_token_dict(state.turn_merged_tokens[companion]),
                "encoded_token": encoded_entry(companion),
            }
        )
    if companions:
        example["companion_tokens"] = companions
    return example


t1 = turn_block(1)
t7 = turn_block(7)
t15 = turn_block(15)

# The negated-adjacent slot: prefer a real NEGATED/PENDING sub-block if the game has one
# before the boundary; this game does not (every declared action executed), so the third
# example is the Liechi-eat turn — whose residual-phase story (berry eat, +1 Atk, poison
# faint) is exactly what the sub-block layer EXCLUDES, making it the negative-space demo.
negated_or_pending = [
    (i, token)
    for i, token in enumerate(state.turn_merged_tokens)
    if token.second.status in (SUB_BLOCK_NEGATED, SUB_BLOCK_PENDING)
]

examples = [
    tm_example(
        7,
        PHASE_TURN,
        [line for line in t7 if line not in ("|", "|upkeep")],
        "THE EXPLOSION TURN, as two tokens. Token one (this one, phase=turn): our declared "
        "Gligar switch executes first (sub-block A), then Weezing's Explosion crits the "
        "fresh Gligar for its full 125/243 and the ko bit lights (sub-block B, "
        "tt2_damage 0.514, tt2_crit, tt2_ko); Explosion's self-faint is move mechanics, "
        "not a KO outcome — it surfaces instead as the TERMINAL self-cost, "
        "NUMERIC_TM2_SELF_HP_COST = 1.0 (the user spends its whole HP), so ONE ko bit "
        "for TWO faints. Both actives are now empty — the engine runs ONE forceSwitch "
        "cycle for both sides, which is the companion token below.",
        extra_slots=((7, PHASE_REPLACEMENT),),
    ),
    tm_example(
        1,
        PHASE_TURN,
        [line for line in t1 if line not in ("|", "|upkeep")],
        "SELF-COST ANATOMY: one turn token with nonzero self-cost on BOTH sub-blocks, "
        "covering both classes short of Explosion's terminal 1.0. Sub-block A: our "
        "Pidgeot's Substitute pays the exact quarter — NUMERIC_TT_SELF_HP_COST = "
        "71/284 = 0.25 (an upfront move price, no target damage). Sub-block B: "
        "Piloswine's Double-Edge breaks the fresh sub (no |-damage| on Pidgeot, so "
        "tt2_damage stays 0) and takes RECOIL — NUMERIC_TM2_SELF_HP_COST = 23/316 = "
        "0.072785, read from the |[from] Recoil| line. The 0.25-vs-0.073 contrast is "
        "the whole column in one card; turn 2 repeats the recoil class at 39/316 = "
        "0.123418 against Gligar. Aside: that Double-Edge is a Choice Band lock — p2's "
        "own requests show its other three moves disabled from turn 2 (ground truth in "
        "the log, invisible to our encode), and by this boundary the Tier-1 belief "
        "layer has independently pinned the item by set elimination: Piloswine's "
        "candidate set is down to 1, so token 7 carries belief:possible_item:choiceband "
        "with NUMERIC_POSSIBLE_ITEM_COUNT = 1.0 — while the Tier-2 damage-evidence CB "
        "channel (NUMERIC_TIER2_CB_PINNED / tt CB bits) stays 0.0 on this replay path.",
    ),
    tm_example(
        15,
        PHASE_REPLACEMENT,
        [line for line in t15 if "Regirock" in line or "Spikes" in line or line.startswith("|faint")],
        "Hazard context lit + a single replacement + an ABSENT half. Regirock replaces the "
        "poison-fainted Pidgeot: phase=replacement, first sub-block is the switch-in, and "
        "the second is ABSENT (tt2_status:absent — no second declaration EXPECTED; only "
        "one side was replacing, unlike the turn-7 cold pair). Cloyster's turn-13 Spikes "
        "shows as the context trio: NUMERIC_TT_OWN_SPIKES = 1/3 = 0.333333, captured as of "
        "this phase — the entry damage itself (227/259) belongs to the current-state "
        "Regirock token, not the history stream.",
    ),
    tm_example(
        15,
        PHASE_TURN,
        [line for line in t15 if line not in ("|", "|upkeep")],
        "The Liechi-eat turn (no NEGATED/PENDING sub-block exists in this game before the "
        "boundary — every declared action executed — so this is the negative-space "
        "example). Piloswine's switch-in is sub-block A; our Return for 31% is sub-block "
        "B with tt2_species:Piloswine as defender. Pidgeot's Liechi eat, +1 Atk boost and "
        "poison faint all happen in the RESIDUAL phase: residuals are not actions, never "
        "form sub-blocks, and surface instead in current-state tokens (the belief item "
        "ledger; here on our own side, the request). The residual FAINT is what spawns "
        "the separate replacement token above; had Pidgeot instead fainted mid-turn "
        "BEFORE acting, Return would have been consumed traceless and encoded as "
        "tt2_status:negated — the consumption-proof rule.",
    ),
]

# Self-cost latch (#519): the self-cost anatomy and Explosion notes above quote exact
# encoder values for the per-sub-block SELF_HP_COST columns. Refuse to write a dump
# whose encode contradicts them — if this fires, the #519 encoder normalizes the
# column differently than the notes claim (e.g. dex-static price vs observed recoil
# line) and the notes must be corrected, not the assert.
_sc = {
    "substitute_A": examples[1]["encoded_token"]["numerics"].get("NUMERIC_TT_SELF_HP_COST"),
    "recoil_B": examples[1]["encoded_token"]["numerics"].get("NUMERIC_TM2_SELF_HP_COST"),
    "explosion_B": examples[0]["encoded_token"]["numerics"].get("NUMERIC_TM2_SELF_HP_COST"),
}
_sc_expected = {"substitute_A": 0.25, "recoil_B": 0.072785, "explosion_B": 1.0}
assert _sc == _sc_expected, (
    f"self-cost worked-example values diverge from the encoder: {_sc} != {_sc_expected}; "
    "fix the example notes in this script to match the real encode, then regenerate."
)

# ------------------------------------------------------------------------ context section
context_start = all_lines.index("|start")
context_lines = [
    line
    for line in all_lines[context_start : boundary_line_index]
    if not line.startswith("|request|")
]
elided_requests = sum(
    1 for line in all_lines[context_start:boundary_line_index] if line.startswith("|request|")
)

opp_beliefs = state.belief_view.opponent_by_species()


def mon_summary(pokemon, belief: bool = False):
    meta = sd._pokemon_metadata(pokemon)
    if meta is None:
        return None
    if belief:
        b = opp_beliefs.get(sd._normalize_identifier(pokemon.species))
        if b is not None:
            cf = sd._condition_features(b.condition if b.condition is not None else pokemon.condition)
            meta["condition"] = b.condition if b.condition is not None else meta["condition"]
            meta["hp_fraction"] = cf.hp_fraction
            meta["status"] = b.status if b.status is not None else cf.status
            meta["fainted"] = cf.fainted
    out = {
        key: meta[key]
        for key in ("species", "condition", "hp_fraction", "status", "fainted", "active", "details")
        if meta.get(key) is not None
    }
    if isinstance(out.get("hp_fraction"), float):
        out["hp_fraction"] = round(out["hp_fraction"], 6)
    if meta.get("moves"):
        out["revealed_moves" if belief else "moves"] = sorted(meta["moves"]) if belief else meta["moves"]
    for key in ("ability", "item"):
        if meta.get(key):
            out[("revealed_" if belief else "") + key] = meta[key]
    if meta.get("stats"):
        out["actual_stats"] = meta["stats"]
    if belief:
        b = opp_beliefs.get(sd._normalize_identifier(pokemon.species))
        if b is not None:
            out["belief"] = {
                "candidate_set_count": b.candidate_set_count,
                "uncertainty": round(b.uncertainty, 6),
                "possible_abilities": sorted(b.possible_abilities),
                "possible_items": sorted(b.possible_items),
                "possible_moves": sorted(b.possible_moves),
            }
    return out


context = {
    "fixture": "tests/fixtures/showdown/explosion-seed148.log.gz",
    "manifest": manifest,
    "battle_id": "explosion-seed148",
    "players": dict(replay.players),
    "perspective": {"player": "p1 (PokeZero p1)", "opponent": "p2 (PokeZero p2)"},
    "boundary": (
        f"the p1 |request| immediately after the |turn|{BOUNDARY_TURN} line — the real "
        "committed server request (this fixture carries requests at every decision; "
        "nothing synthesized)"
    ),
    "protocol_lines_start_to_boundary": context_lines,
    "elided_request_lines": elided_requests,
    "battle_state_summary": {
        "turn": replay.turn_number,
        "weather": replay.weather,
        "self_active": mon_summary(state.self_active),
        "self_bench": [mon_summary(p) for p in state.self_team if not p.active],
        "opponent_active": mon_summary(state.opponent_active, belief=True),
        "opponent_revealed_bench": [mon_summary(p, belief=True) for p in state.opponent_team if not p.active],
        "opponent_unrevealed_count": 6 - len(state.opponent_team),
        "self_side_conditions": list(state.self_side_conditions),
        "opponent_side_conditions": list(state.opponent_side_conditions),
        "boosts": {"self_active": dict(state.self_active_boosts), "opponent_active": dict(state.opponent_active_boosts)},
        "volatiles": {"self_active": list(state.self_active_volatiles), "opponent_active": list(state.opponent_active_volatiles)},
        "legal_actions": [i for i, ok in enumerate(state.legal_action_mask) if ok],
        "request_kind": state.request_kind,
    },
}

# ------------------------------------------------------------------------ layout section
phase_census: dict[str, int] = {}
for token in state.turn_merged_tokens:
    phase_census[token.phase] = phase_census.get(token.phase, 0) + 1

layout = {
    "schema_version": spec.schema_version,
    "token_count": spec.token_count,
    "numeric_feature_count": spec.numeric_feature_count,
    "categorical_feature_count": spec.categorical_feature_count,
    "sections": {
        "field": {"offset": FIELD_TOKEN_OFFSET, "count": FIELD_TOKEN_COUNT, "token_type_id": 0},
        "self_pokemon": {"offset": SELF_POKEMON_TOKEN_OFFSET, "count": SELF_POKEMON_TOKEN_COUNT, "token_type_id": 1},
        "opponent_pokemon": {"offset": OPPONENT_POKEMON_TOKEN_OFFSET, "count": OPPONENT_POKEMON_TOKEN_COUNT, "token_type_id": 2},
        "action_candidate": {"offset": ACTION_CANDIDATE_TOKEN_OFFSET, "count": ACTION_CANDIDATE_TOKEN_COUNT, "token_type_id": 3},
        "stats": {"offset": STATS_TOKEN_OFFSET, "count": STATS_TOKEN_COUNT, "token_type_id": 5},
        "transition": {"offset": TRANSITION_TOKEN_OFFSET, "count": TRANSITION_TOKEN_COUNT, "token_type_id": 6},
    },
    "token_type_id_note": "type id 4 (v1 recent-event section) is retired, not reused",
    "numeric_column_names": NUMERIC_NAMES,
    "numeric_census_boundaries": {
        "v2": V2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
        "v2.1": V2_1_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
        "v2.2": V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
    },
    "categorical_slot_layout": {
        "fixed_count": sd.CATEGORY_FIXED_COUNT,
        "belief_ability_buckets": sd.BELIEF_ABILITY_BUCKET_COUNT,
        "belief_item_buckets": sd.BELIEF_ITEM_BUCKET_COUNT,
        "belief_move_buckets": sd.BELIEF_MOVE_BUCKET_COUNT,
        "volatile_buckets": sd.VOLATILE_BUCKET_COUNT,
        "turn_merged_extra": sd.TURN_MERGED_CATEGORICAL_EXTRA,
        "column_names": CAT_NAMES,
        "census_boundaries": {
            "v2/v2.1": V2_1_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            "v2.2": V2_2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        },
    },
    "category_vocab": {
        "built_with_include_turn_merged": True,
        "tokens": len(vocab.tokens),
        "oov_buckets": vocab.oov_buckets,
        "total_rows": vocab.size,
        "padding_row": 0,
    },
    "attention_mask_at_boundary": {
        "attended_total": sum(attention),
        "field": attention[FIELD_TOKEN_OFFSET],
        "self_pokemon_attended": sum(attention[SELF_POKEMON_TOKEN_OFFSET:OPPONENT_POKEMON_TOKEN_OFFSET]),
        "opponent_pokemon_attended": sum(attention[OPPONENT_POKEMON_TOKEN_OFFSET:ACTION_CANDIDATE_TOKEN_OFFSET]),
        "action_candidates_attended": sum(attention[ACTION_CANDIDATE_TOKEN_OFFSET:STATS_TOKEN_OFFSET]),
        "stats_attended": sum(attention[STATS_TOKEN_OFFSET:TRANSITION_TOKEN_OFFSET]),
        "transition_attended": sum(attention[TRANSITION_TOKEN_OFFSET:]),
        "transition_padded": TRANSITION_TOKEN_COUNT - sum(attention[TRANSITION_TOKEN_OFFSET:]),
        "transition_token_budget_K": masks.transition_token_budget,
    },
    "k_budget_unit_note": (
        "masks.transition_token_budget counts TOKENS in every schema, but a v2.2 token "
        "covers a WHOLE turn/lead/replacement phase — the v2/v2.1 K=64 horizon "
        "(~32 turns) is budget=32 under v2.2; an unchanged K roughly doubles the "
        "temporal horizon. Here: 15 completed turns => 18 merged tokens vs 35 "
        "per-action tokens."
    ),
    "turn_merged_phase_census": phase_census,
    "per_action_token_count_same_boundary": len(state.transition_tokens),
    "legal_action_mask": list(observation.legal_action_mask),
    "populated_transition_token_indices": (
        f"{TRANSITION_TOKEN_OFFSET}..{TRANSITION_TOKEN_OFFSET + merged_count - 1}"
    ),
}

masks_section = {
    "feature_masks": {
        "stats_block": masks.stats_block,
        "exact_state": masks.exact_state,
        "transition_token_budget": masks.transition_token_budget,
        "tier2_residuals": masks.tier2_residuals,
        "tier2_investment": masks.tier2_investment,
    },
    "live_feature_blocks": {
        "stats_block": bool(masks.stats_block and state.tendency_stats is not None),
        "exact_state_layer": masks.exact_state,
        "tier2_residual_gate": masks.tier2_residuals,
        "tier2_investment_gate": masks.tier2_investment,
        "tier2_carrying_sub_blocks_at_this_boundary": sum(
            1
            for token in state.turn_merged_tokens
            for sub in (token.first, token.second)
            if sub.residual_valid or sub.cb_bit or sub.investment
        ),
        "tier2_note": (
            "both tier2 gates are ON but this is the Tier-1 replay-extraction path "
            "(normalize_for_player without a tier2/investment annotation pass), so no "
            "sub-block carries Tier-2 fields and every residual/validity/CB/investment "
            "column — first sub-block (117-120), second sub-block (TM2 148-150, 152) and "
            "the per-mon pinned pair (138-139) — reads 0.0. Live envs populate them via "
            "Tier2LiveTracker.annotate + the #513 investment tracker feeding "
            "annotate_turn_merged_tokens."
        ),
        "belief_candidate_sets": True,
    },
    "belief_set_source": {
        "env": {"POKEZERO_BELIEF_SET_SOURCE": os.environ["POKEZERO_BELIEF_SET_SOURCE"]},
        "format_id": "gen3randombattle",
        "source_hash": set_source.metadata.source_hash,
        "note": (
            "set source attached to PublicBattleBeliefEngine exactly as in "
            "tests/test_corpus_replay.py; POKEZERO_BELIEF_SET_SOURCE is the env-level "
            "flip the harness paths use for the same switch"
        ),
    },
    "dual_schema_story": {
        "current_default": "pokezero.observation.v2.1 (v2.2 is the batch-3 ablation arm, deliberately not the default)",
        "v2": "151 tokens x 121 numeric x 39 categorical; per-action transition tokens; accepts pre-#509 checkpoints and stays byte-identical to the pre-v2.1 encoder (119-column relic family floors lower by design)",
        "v2.1": "151 tokens x 140 numeric x 39 categorical; adds defender identity on move transition rows, per-bucket revealed-move PP-validity bits, substitute HP fraction, per-mon pinned Tier-2 CB/investment surface",
        "v2.2": "151 tokens x 153 numeric x 51 categorical; every v2.1 block carried forward; transition surface swapped to turn-merged tokens (this dump)",
        "resolution": (
            "which schema an env/harness encodes resolves from the loaded checkpoint's "
            "stamped model_config (feature_masks_from_model_config / "
            "env_config_with_checkpoint_masks latch family); "
            "DEFAULT_REPLAY_OBSERVATION_SPEC only covers checkpoint-free encodes. All "
            "three schemas pass require_current_observation_schema; v1/unversioned "
            "artifacts refuse with replay-from-pinned-tag guidance."
        ),
    },
}

dump = {
    "context": context,
    "layout": layout,
    "masks": masks_section,
    "tokens": tokens_out,
    "line_to_token_examples": examples,
    "negated_or_pending_sub_blocks_in_game": [
        {"transition_slot": i, "decoded": merged_token_dict(token)} for i, token in negated_or_pending
    ],
    "extraction": {
        "spec": "V2_2_REPLAY_OBSERVATION_SPEC (pokezero.observation.v2.2)",
        "fixture_sha256_clean_log": manifest["sha256_clean_log"],
        "replay_loop": (
            "line-by-line parser.feed + incremental belief-engine ingest, identical to "
            "tests/test_corpus_replay.py; boundary snapshot right after feeding the p1 "
            f"|request| that follows the |turn|{BOUNDARY_TURN} line; "
            "normalize_for_player(include_turn_merged=True)"
        ),
        "boundary_request_provenance": (
            "the committed fixture log carries the REAL |request| line at every decision "
            "(unlike the capture-corpus games the v2 doc used), so the boundary request "
            "is the exact server payload — no reconstruction"
        ),
        "populated_turn_merged_tokens": merged_count,
        "per_action_tokens_same_fold": len(state.transition_tokens),
        "zero_oov": True,
        "determinism": (
            "belief candidate lists sorted, floats rounded to 6 places, no commit/clock "
            "stamps — tests/test_token_format_doc.py regenerates this dump and asserts "
            "byte-identity with the committed file"
        ),
    },
}


def main() -> None:
    OUT.write_text(json.dumps(dump, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"merged transition tokens: {merged_count} (per-action: {len(state.transition_tokens)}); "
          f"attended total: {sum(attention)}/{spec.token_count}")
    print(f"explosion turn token index: {examples[0]['token_index']}; "
          f"cold pair token index: {examples[0]['companion_tokens'][0]['token_index']}")
    print(f"legal actions: {context['battle_state_summary']['legal_actions']}")


if __name__ == "__main__":
    main()
