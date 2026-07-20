"""Export the v2.2 encoder's vocabulary + layout + dex tables to one JSON artifact.

The Rust encoder (rust/pokezero-search, ``encode_decision``) loads this file —
NO table is ever hand-transcribed into Rust. Everything here is read from the
Python source of truth:

- ``vocab``: the exact ``CategoryVocabulary`` row mapping for the closed gen3
  randbat universe (turn-merged families included, matching the v2.2 encode),
  as an explicit normalized-string -> row-id index (aliases pre-resolved),
  plus the OOV policy constants (blake2b-8 big-endian mod oov_buckets, offset
  1 + len(tokens)) and the pad row (0).
- ``layout``: the token-section offsets and every categorical/numeric column
  index the encoder writes (``CATEGORY_*`` / ``NUMERIC_*``), the v2.2 spec
  census, and the numeric normalization constants.
- ``dex``: the gen3-resolved per-species and per-move tables exactly as
  ``pokezero.dex`` resolves them (effect labels/chances pre-derived).

Deterministic: canonical JSON (sorted keys, no timestamps); the printed
SHA-256 is stable for a given Showdown build.

Usage:

    PYTHONPATH=src python scripts/export_encoder_tables.py \
        --showdown-root <built-showdown> --out corpus/encoder_tables.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pokezero.showdown as showdown  # noqa: E402
from pokezero.actions import ACTION_COUNT, MOVE_ACTION_COUNT, SWITCH_ACTION_COUNT  # noqa: E402
from pokezero.category_vocab import normalize_category_value  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.observation import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION_V2_2,
    ObservationFeatureMasks,
)
from pokezero.randbat_vocab import gen3_category_vocabulary  # noqa: E402
from pokezero.showdown import observation_spec_for_schema  # noqa: E402

TABLES_SCHEMA_VERSION = "pokezero.encoder-tables.v1"


def _vocab_payload(showdown_root: str) -> dict[str, Any]:
    vocab = gen3_category_vocabulary(showdown_root, include_turn_merged=True)
    index: dict[str, int] = {}
    for row, token in enumerate(vocab.tokens, start=1):
        index[normalize_category_value(token)] = row
    for alias, base in vocab.aliases.items():
        base_row = index.get(normalize_category_value(base))
        if base_row is not None:
            index[normalize_category_value(alias)] = base_row
    return {
        "include_turn_merged": True,
        "tokens": list(vocab.tokens),
        "index": index,
        "oov_buckets": vocab.oov_buckets,
        "oov_offset": 1 + len(vocab.tokens),
        "pad_row": 0,
        "size": vocab.size,
        "oov_policy": "blake2b(digest_size=8, big-endian) % oov_buckets + oov_offset",
        "normalization": "strip + lowercase (category_vocab.normalize_category_value)",
    }


def _layout_payload() -> dict[str, Any]:
    spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V2_2)
    masks = ObservationFeatureMasks()
    categorical_columns = {
        name: int(getattr(showdown, name))
        for name in dir(showdown)
        if name.startswith("CATEGORY_") and isinstance(getattr(showdown, name), int)
    }
    numeric_columns = {
        name: int(getattr(showdown, name))
        for name in dir(showdown)
        if name.startswith("NUMERIC_") and isinstance(getattr(showdown, name), int)
    }
    return {
        "schema_version": spec.schema_version,
        "token_count": spec.token_count,
        "categorical_feature_count": spec.categorical_feature_count,
        "numeric_feature_count": spec.numeric_feature_count,
        "action_count": ACTION_COUNT,
        "move_action_count": MOVE_ACTION_COUNT,
        "switch_action_count": SWITCH_ACTION_COUNT,
        "token_offsets": {
            "field": showdown.FIELD_TOKEN_OFFSET,
            "self_pokemon": showdown.SELF_POKEMON_TOKEN_OFFSET,
            "opponent_pokemon": showdown.OPPONENT_POKEMON_TOKEN_OFFSET,
            "action_candidates": showdown.ACTION_CANDIDATE_TOKEN_OFFSET,
            "stats": showdown.STATS_TOKEN_OFFSET,
            "transition": showdown.TRANSITION_TOKEN_OFFSET,
        },
        "token_type_ids": {
            "field": 0,
            "self_pokemon": 1,
            "opponent_pokemon": 2,
            "action": 3,
            "stats": 5,
            "transition": 6,
        },
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "belief_buckets": {
            "ability": showdown.BELIEF_ABILITY_BUCKET_COUNT,
            "item": showdown.BELIEF_ITEM_BUCKET_COUNT,
            "move": showdown.BELIEF_MOVE_BUCKET_COUNT,
        },
        "volatile_bucket_count": showdown.VOLATILE_BUCKET_COUNT,
        "constants": {
            "actual_stat_divisor": showdown._ACTUAL_STAT_DIVISOR,
            "stat_count_divisor": showdown._STAT_COUNT_DIVISOR,
            "timed_condition_duration": showdown._TIMED_CONDITION_DURATION,
            "timed_side_conditions": list(showdown._TIMED_SIDE_CONDITIONS),
            "hazard_conditions": list(showdown._HAZARD_CONDITIONS),
            "screen_conditions": list(showdown._SCREEN_CONDITIONS),
            "trap_abilities": sorted(showdown._TRAP_ABILITIES),
            "pinch_berries": sorted(showdown._PINCH_BERRIES),
            "weather_reveal_order": list(showdown._WEATHER_REVEAL_ORDER),
            "boost_stat_slots": [[stat, slot] for stat, slot in showdown._BOOST_STAT_SLOTS],
            "base_stat_slots": [[stat, slot] for stat, slot in showdown._BASE_STAT_SLOTS],
            "actual_stat_slots": [[stat, slot] for stat, slot in showdown._ACTUAL_STAT_SLOTS],
            "timed_condition_slots": [
                [condition, self_slot, opp_slot]
                for condition, self_slot, opp_slot in showdown._TIMED_CONDITION_SLOTS
            ],
        },
        "default_feature_masks": {
            "stats_block": masks.stats_block,
            "exact_state": masks.exact_state,
            "transition_token_budget": masks.transition_token_budget,
            "tier2_residuals": masks.tier2_residuals,
            "tier2_investment": masks.tier2_investment,
        },
    }


def _dex_payload(showdown_root: str) -> dict[str, Any]:
    dex = load_showdown_dex_cached(showdown_root)
    species = {
        key: {
            "name": info.name,
            "types": list(info.types),
            "base_stats": dict(info.base_stats),
        }
        for key, info in dex.species.items()
    }
    # ``base_power`` is exported as the STATIC dex value, NOT ``resolve_move_base_power``-resolved.
    # Variable-power moves resolve at ENCODE time in the Rust crate, exactly mirroring Python:
    #   - Hidden Power's type/base power is PER-MON (the acting mon's typed request move, e.g.
    #     "hiddenpowerice"), so it can never be a static table keyed by the generic "hiddenpower"
    #     id — the crate resolves the typed variant and looks IT up (encoder.rs::self_move_mechanics_id).
    #   - Return/Frustration (static happiness base power 102/1) could be baked in here, but MUST NOT
    #     be: this same ``base_power`` field is read raw (``base_power > 0``) by the Tier-2
    #     physical-attack heuristic (encoder.rs mirroring showdown._is_physical_attack), where the
    #     static 0 for Return is load-bearing for byte-parity. So the happiness constant lives ONLY
    #     in encoder.rs::resolve_move_base_power (mirroring dex._HAPPINESS_BASE_POWER), never here.
    # Reversal/Flail/Eruption/Water Spout are likewise static here and HP-fraction-resolved at encode.
    moves = {
        key: {
            "name": info.name,
            "type": info.type,
            "category": info.category,
            "gen3_category": info.gen3_category,
            "base_power": info.base_power,
            "accuracy": info.accuracy,
            "priority": info.priority,
            "effect_label": info.effect_label,
            "effect_chance": info.effect_chance,
            "self_hp_cost": info.self_hp_cost,
            "pp": info.pp,
            "max_pp": info.max_pp,
        }
        for key, info in dex.moves.items()
    }
    return {"species": species, "moves": moves}


def build_tables(showdown_root: str) -> dict[str, Any]:
    return {
        "schema_version": TABLES_SCHEMA_VERSION,
        "vocab": _vocab_payload(showdown_root),
        "layout": _layout_payload(),
        "dex": _dex_payload(showdown_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    tables = build_tables(str(args.showdown_root))
    encoded = json.dumps(tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded + "\n", encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "out": str(args.out),
                "bytes": len(encoded) + 1,
                "sha256": digest,
                "vocab_size": tables["vocab"]["size"],
                "vocab_tokens": len(tables["vocab"]["tokens"]),
                "species": len(tables["dex"]["species"]),
                "moves": len(tables["dex"]["moves"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
