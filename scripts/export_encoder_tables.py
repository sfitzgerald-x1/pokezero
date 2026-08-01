"""Export a schema-bound encoder vocabulary + layout + dex artifact.

The Rust encoder (rust/pokezero-search, ``encode_decision``) loads this file —
NO table is ever hand-transcribed into Rust. Everything here is read from the
Python source of truth:

- ``vocab``: the exact ``CategoryVocabulary`` row mapping for the closed gen3
  randbat universe (turn-merged families included, matching V2.2/V3),
  as an explicit normalized-string -> row-id index (aliases pre-resolved),
  plus the OOV policy constants (blake2b-8 big-endian mod oov_buckets, offset
  1 + len(tokens)) and the pad row (0).
- ``layout``: the token-section offsets and every categorical/numeric column
  physical index the encoder writes (``CATEGORY_*`` / ``NUMERIC_*``), the
  selected schema census, and the numeric normalization constants.
- ``dex``: the gen3-resolved per-species and per-move tables exactly as
  ``pokezero.dex`` resolves them (effect labels/chances pre-derived).

Deterministic: canonical JSON (sorted keys, no timestamps); the printed
SHA-256 is stable for a given Showdown build.

Usage:

    PYTHONPATH=src python scripts/export_encoder_tables.py \
        --showdown-root <built-showdown> --observation-schema v3 \
        --out corpus/encoder_tables.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
    ObservationFeatureMasks,
)
from pokezero.randbat_vocab import gen3_category_vocabulary  # noqa: E402
from pokezero.showdown import (  # noqa: E402
    numeric_index_if_present_for_schema,
    observation_schema_version_from_choice,
    observation_spec_for_schema,
)

TABLES_SCHEMA_VERSION = "pokezero.encoder-tables.v1"


def _vocab_payload(
    showdown_root: str,
    trained_tokens: Any = None,
    schema_version: str = OBSERVATION_SCHEMA_VERSION_V2_2,
) -> dict[str, Any]:
    """Emit the token->row map the LEAF encoder will use.

    ``trained_tokens`` is the checkpoint's own ``category_vocab`` and is authoritative
    whenever it is available. The build's enumeration is only correct for a model that
    the build itself created: the model's embedding rows were learned against the
    positions in force at training time, and the enumeration is a positional list, so a
    token added later renumbers everything after it. Exporting build tokens for an older
    checkpoint therefore hands the crate a map that resolves the same string to a row
    the model learned as something else.

    `TransformerPolicyConfig.__post_init__` requires ``category_vocab`` on every valid
    config (neural_policy.py:399), so no vocab-less checkpoint contract exists; the
    build enumeration below is reached only when exporting without a checkpoint at all.
    """
    turn_merged = schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
    vocab = gen3_category_vocabulary(
        showdown_root,
        include_turn_merged=turn_merged,
        # The v4 feature pack's two categorical families are opt-in because they change the
        # vocabulary SIZE. This build-enumeration path is only reached when exporting without a
        # checkpoint; with one, ``trained_tokens`` below overrides it wholesale.
        include_feature_pack_v4=schema_version in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    )
    if trained_tokens:
        vocab = replace(vocab, tokens=tuple(str(t) for t in trained_tokens))
    index: dict[str, int] = {}
    for row, token in enumerate(vocab.tokens, start=1):
        index[normalize_category_value(token)] = row
    for alias, base in vocab.aliases.items():
        base_row = index.get(normalize_category_value(base))
        if base_row is not None:
            index[normalize_category_value(alias)] = base_row
    return {
        "include_turn_merged": turn_merged,
        "tokens": list(vocab.tokens),
        "index": index,
        "oov_buckets": vocab.oov_buckets,
        "oov_offset": 1 + len(vocab.tokens),
        "pad_row": 0,
        "size": vocab.size,
        "oov_policy": "blake2b(digest_size=8, big-endian) % oov_buckets + oov_offset",
        "normalization": "strip + lowercase (category_vocab.normalize_category_value)",
    }


def _numeric_column_payload(schema_version: str) -> dict[str, int]:
    columns: dict[str, int] = {}
    spec = observation_spec_for_schema(schema_version)
    for name in dir(showdown):
        value = getattr(showdown, name)
        if not name.startswith("NUMERIC_") or not isinstance(value, int):
            continue
        # A grouped-layout schema (v3, v4) resolves every named column through its projection
        # map, which also reports the ones it does not carry; the v2 family instead range-checks
        # against its own census, since later columns all sit above it.
        if schema_version not in GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS and not (
            0 <= value < spec.numeric_feature_count
        ):
            continue
        physical = numeric_index_if_present_for_schema(schema_version, value)
        if physical is not None:
            columns[name] = physical
    return columns


def _numeric_slot(schema_version: str, legacy_index: int) -> int:
    physical = numeric_index_if_present_for_schema(schema_version, legacy_index)
    if physical is None:
        raise ValueError(
            f"encoder table slot {legacy_index} is absent from schema {schema_version!r}"
        )
    return physical


def _layout_payload(
    schema_version: str = OBSERVATION_SCHEMA_VERSION_V2_2,
    spec: Any = None,
    masks: Any = None,
) -> dict[str, Any]:
    # A region-trimmed checkpoint's spec differs from the schema default only in
    # transition_token_count (and therefore token_count). The transition region
    # is the LAST token block, so every token_offset above it stays valid and a
    # trimmed spec simply describes a shorter tail. Passing the checkpoint's own
    # spec is what lets the crate encode leaves a trimmed model can consume;
    # without it the tables describe 87 tokens while the model expects 39 and
    # the root/leaf contract check (correctly) refuses to run.
    spec = spec if spec is not None else observation_spec_for_schema(schema_version)
    # The masks must come from the CHECKPOINT for the same reason the spec does.
    # `default_feature_masks` is not decorative: the crate gates real encode work
    # on it (`layout.tier2_investment`, `layout.transition_token_budget`), so a
    # schema default here silently makes the leaf encoder describe a different
    # observation than the Python root encode — the census-mismatch class the
    # contract check exists to prevent. Defaulting cost two live errors:
    # `tier2_investment` (default False, every trained checkpoint True) blanked
    # the investment slots at every leaf, and `transition_token_budget` (default
    # 128, clamped to the region) pinned history to the full region width, so a
    # budget-0 Markov checkpoint was fed 64 synthesized history tokens it was
    # trained to never attend to.
    masks = masks if masks is not None else ObservationFeatureMasks()
    categorical_columns = {
        name: int(getattr(showdown, name))
        for name in dir(showdown)
        if name.startswith("CATEGORY_")
        and isinstance(getattr(showdown, name), int)
        and 0 <= int(getattr(showdown, name)) < spec.categorical_feature_count
    }
    numeric_columns = _numeric_column_payload(schema_version)
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
            "matchup_count_divisor": showdown._MATCHUP_COUNT_DIVISOR,
            "timed_condition_duration": showdown._TIMED_CONDITION_DURATION,
            "timed_side_conditions": list(showdown._TIMED_SIDE_CONDITIONS),
            "hazard_conditions": list(showdown._HAZARD_CONDITIONS),
            "screen_conditions": list(showdown._SCREEN_CONDITIONS),
            "trap_abilities": sorted(showdown._TRAP_ABILITIES),
            "pinch_berries": sorted(showdown._PINCH_BERRIES),
            "weather_reveal_order": list(showdown._WEATHER_REVEAL_ORDER)[
                : 3 if schema_version in GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS else None
            ],
            "boost_stat_slots": [
                [stat, _numeric_slot(schema_version, slot)]
                for stat, slot in showdown._BOOST_STAT_SLOTS
            ],
            "base_stat_slots": [
                [stat, _numeric_slot(schema_version, slot)]
                for stat, slot in showdown._BASE_STAT_SLOTS
            ],
            "actual_stat_slots": [
                [stat, _numeric_slot(schema_version, slot)]
                for stat, slot in showdown._ACTUAL_STAT_SLOTS
            ],
            "timed_condition_slots": [
                [
                    condition,
                    _numeric_slot(schema_version, self_slot),
                    _numeric_slot(schema_version, opp_slot),
                ]
                for condition, self_slot, opp_slot in showdown._TIMED_CONDITION_SLOTS
                if numeric_index_if_present_for_schema(schema_version, self_slot) is not None
                and numeric_index_if_present_for_schema(schema_version, opp_slot) is not None
            ],
        },
        "default_feature_masks": {
            "stats_block": masks.opponent_tendency_stats_block,
            "exact_state": masks.exact_state,
            "transition_token_budget": min(
                masks.transition_token_budget, spec.transition_token_count
            ),
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


def build_tables(
    showdown_root: str,
    *,
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION_V2_2,
    spec: Any = None,
    masks: Any = None,
    trained_tokens: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": TABLES_SCHEMA_VERSION,
        "vocab": _vocab_payload(showdown_root, trained_tokens, observation_schema_version),
        "layout": _layout_payload(observation_schema_version, spec=spec, masks=masks),
        "dex": _dex_payload(showdown_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument(
        "--observation-schema",
        choices=("v2.2", "v3", "v4"),
        default="v2.2",
        help="Observation layout to export (default: v2.2 for compatibility).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Derive the layout from this checkpoint's own model config instead of the "
            "schema default. Required for a REGION-TRIMMED checkpoint, whose transition "
            "region (and therefore token_count) is narrower than the schema's."
        ),
    )
    args = parser.parse_args(argv)

    schema_version = observation_schema_version_from_choice(args.observation_schema)
    if schema_version not in {
        OBSERVATION_SCHEMA_VERSION_V2_2,
        OBSERVATION_SCHEMA_VERSION_V3,
        OBSERVATION_SCHEMA_VERSION_V4,
    }:
        parser.error(f"unsupported encoder-table schema: {args.observation_schema!r}")
    spec = None
    masks = None
    trained_tokens = None
    if args.checkpoint is not None:
        from pokezero.neural_policy import (  # noqa: PLC0415 - optional torch dependency
            feature_masks_from_model_config,
            load_transformer_model_config,
            observation_spec_from_model_config,
        )

        config = load_transformer_model_config(args.checkpoint)
        spec = observation_spec_from_model_config(config)
        masks = feature_masks_from_model_config(config)
        trained_tokens = tuple(str(t) for t in (getattr(config, "category_vocab", ()) or ()))
        if spec.schema_version != schema_version:
            parser.error(
                f"--checkpoint schema {spec.schema_version!r} != --observation-schema "
                f"{schema_version!r}; refusing to emit tables the model cannot consume"
            )
    tables = build_tables(
        str(args.showdown_root),
        observation_schema_version=schema_version,
        spec=spec,
        masks=masks,
        trained_tokens=trained_tokens,
    )
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
