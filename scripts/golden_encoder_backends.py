"""Encoder backends for the golden-corpus bit-exactness harness (track B).

A backend consumes ONE corpus decision row's SANCTIONED input surface (the
"row inputs", see :func:`row_inputs_from_decision_row`) and returns the five
observation arrays in the corpus's canonical dtypes. The harness
(``scripts/validate_rust_encoder.py``) diffs them bit-exactly against the
stored golden arrays.

Input contract (docs/golden_corpus_notes.md, "Encoder input contract"):
the golden observation is a function of the PUBLIC/BELIEF surface only —
the ``public_materialization`` payload plus the acting seat's
``observation_metadata`` (belief_view overlay + request-known self team).
``true_teams`` is oracle-only and is deliberately NOT part of the row
inputs; the stored golden ``observation`` block (arrays, hashes, inline
legal mask) is the OUTPUT under test and is excluded too.

Backends:

- ``python-reference`` — reconstructs a ``PlayerRelativeBattleState`` from
  the row inputs and calls the production ``observation_from_player_state``.
  This backend defines what the stored surface CAN reproduce: any residual
  mismatch it shows is a corpus input gap, not an encoder bug. Known gaps
  (quantified by the harness, headline finding of track B phase 1): the
  per-row surface carries no whole-game event stream, so the turn-merged
  transition tokens (23..150), the tendency aggregates (stats token 22 +
  the per-opponent-mon tendency triple), the pinned Tier-2 conclusions,
  and the transition extent of the attention mask are NOT reconstructable
  per-row. This module reconstructs them as EMPTY (zero history).

- ``rust`` — calls ``pokezero_search.encode_decision(row_inputs_json,
  tables_json)`` from the native crate (rust/pokezero-search). The tables
  artifact comes from ``scripts/export_encoder_tables.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy

from pokezero.actions import ACTION_COUNT
from pokezero.belief import PlayerBeliefView, RevealedPokemonBelief
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import load_showdown_dex_cached
from pokezero.golden_corpus import GOLDEN_ARRAY_FIELDS, GoldenObservationArrays
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    ObservationFeatureMasks,
    ObservationPerspective,
)
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.showdown import (
    ShowdownPokemon,
    observation_from_player_state,
    observation_spec_for_schema,
)
from pokezero.showdown import PlayerRelativeBattleState
from pokezero.transitions import TendencyStats

ARRAY_NAMES = tuple(name for name, _, _ in GOLDEN_ARRAY_FIELDS)
ARRAY_DTYPES = {name: dtype for name, dtype, _ in GOLDEN_ARRAY_FIELDS}

# The five-turn timed-condition duration (gen 3); mirrors showdown._TIMED_CONDITION_DURATION,
# re-declared here because the source constant is module-private and src/pokezero is read-only.
_TIMED_CONDITION_DURATION = 5


# ---------------------------------------------------------------------------
# Row inputs: the sanctioned per-row input surface
# ---------------------------------------------------------------------------


def row_inputs_from_decision_row(row: Any) -> dict[str, Any]:
    """Extract the sanctioned encoder inputs from a ``GoldenDecisionRow``.

    Includes ONLY the public/belief surface: identifiers, the seat's
    ``observation_metadata`` (verbatim) and the ``public_materialization``
    payload (verbatim). Excludes the golden ``observation`` block (the output
    under test) and everything oracle-only (``true_teams`` lives on the game
    record, never on the row).
    """

    return {
        "battle_id": row.battle_id,
        "battle_seed": row.battle_seed,
        "format_id": row.format_id,
        "player_id": row.player_id,
        "observation_schema_version": row.observation_schema_version,
        "observation_metadata": row.observation_metadata,
        "public_materialization": row.public_materialization,
    }


def observation_contract_from_header(header: Mapping[str, Any]) -> tuple[Any, ObservationFeatureMasks]:
    """(spec, feature_masks) as stamped in the corpus header."""

    contract = header.get("observation")
    if not isinstance(contract, Mapping):
        raise ValueError("corpus header is missing its observation contract.")
    spec = observation_spec_for_schema(str(contract["schema_version"]))
    if spec.token_count != int(contract["token_count"]):
        raise ValueError(
            f"corpus token count {contract['token_count']} does not match the "
            f"schema spec's {spec.token_count}."
        )
    if spec.categorical_feature_count != int(contract["categorical_feature_count"]):
        raise ValueError("corpus categorical width does not match the schema spec.")
    if spec.numeric_feature_count != int(contract["numeric_feature_count"]):
        raise ValueError("corpus numeric width does not match the schema spec.")
    raw_masks = contract.get("feature_masks")
    if not isinstance(raw_masks, Mapping):
        raise ValueError("corpus header is missing its feature masks.")
    masks = ObservationFeatureMasks(
        opponent_tendency_stats_block=bool(raw_masks["stats_block"]),
        exact_state=bool(raw_masks["exact_state"]),
        transition_token_budget=int(raw_masks["transition_token_budget"]),
        tier2_residuals=bool(raw_masks["tier2_residuals"]),
        tier2_investment=bool(raw_masks["tier2_investment"]),
    )
    return spec, masks


class EncoderBackend(Protocol):
    name: str

    def encode(self, row_inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Return the five observation arrays in canonical dtypes."""
        ...


def arrays_dict_from_observation_arrays(arrays: GoldenObservationArrays) -> dict[str, Any]:
    return {name: array for name, array in arrays.field_arrays()}


# ---------------------------------------------------------------------------
# python-reference backend
# ---------------------------------------------------------------------------


def _pokemon_from_metadata(entry: Mapping[str, Any]) -> ShowdownPokemon:
    stats = entry.get("stats")
    return ShowdownPokemon(
        ident=str(entry.get("ident") or ""),
        showdown_slot=str(entry.get("showdown_slot") or ""),
        species=str(entry.get("species") or ""),
        condition=entry.get("condition"),
        active=bool(entry.get("active")),
        details=entry.get("details"),
        moves=tuple(str(move) for move in entry.get("moves") or ()),
        ability=entry.get("ability"),
        item=entry.get("item"),
        stats={str(k): int(v) for k, v in stats.items()} if isinstance(stats, Mapping) else None,
        live_type_source=entry.get("live_type_source"),
    )


def _belief_from_overlay(entry: Mapping[str, Any]) -> RevealedPokemonBelief:
    """Rebuild a ``RevealedPokemonBelief`` from its overlay payload.

    The overlay (``RevealedPokemonBelief.to_overlay_payload``) serializes every
    field the encoder consumes. ``evidence`` is serialized as opaque payload
    dicts and is NOT consumed by the encode path, so it is reconstructed empty.
    """

    def _str_tuple(key: str) -> tuple[str, ...]:
        return tuple(str(value) for value in entry.get(key) or ())

    move_uses_raw = entry.get("move_uses") or ()
    move_uses = tuple(
        (str(pair[0]), int(pair[1]))
        for pair in move_uses_raw
        if isinstance(pair, Sequence) and len(pair) == 2
    )
    candidate_set_count = entry.get("candidate_set_count")
    return RevealedPokemonBelief(
        showdown_slot=str(entry.get("showdown_slot") or ""),
        species=str(entry.get("species") or ""),
        condition=entry.get("condition"),
        status=entry.get("status"),
        active=bool(entry.get("active")),
        revealed_moves=_str_tuple("revealed_moves"),
        revealed_ability=entry.get("revealed_ability"),
        revealed_item=entry.get("revealed_item"),
        ruled_out_abilities=_str_tuple("ruled_out_abilities"),
        candidate_set_count=int(candidate_set_count) if candidate_set_count is not None else None,
        uncertainty=float(entry.get("uncertainty", 1.0)),
        possible_abilities=_str_tuple("possible_abilities"),
        possible_items=_str_tuple("possible_items"),
        possible_moves=_str_tuple("possible_moves"),
        candidate_variants=tuple(dict(v) for v in entry.get("candidate_variants") or ()),
        source_metadata=entry.get("source_metadata"),
        evidence=(),
        transformed=bool(entry.get("transformed")),
        transform_species=entry.get("transform_species"),
        move_uses=move_uses,
        sleep_turns=int(entry.get("sleep_turns", 0)),
        rest_sleep=bool(entry.get("rest_sleep")),
        turns_active=int(entry.get("turns_active", 0)),
        ruled_out_items=_str_tuple("ruled_out_items"),
        item_mutated=bool(entry.get("item_mutated")),
    )


def _legal_mask_from_metadata(metadata: Mapping[str, Any]) -> tuple[bool, ...]:
    candidates = metadata.get("action_candidates")
    mask = [False] * ACTION_COUNT
    if isinstance(candidates, Sequence):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            index = candidate.get("action_index")
            if isinstance(index, int) and 0 <= index < ACTION_COUNT:
                mask[index] = bool(candidate.get("legal"))
    return tuple(mask)


def _timed_condition_turns(pm: Mapping[str, Any], slot: str) -> dict[str, int]:
    """Mirror ``showdown._timed_condition_turns`` from the materialization payload."""

    sides = pm.get("sides")
    side = sides.get(slot) if isinstance(sides, Mapping) else None
    if not isinstance(side, Mapping):
        return {}
    set_turns = side.get("sideConditionSetTurns")
    counts = side.get("sideConditions")
    if not isinstance(set_turns, Mapping) or not isinstance(counts, Mapping):
        return {}
    turn = int(pm.get("turn") or 0)
    remaining: dict[str, int] = {}
    for condition, set_turn in set_turns.items():
        if not counts.get(condition):
            continue
        remaining[str(condition)] = max(0, _TIMED_CONDITION_DURATION - (turn - int(set_turn)))
    return remaining


def _synthesized_request(
    pm: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """A minimal request carrying exactly what ``_encode_action_tokens`` reads.

    Only the active move list is consumed on the encode path
    (``_active_request(state.request)`` -> ``moves``); ``request_kind`` and the
    legal mask travel on the state directly.

    The move list merges two sanctioned sources: ``pm.selfActiveMoves``
    carries pp/maxpp/disabled but SKIPS request moves without integer PP
    fields (``local_showdown._request_active_moves``) — e.g. the single
    ``recharge`` pseudo-move on a Hyper Beam recharge turn. The metadata's
    ``action_candidates`` list every request move slot (``move_id`` +
    ``disabled``), so slots missing from the payload are synthesized from
    there without PP fields (``_move_pp_fraction`` then yields the encoder's
    own no-PP-data value, 1.0 — matching the live encode of such moves).
    """

    payload_moves = [
        dict(move)
        for move in (pm.get("selfActiveMoves") or ())
        if isinstance(move, Mapping) and isinstance(move.get("id"), str)
    ]
    move_candidates = [
        candidate
        for candidate in (metadata.get("action_candidates") or ())
        if isinstance(candidate, Mapping) and candidate.get("kind") == "move"
    ]
    moves: list[dict[str, Any]] = []
    cursor = 0
    for candidate in sorted(move_candidates, key=lambda c: int(c.get("action_index", 0))):
        slot = int(candidate.get("move_slot") or 0)
        move_name = str(candidate.get("move_name") or "")
        if move_name == f"slot:{slot}":
            # Absent request slot (the encoder's own fallback naming): stop —
            # request move lists are dense prefixes of the four slots.
            break
        move_id = str(candidate.get("move_id") or "")
        if cursor < len(payload_moves) and str(payload_moves[cursor].get("id")) == move_id:
            moves.append(payload_moves[cursor])
            cursor += 1
        else:
            moves.append({"id": move_name, "disabled": bool(candidate.get("disabled"))})
    if moves:
        return {"active": [{"moves": moves}]}
    return None


def _empty_tendency_stats(self_slot: str, opponent_slot: str) -> TendencyStats:
    """Zero-history tendency aggregates (the per-row surface carries none)."""

    return TendencyStats(
        perspective_slot=self_slot,
        opponent_slot=opponent_slot,
        opponent_switch_count=0,
        opponent_decision_opportunities=0,
        opponent_mon_tendencies=(),
        opponent_weather_reveals=(),
        blocked_on_our_attack_count=0,
        pursuit_intercept_predict_count=0,
        my_switch_turn_count=0,
    )


def state_from_row_inputs(row_inputs: Mapping[str, Any]) -> PlayerRelativeBattleState:
    """Reconstruct the encoder's input state from the sanctioned row inputs.

    History-derived fields (transition/turn-merged token streams, tendency
    aggregates) are reconstructed EMPTY: the per-row corpus surface does not
    carry the public event stream they are extracted from. The harness
    quantifies exactly which golden columns that gap affects.
    """

    metadata = row_inputs["observation_metadata"]
    pm = row_inputs["public_materialization"]
    if not isinstance(metadata, Mapping) or not isinstance(pm, Mapping):
        raise ValueError("row inputs must carry observation_metadata and public_materialization.")

    self_slot = str(metadata["showdown_slot"])
    opponent_slot = str(metadata["opponent_showdown_slot"])
    perspective = ObservationPerspective(
        player_id=str(metadata["player_id"]),
        showdown_slot=self_slot,
        opponent_showdown_slot=opponent_slot,
    )

    overlay = metadata.get("belief_view")
    if not isinstance(overlay, Mapping):
        raise ValueError("row inputs are missing the belief_view overlay.")
    belief_view = PlayerBeliefView(
        self_slot=str(overlay["self_slot"]),
        opponent_slot=str(overlay["opponent_slot"]),
        self_pokemon=tuple(_belief_from_overlay(entry) for entry in overlay.get("self_pokemon") or ()),
        opponent_pokemon=tuple(
            _belief_from_overlay(entry) for entry in overlay.get("opponent_pokemon") or ()
        ),
    )

    return PlayerRelativeBattleState(
        battle_id=str(row_inputs.get("battle_id") or metadata.get("battle_id") or ""),
        player_id=str(metadata["player_id"]),
        perspective=perspective,
        request=_synthesized_request(pm, metadata),
        request_kind=str(metadata["request_kind"]),
        self_team=tuple(_pokemon_from_metadata(entry) for entry in metadata.get("self_team") or ()),
        opponent_team=tuple(
            _pokemon_from_metadata(entry) for entry in metadata.get("opponent_team") or ()
        ),
        self_side_conditions=tuple(str(v) for v in metadata.get("self_side_conditions") or ()),
        opponent_side_conditions=tuple(
            str(v) for v in metadata.get("opponent_side_conditions") or ()
        ),
        self_side_condition_counts={
            str(k): int(v) for k, v in (metadata.get("self_side_condition_counts") or {}).items()
        },
        opponent_side_condition_counts={
            str(k): int(v) for k, v in (metadata.get("opponent_side_condition_counts") or {}).items()
        },
        self_active_boosts={
            str(k): int(v) for k, v in (metadata.get("self_active_boosts") or {}).items()
        },
        opponent_active_boosts={
            str(k): int(v) for k, v in (metadata.get("opponent_active_boosts") or {}).items()
        },
        self_active_volatiles=tuple(str(v) for v in metadata.get("self_active_volatiles") or ()),
        opponent_active_volatiles=tuple(
            str(v) for v in metadata.get("opponent_active_volatiles") or ()
        ),
        self_toxic_stage=int(metadata.get("self_toxic_stage") or 0),
        opponent_toxic_stage=int(metadata.get("opponent_toxic_stage") or 0),
        belief_view=belief_view,
        legal_action_mask=_legal_mask_from_metadata(metadata),
        recent_events=(),
        recent_public_events=tuple(str(v) for v in metadata.get("recent_public_events") or ()),
        weather=metadata.get("weather"),
        turn_number=int(metadata.get("turn_number") or 0),
        self_future_sight_turns=int(metadata.get("self_future_sight_turns") or 0),
        opponent_future_sight_turns=int(metadata.get("opponent_future_sight_turns") or 0),
        winner=None,
        transition_tokens=(),
        tendency_stats=_empty_tendency_stats(self_slot, opponent_slot),
        turn_merged_tokens=(),
        weather_turns_remaining=int(metadata.get("weather_turns_remaining") or 0),
        weather_permanent=bool(metadata.get("weather_permanent")),
        self_timed_condition_turns=_timed_condition_turns(pm, self_slot),
        opponent_timed_condition_turns=_timed_condition_turns(pm, opponent_slot),
        self_wish_pending=bool(metadata.get("self_wish_pending")),
        opponent_wish_pending=bool(metadata.get("opponent_wish_pending")),
        self_sleep_clause_used=bool(metadata.get("self_sleep_clause_used")),
        opponent_sleep_clause_used=bool(metadata.get("opponent_sleep_clause_used")),
        self_sleep_clause_blocks=bool(metadata.get("self_sleep_clause_blocks")),
        opponent_sleep_clause_blocks=bool(metadata.get("opponent_sleep_clause_blocks")),
        self_wish_turns=int(metadata.get("self_wish_turns") or 0),
        opponent_wish_turns=int(metadata.get("opponent_wish_turns") or 0),
        self_stall_counter=int(metadata.get("self_stall_counter") or 0),
        opponent_stall_counter=int(metadata.get("opponent_stall_counter") or 0),
        self_confusion_elapsed=int(metadata.get("self_confusion_elapsed") or 0),
        opponent_confusion_elapsed=int(metadata.get("opponent_confusion_elapsed") or 0),
        self_encore_elapsed=int(metadata.get("self_encore_elapsed") or 0),
        opponent_encore_elapsed=int(metadata.get("opponent_encore_elapsed") or 0),
        self_wrap_trap_elapsed=int(metadata.get("self_wrap_trap_elapsed") or 0),
        opponent_wrap_trap_elapsed=int(metadata.get("opponent_wrap_trap_elapsed") or 0),
        self_meanlook_trap=bool(metadata.get("self_meanlook_trap")),
        opponent_meanlook_trap=bool(metadata.get("opponent_meanlook_trap")),
    )


class PythonReferenceBackend:
    """Re-encode via the production Python encoder from the stored row inputs."""

    name = "python-reference"

    def __init__(self, *, showdown_root: Path | str, header: Mapping[str, Any]) -> None:
        self._spec, self._masks = observation_contract_from_header(header)
        include_turn_merged = self._spec.schema_version in {
            OBSERVATION_SCHEMA_VERSION_V2_2,
            OBSERVATION_SCHEMA_VERSION_V3,
        }
        cached_vocab = gen3_category_vocabulary(
            showdown_root, include_turn_merged=include_turn_merged
        )
        # OOV observations are mutable diagnostics. Keep corpus validation from
        # polluting the process-wide cached vocabulary used by unrelated tests.
        self._vocab = CategoryVocabulary(
            tokens=cached_vocab.tokens,
            oov_buckets=cached_vocab.oov_buckets,
            aliases=cached_vocab.aliases,
        )
        self._dex = load_showdown_dex_cached(showdown_root)

    @property
    def spec(self) -> Any:
        return self._spec

    def encode(self, row_inputs: Mapping[str, Any]) -> dict[str, Any]:
        state = state_from_row_inputs(row_inputs)
        observation = observation_from_player_state(
            state,
            category_vocab=self._vocab,
            spec=self._spec,
            dex=self._dex,
            feature_masks=self._masks,
        )
        return arrays_dict_from_observation_arrays(
            GoldenObservationArrays.from_observation(observation)
        )


# ---------------------------------------------------------------------------
# rust backend
# ---------------------------------------------------------------------------


class RustBackend:
    """Encode through the native crate's PyO3 ``encode_decision`` entry point.

    The crate returns one little-endian byte buffer per array; shapes come
    from the corpus manifest (validated upstream) and are fixed by the spec.
    """

    name = "rust"

    def __init__(self, *, tables_json: str, header: Mapping[str, Any]) -> None:
        import pokezero_search

        if not hasattr(pokezero_search, "encode_decision"):
            raise RuntimeError(
                "the installed pokezero_search wheel has no encode_decision; "
                "rebuild from rust/pokezero-search (maturin build --release)."
            )
        self._encode = pokezero_search.encode_decision
        self._tables_json = tables_json
        spec, _ = observation_contract_from_header(header)
        self._shapes = {
            "categorical_ids": (spec.token_count, spec.categorical_feature_count),
            "numeric_features": (spec.token_count, spec.numeric_feature_count),
            "token_type_ids": (spec.token_count,),
            "attention_mask": (spec.token_count,),
            "legal_action_mask": (ACTION_COUNT,),
        }

    def encode(self, row_inputs: Mapping[str, Any]) -> dict[str, Any]:
        payload = self._encode(json.dumps(row_inputs, sort_keys=True), self._tables_json)
        return _arrays_from_buffers(payload, self._shapes)


def _arrays_from_buffers(
    payload: Any, shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("the encoder must return a dict of array buffers.")
    arrays: dict[str, Any] = {}
    for name in ARRAY_NAMES:
        buffer = payload.get(name)
        if buffer is None:
            raise ValueError(f"the encoder returned no buffer for {name}.")
        arrays[name] = numpy.frombuffer(buffer, dtype=ARRAY_DTYPES[name]).reshape(shapes[name])
    return arrays


class RustFoldBackend:
    """Full-surface native encode: boundary cells from the row inputs plus the
    history cells (transition rows, tendency/stats counters, pinned Tier-2
    conclusions, transition attention extent) from the row's recorded fold
    state, consumed NATIVELY in-crate (``NativeEncoder.encode_with_fold`` —
    no ``products_payload`` Python crossing). Against golden arrays this
    backend must be ALL EXACT: any divergence is an encoder bug, not a
    stored-surface gap.
    """

    name = "rust-fold"

    def __init__(self, *, tables_json: str, header: Mapping[str, Any]) -> None:
        import pokezero_search

        if not hasattr(pokezero_search, "NativeEncoder"):
            raise RuntimeError(
                "the installed pokezero_search wheel has no NativeEncoder; "
                "rebuild from rust/pokezero-search (scripts/build_search_crate_model.sh)."
            )
        self._module = pokezero_search
        self._encoder = pokezero_search.NativeEncoder(tables_json)
        spec, _ = observation_contract_from_header(header)
        self._shapes = {
            "categorical_ids": (spec.token_count, spec.categorical_feature_count),
            "numeric_features": (spec.token_count, spec.numeric_feature_count),
            "token_type_ids": (spec.token_count,),
            "attention_mask": (spec.token_count,),
            "legal_action_mask": (ACTION_COUNT,),
        }

    def encode(self, row_inputs: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("rust-fold requires the row's fold state; use encode_with_fold.")

    def encode_with_fold(
        self, row_inputs: Mapping[str, Any], fold_state_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        fold = self._module.FoldState.from_payload(fold_state_payload)
        payload = self._encoder.encode_with_fold(
            json.dumps(row_inputs, sort_keys=True), fold
        )
        return _arrays_from_buffers(payload, self._shapes)
