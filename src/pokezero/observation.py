"""Versioned fixed-shape observation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .actions import ACTION_COUNT

OBSERVATION_SCHEMA_VERSION = "pokezero.observation.v1"
SHOWDOWN_PLAYER_SLOTS = ("p1", "p2")
FIELD_TOKEN_COUNT = 1
SELF_POKEMON_TOKEN_COUNT = 6
OPPONENT_POKEMON_TOKEN_COUNT = 6
ACTION_CANDIDATE_TOKEN_COUNT = ACTION_COUNT
RECENT_EVENT_TOKEN_COUNT = 24


@dataclass(frozen=True)
class ObservationSpec:
    """Uniform token feature widths for efficient batching.

    Different token sections use different feature subsets; unused categorical
    or numeric columns should be padded by the encoder.
    """

    categorical_feature_count: int
    numeric_feature_count: int
    recent_event_token_count: int = RECENT_EVENT_TOKEN_COUNT

    @property
    def token_count(self) -> int:
        return (
            FIELD_TOKEN_COUNT
            + SELF_POKEMON_TOKEN_COUNT
            + OPPONENT_POKEMON_TOKEN_COUNT
            + ACTION_CANDIDATE_TOKEN_COUNT
            + self.recent_event_token_count
        )


@dataclass(frozen=True)
class ObservationPerspective:
    """Debug/provenance metadata for a player-relative observation.

    Model tensors are normalized to self/opponent sections. Raw Showdown seats
    are retained only so harnesses can audit normalization and submit selected
    actions back to the correct protocol side.
    """

    player_id: str
    showdown_slot: str
    opponent_showdown_slot: str

    def __post_init__(self) -> None:
        _require_showdown_slot("showdown_slot", self.showdown_slot)
        _require_showdown_slot("opponent_showdown_slot", self.opponent_showdown_slot)
        if self.showdown_slot == self.opponent_showdown_slot:
            raise ValueError("showdown_slot and opponent_showdown_slot must differ.")

    @classmethod
    def from_showdown_slot(cls, player_id: str, showdown_slot: str) -> "ObservationPerspective":
        return cls(
            player_id=player_id,
            showdown_slot=showdown_slot,
            opponent_showdown_slot=opponent_showdown_slot(showdown_slot),
        )


@dataclass(frozen=True)
class PokeZeroObservationV0:
    categorical_ids: Any
    numeric_features: Any
    token_type_ids: Any
    attention_mask: Any
    legal_action_mask: Any
    perspective: ObservationPerspective | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def validate(self, spec: ObservationSpec) -> None:
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema version: {self.schema_version!r}.")
        _require_outer_length("categorical_ids", self.categorical_ids, spec.token_count)
        _require_outer_length("numeric_features", self.numeric_features, spec.token_count)
        _require_outer_length("token_type_ids", self.token_type_ids, spec.token_count)
        _require_outer_length("attention_mask", self.attention_mask, spec.token_count)
        _require_outer_length("legal_action_mask", self.legal_action_mask, ACTION_COUNT)
        _require_inner_length("categorical_ids", self.categorical_ids, spec.categorical_feature_count)
        _require_inner_length("numeric_features", self.numeric_features, spec.numeric_feature_count)


def opponent_showdown_slot(showdown_slot: str) -> str:
    _require_showdown_slot("showdown_slot", showdown_slot)
    return "p2" if showdown_slot == "p1" else "p1"


def _require_showdown_slot(name: str, value: str) -> None:
    if value not in SHOWDOWN_PLAYER_SLOTS:
        allowed = ", ".join(SHOWDOWN_PLAYER_SLOTS)
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")


def _require_outer_length(name: str, values: Any, expected: int) -> None:
    actual = _dimension(values, 0)
    if actual != expected:
        raise ValueError(f"{name} must contain {expected} values, got {actual}.")


def _require_inner_length(name: str, rows: Any, expected: int) -> None:
    width = _dimension(rows, 1)
    if width is not None:
        if width != expected:
            raise ValueError(f"{name} rows must contain {expected} values, got {width}.")
        return
    for index, row in enumerate(rows):
        if len(row) != expected:
            raise ValueError(f"{name}[{index}] must contain {expected} values, got {len(row)}.")


def _dimension(values: Any, axis: int) -> int | None:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) > axis:
        return int(shape[axis])
    if axis == 0:
        return len(values)
    return None
