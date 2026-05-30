"""Versioned fixed-shape observation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ACTION_COUNT

OBSERVATION_SCHEMA_VERSION = "pokezero.observation.v0"
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
class PokeZeroObservationV0:
    categorical_ids: Any
    numeric_features: Any
    token_type_ids: Any
    attention_mask: Any
    legal_action_mask: Any
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
