"""Reusable padding helpers for fixed-shape observation tensors."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_ScalarShape = type[bool] | type[int] | type[float]
_Shape = _ScalarShape | tuple[str, int, "_Shape"] | tuple[str, int]


def zeros_like(value: Any) -> Any:
    """Return a cached immutable zero tree with the same structure as ``value``."""

    return _zeros_from_shape(_shape_of(value))


def _shape_of(value: Any) -> _Shape:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if not value:
        return ("tuple", 0)
    # Observation fields are rectangular by schema validation. Follow the first
    # child only so padding lookup is O(depth), not O(tokens * features).
    return ("tuple", len(value), _shape_of(value[0]))


@lru_cache(maxsize=64)
def _zeros_from_shape(shape: _Shape) -> Any:
    if shape is bool:
        return False
    if shape is int:
        return 0
    if shape is float:
        return 0.0
    if len(shape) == 2:
        return ()
    _, length, child_shape = shape
    child = _zeros_from_shape(child_shape)
    return tuple(child for _ in range(length))
