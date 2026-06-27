"""Opponent-pool selection helpers shared by self-play and registry previews."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_MAX_HISTORICAL_OPPONENTS = 3
CHECKPOINT_POLICY_SPEC_PREFIXES = ("linear:", "neural:")
HISTORICAL_OPPONENT_SELECTION_MODES = ("recent", "spread")


def policy_spec_identity(policy_spec: str | None) -> tuple[str, str] | None:
    """Return a comparison key for policy specs that may name checkpoint paths."""
    if policy_spec is None:
        return None
    body = str(policy_spec).strip().partition("?")[0].strip()
    lowered = body.lower()
    for prefix in CHECKPOINT_POLICY_SPEC_PREFIXES:
        if lowered.startswith(prefix):
            checkpoint_path = body[len(prefix) :].strip()
            return (prefix[:-1], str(Path(checkpoint_path).expanduser().resolve(strict=False)))
    return ("named", lowered)


def historical_opponent_policy_specs(
    checkpoint_history: Iterable[str],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
    selection_mode: str = "recent",
) -> tuple[str, ...]:
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if selection_mode not in HISTORICAL_OPPONENT_SELECTION_MODES:
        choices = ", ".join(HISTORICAL_OPPONENT_SELECTION_MODES)
        raise ValueError(f"historical opponent selection mode must be one of: {choices}.")
    if max_historical_opponents == 0:
        return ()
    current_identity = policy_spec_identity(current_policy_spec)
    historical = [
        spec
        for spec in checkpoint_history
        if current_identity is None or policy_spec_identity(spec) != current_identity
    ]
    if selection_mode == "spread":
        return _spread_policy_specs(historical, max_historical_opponents)
    return tuple(historical[-max_historical_opponents:])


def require_historical_opponent_pool_size(
    checkpoint_history: Iterable[str],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
    required_size: int | None,
    pool_label: str = "historical opponent pool",
    selection_mode: str = "recent",
) -> tuple[str, ...]:
    if required_size is None:
        return historical_opponent_policy_specs(
            checkpoint_history,
            current_policy_spec=current_policy_spec,
            max_historical_opponents=max_historical_opponents,
            selection_mode=selection_mode,
        )
    if required_size < 0:
        raise ValueError(f"{pool_label} required size must be non-negative.")
    if required_size > max_historical_opponents:
        raise ValueError(f"{pool_label} required size cannot exceed max_historical_opponents.")
    selected = historical_opponent_policy_specs(
        checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        selection_mode=selection_mode,
    )
    if len(selected) < required_size:
        raise ValueError(
            f"{pool_label} has {len(selected)} selectable opponents after current-policy exclusion; "
            f"required {required_size}."
        )
    return selected


def opponent_pool_policy_specs(
    *,
    fixed_policy_specs: Iterable[str],
    checkpoint_history: Iterable[str],
    current_policy_spec: str,
    max_historical_opponents: int,
    include_current_policy: bool = False,
    historical_selection_mode: str = "recent",
) -> tuple[str, ...]:
    pool = tuple(fixed_policy_specs) + historical_opponent_policy_specs(
        checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        selection_mode=historical_selection_mode,
    )
    if include_current_policy:
        # Mirror match: the current policy plays a copy of itself (current-vs-current),
        # so self-play happens from iteration 1 rather than only once a checkpoint has been
        # promoted into the history pool. Skip if an identical spec is already in the pool.
        current_identity = policy_spec_identity(current_policy_spec)
        if not any(policy_spec_identity(spec) == current_identity for spec in pool):
            pool = pool + (current_policy_spec,)
    return pool


def _spread_policy_specs(policy_specs: list[str], max_count: int) -> tuple[str, ...]:
    if len(policy_specs) <= max_count:
        return tuple(policy_specs)
    if max_count == 1:
        return (policy_specs[-1],)
    last_index = len(policy_specs) - 1
    selected_indices = {
        round(position * last_index / (max_count - 1))
        for position in range(max_count)
    }
    return tuple(policy_specs[index] for index in sorted(selected_indices))
