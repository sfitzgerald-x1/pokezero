"""Opponent-pool selection helpers shared by self-play and registry previews."""

from __future__ import annotations

from typing import Iterable


DEFAULT_MAX_HISTORICAL_OPPONENTS = 3


def historical_opponent_policy_specs(
    checkpoint_history: Iterable[str],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
) -> tuple[str, ...]:
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if max_historical_opponents == 0:
        return ()
    historical = [
        spec
        for spec in checkpoint_history
        if current_policy_spec is None or spec != current_policy_spec
    ]
    return tuple(historical[-max_historical_opponents:])


def require_historical_opponent_pool_size(
    checkpoint_history: Iterable[str],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
    required_size: int | None,
    pool_label: str = "historical opponent pool",
) -> tuple[str, ...]:
    if required_size is None:
        return historical_opponent_policy_specs(
            checkpoint_history,
            current_policy_spec=current_policy_spec,
            max_historical_opponents=max_historical_opponents,
        )
    if required_size < 0:
        raise ValueError(f"{pool_label} required size must be non-negative.")
    if required_size > max_historical_opponents:
        raise ValueError(f"{pool_label} required size cannot exceed max_historical_opponents.")
    selected = historical_opponent_policy_specs(
        checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
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
) -> tuple[str, ...]:
    return tuple(fixed_policy_specs) + historical_opponent_policy_specs(
        checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
    )
