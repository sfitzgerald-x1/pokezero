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
