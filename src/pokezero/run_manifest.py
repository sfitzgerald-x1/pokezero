"""Shared manifest helpers for self-play run metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def path_value(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def opponent_pool_config_dict(
    *,
    fixed_opponent_policy_specs: Iterable[str],
    max_historical_opponents: int,
    promotion_registry_path: Path | None,
    promotion_pool_registry_path: Path | None,
    required_promoted_opponent_pool_size: int | None,
    promoted_checkpoint_policy_specs: Iterable[str] = (),
    historical_opponent_selection: str = "recent",
) -> dict[str, Any]:
    return {
        "fixed_opponent_policy_specs": [str(spec) for spec in fixed_opponent_policy_specs],
        "max_historical_opponents": max_historical_opponents,
        "historical_opponent_selection": historical_opponent_selection,
        "promotion_registry_path": path_value(promotion_registry_path),
        "promotion_pool_registry_path": path_value(promotion_pool_registry_path),
        "required_promoted_opponent_pool_size": required_promoted_opponent_pool_size,
        "promoted_checkpoint_policy_specs": [str(spec) for spec in promoted_checkpoint_policy_specs],
    }


def auto_promotion_config_dict(
    *,
    enabled: bool,
    registry_path: Path | None,
    artifact_dir: Path | None,
    label_prefix: str | None,
    notes: str | None,
    allow_duplicate: bool,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "registry_path": path_value(registry_path),
        "artifact_dir": path_value(artifact_dir),
        "label_prefix": label_prefix,
        "notes": notes,
        "allow_duplicate": allow_duplicate,
    }
