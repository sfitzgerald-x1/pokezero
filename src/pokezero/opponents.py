"""Opponent-pool selection helpers shared by self-play and registry previews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .observation import require_current_observation_schema


DEFAULT_MAX_HISTORICAL_OPPONENTS = 3
CHECKPOINT_POLICY_SPEC_PREFIXES = ("linear:", "neural:")
HISTORICAL_OPPONENT_SELECTION_MODES = ("recent", "spread")
LEGACY_CHECKPOINT_FILTER_MODES = ("reject", "drop")
LEGACY_CHECKPOINT_FAMILY_MARKERS = ("no-belief", "no_belief", "pre-v2", "pre_v2")


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


def checkpoint_policy_spec_observation_schema(policy_spec: str) -> str:
    """Return the stamped observation schema for a checkpoint policy spec.

    This is intentionally metadata-only: linear checkpoints are JSON and neural
    checkpoints use the cheap model-config loader instead of constructing a policy.
    """
    body = str(policy_spec).strip().partition("?")[0].strip()
    lowered = body.lower()
    if lowered.startswith("linear:"):
        checkpoint_path = Path(body[len("linear:") :].strip()).expanduser()
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        try:
            return str(payload["observation_schema_version"])
        except KeyError as exc:
            raise ValueError(
                f"linear checkpoint policy spec {policy_spec!r} has no observation_schema_version"
            ) from exc
    if lowered.startswith("neural:"):
        from .neural_policy import load_transformer_model_config

        checkpoint_path = Path(body[len("neural:") :].strip()).expanduser()
        return str(load_transformer_model_config(checkpoint_path).observation_schema_version)
    raise ValueError(f"policy spec is not a checkpoint policy spec: {policy_spec!r}")


def _checkpoint_path_from_policy_spec(policy_spec: str) -> Path | None:
    body = str(policy_spec).strip().partition("?")[0].strip()
    lowered = body.lower()
    for prefix in CHECKPOINT_POLICY_SPEC_PREFIXES:
        if lowered.startswith(prefix):
            return Path(body[len(prefix) :].strip()).expanduser()
    return None


def _checkpoint_policy_spec_metadata(policy_spec: str, checkpoint_path: Path) -> dict[str, object] | None:
    body = str(policy_spec).strip().partition("?")[0].strip()
    lowered = body.lower()
    if lowered.startswith("linear:") and checkpoint_path.suffix.lower() == ".json":
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    for sidecar_path in _checkpoint_metadata_sidecar_paths(checkpoint_path):
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _checkpoint_metadata_sidecar_paths(checkpoint_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    if checkpoint_path.suffix:
        paths.append(checkpoint_path.with_suffix(".json"))
    paths.append(Path(f"{checkpoint_path}.json"))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return tuple(unique)


def _legacy_marker_reason(value: object, *, source: str) -> str | None:
    if value is None:
        return None
    lowered = str(value).lower()
    for marker in LEGACY_CHECKPOINT_FAMILY_MARKERS:
        if marker in lowered:
            return f"{source} contains {marker!r}"
    return None


def is_current_family_checkpoint_policy_spec(policy_spec: str) -> bool:
    """Return whether a checkpoint spec belongs to the supported v2+ family."""
    family_disqualifier = checkpoint_policy_spec_family_disqualifier(policy_spec)
    if family_disqualifier is not None:
        raise ValueError(
            f"checkpoint opponent {policy_spec!r} belongs to a legacy comparison family: "
            f"{family_disqualifier}"
        )
    schema_version = checkpoint_policy_spec_observation_schema(policy_spec)
    require_current_observation_schema(
        schema_version,
        context=f"checkpoint opponent {policy_spec!r}",
    )
    return True


def checkpoint_policy_spec_family_disqualifier(policy_spec: str) -> str | None:
    """Return a legacy-family reason for otherwise-readable checkpoint specs.

    Observation-schema v2+ is necessary but not sufficient for current strength
    eval pools: old no-belief/pre-v2 lineages are retained as historical
    baselines, not as frozen opponents for new runs. Prefer adjacent checkpoint
    metadata when present, and fall back to path/name markers so managed
    artifacts without sidecars still fail loudly.
    """
    checkpoint_path = _checkpoint_path_from_policy_spec(policy_spec)
    if checkpoint_path is None:
        return None
    path_reason = _legacy_marker_reason(checkpoint_path.name, source="filename")
    if path_reason is not None:
        return path_reason
    metadata = _checkpoint_policy_spec_metadata(policy_spec, checkpoint_path)
    if metadata is None:
        return None
    for key in ("input_family", "name", "parent", "continued_from", "lineage", "run_id"):
        value = metadata.get(key)
        reason = _legacy_marker_reason(value, source=f"metadata.{key}")
        if reason is not None:
            return reason
    return None


def current_family_checkpoint_policy_specs(
    checkpoint_history: Iterable[str],
    *,
    legacy_mode: str = "reject",
) -> tuple[str, ...]:
    """Filter checkpoint history to supported v2+ checkpoints.

    ``legacy_mode='reject'`` is for strength-eval setup where a stale opponent
    should fail loudly. ``legacy_mode='drop'`` is for mixed historical registries
    where older no-belief/pre-v2 checkpoints should simply be unavailable as
    frozen opponents.
    """
    if legacy_mode not in LEGACY_CHECKPOINT_FILTER_MODES:
        choices = ", ".join(LEGACY_CHECKPOINT_FILTER_MODES)
        raise ValueError(f"legacy checkpoint filter mode must be one of: {choices}.")
    selected: list[str] = []
    rejected: list[tuple[str, str]] = []
    for spec in checkpoint_history:
        try:
            is_current_family_checkpoint_policy_spec(spec)
        except OSError as exc:
            if legacy_mode == "reject":
                rejected.append((str(spec), str(exc)))
            else:
                # Keep unknown/missing non-legacy specs visible so registry
                # verification can report the broken artifact instead of
                # silently shrinking the opponent pool.
                selected.append(str(spec))
            continue
        except ValueError as exc:
            if legacy_mode == "reject":
                rejected.append((str(spec), str(exc)))
            continue
        selected.append(str(spec))
    if rejected:
        details = "; ".join(f"{spec}: {reason}" for spec, reason in rejected)
        raise ValueError(
            "legacy or unreadable checkpoint opponents are not allowed in current-family "
            f"strength evals: {details}"
        )
    return tuple(selected)


def current_family_historical_opponent_policy_specs(
    checkpoint_history: Iterable[str],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
    selection_mode: str = "recent",
    legacy_mode: str = "reject",
) -> tuple[str, ...]:
    """Select historical opponents after enforcing the v2+ checkpoint family."""
    return historical_opponent_policy_specs(
        current_family_checkpoint_policy_specs(checkpoint_history, legacy_mode=legacy_mode),
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        selection_mode=selection_mode,
    )


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
