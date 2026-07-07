"""Training-cache helpers for certified G4 refutations.

R1 consumes the R0 fragile-state archive as a separate, explicitly-mixed
training source.  The original rollout records remain unchanged; this module
materializes corrected examples for certified loser-seat deviations.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT
from .collection import RolloutRecord
from .dataset import (
    TrajectoryDatasetConfig,
    TrajectoryExample,
    TrainingCacheSummary,
    examples_from_record,
    write_training_cache_from_examples,
)
from .refutation_mining import FRAGILE_STATE_SCHEMA_VERSION
from .refutation_population import REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION


POLICY_DISTRIBUTION_TARGET_MODE = "policy-distribution-value"
REFUTATION_TRAINING_TARGET_MODES = frozenset(("value", "policy-value", POLICY_DISTRIBUTION_TARGET_MODE))

REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES = {
    "value": ("ppo", "value-only"),
    "policy-value": ("behavior-cloning", "ppo", "reward-weighted"),
    POLICY_DISTRIBUTION_TARGET_MODE: ("behavior-cloning", "ppo", "reward-weighted"),
}
REFUTATION_TRAINING_CACHE_SCHEMA_VERSION = "pokezero.refutation_training_cache.v1"


@dataclass(frozen=True)
class RefutationTrainingConfig:
    """Controls how fragile-state rows become supervised training examples."""

    target_mode: str = "policy-value"
    max_examples: int | None = None
    surprise_weight_scale: float = 0.0
    surprise_weight_max: float = 4.0

    def __post_init__(self) -> None:
        if self.target_mode not in REFUTATION_TRAINING_TARGET_MODES:
            raise ValueError(
                "target_mode must be one of: "
                + ", ".join(sorted(REFUTATION_TRAINING_TARGET_MODES))
            )
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when set.")
        if self.surprise_weight_scale < 0.0:
            raise ValueError("surprise_weight_scale must be non-negative.")
        if self.surprise_weight_max < 1.0:
            raise ValueError("surprise_weight_max must be at least 1.0.")


@dataclass(frozen=True)
class RefutationTrainingSummary:
    source_record_count: int
    fragile_state_count: int
    example_count: int
    skipped_count: int
    target_mode: str
    compatible_objectives: tuple[str, ...]
    surprise_weighting: Mapping[str, Any]
    training_weight_min: float | None
    training_weight_max: float | None
    training_weight_mean: float | None
    cache: TrainingCacheSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_record_count": self.source_record_count,
            "fragile_state_count": self.fragile_state_count,
            "example_count": self.example_count,
            "skipped_count": self.skipped_count,
            "target_mode": self.target_mode,
            "compatible_objectives": list(self.compatible_objectives),
            "surprise_weighting": dict(self.surprise_weighting),
            "training_weight_min": self.training_weight_min,
            "training_weight_max": self.training_weight_max,
            "training_weight_mean": self.training_weight_mean,
            "cache": self.cache.to_dict(),
        }


def refutation_training_examples(
    *,
    records: Sequence[RolloutRecord],
    fragile_states: Iterable[Mapping[str, Any]],
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
) -> tuple[TrajectoryExample, ...]:
    """Build corrected examples from certified fragile-state rows.

    The emitted examples are from the loser perspective at the recorded decision
    point.  Their value target is the terminal-rollout expected value of the
    certified deviation, with ties/caps contributing 0. In ``policy-value`` mode
    the action target is also replaced with the certified deviation. In
    ``policy-distribution-value`` mode, the row must carry
    ``search_policy_distribution`` and one weighted example is emitted per
    non-zero action probability. When ``max_examples`` is set, rows are kept or
    dropped as a unit so distribution rows are never partially truncated.
    """

    examples, _ = _refutation_training_examples_and_skipped(
        records=records,
        fragile_states=fragile_states,
        dataset_config=dataset_config,
        config=config,
    )
    return examples


def refutation_behavior_seed_training_examples(
    *,
    records: Sequence[RolloutRecord],
    behavior_seed_manifest: Mapping[str, Any],
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
) -> tuple[TrajectoryExample, ...]:
    """Build corrected examples from an R2 behavior-seed manifest.

    Behavior-seed manifests are the compact population artifact derived from the
    fragile-state archive. They carry a certified deviation action and aggregate
    terminal counts, but not a full search distribution, so this path supports
    ``value`` and ``policy-value`` target modes only.
    """

    examples, _ = _behavior_seed_training_examples_and_skipped(
        records=records,
        behavior_seed_manifest=behavior_seed_manifest,
        dataset_config=dataset_config,
        config=config,
    )
    return examples


def write_refutation_training_cache(
    *,
    records: Sequence[RolloutRecord],
    fragile_states: Iterable[Mapping[str, Any]],
    output_path: Path,
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
    overwrite: bool = False,
) -> RefutationTrainingSummary:
    fragile_rows = tuple(fragile_states)
    resolved_config = config or RefutationTrainingConfig()
    examples, skipped_count = _refutation_training_examples_and_skipped(
        records=records,
        fragile_states=fragile_rows,
        dataset_config=dataset_config,
        config=resolved_config,
    )
    cache = write_training_cache_from_examples(
        examples,
        output_path,
        config=dataset_config,
        overwrite=overwrite,
    )
    _stamp_refutation_cache_metadata(
        cache.path,
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        surprise_weighting=_surprise_weighting_payload(resolved_config),
        training_weight_stats=_training_weight_stats(examples),
    )
    weight_stats = _training_weight_stats(examples)
    return RefutationTrainingSummary(
        source_record_count=len(records),
        fragile_state_count=len(fragile_rows),
        example_count=len(examples),
        skipped_count=skipped_count,
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        surprise_weighting=_surprise_weighting_payload(resolved_config),
        training_weight_min=weight_stats["min"],
        training_weight_max=weight_stats["max"],
        training_weight_mean=weight_stats["mean"],
        cache=cache,
    )


def write_refutation_behavior_seed_training_cache(
    *,
    records: Sequence[RolloutRecord],
    behavior_seed_manifest: Mapping[str, Any],
    output_path: Path,
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
    overwrite: bool = False,
) -> RefutationTrainingSummary:
    """Write a refutation training cache from an R2 behavior-seed manifest."""

    resolved_config = _behavior_seed_training_config(config)
    seed_rows = tuple(_fragile_rows_from_behavior_seed_manifest(behavior_seed_manifest))
    examples, skipped_count = _behavior_seed_training_examples_and_skipped(
        records=records,
        behavior_seed_manifest=behavior_seed_manifest,
        dataset_config=dataset_config,
        config=resolved_config,
    )
    cache = write_training_cache_from_examples(
        examples,
        output_path,
        config=dataset_config,
        overwrite=overwrite,
    )
    _stamp_refutation_cache_metadata(
        cache.path,
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        surprise_weighting=_surprise_weighting_payload(resolved_config),
        training_weight_stats=_training_weight_stats(examples),
    )
    weight_stats = _training_weight_stats(examples)
    return RefutationTrainingSummary(
        source_record_count=len(records),
        fragile_state_count=len(seed_rows),
        example_count=len(examples),
        skipped_count=skipped_count,
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        surprise_weighting=_surprise_weighting_payload(resolved_config),
        training_weight_min=weight_stats["min"],
        training_weight_max=weight_stats["max"],
        training_weight_mean=weight_stats["mean"],
        cache=cache,
    )


def _refutation_training_examples_and_skipped(
    *,
    records: Sequence[RolloutRecord],
    fragile_states: Iterable[Mapping[str, Any]],
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
) -> tuple[tuple[TrajectoryExample, ...], int]:
    resolved_dataset_config = dataset_config or TrajectoryDatasetConfig()
    resolved_config = config or RefutationTrainingConfig()
    source_records = tuple(records)
    row_groups: list[tuple[float, int, tuple[TrajectoryExample, ...]]] = []
    skipped_count = 0
    for row_index, row in enumerate(fragile_states):
        maybe = _examples_from_fragile_state(
            records=source_records,
            row=row,
            dataset_config=resolved_dataset_config,
            config=resolved_config,
        )
        if not maybe:
            skipped_count += 1
            continue
        row_groups.append((sum(float(example.training_weight) for example in maybe), row_index, maybe))
    examples: list[TrajectoryExample] = []
    row_groups.sort(key=lambda item: (-item[0], item[1]))
    for _, _, group in row_groups:
        if (
            resolved_config.max_examples is not None
            and len(examples) + len(group) > resolved_config.max_examples
        ):
            skipped_count += 1
            continue
        examples.extend(group)
    return tuple(examples), skipped_count


def _behavior_seed_training_examples_and_skipped(
    *,
    records: Sequence[RolloutRecord],
    behavior_seed_manifest: Mapping[str, Any],
    dataset_config: TrajectoryDatasetConfig | None = None,
    config: RefutationTrainingConfig | None = None,
) -> tuple[tuple[TrajectoryExample, ...], int]:
    resolved_config = _behavior_seed_training_config(config)
    return _refutation_training_examples_and_skipped(
        records=records,
        fragile_states=tuple(_fragile_rows_from_behavior_seed_manifest(behavior_seed_manifest)),
        dataset_config=dataset_config,
        config=resolved_config,
    )


def _behavior_seed_training_config(config: RefutationTrainingConfig | None) -> RefutationTrainingConfig:
    resolved = config or RefutationTrainingConfig()
    if resolved.target_mode == POLICY_DISTRIBUTION_TARGET_MODE:
        raise ValueError("behavior-seed training does not support policy-distribution-value targets")
    return resolved


def _fragile_rows_from_behavior_seed_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if manifest.get("schema_version") != REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported behavior-seed manifest schema: {manifest.get('schema_version')!r}")
    seeds = _sequence(manifest.get("seeds"), label="behavior_seed_manifest.seeds")
    rows = []
    for seed in seeds:
        rows.append(_fragile_row_from_behavior_seed(_mapping(seed, label="behavior_seed")))
    return tuple(rows)


def _fragile_row_from_behavior_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    seed_count = _int(seed.get("certification_seed_count"), label="behavior_seed.certification_seed_count")
    deviation_wins = _int(seed.get("deviation_wins"), label="behavior_seed.deviation_wins")
    champion_wins = _int(seed.get("champion_wins"), label="behavior_seed.champion_wins")
    ties_or_caps = _int(seed.get("ties_or_caps"), label="behavior_seed.ties_or_caps")
    if seed_count != deviation_wins + champion_wins + ties_or_caps:
        raise ValueError("behavior_seed terminal counts must sum to certification_seed_count")
    return {
        "schema_version": FRAGILE_STATE_SCHEMA_VERSION,
        "mode": _required_str(seed.get("mode"), label="behavior_seed.mode"),
        "candidate": {
            "source_record_index": _int(seed.get("source_record_index"), label="behavior_seed.source_record_index"),
            "battle_id": _required_str(seed.get("battle_id"), label="behavior_seed.battle_id"),
            "seed": _int(seed.get("seed"), label="behavior_seed.seed"),
            "format_id": _required_str(seed.get("format_id"), label="behavior_seed.format_id"),
            "champion_player_id": _required_str(seed.get("champion_player_id"), label="behavior_seed.champion_player_id"),
            "loser_player_id": _required_str(seed.get("loser_player_id"), label="behavior_seed.loser_player_id"),
            "decision_round_index": _int(seed.get("decision_round_index"), label="behavior_seed.decision_round_index"),
            "step_index": _int(seed.get("step_index"), label="behavior_seed.step_index"),
            "recorded_action_index": _int(seed.get("recorded_action_index"), label="behavior_seed.recorded_action_index"),
            "deviation_action_index": _int(seed.get("deviation_action_index"), label="behavior_seed.deviation_action_index"),
        },
        "certification": {
            "passed": True,
            "seed_count": seed_count,
            "deviation_wins": deviation_wins,
            "champion_wins": champion_wins,
            "ties_or_caps": ties_or_caps,
            "flip_rate": _float(seed.get("flip_rate"), label="behavior_seed.flip_rate"),
        },
    }


def _stamp_refutation_cache_metadata(
    cache_path: Path,
    *,
    target_mode: str,
    compatible_objectives: Sequence[str],
    surprise_weighting: Mapping[str, Any],
    training_weight_stats: Mapping[str, float | None],
) -> None:
    metadata_path = cache_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["refutation_training"] = {
        "schema_version": REFUTATION_TRAINING_CACHE_SCHEMA_VERSION,
        "target_mode": target_mode,
        "compatible_objectives": list(compatible_objectives),
        "surprise_weighting": dict(surprise_weighting),
        "training_weight_stats": dict(training_weight_stats),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _examples_from_fragile_state(
    *,
    records: Sequence[RolloutRecord],
    row: Mapping[str, Any],
    dataset_config: TrajectoryDatasetConfig,
    config: RefutationTrainingConfig,
) -> tuple[TrajectoryExample, ...]:
    if row.get("schema_version") != FRAGILE_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported fragile-state schema: {row.get('schema_version')!r}")
    candidate = _mapping(row.get("candidate"), label="candidate")
    certification = _mapping(row.get("certification"), label="certification")
    if certification.get("passed") is not True:
        return ()
    source_record_index = _int(candidate.get("source_record_index"), label="candidate.source_record_index")
    if source_record_index < 0 or source_record_index >= len(records):
        raise ValueError("candidate.source_record_index is outside the supplied records")
    record = records[source_record_index]
    if str(candidate.get("battle_id")) != record.battle_id:
        raise ValueError("fragile-state battle_id does not match the source record")
    step_index = _int(candidate.get("step_index"), label="candidate.step_index")
    if step_index < 0 or step_index >= len(record.trajectory.steps):
        raise ValueError("candidate.step_index is outside the source record trajectory")
    step = record.trajectory.steps[step_index]
    loser_player_id = str(candidate.get("loser_player_id"))
    if step.player_id != loser_player_id:
        raise ValueError("fragile-state loser_player_id does not match the source step")
    base_examples = tuple(examples_from_record(record, config=dataset_config))
    base = base_examples[step_index]
    certified_value = _certified_loser_value(row, candidate=candidate, certification=certification)
    base_weight = _surprise_training_weight(certification, config=config)
    action_targets = _action_targets_for_mode(
        row=row,
        base=base,
        candidate=candidate,
        config=config,
    )
    examples: list[TrajectoryExample] = []
    for action_index, policy_target_probability in action_targets:
        metadata = dict(base.step_metadata or {})
        # Training caches do not currently persist step_metadata; this provenance is
        # available to direct callers and mirrors the source fragile-state archive.
        metadata["refutation_training"] = {
            "source_record_index": source_record_index,
            "step_index": step_index,
            "decision_round_index": candidate.get("decision_round_index"),
            "recorded_action_index": candidate.get("recorded_action_index"),
            "deviation_action_index": candidate.get("deviation_action_index"),
            "certified_value": certified_value,
            "flip_rate": certification.get("flip_rate"),
            "training_weight": base_weight * policy_target_probability,
            "target_mode": config.target_mode,
            "mode": row.get("mode"),
            "policy_target_probability": policy_target_probability,
        }
        examples.append(
            replace(
                base,
                action_index=action_index,
                return_value=certified_value,
                ppo_value_target=certified_value,
                ppo_advantage=(
                    certified_value - float(base.value_estimate)
                    if base.value_estimate is not None
                    else None
                ),
                # The deviation/search target was not produced by the rollout behavior policy.
                # Leaving this missing prevents PPO from treating the target action as an
                # importance-sampled on-policy action unless a later R1 variant defines
                # a proper behavior probability.
                action_probability=None,
                step_metadata=metadata,
                training_weight=base_weight * policy_target_probability,
            )
        )
    return tuple(examples)


def _action_targets_for_mode(
    *,
    row: Mapping[str, Any],
    base: TrajectoryExample,
    candidate: Mapping[str, Any],
    config: RefutationTrainingConfig,
) -> tuple[tuple[int, float], ...]:
    if config.target_mode == "value":
        return ((base.action_index, 1.0),)
    if config.target_mode == "policy-value":
        return ((_int(candidate.get("deviation_action_index"), label="candidate.deviation_action_index"), 1.0),)
    if config.target_mode == POLICY_DISTRIBUTION_TARGET_MODE:
        return _search_policy_distribution_targets(row, base=base)
    raise ValueError(f"unsupported target mode: {config.target_mode!r}")


def _search_policy_distribution_targets(
    row: Mapping[str, Any],
    *,
    base: TrajectoryExample,
) -> tuple[tuple[int, float], ...]:
    raw_distribution = row.get("search_policy_distribution")
    if raw_distribution is None:
        raise ValueError("policy-distribution-value mode requires search_policy_distribution on each fragile row")
    weights_by_action = [0.0] * ACTION_COUNT
    if isinstance(raw_distribution, Mapping):
        for raw_action, raw_weight in raw_distribution.items():
            action_index = _int(raw_action, label="search_policy_distribution action")
            if action_index < 0 or action_index >= ACTION_COUNT:
                raise ValueError("search_policy_distribution action is outside the action space")
            weights_by_action[action_index] += _nonnegative_finite_float(
                raw_weight,
                label=f"search_policy_distribution[{action_index}]",
            )
    else:
        values = _sequence(raw_distribution, label="search_policy_distribution")
        if len(values) != ACTION_COUNT:
            raise ValueError(f"search_policy_distribution must contain {ACTION_COUNT} entries")
        weights_by_action = [
            _nonnegative_finite_float(value, label=f"search_policy_distribution[{index}]")
            for index, value in enumerate(values)
        ]
    targets = []
    for action_index, weight in enumerate(weights_by_action):
        if weight <= 0.0:
            continue
        if not base.legal_action_mask[action_index]:
            raise ValueError(f"search_policy_distribution assigns weight to illegal action {action_index}")
        targets.append((action_index, weight))
    total = sum(weight for _, weight in targets)
    if total <= 0.0:
        raise ValueError("search_policy_distribution must assign positive mass to at least one legal action")
    return tuple(
        (action_index, weight / total)
        for action_index, weight in sorted(targets, key=lambda item: (-item[1], item[0]))
    )


def _nonnegative_finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise ValueError(f"{label} must be a sequence")
    return value


def _surprise_training_weight(
    certification: Mapping[str, Any],
    *,
    config: RefutationTrainingConfig,
) -> float:
    if config.surprise_weight_scale <= 0.0:
        return 1.0
    flip_rate = float(certification.get("flip_rate", 0.0))
    min_flip_rate = float(certification.get("min_flip_rate", 0.60))
    denominator = max(1e-9, 1.0 - min_flip_rate)
    normalized_surprise = max(0.0, (flip_rate - min_flip_rate) / denominator)
    return min(
        float(config.surprise_weight_max),
        1.0 + (float(config.surprise_weight_scale) * normalized_surprise),
    )


def _surprise_weighting_payload(config: RefutationTrainingConfig) -> dict[str, Any]:
    mode = "certification-flip-rate" if config.surprise_weight_scale > 0.0 else "none"
    return {
        "mode": mode,
        "scale": float(config.surprise_weight_scale),
        "max": float(config.surprise_weight_max),
        "field": "training_weights",
    }


def _training_weight_stats(examples: Sequence[TrajectoryExample]) -> dict[str, float | None]:
    if not examples:
        return {"min": None, "max": None, "mean": None}
    weights = tuple(float(example.training_weight) for example in examples)
    return {
        "min": min(weights),
        "max": max(weights),
        "mean": sum(weights) / len(weights),
    }


def _certified_loser_value(
    row: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    certification: Mapping[str, Any],
) -> float:
    loser_player_id = str(candidate.get("loser_player_id"))
    champion_player_id = str(candidate.get("champion_player_id"))
    terminal_results = row.get("terminal_results")
    if isinstance(terminal_results, list) and terminal_results:
        score = 0.0
        count = 0
        for result in terminal_results:
            if not isinstance(result, Mapping):
                continue
            winner = result.get("winner")
            if winner == loser_player_id:
                score += 1.0
            elif winner == champion_player_id:
                score -= 1.0
            count += 1
        if count:
            return _clip(score / count)
    deviation_wins = _int(certification.get("deviation_wins"), label="certification.deviation_wins")
    champion_wins = _int(certification.get("champion_wins"), label="certification.champion_wins")
    seed_count = _int(certification.get("seed_count"), label="certification.seed_count")
    if seed_count <= 0:
        raise ValueError("certification.seed_count must be positive")
    return _clip((deviation_wins - champion_wins) / seed_count)


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"fragile-state {label} must be an object")
    return value


def _required_str(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _int(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result
