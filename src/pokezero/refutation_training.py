"""Training-cache helpers for certified G4 refutations.

R1 consumes the R0 fragile-state archive as a separate, explicitly-mixed
training source.  The original rollout records remain unchanged; this module
materializes corrected examples for certified loser-seat deviations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .collection import RolloutRecord
from .dataset import (
    TrajectoryDatasetConfig,
    TrajectoryExample,
    TrainingCacheSummary,
    examples_from_record,
    write_training_cache_from_examples,
)
from .refutation_mining import FRAGILE_STATE_SCHEMA_VERSION


REFUTATION_TRAINING_TARGET_MODES = frozenset(("value", "policy-value"))

REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES = {
    "value": ("ppo", "value-only"),
    "policy-value": ("behavior-cloning", "ppo", "reward-weighted"),
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
    the action target is also replaced with the certified deviation.
    """

    resolved_dataset_config = dataset_config or TrajectoryDatasetConfig()
    resolved_config = config or RefutationTrainingConfig()
    source_records = tuple(records)
    examples: list[TrajectoryExample] = []
    for row in fragile_states:
        maybe = _example_from_fragile_state(
            records=source_records,
            row=row,
            dataset_config=resolved_dataset_config,
            config=resolved_config,
        )
        if maybe is not None:
            examples.append(maybe)
    examples.sort(key=lambda example: float(example.training_weight), reverse=True)
    if resolved_config.max_examples is not None:
        examples = examples[: resolved_config.max_examples]
    return tuple(examples)


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
    examples = refutation_training_examples(
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
        skipped_count=len(fragile_rows) - len(examples),
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        surprise_weighting=_surprise_weighting_payload(resolved_config),
        training_weight_min=weight_stats["min"],
        training_weight_max=weight_stats["max"],
        training_weight_mean=weight_stats["mean"],
        cache=cache,
    )


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


def _example_from_fragile_state(
    *,
    records: Sequence[RolloutRecord],
    row: Mapping[str, Any],
    dataset_config: TrajectoryDatasetConfig,
    config: RefutationTrainingConfig,
) -> TrajectoryExample | None:
    if row.get("schema_version") != FRAGILE_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported fragile-state schema: {row.get('schema_version')!r}")
    candidate = _mapping(row.get("candidate"), label="candidate")
    certification = _mapping(row.get("certification"), label="certification")
    if certification.get("passed") is not True:
        return None
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
    action_index = (
        _int(candidate.get("deviation_action_index"), label="candidate.deviation_action_index")
        if config.target_mode == "policy-value"
        else base.action_index
    )
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
        "training_weight": _surprise_training_weight(certification, config=config),
        "target_mode": config.target_mode,
        "mode": row.get("mode"),
    }
    return replace(
        base,
        action_index=action_index,
        return_value=certified_value,
        ppo_value_target=certified_value,
        ppo_advantage=(
            certified_value - float(base.value_estimate)
            if base.value_estimate is not None
            else None
        ),
        # The deviation was produced by search, not the rollout behavior policy.
        # Leaving this missing prevents PPO from treating the branch action as an
        # importance-sampled on-policy action unless a later R1 variant defines
        # a proper behavior probability.
        action_probability=None,
        step_metadata=metadata,
        training_weight=_surprise_training_weight(certification, config=config),
    )


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


def _int(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
