"""Training-cache helpers for certified G4 refutations.

R1 consumes the R0 fragile-state archive as a separate, explicitly-mixed
training source.  The original rollout records remain unchanged; this module
materializes corrected examples for certified loser-seat deviations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class RefutationTrainingConfig:
    """Controls how fragile-state rows become supervised training examples."""

    target_mode: str = "policy-value"
    max_examples: int | None = None

    def __post_init__(self) -> None:
        if self.target_mode not in REFUTATION_TRAINING_TARGET_MODES:
            raise ValueError(
                "target_mode must be one of: "
                + ", ".join(sorted(REFUTATION_TRAINING_TARGET_MODES))
            )
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when set.")


@dataclass(frozen=True)
class RefutationTrainingSummary:
    source_record_count: int
    fragile_state_count: int
    example_count: int
    skipped_count: int
    target_mode: str
    compatible_objectives: tuple[str, ...]
    cache: TrainingCacheSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_record_count": self.source_record_count,
            "fragile_state_count": self.fragile_state_count,
            "example_count": self.example_count,
            "skipped_count": self.skipped_count,
            "target_mode": self.target_mode,
            "compatible_objectives": list(self.compatible_objectives),
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
        if resolved_config.max_examples is not None and len(examples) >= resolved_config.max_examples:
            break
        maybe = _example_from_fragile_state(
            records=source_records,
            row=row,
            dataset_config=resolved_dataset_config,
            config=resolved_config,
        )
        if maybe is not None:
            examples.append(maybe)
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
    return RefutationTrainingSummary(
        source_record_count=len(records),
        fragile_state_count=len(fragile_rows),
        example_count=len(examples),
        skipped_count=len(fragile_rows) - len(examples),
        target_mode=resolved_config.target_mode,
        compatible_objectives=REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[resolved_config.target_mode],
        cache=cache,
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
    )


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
