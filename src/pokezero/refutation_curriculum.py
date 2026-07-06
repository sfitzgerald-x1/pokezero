"""Curriculum collection from certified G4 fragile states.

R1(d) starts an explicit fraction of collection from fragile states instead of
turn 0.  This module produces that curriculum slice as ordinary rollout JSONL:
callers can mix the emitted records with normal collection at the configured
epsilon/fraction without changing the rollout-record schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .collection import RolloutRecord, record_from_result, write_rollout_record
from .env import PokeZeroEnv
from .policy import Policy
from .refutation_mining import FRAGILE_STATE_SCHEMA_VERSION, RefutationCandidate
from .replay_branching import replay_trajectory_prefix
from .rollout import RolloutConfig, continue_rollout_from_current_state


REFUTATION_CURRICULUM_SUMMARY_SCHEMA_VERSION = "pokezero.refutation_curriculum_summary.v1"
REFUTATION_CURRICULUM_METADATA_SCHEMA_VERSION = "pokezero.refutation_curriculum_metadata.v1"


@dataclass(frozen=True)
class RefutationCurriculumConfig:
    """Controls how many fragile-state starts are materialized."""

    total_games: int
    curriculum_fraction: float
    seed_start: int = 1
    max_starts: int | None = None
    reset_policies: bool = True

    def __post_init__(self) -> None:
        if self.total_games < 0:
            raise ValueError("total_games must be non-negative.")
        if not 0.0 <= self.curriculum_fraction <= 1.0:
            raise ValueError("curriculum_fraction must be between 0.0 and 1.0.")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative.")
        if self.max_starts is not None and self.max_starts < 0:
            raise ValueError("max_starts must be non-negative when set.")


@dataclass(frozen=True)
class RefutationCurriculumSummary:
    source_record_count: int
    fragile_state_count: int
    requested_start_count: int
    emitted_count: int
    output_path: Path
    config: RefutationCurriculumConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUTATION_CURRICULUM_SUMMARY_SCHEMA_VERSION,
            "source_record_count": self.source_record_count,
            "fragile_state_count": self.fragile_state_count,
            "requested_start_count": self.requested_start_count,
            "emitted_count": self.emitted_count,
            "output_path": str(self.output_path),
            "config": {
                "total_games": self.config.total_games,
                "curriculum_fraction": self.config.curriculum_fraction,
                "seed_start": self.config.seed_start,
                "max_starts": self.config.max_starts,
                "reset_policies": self.config.reset_policies,
            },
        }


def refutation_curriculum_start_count(config: RefutationCurriculumConfig) -> int:
    """Number of fragile starts needed for the configured epsilon slice."""

    requested = int(math.ceil(config.total_games * config.curriculum_fraction))
    if config.max_starts is not None:
        requested = min(requested, config.max_starts)
    return requested


def collect_refutation_curriculum_rollouts(
    *,
    records: Iterable[RolloutRecord],
    fragile_states: Iterable[Mapping[str, Any]],
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    output_path: Path,
    config: RefutationCurriculumConfig,
) -> RefutationCurriculumSummary:
    """Materialize rollout records that start from certified fragile states.

    The replay prefix stops at the fragile decision boundary before any branch
    action is submitted, so the supplied policies choose from the fragile state
    itself.  If the requested curriculum slice is larger than the archive, rows
    are cycled deterministically; the repeat index is stamped in metadata.
    """

    source_records = tuple(records)
    rows = tuple(_certified_fragile_rows(fragile_states))
    requested_count = refutation_curriculum_start_count(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_count > 0 and not rows:
        raise ValueError("cannot collect refutation curriculum starts: fragile archive has no certified rows.")
    emitted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if requested_count == 0:
            return RefutationCurriculumSummary(
                source_record_count=len(source_records),
                fragile_state_count=len(rows),
                requested_start_count=0,
                emitted_count=0,
                output_path=output_path,
                config=config,
            )
        for curriculum_index, row in enumerate(itertools.islice(itertools.cycle(rows), requested_count)):
            record = _source_record_for_row(source_records, row)
            curriculum_record = _collect_one_curriculum_record(
                record=record,
                row=row,
                row_index=curriculum_index % len(rows),
                repeat_index=curriculum_index // len(rows),
                curriculum_index=curriculum_index,
                env_factory=env_factory,
                policies=policies,
                rollout_config=rollout_config,
                seed=config.seed_start + curriculum_index,
                reset_policies=config.reset_policies,
            )
            write_rollout_record(handle, curriculum_record)
            emitted += 1
    return RefutationCurriculumSummary(
        source_record_count=len(source_records),
        fragile_state_count=len(rows),
        requested_start_count=requested_count,
        emitted_count=emitted,
        output_path=output_path,
        config=config,
    )


def write_refutation_curriculum_summary(path: Path, summary: RefutationCurriculumSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _collect_one_curriculum_record(
    *,
    record: RolloutRecord,
    row: Mapping[str, Any],
    row_index: int,
    repeat_index: int,
    curriculum_index: int,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed: int,
    reset_policies: bool,
) -> RolloutRecord:
    candidate = _candidate_from_row(row)
    if rollout_config.format_id != record.format_id:
        raise ValueError(
            "curriculum rollout format must match the source record format: "
            f"rollout_config={rollout_config.format_id!r}, source={record.format_id!r}."
        )
    env = env_factory()
    start = perf_counter()
    try:
        prefix = replay_trajectory_prefix(
            env,
            record.trajectory,
            decision_round_count=candidate.decision_round_index,
            check_prefix_observations=False,
        )
        if prefix.terminal is not None:
            raise ValueError("cannot start curriculum from a terminal replay prefix.")
        if candidate.loser_player_id not in prefix.requested_players:
            raise ValueError(
                "replay prefix did not land on the fragile loser decision: "
                f"loser={candidate.loser_player_id!r}, requested={prefix.requested_players!r}."
            )
        available_observations = {
            player_id: env.observe(player_id)
            for player_id in prefix.requested_players
        }
        result = continue_rollout_from_current_state(
            env=env,
            policies=policies,
            config=rollout_config,
            seed=seed,
            battle_id=f"refutation-curriculum-{record.battle_id}-{candidate.decision_round_index}-{curriculum_index}",
            starting_decision_round_index=candidate.decision_round_index,
            available_observations=available_observations,
            reset_policies=reset_policies,
        )
        result.trajectory.metadata = {
            **dict(result.trajectory.metadata),
            "refutation_curriculum": _curriculum_metadata(
                row=row,
                candidate=candidate,
                source_record=record,
                row_index=row_index,
                repeat_index=repeat_index,
                curriculum_index=curriculum_index,
            ),
        }
        return record_from_result(
            result,
            policies=policies,
            elapsed_seconds=perf_counter() - start,
            belief_set_source_hash=getattr(env, "belief_set_source_hash", record.belief_set_source_hash),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _curriculum_metadata(
    *,
    row: Mapping[str, Any],
    candidate: RefutationCandidate,
    source_record: RolloutRecord,
    row_index: int,
    repeat_index: int,
    curriculum_index: int,
) -> dict[str, Any]:
    certification = _mapping(row.get("certification"))
    return {
        "schema_version": REFUTATION_CURRICULUM_METADATA_SCHEMA_VERSION,
        "source_schema_version": row.get("schema_version"),
        "source_battle_id": source_record.battle_id,
        "source_record_index": candidate.source_record_index,
        "source_seed": source_record.seed,
        "curriculum_index": curriculum_index,
        "fragile_state_index": row_index,
        "repeat_index": repeat_index,
        "mode": row.get("mode"),
        "evaluation_source": row.get("evaluation_source"),
        "champion_player_id": candidate.champion_player_id,
        "loser_player_id": candidate.loser_player_id,
        "decision_round_index": candidate.decision_round_index,
        "step_index": candidate.step_index,
        "recorded_action_index": candidate.recorded_action_index,
        "deviation_action_index": candidate.deviation_action_index,
        "flip_rate": certification.get("flip_rate"),
        "certification_seed_count": certification.get("seed_count"),
        "min_flip_rate": certification.get("min_flip_rate"),
    }


def _certified_fragile_rows(fragile_states: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for row in fragile_states:
        if row.get("schema_version") != FRAGILE_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported fragile-state schema: {row.get('schema_version')!r}")
        certification = _mapping(row.get("certification"))
        if certification.get("passed") is not True:
            continue
        _candidate_from_row(row)
        rows.append(dict(row))
    return tuple(rows)


def _source_record_for_row(records: tuple[RolloutRecord, ...], row: Mapping[str, Any]) -> RolloutRecord:
    candidate = _candidate_from_row(row)
    try:
        record = records[candidate.source_record_index]
    except IndexError as exc:
        raise ValueError(
            f"fragile-state source_record_index {candidate.source_record_index} is outside "
            f"the supplied records ({len(records)})."
        ) from exc
    if record.battle_id != candidate.battle_id:
        raise ValueError("fragile-state battle_id does not match the source record")
    if record.seed != candidate.seed:
        raise ValueError("fragile-state seed does not match the source record")
    if record.format_id != candidate.format_id:
        raise ValueError("fragile-state format_id does not match the source record")
    return record


def _candidate_from_row(row: Mapping[str, Any]) -> RefutationCandidate:
    candidate = _mapping(row.get("candidate"))
    return RefutationCandidate(
        battle_id=str(candidate["battle_id"]),
        source_record_index=int(candidate["source_record_index"]),
        seed=int(candidate["seed"]),
        format_id=str(candidate["format_id"]),
        champion_player_id=str(candidate["champion_player_id"]),
        loser_player_id=str(candidate["loser_player_id"]),
        decision_round_index=int(candidate["decision_round_index"]),
        step_index=int(candidate["step_index"]),
        recorded_action_index=int(candidate["recorded_action_index"]),
        deviation_action_index=int(candidate["deviation_action_index"]),
        branch_actions={str(player): int(action) for player, action in _mapping(candidate["branch_actions"]).items()},
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected an object")
    return value
