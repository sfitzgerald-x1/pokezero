"""Import normalized replay decisions into rollout JSONL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

from .collection import RolloutRecord, write_rollout_record
from .trajectory import trajectory_from_dict

NORMALIZED_REPLAY_SCHEMA_VERSION = "pokezero.normalized_replay.v1"
REPLAY_IMPORT_SCHEMA_VERSION = "pokezero.replay_import.v1"
DEFAULT_REPLAY_POLICY_ID = "replay-human"


@dataclass(frozen=True)
class ReplayImportResult:
    output_path: Path
    input_paths: tuple[Path, ...]
    records_written: int
    elapsed_seconds: float
    append: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_IMPORT_SCHEMA_VERSION,
            "output_path": str(self.output_path),
            "input_paths": [str(path) for path in self.input_paths],
            "records_written": self.records_written,
            "elapsed_seconds": self.elapsed_seconds,
            "append": self.append,
        }


def import_replay_files(
    input_paths: Iterable[Path],
    *,
    output_path: Path,
    append: bool = False,
) -> ReplayImportResult:
    """Convert normalized replay JSON files into training rollout JSONL."""
    paths = tuple(input_paths)
    if not paths:
        raise ValueError("at least one replay input path is required.")

    start = perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_path = output_path if append else _temporary_output_path(output_path)
    records_written = 0
    try:
        with write_path.open("a" if append else "w", encoding="utf-8") as handle:
            for path in paths:
                payload = _read_json(path)
                record = rollout_record_from_normalized_replay(payload)
                write_rollout_record(handle, record)
                records_written += 1
        if not append:
            write_path.replace(output_path)
    except Exception:
        if not append:
            write_path.unlink(missing_ok=True)
        raise

    return ReplayImportResult(
        output_path=output_path,
        input_paths=paths,
        records_written=records_written,
        elapsed_seconds=perf_counter() - start,
        append=append,
    )


def rollout_record_from_normalized_replay(payload: Mapping[str, Any]) -> RolloutRecord:
    """Build a rollout record from the normalized replay import schema."""
    if payload.get("schema_version") != NORMALIZED_REPLAY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported normalized replay schema: {payload.get('schema_version')!r}.")

    trajectory = trajectory_from_dict(
        {
            "battle_id": str(payload["battle_id"]),
            "format_id": str(payload["format_id"]),
            "seed": int(payload.get("seed", 0)),
            "metadata": _mapping(payload.get("metadata", {})),
            "terminal": _mapping(payload["terminal"]),
            "steps": _sequence(payload["steps"]),
        }
    )
    if trajectory.terminal is None:
        raise ValueError("normalized replay payload must include terminal state.")
    if not trajectory.steps:
        raise ValueError("normalized replay payload must include at least one trajectory step.")

    policy_ids = _policy_ids(payload.get("policy_ids"), trajectory.players())
    decision_round_count = int(payload.get("decision_round_count", _decision_round_count(trajectory)))
    if decision_round_count <= 0:
        raise ValueError("decision_round_count must be positive.")

    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids=policy_ids,
        decision_round_count=decision_round_count,
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def normalized_replay_payload_from_rollout_record(record: RolloutRecord) -> dict[str, Any]:
    """Return a normalized replay payload equivalent to an existing rollout record."""
    trajectory = record.trajectory
    return {
        "schema_version": NORMALIZED_REPLAY_SCHEMA_VERSION,
        "battle_id": record.battle_id,
        "format_id": record.format_id,
        "seed": record.seed,
        "policy_ids": dict(record.policy_ids),
        "decision_round_count": record.decision_round_count,
        "elapsed_seconds": record.elapsed_seconds,
        "metadata": dict(trajectory.metadata),
        "terminal": {
            "winner": trajectory.terminal.winner if trajectory.terminal else None,
            "turn_count": trajectory.terminal.turn_count if trajectory.terminal else 0,
            "capped": trajectory.terminal.capped if trajectory.terminal else False,
        },
        "steps": [
            {
                "player_id": step.player_id,
                "turn_index": step.turn_index,
                "observation": {
                    "schema_version": step.observation.schema_version,
                    "categorical_ids": step.observation.categorical_ids,
                    "numeric_features": step.observation.numeric_features,
                    "token_type_ids": step.observation.token_type_ids,
                    "attention_mask": step.observation.attention_mask,
                    "legal_action_mask": step.observation.legal_action_mask,
                    "perspective": (
                        {
                            "player_id": step.observation.perspective.player_id,
                            "showdown_slot": step.observation.perspective.showdown_slot,
                            "opponent_showdown_slot": step.observation.perspective.opponent_showdown_slot,
                        }
                        if step.observation.perspective is not None
                        else None
                    ),
                    "metadata": dict(step.observation.metadata),
                },
                "legal_action_mask": list(step.legal_action_mask),
                "action_index": step.action_index,
                "reward": step.reward,
                "opponent_action_index": step.opponent_action_index,
                "action_probability": step.action_probability,
                "value_estimate": step.value_estimate,
                "metadata": dict(step.metadata),
            }
            for step in trajectory.steps
        ],
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle))


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _decision_round_count(trajectory) -> int:
    return len({step.turn_index for step in trajectory.steps})


def _policy_ids(value: Any, players: tuple[str, ...]) -> dict[str, str]:
    if value is None:
        return {player: DEFAULT_REPLAY_POLICY_ID for player in players}
    policy_ids = {str(player): str(policy_id) for player, policy_id in _mapping(value).items()}
    missing = sorted(set(players) - set(policy_ids))
    if missing:
        raise ValueError(f"policy_ids missing player(s): {', '.join(missing)}.")
    return policy_ids


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("expected JSON array payload.")
    return tuple(value)
