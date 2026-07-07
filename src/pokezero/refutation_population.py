"""R2 population seed artifacts derived from certified refutations.

Certified fragile-state rows are useful outside the direct training-cache path:
they identify concrete decision boundaries where a candidate policy can be
distilled, admitted, or probed by the population tooling.  This module keeps the
interface artifact-only; it does not run admission, schedule training, or change
any evaluation opponent policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .refutation_mining import FRAGILE_STATE_SCHEMA_VERSION, TERMINAL_ROLLOUT_EVALUATION_SOURCE


REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION = "pokezero.refutation_behavior_seed_manifest.v1"


@dataclass(frozen=True)
class RefutationBehaviorSeedConfig:
    """Controls which certified refutations become R2 behavior seeds."""

    max_seeds: int | None = None
    min_flip_rate: float = 0.0
    mode: str | None = None

    def __post_init__(self) -> None:
        if self.max_seeds is not None and self.max_seeds <= 0:
            raise ValueError("max_seeds must be positive when set.")
        if not 0.0 <= self.min_flip_rate <= 1.0:
            raise ValueError("min_flip_rate must be between 0 and 1.")
        if self.mode is not None and self.mode not in {"oracle", "fair"}:
            raise ValueError("mode must be 'oracle', 'fair', or None.")


@dataclass(frozen=True)
class RefutationBehaviorSeedManifest:
    """A reproducible seed list for R2 candidate/population work."""

    config: RefutationBehaviorSeedConfig
    source_digest: str
    source_row_count: int
    skipped_count: int
    seeds: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION,
            "source_schema_version": FRAGILE_STATE_SCHEMA_VERSION,
            "source_digest": self.source_digest,
            "source_row_count": self.source_row_count,
            "seed_count": len(self.seeds),
            "skipped_count": self.skipped_count,
            "config": {
                "max_seeds": self.config.max_seeds,
                "min_flip_rate": self.config.min_flip_rate,
                "mode": self.config.mode,
            },
            "seeds": [dict(seed) for seed in self.seeds],
        }


def build_refutation_behavior_seed_manifest(
    fragile_states: Iterable[Mapping[str, Any]],
    *,
    config: RefutationBehaviorSeedConfig | None = None,
) -> RefutationBehaviorSeedManifest:
    """Build an ordered R2 seed manifest from certified fragile-state rows."""

    resolved_config = config or RefutationBehaviorSeedConfig()
    source_rows = tuple(fragile_states)
    seeds: list[dict[str, Any]] = []
    skipped_count = 0
    for source_row_index, row in enumerate(source_rows):
        maybe = _seed_from_row(row, source_row_index=source_row_index, config=resolved_config)
        if maybe is None:
            skipped_count += 1
            continue
        seeds.append(maybe)
    seeds.sort(
        key=lambda seed: (
            -float(seed["flip_rate"]),
            str(seed["battle_id"]),
            int(seed["decision_round_index"]),
            int(seed["step_index"]),
            int(seed["deviation_action_index"]),
        )
    )
    if resolved_config.max_seeds is not None:
        skipped_count += max(0, len(seeds) - resolved_config.max_seeds)
        seeds = seeds[: resolved_config.max_seeds]
    return RefutationBehaviorSeedManifest(
        config=resolved_config,
        source_digest=_source_rows_digest(source_rows),
        source_row_count=len(source_rows),
        skipped_count=skipped_count,
        seeds=tuple(seeds),
    )


def write_refutation_behavior_seed_manifest(path: Path, manifest: RefutationBehaviorSeedManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_from_row(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    config: RefutationBehaviorSeedConfig,
) -> dict[str, Any] | None:
    if row.get("schema_version") != FRAGILE_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported fragile-state schema: {row.get('schema_version')!r}")
    if row.get("evaluation_source") != TERMINAL_ROLLOUT_EVALUATION_SOURCE:
        raise ValueError("behavior seeds require terminal-rollout refutations")
    search_stats = _mapping(row.get("search_stats"), label="search_stats")
    if search_stats.get("value_head_used") is not False:
        raise ValueError("behavior seeds require refutations that did not use a value head")
    mode = _required_str(row.get("mode"), label="mode")
    if config.mode is not None and mode != config.mode:
        return None
    certification = _mapping(row.get("certification"), label="certification")
    if certification.get("passed") is not True:
        return None
    candidate = _mapping(row.get("candidate"), label="candidate")
    _validate_certification_counts(row=row, candidate=candidate, certification=certification)
    flip_rate = _float(certification.get("flip_rate"), label="certification.flip_rate")
    if flip_rate < config.min_flip_rate:
        return None
    battle_id = _required_str(candidate.get("battle_id"), label="candidate.battle_id")
    decision_round_index = _int(candidate.get("decision_round_index"), label="candidate.decision_round_index")
    step_index = _int(candidate.get("step_index"), label="candidate.step_index")
    deviation_action_index = _int(candidate.get("deviation_action_index"), label="candidate.deviation_action_index")
    recorded_action_index = _int(candidate.get("recorded_action_index"), label="candidate.recorded_action_index")
    seed_id = f"{battle_id}:round-{decision_round_index}:step-{step_index}:action-{deviation_action_index}"
    return {
        "seed_id": seed_id,
        "source_row_index": source_row_index,
        "source_record_index": _int(candidate.get("source_record_index"), label="candidate.source_record_index"),
        "battle_id": battle_id,
        "seed": _int(candidate.get("seed"), label="candidate.seed"),
        "format_id": _required_str(candidate.get("format_id"), label="candidate.format_id"),
        "mode": mode,
        "evaluation_source": TERMINAL_ROLLOUT_EVALUATION_SOURCE,
        "champion_player_id": _required_str(candidate.get("champion_player_id"), label="candidate.champion_player_id"),
        "loser_player_id": _required_str(candidate.get("loser_player_id"), label="candidate.loser_player_id"),
        "decision_round_index": decision_round_index,
        "step_index": step_index,
        "recorded_action_index": recorded_action_index,
        "deviation_action_index": deviation_action_index,
        "flip_rate": flip_rate,
        "min_flip_rate": _float(certification.get("min_flip_rate"), label="certification.min_flip_rate"),
        "certification_seed_count": _int(
            certification.get("seed_count"),
            label="certification.seed_count",
        ),
        "deviation_wins": _int(certification.get("deviation_wins"), label="certification.deviation_wins"),
        "champion_wins": _int(certification.get("champion_wins"), label="certification.champion_wins"),
        "ties_or_caps": _int(certification.get("ties_or_caps"), label="certification.ties_or_caps"),
        "population_use": {
            "kind": "refutation_behavior_seed",
            "intended_for": [
                "candidate_distillation",
                "admission_gauntlet_seed",
                "heldout_exploiter_seed",
            ],
            "not_for": [
                "reward_shaping",
                "legacy_checkpoint_strength_eval",
            ],
        },
    }


def _source_rows_digest(rows: tuple[Mapping[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _validate_certification_counts(
    *,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    certification: Mapping[str, Any],
) -> None:
    terminal_results = row.get("terminal_results")
    if not isinstance(terminal_results, list):
        raise ValueError("behavior seeds require terminal_results")
    champion_player_id = _required_str(candidate.get("champion_player_id"), label="candidate.champion_player_id")
    loser_player_id = _required_str(candidate.get("loser_player_id"), label="candidate.loser_player_id")
    deviation_wins = 0
    champion_wins = 0
    for result in terminal_results:
        result_row = _mapping(result, label="terminal_results[]")
        winner = result_row.get("winner")
        if winner == loser_player_id:
            deviation_wins += 1
        elif winner == champion_player_id:
            champion_wins += 1
    ties_or_caps = len(terminal_results) - deviation_wins - champion_wins
    seed_count = _int(certification.get("seed_count"), label="certification.seed_count")
    if len(terminal_results) != seed_count:
        raise ValueError("terminal_results length must match certification.seed_count")
    if _int(certification.get("deviation_wins"), label="certification.deviation_wins") != deviation_wins:
        raise ValueError("certification.deviation_wins does not match terminal_results")
    if _int(certification.get("champion_wins"), label="certification.champion_wins") != champion_wins:
        raise ValueError("certification.champion_wins does not match terminal_results")
    if _int(certification.get("ties_or_caps"), label="certification.ties_or_caps") != ties_or_caps:
        raise ValueError("certification.ties_or_caps does not match terminal_results")
    flip_rate = deviation_wins / len(terminal_results) if terminal_results else 0.0
    if abs(_float(certification.get("flip_rate"), label="certification.flip_rate") - flip_rate) > 1e-9:
        raise ValueError("certification.flip_rate does not match terminal_results")


def _required_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
