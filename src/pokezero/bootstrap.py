"""Bootstrap data and checkpoint workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkMatchupResult,
    BenchmarkReport,
    CollectionMetrics,
    benchmark_rollouts,
    iter_rollout_records,
    policy_from_spec,
    run_rollout_record,
    RolloutRecord,
)
from .env import PokeZeroEnv
from .linear_policy import LinearSoftmaxPolicy, LinearTrainingConfig, LinearTrainingResult, save_linear_model, train_linear_policy
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig
from .selfplay import collect_selfplay_rollouts
from .source_metadata import collect_source_metadata


TEACHER_BOOTSTRAP_SCHEMA_VERSION = "pokezero.teacher_bootstrap.v1"
DEFAULT_BASELINE_OPPONENT_POLICY_SPECS = ("simple-legal", "random-legal")
DEFAULT_BENCHMARK_GAMES = 10
DEFAULT_PREFLIGHT_GAMES = 2
DEFAULT_PREFLIGHT_SEED_START = 3_000_000
_FALLBACK_TEACHER_REASONS = frozenset(
    {
        "dex unavailable",
        "missing observation metadata",
        "missing legal candidate metadata",
        "fallback",
    }
)


@dataclass(frozen=True)
class TeacherBootstrapResult:
    run_dir: Path
    manifest_path: Path
    full_train_rollout_path: Path
    train_rollout_path: Path
    full_validation_rollout_path: Path
    validation_rollout_path: Path
    checkpoint_path: Path
    teacher_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    preflight_seed_start: int
    preflight_metrics: CollectionMetrics | None
    train_metrics: CollectionMetrics
    validation_metrics: CollectionMetrics
    training: LinearTrainingResult
    teacher_decision_summary: Mapping[str, Any]
    benchmark: BenchmarkReport | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TEACHER_BOOTSTRAP_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "source": dict(self.source),
            "full_train_rollout_path": str(self.full_train_rollout_path),
            "train_rollout_path": str(self.train_rollout_path),
            "full_validation_rollout_path": str(self.full_validation_rollout_path),
            "validation_rollout_path": str(self.validation_rollout_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_policy_spec": f"linear:{self.checkpoint_path}",
            "teacher_policy_spec": self.teacher_policy_spec,
            "opponent_policy_specs": list(self.opponent_policy_specs),
            "preflight": {
                "seed_start": self.preflight_seed_start,
                "metrics": self.preflight_metrics.to_dict() if self.preflight_metrics is not None else None,
            },
            "train_collection_metrics": self.train_metrics.to_dict(),
            "validation_collection_metrics": self.validation_metrics.to_dict(),
            "teacher_decision_summary": dict(self.teacher_decision_summary),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
        }


@dataclass(frozen=True)
class TeacherBenchmarkResult:
    benchmark: BenchmarkReport
    teacher_decision_summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark.to_dict(),
            "teacher_decision_summary": dict(self.teacher_decision_summary),
        }


def run_teacher_bootstrap(
    *,
    run_dir: Path,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    training_config: LinearTrainingConfig,
    train_games: int,
    validation_games: int,
    teacher_policy_spec: str = "scripted-teacher",
    opponent_policy_specs: Iterable[str] | None = None,
    seed_start: int = 1,
    validation_seed_start: int = 1_000_000,
    benchmark_games: int = DEFAULT_BENCHMARK_GAMES,
    benchmark_seed_start: int = 2_000_000,
    preflight_games: int = DEFAULT_PREFLIGHT_GAMES,
    preflight_seed_start: int = DEFAULT_PREFLIGHT_SEED_START,
    worker_count: int = 1,
) -> TeacherBootstrapResult:
    if train_games <= 0:
        raise ValueError("train_games must be positive.")
    if validation_games <= 0:
        raise ValueError("validation_games must be positive.")
    if benchmark_games < 0:
        raise ValueError("benchmark_games must be non-negative.")
    if preflight_games < 0:
        raise ValueError("preflight_games must be non-negative.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    opponents = (
        _default_opponent_policy_specs(teacher_policy_spec)
        if opponent_policy_specs is None
        else tuple(opponent_policy_specs)
    )
    if not opponents:
        raise ValueError("at least one opponent policy spec is required.")
    source_metadata = collect_source_metadata()
    _validate_seed_ranges(
        (
            ("train", seed_start, train_games),
            ("validation", validation_seed_start, validation_games),
            ("benchmark", benchmark_seed_start, benchmark_games),
            ("preflight", preflight_seed_start, preflight_games),
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("bootstrap run manifest already exists; choose a new run_dir.")

    full_train_rollout_path = run_dir / "train-full-rollouts.jsonl"
    train_rollout_path = run_dir / "train-rollouts.jsonl"
    full_validation_rollout_path = run_dir / "validation-full-rollouts.jsonl"
    validation_rollout_path = run_dir / "validation-rollouts.jsonl"
    checkpoint_path = run_dir / "linear-bootstrap.json"
    _require_outputs_absent(
        full_train_rollout_path,
        train_rollout_path,
        full_validation_rollout_path,
        validation_rollout_path,
        checkpoint_path,
    )

    preflight_metrics = None
    if preflight_games:
        preflight_metrics = _run_preflight(
            run_dir=run_dir,
            games=preflight_games,
            env_factory=env_factory,
            rollout_config=rollout_config,
            seed_start=preflight_seed_start,
            teacher_policy_spec=teacher_policy_spec,
            opponent_policy_specs=opponents,
        )

    train_metrics = collect_selfplay_rollouts(
        output_path=full_train_rollout_path,
        training_output_path=train_rollout_path,
        games=train_games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        current_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        worker_count=worker_count,
    )
    validation_metrics = collect_selfplay_rollouts(
        output_path=full_validation_rollout_path,
        training_output_path=validation_rollout_path,
        games=validation_games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=validation_seed_start,
        current_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        worker_count=worker_count,
    )
    training = train_linear_policy(
        train_rollout_path,
        config=training_config,
        validation_paths=validation_rollout_path,
    )
    save_linear_model(checkpoint_path, training.model)
    teacher_decision_summary = _teacher_decision_summary(
        train_rollout_path,
        validation_rollout_path,
    )
    benchmark = None
    if benchmark_games:
        benchmark = _benchmark_bootstrap_checkpoint(
            model_policy=LinearSoftmaxPolicy(model=training.model),
            teacher_policy_spec=teacher_policy_spec,
            env_factory=env_factory,
            rollout_config=rollout_config,
            games=benchmark_games,
            seed_start=benchmark_seed_start,
        )

    result = TeacherBootstrapResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        full_train_rollout_path=full_train_rollout_path,
        train_rollout_path=train_rollout_path,
        full_validation_rollout_path=full_validation_rollout_path,
        validation_rollout_path=validation_rollout_path,
        checkpoint_path=checkpoint_path,
        teacher_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        preflight_seed_start=preflight_seed_start,
        preflight_metrics=preflight_metrics,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        training=training,
        teacher_decision_summary=teacher_decision_summary,
        benchmark=benchmark,
        source=source_metadata,
    )
    _write_json(manifest_path, result.to_dict())
    return result


def benchmark_teacher_policy(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    teacher_policy_spec: str = "scripted-teacher",
    baseline_policy_specs: Iterable[str] | None = None,
    games: int = DEFAULT_BENCHMARK_GAMES,
    seed_start: int = 1,
) -> TeacherBenchmarkResult:
    if games <= 0:
        raise ValueError("games must be positive.")
    baselines = (
        DEFAULT_BASELINE_OPPONENT_POLICY_SPECS
        if baseline_policy_specs is None
        else tuple(dict.fromkeys(baseline_policy_specs))
    )
    if not baselines:
        raise ValueError("at least one baseline policy spec is required.")
    teacher_policy_id = policy_from_spec(teacher_policy_spec).policy_id
    matchups: list[BenchmarkMatchup] = []
    for baseline_spec in baselines:
        baseline_policy_id = policy_from_spec(baseline_spec).policy_id
        if baseline_policy_id == teacher_policy_id:
            continue
        matchups.extend(
            (
                BenchmarkMatchup(
                    f"{teacher_policy_id} vs {baseline_policy_id}",
                    policy_from_spec(teacher_policy_spec),
                    policy_from_spec(baseline_spec),
                ),
                BenchmarkMatchup(
                    f"{baseline_policy_id} vs {teacher_policy_id}",
                    policy_from_spec(baseline_spec),
                    policy_from_spec(teacher_policy_spec),
                ),
            )
        )
    if not matchups:
        raise ValueError("teacher benchmark requires at least one baseline distinct from the teacher.")
    matchup_results: list[BenchmarkMatchupResult] = []
    records: list[RolloutRecord] = []
    for matchup in matchups:
        accumulator = _BenchmarkMetricsAccumulator()
        matchup_start = perf_counter()
        for game_index in range(games):
            seed = seed_start + game_index
            record = run_rollout_record(
                env_factory=env_factory,
                policies={"p1": matchup.p1_policy, "p2": matchup.p2_policy},
                rollout_config=rollout_config,
                seed=seed,
                battle_id=f"teacher-benchmark-{_slugify_label(matchup.label)}-{seed}",
            )
            accumulator.add(record)
            records.append(record)
        matchup_results.append(
            BenchmarkMatchupResult(
                label=matchup.label,
                p1_policy_id=matchup.p1_policy.policy_id,
                p2_policy_id=matchup.p2_policy.policy_id,
                seed_start=seed_start,
                metrics=accumulator.to_metrics(elapsed_seconds=perf_counter() - matchup_start),
            )
        )
    benchmark = BenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games_per_matchup=games,
        matchups=tuple(matchup_results),
    )
    return TeacherBenchmarkResult(
        benchmark=benchmark,
        teacher_decision_summary=_teacher_decision_summary_from_records(records),
    )


def _default_opponent_policy_specs(teacher_policy_spec: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((teacher_policy_spec, *DEFAULT_BASELINE_OPPONENT_POLICY_SPECS)))


def _run_preflight(
    *,
    run_dir: Path,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    seed_start: int,
    teacher_policy_spec: str,
    opponent_policy_specs: tuple[str, ...],
) -> CollectionMetrics:
    with tempfile.TemporaryDirectory(prefix=".teacher-preflight-", dir=run_dir) as temp_dir:
        temp_path = Path(temp_dir)
        return collect_selfplay_rollouts(
            output_path=temp_path / "full-rollouts.jsonl",
            training_output_path=temp_path / "training-rollouts.jsonl",
            games=games,
            env_factory=env_factory,
            rollout_config=rollout_config,
            seed_start=seed_start,
            current_policy_spec=teacher_policy_spec,
            opponent_policy_specs=opponent_policy_specs,
            worker_count=1,
        )


def _teacher_decision_summary(*paths: Path) -> dict[str, Any]:
    return _teacher_decision_summary_from_records(
        record
        for path in paths
        for record in iter_rollout_records(path)
    )


def _teacher_decision_summary_from_records(records: Iterable[RolloutRecord]) -> dict[str, Any]:
    total_decisions = 0
    scripted_teacher_decisions = 0
    unknown_move_decisions = 0
    fallback_decisions = 0
    fallback_reasons: dict[str, int] = {}
    for record in records:
        for step in record.trajectory.steps:
            total_decisions += 1
            if step.metadata.get("policy_family") != "scripted-teacher":
                continue
            scripted_teacher_decisions += 1
            reason = str(step.metadata.get("teacher_reason") or "")
            if reason == "unknown move":
                unknown_move_decisions += 1
            if reason in _FALLBACK_TEACHER_REASONS:
                fallback_decisions += 1
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    return {
        "total_decisions": total_decisions,
        "scripted_teacher_decisions": scripted_teacher_decisions,
        "unknown_move_decisions": unknown_move_decisions,
        "fallback_decisions": fallback_decisions,
        "fallback_reasons": fallback_reasons,
    }


def _validate_seed_ranges(ranges: Iterable[tuple[str, int, int]]) -> None:
    active_ranges = tuple((name, start, start + count) for name, start, count in ranges if count > 0)
    for index, (left_name, left_start, left_end) in enumerate(active_ranges):
        for right_name, right_start, right_end in active_ranges[index + 1 :]:
            if left_start < right_end and right_start < left_end:
                raise ValueError(
                    f"{left_name} seed range [{left_start}, {left_end}) overlaps "
                    f"{right_name} seed range [{right_start}, {right_end})."
                )


def _benchmark_bootstrap_checkpoint(
    *,
    model_policy: LinearSoftmaxPolicy,
    teacher_policy_spec: str,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
) -> BenchmarkReport:
    policy_id = str(model_policy.policy_id)
    teacher_policy_as_p1 = policy_from_spec(teacher_policy_spec)
    teacher_policy_as_p2 = policy_from_spec(teacher_policy_spec)
    teacher_policy_id = teacher_policy_as_p1.policy_id
    return benchmark_rollouts(
        games=games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        matchups=(
            BenchmarkMatchup(f"{policy_id} vs random-legal", LinearSoftmaxPolicy(model=model_policy.model), RandomLegalPolicy()),
            BenchmarkMatchup(f"random-legal vs {policy_id}", RandomLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
            BenchmarkMatchup(f"{policy_id} vs simple-legal", LinearSoftmaxPolicy(model=model_policy.model), SimpleLegalPolicy()),
            BenchmarkMatchup(f"simple-legal vs {policy_id}", SimpleLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
            BenchmarkMatchup(f"{policy_id} vs {teacher_policy_id}", LinearSoftmaxPolicy(model=model_policy.model), teacher_policy_as_p2),
            BenchmarkMatchup(f"{teacher_policy_id} vs {policy_id}", teacher_policy_as_p1, LinearSoftmaxPolicy(model=model_policy.model)),
        ),
    )


def _training_result_to_dict(result: LinearTrainingResult) -> dict[str, Any]:
    return {
        "config": {
            "feature_count": result.config.feature_count,
            "window_size": result.config.window_size,
            "discount": result.config.discount,
            "capped_terminal_value": result.config.capped_terminal_value,
            "objective": result.config.objective,
            "epochs": result.config.epochs,
            "learning_rate": result.config.learning_rate,
            "l2": result.config.l2,
            "shuffle_buffer_size": result.config.shuffle_buffer_size,
            "shuffle_seed": result.config.shuffle_seed,
            "max_examples": result.config.max_examples,
            "policy_id": result.config.policy_id,
        },
        "epochs": [metrics.to_dict() for metrics in result.epochs],
        "validation_metrics": result.validation_metrics.to_dict() if result.validation_metrics is not None else None,
        "model": {
            "policy_id": result.model.policy_id,
            "action_schema_version": result.model.action_schema_version,
            "observation_schema_version": result.model.observation_schema_version,
            "feature_schema_version": result.model.feature_schema_version,
            "feature_fingerprint": result.model.feature_fingerprint,
            "feature_count": result.model.feature_count,
            "window_size": result.model.window_size,
        },
    }


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(f"bootstrap output path(s) already exist: {', '.join(existing)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _slugify_label(label: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in label.strip())
    return "-".join(part for part in slug.split("-") if part)


@dataclass
class _BenchmarkMetricsAccumulator:
    games: int = 0
    total_decision_rounds: int = 0
    total_simulator_turns: int = 0
    p1_wins: int = 0
    p2_wins: int = 0
    ties: int = 0
    capped_games: int = 0

    def add(self, record: RolloutRecord) -> None:
        self.games += 1
        self.total_decision_rounds += record.decision_round_count
        self.total_simulator_turns += record.terminal.turn_count
        if record.terminal.winner == "p1":
            self.p1_wins += 1
        elif record.terminal.winner == "p2":
            self.p2_wins += 1
        elif not record.terminal.capped:
            self.ties += 1
        if record.terminal.capped:
            self.capped_games += 1

    def to_metrics(self, *, elapsed_seconds: float) -> CollectionMetrics:
        return CollectionMetrics(
            games=self.games,
            elapsed_seconds=elapsed_seconds,
            total_decision_rounds=self.total_decision_rounds,
            total_simulator_turns=self.total_simulator_turns,
            p1_wins=self.p1_wins,
            p2_wins=self.p2_wins,
            ties=self.ties,
            capped_games=self.capped_games,
        )
