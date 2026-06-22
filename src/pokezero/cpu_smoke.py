"""CPU-only end-to-end smoke workflow for local self-play plumbing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from .bootstrap import TeacherBootstrapResult, run_teacher_bootstrap
from .env import PokeZeroEnv
from .evaluation_profiles import evaluation_profile
from .linear_policy import LinearTrainingConfig
from .rollout import RolloutConfig
from .run_audit import RunAuditResult, calibrate_run_audit, RunAuditCalibrationResult, audit_run
from .selfplay import SelfPlayPromotionConfig, SelfPlayRunResult, run_selfplay_iterations


CPU_SMOKE_RUN_SCHEMA_VERSION = "pokezero.cpu_smoke_run.v1"


@dataclass(frozen=True)
class CPUSmokeRunResult:
    run_dir: Path
    summary_path: Path
    audit_profile: str
    bootstrap: TeacherBootstrapResult
    selfplay: SelfPlayRunResult
    promotion_registry_path: Path
    promotion_artifact_dir: Path
    audit: RunAuditResult
    calibration: RunAuditCalibrationResult

    @property
    def passed(self) -> bool:
        return self.audit.passed

    def to_dict(self) -> dict:
        return {
            "schema_version": CPU_SMOKE_RUN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "passed": self.passed,
            "audit_profile": self.audit_profile,
            "bootstrap": {
                "run_dir": str(self.bootstrap.run_dir),
                "manifest_path": str(self.bootstrap.manifest_path),
                "checkpoint_path": str(self.bootstrap.checkpoint_path),
                "validation_rollout_path": str(self.bootstrap.validation_rollout_path),
                "train_games": self.bootstrap.train_metrics.games,
                "validation_games": self.bootstrap.validation_metrics.games,
                "benchmark_games": self.bootstrap.benchmark.total_games if self.bootstrap.benchmark is not None else 0,
                "teacher_decision_summary": dict(self.bootstrap.teacher_decision_summary),
            },
            "selfplay": {
                "run_dir": str(self.selfplay.run_dir),
                "manifest_path": str(self.selfplay.run_dir / "manifest.json"),
                "iterations": len(self.selfplay.iterations),
                "latest_checkpoint_path": (
                    str(self.selfplay.latest_checkpoint_path) if self.selfplay.latest_checkpoint_path is not None else None
                ),
            },
            "promotion_registry_path": str(self.promotion_registry_path),
            "promotion_artifact_dir": str(self.promotion_artifact_dir),
            "audit": self.audit.to_dict(),
            "calibration": self.calibration.to_dict(),
        }


def run_cpu_smoke_experiment(
    *,
    run_dir: Path,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig = RolloutConfig(),
    audit_profile: str = "smoke",
    train_games: int = 4,
    validation_games: int = 2,
    bootstrap_benchmark_games: int = 2,
    preflight_games: int = 1,
    selfplay_iterations: int = 1,
    games_per_iteration: int = 4,
    evaluation_games: int = 2,
    worker_count: int = 1,
    teacher_policy_spec: str = "scripted-teacher",
    bootstrap_opponent_policy_specs: tuple[str, ...] | None = None,
    fixed_opponent_policy_specs: tuple[str, ...] = ("random-legal", "simple-legal"),
    seed_start: int = 1,
    validation_seed_start: int = 1_000_000,
    benchmark_seed_start: int = 2_000_000,
    selfplay_seed_start: int = 3_000_000,
    evaluation_seed_start: int = 4_000_000,
    feature_count: int = 8192,
    window_size: int = 4,
    epochs: int = 1,
    learning_rate: float = 0.05,
) -> CPUSmokeRunResult:
    if run_dir.exists() and (run_dir / "summary.json").exists():
        raise ValueError(f"CPU smoke summary already exists: {run_dir / 'summary.json'}")
    profile = evaluation_profile(audit_profile)
    resolved_run_dir = run_dir.expanduser().resolve(strict=False)
    bootstrap_dir = resolved_run_dir / "bootstrap"
    selfplay_dir = resolved_run_dir / "selfplay"
    promotion_registry_path = resolved_run_dir / "promotions.json"
    promotion_artifact_dir = resolved_run_dir / "promoted-checkpoints"
    summary_path = resolved_run_dir / "summary.json"
    training_config = LinearTrainingConfig(
        feature_count=feature_count,
        window_size=window_size,
        objective="behavior-cloning",
        epochs=epochs,
        learning_rate=learning_rate,
        shuffle_buffer_size=0,
        policy_id="cpu-smoke-linear",
    )
    bootstrap = run_teacher_bootstrap(
        run_dir=bootstrap_dir,
        env_factory=env_factory,
        rollout_config=rollout_config,
        training_config=training_config,
        train_games=train_games,
        validation_games=validation_games,
        teacher_policy_spec=teacher_policy_spec,
        opponent_policy_specs=bootstrap_opponent_policy_specs,
        seed_start=seed_start,
        validation_seed_start=validation_seed_start,
        benchmark_games=bootstrap_benchmark_games,
        benchmark_seed_start=benchmark_seed_start,
        preflight_games=preflight_games,
        worker_count=worker_count,
    )
    selfplay = run_selfplay_iterations(
        run_dir=selfplay_dir,
        iterations=selfplay_iterations,
        games_per_iteration=games_per_iteration,
        env_factory=env_factory,
        rollout_config=rollout_config,
        training_config=training_config,
        seed_start=selfplay_seed_start,
        initial_policy_spec=f"linear:{bootstrap.checkpoint_path}",
        fixed_opponent_policy_specs=fixed_opponent_policy_specs,
        benchmark_reference_policy_specs=(f"linear:{bootstrap.checkpoint_path}",),
        evaluation_games=evaluation_games,
        evaluation_seed_start=evaluation_seed_start,
        validation_rollout_paths=(bootstrap.validation_rollout_path,),
        promotion_registry_path=promotion_registry_path,
        auto_promotion_config=SelfPlayPromotionConfig(
            registry_path=promotion_registry_path,
            gate_config=profile.gate_config,
            artifact_dir=promotion_artifact_dir,
            label_prefix="cpu-smoke",
            notes=f"CPU smoke workflow using {audit_profile} profile",
        ),
        worker_count=worker_count,
    )
    audit = audit_run(selfplay_dir, config=profile.audit_config)
    calibration = calibrate_run_audit(selfplay_dir)
    result = CPUSmokeRunResult(
        run_dir=resolved_run_dir,
        summary_path=summary_path,
        audit_profile=audit_profile,
        bootstrap=bootstrap,
        selfplay=selfplay,
        promotion_registry_path=promotion_registry_path,
        promotion_artifact_dir=promotion_artifact_dir,
        audit=audit,
        calibration=calibration,
    )
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result
