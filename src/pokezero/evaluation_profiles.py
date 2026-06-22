"""Named evaluation profiles shared by gate and audit CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation import (
    DEFAULT_MAX_COLLECTION_CAPPED_RATE,
    DEFAULT_MAX_TEACHER_DEGRADATION_RATE,
    DEFAULT_MIN_INCUMBENT_GAMES,
    DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND,
    PromotionGateConfig,
)
from .run_audit import (
    DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES,
    RunAuditConfig,
)


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    description: str
    gate_config: PromotionGateConfig
    audit_config: RunAuditConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "gate": {
                "min_benchmark_win_rate": self.gate_config.min_benchmark_win_rate,
                "min_incumbent_win_rate": self.gate_config.min_incumbent_win_rate,
                "min_benchmark_games": self.gate_config.min_benchmark_games,
                "min_incumbent_games": self.gate_config.min_incumbent_games,
                "max_collection_capped_rate": self.gate_config.max_collection_capped_rate,
                "max_benchmark_capped_rate": self.gate_config.max_benchmark_capped_rate,
                "max_incumbent_capped_rate": self.gate_config.max_incumbent_capped_rate,
                "max_teacher_degradation_rate": self.gate_config.max_teacher_degradation_rate,
                "min_incumbent_win_rate_lower_bound": self.gate_config.min_incumbent_win_rate_lower_bound,
                "incumbent_confidence_z": self.gate_config.incumbent_confidence_z,
                "require_benchmark": self.gate_config.require_benchmark,
                "required_benchmark_opponents": list(self.gate_config.required_benchmark_opponents),
                "opponent_min_win_rates": dict(self.gate_config.opponent_min_win_rates),
                "incumbent_policy_id": self.gate_config.incumbent_policy_id,
            },
            "audit": {
                "min_latest_benchmark_win_rate": self.audit_config.min_latest_benchmark_win_rate,
                "min_latest_benchmark_games": self.audit_config.min_latest_benchmark_games,
                "max_latest_collection_capped_rate": self.audit_config.max_latest_collection_capped_rate,
                "max_latest_benchmark_capped_rate": self.audit_config.max_latest_benchmark_capped_rate,
                "max_latest_average_decision_rounds": self.audit_config.max_latest_average_decision_rounds,
                "max_latest_benchmark_average_decision_rounds": (
                    self.audit_config.max_latest_benchmark_average_decision_rounds
                ),
                "max_benchmark_win_rate_drop": self.audit_config.max_benchmark_win_rate_drop,
                "max_consecutive_promotion_failures": self.audit_config.max_consecutive_promotion_failures,
                "require_benchmark": self.audit_config.require_benchmark,
                "require_latest_promotion": self.audit_config.require_latest_promotion,
                "require_benchmark_opponent_coverage": self.audit_config.require_benchmark_opponent_coverage,
            },
        }


SMOKE_EVALUATION_PROFILE = EvaluationProfile(
    name="smoke",
    description="Permissive thresholds for plumbing checks where benchmark evidence may be absent.",
    gate_config=PromotionGateConfig(
        min_benchmark_win_rate=0.0,
        min_incumbent_win_rate=0.0,
        min_benchmark_games=0,
        min_incumbent_games=0,
        max_collection_capped_rate=1.0,
        max_benchmark_capped_rate=1.0,
        max_incumbent_capped_rate=1.0,
        max_teacher_degradation_rate=1.0,
        min_incumbent_win_rate_lower_bound=0.0,
        require_benchmark=False,
    ),
    audit_config=RunAuditConfig(
        min_latest_benchmark_win_rate=0.0,
        min_latest_benchmark_games=0,
        max_latest_collection_capped_rate=1.0,
        max_latest_benchmark_capped_rate=1.0,
        max_benchmark_win_rate_drop=1.0,
        max_consecutive_promotion_failures=1000,
        require_benchmark=False,
        require_latest_promotion=False,
        require_benchmark_opponent_coverage=False,
    ),
)

DEFAULT_EVALUATION_PROFILE = EvaluationProfile(
    name="default",
    description="Current default guardrails used by gate and audit commands.",
    gate_config=PromotionGateConfig(),
    audit_config=RunAuditConfig(),
)

LONG_RUN_EVALUATION_PROFILE = EvaluationProfile(
    name="long-run",
    description="Stricter provisional CPU long-run guardrails for benchmarked self-play experiments.",
    gate_config=PromotionGateConfig(
        min_benchmark_win_rate=0.60,
        min_incumbent_win_rate=0.57,
        min_benchmark_games=100,
        min_incumbent_games=DEFAULT_MIN_INCUMBENT_GAMES,
        max_collection_capped_rate=DEFAULT_MAX_COLLECTION_CAPPED_RATE,
        max_benchmark_capped_rate=0.05,
        max_incumbent_capped_rate=0.05,
        max_teacher_degradation_rate=DEFAULT_MAX_TEACHER_DEGRADATION_RATE,
        min_incumbent_win_rate_lower_bound=DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND,
        require_benchmark=True,
    ),
    audit_config=RunAuditConfig(
        min_latest_benchmark_win_rate=0.60,
        min_latest_benchmark_games=100,
        max_latest_collection_capped_rate=DEFAULT_MAX_COLLECTION_CAPPED_RATE,
        max_latest_benchmark_capped_rate=0.05,
        max_latest_average_decision_rounds=200.0,
        max_latest_benchmark_average_decision_rounds=200.0,
        max_benchmark_win_rate_drop=0.03,
        max_consecutive_promotion_failures=DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES,
        require_benchmark=True,
        require_latest_promotion=False,
        require_benchmark_opponent_coverage=True,
    ),
)

EVALUATION_PROFILES: dict[str, EvaluationProfile] = {
    profile.name: profile
    for profile in (
        SMOKE_EVALUATION_PROFILE,
        DEFAULT_EVALUATION_PROFILE,
        LONG_RUN_EVALUATION_PROFILE,
    )
}


def evaluation_profile(name: str | None) -> EvaluationProfile:
    resolved_name = name or DEFAULT_EVALUATION_PROFILE.name
    try:
        return EVALUATION_PROFILES[resolved_name]
    except KeyError as exc:
        choices = ", ".join(sorted(EVALUATION_PROFILES))
        raise ValueError(f"unknown evaluation profile {resolved_name!r}; choose one of: {choices}") from exc
