"""Seed-paired strength analysis for the test-time-search capstone.

The capstone plays each BattleStream seed from both seats.  This module keeps the
two seats together when bootstrapping, preventing a seat-specific result from
being counted as an independent seed-level sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping


OUTCOME_SCORING: Mapping[str, float] = {
    "win": 1.0,
    "tie": 0.5,
    "capped": 0.5,
    "loss": 0.0,
}
_MIRRORED_SEATS = frozenset(("p1", "p2"))


@dataclass(frozen=True)
class CapstoneGameOutcome:
    """One arm's public outcome for a single seed and seat."""

    band: str
    seat: str
    seed: int
    score: float
    tied: bool = False
    capped: bool = False
    root_puct_fallbacks: int = 0
    privileged_fallbacks: int = 0

    def __post_init__(self) -> None:
        if not self.band.strip():
            raise ValueError("band must be non-empty.")
        if self.seat not in _MIRRORED_SEATS:
            raise ValueError("seat must be p1 or p2.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.tied and self.capped:
            raise ValueError("an outcome cannot be both tied and capped.")
        if not math.isfinite(self.score) or self.score not in set(OUTCOME_SCORING.values()):
            raise ValueError("score must be one of 0.0, 0.5, or 1.0.")
        if (self.tied or self.capped) != (self.score == 0.5):
            raise ValueError("ties and capped games must score 0.5, and only those outcomes may do so.")
        for field_name, value in (
            ("root_puct_fallbacks", self.root_puct_fallbacks),
            ("privileged_fallbacks", self.privileged_fallbacks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.band, self.seed, self.seat)


@dataclass(frozen=True)
class CapstoneArmEvidence:
    """One capstone arm plus the provenance required for primary-row eligibility."""

    arm_id: str
    outcomes: tuple[CapstoneGameOutcome, ...]
    uses_value_leaves: bool
    calibrated_value_copy: str | None = None

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise ValueError("arm_id must be non-empty.")
        if self.uses_value_leaves and not (self.calibrated_value_copy or "").strip():
            raise ValueError("value-leaf arms require a frozen calibrated_value_copy label.")
        if not self.outcomes:
            raise ValueError("an arm requires at least one outcome.")

    @property
    def fallback_count(self) -> int:
        return sum(outcome.root_puct_fallbacks for outcome in self.outcomes)

    @property
    def privileged_fallback_count(self) -> int:
        return sum(outcome.privileged_fallbacks for outcome in self.outcomes)


@dataclass(frozen=True)
class _PairedGame:
    baseline: CapstoneGameOutcome
    candidate: CapstoneGameOutcome

    @property
    def delta(self) -> float:
        return self.candidate.score - self.baseline.score


@dataclass(frozen=True)
class _SeedGroup:
    band: str
    seed: int
    baseline_score: float
    candidate_score: float
    pairs: tuple[_PairedGame, ...]

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


def paired_capstone_delta(
    baseline_outcomes: Iterable[CapstoneGameOutcome],
    candidate_outcomes: Iterable[CapstoneGameOutcome],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260710,
    expected_keys: Iterable[tuple[str, int, str]] | None = None,
) -> dict[str, Any]:
    """Return deterministic paired deltas and bootstrap intervals for one capstone arm.

    Inputs must have exactly matching ``(band, seed, seat)`` keys and both p1/p2
    seats for every seed. The two seats are first averaged and that seed-level
    value is the bootstrap unit.
    """

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer.")

    baseline_by_key = _outcomes_by_key(baseline_outcomes, label="baseline")
    candidate_by_key = _outcomes_by_key(candidate_outcomes, label="candidate")
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        raise ValueError(
            "baseline and candidate outcomes must have identical band/seed/seat keys; "
            f"missing baseline={missing_baseline!r}, missing candidate={missing_candidate!r}."
        )
    if not baseline_by_key:
        raise ValueError("at least one paired outcome is required.")
    if expected_keys is not None:
        expected_key_set = set(expected_keys)
        if baseline_by_key.keys() != expected_key_set:
            raise ValueError("outcome keys do not match the required shared capstone seed roster.")

    pairs = tuple(
        _PairedGame(baseline=baseline_by_key[key], candidate=candidate_by_key[key])
        for key in sorted(baseline_by_key)
    )
    groups = _seed_groups(pairs)
    result = {
        "analysis_method": "paired_bootstrap_seed_group_mean_scores",
        "outcome_scoring": dict(OUTCOME_SCORING),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "unit": "seed_mean_over_mirrored_seats",
            "stratified_by_band": True,
        },
        "overall": _paired_summary(
            pairs,
            groups,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "by_band": {},
    }
    for band in sorted({pair.baseline.band for pair in pairs}):
        band_pairs = tuple(pair for pair in pairs if pair.baseline.band == band)
        band_groups = tuple(group for group in groups if group.band == band)
        result["by_band"][band] = _paired_summary(
            band_pairs,
            band_groups,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_scoped_seed(bootstrap_seed, band),
        )
    return result


def analyze_primary_capstone(
    baseline: CapstoneArmEvidence,
    candidate_arms: Iterable[CapstoneArmEvidence],
    *,
    expected_keys: Iterable[tuple[str, int, str]],
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    """Analyze every primary arm against one baseline under the fixed capstone contract.

    A primary row is rejected before statistical analysis unless it uses the exact shared
    seed roster, has zero ordinary and privileged fallbacks, and labels its frozen calibrated
    copy whenever it evaluates value leaves.
    """

    expected = tuple(sorted(set(expected_keys)))
    if not expected:
        raise ValueError("expected_keys must contain the complete capstone seed roster.")
    _validate_mirrored_roster(expected)
    arms = tuple(candidate_arms)
    arm_ids = [baseline.arm_id, *(arm.arm_id for arm in arms)]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("baseline and candidate arm ids must be unique.")

    _validate_primary_arm(baseline, expected_keys=expected)
    rows: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        _validate_primary_arm(arm, expected_keys=expected)
        rows.append(
            {
                "arm_id": arm.arm_id,
                "uses_value_leaves": arm.uses_value_leaves,
                "calibrated_value_copy": arm.calibrated_value_copy,
                "fallback_count": arm.fallback_count,
                "privileged_fallback_count": arm.privileged_fallback_count,
                "paired_delta_vs_baseline": paired_capstone_delta(
                    baseline.outcomes,
                    arm.outcomes,
                    expected_keys=expected,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=_scoped_seed(bootstrap_seed, f"{arm.arm_id}:{index}"),
                ),
            }
        )
    return {
        "analysis_method": "primary_capstone_shared_roster_paired_bootstrap",
        "outcome_scoring": dict(OUTCOME_SCORING),
        "shared_seed_roster": _roster_readout(expected),
        "baseline": {
            "arm_id": baseline.arm_id,
            "uses_value_leaves": baseline.uses_value_leaves,
            "calibrated_value_copy": baseline.calibrated_value_copy,
            "fallback_count": baseline.fallback_count,
            "privileged_fallback_count": baseline.privileged_fallback_count,
        },
        "primary_arms": rows,
    }


def _validate_primary_arm(
    arm: CapstoneArmEvidence,
    *,
    expected_keys: tuple[tuple[str, int, str], ...],
) -> None:
    observed = tuple(sorted(_outcomes_by_key(arm.outcomes, label=arm.arm_id)))
    if observed != expected_keys:
        raise ValueError(f"{arm.arm_id}: outcomes do not match the shared capstone seed roster.")
    if arm.fallback_count:
        raise ValueError(f"{arm.arm_id}: primary rows require zero search fallbacks.")
    if arm.privileged_fallback_count:
        raise ValueError(f"{arm.arm_id}: primary rows require zero privileged fallbacks.")


def _validate_mirrored_roster(keys: tuple[tuple[str, int, str], ...]) -> None:
    by_seed: dict[tuple[str, int], set[str]] = {}
    for band, seed, seat in keys:
        if seat not in _MIRRORED_SEATS:
            raise ValueError("expected_keys may contain only p1/p2 seats.")
        by_seed.setdefault((band, seed), set()).add(seat)
    for key, seats in by_seed.items():
        if seats != _MIRRORED_SEATS:
            raise ValueError(f"shared capstone seed roster lacks mirrored seats for {key!r}.")


def _roster_readout(keys: tuple[tuple[str, int, str], ...]) -> dict[str, Any]:
    encoded = json.dumps(keys, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    bands: dict[str, int] = {}
    for band, _seed, _seat in keys:
        bands[band] = bands.get(band, 0) + 1
    return {
        "outcomes": len(keys),
        "seed_groups": len(keys) // len(_MIRRORED_SEATS),
        "outcomes_by_band": dict(sorted(bands.items())),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _outcomes_by_key(
    outcomes: Iterable[CapstoneGameOutcome],
    *,
    label: str,
) -> dict[tuple[str, int, str], CapstoneGameOutcome]:
    result: dict[tuple[str, int, str], CapstoneGameOutcome] = {}
    for outcome in outcomes:
        if outcome.key in result:
            raise ValueError(f"duplicate {label} outcome for {outcome.key!r}.")
        result[outcome.key] = outcome
    return result


def _seed_groups(pairs: tuple[_PairedGame, ...]) -> tuple[_SeedGroup, ...]:
    by_seed: dict[tuple[str, int], list[_PairedGame]] = {}
    for pair in pairs:
        key = (pair.baseline.band, pair.baseline.seed)
        by_seed.setdefault(key, []).append(pair)

    groups: list[_SeedGroup] = []
    for (band, seed), seed_pairs in sorted(by_seed.items()):
        seats = {pair.baseline.seat for pair in seed_pairs}
        if seats != _MIRRORED_SEATS:
            raise ValueError(f"seed {(band, seed)!r} must contain exactly mirrored p1 and p2 outcomes.")
        groups.append(
            _SeedGroup(
                band=band,
                seed=seed,
                baseline_score=sum(pair.baseline.score for pair in seed_pairs) / len(seed_pairs),
                candidate_score=sum(pair.candidate.score for pair in seed_pairs) / len(seed_pairs),
                pairs=tuple(seed_pairs),
            )
        )
    return tuple(groups)


def _paired_summary(
    pairs: tuple[_PairedGame, ...],
    groups: tuple[_SeedGroup, ...],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not groups:
        raise ValueError("paired summary requires at least one seed group.")
    baseline = tuple(pair.baseline for pair in pairs)
    candidate = tuple(pair.candidate for pair in pairs)
    bootstrap_deltas = _stratified_bootstrap_means(
        groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "paired_games": len(pairs),
        "paired_seed_groups": len(groups),
        "baseline": _outcome_readout(baseline),
        "candidate": _outcome_readout(candidate),
        "candidate_minus_baseline_score_rate": sum(group.delta for group in groups) / len(groups),
        "paired_bootstrap_95": {
            "lower": _percentile(bootstrap_deltas, 0.025),
            "upper": _percentile(bootstrap_deltas, 0.975),
        },
        "score_flip_counts": {
            "candidate_better": sum(1 for group in groups if group.delta > 0.0),
            "baseline_better": sum(1 for group in groups if group.delta < 0.0),
            "equal": sum(1 for group in groups if group.delta == 0.0),
        },
        "win_flip_counts": {
            "both_won": sum(1 for group in groups if _group_all_won(group, "baseline") and _group_all_won(group, "candidate")),
            "baseline_only_won": sum(
                1 for group in groups if _group_all_won(group, "baseline") and not _group_all_won(group, "candidate")
            ),
            "candidate_only_won": sum(
                1 for group in groups if _group_all_won(group, "candidate") and not _group_all_won(group, "baseline")
            ),
            "neither_won": sum(
                1 for group in groups if not _group_all_won(group, "baseline") and not _group_all_won(group, "candidate")
            ),
        },
    }


def _outcome_readout(outcomes: tuple[CapstoneGameOutcome, ...]) -> dict[str, Any]:
    score = sum(outcome.score for outcome in outcomes)
    return {
        "games": len(outcomes),
        "wins": sum(1 for outcome in outcomes if outcome.score == 1.0),
        "ties": sum(1 for outcome in outcomes if outcome.tied),
        "capped_games": sum(1 for outcome in outcomes if outcome.capped),
        "losses": sum(1 for outcome in outcomes if outcome.score == 0.0),
        "score": score,
        "score_rate": score / len(outcomes),
    }


def _stratified_bootstrap_means(
    groups: tuple[_SeedGroup, ...],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, ...]:
    by_band: dict[str, tuple[float, ...]] = {}
    for band in sorted({group.band for group in groups}):
        by_band[band] = tuple(group.delta for group in groups if group.band == band)
    rng = random.Random(seed)
    total_groups = len(groups)
    return tuple(
        sum(
            sum(values[rng.randrange(len(values))] for _ in range(len(values)))
            for values in by_band.values()
        )
        / total_groups
        for _ in range(replicates)
    )


def _group_all_won(group: _SeedGroup, arm: str) -> bool:
    outcomes = (
        (pair.baseline for pair in group.pairs)
        if arm == "baseline"
        else (pair.candidate for pair in group.pairs)
    )
    return all(outcome.score == 1.0 for outcome in outcomes)


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _scoped_seed(seed: int, scope: str) -> int:
    value = seed
    for character in scope:
        value = ((value * 31) + ord(character)) & 0xFFFFFFFF
    return value
