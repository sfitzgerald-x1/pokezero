"""Seed-paired strength analysis for the test-time-search capstone.

The capstone plays each BattleStream seed from both seats.  This module keeps the
two seats together when bootstrapping, preventing a seat-specific result from
being counted as an independent seed-level sample.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.band, self.seed, self.seat)


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

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


def paired_capstone_delta(
    baseline_outcomes: Iterable[CapstoneGameOutcome],
    candidate_outcomes: Iterable[CapstoneGameOutcome],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260710,
    require_mirrored_seats: bool = True,
) -> dict[str, Any]:
    """Return deterministic paired deltas and bootstrap intervals for one capstone arm.

    Inputs must have exactly matching ``(band, seed, seat)`` keys.  When mirrored
    seats are required, the p1/p2 outcomes for each seed are first averaged and
    that seed-level value is the bootstrap unit.
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

    pairs = tuple(
        _PairedGame(baseline=baseline_by_key[key], candidate=candidate_by_key[key])
        for key in sorted(baseline_by_key)
    )
    groups = _seed_groups(pairs, require_mirrored_seats=require_mirrored_seats)
    result = {
        "analysis_method": "paired_bootstrap_seed_group_mean_scores",
        "outcome_scoring": dict(OUTCOME_SCORING),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "unit": "seed_mean_over_mirrored_seats" if require_mirrored_seats else "seed_seat_game",
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


def _seed_groups(
    pairs: tuple[_PairedGame, ...],
    *,
    require_mirrored_seats: bool,
) -> tuple[_SeedGroup, ...]:
    by_seed: dict[tuple[str, int], list[_PairedGame]] = {}
    for pair in pairs:
        key = (pair.baseline.band, pair.baseline.seed)
        by_seed.setdefault(key, []).append(pair)

    groups: list[_SeedGroup] = []
    for (band, seed), seed_pairs in sorted(by_seed.items()):
        seats = {pair.baseline.seat for pair in seed_pairs}
        if require_mirrored_seats and seats != _MIRRORED_SEATS:
            raise ValueError(f"seed {(band, seed)!r} must contain exactly mirrored p1 and p2 outcomes.")
        groups.append(
            _SeedGroup(
                band=band,
                seed=seed,
                baseline_score=sum(pair.baseline.score for pair in seed_pairs) / len(seed_pairs),
                candidate_score=sum(pair.candidate.score for pair in seed_pairs) / len(seed_pairs),
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
    deltas = tuple(group.delta for group in groups)
    bootstrap_deltas = _bootstrap_means(
        deltas,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "paired_games": len(pairs),
        "paired_seed_groups": len(groups),
        "baseline": _outcome_readout(baseline),
        "candidate": _outcome_readout(candidate),
        "candidate_minus_baseline_score_rate": sum(deltas) / len(deltas),
        "paired_bootstrap_95": {
            "lower": _percentile(bootstrap_deltas, 0.025),
            "upper": _percentile(bootstrap_deltas, 0.975),
        },
        "score_flip_counts": {
            "candidate_better": sum(1 for pair in pairs if pair.delta > 0.0),
            "baseline_better": sum(1 for pair in pairs if pair.delta < 0.0),
            "equal": sum(1 for pair in pairs if pair.delta == 0.0),
        },
        "win_flip_counts": {
            "both_won": sum(1 for pair in pairs if pair.baseline.score == 1.0 and pair.candidate.score == 1.0),
            "baseline_only_won": sum(
                1 for pair in pairs if pair.baseline.score == 1.0 and pair.candidate.score != 1.0
            ),
            "candidate_only_won": sum(
                1 for pair in pairs if pair.candidate.score == 1.0 and pair.baseline.score != 1.0
            ),
            "neither_won": sum(1 for pair in pairs if pair.baseline.score != 1.0 and pair.candidate.score != 1.0),
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


def _bootstrap_means(values: tuple[float, ...], *, replicates: int, seed: int) -> tuple[float, ...]:
    rng = random.Random(seed)
    count = len(values)
    return tuple(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )


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
