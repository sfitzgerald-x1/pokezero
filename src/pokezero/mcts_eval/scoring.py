"""Mirrored-pair scoring, bootstrap intervals, and the fail-closed merge (plan D7/§3).

Scoring convention (existing controlled-FoulPlay convention, plan section 3):
win = 1, tie or decision cap = 0.5, loss = 0. PokeZero plays BOTH seats of every
team seed; the two seats average into one pair score, giving 50 independent pair
scores from 100 games. The headline is the mean over pairs with a deterministic
percentile bootstrap; comparisons bootstrap the PAIRED deltas using the same
resampled pair indices for every configuration, so two rows differ only by their
own results and never by resampling noise.

Parity language is deliberately constrained (section 3): 100 games is a
screening sample. ``parity_label`` only ever returns "clearly below parity",
"parity-compatible", or "directionally above parity" — never "parity achieved".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from typing import Any, Iterable, Mapping, Sequence

# win / tie / cap / loss -> score
OUTCOME_SCORES: Mapping[str, float] = {
    "win": 1.0,
    "tie": 0.5,
    "cap": 0.5,  # decision-cap termination scores as a tie by convention
    "loss": 0.0,
}
VALID_OUTCOMES = tuple(OUTCOME_SCORES)


class MergeError(RuntimeError):
    """Terminal merge failure — conflicting duplicates, missing seats, drift."""


@dataclass(frozen=True)
class GameResult:
    """One completed game. ``provenance_sha256`` binds the row to the exact
    checkpoint/engine/corpus contract that produced it."""

    config_id: str
    seed: int
    seat: str
    outcome: str
    turns: int
    provenance_sha256: str
    decision_walls_s: tuple[float, ...] = ()
    chosen_actions: tuple[str, ...] = ()
    opponent_crashed: bool = False

    def __post_init__(self) -> None:
        if self.seat not in {"p1", "p2"}:
            raise ValueError("seat must be p1 or p2.")
        if self.outcome not in OUTCOME_SCORES:
            raise ValueError(f"outcome must be one of {VALID_OUTCOMES}, got {self.outcome!r}.")

    @property
    def score(self) -> float:
        return OUTCOME_SCORES[self.outcome]

    @property
    def canonical_key(self) -> tuple[str, int, str]:
        return (self.config_id, self.seed, self.seat)

    @property
    def canonical_outcome(self) -> tuple[Any, ...]:
        """Fields that must match for a duplicate to be idempotent. Timing and
        telemetry are deliberately excluded: a retry's different wall time is a
        diagnostic, not a conflict (plan B3)."""
        return (self.outcome, self.score, self.turns, self.provenance_sha256, self.chosen_actions)


def merge_game_results(results: Iterable[GameResult]) -> dict[tuple[str, int, str], GameResult]:
    """Idempotent on canonically-matching duplicates; terminal on a real conflict."""
    merged: dict[tuple[str, int, str], GameResult] = {}
    for result in results:
        key = result.canonical_key
        existing = merged.get(key)
        if existing is None:
            merged[key] = result
            continue
        if existing.canonical_outcome != result.canonical_outcome:
            raise MergeError(
                f"canonical outcome conflict for {key}: {existing.canonical_outcome} != "
                f"{result.canonical_outcome}. A duplicate that disagrees on result, score, "
                "turns, provenance, or action sequence is never silently resolved."
            )
    return merged


def pair_scores(
    results: Iterable[GameResult],
    *,
    seeds: Sequence[int],
    config_id: str,
) -> list[float]:
    """Average the two seats per seed into one pair score, in ``seeds`` order.

    Both seats are required: a half-played pair would silently bias a row, so a
    missing seat fails closed rather than scoring the seed on one side.
    """
    merged = merge_game_results(r for r in results if r.config_id == config_id)
    scores: list[float] = []
    for seed in seeds:
        seats = {}
        for seat in ("p1", "p2"):
            result = merged.get((config_id, seed, seat))
            if result is None:
                raise MergeError(
                    f"{config_id}: seed {seed} is missing seat {seat}; every scored pair must "
                    "have both seats (mirrored-pair design)."
                )
            seats[seat] = result.score
        scores.append((seats["p1"] + seats["p2"]) / 2.0)
    return scores


def bootstrap_indices(
    *, sample_size: int, resamples: int, seed: int
) -> list[list[int]]:
    """Deterministic resample indices, shared across configurations.

    Reusing the same indices for every row is what makes a paired delta interval
    reflect the configurations rather than the resampling draw.
    """
    rng = random.Random(seed)
    return [
        [rng.randrange(sample_size) for _ in range(sample_size)] for _ in range(resamples)
    ]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample.")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def to_payload(self) -> dict[str, float]:
        return asdict(self)


def bootstrap_mean(
    values: Sequence[float], indices: Sequence[Sequence[int]]
) -> Interval:
    """Percentile bootstrap of the mean over precomputed resample indices."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample.")
    means = [sum(values[i] for i in draw) / len(draw) for draw in indices]
    return Interval(
        point=sum(values) / len(values),
        low=_percentile(means, 0.025),
        high=_percentile(means, 0.975),
    )


def bootstrap_paired_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
    indices: Sequence[Sequence[int]],
) -> Interval:
    """Paired delta over mirrored seed pairs, using the SAME resample indices."""
    if len(treatment) != len(baseline):
        raise ValueError("paired delta requires equal-length pair-score vectors.")
    deltas = [t - b for t, b in zip(treatment, baseline, strict=True)]
    return bootstrap_mean(deltas, indices)


def parity_label(interval: Interval) -> str:
    """Screening labels only — section 3 forbids 'parity achieved' at n=100."""
    if interval.high < 0.5:
        return "clearly below parity"
    if interval.low <= 0.5 <= interval.high:
        return "parity-compatible"
    if interval.point > 0.5:
        return "directionally above parity"
    return "clearly below parity"


def outcome_record(results: Iterable[GameResult], *, config_id: str) -> dict[str, int]:
    """Raw wins/ties/caps/losses, retained separately from the score (section 3)."""
    counts = {name: 0 for name in VALID_OUTCOMES}
    counts["opponent_crashes"] = 0
    for result in results:
        if result.config_id != config_id:
            continue
        counts[result.outcome] += 1
        if result.opponent_crashed:
            counts["opponent_crashes"] += 1
    return counts


def promote_spare_pairs(
    *,
    primary_seeds: Sequence[int],
    spare_seeds: Sequence[int],
    excluded: Iterable[int],
) -> tuple[list[int], dict[int, int]]:
    """Substitute spares for excluded pairs in a pre-registered, fixed order.

    A pair that still crashes after its retry in ANY entry is excluded from
    EVERY entry, so all entries keep a shared 50-pair scoring set. Exhausting
    the spare band is terminal.
    """
    excluded_set = list(dict.fromkeys(excluded))
    remaining = [seed for seed in primary_seeds if seed not in excluded_set]
    substitutions: dict[int, int] = {}
    spares = list(spare_seeds)
    for dropped in excluded_set:
        if not spares:
            raise MergeError(
                f"spare seed band exhausted while excluding pair {dropped}; the entries can no "
                "longer share 50 complete mirrored pairs."
            )
        promoted = spares.pop(0)
        substitutions[dropped] = promoted
        remaining.append(promoted)
    return remaining, substitutions
