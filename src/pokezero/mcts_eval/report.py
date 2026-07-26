"""Eligibility, pruning, Pareto frontier, and the report table (plan D8/A4/B4).

The output of the study is a FRONTIER, not a single ladder setting: a
configuration is on the frontier when no other measured configuration is both
faster and stronger. Everything here is descriptive screening — no multiplicity
adjustment is applied or implied (section 3).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Mapping, Sequence

from .manifest import SearchConfig
from .scoring import Interval, parity_label

DECISION_WALL_GATE_S = 15.0
MAX_SEARCH_ENTRIES = 7
MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP = 64


@dataclass(frozen=True)
class TimingRow:
    """One A3 lattice cell's measured timing + realized-depth telemetry."""

    config_id: str
    depth: int
    sims: int
    decisions_timed: int
    mean_wall_s: float
    median_wall_s: float
    p95_wall_s: float
    max_wall_s: float
    realized_depth_mean: float
    realized_depth_max: int
    cap_hit_rate: float
    encode_s: float = 0.0
    model_s: float = 0.0
    tree_s: float = 0.0
    fallbacks: int = 0
    invalid_actions: int = 0
    gate_failed: bool = False
    provenance_exact: bool = True
    root_argmax_by_decision: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        """Plan A4: strictly under the gate, complete, zero fallbacks/invalids,
        exact provenance. The gate is an EXPERIMENT gate, not a ladder safety
        guarantee — p95/max are reported regardless."""
        return (
            not self.gate_failed
            and self.provenance_exact
            and self.mean_wall_s < DECISION_WALL_GATE_S
            and self.fallbacks == 0
            and self.invalid_actions == 0
            and self.decisions_timed > 0
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible"] = self.eligible
        payload["root_argmax_by_decision"] = list(self.root_argmax_by_decision)
        return payload


def dominated(row: TimingRow, other: TimingRow) -> bool:
    """``row`` is dominated when ``other`` is no slower and reaches at least as deep.

    Plan A4's first pruning rule, applied within a breadth (same sims): discard a
    cell that is slower and reaches no deeper than another.
    """
    if row.config_id == other.config_id or row.sims != other.sims:
        return False
    return other.mean_wall_s <= row.mean_wall_s and other.realized_depth_max >= row.realized_depth_max


def select_candidates(
    rows: Sequence[TimingRow], *, max_entries: int = MAX_SEARCH_ENTRIES
) -> tuple[TimingRow, ...]:
    """Plan A4 selection: per configured depth keep the largest eligible
    simulation count (ties broken by lower mean wall); if that yields fewer than
    ``max_entries``, backfill with the fastest remaining eligible cell and then
    the cell nearest the median eligible wall. Caps at ``max_entries``."""
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        return ()
    survivors = [
        row
        for row in eligible
        if not any(dominated(row, other) for other in eligible)
    ]
    by_depth: dict[int, TimingRow] = {}
    for row in survivors:
        current = by_depth.get(row.depth)
        if current is None or (row.sims, -row.mean_wall_s) > (current.sims, -current.mean_wall_s):
            by_depth[row.depth] = row
    selected = sorted(by_depth.values(), key=lambda row: (row.depth, row.sims))
    if len(selected) < max_entries:
        chosen_ids = {row.config_id for row in selected}
        remaining = [row for row in eligible if row.config_id not in chosen_ids]
        if remaining:
            fastest = min(remaining, key=lambda row: row.mean_wall_s)
            selected.append(fastest)
            chosen_ids.add(fastest.config_id)
        remaining = [row for row in eligible if row.config_id not in chosen_ids]
        if remaining and len(selected) < max_entries:
            walls = sorted(row.mean_wall_s for row in eligible)
            median_wall = walls[len(walls) // 2]
            nearest = min(remaining, key=lambda row: abs(row.mean_wall_s - median_wall))
            selected.append(nearest)
    return tuple(selected[:max_entries])


@dataclass(frozen=True)
class StrengthRow:
    """One Phase-B entry: 100 games, 50 mirrored pairs."""

    config_id: str
    foulplay_rung: str
    record: Mapping[str, int]
    score: Interval
    delta_vs_raw: Interval | None
    timing: TimingRow | None = None
    fallbacks: int = 0
    timeouts: int = 0

    @property
    def parity(self) -> str:
        return parity_label(self.score)

    def to_payload(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "foulplay_rung": self.foulplay_rung,
            "record": dict(self.record),
            "score_95ci": self.score.to_payload(),
            "parity_label": self.parity,
            "delta_vs_raw_95ci": self.delta_vs_raw.to_payload() if self.delta_vs_raw else None,
            "mean_s": self.timing.mean_wall_s if self.timing else None,
            "p95_s": self.timing.p95_wall_s if self.timing else None,
            "max_s": self.timing.max_wall_s if self.timing else None,
            "realized_depth": (
                {
                    "mean": self.timing.realized_depth_mean,
                    "max": self.timing.realized_depth_max,
                    "cap_hit_rate": self.timing.cap_hit_rate,
                }
                if self.timing
                else None
            ),
            "sims": self.timing.sims if self.timing else None,
            "fallbacks": self.fallbacks,
            "timeouts": self.timeouts,
        }


def pareto_frontier(rows: Sequence[StrengthRow]) -> tuple[StrengthRow, ...]:
    """On the frontier when no other row is both faster AND stronger.

    Comparison uses point estimates; the intervals are published alongside so a
    reader can see how much of the ordering is screening noise.
    """
    scored = [row for row in rows if row.timing is not None]
    frontier: list[StrengthRow] = []
    for row in scored:
        beaten = any(
            other is not row
            and other.timing.mean_wall_s <= row.timing.mean_wall_s
            and other.score.point >= row.score.point
            and (
                other.timing.mean_wall_s < row.timing.mean_wall_s
                or other.score.point > row.score.point
            )
            for other in scored
        )
        if not beaten:
            frontier.append(row)
    return tuple(sorted(frontier, key=lambda row: row.timing.mean_wall_s))


def root_action_agreement(left: TimingRow, right: TimingRow) -> float | None:
    """Fraction of corpus decisions where two cells pick the same root action.

    Adjacent-cell agreement is how the report shows whether extra depth/breadth
    actually changes decisions or merely costs wall time.
    """
    if not left.root_argmax_by_decision or not right.root_argmax_by_decision:
        return None
    pairs = list(zip(left.root_argmax_by_decision, right.root_argmax_by_decision, strict=False))
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def render_report(
    strength_rows: Sequence[StrengthRow],
    *,
    timing_rows: Sequence[TimingRow] = (),
    manifest_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable report: the B4 table, the frontier, and the raw ledgers."""
    frontier_ids = [row.config_id for row in pareto_frontier(strength_rows)]
    return {
        "schema_version": "pokezero.mcts-depth-eval.report.v1",
        "manifest": dict(manifest_payload) if manifest_payload else None,
        "strength_table": [row.to_payload() for row in strength_rows],
        "timing_table": [row.to_payload() for row in timing_rows],
        "pareto_frontier": frontier_ids,
        "notes": [
            "100 games per entry is a screening sample, not proof of parity "
            "(plan section 3); no multiplicity adjustment is applied.",
            "A depth label is valid only alongside its realized-depth telemetry.",
        ],
    }


def render_markdown_table(rows: Sequence[StrengthRow]) -> str:
    """The plan's B4 table as Markdown."""
    header = (
        "| config_id | rung | record (W/T/C/L) | score 95% CI | parity | "
        "delta vs raw | mean/p95/max s | realized depth | fallbacks |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        record = row.record
        delta = (
            f"{row.delta_vs_raw.point:+.3f} [{row.delta_vs_raw.low:+.3f},{row.delta_vs_raw.high:+.3f}]"
            if row.delta_vs_raw
            else "—"
        )
        timing = row.timing
        walls = (
            f"{timing.mean_wall_s:.2f}/{timing.p95_wall_s:.2f}/{timing.max_wall_s:.2f}"
            if timing
            else "—"
        )
        depth = (
            f"{timing.realized_depth_mean:.1f} (max {timing.realized_depth_max}, "
            f"cap {timing.cap_hit_rate:.0%})"
            if timing
            else "—"
        )
        lines.append(
            f"| {row.config_id} | {row.foulplay_rung} | "
            f"{record.get('win', 0)}/{record.get('tie', 0)}/{record.get('cap', 0)}/{record.get('loss', 0)} | "
            f"{row.score.point:.3f} [{row.score.low:.3f},{row.score.high:.3f}] | {row.parity} | "
            f"{delta} | {walls} | {depth} | {row.fallbacks} |"
        )
    return header + "\n".join(lines) + "\n"
