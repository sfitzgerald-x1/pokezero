"""Join timing and strength rows into the Pareto frontier (plan deliverable 8).

The study's output is explicitly "a Pareto frontier, not a single hard-coded
ladder setting": a configuration is ON the frontier when no other measured
configuration is both faster AND stronger. Timing rows and strength rows are
produced by different jobs and joined on ``config_id``, which is why that id is
the immutable identity of a lattice cell.

Parity language follows plan section 3 and is deliberately conservative: a
screening sample never says "parity achieved".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FrontierRow:
    """One configuration with both axes measured."""

    config_id: str
    depth: int
    sims: int
    batch: int
    worlds: int
    mean_s: float
    p95_s: float
    max_s: float
    encode_share: float
    model_share: float
    gate_pass: bool
    games: int | None = None
    wins: int | None = None
    win_rate: float | None = None
    win_rate_lo: float | None = None
    win_rate_hi: float | None = None
    parity_label: str | None = None
    on_frontier: bool = False

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(wins: int, games: int, *, z: float = 1.96) -> tuple[float, float]:
    """Binomial Wilson interval — the plan's illustration of screening uncertainty.

    Wilson rather than normal-approximation because it stays inside [0, 1] and
    behaves at the extremes a 20-game screen can produce (0 or n wins).
    """
    if games <= 0:
        return (0.0, 1.0)
    phat = wins / games
    denom = 1.0 + z * z / games
    center = (phat + z * z / (2 * games)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def parity_label(lo: float, hi: float, point: float) -> str:
    """Plan section 3 vocabulary. Never returns 'parity achieved'."""
    if hi < 0.5:
        return "clearly below parity"
    if lo > 0.5:
        return "directionally above parity"
    if point > 0.5:
        return "parity-compatible (point above 50%)"
    return "parity-compatible"


def mark_frontier(rows: Sequence[FrontierRow]) -> tuple[FrontierRow, ...]:
    """A row is on the frontier when nothing measured is both faster and stronger.

    Only rows carrying BOTH axes can be placed: a configuration with no strength
    read is not "not on the frontier", it is unmeasured, and is returned with
    ``on_frontier=False`` without being used to dominate anything.
    """
    scored = [row for row in rows if row.win_rate is not None and row.gate_pass]
    marked: list[FrontierRow] = []
    for row in rows:
        if row.win_rate is None or not row.gate_pass:
            marked.append(row)
            continue
        dominated = any(
            other.mean_s <= row.mean_s
            and other.win_rate >= row.win_rate
            and (other.mean_s < row.mean_s or other.win_rate > row.win_rate)
            for other in scored
            if other.config_id != row.config_id
        )
        marked.append(
            FrontierRow(**{**row.to_payload(), "on_frontier": not dominated})
        )
    return tuple(marked)


def _timing_key(payload: Mapping[str, Any]) -> str:
    """Timing rows carry the full id; strength rows carry the same minus mode."""
    return str(payload.get("config", "")).replace("-w4", "")


def build_frontier(
    timing_dir: str | Path,
    strength_dir: str | Path,
) -> tuple[FrontierRow, ...]:
    """Join the two artifact directories on config id and rank the frontier."""
    timing: dict[str, Mapping[str, Any]] = {}
    for path in sorted(Path(timing_dir).glob("timing-*.json")):
        payload = json.loads(path.read_text())
        if payload.get("n"):
            timing[_timing_key(payload)] = payload

    strength: dict[str, Mapping[str, Any]] = {}
    for path in sorted(Path(strength_dir).glob("strength-*.json")):
        payload = json.loads(path.read_text())
        if "error" not in payload:
            strength[_timing_key(payload)] = payload

    rows: list[FrontierRow] = []
    for key, cell in sorted(timing.items()):
        total = cell["encode_s"] + cell["model_s"] + cell["tree_s"]
        played = strength.get(key)
        wins = games = None
        rate = lo = hi = None
        label = None
        if played:
            wins, games = int(played["wins"]), int(played["games"])
            rate = wins / games if games else None
            lo, hi = wilson_interval(wins, games)
            label = parity_label(lo, hi, rate or 0.0)
        rows.append(
            FrontierRow(
                config_id=str(cell["config"]),
                depth=int(cell["depth"]), sims=int(cell["sims"]),
                batch=int(cell["batch"]), worlds=int(cell["worlds"]),
                mean_s=float(cell["mean_s"]), p95_s=float(cell["p95_s"]),
                max_s=float(cell["max_s"]),
                encode_share=round(cell["encode_s"] / total, 3) if total else 0.0,
                model_share=round(cell["model_s"] / total, 3) if total else 0.0,
                gate_pass=bool(cell.get("gate_pass_15s")),
                games=games, wins=wins,
                win_rate=round(rate, 3) if rate is not None else None,
                win_rate_lo=round(lo, 3) if lo is not None else None,
                win_rate_hi=round(hi, 3) if hi is not None else None,
                parity_label=label,
            )
        )
    return mark_frontier(rows)


def render_markdown(rows: Iterable[FrontierRow]) -> str:
    """Report table. Rows without a strength read are shown as timing-only."""
    lines = [
        "| config | mean s | p95 s | enc% | mdl% | gate | record | win rate (95% CI) | reading | frontier |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r.mean_s,)):
        record = f"{row.wins}/{row.games}" if row.games else "—"
        if row.win_rate is None:
            rate = "—"
        elif row.win_rate_lo is None or row.win_rate_hi is None:
            # A point estimate without an interval is reported bare rather than
            # dressed with a fabricated range.
            rate = f"{row.win_rate:.0%}"
        else:
            rate = f"{row.win_rate:.0%} ({row.win_rate_lo:.0%}–{row.win_rate_hi:.0%})"
        lines.append(
            f"| {row.config_id} | {row.mean_s:.2f} | {row.p95_s:.2f} | "
            f"{row.encode_share:.0%} | {row.model_share:.0%} | "
            f"{'PASS' if row.gate_pass else 'FAIL'} | {record} | {rate} | "
            f"{row.parity_label or '—'} | {'★' if row.on_frontier else ''} |"
        )
    return "\n".join(lines)
