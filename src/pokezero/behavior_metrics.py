"""Read-only behavior measurement helpers for population/diversity probes."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Protocol

from .dex import normalize_id
from .randbat import RECOVERY_MOVES, SETUP_MOVES, STATUS_INFLICTING_MOVES

MOVE_CLASS_ORDER = ("hazard", "clear", "setup", "status", "heal", "phaze", "attack", "other")

_HAZARD_MOVES = frozenset({"spikes", "stealthrock", "toxicspikes"})
_HAZARD_CLEAR_MOVES = frozenset({"rapidspin", "defog"})
_PHASE_MOVES = frozenset({"roar", "whirlwind"})
_STATUS_MOVES = frozenset(STATUS_INFLICTING_MOVES) | frozenset(
    {
        "confuseray",
        "glare",
        "grasswhistle",
        "hypnosis",
        "leechseed",
        "lovelykiss",
        "sleeppowder",
        "spore",
        "supersonic",
    }
)
_HEAL_MOVES = frozenset(RECOVERY_MOVES) | frozenset({"aromatherapy", "healbell", "rest", "wish"})


class _MoveInfo(Protocol):
    base_power: int
    gen3_category: str
    heal: bool
    status: str | None
    effect_label: str


class _Dex(Protocol):
    def move_info(self, move: str | None) -> _MoveInfo | None:
        ...


def classify_move(move_name: str, *, dex: _Dex | None = None) -> str:
    """Classify an observed move for read-only behavior dashboards.

    The classes are intentionally coarse measurement buckets. They are not used
    as rewards, selection rules, or matchmaking weights.
    """
    move_id = normalize_id(move_name)
    if move_id in _HAZARD_MOVES:
        return "hazard"
    if move_id in _HAZARD_CLEAR_MOVES:
        return "clear"
    if move_id in SETUP_MOVES:
        return "setup"
    if move_id in _PHASE_MOVES:
        return "phaze"
    if move_id in _HEAL_MOVES:
        return "heal"
    if move_id in _STATUS_MOVES:
        return "status"

    info = dex.move_info(move_name) if dex is not None else None
    if info is not None:
        if info.heal:
            return "heal"
        if info.status or info.effect_label in {"confusion", "leechseed"}:
            return "status"
        if info.base_power > 0 and info.gen3_category != "Status":
            return "attack"
        return "other"
    return "other"


def move_class_summary(move_counts: Mapping[str, int], *, dex: _Dex | None = None) -> dict[str, dict[str, float | int]]:
    class_counts: Counter[str] = Counter()
    total = 0
    for move_name, count in move_counts.items():
        count = int(count)
        if count <= 0:
            continue
        total += count
        class_counts[classify_move(move_name, dex=dex)] += count
    return {
        move_class: {
            "count": class_counts.get(move_class, 0),
            "rate": round(class_counts.get(move_class, 0) / total, 4) if total else 0.0,
        }
        for move_class in MOVE_CLASS_ORDER
    }
