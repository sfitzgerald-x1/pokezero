"""Per-POKEMON HP comparison for boundaries where the active changed.

WHY THIS EXISTS. Analyses across this program have repeatedly compared a slot's
HP before and after a boundary. That is only valid while the same Pokemon
occupies the slot. When the active changes, ``pre_features[f"{slot}_hp"]`` and
``observed[f"{slot}_hp"]`` describe DIFFERENT Pokemon and their difference is
meaningless.

The failure is not hypothetical and it is not rare. It invalidated 105 of 341
residue rows in the C58 partition -- which had to be corrected twice -- and 37
of the 108 rows examined in C61. Roughly a third of the residue is affected, and
the error is silent: the subtraction succeeds and returns a plausible number.
s18000268/37 is the canonical case, where observed p2 nets +17 while the engine
branch nets -96, because p2 switched to Cradily mid-boundary.

WHAT THIS PROVIDES. ``slot_hp_comparable`` answers the only question a per-slot
comparison may safely ask: is this slot's HP delta meaningful at all? Callers
that need a verdict on a switched slot must compare the identified Pokemon
rather than the slot, which requires the protocol's switch lines; that is
deliberately NOT hidden behind a convenience wrapper, because a caller silently
receiving 0 for an uncomparable slot is how this class of error spreads.
"""

from __future__ import annotations

from typing import Any, Mapping


def slot_hp_comparable(row: Mapping[str, Any], slot: str) -> bool:
    """Whether ``row``'s pre/post HP for ``slot`` describe the same Pokemon.

    FAILS CLOSED. A row with no ``active_changed`` map, or a map that does not
    mention this slot, is NOT comparable. The first version of this returned
    ``not bool(active_changed.get(slot))``, so a missing key read as False --
    "the active did not change" -- and the helper produced exactly the +55
    cross-Pokemon delta on s18000268/37 that it exists to prevent. Only
    scripts/engine_transition_differential.py emits ``active_changed``, so any
    hand-built row, older artifact, or JSON from another producer hit that path.
    Absence of evidence that the slot held still is not evidence that it did.
    """

    active_changed = row.get("active_changed")
    if not isinstance(active_changed, Mapping) or slot not in active_changed:
        return False
    flag = active_changed[slot]
    # The value must be an explicit boolean. A JSON `null` from a producer that
    # encodes "unknown" that way is absence of evidence, and `not bool(None)`
    # would read it as "the active did not change" -- the same fail-open this
    # guard exists to close, one level down.
    if not isinstance(flag, bool):
        return False
    return not flag


def slot_hp_delta(row: Mapping[str, Any], slot: str) -> int | None:
    """Signed HP delta for ``slot``, or ``None`` when a switch makes it meaningless.

    Returning ``None`` rather than 0 is the point: a caller that forgets to
    handle the switch case gets a ``TypeError`` on the next arithmetic operation
    instead of a wrong number that flows into a published partition.
    """

    if not slot_hp_comparable(row, slot):
        return None
    pre = (row.get("pre_features") or {}).get(f"{slot}_hp")
    post = (row.get("observed") or {}).get(f"{slot}_hp")
    if pre is None or post is None:
        return None
    return int(post) - int(pre)


def comparable_slots(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The slots whose HP endpoints describe one Pokemon throughout."""

    return tuple(slot for slot in ("p1", "p2") if slot_hp_comparable(row, slot))
