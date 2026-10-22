"""Exact checkpoint support for a certified Gen3 private maybe-trap request.

Protocol-offered actions, the frozen checkpoint mask and executable native
arms are different objects. This is NOT a generic subset/fallback rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .actions import ACTION_COUNT


SCHEMA = "pokezero.gen3-maybe-trap-request.v1"


def _digest(value: Any, domain: bytes) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


def request_digest(request: Mapping[str, Any]) -> str:
    return _digest(request, b"pokezero.gen3-maybe-trap-own-request.v1")


def public_binding(lines: Sequence[str]) -> dict[str, Any]:
    """Bind an internal observer receipt to this exact public entry/turn.

    Full public history and request semantics are separately certified by the
    native observer and canonical replay parser. These are not FFI credentials.
    """
    entries = {"p1": 0, "p2": 0}
    actives = {"p1": "", "p2": ""}
    turn = 0
    for line in lines:
        fields = line.split("|")
        if len(fields) > 3 and fields[1] in {"switch", "drag"}:
            slot = fields[2][:2]
            if slot in entries:
                entries[slot] += 1
                actives[slot] = fields[2]
        if len(fields) > 2 and fields[1] == "turn":
            turn = int(fields[2])
    return {"sha256": _digest(list(lines), b"pokezero.gen3-maybe-trap-public-history.v1"),
            "public_line_count": len(lines), "turn": turn,
            "entries": entries, "actives": actives}


def _slots(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, (tuple, list)) or not raw:
        raise ValueError("empty observer action surface")
    slots = tuple(raw)
    if any(type(index) is not int or not 0 <= index < ACTION_COUNT for index in slots):
        raise ValueError("invalid observer action slot")
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate observer action slot")
    return slots


@dataclass(frozen=True)
class PolicyActionContract:
    native: tuple[int, ...]
    protocol: tuple[int, ...]
    policy: tuple[int, ...]

    @classmethod
    def certify(cls, *, request: Mapping[str, Any], receipt: Mapping[str, Any],
                native_indices: Sequence[int], canonical_mask: Sequence[bool],
                public_lines: Sequence[str], slot: str, format_id: str) -> PolicyActionContract:
        expected_fields = {"schema", "seat", "format_id", "public_binding", "request_sha256",
                           "protocol_action_indices", "policy_action_indices"}
        if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
            raise ValueError("incomplete private-request observer receipt")
        if (receipt["schema"] != SCHEMA or format_id != "gen3randombattle"
            or receipt["format_id"] != format_id or receipt["seat"] != slot):
            raise ValueError("private-request observer format/seat mismatch")
        if receipt["public_binding"] != public_binding(public_lines):
            raise ValueError("private-request observer public-entry binding mismatch")
        if receipt["request_sha256"] != request_digest(request):
            raise ValueError("private-request observer own-request binding mismatch")
        if request.get("wait") or request.get("forceSwitch") or request.get("teamPreview"):
            raise ValueError("private-request observer requires ordinary choice")
        rows = request.get("active")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError("private-request observer requires one active")
        active = rows[0]
        if active.get("maybeTrapped") is not True or active.get("trapped") is True:
            raise ValueError("private-request observer requires faithful maybeTrapped")
        if active.get("maybeDisabled") or active.get("maybeLocked"):
            raise ValueError("unsupported private-request observer flag")
        moves = active.get("moves")
        if not isinstance(moves, list) or not 1 <= len(moves) <= 4:
            raise ValueError("invalid observer move bank")
        if any(not isinstance(move, Mapping) or type(move.get("disabled")) is not bool for move in moves):
            raise ValueError("ambiguous observer move legality")
        policy = tuple(index for index, move in enumerate(moves) if not move["disabled"])
        team = request.get("side", {}).get("pokemon")
        if (request.get("side", {}).get("id") != slot or not isinstance(team, list)
            or not 2 <= len(team) <= 6 or any(not isinstance(mon, Mapping) for mon in team)
            or team[0].get("active") is not True
            or any(mon.get("active") is not False for mon in team[1:])):
            raise ValueError("observer request must use active-first complete own party")
        switches = []
        for index, mon in enumerate(team[1:]):
            condition = str(mon.get("condition", "")).split()
            try:
                hp = int(condition[0].split("/")[0])
            except (IndexError, ValueError) as error:
                raise ValueError("ambiguous observer bench life") from error
            if hp > 0:
                if "fnt" in condition:
                    raise ValueError("inconsistent observer bench life")
                switches.append(4 + index)
        if not policy or not switches:
            raise ValueError("observer certificate needs moves and a live bench")
        protocol = (*policy, *switches)
        native = _slots(native_indices)
        if (len(canonical_mask) != ACTION_COUNT or any(type(flag) is not bool for flag in canonical_mask)
            or tuple(index for index, flag in enumerate(canonical_mask) if flag) != policy):
            raise ValueError("canonical maybe-trap zero-switch law changed")
        if (_slots(receipt["policy_action_indices"]) != policy
            or _slots(receipt["protocol_action_indices"]) != protocol):
            raise ValueError("observer declared action surface differs from own request")
        if set(native) not in (set(policy), set(protocol)):
            raise ValueError("unexplained native/protocol/policy action difference")
        return cls(native, protocol, policy)

    def gather(self, row: Sequence[float]) -> tuple[float, ...]:
        if len(row) != ACTION_COUNT or any(not math.isfinite(p) or p < 0 for p in row):
            raise ValueError("invalid full checkpoint probability row")
        if any(row[index] != 0 for index in range(ACTION_COUNT) if index not in self.policy):
            raise ValueError("private-request observer would discard positive policy mass")
        gathered = tuple(row[index] for index in self.native)
        if math.fsum(gathered) != math.fsum(row) or math.fsum(row) <= 0:
            raise ValueError("private-request observer probability mass changed")
        return gathered
