"""Our own Pokemon's real movesets and exact PP from the owner's own protocol history (V2 ``self_moves``).

The owner knows its PP exactly: every move request carries the active Pokemon's ``pp``/``maxpp``, and a benched
Pokemon's PP is its latest active snapshot minus the public charges since. This is
``gpu_engine.phase_d_constructor_rules.own_pp_carry``'s rule, the PP the protocol constructor gives its engine worlds,
ported here PER POKEMON so that the encoders (the Python encoder and B2) and the worlds read one quantity. Where
own_pp_carry refuses the whole team, a Pokemon whose PP the history cannot state gets ``pp=None`` with that reason; a
history that fails a team-wide precondition gives every Pokemon ``pp=None`` with the reason. Where own_pp_carry succeeds,
the PP of every non-transformed Pokemon is exactly its result (tests/test_own_moves.py pins this).

The moves are the REAL moveset, never Transform's copies: a transformed active's request lists the copied move slots
(5 PP each), so the real set is taken from the owner's battle-start request (``side.pokemon[i].moves``, which predates
any Transform), and a transformed Pokemon's PP is its own ledger as the ``-transform`` line found it (own_pp_carry's
``base``, which it restores at switch-out). Without a battle-start request in the history the current request's moves are
used unless that Pokemon is transformed, in which case its moves and PP are unknown (empty moves, ``pp=None``).

Move ids are canonical request ids (``return102`` -> ``return``; Hidden Power keeps its type, ``hiddenpowerfire``), the
belief:possible_move labels; PP is looked up under own_pp_carry's key (Hidden Power generic). Pure Python, no JAX.
"""
from __future__ import annotations

import json
from typing import Callable, Mapping, NamedTuple, Sequence

from .belief import _CALLER_MOVES, _NEVER_PRESSURED_POOL_MOVES, _called_move_source
from .dex import normalize_id
from .tier2 import canonical_move_id

__all__ = ("OwnMoveState", "own_move_states", "move_label", "move_key")


class OwnMoveState(NamedTuple):
    ident: str                      # the request row's identity (casefolded name after the side prefix)
    moves: tuple[str, ...]          # canonical real-set move ids (labels), request order; () when unknown
    pp: tuple[int, ...] | None      # exact remaining PP per move, same order; None when the history cannot state it
    reason: str                     # '' when pp is exact, else the refusal code (own_pp_carry's names)


def move_label(value: str) -> str:
    """A request move id as the move it names: Return/Frustration (and Hidden Power from gen 6) lose the resolved-power
    suffix (``tier2.canonical_move_id``'s rule); Hidden Power keeps its type."""
    return canonical_move_id(value)


def move_key(value: str) -> str:
    """own_pp_carry's PP key (``phase_d_constructor_rules._move_key``): Hidden Power generic, others canonical."""
    value = normalize_id(value)
    return "hiddenpower" if value.startswith("hiddenpower") else canonical_move_id(value)


def _identity(value: str) -> str:
    return str(value).split(":", 1)[-1].strip().casefold()


def _request(line: str, owner: str) -> Mapping | None:
    """The owner's request packet of a ``|request|`` line; None for another seat's (the env's log interleaves both)."""
    raw = json.loads(line[len("|request|"):])
    side = raw.get("side") if isinstance(raw, dict) else None
    if not isinstance(side, dict) or side.get("id") != owner:
        return None
    return raw


def _end_transform(ledger: dict, base: dict, identity: str) -> None:
    value = base.pop(identity)
    if value is None:
        ledger.pop(identity, None)
    else:
        ledger[identity] = value


def own_move_states(lines: Sequence[str], request: Mapping, *, maximum_pp: Callable[[str], int]) -> tuple[OwnMoveState, ...]:
    """Per row of ``request['side']['pokemon']`` (request order): the real moves and exact PP (module docstring).

    ``lines`` is the owner's protocol history up to and including ``request`` (``|request|`` lines included, another
    seat's requests ignored); ``maximum_pp(key)`` is the catalog max PP of a move key (a Pokemon never active and never
    moved has full PP).
    """
    owner = request["side"]["id"]
    party = request["side"]["pokemon"]
    identities = [_identity(p.get("ident", "")) for p in party]
    current = {identity: [move_label(m) for m in row.get("moves", ()) if move_label(m)] for identity, row in zip(identities, party)}
    # The real set per member: its moves in the earliest owner request taken while it was NOT transformed (the battle-start
    # request for every member of a complete history). A member transformed in every request seen has no known real set.
    real: dict = {}
    transformed: set = set()
    on_field = None
    for line in lines:
        p = line.split("|")
        kind = p[1] if len(p) > 1 else ""
        if kind == "request":
            packet = _request(line, owner)
            if packet is None:
                continue
            for r in packet["side"]["pokemon"]:
                identity = _identity(r.get("ident", ""))
                if identity not in transformed and identity not in real:
                    real[identity] = [move_label(m) for m in r.get("moves", ()) if move_label(m)]
        elif kind in ("switch", "drag", "replace") and len(p) > 3 and p[2][:2] == owner:
            if on_field is not None and on_field != _identity(p[2]):
                transformed.discard(on_field)
            on_field = _identity(p[2])
        elif kind == "-transform" and len(p) > 3 and p[2][:2] == owner:
            transformed.add(_identity(p[2]))
        elif kind == "faint" and len(p) > 2 and p[2][:2] == owner:
            transformed.discard(_identity(p[2]))

    def states(ledger_value: Callable[[str], object], team_reason: str = "") -> tuple[OwnMoveState, ...]:
        out = []
        for identity in identities:
            if identity in real:
                moves = real[identity]
            elif identity in transformed:
                out.append(OwnMoveState(identity, (), None, "OWN_REAL_MOVESET_UNAVAILABLE"))
                continue
            else:
                moves = current[identity]
            if team_reason:
                out.append(OwnMoveState(identity, tuple(moves), None, team_reason))
                continue
            value = ledger_value(identity)
            keys = [move_key(m) for m in moves]
            if not isinstance(value, dict):
                out.append(OwnMoveState(identity, tuple(moves), None, value or "OWN_PP_LAST_ACTIVE_REQUEST_UNAVAILABLE"))
            elif set(value) != set(keys):
                out.append(OwnMoveState(identity, tuple(moves), None, "OWN_PP_REQUEST_MOVESET_DIFFERS"))
            else:
                out.append(OwnMoveState(identity, tuple(moves), tuple(int(value[k]) for k in keys), ""))
        return tuple(out)

    if len(identities) != 6 or not all(identities) or len(set(identities)) != 6 or "|start" not in lines:
        return states(lambda _: None, "OWN_PP_COMPLETE_UNIQUE_START_HISTORY_REQUIRED")
    # ---- own_pp_carry's ledger, line for line (per Pokemon instead of a team-wide raise) ----
    ledger: dict = {}
    active: dict = {}
    ability: dict = {}
    moved: set = set()
    base: dict = {}
    last = None
    for line in lines:
        p = line.split("|")
        kind = p[1] if len(p) > 1 else ""
        if kind == "request":
            packet = _request(line, owner)
            if packet is None:
                continue
            last = packet
            rows = packet["side"]["pokemon"]
            if {_identity(r.get("ident", "")) for r in rows} != set(identities):
                return states(lambda _: None, "OWN_PP_REQUEST_IDENTITY_CHANGED")
            now = [r for r in rows if r.get("active") is True]
            if len(now) != 1:
                continue
            identity = _identity(now[0]["ident"])
            if packet.get("wait") or any(packet.get("forceSwitch", ())) or packet.get("teamPreview"):
                continue
            moves = packet.get("active", ())
            moves = moves[0].get("moves", ()) if len(moves) == 1 else ()
            expected = [move_key(m) for m in now[0].get("moves", ())]
            values: dict = {}
            for row in moves:
                key = move_key(row.get("id", ""))
                pp, maximum = row.get("pp"), row.get("maxpp")
                if key in values or type(pp) is not int or type(maximum) is not int or not 0 <= pp <= maximum:
                    values = {}
                    break
                values[key] = pp
            if values and len(expected) == len(set(expected)) and set(values) == set(expected):
                ledger[identity] = values
            elif not moves or any(move_key(row.get("id", "")) not in ("struggle", "recharge", "solarbeam") for row in moves):
                ledger[identity] = "OWN_PP_LAST_ACTIVE_REQUEST_UNAVAILABLE"
        elif kind in ("switch", "drag", "replace") and len(p) > 3:
            side = p[2][:2]
            identity = _identity(p[2])
            if side == owner and active.get(side) in base and active[side] != identity:
                _end_transform(ledger, base, active[side])
            active[side] = identity
            ability[side] = ""
            if side == owner and identity not in identities:
                return states(lambda _: None, "OWN_PP_PUBLIC_IDENTITY_UNAVAILABLE")
        elif kind in ("-ability", "-endability") and len(p) > 3:
            side = p[2][:2]
            if active.get(side) == _identity(p[2]):
                ability[side] = normalize_id(p[3]) if kind == "-ability" else ""
        elif kind == "-transform" and len(p) > 3 and p[2][:2] == owner:
            identity = _identity(p[2])
            value = ledger.get(identity)
            base[identity] = dict(value) if isinstance(value, dict) else value
        elif kind == "faint" and len(p) > 2 and p[2][:2] == owner and _identity(p[2]) in base:
            _end_transform(ledger, base, _identity(p[2]))
        elif kind == "move" and len(p) > 3 and p[2][:2] == owner:
            identity = _identity(p[2])
            moved.add(identity)
            value = ledger.get(identity)
            if not isinstance(value, dict):
                continue
            move = move_key(p[3])
            caller = _called_move_source(line)
            if move == "struggle" or caller in _CALLER_MOVES | {"sleeptalk", "lockedmove"}:
                continue
            if caller is not None:
                ledger[identity] = "OWN_PP_MOVE_COST_UNAVAILABLE"
                continue
            if active.get(owner) != identity or move not in value:
                ledger[identity] = "OWN_PP_MOVE_SLOT_UNAVAILABLE"
                continue
            cost = 1 + int(ability.get("p2" if owner == "p1" else "p1") == "pressure" and move not in _NEVER_PRESSURED_POOL_MOVES)
            value[move] = max(0, value[move] - cost)
        if any(token in line.lower() for token in ("spite", "grudge", "leppaberry", "leppa berry")):
            identity = active.get(owner)
            if identity in ledger:
                ledger[identity] = "OWN_PP_NONMOVE_CHANGE_UNAVAILABLE"
    if last != request:
        return states(lambda _: None, "OWN_PP_CURRENT_OWN_REQUEST_REQUIRED")

    def value_of(identity: str) -> object:
        # A transformed Pokemon's real-set PP is its ledger as the -transform line found it (own_pp_carry's base).
        if identity in base:
            return base[identity]
        if identity not in ledger and identity not in moved:
            return {move_key(m): int(maximum_pp(move_key(m))) for m in real.get(identity, current[identity])}
        return ledger.get(identity)

    return states(value_of)
