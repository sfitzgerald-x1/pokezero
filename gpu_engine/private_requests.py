"""Seat-scoped action-request inputs for the public device encoder.

This ports request legality, canonical targets and PP fractions. Full own-team
feature packets and production observation columns are subsequent B2 work.
The host adapter reads exactly one seat's request; it accepts no engine state
or opponent request and performs no policy-feature encoding.
"""
from collections.abc import Mapping
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.showdown import _request_kind


SCHEMA = "pokezero.private_action_request.v1"


class RequestKind(IntEnum):
    NONE = 0
    WAIT = 1
    TEAM_PREVIEW = 2
    FORCE_SWITCH = 3
    MOVE = 4
    UNKNOWN = 5


_KINDS = dict(none=RequestKind.NONE, wait=RequestKind.WAIT,
              team_preview=RequestKind.TEAM_PREVIEW, force_switch=RequestKind.FORCE_SWITCH,
              move=RequestKind.MOVE, unknown=RequestKind.UNKNOWN)


class ActionRequest(NamedTuple):
    owner_side: jax.Array           # leading shape: absolute seat 0/1
    present: jax.Array
    request_id: jax.Array           # -1 when no rqid was supplied
    kind: jax.Array
    wait: jax.Array
    force_switch: jax.Array
    trapped: jax.Array              # source's exact True trap/maybeTrapped flags
    team_size: jax.Array
    team_active: jax.Array          # [...,6], own request order, not public slots
    team_fainted_prefix: jax.Array  # source _can_switch_to: condition starts '0 '
    move_present: jax.Array         # [...,4], mapping rows in active request
    move_disabled: jax.Array
    pp: jax.Array
    max_pp: jax.Array
    pp_known: jax.Array


class ActionInputs(NamedTuple):
    legal: jax.Array                # [...,9] bool
    switch_targets: jax.Array       # [...,5] own party indices, -1 for absent
    move_pp_fraction: jax.Array     # [...,4] valid row/missing PP=1; absent row=0
    ready: jax.Array                # owner matched, request present/nonwaiting


def parse_action_request(request, *, owner_side: int) -> ActionRequest:
    """Parse one singles request without deriving an opponent's private view.

    Unsupported multi-active or oversized packets fail admission explicitly.
    The source's unusual legality details are retained: PP zero alone does not
    disable a move, and forced switching overrides trapping. No request or a
    waiting request produces no fabricated actions.
    """
    if type(owner_side) is not int or owner_side not in (0, 1):
        raise ValueError("private request owner must be absolute side 0 or 1")
    if request is not None and not isinstance(request, Mapping):
        raise ValueError("private request must be a mapping or absent")
    raw = request or {}
    side = raw.get("side") or {}
    if not isinstance(side, Mapping):
        raise ValueError("invalid private request side")
    if side.get("id") is not None and side["id"] != f"p{owner_side + 1}":
        raise ValueError("private request belongs to a different seat")
    team = side.get("pokemon") or []
    if not isinstance(team, list) or len(team) > 6 or any(not isinstance(p, Mapping) for p in team):
        raise ValueError("private request needs at most six valid team rows")
    active_rows = raw.get("active") or []
    if not isinstance(active_rows, list) or len(active_rows) > 1 or any(not isinstance(p, Mapping) for p in active_rows):
        raise ValueError("only singles active requests are supported")
    active = active_rows[0] if active_rows else {}
    moves = active.get("moves") or []
    if not isinstance(moves, list) or len(moves) > 4:
        raise ValueError("active request exceeds four move slots")
    force = raw.get("forceSwitch")
    if force is not None and (not isinstance(force, list) or len(force) > 1):
        raise ValueError("only singles force-switch requests are supported")
    rqid = raw.get("rqid", -1)
    if type(rqid) is not int or not -1 <= rqid < 2**31:
        raise ValueError("private request ID must fit int32")
    team_active = np.zeros(6, bool)
    team_fainted = np.zeros(6, bool)
    for index, mon in enumerate(team):
        team_active[index] = bool(mon.get("active"))
        team_fainted[index] = str(mon.get("condition") or "").startswith("0 ")
    present, disabled = np.zeros(4, bool), np.zeros(4, bool)
    pp, max_pp, known = np.zeros(4, np.float32), np.zeros(4, np.float32), np.zeros(4, bool)
    for index, move in enumerate(moves):
        if not isinstance(move, Mapping):
            continue  # source's positional invalid-row mask; do not compact slots
        present[index], disabled[index] = True, bool(move.get("disabled", False))
        a, b = move.get("pp"), move.get("maxpp")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            try:
                a, b = float(a), float(b)
            except OverflowError as exc:
                raise ValueError("request PP exceeds float32 transport") from exc
            limits = np.finfo(np.float32)
            if any(not np.isfinite(x) or abs(x) > limits.max or (x != 0 and abs(x) < limits.tiny) for x in (a, b)):
                raise ValueError("request PP must be zero or finite normal float32 values")
            pp[index], max_pp[index], known[index] = a, b, True
    values = (np.int32(owner_side), np.bool_(request is not None), np.int32(rqid),
        np.int32(_KINDS[_request_kind(request)]), np.bool_(raw.get("wait", False)),
        np.bool_(isinstance(force, list) and any(bool(x) for x in force)),
        np.bool_(active.get("trapped") is True or active.get("maybeTrapped") is True),
        np.int32(len(team)), team_active, team_fainted, present, disabled, pp, max_pp, known)
    return ActionRequest(*(jnp.asarray(x) for x in values))


def action_inputs(request: ActionRequest, *, observer_side) -> ActionInputs:
    """Compute source-compatible actions entirely on device for the owning seat.

    Wrong-seat access returns no actions, targets or PP facts. The later encoder
    must use the same owner check before consuming private own-team features.
    """
    if type(observer_side) is int:
        if observer_side not in (0, 1):
            raise ValueError("observer side must be 0 or 1")
        observer = jnp.asarray(observer_side, jnp.int32)
    else:
        # Reject floats/bools and wide integer coercions before a cast can
        # alias an invalid observer into an authorized seat.
        if getattr(observer_side, "dtype", None) != jnp.dtype(jnp.int32):
            raise ValueError("device observer sides must be int32, without coercion")
        observer = jnp.asarray(observer_side)
    owner = (observer >= 0) & (observer <= 1) & (request.owner_side == observer)
    ready = owner & request.present & ~request.wait
    has_active = jnp.any(request.team_active, axis=-1)
    active_index = jnp.argmax(request.team_active, axis=-1)
    ordinal = jnp.arange(5)
    targets = ordinal + (ordinal >= active_index[..., None])
    target_valid = owner[..., None] & has_active[..., None] & (request.team_size[..., None] >= 2) & (targets < request.team_size[..., None])
    targets = jnp.where(target_valid, targets, -1).astype(jnp.int32)
    gather = jnp.clip(targets, 0, 5)
    target_fainted = jnp.take_along_axis(request.team_fainted_prefix, gather, axis=-1)
    target_active = jnp.take_along_axis(request.team_active, gather, axis=-1)
    switching = request.force_switch | ((request.kind == RequestKind.MOVE) & ~request.trapped)
    legal_switch = ready[..., None] & switching[..., None] & target_valid & ~target_fainted & ~target_active
    legal_move = ready[..., None] & ~request.force_switch[..., None] & request.move_present & ~request.move_disabled
    denominator = jnp.where(request.max_pp != 0, request.max_pp, 1)
    pp = jnp.where(request.pp_known & (request.max_pp != 0), jnp.clip(request.pp / denominator, 0, 1), 1)
    return ActionInputs(jnp.concatenate((legal_move, legal_switch), axis=-1), targets,
                         jnp.where(owner[..., None] & request.move_present, pp, 0), ready)
