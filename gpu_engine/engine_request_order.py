"""Source singles request order over fixed engine physical party storage.

Pinned Showdown switchIn swaps side.pokemon[0] and the incoming position;
BattleState instead changes its active physical index. Keep that permutation
explicit. Only an actual installed-switch callback may advance this ledger,
including ordinary, forced, Baton and drag installations. Do not infer a switch
from chosen actions, a species match, or a final-state difference.

Source: Showdown f76228a1354b5d0f307ca2d16101294ad3a2308b,
sim/battle-actions.ts:switchIn and sim/pokemon.ts:getSwitchRequestData;
engine actions use state.py:bench_indices. These APIs do not attest that the
missing transition callbacks or complete request builder have been integrated.

This is a private owner packet/choice seam, NOT a public event emitter, request
builder, observation encoder or rollout integration. Stream IDs and engine
bindings are constructor attestations supplied by the caller. The caller must
feed every actual installation and provide masks/move-slot maps from the SAME
own request. Missing move routing remains unknown; hard locks and Struggle do
not justify assuming that request slot zero is physical move slot zero.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .engine_private_team import OwnedTeam

SCHEMA = 'pokezero.engine_request_order.v1'


class Order(NamedTuple):
    stream_id: object
    owner_side: object
    engine_binding: object
    team_size: object
    generation: object
    request_to_physical: object
    admitted: object


class SwitchInstalled(NamedTuple):
    """One owner's next source callback; false installed rows carry no switch."""
    stream_id: object
    owner_side: object
    engine_binding: object
    generation: object
    installed: object
    outgoing_physical: object
    incoming_physical: object


class RequestBoundary(NamedTuple):
    """Source packet identity, including independently gathered active index.

    Read active_physical from the actual owner state, not from this ledger.
    Generation is the installed-switch count, not a request ID or turn number;
    this routing helper does not consume or authorize a request submission.
    """
    stream_id: object
    owner_side: object
    engine_binding: object
    generation: object
    active_physical: object


def _shape(value, shape, dtype, name):
    if getattr(value, 'shape', None) != shape or getattr(value, 'dtype', None) != jnp.dtype(dtype):
        raise ValueError('exact shape/dtype required: ' + name)


def initialize_order(stream_ids, owner_sides, team_sizes, *, engine_binding):
    """Register constructor order, before initial entries; physical zero starts.

    Host registration is metadata only and is not an engine initialization.
    One (stream, owner) pair identifies one lifetime; no automatic reset exists.
    """
    if getattr(stream_ids, 'ndim', None) != 1 or not stream_ids.shape[0]:
        raise ValueError('nonempty owner registration required')
    batch = stream_ids.shape[0]
    for value, name in ((stream_ids, 'stream'), (owner_sides, 'owner'), (team_sizes, 'team size')):
        _shape(value, (batch,), jnp.int32, name)
    _shape(engine_binding, (batch, 8), jnp.uint32, 'constructor binding')
    ids, owners, sizes = (np.asarray(x) for x in (stream_ids, owner_sides, team_sizes))
    if (np.any(ids <= 0) or np.any((owners < 0) | (owners > 1))
            or np.any((sizes < 1) | (sizes > 6))
            or len(set(zip(ids.tolist(), owners.tolist()))) != batch):
        raise ValueError('unique positive stream/owner and one to six party members required')
    order = jnp.where(jnp.arange(6) < team_sizes[:, None], jnp.arange(6), -1).astype(jnp.int32)
    return Order(stream_ids, owner_sides, engine_binding, team_sizes,
        jnp.zeros(batch, jnp.int32), order, jnp.ones(batch, jnp.bool_))


def _valid(order):
    if type(order) is not Order:
        raise ValueError('registered request order required')
    if getattr(order.owner_side, 'ndim', None) != 1 or not order.owner_side.shape[0]:
        raise ValueError('nonempty owner order required')
    batch = order.owner_side.shape[0]
    for field in ('stream_id', 'owner_side', 'team_size', 'generation'):
        _shape(getattr(order, field), (batch,), jnp.int32, field)
    _shape(order.engine_binding, (batch, 8), jnp.uint32, 'engine binding')
    _shape(order.request_to_physical, (batch, 6), jnp.int32, 'party permutation')
    _shape(order.admitted, (batch,), jnp.bool_, 'order admission')
    positions = jnp.arange(6)
    present = positions < order.team_size[:, None]
    actual = jnp.sort(jnp.where(order.request_to_physical >= 0, order.request_to_physical, 6), axis=-1)
    expected = jnp.where(present, positions, 6)
    return (order.admitted & (order.stream_id > 0) & (order.owner_side >= 0) & (order.owner_side <= 1)
        & (order.team_size >= 1) & (order.team_size <= 6) & (order.generation >= 0)
        & jnp.all((order.request_to_physical >= 0) == present, axis=-1)
        & jnp.all(present | (order.request_to_physical == -1), axis=-1)
        & jnp.all(actual == expected, axis=-1))


def _identity(order, boundary):
    batch = order.owner_side.shape[0]
    for field in ('stream_id', 'owner_side', 'generation'):
        _shape(getattr(boundary, field), (batch,), jnp.int32, field)
    _shape(boundary.engine_binding, (batch, 8), jnp.uint32, 'boundary engine binding')
    return ((boundary.stream_id == order.stream_id) & (boundary.owner_side == order.owner_side)
        & (boundary.generation == order.generation)
        & jnp.all(boundary.engine_binding == order.engine_binding, axis=-1))


def _request_identity(order, boundary):
    if type(boundary) is not RequestBoundary:
        raise ValueError('actual owning request boundary required')
    _shape(boundary.active_physical, order.owner_side.shape, jnp.int32, 'request active physical index')
    return _identity(order, boundary) & (boundary.active_physical == order.request_to_physical[:, 0])


def install_switch(order, receipt):
    """Swap request positions exactly once at an actual switch installation.

    Bad identity, sequence or party rows poison that owner's ledger permanently.
    Uninstalled rows ignore unused physical indices and do not advance sequence.
    """
    ok = _valid(order)
    if type(receipt) is not SwitchInstalled:
        raise ValueError('actual installed-switch receipt required')
    batch = order.owner_side.shape[0]
    for field in ('stream_id', 'owner_side', 'generation', 'outgoing_physical', 'incoming_physical'):
        _shape(getattr(receipt, field), (batch,), jnp.int32, field)
    _shape(receipt.engine_binding, (batch, 8), jnp.uint32, 'receipt engine binding')
    _shape(receipt.installed, (batch,), jnp.bool_, 'installed mask')
    ok &= _identity(order, receipt)
    active = order.request_to_physical[:, 0]
    incoming = receipt.incoming_physical
    valid_switch = ((receipt.outgoing_physical == active) & (incoming >= 0)
        & (incoming < order.team_size) & (incoming != active) & (order.generation < 2**31-1))
    ok &= ~receipt.installed | valid_switch
    position = jnp.argmax(order.request_to_physical == incoming[:, None], axis=-1)
    rows = jnp.arange(batch)
    swapped = order.request_to_physical.at[rows, position].set(active).at[:, 0].set(incoming)
    changed = jnp.where(receipt.installed[:, None], swapped, order.request_to_physical)
    return order._replace(request_to_physical=jnp.where(ok[:, None], changed, -1),
        generation=order.generation + (ok & receipt.installed).astype(jnp.int32), admitted=ok)


class ReorderedTeam(NamedTuple):
    owned: OwnedTeam
    admitted: object


def reorder_owned_team(owned, order, *, boundary):
    """Gather already-selected OWN fields into source request order.

    No opposing state is accepted. Invalid rows have owner=-1 and zero contents,
    so the existing private-team writer refuses even if admission is ignored.
    """
    ok = _valid(order)
    if type(owned) is not OwnedTeam:
        raise ValueError('already-selected owned team required')
    ok &= _request_identity(order, boundary)
    batch = order.owner_side.shape[0]
    for field in OwnedTeam._fields:
        shape = (batch, 8) if field == 'engine_binding' else (batch,) if field in ('owner_side', 'active') else (
            batch, 6, 6) if field == 'stats' else (batch, 6)
        _shape(getattr(owned, field), shape, jnp.uint32 if field == 'engine_binding' else jnp.int32, field)
    ok &= ((owned.owner_side == order.owner_side) & (owned.active == order.request_to_physical[:, 0])
        & jnp.all(owned.engine_binding == order.engine_binding, axis=-1)
        & jnp.all((owned.species > 0) == (jnp.arange(6) < order.team_size[:, None]), axis=-1))
    indices = jnp.clip(order.request_to_physical, 0, 5)
    values = []
    for field in OwnedTeam._fields[2:]:
        if field == 'active':
            values.append(jnp.zeros(batch, jnp.int32))
            continue
        value = getattr(owned, field)
        extra = (1,) if field == 'stats' else ()
        gathered = jnp.take_along_axis(value, indices.reshape((batch, 6) + extra), axis=1)
        visible = ok[:, None] & (jnp.arange(6) < order.team_size[:, None])
        values.append(jnp.where(visible.reshape((batch, 6) + extra), gathered, 0))
    return ReorderedTeam(OwnedTeam(jnp.where(ok, owned.owner_side, -1), owned.engine_binding, *values), ok)


class Choice(NamedTuple):
    engine_action: object
    physical_switch_target: object
    admitted: object


def route_choice(order, choices, *, boundary, requested, request_legal, move_action_map, engine_legal):
    """Route one actual owning request's action to fixed physical engine slots.

    The caller supplies explicit request-move-slot -> engine-action routing.
    No PP/lock/Struggle inference or move-ID matching occurs here. Both legality
    masks are required; waiting, missing-map and invalid choices return -1.
    """
    ok = _valid(order)
    ok &= _request_identity(order, boundary)
    batch = order.owner_side.shape[0]
    for value, shape, dtype, name in ((choices, (batch,), jnp.int32, 'choices'),
            (requested, (batch,), jnp.bool_, 'requested'),
            (request_legal, (batch, 9), jnp.bool_, 'request legal'),
            (move_action_map, (batch, 4), jnp.int32, 'explicit move routing'),
            (engine_legal, (batch, 9), jnp.bool_, 'engine legal')):
        _shape(value, shape, dtype, name)
    safe = jnp.clip(choices, 0, 8)
    rows = jnp.arange(batch)
    moving = choices < 4
    physical = order.request_to_physical[rows, jnp.clip(choices-3, 0, 5)]
    active = order.request_to_physical[:, 0]
    switch_action = 4 + physical - (physical > active).astype(jnp.int32)
    move_action = move_action_map[rows, jnp.clip(choices, 0, 3)]
    engine_action = jnp.where(moving, move_action, switch_action)
    valid_target = (physical >= 0) & (physical < order.team_size) & (physical != active)
    ok &= (requested & (choices >= 0) & (choices < 9) & request_legal[rows, safe]
        & jnp.where(moving, (move_action >= 0) & (move_action < 4), valid_target)
        & (engine_action >= 0) & (engine_action < 9)
        & engine_legal[rows, jnp.clip(engine_action, 0, 8)])
    return Choice(jnp.where(ok, engine_action, -1), jnp.where(ok & ~moving, physical, -1), ok)
