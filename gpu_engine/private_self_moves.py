"""V2 self_moves for B2: the owner's own moves and PP on its self tokens.

Off by default (v4_encoder.Masks.self_moves): with the mask off nothing here is compiled,
parsed or traced, so B2's encoding is unchanged. On, every own token (request order, every
listed mon: bench, active and fainted) carries its moves as a FULLY revealed opponent mon's
are written (showdown._encode_pokemon_tokens' self branch and _encode_self_move_pp_fractions
are the reference):

- categorical CATEGORY_BELIEF_MOVE_OFFSET + k: belief:possible_move:<id> of the request's
  moves (showdown._self_move_id: Return/Frustration without the request's resolved-power suffix,
  Hidden Power typed, as the request names it), in lexical order (this catalog's move
  namespace is lexically ordered, so ascending IDs), padding 0;
- numeric REVEALED_MOVE_COUNT = POSSIBLE_MOVE_COUNT = the number of own moves;
- under exact_state, per bucket: the PP validity bit 1 (NUMERIC_OPP_MOVE_PP_VALID_OFFSET + k)
  and the remaining fraction (NUMERIC_OPP_MOVE_PP_OFFSET + k) = (catalog max PP - uses) /
  max PP, where uses is the owner fold's own public PP ledger (public_moves.MoveState.uses on
  the owner's seat at the mon's public encounter, joined by exact public species as
  public_exact joins sleep history): the opponent ledger's charging rules. Typed Hidden Power
  reads the generic Hidden Power entry the protocol line charged. A mon with no public
  encounter (never sent out) has used nothing.

The move list is request-private and parsed on host (parse_own_moves); PP use is the fold's.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from .private_requests import parse_action_request
from .private_team import TeamFeaturePatch, vocabulary_binding

SCHEMA = 'pokezero.private_self_moves.v1'
SLOTS = 4
BUCKETS = reference.BELIEF_MOVE_BUCKET_COUNT
COUNT_COLUMNS = tuple(reference.v4_numeric_index(x) for x in
    (reference.NUMERIC_REVEALED_MOVE_COUNT, reference.NUMERIC_POSSIBLE_MOVE_COUNT))
PP_COLUMNS = tuple(reference.v4_numeric_index(reference.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(BUCKETS))
VALID_COLUMNS = tuple(reference.v4_numeric_index(reference.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(BUCKETS))
NUMERIC_COLUMNS = COUNT_COLUMNS + PP_COLUMNS + VALID_COLUMNS
CATEGORY_COLUMNS = tuple(range(reference.CATEGORY_BELIEF_MOVE_OFFSET, reference.CATEGORY_BELIEF_MOVE_OFFSET + BUCKETS))


class Tables(NamedTuple):
    ledger: jax.Array               # [M] move ID -> the PP-ledger ID charged for it (typed Hidden Power -> generic)
    max_pp: jax.Array               # [M] public_moves catalog rule floor(base PP * 8 / 5)
    category: jax.Array             # [M] vocabulary row of belief:possible_move:<name>
    move_binding: jax.Array         # [8] public_moves catalog contract
    vocabulary_binding: jax.Array   # [8]
    binding: jax.Array              # [8] this writer's contract


@dataclass(frozen=True)
class Catalog:
    moves: object                   # public_moves.MoveCatalog (host ID lookup)
    device: Tables


def compile_self_moves(moves, features, vocabulary):
    """Join the existing lexical move namespace, its max-PP rule and the belief move categories."""
    vb = vocabulary_binding(vocabulary)
    if (not np.array_equal(np.asarray(features.move_tables.binding), np.asarray(moves.device.binding))
            or not np.array_equal(np.asarray(features.vocabulary_binding), np.asarray(vb))):
        raise ValueError('self moves require the belief writer\'s move catalog and vocabulary')
    if moves.names != ('',) + tuple(sorted(moves.names[1:])):
        raise ValueError('self move buckets require the lexically ordered move namespace')
    generic = moves.ids.get('hiddenpower', 0)
    ledger = [0] + [generic if generic and name.startswith('hiddenpower') else i
        for i, name in enumerate(moves.names) if i]
    payload = dict(schema=SCHEMA, ledger=ledger, slots=SLOTS, numeric=NUMERIC_COLUMNS, categories=CATEGORY_COLUMNS,
        moves=np.asarray(moves.device.binding).tolist(), features=np.asarray(features.binding).tolist(),
        vocabulary=np.asarray(vb).tolist())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    return Catalog(moves, Tables(jnp.asarray(ledger, jnp.int32), moves.device.max_pp, features.move_category,
        moves.device.binding, vb, jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32))))


class OwnMoves(NamedTuple):
    owner_side: jax.Array
    move_binding: jax.Array         # [8]
    present: jax.Array              # [6] request order
    moves: jax.Array                # [6, SLOTS] move IDs, ascending (lexical), padding 0


def parse_own_moves(request, *, owner_side, catalog):
    """The owner's request moves per request slot; never the opponent's request."""
    owner = parse_action_request(request, owner_side=owner_side).owner_side
    team = reference._self_team_from_request(request, f'p{owner_side + 1}')
    present = np.zeros(6, bool); moves = np.zeros((6, SLOTS), np.int32)
    for i, mon in enumerate(team):
        if mon.showdown_slot != f'p{owner_side + 1}':raise ValueError('own move row belongs to another seat')
        present[i] = True
        # The Python writer's list: request IDs as the moves they name (return102 -> return), deduplicated, sorted.
        ids = sorted({catalog.moves.move_id(reference._self_move_id(name)) for name in mon.moves if reference._self_move_id(name)})
        if len(ids) > SLOTS:raise ValueError('own request row lists more than four moves')
        moves[i, :len(ids)] = ids
    return OwnMoves(owner, catalog.device.move_binding, jnp.asarray(present), jnp.asarray(moves))


def project(own_moves, moves_state, mons, own_status, tables, *, observer_side, exact_state=True):
    """Own-section patch in request order; refuses (admitted False) on any namespace or join mismatch."""
    if not isinstance(own_moves, OwnMoves):raise ValueError('owner-bound request move facts required')
    b = mons.encounter.shape[0]; rows = jnp.arange(b)
    def packet(x, shape, dtype):
        if getattr(x, 'shape', None) != shape or getattr(x, 'dtype', None) != jnp.dtype(dtype):
            raise ValueError('self move packet shape/dtype mismatch')
        return x
    owner = packet(own_moves.owner_side, (b,), jnp.int32)
    present = packet(own_moves.present, (b, 6), jnp.bool_)
    moves = packet(own_moves.moves, (b, 6, SLOTS), jnp.int32)
    packet(own_moves.move_binding, (b, 8), jnp.uint32)
    observer = packet(observer_side, (b,), jnp.int32); seat = jnp.clip(observer, 0, 1)
    exact = jnp.broadcast_to(jnp.asarray(exact_state, bool), (b,))
    count = tables.max_pp.shape[0]
    ok = (owner == observer) & (observer >= 0) & (observer < 2)
    ok &= jnp.all(own_moves.move_binding == tables.move_binding, axis=-1) & jnp.all(moves_state.binding == tables.move_binding, axis=-1)
    ok &= jnp.all((moves >= 0) & (moves < count) & (present[..., None] | (moves == 0)), axis=(1, 2))
    ok &= jnp.all(present == own_status.present, axis=-1) & (own_status.owner_side == observer)
    ok &= jnp.all(own_status.species_binding == mons.species_binding, axis=-1)
    # The fold's ledger for this mon: its public encounter on the owner's seat, by exact public species.
    match = (own_status.species[:, :, None] == mons.species[rows, seat][:, None, :]) & (
        mons.encounter[rows, seat][:, None, :] > 0) & present[..., None]
    ok &= jnp.all(match.sum(axis=-1) <= 1, axis=-1)
    slot = jnp.argmax(match, axis=-1); seen = jnp.any(match, axis=-1)
    ledger_ids = moves_state.move_ids[rows[:, None], seat[:, None], slot]        # [B, 6, capacity]
    ledger_uses = moves_state.uses[rows[:, None], seat[:, None], slot]
    safe = jnp.clip(moves, 0, count - 1); have = moves > 0
    key = tables.ledger[safe]
    hit = (ledger_ids[:, :, None, :] == key[..., None]) & (key[..., None] > 0) & seen[:, :, None, None]
    uses = jnp.sum(jnp.where(hit, ledger_uses[:, :, None, :], 0), axis=-1)
    maximum = tables.max_pp[safe]
    fraction = jnp.where(have & (maximum > 0), jnp.maximum(0, maximum - uses) / jnp.maximum(maximum, 1), 0)
    number = have.sum(axis=-1).astype(jnp.float32)
    pad = jnp.zeros((b, 6, BUCKETS - SLOTS), jnp.float32)
    on = exact[:, None, None]
    values = jnp.concatenate((number[..., None], number[..., None],
        jnp.where(on, fraction, 0).astype(jnp.float32), pad,
        jnp.where(on & have, 1., 0.).astype(jnp.float32), pad), axis=-1)
    cats = jnp.concatenate((jnp.where(have, tables.category[safe], 0),
        jnp.zeros((b, 6, BUCKETS - SLOTS), jnp.int32)), axis=-1)
    visible = present & ok[:, None]
    numeric = jnp.zeros((b, 6, 132), jnp.float32).at[..., jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(visible[..., None], values, 0))
    categorical = jnp.zeros((b, 6, 41), jnp.int32).at[..., jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(visible[..., None], cats, 0))
    return TeamFeaturePatch(numeric, categorical, jnp.zeros(132, bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41, bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True), ok)
