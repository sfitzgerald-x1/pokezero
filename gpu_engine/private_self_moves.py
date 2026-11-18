"""V2 self_moves for B2: the owner's own moves and PP on its self tokens.

Off by default (v4_encoder.Masks.self_moves): with the mask off nothing here is compiled,
parsed or traced, so B2's encoding is unchanged. On, every own token (request order, every
listed mon: bench, active and fainted) carries its moves as a FULLY revealed opponent mon's
are written (showdown._encode_pokemon_tokens' self branch and _encode_self_move_pp_fractions
are the reference). The moves and PP are pokezero.own_moves.own_move_states over the owner's
protocol history (the same function the Python encoder reads, a per-mon port of the
constructor's own_pp_carry), parsed on host with the request:

- categorical CATEGORY_BELIEF_MOVE_OFFSET + k: belief:possible_move:<label> of the REAL moveset
  (the battle-start request's; never Transform's copies; Return/Frustration without the
  request's resolved-power suffix, Hidden Power typed), in lexical order (this catalog's move
  namespace is lexically ordered, so ascending IDs), padding 0;
- numeric REVEALED_MOVE_COUNT = POSSIBLE_MOVE_COUNT = the number of own moves (0 when unknown);
- under exact_state, per bucket, when the mon's PP is exact: the validity bit 1
  (NUMERIC_OPP_MOVE_PP_VALID_OFFSET + k) and remaining / catalog max PP (NUMERIC_OPP_MOVE_PP_OFFSET
  + k); when the history cannot state it (Spite, Grudge, Leppa Berry, an unknown called-move
  cost, ...) both stay 0.
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

SCHEMA = 'pokezero.private_self_moves.v2'
SLOTS = 4
BUCKETS = reference.BELIEF_MOVE_BUCKET_COUNT
COUNT_COLUMNS = tuple(reference.v4_numeric_index(x) for x in
    (reference.NUMERIC_REVEALED_MOVE_COUNT, reference.NUMERIC_POSSIBLE_MOVE_COUNT))
PP_COLUMNS = tuple(reference.v4_numeric_index(reference.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(BUCKETS))
VALID_COLUMNS = tuple(reference.v4_numeric_index(reference.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(BUCKETS))
NUMERIC_COLUMNS = COUNT_COLUMNS + PP_COLUMNS + VALID_COLUMNS
CATEGORY_COLUMNS = tuple(range(reference.CATEGORY_BELIEF_MOVE_OFFSET, reference.CATEGORY_BELIEF_MOVE_OFFSET + BUCKETS))


SEARCH_FOLLOW_UP = ('self_moves search serving is not threaded yet: the search stack captures the Python root under '
    'ActorContract.champion() (phase_d_protocol_constructor, depthn_live_search, d3_depthn_gate, owning_python_reference, '
    'mc_ladder_captures); the follow-up threads the served contract there')


def refuse_search_serving(*, catalog=None, meta=None, context):
    """Search serving fails closed on self_moves until the follow-up lands: a served catalog compiled with it, or a served
    archive declaring it (meta['self_moves']), stops the server at build instead of pricing an on catalog under an off
    Python root capture."""
    if bool(getattr(getattr(catalog, 'masks', None), 'self_moves', False)) or bool((meta or {}).get('self_moves', False)):
        raise ValueError(f'{context}: {SEARCH_FOLLOW_UP}')


UNTHREADED = ('self_moves is not threaded through this tool yet: it encodes with the champion contract (self_moves off), so '
    'an archive that declares self_moves would be played or scored on observations it was not trained on')


def refuse_unthreaded(meta=None, *, plan=None, catalog=None, context):
    """Off-only tools reject on archives, plans or catalogs before evaluating actors."""
    if (bool((meta or {}).get('self_moves', False)) or bool(getattr(plan, 'self_moves', False))
            or bool(getattr(getattr(catalog, 'masks', None), 'self_moves', False))):
        raise ValueError(f'{context}: {UNTHREADED}')


def check_run_actor(meta, *, plan_self_moves, iteration, context):
    """A V2 run's actor archive against its plan's declaration (meta['self_moves'], absent = off). An on archive in an off
    run is refused. In an on run every archive the run exports is stamped on; only the initial snapshot (iteration -1,
    the checkpoint the run starts from, e.g. a shared off init) may be off: the plan's declaration wins and its tables
    encode that actor on."""
    declared = bool((meta or {}).get('self_moves', False))
    if declared == bool(plan_self_moves) or plan_self_moves and iteration == -1:
        return
    raise ValueError(f'{context}: actor archive self_moves={declared} differs from the plan self_moves={bool(plan_self_moves)}')


class Tables(NamedTuple):
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
    payload = dict(schema=SCHEMA, slots=SLOTS, numeric=NUMERIC_COLUMNS, categories=CATEGORY_COLUMNS,
        moves=np.asarray(moves.device.binding).tolist(), features=np.asarray(features.binding).tolist(),
        vocabulary=np.asarray(vb).tolist())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    return Catalog(moves, Tables(moves.device.max_pp, features.move_category,
        moves.device.binding, vb, jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32))))


class OwnMoves(NamedTuple):
    owner_side: jax.Array
    move_binding: jax.Array         # [8]
    present: jax.Array              # [6] request order
    moves: jax.Array                # [6, SLOTS] move IDs, ascending (lexical), padding 0
    pp: jax.Array                   # [6, SLOTS] exact remaining PP aligned with moves (0 where unknown)
    pp_known: jax.Array             # [6] bool: the mon's PP is exact


def parse_own_moves(request, *, owner_side, catalog, lines):
    """The owner's real moves and exact PP per request slot (pokezero.own_moves.own_move_states over `lines`, the
    owner's protocol history up to and including `request`); never the opponent's request."""
    from pokezero.own_moves import own_move_states
    owner = parse_action_request(request, owner_side=owner_side).owner_side
    team = reference._self_team_from_request(request, f'p{owner_side + 1}')
    maximum = np.asarray(catalog.device.max_pp)
    states = own_move_states(tuple(lines), request,
        maximum_pp=lambda key: int(maximum[catalog.moves.ids[key]]) if key in catalog.moves.ids else 0)
    by_identity = {state.ident: state for state in states}
    present = np.zeros(6, bool); moves = np.zeros((6, SLOTS), np.int32); pp = np.zeros((6, SLOTS), np.int32)
    known = np.zeros(6, bool)
    for i, mon in enumerate(team):
        if mon.showdown_slot != f'p{owner_side + 1}':raise ValueError('own move row belongs to another seat')
        present[i] = True
        state = by_identity.get(str(mon.ident).split(':', 1)[-1].strip().casefold())
        if state is None:raise ValueError('own move state missing for a request row')
        rows = sorted({catalog.moves.move_id(label): (state.pp[k] if state.pp is not None else 0)
            for k, label in enumerate(state.moves)}.items())
        if len(rows) > SLOTS:raise ValueError('own request row lists more than four moves')
        for k, (move, value) in enumerate(rows):
            moves[i, k], pp[i, k] = move, value
        known[i] = state.pp is not None
    return OwnMoves(owner, catalog.device.move_binding, jnp.asarray(present), jnp.asarray(moves), jnp.asarray(pp),
        jnp.asarray(known))


def project(own_moves, mons, own_status, tables, *, observer_side, exact_state=True):
    """Own-section patch in request order; refuses (admitted False) on any namespace or shape mismatch."""
    if not isinstance(own_moves, OwnMoves):raise ValueError('owner-bound request move facts required')
    b = mons.encounter.shape[0]
    def packet(x, shape, dtype):
        if getattr(x, 'shape', None) != shape or getattr(x, 'dtype', None) != jnp.dtype(dtype):
            raise ValueError('self move packet shape/dtype mismatch')
        return x
    owner = packet(own_moves.owner_side, (b,), jnp.int32)
    present = packet(own_moves.present, (b, 6), jnp.bool_)
    moves = packet(own_moves.moves, (b, 6, SLOTS), jnp.int32)
    pp = packet(own_moves.pp, (b, 6, SLOTS), jnp.int32)
    known = packet(own_moves.pp_known, (b, 6), jnp.bool_)
    packet(own_moves.move_binding, (b, 8), jnp.uint32)
    observer = packet(observer_side, (b,), jnp.int32)
    exact = jnp.broadcast_to(jnp.asarray(exact_state, bool), (b,))
    count = tables.max_pp.shape[0]
    ok = (owner == observer) & (observer >= 0) & (observer < 2)
    ok &= jnp.all(own_moves.move_binding == tables.move_binding, axis=-1)
    ok &= jnp.all((moves >= 0) & (moves < count) & (present[..., None] | (moves == 0)), axis=(1, 2))
    ok &= jnp.all(present == own_status.present, axis=-1) & (own_status.owner_side == observer)
    safe = jnp.clip(moves, 0, count - 1); have = moves > 0
    maximum = tables.max_pp[safe]
    valid = have & known[..., None]
    fraction = jnp.where(valid & (maximum > 0), jnp.clip(jnp.maximum(0, pp) / jnp.maximum(maximum, 1), 0, 1), 0)
    number = have.sum(axis=-1).astype(jnp.float32)
    pad = jnp.zeros((b, 6, BUCKETS - SLOTS), jnp.float32)
    on = exact[:, None, None]
    values = jnp.concatenate((number[..., None], number[..., None],
        jnp.where(on, fraction, 0).astype(jnp.float32), pad,
        jnp.where(on & valid, 1., 0.).astype(jnp.float32), pad), axis=-1)
    cats = jnp.concatenate((jnp.where(have, tables.category[safe], 0),
        jnp.zeros((b, 6, BUCKETS - SLOTS), jnp.int32)), axis=-1)
    visible = present & ok[:, None]
    numeric = jnp.zeros((b, 6, 132), jnp.float32).at[..., jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(visible[..., None], values, 0))
    categorical = jnp.zeros((b, 6, 41), jnp.int32).at[..., jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(visible[..., None], cats, 0))
    return TeamFeaturePatch(numeric, categorical, jnp.zeros(132, bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41, bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True), ok)
