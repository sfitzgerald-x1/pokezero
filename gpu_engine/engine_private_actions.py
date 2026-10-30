"""Bound own-engine inputs to the existing source action-token writer.

Supported requests are explicitly evidenced uncompressed ordinary move rows,
forced replacement and wait packets. UNKNOWN includes compressed locks,
Struggle, recharge, Encore and Transform; none is reconstructed from a chosen
action, current PP, a legal mask, or an opposing physical world.

The actual request producer is still missing. SourceRequest is a caller
attestation, not a certificate that getMoveRequestData ran. MaxPP is retained
separately from the actual immutable constructor pool and selection indices.
generate_teams.cjs retains initial full PP, but BattleState.pp is mutable and
MoveTable/metadata base PP cannot recover source maxpp (including noPPBoosts).
The producer must supply source-exposed disabled/trapped flags, not physical
server legality. PP zero alone does not disable a parsed Python request row.

Source: Showdown f76228a1354b5d0f307ca2d16101294ad3a2308b,
pokemon.ts:getMoves/getMoveRequestData/getSwitchRequestData and
battle.ts:getRequests. No transition, request producer, full observation,
model, rollout or B4 qualification is added by this module.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from pokezero.showdown import _HIDDEN_POWER_TYPES
from . import engine_private_team as team
from . import engine_request_order as order_api
from . import private_actions as actions
from . import private_requests as requests
from . import source_provenance as provenance
from .public_moves import _static_category
from .state import BattleState

SCHEMA = 'pokezero.engine_private_actions.v1'
UNKNOWN, ORDINARY, FORCE_SWITCH, WAIT = range(4)


@dataclass(frozen=True)
class Exposure:
    """Pinned source-producer contract, not a claim that it is integrated."""
    producer_sha256: str
    max_pp_source: str

    def __post_init__(self):
        if (type(self.producer_sha256) is not str or len(self.producer_sha256) != 64
                or any(x not in '0123456789abcdef' for x in self.producer_sha256)
                or self.max_pp_source != 'retained-constructor'):
            raise ValueError('exact source producer hash and explicit maxPP origin required')


class SourceRequest(NamedTuple):
    """One actual owner request and its same-boundary physical dispatch mask.

    complete certifies that no source exposure field was omitted. unchanged
    requires actual own provenance excluding Transform/Mimic/other replaced
    moves/stats, recharge and unrepresented request locks. Absence means false,
    never a default inferred from BattleState. ORDINARY certifies getMoves was
    uncompressed and not replaced by Struggle; rows retain physical move order.
    engine_legal is the owning server mask, never the source request mask.
    """
    boundary: order_api.RequestBoundary
    binding: object
    constructor_binding: object
    request_id: object
    mode: object
    complete: object
    unchanged: object
    disabled: object
    trapped: object
    engine_legal: object


class Tables(NamedTuple):
    binding: object
    engine_binding: object
    primary: object
    mechanics: object
    supported: object
    ghost: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Tables


_COMPILED = weakref.WeakKeyDictionary()
_CONSTRUCTORS = weakref.WeakKeyDictionary()


def _shape(value, shape, dtype, name):
    if getattr(value, 'shape', None) != shape or getattr(value, 'dtype', None) != jnp.dtype(dtype):
        raise ValueError('exact action shape/dtype required: ' + name)


def _immutable_arrays(arrays, name):
    # Equality and object identity do not freeze NumPy or foreign array storage,
    # even when its current writeable flag is false. The existing source
    # compilers produce genuine immutable JAX arrays for every static field.
    if any(not isinstance(value, jax.Array) for value in arrays):
        raise ValueError('immutable JAX static arrays required: ' + name)


class ConstructorData(NamedTuple):
    binding: object
    stream_id: object
    owner_side: object
    engine_binding: object
    species: object
    moves: object
    max_pp: object


@dataclass(frozen=True, eq=False)
class ConstructorPP:
    device: ConstructorData


def retain_constructor_pp(pool, team_indices, *, stream_ids, owner_sides):
    """Retain exact pool.pp using state_from_pool's [B,2] selection schema.

    Call beside actual construction, before mutable state.pp loses maxPP.
    This never initializes/steps an engine or reads the unselected team's data.
    It does not certify that a caller actually used those construction inputs.
    Registration arrays are owned copies before hashing and retention; caller
    mutation can neither change this lifetime nor reuse its original binding.
    """
    receipt = provenance._pool(pool, rehash=True)
    indices = np.asarray(team_indices)
    if (indices.ndim != 2 or indices.shape[1] != 2 or not indices.shape[0]
            or indices.dtype.kind not in 'iu'):
        raise ValueError('same nonempty integer [B,2] constructor selection required')
    b = indices.shape[0]
    _shape(stream_ids, (b,), jnp.int32, 'constructor streams')
    _shape(owner_sides, (b,), jnp.int32, 'constructor owners')
    stream_ids = jnp.array(stream_ids, jnp.int32, copy=True)
    owner_sides = jnp.array(owner_sides, jnp.int32, copy=True)
    streams, owners = np.asarray(stream_ids), np.asarray(owner_sides)
    count = pool.arrays['species'].shape[0]
    if (np.any(indices < 0) or np.any(indices >= count) or np.any(streams <= 0)
            or np.any((owners < 0) | (owners > 1))
            or len(set(zip(streams.tolist(), owners.tolist()))) != b):
        raise ValueError('valid constructor selection and unique stream/owner pairs required')
    selected = indices[np.arange(b), owners]
    values = {}
    for name, tail in (('species', (6,)), ('moves', (6, 4)), ('pp', (6, 4))):
        array = pool.arrays[name]
        if array.shape != (count, *tail):
            raise ValueError('constructor pool shape differs: ' + name)
        # Owner selection precedes content/range checks.
        own = array[selected]
        if np.any(own < 0) or np.any(own > np.iinfo(np.int32).max):
            raise ValueError('own constructor values must fit nonnegative int32')
        values[name] = jnp.asarray(own, jnp.int32)
    if (np.any((np.asarray(values['moves']) == 0) != (np.asarray(values['pp']) == 0))
            or np.any((np.asarray(values['species']) == 0)[..., None] & (np.asarray(values['moves']) != 0))):
        raise ValueError('constructor move/maxPP padding differs')
    engine = team.engine_metadata_binding(pool.metadata())
    words = []
    for index in range(b):
        payload = dict(schema=SCHEMA + '.constructor', pool=receipt['pool_sha256'],
            receipt=pool.receipt_sha256, selection=int(selected[index]), stream=int(streams[index]),
            owner=int(owners[index]), engine=np.asarray(engine).tolist())
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
        words.append(np.frombuffer(digest, dtype='>u4').astype(np.uint32))
    binding = jnp.asarray(np.stack(words))
    device = ConstructorData(binding, stream_ids, owner_sides, jnp.broadcast_to(engine, (b, 8)),
        values['species'], values['moves'], values['pp'])
    retained = ConstructorPP(device)
    _CONSTRUCTORS[retained] = (device, tuple(device))
    return retained


def _constructor(retained):
    record = _CONSTRUCTORS.get(retained) if type(retained) is ConstructorPP else None
    if record is None or retained.device is not record[0] or any(a is not b for a, b in zip(retained.device, record[1])):
        raise ValueError('original retained constructor maxPP holder required')
    _immutable_arrays(retained.device, 'retained constructor')
    return retained.device


def _checked(compiled):
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record['tables']
            or any(a is not b for a, b in zip(compiled.device, record['edges']))
            or record['catalog'].device is not record['action_tables']
            or any(a is not b for a, b in zip(record['catalog'].device, record['action_edges']))):
        raise ValueError('original compiled action holder and immutable catalog edges required')
    _immutable_arrays(compiled.device, 'compiled adapter')
    _immutable_arrays(record['action_tables'], 'original action catalog')
    return record


def compile_engine_actions(metadata, *, catalog, vocabulary, private_team, exposure):
    """Compile source catalog lookup IDs; never use engine IDs as policy rows.

    Retain the original guarded private-team handle and original action catalog.
    Recompile its frozen source snapshot once to reject a catalog whose copied
    binding words conceal replaced tables. Foreign/mutable static array leaves
    are refused before capture. All later calls check array identity and storage.
    """
    if type(catalog) is not actions.ActionCatalog or type(exposure) is not Exposure:
        raise ValueError('original ActionCatalog and explicit Exposure required')
    if type(catalog.device) is not actions.ActionTables:
        raise ValueError('original typed action tables required')
    _immutable_arrays(catalog.device, 'action catalog')
    team_fn = team.prepare_team_facts(private_team, expected_binding=private_team.binding)
    rebuilt = actions.compile_actions(catalog.dex, vocabulary, hp_limit=catalog.device.hp_power.shape[1]-1)
    if (catalog.names != rebuilt.names or dict(catalog.ids) != dict(rebuilt.ids)
            or any(not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(catalog.device, rebuilt.device))):
        raise ValueError('action catalog differs from its frozen source compilation')
    engine = team.engine_metadata_binding(metadata)
    if (not np.array_equal(np.asarray(engine), np.asarray(private_team.engine_binding))
            or not np.array_equal(np.asarray(catalog.device.vocabulary_binding), np.asarray(private_team.vocabulary_binding))):
        raise ValueError('constructor and action/private-team vocabulary bindings differ')
    def names(kind):
        values = metadata['vocabulary'][kind]
        if (type(values) is not list or not values or values[0] not in ('', '_none')
                or any(type(x) is not str or not normalize_id(x) for x in values[1:])
                or len(set(normalize_id(x) for x in values)) != len(values)):
            raise ValueError('exact ordered engine namespace required: ' + kind)
        return values
    moves, types, species = (names(k) for k in ('moves', 'types', 'species'))
    primary, mechanics, supported = [], [], []
    for index, raw in enumerate(moves):
        name = normalize_id(raw) if index else ''
        hp_aliases = {'hiddenpower' + kind for kind in _HIDDEN_POWER_TYPES}
        identity = 'hiddenpower' if name == 'hiddenpower' or name in hp_aliases else name
        label = 'move:' + identity
        primary.append(_static_category(vocabulary, label) if identity else 0)
        mr, sr = [], []
        for kind in types:
            lookup = 'hiddenpower' + normalize_id(kind) if identity == 'hiddenpower' else name
            mid = catalog.ids.get(lookup, 0)
            mr.append(mid)
            sr.append(bool(index and identity not in ('struggle', 'recharge') and mid and vocabulary.is_enumerated(label)
                and (identity != 'hiddenpower' or normalize_id(kind) in _HIDDEN_POWER_TYPES)
                and catalog.dex.move_info(lookup) is not None))
        mechanics.append(mr); supported.append(sr)
    ghosts = []
    for index, name in enumerate(species):
        info = catalog.dex.species_info(name) if index else None
        ghosts.append(bool(info and any(str(t).lower() == 'ghost' for t in info.types)))
        # Both writers must resolve the same static source species fields.
        if info is not None:
            values = [info.base_stats.get(k, 0) for k in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')]
            if not np.array_equal(np.asarray(private_team.base_stats[index]), np.asarray(values, np.float32)):
                raise ValueError('action and private-team source species snapshots differ')
            kinds = [_static_category(vocabulary, 'type:' + info.types[i]) if i < len(info.types) else 0 for i in (0, 1)]
            if not np.array_equal(np.asarray(private_team.type_categories[index]), np.asarray(kinds, np.int32)):
                raise ValueError('action and private-team source species types differ')
    payload = dict(schema=SCHEMA, engine=np.asarray(engine).tolist(), exposure=asdict(exposure),
        actions=np.asarray(catalog.device.binding).tolist(), team=np.asarray(private_team.binding).tolist(),
        primary=primary, mechanics=mechanics, supported=supported, ghost=ghosts)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32))
    tables = Tables(binding, engine, jnp.asarray(primary, jnp.int32), jnp.asarray(mechanics, jnp.int32),
        jnp.asarray(supported, jnp.bool_), jnp.asarray(ghosts, jnp.bool_))
    compiled = Compiled(tables)
    _COMPILED[compiled] = dict(tables=tables, edges=tuple(tables), catalog=catalog,
        action_tables=catalog.device, action_edges=tuple(catalog.device), team=private_team,
        team_fn=team_fn)
    return compiled


class Selected(NamedTuple):
    moves: object
    pp: object
    hidden_power_type: object
    charge_move: object
    encore_move: object
    encore_counter: object


def _select(state, owners):
    """Read owner storage first; no opposing contents or legality derivation."""
    if type(state) is not BattleState:
        raise ValueError('actual engine BattleState required')
    b = owners.shape[0]
    _shape(state.active, (b, 2), jnp.int32, 'actual active')
    rows, seats = jnp.arange(b), jnp.clip(owners, 0, 1)
    active = state.active[rows, seats]
    physical = jnp.clip(active, 0, 5)
    selected = []
    for name in Selected._fields:
        value = getattr(state, name)
        tail = (6, 4) if name in ('moves', 'pp') else (6,)
        _shape(value, (b, 2, *tail), jnp.int32, name)
        owned = value[rows, seats]
        selected.append(owned[rows, physical])
    return Selected(*selected)


class Result(NamedTuple):
    facts: actions.ActionFacts
    features: actions.ActionFeatures
    move_action_map: object
    engine_legal: object
    boundary: order_api.RequestBoundary
    admitted: object


def _device(selected, owned, facts, ledger, source, constructor, tables, action_tables, pre_admitted):
    b = ledger.owner_side.shape[0]
    if type(source) is not SourceRequest:
        raise ValueError('actual source request exposure required')
    for field in ('request_id', 'mode'):
        _shape(getattr(source, field), (b,), jnp.int32, field)
    for field in ('complete', 'unchanged', 'trapped'):
        _shape(getattr(source, field), (b,), jnp.bool_, field)
    _shape(source.binding, (b, 8), jnp.uint32, 'request exposure binding')
    _shape(source.constructor_binding, (b, 8), jnp.uint32, 'actual owning constructor binding')
    _shape(source.disabled, (b, 4), jnp.bool_, 'source disabled rows')
    _shape(source.engine_legal, (b, 9), jnp.bool_, 'own server legal mask')
    for field in ('stream_id', 'owner_side'):
        _shape(getattr(constructor, field), (b,), jnp.int32, 'retained ' + field)
    _shape(constructor.engine_binding, (b, 8), jnp.uint32, 'retained namespace')
    _shape(constructor.binding, (b, 8), jnp.uint32, 'retained constructor binding')
    _shape(constructor.species, (b, 6), jnp.int32, 'retained species')
    _shape(constructor.moves, (b, 6, 4), jnp.int32, 'retained moves')
    _shape(constructor.max_pp, (b, 6, 4), jnp.int32, 'retained maxPP')
    rows = jnp.arange(b)
    active = jnp.clip(owned.active, 0, 5)
    max_pp = constructor.max_pp[rows, active]
    mode = source.mode
    ordinary = mode == ORDINARY
    forced = mode == FORCE_SWITCH
    waiting = mode == WAIT
    present = ordinary[:, None] & (selected.moves > 0)
    safe_moves = jnp.clip(selected.moves, 0, tables.primary.shape[0]-1)
    safe_type = jnp.clip(selected.hidden_power_type, 0, tables.mechanics.shape[1]-1)
    mechanics = tables.mechanics[safe_moves, safe_type[:, None]]
    supported = tables.supported[safe_moves, safe_type[:, None]]
    ok = pre_admitted & source.complete & source.unchanged
    ok &= jnp.all(source.constructor_binding == constructor.binding, axis=-1)
    ok &= (constructor.stream_id == ledger.stream_id) & (constructor.owner_side == ledger.owner_side)
    ok &= jnp.all(constructor.engine_binding == ledger.engine_binding, axis=-1)
    ok &= jnp.all(constructor.species == owned.species, axis=-1)
    ok &= jnp.all(constructor.moves[rows, active] == selected.moves, axis=-1)
    ok &= jnp.all(source.binding == tables.binding, axis=-1)
    ok &= (source.request_id >= -1) & (ordinary | forced | waiting)
    ok &= ~ordinary | ((selected.charge_move == 0) & (selected.encore_move == 0)
        & (selected.encore_counter == 0) & jnp.any(present & ~source.disabled, axis=-1))
    ok &= jnp.all(~present | ((selected.moves < tables.primary.shape[0]) & supported
        & (selected.pp >= 0) & (max_pp > 0) & (selected.pp <= max_pp)), axis=-1)
    needs_type = jnp.any(present & ~tables.supported[safe_moves, 0], axis=-1)
    ok &= ~needs_type | ((selected.hidden_power_type > 0)
        & (selected.hidden_power_type < tables.mechanics.shape[1]))
    ok &= ~ordinary | ~jnp.any((selected.moves[:, 1:] > 0) & (selected.moves[:, :-1] <= 0), axis=-1)
    ok &= ~ordinary | jnp.all(selected.moves >= 0, axis=-1)
    owner = jnp.where(ok, ledger.owner_side, -1)
    team_facts = facts._replace(owner_side=owner)
    kind = jnp.where(ordinary, requests.RequestKind.MOVE,
        jnp.where(forced, requests.RequestKind.FORCE_SWITCH, requests.RequestKind.WAIT)).astype(jnp.int32)
    packet = requests.ActionRequest(owner, ok, source.request_id, kind, waiting, forced,
        ordinary & source.trapped, ledger.team_size, facts.active, facts.fainted, present, source.disabled & present,
        jnp.where(present, selected.pp, 0).astype(jnp.float32),
        jnp.where(present, max_pp, 0).astype(jnp.float32), present)
    primary = jnp.where(present, tables.primary[safe_moves], action_tables.fallback_primary[:4])
    mechanics = jnp.where(present, mechanics, 0)
    # TeamFacts now follows request order, so actual active is request row zero.
    actor_num = facts.hp_numerator[:, 0]
    actor_den = facts.hp_denominator[:, 0]
    exact = ((actor_den > 0) & (actor_den < action_tables.hp_power.shape[1])
        & (actor_num >= 0) & (actor_num <= actor_den))
    return packet, team_facts, primary, mechanics, exact, actor_num, actor_den, ok


def _sanitize(facts, admitted):
    """Refused packets cannot leak carried facts if admission is ignored."""
    def mask(value):
        return jnp.where(admitted.reshape(admitted.shape + (1,) * (value.ndim-1)), value, jnp.zeros_like(value))
    safe = jax.tree.map(mask, facts)
    return safe._replace(request=safe.request._replace(owner_side=jnp.where(admitted, facts.request.owner_side, -1),
        request_id=jnp.where(admitted, facts.request.request_id, -1)),
        team=safe.team._replace(owner_side=jnp.where(admitted, facts.team.owner_side, -1)))


def prepare_engine_actions(compiled, *, expected_binding):
    """Guard on EVERY host call, then select, reorder and run existing writers.

    The returned host boundary must not be wrapped in an outer JIT. Its device
    kernels are already jitted; tracing would cache away holder guards.
    """
    record = _checked(compiled)
    tables, catalog = record['tables'], record['catalog']
    _shape(expected_binding, (8,), jnp.uint32, 'expected engine action binding')
    if not np.array_equal(np.asarray(expected_binding), np.asarray(tables.binding)):
        raise ValueError('pinned engine action binding required')
    gather = jax.jit(order_api.reorder_owned_team)
    assemble = jax.jit(_device)
    project = jax.jit(actions.project_actions)
    sanitize = jax.jit(_sanitize)
    def checked(state, ledger, source, constructor, *, engine_binding):
        _checked(compiled)
        retained = _constructor(constructor)
        if isinstance(jnp.zeros((), jnp.int32), jax.core.Tracer):
            raise ValueError('checked engine action host boundary cannot be traced')
        if type(source) is not SourceRequest:
            raise ValueError('actual source request exposure required')
        owned = team.select_owned_team(state, ledger.owner_side, engine_binding=engine_binding)
        reordered = gather(owned, ledger, boundary=source.boundary)
        team_result = record['team_fn'](reordered.owned)
        selected = _select(state, ledger.owner_side)
        packet, tf, primary, mechanics, exact, num, den, ok = assemble(selected, owned, team_result.facts,
            ledger, source, retained, tables, catalog.device, reordered.admitted & team_result.admitted)
        rows = jnp.arange(ledger.owner_side.shape[0])
        species = owned.species[rows, jnp.clip(owned.active, 0, 5)]
        ghost = tables.ghost[jnp.clip(species, 0, tables.ghost.shape[0]-1)]
        af = actions.ActionFacts(packet, tf, jnp.broadcast_to(catalog.device.binding, source.binding.shape),
            primary, mechanics, ghost, ok, exact & ok, num.astype(jnp.int32), den.astype(jnp.int32))
        features = project(af, catalog.device, observer_side=ledger.owner_side)
        admitted = ok & features.admitted
        mapping = jnp.where(admitted[:, None] & packet.move_present,
            jnp.arange(4, dtype=jnp.int32), -1)
        return Result(sanitize(af, admitted), features, mapping,
            source.engine_legal & admitted[:, None], source.boundary, admitted)
    return checked


def route_choice(result, ledger, choices):
    """Feed the SAME packet, explicit map and server mask to the merged router."""
    if type(result) is not Result:
        raise ValueError('bound engine action result required')
    ready = requests.action_inputs(result.facts.request, observer_side=ledger.owner_side).ready
    return order_api.route_choice(ledger, choices, boundary=result.boundary,
        requested=result.admitted & ready, request_legal=result.features.legal,
        move_action_map=result.move_action_map, engine_legal=result.engine_legal)
