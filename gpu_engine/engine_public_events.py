"""Partial public receipts from actual transition callbacks, never final-state diff.

The INTERNAL device tape carries physical addresses/exact HP and MUST NOT enter
a policy or public fold. project() emits only already-public identities and
owner-visible HP. It implements MOVE, direct main-hit CONDITION and direct-hit
FAINT component facts; no trace has complete raw-line consumers or observation
admission. Entry, prevention, proc, status, recoil/residual, terminal and boundary
callbacks remain uninstrumented. Overflow refuses the entire projection.

Declaration admission uses the exact canonical target enum and existing source
dynamic-effect resolver, never a selected move roster. Unsupported engine rows
remain untrimmed but cannot manufacture target/called/locked lexical provenance.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id, resolve_move_effect
from .engine_private_team import engine_metadata_binding
from .public_context_protocol import PublicAddress
from .public_inputs import PublicEvent, Kind, Phase

SCHEMA = 'pokezero.engine_public_callback_subset.v1'
SHOWDOWN = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'
MOVE, MAIN_HP, DIRECT_FAINT = range(1, 4)
UNINSTRUMENTED, OVERFLOW, INVALID_ACTION = 1, 2, 4
CALLBACK_WORDS = 10


@dataclass(frozen=True)
class HealthExposure:
    """Explicit source visibility facts, never inferred from physical HP.

    This cut supports only the pinned Gen3 Random Battle HP Percentage Mod.
    Pixel/exact-shared formats require another reviewed producer surface.
    """
    showdown_revision: str
    format_id: str
    report_percentages: bool
    report_exact_hp: bool


class CallbackTape(NamedTuple):
    """INTERNAL [B,C,10] exact callback facts, not a public event transport."""
    words: jax.Array
    count: jax.Array
    refusal: jax.Array
    turn: jax.Array
    engine_binding: jax.Array
    source_binding: jax.Array
    move_binding: jax.Array
    producer_binding: jax.Array


class TracedStep(NamedTuple):
    step: object
    callbacks: CallbackTape


def trace_step(state, actions, table, *, catalog, expected_binding, expected_engine_binding,
               chance_inputs=None, capacity=64):
    """Opt-in SAME transition core; ordinary StepResult APIs stay unchanged.

    Fresh bounded receipts per invocation; no whole-game runner or observation
    substitute. The supplied finite chance API permits small component tests.
    """
    from .transition import step_with_chance, sample_chance_inputs
    check_catalog(catalog,table,expected_binding=expected_binding,
        expected_engine_binding=expected_engine_binding)
    chance_inputs = sample_chance_inputs(state) if chance_inputs is None else chance_inputs
    return step_with_chance(state, actions, table, chance_inputs,
        _public_trace=initialize(state.hp.shape[0],capacity,catalog=catalog))


def initialize(batch, capacity=64, *, catalog):
    if type(batch) is not int or batch <= 0 or type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError('bounded positive callback tape shape required')
    check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32))
    producer=np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)
    return CallbackTape(jnp.zeros((batch, capacity, CALLBACK_WORDS), jnp.int32),
        jnp.zeros((batch,), jnp.int32), jnp.zeros((batch,), jnp.int32), jnp.zeros((batch,), jnp.int32),
        *[jnp.asarray(value,jnp.uint32) for value in (catalog.engine_binding,catalog.source_binding,catalog.move_binding,producer)])


def _append(tape, kind, actor_side, actor_slot, target_side, target_slot, values, eligible):
    """Trusted device callback primitive; never a caller's state-diff adapter."""
    rows = jnp.arange(tape.count.shape[0])
    columns = [jnp.full_like(tape.count, kind), actor_side, actor_slot, target_side, target_slot, *values]
    value = jnp.stack(columns, axis=-1)
    room = tape.count < tape.words.shape[1]
    index = jnp.minimum(tape.count, tape.words.shape[1]-1)
    previous = tape.words[rows, index]
    words = tape.words.at[rows, index].set(jnp.where((eligible & room)[:, None], value, previous))
    return tape._replace(words=words, count=tape.count+(eligible & room).astype(jnp.int32),
        refusal=tape.refusal | jnp.where(eligible & ~room, OVERFLOW, 0))


def move(tape, side, actor_slot, target_slot, move_id, *, eligible, unsupported=False):
    z = jnp.zeros_like(tape.count)
    flags = jnp.broadcast_to(jnp.asarray(unsupported,jnp.int32),z.shape)
    return _append(tape, MOVE, side, actor_slot, 1-side, target_slot, (move_id, flags, z, z, z), eligible)


def condition(tape, side, slot, hp, maximum, status, *, eligible):
    absent = jnp.full_like(tape.count, -1)
    z = jnp.zeros_like(tape.count)
    return _append(tape, MAIN_HP, absent, absent, side, slot, (hp, maximum, status, z, z), eligible)


def faint(tape, side, slot, *, eligible):
    absent = jnp.full_like(tape.count, -1)
    z = jnp.zeros_like(tape.count)
    return _append(tape, DIRECT_FAINT, absent, absent, side, slot, (z, z, z, z, z), eligible)


@dataclass(frozen=True, eq=False)
class Catalog:
    """Compiler-registered host holder, not a copied namespace tag."""
    engine_binding: tuple
    source_binding: tuple
    move_binding: tuple
    binding: str
    ids: tuple
    targets: tuple
    engine_table: object
    public_moves: object


_COMPILED = weakref.WeakKeyDictionary()


def _binding(value):
    a = np.asarray(value)
    if a.shape != (8,) or a.dtype != np.uint32:
        raise ValueError('eight uint32 namespace words required')
    return tuple(int(x) for x in a)


def _immutable_tables(engine_table, public_moves):
    """Require the real compiled types and immutable numerical leaves.

    Object-edge identity cannot bind a writable NumPy buffer's contents.
    No implicit coercion, copying, content fingerprinting or duck-typed holder
    can stand in for the original compiled JAX tables at this host boundary.
    """
    from .transition import MoveTable
    from .public_moves import MoveCatalog, MoveTables
    def immutable(value):
        if isinstance(value,jax.Array) and not isinstance(value,jax.core.Tracer):
            return True
        return value is None or (isinstance(value,tuple) and hasattr(value,'_fields')
            and all(immutable(field) for field in value))
    if (type(engine_table) is not MoveTable or not immutable(engine_table)
            or type(public_moves) is not MoveCatalog or type(public_moves.device) is not MoveTables
            or not immutable(public_moves.device) or type(public_moves.names) is not tuple
            or type(public_moves.ids) is not MappingProxyType):
        raise ValueError('real compiled holders with immutable JAX array leaves required')


def compile_catalog(metadata, engine_table, *, public_moves, dex, exposure):
    """Exact ID mapping; unsupported rows retain their original engine ID.

    The constructor/metadata/engine-table relationship and upstream public
    compiler provenance are trusted host boundaries. Real compiled types and
    immutable JAX leaves are required; these guards do not attest which upstream
    factory minted them. Basic engine columns are checked, not every mechanic.
    Retained exact array edges reject copied bindings on replaced runtime rows.
    """
    _immutable_tables(engine_table,public_moves)
    if (type(exposure) is not HealthExposure or exposure.showdown_revision != SHOWDOWN
            or exposure.format_id != 'gen3randombattle' or exposure.report_percentages is not True
            or exposure.report_exact_hp is not False):
        raise ValueError('pinned Gen3 percentage-health source required')
    if any(metadata[key] != exposure.format_id for key in ('format','format_id') if key in metadata):
        raise ValueError('engine/source format exposure drift')
    names = metadata['vocabulary']['moves']
    if (type(names) is not list or not names or names[0] not in ('', '_none')
            or any(type(s) is not str or not normalize_id(s) for s in names[1:])
            or len({normalize_id(s) for s in names}) != len(names)
            or len(metadata['moves']) != len(names)):
        raise ValueError('exact ordered engine move namespace required')
    ids, targets, wire_names = [0], [-1], ['']
    for i, name in enumerate(names[1:], 1):
        row = metadata['moves'][i]
        if not isinstance(row, dict) or row.get('id') != name:
            raise ValueError('engine metadata move row drift')
        info = dex.move_info(name)
        if info is not None and (normalize_id(info.id) != normalize_id(name)
                or normalize_id(info.name) != normalize_id(name)):
            raise ValueError('dex move identity drift: ' + name)
        # The engine's variable_moves treats Hidden Power suffixes as metadata
        # aliases, not public move names. Pinned dex-data BasicEffect.realMove
        # canonicalizes their runtime ID; Gen3 scripts.ts163-166 then prints
        # literal 'Hidden Power', hiding type/power from both owners' MOVE.
        key=normalize_id(name)
        wire_key='hiddenpower' if key.startswith('hiddenpower') else key
        wire_info=dex.move_info(wire_key) if wire_key!=key else info
        if wire_info is not None and (normalize_id(wire_info.id)!=wire_key
                or normalize_id(wire_info.name)!=wire_key):
            raise ValueError('source lexical move identity drift: '+wire_key)
        public_id = public_moves.ids.get(wire_key, 0)
        if type(public_id) is not int or public_id and (not 0 < public_id < len(public_moves.names)
                or public_moves.names[public_id] != wire_key):
            raise ValueError('public move namespace row drift')
        target = (0 if wire_info.target == 'self' else 1 if wire_info.target in
            ('normal', 'adjacentFoe', 'any', 'allAdjacentFoes', 'allAdjacent', 'randomNormal') else -1) if wire_info else -1
        # The source resolver documents the one Gen3 type-dependent case.
        # Never infer its target rewrite from a static row or physical types.
        dynamic = bool(wire_info and resolve_move_effect(wire_info,()) != resolve_move_effect(wire_info,('Ghost',)))
        supported = bool(public_id and info and wire_info and target >= 0 and not dynamic)
        ids.append(int(public_id) if supported else 0)
        targets.append(target if supported else -1)
        wire_names.append(wire_key)
    checks = dict(power=[0]+[r['base_power'] for r in metadata['moves'][1:]],
        type=[0]+[r['type'] for r in metadata['moves'][1:]],
        physical=[False]+[r['category']=='Physical' for r in metadata['moves'][1:]])
    for field, values in checks.items():
        actual=np.asarray(getattr(engine_table, field))
        dtype=np.dtype(bool if field=='physical' else np.int32)
        if actual.shape!=(len(names),) or actual.dtype!=dtype or not np.array_equal(actual, np.asarray(values)):
            raise ValueError('engine move table differs from metadata: ' + field)
    engine = _binding(engine_metadata_binding(metadata))
    source, moves = _binding(public_moves.device.source_binding), _binding(public_moves.device.binding)
    payload = dict(schema=SCHEMA, exposure=asdict(exposure), engine=engine, source=source,
        moves=moves, names=names, wire_names=wire_names, ids=ids, targets=targets,
        hp='gen3_percentage_mod_owner_exact_nearfull99_faint0fnt')
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    catalog = Catalog(engine, source, moves, binding, tuple(ids), tuple(targets), engine_table, public_moves)
    _COMPILED[catalog] = (tuple(jax.tree.leaves(engine_table)), tuple(public_moves.device),
        tuple(public_moves.names), tuple(sorted(public_moves.ids.items())),
        (catalog.engine_binding,catalog.source_binding,catalog.move_binding,catalog.binding,catalog.ids,catalog.targets),
        public_moves)
    return catalog


def check_catalog(catalog, engine_table, *, expected_binding, expected_engine_binding):
    record = _COMPILED.get(catalog) if type(catalog) is Catalog else None
    if record is not None:
        _immutable_tables(engine_table,catalog.public_moves)
    if (record is None or engine_table is not catalog.engine_table or expected_binding != catalog.binding
            or _binding(expected_engine_binding) != catalog.engine_binding
            or len(jax.tree.leaves(engine_table)) != len(record[0])
            or any(a is not b for a, b in zip(jax.tree.leaves(engine_table), record[0]))
            or len(tuple(catalog.public_moves.device)) != len(record[1])
            or any(a is not b for a, b in zip(catalog.public_moves.device, record[1]))
            or tuple(catalog.public_moves.names) != record[2]
            or tuple(sorted(catalog.public_moves.ids.items())) != record[3]
            or (catalog.engine_binding,catalog.source_binding,catalog.move_binding,catalog.binding,catalog.ids,catalog.targets) != record[4]
            or catalog.public_moves is not record[5]):
        raise ValueError('original compiled host catalog/table/immutable edges required')


@dataclass(frozen=True)
class PublicRegistry:
    """Prior public evidence only; NEVER derive this from physical party order.

    active maps side -> (current physical slot, PublicAddress). It is supplied
    from prior established public arrival evidence. This cut cannot manufacture
    arrival evidence, so a different active slot refuses until registered.
    """
    active: object

    def resolve(self, side, slot):
        pair = self.active.get(side)
        if (type(side) is not int or side not in (0, 1) or type(slot) is not int or not 0 <= slot < 6
                or type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not int or pair[0] != slot
                or type(pair[1]) is not PublicAddress or pair[1].side != side):
            raise ValueError('callback requires the already-public active encounter, never private order')
        return pair[1]


@dataclass(frozen=True)
class Projection:
    stream_id: int
    owner: int
    events: tuple
    source_binding: tuple
    move_binding: tuple
    producer_binding: str
    refusal: int
    full_line_consumers_complete: bool = False
    full_encoder_admitted: bool = False
    B4_qualified: bool = False


def project(tape, *, battle, stream_id, owner, next_sequence, catalog, registry,
            expected_binding, expected_engine_binding):
    """Host-only component projection; not a v4 encoder-admission interface.

    Uses only callback receipts, never full BattleState. Owner selection is
    explicit before reading HP. The primitive tape itself is trusted to come
    from the real core; its scalar arrays are not cryptographic attestations.
    """
    check_catalog(catalog, catalog.engine_table, expected_binding=expected_binding,
        expected_engine_binding=expected_engine_binding)
    if (type(stream_id) is not int or not 0 < stream_id < 2**31
            or type(battle) is not int or not 0 <= battle < tape.words.shape[0] or type(owner) is not int
            or owner not in (0, 1) or type(next_sequence) is not int or not 1 <= next_sequence < 2**31
            or type(registry) is not PublicRegistry):
        raise ValueError('exact owner/battle/sequence/public registry required')
    if (tape.words.ndim != 3 or tape.words.shape[2] != CALLBACK_WORDS or tape.words.dtype != jnp.int32
            or any(a.shape != (tape.words.shape[0],) or a.dtype != jnp.int32 for a in (tape.count,tape.refusal,tape.turn))):
        raise ValueError('exact callback tape shapes/dtypes required')
    producer=tuple(int(x) for x in np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4'))
    if any(_binding(a)!=b for a,b in zip(tape[4:],
            (catalog.engine_binding,catalog.source_binding,catalog.move_binding,producer))):
        raise ValueError('callback producer/engine/public catalog namespace drift')
    count, refusal, turn = (int(np.asarray(a)[battle]) for a in (tape.count, tape.refusal, tape.turn))
    if turn < 0 or refusal < 0 or refusal & ~(UNINSTRUMENTED|OVERFLOW|INVALID_ACTION):
        raise ValueError('callback turn/refusal flags outside the named contract')
    if count < 0 or count > tape.words.shape[1] or refusal & OVERFLOW:
        raise ValueError('callback overflow refuses the entire projection; no truncated input')
    events = []
    for raw in np.asarray(tape.words)[battle, :count]:
        kind, actor_side, actor_slot, target_side, target_slot, a, b, c, d, e = map(int, raw)
        if d or e:
            raise ValueError('callback reserved words must be zero')
        actor = target = None
        if kind == MOVE:
            actor = registry.resolve(actor_side, actor_slot)
            if not 0 < a < len(catalog.ids) or not catalog.ids[a] or b or c:
                raise ValueError('unsupported/unknown engine move declaration, no guessed lexical tags')
            target = actor if catalog.targets[a] == 0 else registry.resolve(target_side, target_slot)
            payload = (actor.encounter, catalog.ids[a], 0, 0, 0, 0, 0, 0)
            event_kind = Kind.MOVE
        elif kind in (MAIN_HP, DIRECT_FAINT):
            if (actor_side, actor_slot) != (-1, -1):
                raise ValueError('condition/faint callback cannot carry actor fields')
            target = registry.resolve(target_side, target_slot)
            if kind == MAIN_HP:
                if not 0 <= a <= b <= 1_000_000 or b <= 0 or not 0 <= c <= 6:
                    raise ValueError('bounded exact callback HP/status required')
                # pinned Pokemon.getHealth: hp0 -> literal0fnt without status;
                # percentage ceiling has a 99 near-full guard, not 100.
                hp, maximum = (a, b) if owner == target_side else ((100*a+b-1)//b, 100)
                if owner != target_side and a < b and hp == 100:
                    hp = 99
                status = (0, 2, 3, 5, 6, 1, 4)[c]
                if a == 0:
                    hp, maximum, status = 0, 0, 0
                payload = (target.encounter, hp, maximum, status, int(a == 0), 1, 0, 0)
                event_kind = Kind.CONDITION
            else:
                if a or b or c:
                    raise ValueError('faint callback payload must be empty')
                payload = (target.encounter, 0, 0, 0, 0, 0, 0, 0)
                event_kind = Kind.FAINT
        else:
            raise ValueError('unsupported callback kind')
        event = PublicEvent(event_kind, next_sequence+len(events), turn, Phase.ACTION,
            actor_side=actor.side if actor else -1, actor_slot=actor.slot if actor else -1,
            target_side=target.side, target_slot=target.slot, payload=payload)
        event.words()
        events.append(event)
    return Projection(stream_id, owner, tuple(events), catalog.source_binding, catalog.move_binding, catalog.binding,
        refusal | UNINSTRUMENTED)
