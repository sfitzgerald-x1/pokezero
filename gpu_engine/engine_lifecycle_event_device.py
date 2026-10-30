"""Causal lifecycle callbacks -> existing typed public consumers, PARTIAL ONLY.

The device scan owns a persistent public encounter map. It starts before real
initial reveals and advances only on actual installation receipts. Per-event
BEFORE generation tags are checked before resolving any following declaration.
Before/after snapshots validate continuity; neither final BattleState nor the
post-step active index is an identity source for emitted words.

The caller must deliver each genuine source frame exactly once. LifecycleTape
v1 has no monotonic frame/event ordinal: available clocks/SWAP generations do
not independently distinguish every same-turn, no-install frame replay.

Unknown details/Baton, missing installations, unsupported callbacks and overflow
poison the row. No complete raw-line, full encoder or B4 coverage is claimed.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from . import engine_lifecycle_callbacks as life, engine_public_event_device as base
from . import engine_public_events as public, public_inputs as inputs
from . import public_belief, public_context, public_team
from .engine_private_team import engine_metadata_binding

SCHEMA = 'pokezero.lifecycle_device_events.partial.v1'


class Tables(NamedTuple):
    base: base.DeviceTables
    species: object
    source_species: object
    species_binding: object
    ability_binding: object
    item_binding: object
    context_binding: object
    field_binding: object
    damage_binding: object
    raw_begin: object
    binding: object


@dataclass(frozen=True, eq=False)
class Compiled:
    projector: object
    species_catalog: object
    source: object
    context_tables: object
    device: Tables


_COMPILED = weakref.WeakKeyDictionary()


def _immutable(value):
    return (isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer)) or value is None or (
        isinstance(value, tuple) and hasattr(value, '_fields') and all(_immutable(v) for v in value))


def _host_inputs(projector, species_catalog, source, context_tables):
    b = base.check_projector(projector)
    if (type(species_catalog) is not public_team.PublicSpeciesCatalog
            or type(species_catalog.device) is not public_team.PublicTeamTables
            or type(species_catalog.names) is not tuple or type(species_catalog.ids) is not MappingProxyType
            or type(source) is not public_belief.SourceTables or type(source.device) is not public_belief.DeviceTables
            or not _immutable(source.device)
            or not _immutable(species_catalog.device)
            or type(context_tables) is not public_context.ContextTables or not _immutable(context_tables)):
        raise ValueError('original typed immutable source/species/context tables required')
    if context_tables.moves is not projector.catalog.public_moves.device:
        raise ValueError('context requires the original reviewed public move table holder')
    if (context_tables.refresh is not None or context_tables.lifecycle is not None
            or context_tables.sleep_clause is not None or context_tables.switches is not None
            or context_tables.nonproc is not None or context_tables.mudshot is not None
            or context_tables.toxic is not None or context_tables.tendency is not None
            or context_tables.sleep is not None or context_tables.types is not None or context_tables.active is not None):
        raise ValueError('enhanced raw semantic/closure producers remain unknown in this component')
    for a, expected in ((source.device.binding, b.source_binding),
            (species_catalog.device.source_binding, b.source_binding),
            (context_tables.abilities.source_binding, b.source_binding),
            (context_tables.moves.source_binding, b.source_binding),
            (context_tables.moves.binding, b.move_binding),
            (species_catalog.device.ability_binding, context_tables.abilities.binding)):
        if not np.array_equal(np.asarray(a), np.asarray(expected)):
            raise ValueError('lifecycle source/public namespace drift')


def compile_projector(metadata, projector, *, species_catalog, source, context_tables):
    """Host-only compilation, retaining original guarded upstream handles.

    Upstream constructor/compiler provenance is trusted, not attested by tags.
    Missing public species rows stay unsupported. Off-source species retain
    source ID0, exactly as the existing source species_id lookup does.
    """
    _host_inputs(projector, species_catalog, source, context_tables)
    if not np.array_equal(np.asarray(engine_metadata_binding(metadata)), np.asarray(projector.device.engine_binding)):
        raise ValueError('exact original constructor metadata required')
    names = metadata['vocabulary']['species']
    if (type(names) is not list or not names or names[0] not in ('', '_none')
            or any(type(n) is not str or not normalize_id(n) for n in names[1:])
            or len(set(map(normalize_id, names))) != len(names)):
        raise ValueError('ordered original engine species namespace required')
    species, source_ids = [0], [0]
    for name in names[1:]:
        value = species_catalog.ids.get(normalize_id(name), 0)
        if value and (type(value) is not int or not 0 < value < len(species_catalog.names)
                or normalize_id(species_catalog.names[value]) != normalize_id(name)):
            raise ValueError('public dex species row drift')
        species.append(value)
        source_ids.append(source.species_id(name))
    raw = context_tables.items is not None
    payload = dict(schema=SCHEMA, base=np.asarray(projector.device.binding).tolist(), names=names,
        species=species, source_species=source_ids, species_binding=np.asarray(species_catalog.device.binding).tolist(),
        context=[None if a is None else np.asarray(a).tolist() for a in (
            context_tables.binding, context_tables.abilities.binding,
            context_tables.items.binding if raw else None,
            context_tables.field.binding if context_tables.field is not None else None,
            context_tables.damage.binding if context_tables.damage is not None else None)],
        raw_begin='source_basic_item_nonitem_zero_before_line' if raw else 'absent', admission=False)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    device = Tables(projector.device, jnp.array(species, jnp.int32), jnp.array(source_ids, jnp.int32),
        species_catalog.device.binding, context_tables.abilities.binding,
        context_tables.items.binding if raw else None, context_tables.binding if raw else None,
        context_tables.field.binding if context_tables.field is not None else None,
        context_tables.damage.binding if context_tables.damage is not None else None,
        jnp.array(raw), jnp.array(np.frombuffer(digest, dtype='>u4').astype(np.uint32)))
    result = Compiled(projector, species_catalog, source, context_tables, device)
    _COMPILED[result] = (projector, species_catalog, source, context_tables, device,
        tuple(jax.tree.leaves((species_catalog.device, source.device, context_tables, device))),
        tuple(species_catalog.names), tuple(sorted(species_catalog.ids.items())),
        tuple(source.species_names), tuple(sorted(source.species_ids.items())))
    return result


def check_projector(compiled):
    """Host-only original-holder/immutable-edge guard, never inside hot JIT."""
    r = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (r is None or compiled.projector is not r[0] or compiled.species_catalog is not r[1]
            or compiled.source is not r[2] or compiled.context_tables is not r[3] or compiled.device is not r[4]):
        raise ValueError('original compiled lifecycle projector required')
    _host_inputs(compiled.projector, compiled.species_catalog, compiled.source, compiled.context_tables)
    leaves = tuple(jax.tree.leaves((compiled.species_catalog.device, compiled.source.device,
        compiled.context_tables, compiled.device)))
    if (len(leaves) != len(r[5]) or any(a is not b for a, b in zip(leaves, r[5]))
            or tuple(compiled.species_catalog.names) != r[6]
            or tuple(sorted(compiled.species_catalog.ids.items())) != r[7]
            or tuple(compiled.source.species_names) != r[8]
            or tuple(sorted(compiled.source.species_ids.items())) != r[9]):
        raise ValueError('original immutable lifecycle namespace edges required')
    return compiled.device


class StreamState(NamedTuple):
    cursor: life.Cursor
    owner: object
    sequence: object
    admitted: object


def register(stream_ids, owners, *, compiled):
    """Host-only NEW stream, before actual initialize/reveals; no party join."""
    check_projector(compiled)
    base._shape(owners, stream_ids.shape, jnp.int32)
    if not isinstance(owners, jax.Array) or isinstance(owners, jax.core.Tracer) or not bool(np.all((np.asarray(owners) >= 0) & (np.asarray(owners) <= 1))):
        raise ValueError('explicit immutable observer owner vector required')
    catalog = compiled.projector.catalog
    cursor = life.register(stream_ids, catalog=catalog, expected_binding=catalog.binding,
        expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
    return StreamState(cursor, owners, jnp.zeros_like(stream_ids), jnp.ones_like(stream_ids, bool))


class Projection(NamedTuple):
    state: StreamState
    chunk: inputs.EventChunk
    error: object
    refusal: object
    component_admitted: object
    full_line_consumers_complete: object
    full_encoder_admitted: object
    B4_qualified: object


def _equal_rows(a, b):
    return jnp.all((a == b).reshape((a.shape[0], -1)), axis=1)


def _snapshot_equal(cursor, snapshot):
    # Explicit same-lifecycle metadata comparison; not an emission identity join.
    return (_equal_rows(cursor.stream_id, snapshot.stream_id)
        & _equal_rows(cursor.engine_binding, snapshot.engine_binding)
        & _equal_rows(cursor.generation, snapshot.generation)
        & _equal_rows(cursor.physical_active, snapshot.physical_active)
        & _equal_rows(cursor.public_slot, snapshot.public_slot)
        & _equal_rows(cursor.encounter, snapshot.encounter)
        & _equal_rows(cursor.established, snapshot.established)
        & _equal_rows(cursor.revealed_species_id, snapshot.revealed_species_id))


def _registry(cursor):
    return base.PublicRegistry(stream_id=cursor.stream_id, engine_binding=cursor.engine_binding,
        generation=cursor.generation, physical_active=cursor.physical_active,
        public_slot=cursor.public_slot, encounter=cursor.encounter,
        revealed_species_id=cursor.revealed_species_id, established=cursor.established)


def _snapshot_shape(snapshot, batch):
    if type(snapshot) is not life.PublicArrivalSnapshot:
        raise ValueError('actual lifecycle public-arrival snapshot required')
    base._shape(snapshot.stream_id, (batch,), jnp.int32)
    base._shape(snapshot.engine_binding, (batch, 8), jnp.uint32)
    base._shape(snapshot.generation, (batch, 2), jnp.int32)
    base._shape(snapshot.physical_active, (batch, 2), jnp.int32)
    base._shape(snapshot.public_slot, (batch, 2), jnp.int32)
    base._shape(snapshot.encounter, (batch, 2), jnp.int32)
    base._shape(snapshot.established, (batch, 2), jnp.bool_)
    base._shape(snapshot.revealed_species_id, (batch, 2), jnp.int32)


def project_device(state, tape, before_registry, after_registry, tables):
    """Trusted device primitive: real LifecycleTape and causal source snapshots.

    All host guards are separate. No host reads, string reconstruction or event
    loop runs here: a device scan resolves each receipt using its BEFORE state.
    Basic item RAW_BEGIN words are emitted only when that exact source consumer
    is configured; their headers use PRE-line clocks, including TURN/UPKEEP.
    Enhanced raw-family/closure compositions are explicitly unsupported.
    Exactly-once delivery of genuine frames remains a caller obligation; this
    partial source surface does not manufacture a missing source ordinal.
    """
    if type(state) is not StreamState or type(tape) is not life.LifecycleTape or type(tables) is not Tables:
        raise ValueError('typed lifecycle stream/tape/static tables required')
    b, capacity, _ = tape.words.shape
    if b <= 0 or not 1 <= capacity <= 64:
        raise ValueError('bounded lifecycle tape required')
    base._shape(tape.words, (b, capacity, 10), jnp.int32)
    base._shape(tape.count, (b,), jnp.int32)
    base._shape(tape.refusal, (b,), jnp.int32)
    base._shape(tape.stream_id, (b,), jnp.int32)
    base._shape(tape.event_turn, (b, capacity), jnp.int32)
    base._shape(tape.event_phase, (b, capacity), jnp.int32)
    base._shape(tape.event_generation, (b, capacity, 2), jnp.int32)
    base._shape(tape.switch_details, (b, capacity, 6), jnp.int32)
    base._shape(state.owner, (b,), jnp.int32)
    base._shape(state.sequence, (b,), jnp.int32)
    base._shape(state.admitted, (b,), jnp.bool_)
    _snapshot_shape(before_registry, b)
    _snapshot_shape(after_registry, b)
    base._shape(state.cursor.stream_id, (b,), jnp.int32)
    base._shape(tape.cursor.stream_id, (b,), jnp.int32)
    base._shape(tables.species, tables.source_species.shape, jnp.int32)
    base._shape(tables.source_species, tables.species.shape, jnp.int32)
    if tables.species.ndim != 1 or tables.species.shape[0] <= 0:
        raise ValueError('static engine/public species maps required')
    base._shape(tables.raw_begin, (), jnp.bool_)
    rows = jnp.arange(b)
    known = public.UNINSTRUMENTED | public.OVERFLOW | public.INVALID_ACTION | life.UNKNOWN_DETAILS | life.UNKNOWN_BATON | life.MISSING_INSTALL | life.CLOCK
    fatal = (tape.refusal & ~public.UNINSTRUMENTED) != 0
    ok = (state.admitted & life._valid(state.cursor) & life._valid(tape.cursor)
        & _snapshot_equal(state.cursor, before_registry)
        & (tape.stream_id == state.cursor.stream_id) & (state.sequence >= 0)
        & (state.owner >= 0) & (state.owner <= 1) & (tape.count >= 0) & (tape.count <= capacity)
        & ((tape.refusal & ~known) == 0) & ~fatal
        & jnp.all(tape.engine_binding == tables.base.engine_binding)
        & jnp.all(tape.source_binding == tables.base.source_binding)
        & jnp.all(tape.move_binding == tables.base.move_binding)
        & jnp.all(tape.producer_binding == tables.base.producer_binding))
    error = jnp.where(ok, 0, int(inputs.Error.SOURCE)).astype(jnp.int32)
    buffer = jnp.zeros((b, 64, 16), jnp.int32)
    count = jnp.zeros((b,), jnp.int32)

    def step(carry, index):
        cursor, sequence, buffer, count, error, overflow = carry
        raw = tape.words[:, index]
        kind, side, incoming = raw[:, 0], raw[:, 3], raw[:, 4]
        present = index < tape.count
        initial = kind == life.INITIAL_REVEAL
        arrival = initial | (kind == life.SWITCH_INSTALLED)
        clock = (kind == life.TURN) | (kind == life.UPKEEP) | (kind == life.ACTION_END)
        component = (kind == public.MOVE) | (kind == public.MAIN_HP) | (kind == public.DIRECT_FAINT)
        safe_side, safe_in = jnp.clip(side, 0, 1), jnp.clip(incoming, 0, 5)
        old = cursor.physical_to_public[rows, safe_side, safe_in]
        slot = jnp.where(old < 0, cursor.next_slot[rows, safe_side], old)
        generation_ok = jnp.all(tape.event_generation[:, index] == cursor.generation, axis=1)
        turn, phase = tape.event_turn[:, index], tape.event_phase[:, index]
        clock_ok = ((turn >= 0) & ((phase == 0) | (phase == 1))
            & jnp.where(kind == life.TURN, (turn == cursor.turn+1) & (cursor.turn < 2**31-1) & (phase == 0),
                (turn == cursor.turn) & jnp.where(kind == life.UPKEEP, phase == 1, phase == cursor.phase)))
        details = tape.switch_details[:, index]
        species, level, gender, hp, maximum, status = (details[:, i] for i in range(6))
        species_index = jnp.clip(species, 0, tables.species.shape[0]-1)
        public_species = tables.species[species_index]
        arrival_ok = ((side >= 0) & (side <= 1) & (incoming >= 0) & (incoming < 6)
            & (slot >= 0) & (slot < 6) & (species > 0) & (species < tables.species.shape[0]) & (public_species > 0)
            & (level > 0) & (level <= 100) & (gender >= 0) & (gender <= 2)
            & (hp > 0) & (hp <= maximum) & (maximum <= 1_000_000) & (status >= 0) & (status <= 6)
            & jnp.all(raw[:, 6:] == 0, axis=1) & ((raw[:, 5] == 0) | (raw[:, 5] == 1))
            & jnp.where(initial, (raw[:, 1] == -1) & (raw[:, 2] == -1) & ~cursor.initialized
                & ~cursor.established[rows, safe_side] & (raw[:, 5] == 0) & (turn == 0) & (phase == 0),
                (raw[:, 1] == side) & (raw[:, 2] == cursor.physical_active[rows, safe_side])
                & (raw[:, 2] != incoming) & cursor.initialized & (cursor.generation[rows, safe_side] < 2**31-1)))
        clock_payload = jnp.all(raw[:, 1:5] == -1, axis=1) & jnp.all(raw[:, 5:] == 0, axis=1)
        clock_payload &= jnp.where(kind == life.TURN, jnp.all(cursor.established, axis=1), cursor.initialized)
        ordinary = public.CallbackTape(raw[:, None, :], (present & component).astype(jnp.int32),
            jnp.full((b,), public.UNINSTRUMENTED, jnp.int32), turn, tape.engine_binding,
            tape.source_binding, tape.move_binding, tape.producer_binding)
        semantic_sequence = sequence + 1 + tables.raw_begin.astype(jnp.int32)
        observer = base.ObserverContext(cursor.stream_id, state.owner, semantic_sequence, cursor.generation)
        projected = base.project_device(ordinary, tables.base, _registry(cursor), observer)
        component_ok = (projected.error == 0) & (phase == 0)
        event_ok = generation_ok & clock_ok & jnp.where(arrival, arrival_ok,
            jnp.where(clock, clock_payload, component & component_ok))
        amount = 1 + tables.raw_begin.astype(jnp.int32)
        capacity_ok = count <= 64-amount
        sequence_ok = sequence <= 2**31-1-amount
        overflow |= present & (error == 0) & ~capacity_ok
        room = capacity_ok & sequence_ok
        event_ok &= room
        error = jnp.where(present & (error == 0) & ~event_ok,
            jnp.where(sequence_ok, int(inputs.Error.PAYLOAD), int(inputs.Error.SEQUENCE)), error)
        apply = present & (error == 0)
        own = state.owner == side
        percent = (100*hp + jnp.maximum(1, maximum)-1)//jnp.maximum(1, maximum)
        percent = jnp.where((hp < maximum) & (percent == 100), 99, percent)
        h = jnp.where(own, hp, percent)
        m = jnp.where(own, maximum, 100)
        s = jnp.array([0, 2, 3, 5, 6, 1, 4], jnp.int32)[jnp.clip(status, 0, 6)]
        absent = jnp.full((b,), -1, jnp.int32)
        z = jnp.zeros((b,), jnp.int32)
        public_kind = jnp.where(arrival, int(inputs.Kind.SWITCH), jnp.where(kind == life.TURN,
            int(inputs.Kind.TURN), jnp.where(kind == life.UPKEEP, int(inputs.Kind.UPKEEP), int(inputs.Kind.ACTION_CHUNK_END))))
        arrival_payload = jnp.stack((slot+1, public_species, tables.source_species[species_index], level, gender, h, m, s), axis=1)
        semantic = jnp.concatenate((jnp.stack((public_kind, semantic_sequence, turn, phase, absent, absent,
            jnp.where(arrival, side, -1), jnp.where(arrival, slot, -1)), axis=1),
            jnp.where(arrival[:, None], arrival_payload, 0)), axis=1)
        semantic = jnp.where(component[:, None], projected.chunk.words[:, 0], semantic)
        raw_header = jnp.stack((jnp.full((b,), int(inputs.Kind.RAW_BEGIN), jnp.int32), sequence+1,
            cursor.turn, cursor.phase, absent, absent, absent, absent, z, z, z, z, z, z, z, z), axis=1)
        raw_index = jnp.minimum(count, 63)
        old_words = buffer[rows, raw_index]
        buffer = buffer.at[rows, raw_index].set(jnp.where((apply & tables.raw_begin)[:, None], raw_header, old_words))
        semantic_index = jnp.minimum(count+tables.raw_begin.astype(jnp.int32), 63)
        old_words = buffer[rows, semantic_index]
        buffer = buffer.at[rows, semantic_index].set(jnp.where(apply[:, None], semantic, old_words))
        use = apply & arrival
        def update(array, value):
            return array.at[rows, safe_side].set(jnp.where(use, value, array[rows, safe_side]))
        mapping = cursor.physical_to_public.at[rows, safe_side, safe_in].set(
            jnp.where(use, slot, cursor.physical_to_public[rows, safe_side, safe_in]))
        cursor = cursor._replace(physical_to_public=mapping,
            generation=update(cursor.generation, cursor.generation[rows, safe_side]+(~initial).astype(jnp.int32)),
            next_slot=update(cursor.next_slot, cursor.next_slot[rows, safe_side]+(old < 0).astype(jnp.int32)),
            physical_active=update(cursor.physical_active, incoming), public_slot=update(cursor.public_slot, slot),
            encounter=update(cursor.encounter, slot+1), established=update(cursor.established, jnp.ones_like(use)),
            revealed_species_id=update(cursor.revealed_species_id, species),
            turn=jnp.where(apply, turn, cursor.turn), phase=jnp.where(apply, phase, cursor.phase),
            initialized=cursor.initialized | (apply & (kind == life.TURN)))
        return (cursor, sequence+jnp.where(apply, amount, 0), buffer,
            count+jnp.where(apply, amount, 0), error, overflow), None

    (cursor, sequence, words, count, error, overflow), _ = jax.lax.scan(step,
        (state.cursor, state.sequence, buffer, count, error, jnp.zeros((b,), bool)), jnp.arange(capacity))
    consistent = (_snapshot_equal(cursor, after_registry) & (cursor.turn == tape.cursor.turn)
        & (cursor.phase == tape.cursor.phase) & _equal_rows(cursor.physical_to_public, tape.cursor.physical_to_public)
        & _equal_rows(cursor.next_slot, tape.cursor.next_slot))
    error = jnp.where((error == 0) & ~consistent, int(inputs.Error.IDENTITY), error)
    admitted = error == 0
    def commit(old, new):
        return jnp.where(admitted.reshape((b,)+(1,)*(old.ndim-1)), new, old)
    cursor = jax.tree.map(commit, state.cursor, cursor)._replace(admitted=state.cursor.admitted & admitted)
    next_state = StreamState(cursor, state.owner, jnp.where(admitted, sequence, state.sequence), state.admitted & admitted)
    valid = (jnp.arange(64)[None, :] < count[:, None]) & admitted[:, None]
    words = jnp.where(valid[:, :, None], words, 0)
    no = jnp.zeros((b,), bool)
    chunk = inputs.EventChunk(tape.stream_id, words, valid, no, tables.base.source_binding,
        species_binding=tables.species_binding, ability_binding=tables.ability_binding,
        move_binding=tables.base.move_binding, item_binding=tables.item_binding, context_binding=tables.context_binding,
        field_binding=tables.field_binding, damage_binding=tables.damage_binding)
    refusal = tape.refusal | public.UNINSTRUMENTED | jnp.where(overflow, public.OVERFLOW, 0)
    return Projection(next_state, chunk, error, refusal, admitted, no, no, no)


def project_frame(state, frame, tables):
    """Device connection from an actual lifecycle result; never reads its state."""
    if type(frame) is not life.LifecycleResult:
        raise ValueError('actual lifecycle result required')
    return project_device(state, frame.callbacks, frame.before_registry, frame.after_registry, tables)


class Consumed(NamedTuple):
    projection: Projection
    fold: inputs.FoldResult


def project_and_fold(state, frame, tables, cursor, beliefs, source_tables, *, mon_state, context, context_tables):
    """Existing typed consumer connection; atomic for projection OR fold error.

    Successful named consumers do not upgrade missing raw-family coverage.
    Host guards on original compiled handles remain required outside JIT.
    """
    p = project_frame(state, frame, tables)
    f = inputs.fold_chunk(cursor, beliefs, source_tables, p.chunk,
        mon_state=mon_state, context=context, context_tables=context_tables)
    error = jnp.where(p.error != 0, p.error, f.error)
    ok = error == 0
    b = error.shape[0]
    def commit(old, new):
        return jnp.where(ok.reshape((b,)+(1,)*(old.ndim-1)), new, old)
    next_state = jax.tree.map(commit, state, p.state)
    next_state = next_state._replace(admitted=state.admitted & ok,
        cursor=next_state.cursor._replace(admitted=state.cursor.admitted & ok))
    p = p._replace(state=next_state, error=error, component_admitted=ok,
        chunk=p.chunk._replace(words=jnp.where(ok[:, None, None], p.chunk.words, 0), valid=p.chunk.valid & ok[:, None]))
    f = inputs.FoldResult(jax.tree.map(commit, cursor, f.cursor), jax.tree.map(commit, beliefs, f.beliefs), error,
        jax.tree.map(commit, mon_state, f.mon_state), jax.tree.map(commit, context, f.context))
    return Consumed(p, f)
