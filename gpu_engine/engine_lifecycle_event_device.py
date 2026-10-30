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
from . import engine_status_callbacks as status_callbacks, engine_item_callbacks as item_callbacks
from . import public_items, public_active
from . import engine_weather_callbacks as weather_callbacks, weather, public_field, v4_encoder
from . import engine_weather_progress_callbacks as weather_progress
from . import engine_weather_effect_callbacks as weather_effects
from . import engine_weather_speed_profile as weather_speed
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
    status_events: object
    lum_engine_item: object
    lum_public_item: object
    lum_events: object
    weather_public: object
    weather_ability: object
    weather_guard_species: object
    weather_guard_item_possible: object
    weather_events: object
    weather_moves: object
    weather_duration_safe: object
    binding: object


@dataclass(frozen=True, eq=False)
class Compiled:
    projector: object
    species_catalog: object
    source: object
    context_tables: object
    device: Tables
    items: object = None
    weather_catalog: object = None


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


def _item_inputs(items, source):
    if items is None:
        return
    if (type(items) is not public_items.ItemCatalog or type(items.device) is not public_items.ItemTables
            or type(items.names) is not tuple or type(items.ids) is not MappingProxyType
            or not _immutable(items.device)
            or not np.array_equal(np.asarray(items.device.source_binding), np.asarray(source.device.binding))):
        raise ValueError('original immutable public item namespace required')


def compile_projector(metadata, projector, *, species_catalog, source, context_tables, status_events=False,
        items=None, weather_catalog=None):
    """Host-only compilation, retaining original guarded upstream handles.

    Upstream constructor/compiler provenance is trusted, not attested by tags.
    Missing public species rows stay unsupported. Off-source species retain
    source ID0, exactly as the existing source species_id lookup does.
    """
    _host_inputs(projector, species_catalog, source, context_tables)
    _item_inputs(items, source)
    if type(status_events) is not bool:
        raise ValueError('explicit static status-event mode required')
    if status_events and context_tables.items is not None:
        raise ValueError('status receipts require typed mode plus actual full raw-control composition')
    if items is not None and not status_events:
        raise ValueError('Lum receipts require full typed status/raw composition')
    if weather_catalog is not None:
        v4_encoder.check_catalog(weather_catalog)
        if (not status_events or weather_catalog.source is not source or weather_catalog.species is not species_catalog
                or weather_catalog.items is not items or weather_catalog.abilities.device is not context_tables.abilities
                or weather_catalog.moves.device is not context_tables.moves):
            raise ValueError('original full weather catalogue edges required')
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
    item_names = metadata['vocabulary'].get('items', [])
    lum_engine = next((i for i, name in enumerate(item_names) if normalize_id(name) == 'lumberry'), 0)
    lum_public = 0 if items is None else items.ids.get('lumberry', 0)
    lum = items is not None and lum_engine > 0 and lum_public > 0
    weather_ids = [0]*5; ability_ids = [0]*4; guard_species = [0]*4; guard_item = [False]*4
    weather_moves=np.array([{'raindance':weather.RAIN,'sunnyday':weather.SUN,
        'sandstorm':weather.SAND,'hail':weather.HAIL}.get(normalize_id(n),0)
        for n in metadata['vocabulary']['moves']],np.int32)
    # Original Gen3 generator excludes these Gen4 duration items. Supplied
    # metadata provenance alone does not establish that generation constraint.
    duration_safe=not any(normalize_id(n) in ('damprock','heatrock','smoothrock','icyrock') for n in item_names)
    if weather_catalog is not None:
        for code, name in ((weather.RAIN,'raindance'),(weather.SUN,'sunnyday'),
                (weather.SAND,'sandstorm'),(weather.HAIL,'hail')):
            weather_ids[code] = weather_catalog.field.weather_ids.get(name,0)
        ability_ids = [weather_catalog.abilities.ids.get(name,0) for name in weather_callbacks.CANONICAL_ABILITIES]
        # These are source lexical guards, not private held-item reads. If an
        # Orb exists in the bound engine namespace, only the publicly revealed
        # corresponding species is ambiguous; other actors are still proved.
        for proc, species_name, item_name in ((1,'kyogre','blueorb'),(2,'groudon','redorb')):
            guard_species[proc] = next((i for i,n in enumerate(names) if normalize_id(n)==species_name),0)
            guard_item[proc] = any(normalize_id(n)==item_name for n in item_names)
    payload = dict(schema=SCHEMA, base=np.asarray(projector.device.binding).tolist(), names=names,
        species=species, source_species=source_ids, species_binding=np.asarray(species_catalog.device.binding).tolist(),
        context=[None if a is None else np.asarray(a).tolist() for a in (
            context_tables.binding, context_tables.abilities.binding,
            context_tables.items.binding if raw else None,
            context_tables.field.binding if context_tables.field is not None else None,
            context_tables.damage.binding if context_tables.damage is not None else None)],
        raw_begin='source_basic_item_nonitem_zero_before_line' if raw else 'absent',
        status_events=status_events, lum_engine_item=lum_engine, lum_public_item=lum_public,
        lum_events=lum, item_binding=None if items is None else np.asarray(items.device.binding).tolist(),
        weather_public=weather_ids,weather_ability=ability_ids,weather_guard_species=guard_species,
        weather_guard_item_possible=guard_item,
        weather_moves=weather_moves.tolist(),weather_duration_safe=duration_safe,
        weather_binding=None if weather_catalog is None else np.asarray(weather_catalog.binding).tolist(),admission=False)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    device = Tables(projector.device, jnp.array(species, jnp.int32), jnp.array(source_ids, jnp.int32),
        species_catalog.device.binding, context_tables.abilities.binding,
        context_tables.items.binding if raw else None if items is None else items.device.binding,
        context_tables.binding if raw else None,
        context_tables.field.binding if context_tables.field is not None else None,
        context_tables.damage.binding if context_tables.damage is not None else None,
        jnp.array(raw), jnp.array(status_events), jnp.array(lum_engine, jnp.int32),
        jnp.array(lum_public, jnp.int32), jnp.array(lum),
        jnp.array(weather_ids,jnp.int32),jnp.array(ability_ids,jnp.int32),
        jnp.array(guard_species,jnp.int32),jnp.array(guard_item),jnp.array(weather_catalog is not None),
        jnp.array(weather_moves,jnp.int32),
        jnp.array(duration_safe),
        jnp.array(np.frombuffer(digest, dtype='>u4').astype(np.uint32)))
    result = Compiled(projector, species_catalog, source, context_tables, device, items, weather_catalog)
    _COMPILED[result] = (projector, species_catalog, source, context_tables, device,
        tuple(jax.tree.leaves((species_catalog.device, source.device, context_tables, device))),
        tuple(species_catalog.names), tuple(sorted(species_catalog.ids.items())),
        tuple(source.species_names), tuple(sorted(source.species_ids.items())), items,
        () if items is None else tuple(jax.tree.leaves(items.device)),
        () if items is None else tuple(items.names), () if items is None else tuple(sorted(items.ids.items())),weather_catalog)
    return result


def check_projector(compiled):
    """Host-only original-holder/immutable-edge guard, never inside hot JIT."""
    r = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (r is None or compiled.projector is not r[0] or compiled.species_catalog is not r[1]
            or compiled.source is not r[2] or compiled.context_tables is not r[3] or compiled.device is not r[4]
            or compiled.items is not r[10] or compiled.weather_catalog is not r[14]):
        raise ValueError('original compiled lifecycle projector required')
    _host_inputs(compiled.projector, compiled.species_catalog, compiled.source, compiled.context_tables)
    _item_inputs(compiled.items, compiled.source)
    if compiled.weather_catalog is not None:
        v4_encoder.check_catalog(compiled.weather_catalog)
    if compiled.items is not None:
        item_leaves = tuple(jax.tree.leaves(compiled.items.device))
        if (len(item_leaves) != len(r[11]) or any(a is not b for a, b in zip(item_leaves, r[11]))
                or tuple(compiled.items.names) != r[12] or tuple(sorted(compiled.items.ids.items())) != r[13]):
            raise ValueError('original immutable item namespace edges required')
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


def project_device(state, tape, before_registry, after_registry, tables, *, weather_hp=None):
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
    base._shape(tables.status_events, (), jnp.bool_)
    base._shape(tables.lum_events, (), jnp.bool_)
    base._shape(tables.lum_engine_item, (), jnp.int32)
    base._shape(tables.lum_public_item, (), jnp.int32)
    base._shape(tables.weather_events, (), jnp.bool_)
    base._shape(tables.weather_moves,tables.base.public_ids.shape,jnp.int32)
    base._shape(tables.weather_duration_safe,(),jnp.bool_)
    for value,shape,dtype in ((tables.weather_public,(5,),jnp.int32),
            (tables.weather_ability,(4,),jnp.int32),(tables.weather_guard_species,(4,),jnp.int32),
            (tables.weather_guard_item_possible,(4,),jnp.bool_)):
        base._shape(value,shape,dtype)
    # Optional trusted owning carry, never a naked admission boolean. Host
    # Session checks the original compiled profile and actual birth boundary.
    hp_profile = jnp.zeros(b, bool)
    if weather_hp is not None:
        if (type(weather_hp) is not tuple or len(weather_hp) != 2
                or type(weather_hp[0]) is not weather_speed.BirthRecord
                or type(weather_hp[1]) is not weather_speed.ProfileData):
            raise ValueError('owning weather birth carry and original static profile required')
        record, data = weather_hp
        for value in (data.binding,data.engine_binding,data.source_binding):
            base._shape(value,(8,),jnp.uint32)
        for value,shape,dtype in ((record.qualified_birth,(b,),jnp.bool_),
                (record.stream_id,(b,),jnp.int32),(record.engine_binding,(b,8),jnp.uint32),
                (record.source_binding,(b,8),jnp.uint32),(record.profile_binding,(b,8),jnp.uint32)):
            base._shape(value,shape,dtype)
        hp_profile = (record.qualified_birth & (record.stream_id == tape.stream_id)
            & jnp.all(record.engine_binding == tape.engine_binding,axis=1)
            & jnp.all(record.source_binding == tape.source_binding,axis=1)
            & jnp.all(record.profile_binding == data.binding,axis=1)
            & jnp.all(data.engine_binding == tape.engine_binding)
            & jnp.all(data.source_binding == tape.source_binding))
    rows = jnp.arange(b)
    known = public.UNINSTRUMENTED | public.OVERFLOW | public.INVALID_ACTION | life.UNKNOWN_DETAILS | life.UNKNOWN_BATON | life.MISSING_INSTALL | life.CLOCK | status_callbacks.UNKNOWN_STATUS | weather_callbacks.UNKNOWN_WEATHER
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
        applied = kind == status_callbacks.STATUS_APPLIED
        cured = kind == status_callbacks.STATUS_CURED
        status_event = applied | cured
        lum = kind == item_callbacks.LUM_CONSUMED
        entry_weather = kind == weather_callbacks.ENTRY_WEATHER
        move_weather = kind == weather_progress.MOVE_WEATHER
        clock_weather = kind == weather_progress.WEATHER_CLOCK
        failed_weather = kind == weather_effects.SAME_WEATHER_FAILURE
        weather_chip = kind == weather_effects.WEATHER_CHIP
        any_weather=entry_weather|move_weather|clock_weather
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
        status_slot,status_enc,status_identity = (value[:,0] for value in
            base._resolve(_registry(cursor),side[:,None],incoming[:,None]))
        old_status,new_status,reason = raw[:,5],raw[:,6],raw[:,7]
        status_ok = (tables.status_events & ~tables.raw_begin & status_identity & (phase == 0)
            & jnp.all(raw[:,1:3]==-1,axis=1) & jnp.all(raw[:,8:]==0,axis=1)
            & jnp.where(applied,(old_status==0)&(new_status>0)&(new_status<=6)&(reason==status_callbacks.PRIMARY),
                (new_status==0)&(((old_status==5)&(reason==status_callbacks.WAKE))
                    |((old_status==6)&(reason==status_callbacks.THAW)))))
        lum_ok = (tables.lum_events & tables.status_events & ~tables.raw_begin & status_identity
            & jnp.all(raw[:,1:3] == -1, axis=1)
            & (incoming == cursor.physical_active[rows,safe_side])
            & (raw[:,5] == tables.lum_engine_item) & (tables.lum_engine_item > 0)
            & (raw[:,6] >= 0) & (raw[:,6] <= 6) & (raw[:,7] == 0)
            & ((raw[:,8] == 0) | (raw[:,8] == 1)) & (raw[:,9] == 0)
            & ((raw[:,6] != 0) | (raw[:,8] == 1)))
        actor_side,actor_physical = raw[:,1],raw[:,2]
        actor_slot,actor_enc,actor_ok = (v[:,0] for v in
            base._resolve(_registry(cursor),actor_side[:,None],actor_physical[:,None]))
        proc=jnp.clip(raw[:,7],0,3)
        new_weather=jnp.array([0,weather.RAIN,weather.SUN,weather.SAND],jnp.int32)[proc]
        source_species=cursor.revealed_species_id[rows,jnp.clip(actor_side,0,1)]
        orb_ambiguous=tables.weather_guard_item_possible[proc] & (source_species==tables.weather_guard_species[proc])
        weather_ok=(tables.weather_events & tables.status_events & ~tables.raw_begin & actor_ok
            & (actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            & jnp.all(raw[:,3:5]==-1,axis=1) & (raw[:,5]>=weather.NONE)&(raw[:,5]<=weather.HAIL)
            & (raw[:,7]>0)&(raw[:,7]<4)&(raw[:,6]==new_weather)
            & ((raw[:,8]==0)|(raw[:,8]==1))&(raw[:,9]==1)
            & ((raw[:,5]!=raw[:,6])|(raw[:,8]==1))
            & (tables.weather_public[jnp.clip(raw[:,6],0,4)]>0)&(tables.weather_ability[proc]>0)&~orb_ambiguous)
        move_index=jnp.clip(raw[:,7],0,tables.weather_moves.shape[0]-1)
        move_weather_ok=(tables.weather_events & tables.weather_duration_safe & tables.status_events & ~tables.raw_begin & actor_ok
            & (phase==0) & (actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            & jnp.all(raw[:,3:5]==-1,axis=1)&(raw[:,5]>=weather.NONE)&(raw[:,5]<=weather.HAIL)
            & (raw[:,6]>=weather.RAIN)&(raw[:,6]<=weather.HAIL)&(raw[:,5]!=raw[:,6])
            & (raw[:,7]>0)&(raw[:,7]<tables.weather_moves.shape[0])
            & (tables.weather_moves[move_index]==raw[:,6])&(tables.base.public_ids[move_index]>0)
            & (raw[:,8]>=0)&(raw[:,8]<=5)&(raw[:,9]==5)
            & (tables.weather_public[jnp.clip(raw[:,6],0,4)]>0))
        failed_weather_ok=(tables.weather_events & tables.status_events & ~tables.raw_begin & actor_ok
            & (phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            & jnp.all(raw[:,3:5]==-1,axis=1)&(raw[:,5]>=weather.RAIN)&(raw[:,5]<=weather.HAIL)
            & (raw[:,6]==raw[:,5])&(raw[:,8]==raw[:,5])&(raw[:,9]==weather_effects.STILL_REQUIRED)
            & (raw[:,7]>0)&(raw[:,7]<tables.weather_moves.shape[0])
            & (tables.weather_moves[move_index]==raw[:,5])&(tables.base.public_ids[move_index]>0)
            & (tables.weather_public[jnp.clip(raw[:,5],0,4)]>0))
        upkeep=(raw[:,9]==weather_progress.UPKEEP)&(raw[:,6]==raw[:,5])&(
            ((raw[:,7]==0)&(raw[:,8]==0))|((raw[:,7]>1)&(raw[:,8]==raw[:,7]-1)))
        expires=(raw[:,9]==weather_progress.EXPIRE)&(raw[:,7]==1)&(raw[:,8]==0)&(raw[:,6]==weather.NONE)
        clock_weather_ok=(tables.weather_events & tables.status_events & ~tables.raw_begin & (phase==0)
            & (tables.weather_duration_safe|(raw[:,7]==0))
            & jnp.all(raw[:,1:5]==-1,axis=1)&(raw[:,5]>=weather.RAIN)&(raw[:,5]<=weather.HAIL)
            & (raw[:,7]>=0)&(raw[:,7]<=5)&(upkeep|expires)
            & (tables.weather_public[jnp.clip(raw[:,5],0,4)]>0))
        chip_ok=(hp_profile & tables.weather_events & tables.status_events & ~tables.raw_begin
            & status_identity & (phase==0) & (incoming==cursor.physical_active[rows,safe_side])
            & (raw[:,1]>0)&(raw[:,2]>0)&(raw[:,1]!=raw[:,2])
            & (raw[:,5]>raw[:,6])&(raw[:,6]>0)&(raw[:,7]>=raw[:,5])&(raw[:,7]<=1_000_000)
            & (raw[:,8]>=0)&(raw[:,8]<=6)&((raw[:,9]==weather.SAND)|(raw[:,9]==weather.HAIL)))
        event_ok = generation_ok & clock_ok & jnp.where(arrival, arrival_ok,
            jnp.where(clock, clock_payload, jnp.where(status_event,status_ok,
                jnp.where(lum,lum_ok,jnp.where(entry_weather,weather_ok,
                    jnp.where(move_weather,move_weather_ok,jnp.where(clock_weather,clock_weather_ok,
                        jnp.where(failed_weather,failed_weather_ok,jnp.where(weather_chip,chip_ok,component & component_ok)))))))))
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
        status_payload = jnp.stack((status_enc,jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(new_status,0,6)],z,z,z,z,z,z),axis=1)
        status_semantic = jnp.concatenate((jnp.stack((jnp.full((b,),int(inputs.Kind.STATUS),jnp.int32),
            semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),status_payload),axis=1)
        semantic=jnp.where(status_event[:,None],status_semantic,semantic)
        item_payload = jnp.stack((status_enc, jnp.full_like(z, tables.lum_public_item),
            jnp.full_like(z, public_items.CONSUME), z,z,z,z,z), axis=1)
        item_semantic = jnp.concatenate((jnp.stack((jnp.full_like(z, int(inputs.Kind.ITEM)),
            semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),item_payload),axis=1)
        semantic=jnp.where(lum[:,None],item_semantic,semantic)
        weather_payload=jnp.stack((jnp.full_like(z,public_field.WEATHER),
            tables.weather_public[jnp.clip(raw[:,6],0,4)],
            (clock_weather&(raw[:,9]==weather_progress.UPKEEP)).astype(jnp.int32),
            entry_weather.astype(jnp.int32),z,z,z,z),axis=1)
        weather_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.FIELD)),
            semantic_sequence,turn,phase,absent,absent,absent,absent),axis=1),weather_payload),axis=1)
        semantic=jnp.where(any_weather[:,None],weather_semantic,semantic)
        failed_payload=jnp.stack((actor_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        failed_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),
            semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),failed_payload),axis=1)
        semantic=jnp.where(failed_weather[:,None],failed_semantic,semantic)
        chip_percent=(100*raw[:,6]+jnp.maximum(1,raw[:,7])-1)//jnp.maximum(1,raw[:,7])
        chip_percent=jnp.where((raw[:,6]<raw[:,7])&(chip_percent==100),99,chip_percent)
        chip_payload=jnp.stack((status_enc,jnp.where(own,raw[:,6],chip_percent),
            jnp.where(own,raw[:,7],100),jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(raw[:,8],0,6)],
            z,jnp.ones_like(z),jnp.ones_like(z),z),axis=1)  # damage, explicit [from], no Spikes
        chip_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),
            semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),chip_payload),axis=1)
        semantic=jnp.where(weather_chip[:,None],chip_semantic,semantic)
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


def project_frame(state, frame, tables, *, weather_hp=None):
    """Device connection from an actual lifecycle result; never reads its state."""
    if type(frame) is not life.LifecycleResult:
        raise ValueError('actual lifecycle result required')
    return project_device(state, frame.callbacks, frame.before_registry, frame.after_registry, tables,weather_hp=weather_hp)


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
