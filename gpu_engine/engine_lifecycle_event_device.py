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
from . import engine_self_destruct_callbacks as self_destruct_callbacks
from . import engine_before_move_blocked_callbacks as before_move_blocked_callbacks
from . import engine_lifecycle_callbacks as life, engine_public_event_device as base
from . import engine_public_events as public, public_inputs as inputs, public_toxic
from . import engine_secondary_status_callbacks as secondary_callbacks
from . import public_belief, public_context, public_team
from . import engine_status_callbacks as status_callbacks, engine_item_callbacks as item_callbacks
from . import public_items, public_active
from . import engine_weather_callbacks as weather_callbacks, weather, public_field, v4_encoder
from . import engine_weather_progress_callbacks as weather_progress
from . import engine_weather_effect_callbacks as weather_effects
from . import engine_weather_speed_profile as weather_speed
from . import engine_leftovers_callbacks as leftovers_callbacks
from . import engine_status_residual_callbacks as status_residual_callbacks
from . import engine_direct_hit_controls as direct_hit_controls
from . import engine_direct_ko_callbacks as direct_ko_callbacks
from . import engine_no_damage_callbacks as no_damage
from . import engine_flash_fire_end_callbacks as flash_end
from . import engine_substitute_damage_callbacks as substitute_damage
from . import engine_protect_callbacks as protect_callbacks
from . import engine_pinch_item_callbacks as pinch_callbacks
from . import engine_bulk_up_callbacks as bulk_up_callbacks
from . import engine_self_boost_callbacks as self_boost_callbacks
from . import engine_baton_pass_callbacks as baton_callbacks
from . import engine_self_heal_callbacks as self_heal_callbacks
from . import engine_belly_drum_callbacks as belly_callbacks
from . import engine_sleep_blocked_callbacks as sleep_blocked_callbacks
from . import engine_sleep_talk_callbacks as sleep_talk_callbacks
from . import engine_focus_punch_callbacks as focus_callbacks
from . import engine_encore_callbacks as encore_callbacks
from . import engine_self_move_callbacks as self_move_callbacks
from . import engine_heal_bell_callbacks as heal_bell_callbacks
from . import engine_leech_seed_callbacks as leech_callbacks
from . import engine_primary_plain_status_callbacks as plain_primary_callbacks
from . import engine_color_change_callbacks as color_callbacks
from . import engine_pressure_callbacks as pressure_callbacks
from . import engine_trace_callbacks as trace_callbacks
from . import engine_substitute_callbacks as substitute_callbacks
from . import engine_toxic_callbacks as toxic_callbacks
from . import engine_recoil_callbacks as recoil_callbacks
from . import engine_rough_skin_callbacks as rough_callbacks
from . import engine_swords_dance_callbacks as swords_dance_callbacks
from . import engine_intimidate_callbacks as intimidate_callbacks
from . import engine_entry_spikes_callbacks as entry_spikes_callbacks
from . import engine_spikes_callbacks as spikes_callbacks
from . import engine_thunder_wave_callbacks as thunder_callbacks
from . import engine_sleep_powder_callbacks as powder_callbacks
from . import public_mon_state
from . import engine_calm_mind_callbacks as calm_mind_callbacks
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
    leftovers_engine_item: object
    leftovers_public_item: object
    pressure_engine_ability: object
    pressure_public_ability: object
    trace_engine_ability: object
    sturdy_engine_ability: object
    sturdy_public_ability: object
    substitute_move: object
    toxic_move: object
    recoil_move: object
    intimidate_engine_ability: object
    intimidate_public_ability: object
    calm_mind_move: object
    swords_dance_move: object
    spikes_move: object
    thunder_wave_move: object
    trace_public_ability: object
    no_damage: object
    color_categories: object
    primary_sleep_moves: object
    sleep_powder_move: object
    wonder_guard_engine_ability: object
    wonder_guard_public_ability: object
    entry_spikes_flying_type: object
    entry_spikes_type_count: object
    entry_spikes_levitate: object
    entry_spikes_item_count: object
    entry_spikes_boots_item: object
    heal_bell_move: object
    heal_bell_soundproof: object
    heal_bell_public_soundproof: object
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
    leftovers_engine = next((i for i, name in enumerate(item_names) if normalize_id(name) == 'leftovers'), 0)
    leftovers_public = 0 if items is None else items.ids.get('leftovers', 0)
    pressure_engine = next((i for i,name in enumerate(metadata['vocabulary'].get('abilities',[]))
        if normalize_id(name)=='pressure'),0)
    pressure_public = 0 if weather_catalog is None else weather_catalog.abilities.ids.get('pressure',0)
    intimidate_engine=next((i for i,n in enumerate(metadata['vocabulary'].get('abilities',[])) if normalize_id(n)=='intimidate'),0)
    intimidate_public=0 if weather_catalog is None else weather_catalog.abilities.ids.get('intimidate',0)
    entry_spikes_flying=next((i for i,n in enumerate(metadata['vocabulary'].get('types',[])) if n=='flying'),0)
    entry_spikes_types=len(metadata['vocabulary'].get('types',[]))
    entry_spikes_levitate=[n=='levitate' for n in metadata['vocabulary'].get('abilities',[])] or [False]
    entry_spikes_items=len(item_names)
    entry_spikes_boots=next((i for i,n in enumerate(item_names) if n=='heavydutyboots'),0)
    calm_mind_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if n=='calmmind' and projector.catalog.ids[i]>0 and projector.catalog.targets[i]==0),0)
    swords_dance_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if n=='swordsdance' and projector.catalog.ids[i]>0 and projector.catalog.targets[i]==0),0)
    spikes_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if n=='spikes'),0)
    thunder_wave_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='thunderwave'),0)
    sleep_powder_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='sleeppowder'),0)
    no_damage_data=no_damage.compile_data(projector.catalog)
    wonder_rows=[i for i,n in enumerate(metadata["vocabulary"].get("abilities",[])) if normalize_id(n)=="wonderguard"]
    if len(wonder_rows)>1:raise ValueError("one original Wonder Guard ability row required")
    wonder_engine=wonder_rows[0] if wonder_rows else 0
    wonder_public=0 if weather_catalog is None else weather_catalog.abilities.ids.get("wonderguard",0)
    heal_bell_move=next((i for i,n in enumerate(metadata['vocabulary']['moves'])if n=='healbell'and projector.catalog.ids[i]>0 and projector.catalog.targets[i]==0),0)
    heal_bell_soundproof=np.asarray(projector.catalog.engine_table.effects.soundproof,bool)if projector.catalog.engine_table.effects is not None else np.zeros(1,bool)
    heal_bell_public_soundproof=0 if weather_catalog is None else weather_catalog.abilities.ids.get('soundproof',0)
    from .public_moves import _static_category
    color_categories=np.asarray([0 if i==0 or weather_catalog is None else _static_category(weather_catalog.vocabulary,'type:'+name) for i,name in enumerate(metadata['vocabulary']['types'])],np.int32)
    primary_sleep_moves=np.array([bool(i and projector.catalog.ids[i]>0 and projector.catalog.targets[i]==1
        and metadata['moves'][i].get('category')=='Status' and int(projector.catalog.engine_table.power[i])==0
        and projector.catalog.engine_table.effects is not None and int(projector.catalog.engine_table.effects.status[i])==5)
        for i in range(len(metadata['moves']))],bool)
    ability_names=metadata['vocabulary'].get('abilities',[])
    trace_engine=next((i for i,n in enumerate(ability_names) if normalize_id(n)=='trace'),0)
    sturdy_engine=next((i for i,n in enumerate(ability_names) if normalize_id(n)=='sturdy'),0)
    sturdy_public=0 if weather_catalog is None else weather_catalog.abilities.ids.get('sturdy',0)
    # Closed source-specific copy vocabulary, indexed in the ORIGINAL engine
    # ability namespace. Never infer support from arbitrary copied IDs.
    trace_public=np.array([weather_catalog.abilities.ids.get(normalize_id(n),0)
        if weather_catalog is not None and normalize_id(n) in ('sturdy','naturalcure','oblivious','levitate','pressure') else 0
        for n in ability_names],np.int32)
    substitute_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='substitute'),0)
    toxic_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='toxic'),0)
    recoil_move=next((i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='doubleedge'),0)
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
    payload = dict(heal_bell_move=heal_bell_move,heal_bell_soundproof=heal_bell_soundproof.tolist(),heal_bell_public_soundproof=heal_bell_public_soundproof,schema=SCHEMA, base=np.asarray(projector.device.binding).tolist(), names=names,
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
        weather_binding=None if weather_catalog is None else np.asarray(weather_catalog.binding).tolist(),
        leftovers_engine_item=leftovers_engine,leftovers_public_item=leftovers_public,
        trace_engine_ability=trace_engine,sturdy_engine_ability=sturdy_engine,sturdy_public_ability=sturdy_public,
        trace_public_ability=trace_public.tolist(),sleep_powder_move=sleep_powder_move,primary_sleep_moves=primary_sleep_moves.tolist(),no_damage_binding=np.asarray(no_damage_data.binding).tolist(),color_categories=color_categories.tolist(),
        wonder_guard_engine_ability=wonder_engine,wonder_guard_public_ability=wonder_public,
        pressure_engine_ability=pressure_engine,pressure_public_ability=pressure_public,substitute_move=substitute_move,toxic_move=toxic_move,recoil_move=recoil_move,intimidate_engine_ability=intimidate_engine,intimidate_public_ability=intimidate_public,calm_mind_move=calm_mind_move,swords_dance_move=swords_dance_move,spikes_move=spikes_move,thunder_wave_move=thunder_wave_move,
        entry_spikes=(entry_spikes_flying,entry_spikes_types,entry_spikes_levitate,entry_spikes_items,entry_spikes_boots),admission=False)
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
        jnp.array(leftovers_engine,jnp.int32),jnp.array(leftovers_public,jnp.int32),
        jnp.array(pressure_engine,jnp.int32),jnp.array(pressure_public,jnp.int32),jnp.array(trace_engine,jnp.int32),jnp.array(sturdy_engine,jnp.int32),jnp.array(sturdy_public,jnp.int32),jnp.array(substitute_move,jnp.int32),jnp.array(toxic_move,jnp.int32),jnp.array(recoil_move,jnp.int32),jnp.array(intimidate_engine,jnp.int32),jnp.array(intimidate_public,jnp.int32),jnp.array(calm_mind_move,jnp.int32),jnp.array(swords_dance_move,jnp.int32),jnp.array(spikes_move,jnp.int32),jnp.array(thunder_wave_move,jnp.int32),jnp.array(trace_public,jnp.int32),no_damage_data,jnp.array(color_categories),jnp.array(primary_sleep_moves),jnp.array(sleep_powder_move,jnp.int32),jnp.array(wonder_engine,jnp.int32),jnp.array(wonder_public,jnp.int32),
        jnp.array(entry_spikes_flying,jnp.int32),jnp.array(entry_spikes_types,jnp.int32),jnp.array(entry_spikes_levitate,bool),jnp.array(entry_spikes_items,jnp.int32),jnp.array(entry_spikes_boots,jnp.int32),
        jnp.array(heal_bell_move,jnp.int32),jnp.array(heal_bell_soundproof),jnp.array(heal_bell_public_soundproof,jnp.int32),
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


def project_device(state, tape, before_registry, after_registry, tables, *, weather_hp=None, leftovers=None,
                   status_residual=None,hypothetical_hp=None):
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
    base._shape(tables.toxic_move, (), jnp.int32)
    base._shape(tables.recoil_move, (), jnp.int32)
    base._shape(tables.intimidate_engine_ability,(),jnp.int32)
    base._shape(tables.intimidate_public_ability,(),jnp.int32)
    base._shape(tables.calm_mind_move,(),jnp.int32)
    base._shape(tables.swords_dance_move,(),jnp.int32)
    base._shape(tables.spikes_move,(),jnp.int32)
    base._shape(tables.thunder_wave_move,(),jnp.int32)
    no_damage._shape(tables.no_damage)
    base._shape(tables.color_categories,(18,),jnp.int32)
    base._shape(tables.primary_sleep_moves,(len(tables.base.public_ids),),jnp.bool_)
    base._shape(tables.sleep_powder_move,(),jnp.int32)
    base._shape(tables.wonder_guard_engine_ability,(),jnp.int32)
    base._shape(tables.wonder_guard_public_ability,(),jnp.int32)
    base._shape(tables.substitute_move, (), jnp.int32)
    base._shape(tables.pressure_engine_ability, (), jnp.int32)
    base._shape(tables.pressure_public_ability, (), jnp.int32)
    if tables.trace_public_ability.ndim!=1 or tables.trace_public_ability.shape[0]<=0:
        raise ValueError('original engine-indexed Trace copy vocabulary required')
    base._shape(tables.trace_public_ability,tables.trace_public_ability.shape,jnp.int32)
    for v in (tables.trace_engine_ability,tables.sturdy_engine_ability,tables.sturdy_public_ability):
        base._shape(v,(),jnp.int32)
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
    base._shape(tables.leftovers_engine_item,(),jnp.int32)
    base._shape(tables.leftovers_public_item,(),jnp.int32)
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
    if hypothetical_hp is not None:
        from . import v5_hypothetical_profile as hypothesis
        if weather_hp is not None:raise ValueError('hypothetical HP observer cannot replace ordinary birth carry')
        hypothesis.check_context(hypothetical_hp,b)
        record,data=hypothetical_hp
        hp_profile=(record.qualified_origin & jnp.any(record.origin_binding!=0,axis=1)
            & (record.stream_id==tape.stream_id)
            & jnp.all(record.engine_binding==tape.engine_binding,axis=1)
            & jnp.all(record.source_binding==tape.source_binding,axis=1)
            & jnp.all(record.profile_binding==data.binding,axis=1)
            & jnp.all(data.engine_binding==tape.engine_binding)
            & jnp.all(data.source_binding==tape.source_binding))
    # Optional original helper data is supplied only by the owning host guard.
    # A canonical item integer or a caller boolean is not a registration.
    leftovers_profile=jnp.zeros(b,bool)
    if leftovers is not None:
        if type(leftovers) is not leftovers_callbacks.Data:
            raise ValueError('original checked Leftovers helper data required')
        for value in (leftovers.binding,leftovers.engine_binding,leftovers.source_binding,
                leftovers.move_binding,leftovers.producer_binding):
            base._shape(value,(8,),jnp.uint32)
        base._shape(leftovers.leftovers_item,(),jnp.int32)
        leftovers_profile=(hp_profile & (tables.leftovers_engine_item>0)&(tables.leftovers_public_item>0)
            & (leftovers.leftovers_item==tables.leftovers_engine_item)
            & jnp.all(leftovers.engine_binding==tape.engine_binding)
            & jnp.all(leftovers.source_binding==tape.source_binding)
            & jnp.all(leftovers.move_binding==tape.move_binding)
            & jnp.all(leftovers.producer_binding==tape.producer_binding))
    status_profile=jnp.zeros(b,bool)
    if status_residual is not None:
        if type(status_residual) is not status_residual_callbacks.Data:
            raise ValueError('original checked status residual helper data required')
        for value in (status_residual.binding,status_residual.engine_binding,status_residual.source_binding,
                status_residual.move_binding,status_residual.producer_binding):
            base._shape(value,(8,),jnp.uint32)
        base._shape(status_residual.reason_by_status,(7,),jnp.int32)
        base._shape(status_residual.damage_tag_by_reason,(4,),jnp.int32)
        status_profile=(hp_profile & jnp.all(status_residual.engine_binding==tape.engine_binding)
            & jnp.all(status_residual.source_binding==tape.source_binding)
            & jnp.all(status_residual.move_binding==tape.move_binding)
            & jnp.all(status_residual.producer_binding==tape.producer_binding)
            & jnp.all(status_residual.reason_by_status==jnp.array([0,0,1,2,3,0,0],jnp.int32))
            & jnp.all(status_residual.damage_tag_by_reason==jnp.array([0,1,2,2],jnp.int32)))
    for value in (tables.entry_spikes_flying_type,tables.entry_spikes_type_count,
            tables.entry_spikes_item_count,tables.entry_spikes_boots_item):base._shape(value,(),jnp.int32)
    if tables.entry_spikes_levitate.ndim!=1 or not len(tables.entry_spikes_levitate):raise ValueError('original entry ability namespace required')
    base._shape(tables.entry_spikes_levitate,tables.entry_spikes_levitate.shape,jnp.bool_)
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
        turnloop_prefix = kind == life.TURNLOOP_PREFIX
        component = (kind == public.MOVE) | (kind == public.MAIN_HP) | (kind == public.DIRECT_FAINT)
        applied = kind == status_callbacks.STATUS_APPLIED
        cured = kind == status_callbacks.STATUS_CURED
        status_event = applied | cured
        pressure = kind == pressure_callbacks.PRESSURE_START
        trace_copy = kind == trace_callbacks.TRACE_COPY
        substitute = kind == substitute_callbacks.SUBSTITUTE_CREATED
        sub_damage = kind == substitute_damage.SUBSTITUTE_DAMAGE
        toxic = kind == toxic_callbacks.TOXIC_ATTEMPT
        recoil = kind == recoil_callbacks.RECOIL_HP
        rough = kind == rough_callbacks.ROUGH_SKIN_HP
        belly = kind == belly_callbacks.BELLY_DRUM
        sleep_blocked = kind == sleep_blocked_callbacks.SLEEP_BLOCKED
        focus = kind == focus_callbacks.FOCUS_PREPARATION
        sleep_talk = kind == sleep_talk_callbacks.SLEEP_TALK
        sleep_talk_owned = sleep_talk_callbacks.owned(tape,index)
        sleep_talk_child = (kind==public.MOVE)&sleep_talk_owned
        before_move_blocked = kind == before_move_blocked_callbacks.BEFORE_MOVE_BLOCKED
        heal_bell = kind == heal_bell_callbacks.HEAL_BELL
        self_move = kind == self_move_callbacks.SELF_MOVE_INPUT
        self_destruct = kind == self_destruct_callbacks.SELF_DESTRUCT
        encore = kind == encore_callbacks.ENCORE
        leech = kind == leech_callbacks.LEECH
        intimidate = kind == intimidate_callbacks.INTIMIDATE
        calm_mind = kind == calm_mind_callbacks.CALM_MIND
        pinch = kind == pinch_callbacks.PINCH_ITEM
        bulk_up = kind == bulk_up_callbacks.BULK_UP
        self_boost = kind == self_boost_callbacks.SELF_BOOST_INPUT
        baton = kind == baton_callbacks.BATON_PASS
        color_change=kind==color_callbacks.COLOR_CHANGE
        swords_dance = kind == swords_dance_callbacks.SWORDS_DANCE
        spikes = kind == spikes_callbacks.SPIKES
        lum = kind == item_callbacks.LUM_CONSUMED
        entry_weather = kind == weather_callbacks.ENTRY_WEATHER
        move_weather = kind == weather_progress.MOVE_WEATHER
        clock_weather = kind == weather_progress.WEATHER_CLOCK
        failed_weather = kind == weather_effects.SAME_WEATHER_FAILURE
        weather_chip = kind == weather_effects.WEATHER_CHIP
        leftovers_heal = kind == leftovers_callbacks.LEFTOVERS_HEAL
        self_heal = kind == self_heal_callbacks.SELF_HEAL
        heal_failure=self_heal&((raw[:,9]&~self_heal_callbacks.SECONDARY)!=0)
        status_damage = kind == status_residual_callbacks.STATUS_RESIDUAL
        entry_spikes = kind == entry_spikes_callbacks.ENTRY_SPIKES_HP
        no_damage_header=kind==no_damage.TARGET_NO_DAMAGE
        no_damage_facts=kind==no_damage.ABSORB_FACTS
        flash_ended=kind==flash_end.FLASH_FIRE_END
        protect_attempt=kind==protect_callbacks.PROTECT_ATTEMPT
        protect_block=kind==protect_callbacks.PROTECT_BLOCK
        direct_control = ((kind == direct_hit_controls.CRITICAL)
            | (kind == direct_hit_controls.SUPER_EFFECTIVE) | (kind == direct_hit_controls.RESISTED))
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
        baton_arrival=arrival&(raw[:,6]==1)&baton_callbacks.owned(tape,index,baton_callbacks.INSTALL)
        arrival_ok = ((side >= 0) & (side <= 1) & (incoming >= 0) & (incoming < 6)
            & (slot >= 0) & (slot < 6) & (species > 0) & (species < tables.species.shape[0]) & (public_species > 0)
            & (level > 0) & (level <= 100) & (gender >= 0) & (gender <= 2)
            & (hp > 0) & (hp <= maximum) & (maximum <= 1_000_000) & (status >= 0) & (status <= 6)
            & jnp.where(baton_arrival,jnp.all(raw[:,7:]==0,axis=1),jnp.all(raw[:,6:]==0,axis=1)) & ((raw[:, 5] == 0) | (raw[:, 5] == 1))
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
        leftovers_ok=(leftovers_profile & tables.status_events & ~tables.raw_begin & status_identity
            & (phase==0)&(incoming==cursor.physical_active[rows,safe_side])
            & jnp.all(raw[:,1:3]==-1,axis=1)
            & (raw[:,5]>0)&(raw[:,6]>raw[:,5])&(raw[:,7]>=raw[:,6])&(raw[:,7]<=1_000_000)
            & (raw[:,8]>=0)&(raw[:,8]<=6)&(raw[:,9]==tables.leftovers_engine_item))
        event_ok = generation_ok & clock_ok & jnp.where(arrival, arrival_ok,
            jnp.where(clock, clock_payload, jnp.where(status_event,status_ok,
                jnp.where(lum,lum_ok,jnp.where(entry_weather,weather_ok,
                    jnp.where(move_weather,move_weather_ok,jnp.where(clock_weather,clock_weather_ok,
                        jnp.where(failed_weather,failed_weather_ok,jnp.where(weather_chip,chip_ok,
                            jnp.where(leftovers_heal,leftovers_ok,component & component_ok))))))))))
        status_damage_ok=(status_profile & tables.status_events & ~tables.raw_begin & status_identity
            & (phase==0)&(incoming==cursor.physical_active[rows,safe_side])
            & (raw[:,1]>=0)&(raw[:,1]<=15)&(raw[:,2]>=0)&(raw[:,2]<=15)
            & (raw[:,5]>raw[:,6])&((raw[:,6]>0)|(status_residual_callbacks.owned_lethal(tape,index,status_residual) if status_residual is not None else jnp.zeros(b,bool)))&(raw[:,7]>=raw[:,5])&(raw[:,7]<=1_000_000)
            & ((((raw[:,8]==2)&(raw[:,9]==1))|((raw[:,8]==3)&(raw[:,9]==2)))
                &(raw[:,2]==raw[:,1])
                |((raw[:,8]==4)&(raw[:,9]==3)&(raw[:,2]==jnp.minimum(raw[:,1]+1,15)))))
        event_ok=jnp.where(status_damage,generation_ok & clock_ok & status_damage_ok,event_ok)
        entry_spikes_ok=(tables.status_events&~tables.raw_begin&status_identity
            &entry_spikes_callbacks.claimed(tape,index,cursor)
            &entry_spikes_callbacks.ground(raw,tables.entry_spikes_levitate,tables.entry_spikes_flying_type,
                tables.entry_spikes_type_count,tables.entry_spikes_item_count,tables.entry_spikes_boots_item))
        event_ok=jnp.where(entry_spikes,generation_ok&clock_ok&entry_spikes_ok,event_ok)
        direct_move=jnp.clip(raw[:,5],0,tables.base.public_ids.shape[0]-1)
        direct_ok=(tables.status_events & ~tables.raw_begin & status_identity & actor_ok
            & (direct_hit_controls.validate_raw_device(raw)|direct_ko_callbacks.validate_owned_control_device(tape,index,raw)|recoil_callbacks.validate_owned_ko_control(tape,index,raw)|self_destruct_callbacks.owned(tape,index)) & (phase==0)
            & (actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            & (incoming==cursor.physical_active[rows,safe_side])
            & (raw[:,5]<tables.base.public_ids.shape[0]) & (tables.base.public_ids[direct_move]>0))
        event_ok=jnp.where(direct_control,generation_ok & clock_ok & direct_ok,event_ok)
        sub_damage_ok=(tables.status_events & ~tables.raw_begin & cursor.initialized & actor_ok & status_identity
            &(phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(incoming==cursor.physical_active[rows,safe_side])
            &(raw[:,5]<tables.base.public_ids.shape[0])&(tables.base.public_ids[direct_move]>0)
            &substitute_damage.owned_raw(tape,index,raw,cursor))
        event_ok=jnp.where(sub_damage,generation_ok&clock_ok&sub_damage_ok,event_ok)
        no_damage_start=jnp.where(no_damage_facts,index-1,index)
        header=tape.words[rows,jnp.clip(no_damage_start,0,capacity-1)]
        facts=tape.words[rows,jnp.clip(no_damage_start+1,0,capacity-1)]
        no_damage_ok=(tables.status_events & ~tables.raw_begin & actor_ok & status_identity & cursor.initialized
            &(phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(incoming==cursor.physical_active[rows,safe_side])&(no_damage_start>0)&(no_damage_start+1<tape.count)
            &no_damage.valid_raw_pair(header,facts,tables.no_damage))
        # Gen3 runs the actual ability TryHit before natural type immunity.
        # A captured Wonder Guard holder therefore requires its tagged source
        # schema, even though the unchanged kernel ratio is already zero.
        wonder_immunity=(header[:,6]==no_damage.TYPE_IMMUNE)&(tables.wonder_guard_engine_ability>0)&(facts[:,6]==tables.wonder_guard_engine_ability)
        no_damage_ok &= ~wonder_immunity|(tables.wonder_guard_public_ability>0)
        levitate_immunity=header[:,6]==no_damage.LEVITATE_IMMUNE
        levitate_public=tables.trace_public_ability[jnp.clip(facts[:,6],0,len(tables.trace_public_ability)-1)]
        no_damage_ok &= ~levitate_immunity|((levitate_public>0)&no_damage.owned_pair(tape,index,tables.no_damage)[0])
        for offset in (0,1):
            at=jnp.clip(no_damage_start+offset,0,capacity-1)
            no_damage_ok &= (tape.event_turn[rows,at]==turn)&(tape.event_phase[rows,at]==phase)&jnp.all(tape.event_generation[rows,at]==cursor.generation,axis=1)
        event_ok=jnp.where(no_damage_header|no_damage_facts,generation_ok & clock_ok & no_damage_ok,event_ok)
        recoil_ok=(tables.status_events & ~tables.raw_begin & actor_ok & status_identity & (phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(side==1-actor_side)&(incoming==cursor.physical_active[rows,safe_side])
            &(tables.recoil_move>0)&(raw[:,5]==tables.recoil_move)
            &(raw[:,6]>raw[:,7])&(raw[:,7]>0)&(raw[:,8]>=raw[:,6])&(raw[:,8]<=1_000_000)
            &(raw[:,9]>=0)&(raw[:,9]<=6))
        recoil_ok |= tables.status_events & ~tables.raw_begin & actor_ok & status_identity & (phase==0) & (actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)]) & (side==1-actor_side)&(incoming==cursor.physical_active[rows,safe_side]) & recoil_callbacks.owned_silent(tape,index,cursor)
        event_ok=jnp.where(recoil,generation_ok & clock_ok & recoil_ok,event_ok)
        rough_ok=(tables.status_events&~tables.raw_begin&actor_ok&status_identity&(phase==0)
            &rough_callbacks.owned(tape,index,cursor))
        event_ok=jnp.where(rough,generation_ok&clock_ok&rough_ok,event_ok)
        belly_a,belly_z,belly_owned=belly_callbacks.pair(tape,index)
        belly_ok=(tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)&belly_owned
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)]))
        event_ok=jnp.where(belly,generation_ok&clock_ok&belly_ok,event_ok)
        leech_ok=(tables.status_events&~tables.raw_begin&actor_ok&status_identity&cursor.initialized
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(incoming==cursor.physical_active[rows,safe_side])&leech_callbacks.owned(tape,index))
        event_ok=jnp.where(leech,generation_ok&clock_ok&leech_ok,event_ok)
        self_move_ok=tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&self_move_callbacks.owned(tape,index)
        event_ok=jnp.where(self_move,generation_ok&clock_ok&self_move_ok,event_ok)
        event_ok=jnp.where(self_destruct,generation_ok&clock_ok&self_destruct_callbacks.owned(tape,index),event_ok)
        bell_status=heal_bell&(raw[:,5]==heal_bell_callbacks.STATUS)
        bell_cure=bell_status&(raw[:,6]>0)&(raw[:,7]==0)
        bell_sp=tables.heal_bell_soundproof[jnp.clip(raw[:,9],0,len(tables.heal_bell_soundproof)-1)]
        bell_immune=bell_status&(raw[:,4]==raw[:,2])&bell_sp
        bell_member_slot=cursor.physical_to_public[rows,jnp.clip(raw[:,3],0,1),jnp.clip(raw[:,4],0,5)]
        bell_visible=bell_cure|bell_immune
        bell_ok=tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&heal_bell_callbacks.owned(tape,index)
        bell_ok &= (~bell_visible|((bell_member_slot>=0)&(bell_member_slot<6)))&(~bell_immune|(tables.heal_bell_public_soundproof>0))
        if tape.execution is not None and tape.execution.risks is not None:
            bell_ok &= (tables.heal_bell_move==tape.execution.risks.tables.heal_bell_move)&jnp.all(tables.heal_bell_soundproof==tape.execution.risks.tables.heal_bell_soundproof)
        else:bell_ok &= False
        event_ok=jnp.where(heal_bell,generation_ok&clock_ok&bell_ok,event_ok)
        encore_ok=(tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&encore_callbacks.owned(tape,index))
        event_ok=jnp.where(encore,generation_ok&clock_ok&encore_ok,event_ok)
        sleep_blocked_ok=(tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&sleep_blocked_callbacks.owned(tape,index))
        event_ok=jnp.where(sleep_blocked,generation_ok&clock_ok&sleep_blocked_ok,event_ok)
        talk_ok=tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&sleep_talk_owned
        event_ok=jnp.where(sleep_talk,generation_ok&clock_ok&talk_ok,event_ok)
        bm_ok=tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)&(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&before_move_blocked_callbacks.owned(tape,index)
        event_ok=jnp.where(before_move_blocked,generation_ok&clock_ok&bm_ok,event_ok)
        intimidate_ok=(tables.status_events & ~tables.raw_begin & actor_ok & status_identity
            &~cursor.initialized&(turn==0)&(phase==0)&(actor_physical==0)&(incoming==0)
            &(actor_side>=0)&(actor_side<2)&(side==1-actor_side)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&(incoming==cursor.physical_active[rows,safe_side])
            &jnp.all(cursor.generation==0,axis=1)&(tables.intimidate_engine_ability>0)&(tables.intimidate_public_ability>0)
            &(raw[:,5]==tables.intimidate_engine_ability)&(raw[:,6]==0)&(raw[:,7]==-1)&jnp.all(raw[:,8:]==0,axis=1))
        ordinary_intimidate=(tables.status_events&~tables.raw_begin&actor_ok&status_identity
            &(tables.intimidate_engine_ability>0)&(tables.intimidate_public_ability>0)
            &(raw[:,5]==tables.intimidate_engine_ability)
            &intimidate_callbacks.ordinary_owned(tape,index,cursor))
        event_ok=jnp.where(intimidate,generation_ok & clock_ok & (intimidate_ok|ordinary_intimidate),event_ok)
        calm_mind_ok=(tables.status_events & ~tables.raw_begin & actor_ok & cursor.initialized
            &(phase==0)&(actor_side>=0)&(actor_side<2)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(tables.calm_mind_move>0)&(raw[:,3]==tables.calm_mind_move)
            &((raw[:,4]==2)|(raw[:,4]==3))&(raw[:,5]>=-6)&(raw[:,5]<6)
            &(raw[:,6]==raw[:,5]+1)&((raw[:,7]==0)|(raw[:,7]==1))&jnp.all(raw[:,8:]==0,axis=1))
        event_ok=jnp.where(calm_mind,generation_ok & clock_ok & calm_mind_ok,event_ok)
        pinch_ok=(tables.status_events&~tables.raw_begin&actor_ok&status_identity&cursor.initialized&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &pinch_callbacks.owned(tape,index))
        event_ok=jnp.where(pinch,generation_ok&clock_ok&pinch_ok,event_ok)
        bulk_ok=(tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &bulk_up_callbacks.owned_pair(tape,index))
        event_ok=jnp.where(bulk_up,generation_ok&clock_ok&bulk_ok,event_ok)
        self_boost_ok=(tables.status_events&~tables.raw_begin&actor_ok&cursor.initialized&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&self_boost_callbacks.owned(tape,index))
        event_ok=jnp.where(self_boost,generation_ok&clock_ok&self_boost_ok,event_ok)
        event_ok=jnp.where(baton,generation_ok&clock_ok&baton_callbacks.owned(tape,index),event_ok)
        swords_dance_ok=(tables.status_events & ~tables.raw_begin & actor_ok & cursor.initialized
            &(phase==0)&(actor_side>=0)&(actor_side<2)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(tables.swords_dance_move>0)&(raw[:,3]==tables.swords_dance_move)
            &(raw[:,4]>=-6)&(raw[:,4]<=4)&(raw[:,5]==raw[:,4]+2)
            &((raw[:,6]==0)|(raw[:,6]==1))&jnp.all(raw[:,7:]==0,axis=1))
        event_ok=jnp.where(swords_dance,generation_ok & clock_ok & swords_dance_ok,event_ok)

        spike_side=raw[:,1];spike_safe=jnp.clip(spike_side,0,1)
        spikes_ok=(tables.status_events & ~tables.raw_begin & cursor.initialized & (phase==0) & (turn>0)
            &(spike_side>=0)&(spike_side<2)&(raw[:,2]==cursor.physical_active[rows,spike_safe])
            &(raw[:,3]==tables.spikes_move)&(tables.spikes_move>0)&(raw[:,4]==1-spike_side)
            &(raw[:,5]>=0)&(raw[:,5]<3)&(raw[:,6]==raw[:,5]+1)
            &((raw[:,7]==0)|(raw[:,7]==1))&jnp.all(raw[:,8:]==0,axis=1))
        event_ok=jnp.where(spikes,generation_ok & clock_ok & spikes_ok,event_ok)
        pressure_ok=(tables.status_events & ~tables.raw_begin & status_identity & (phase==0)
            & (incoming==cursor.physical_active[rows,safe_side])
            & jnp.all(raw[:,1:3]==-1,axis=1) & jnp.all(raw[:,6:]==0,axis=1)
            & (tables.pressure_engine_ability>0) & (tables.pressure_public_ability>0)
            & (raw[:,5]==tables.pressure_engine_ability))
        event_ok=jnp.where(pressure,generation_ok & clock_ok & pressure_ok,event_ok)
        donor_slot,donor_enc,donor_ok=(v[:,0] for v in base._resolve(_registry(cursor),raw[:,5,None],raw[:,6,None]))
        trace_ok=(tables.status_events & ~tables.raw_begin & status_identity & donor_ok & (phase==0)
            &(incoming==cursor.physical_active[rows,safe_side])&(raw[:,5]==1-side)
            &(raw[:,6]==cursor.physical_active[rows,1-safe_side])
            &(tables.trace_engine_ability>0)&(raw[:,1]==tables.trace_engine_ability)
            &(raw[:,2]>0)&(raw[:,2]<tables.trace_public_ability.shape[0])
            &(tables.trace_public_ability[jnp.clip(raw[:,2],0,tables.trace_public_ability.shape[0]-1)]>0)
            &(raw[:,7]==raw[:,2])&jnp.all(raw[:,8:]==0,axis=1))
        event_ok=jnp.where(trace_copy,generation_ok & clock_ok & trace_ok,event_ok)
        substitute_ok=(tables.status_events & ~tables.raw_begin & status_identity & (phase==0)
            & (incoming==cursor.physical_active[rows,safe_side])&(raw[:,1]==0)&(raw[:,2]>0)
            & (raw[:,5]>raw[:,6])&(raw[:,6]>0)&(raw[:,7]>=raw[:,5])&(raw[:,7]<=1_000_000)
            & (raw[:,2]==raw[:,5]-raw[:,6])&(raw[:,8]>=0)&(raw[:,8]<=6)
            & (tables.substitute_move>0)&(raw[:,9]==tables.substitute_move))
        substitute_failure=substitute_callbacks.owned_failure(tape,index)
        event_ok=jnp.where(substitute,generation_ok & clock_ok & (substitute_ok|substitute_failure),event_ok)
        toxic_ok=(tables.status_events & ~tables.raw_begin & actor_ok & status_identity & (phase==0)
            & (actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            & (incoming==cursor.physical_active[rows,safe_side])
            & (((toxic_callbacks.supported_raw(raw)|toxic_callbacks.owned_substitute_failure(tape,index)|toxic_callbacks.owned_existing_status(tape,index))&(tables.toxic_move>0)&(raw[:,5]==tables.toxic_move))
                |((thunder_callbacks.supported_raw(raw)|thunder_callbacks.owned_substitute_failure(tape,index)|thunder_callbacks.owned_ground_immunity(tape,index))&(tables.thunder_wave_move>0)&(raw[:,5]==tables.thunder_wave_move))
                |(powder_callbacks.supported_raw(raw,primary_sleep=True)&(raw[:,5]>0)&(raw[:,5]<len(tables.primary_sleep_moves))&tables.primary_sleep_moves[jnp.clip(raw[:,5],0,len(tables.primary_sleep_moves)-1)])))
        toxic_ok |= (tables.status_events&~tables.raw_begin&actor_ok&status_identity&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&(incoming==cursor.physical_active[rows,safe_side])
            &plain_primary_callbacks.owned_pair(tape,index))
        toxic_ok |= (tables.status_events&~tables.raw_begin&actor_ok&status_identity&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&(incoming==cursor.physical_active[rows,safe_side])
            &powder_callbacks.owned_miss_device(tape,index))
        event_ok=jnp.where(toxic,generation_ok & clock_ok & toxic_ok,event_ok)
        secondary_cont=kind==secondary_callbacks.SECONDARY_STATUS
        following=tape.words[:,jnp.minimum(index+1,capacity-1)]
        secondary_write=applied&(following[:,0]==secondary_callbacks.SECONDARY_STATUS)
        secondary_pair=secondary_callbacks.owned_pair(tape,jnp.where(secondary_cont,index,index+1))
        secondary_ok=(tables.status_events&~tables.raw_begin&(phase==0)&secondary_pair
            &jnp.where(secondary_cont,actor_ok&status_identity
                &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
                &(incoming==cursor.physical_active[rows,safe_side]),status_ok))
        event_ok=jnp.where(secondary_cont|secondary_write,generation_ok&clock_ok&secondary_ok,event_ok)
        color_ok=(tables.status_events&~tables.raw_begin&actor_ok&status_identity&(phase==0)&cursor.initialized
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(incoming==cursor.physical_active[rows,safe_side])&color_callbacks.owned(tape,index))
        event_ok=jnp.where(color_change,generation_ok&clock_ok&color_ok,event_ok)
        flash_end_ok=(tables.status_events&~tables.raw_begin&status_identity&(phase==0)
            &(incoming==cursor.physical_active[rows,safe_side])&cursor.initialized
            &flash_end.owned(tape,index,tables.no_damage))
        event_ok=jnp.where(flash_ended,generation_ok&clock_ok&flash_end_ok,event_ok)
        protect_identity=(tables.status_events&~tables.raw_begin&actor_ok&status_identity&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])
            &(incoming==cursor.physical_active[rows,safe_side]))
        event_ok=jnp.where(protect_attempt,generation_ok&clock_ok&protect_identity&protect_callbacks.owned_attempt(tape,index),event_ok)
        event_ok=jnp.where(protect_block,generation_ok&clock_ok&protect_identity&protect_callbacks.owned_block(tape,index),event_ok)
        self_heal_ok=(tables.status_events&~tables.raw_begin&status_identity&(phase==0)
            &(incoming==cursor.physical_active[rows,safe_side])&self_heal_callbacks.owned(tape,index))
        event_ok=jnp.where(self_heal,generation_ok&clock_ok&self_heal_ok,event_ok)
        prefix_ok=(tables.status_events&~tables.raw_begin&clock_payload&(index==0)
            &jnp.all(cursor.established,axis=1)&jnp.all(details==0,axis=1))
        event_ok=jnp.where(turnloop_prefix,generation_ok&clock_ok&prefix_ok,event_ok)
        focus_ok=(tables.status_events&~tables.raw_begin&actor_ok&(phase==0)
            &(actor_physical==cursor.physical_active[rows,jnp.clip(actor_side,0,1)])&focus_callbacks.owned(tape,index))
        event_ok=jnp.where(focus,generation_ok&clock_ok&focus_ok,event_ok)
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
        public_kind = jnp.where(arrival,jnp.where(baton_arrival,int(inputs.Kind.BATON_PASS),int(inputs.Kind.SWITCH)), jnp.where(kind == life.TURN,
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
        bell_payload=jnp.stack((bell_member_slot+1,z,z,z,z,z,z,z),axis=1)
        bell_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.STATUS)),semantic_sequence,turn,phase,absent,absent,actor_side,bell_member_slot),axis=1),bell_payload),axis=1)
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
        spikes_payload=jnp.stack((jnp.full_like(z,public_field.SPIKES),jnp.ones_like(z),z,z,z,z,z,z),axis=1)
        spikes_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.FIELD)),
            semantic_sequence,turn,phase,absent,absent,raw[:,4],absent),axis=1),spikes_payload),axis=1)
        semantic=jnp.where(spikes[:,None],spikes_semantic,semantic)
        failed_payload=jnp.stack((actor_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        failed_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),
            semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),failed_payload),axis=1)
        semantic=jnp.where((failed_weather|self_move)[:,None],failed_semantic,semantic)
        sm_code=raw[:,3]&~self_move_callbacks.SECONDARY
        sm_status=self_move&(sm_code==self_move_callbacks.REST_STATUS)
        sm_heal=self_move&(sm_code==self_move_callbacks.REST_HEAL)
        sm_status_payload=jnp.stack((actor_enc,jnp.ones_like(z),z,z,z,z,z,z),axis=1)
        sm_status_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.STATUS)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),sm_status_payload),axis=1)
        sm_own=state.owner==actor_side
        sm_percent=(100*raw[:,6]+jnp.maximum(1,raw[:,7])-1)//jnp.maximum(1,raw[:,7])
        sm_heal_payload=jnp.stack((actor_enc,jnp.where(sm_own,raw[:,6],sm_percent),jnp.where(sm_own,raw[:,7],100),jnp.ones_like(z),z,z,z,z),axis=1)
        sm_heal_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),sm_heal_payload),axis=1)
        semantic=jnp.where(sm_status[:,None],sm_status_semantic,semantic)
        semantic=jnp.where(sm_heal[:,None],sm_heal_semantic,semantic)
        chip_percent=(100*raw[:,6]+jnp.maximum(1,raw[:,7])-1)//jnp.maximum(1,raw[:,7])
        chip_percent=jnp.where((raw[:,6]<raw[:,7])&(chip_percent==100),99,chip_percent)
        chip_payload=jnp.stack((status_enc,jnp.where(own,raw[:,6],chip_percent),
            jnp.where(own,raw[:,7],100),jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(raw[:,8],0,6)],
            z,jnp.where(leftovers_heal|self_heal,0,1),jnp.where(leftovers_heal|substitute|self_heal,0,1),entry_spikes.astype(jnp.int32)),axis=1)  # heal tags are not damage metadata
        lethal_status=status_damage&(raw[:,6]==0)
        chip_payload=chip_payload.at[:,2].set(jnp.where(lethal_status,0,chip_payload[:,2])).at[:,3].set(jnp.where(lethal_status,0,chip_payload[:,3])).at[:,4].set(lethal_status.astype(jnp.int32))
        chip_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),
            semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),chip_payload),axis=1)
        semantic=jnp.where((weather_chip|leftovers_heal|status_damage|substitute|(self_heal&~heal_failure)|entry_spikes)[:,None],chip_semantic,semantic)
        sub_fail_payload=jnp.stack((status_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        sub_fail_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),sub_fail_payload),axis=1)
        semantic=jnp.where((substitute&substitute_failure)[:,None],sub_fail_semantic,semantic)
        heal_fail_payload=jnp.stack((status_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        heal_fail_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),heal_fail_payload),axis=1)
        semantic=jnp.where(heal_failure[:,None],heal_fail_semantic,semantic)
        recoil_own=state.owner==actor_side
        recoil_percent=(100*raw[:,7]+jnp.maximum(1,raw[:,8])-1)//jnp.maximum(1,raw[:,8])
        recoil_percent=jnp.where((raw[:,7]<raw[:,8])&(recoil_percent==100),99,recoil_percent)
        recoil_payload=jnp.stack((actor_enc,jnp.where(recoil_own,raw[:,7],recoil_percent),jnp.where(recoil_own,raw[:,8],100),
            jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(raw[:,9],0,6)],z,jnp.ones_like(z),jnp.ones_like(z),z),axis=1)
        recoil_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),recoil_payload),axis=1)
        semantic=jnp.where((recoil|rough)[:,None],recoil_semantic,semantic)
        belly_percent=(100*raw[:,6]+jnp.maximum(1,raw[:,7])-1)//jnp.maximum(1,raw[:,7])
        belly_percent=jnp.where((raw[:,6]<raw[:,7])&(belly_percent==100),99,belly_percent)
        belly_own=state.owner==actor_side
        belly_hp_payload=jnp.stack((actor_enc,jnp.where(belly_own,raw[:,6],belly_percent),jnp.where(belly_own,raw[:,7],100),
            jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(raw[:,8],0,6)],z,jnp.ones_like(z),z,z),axis=1)
        belly_hp_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),belly_hp_payload),axis=1)
        belly_boost_payload=jnp.stack((actor_enc,z,raw[:,6],jnp.full_like(z,public_mon_state.SET),z,z,z,z),axis=1)
        belly_boost_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),belly_boost_payload),axis=1)
        semantic=jnp.where((belly&(raw[:,4]==0))[:,None],belly_hp_semantic,semantic)
        semantic=jnp.where((belly&(raw[:,4]==1))[:,None],belly_boost_semantic,semantic)
        leech_start=leech&(raw[:,5]==leech_callbacks.START)
        leech_heal=leech&(raw[:,5]==leech_callbacks.HEAL)
        leech_slot=jnp.where(leech_heal,actor_slot,status_slot);leech_enc=jnp.where(leech_heal,actor_enc,status_enc);leech_side=jnp.where(leech_heal,actor_side,side)
        leech_percent=(100*raw[:,7]+jnp.maximum(1,raw[:,8])-1)//jnp.maximum(1,raw[:,8]);leech_percent=jnp.where((raw[:,7]<raw[:,8])&(leech_percent==100),99,leech_percent)
        leech_own=state.owner==leech_side
        leech_hp=jnp.stack((leech_enc,jnp.where(leech_own,raw[:,7],leech_percent),jnp.where(leech_own,raw[:,8],100),jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(raw[:,9],0,6)],z,(~leech_heal).astype(jnp.int32),(~leech_heal).astype(jnp.int32),z),axis=1)
        leech_hp_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.CONDITION)),semantic_sequence,turn,phase,absent,absent,leech_side,leech_slot),axis=1),leech_hp),axis=1)
        leech_start_payload=jnp.stack((status_enc,jnp.full_like(z,public_active.ADD),jnp.full_like(z,public_active.NAMES.index('leechseed')),z,z,z,z,z),axis=1)
        leech_start_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),leech_start_payload),axis=1)
        semantic=jnp.where((leech&~leech_start)[:,None],leech_hp_semantic,semantic)
        semantic=jnp.where(leech_start[:,None],leech_start_semantic,semantic)
        encore_payload=jnp.stack((actor_enc,jnp.where(raw[:,3]==encore_callbacks.END,public_active.REMOVE,public_active.ADD),jnp.full_like(z,public_active.NAMES.index('encore')),z,z,z,z,z),axis=1)
        encore_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),encore_payload),axis=1)
        semantic=jnp.where(encore[:,None],encore_semantic,semantic)
        encore_fail=encore&(raw[:,3]==encore_callbacks.FAIL_NO_LAST)
        encore_caller=1-actor_side;encore_caller_slot=cursor.public_slot[rows,jnp.clip(encore_caller,0,1)];encore_caller_enc=cursor.encounter[rows,jnp.clip(encore_caller,0,1)]
        encore_failure=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,encore_caller,encore_caller_slot),axis=1),jnp.stack((encore_caller_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)),axis=1)
        semantic=jnp.where(encore_fail[:,None],encore_failure,semantic)
        from . import public_sleep
        sleep_payload=jnp.stack((actor_enc,jnp.full_like(z,public_sleep.CANT_SLEEP),z,z,z,z,z,z),axis=1)
        sleep_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.SLEEP)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),sleep_payload),axis=1)
        semantic=jnp.where((sleep_blocked|(sleep_talk&(raw[:,3]==sleep_talk_callbacks.CLOCK)))[:,None],sleep_semantic,semantic)
        talk_payload=jnp.stack((actor_enc,tables.base.public_ids[jnp.clip(raw[:,4],0,len(tables.base.public_ids)-1)],z,z,z,z,z,z),axis=1)
        talk_move=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.MOVE)),semantic_sequence,turn,phase,actor_side,actor_slot,actor_side,actor_slot),axis=1),talk_payload),axis=1)
        semantic=jnp.where((sleep_talk&(raw[:,3]==sleep_talk_callbacks.CALL))[:,None],talk_move,semantic)
        semantic=semantic.at[:,10].set(jnp.where(sleep_talk_child,1,semantic[:,10])).at[:,11].set(jnp.where(sleep_talk_child,1,semantic[:,11]))
        bm_payload=jnp.stack((actor_enc,jnp.full_like(z,public_active.STALL_BREAK),z,z,z,z,z,z),axis=1)
        bm_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),bm_payload),axis=1)
        semantic=jnp.where(before_move_blocked[:,None],bm_semantic,semantic)
        intimidate_payload=jnp.stack((status_enc,z,jnp.full_like(z,-1),jnp.full_like(z,public_mon_state.DELTA),jnp.ones_like(z),z,z,z),axis=1)
        intimidate_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,
            turn,phase,absent,absent,side,status_slot),axis=1),intimidate_payload),axis=1)
        semantic=jnp.where(intimidate[:,None],intimidate_semantic,semantic)
        calm_mind_payload=jnp.stack((actor_enc,raw[:,4],jnp.ones_like(z),jnp.full_like(z,public_mon_state.DELTA),z,z,z,z),axis=1)
        calm_mind_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),calm_mind_payload),axis=1)
        semantic=jnp.where(calm_mind[:,None],calm_mind_semantic,semantic)
        bulk_payload=jnp.stack((actor_enc,raw[:,4],jnp.ones_like(z),jnp.full_like(z,public_mon_state.DELTA),z,z,z,z),axis=1)
        bulk_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),bulk_payload),axis=1)
        semantic=jnp.where(bulk_up[:,None],bulk_semantic,semantic)
        self_boost_payload=jnp.stack((actor_enc,raw[:,4],raw[:,6]-raw[:,5],jnp.full_like(z,public_mon_state.DELTA),z,z,z,z),axis=1)
        self_boost_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),self_boost_payload),axis=1)
        semantic=jnp.where(self_boost[:,None],self_boost_semantic,semantic)
        baton_semantic=jnp.stack((jnp.full_like(z,int(inputs.Kind.NONPROC)),semantic_sequence,turn,phase,absent,absent,absent,absent,z,z,z,z,z,z,z,z),axis=1)
        semantic=jnp.where(baton[:,None],baton_semantic,semantic)
        pinch_payload=jnp.stack((actor_enc,raw[:,6],jnp.ones_like(z),jnp.full_like(z,public_mon_state.DELTA),z,z,z,z),axis=1)
        pinch_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),pinch_payload),axis=1)
        semantic=jnp.where(pinch[:,None],pinch_semantic,semantic)
        color_payload=jnp.stack((status_enc,jnp.full_like(z,2),tables.color_categories[jnp.clip(raw[:,8],0,17)],z,z,z,z,z),axis=1)
        color_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.TYPES)),semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),color_payload),axis=1)
        semantic=jnp.where(color_change[:,None],color_semantic,semantic)
        swords_dance_payload=jnp.stack((actor_enc,z,jnp.full_like(z,2),jnp.full_like(z,public_mon_state.DELTA),z,z,z,z),axis=1)
        swords_dance_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.BOOST)),semantic_sequence,
            turn,phase,absent,absent,actor_side,actor_slot),axis=1),swords_dance_payload),axis=1)
        semantic=jnp.where(swords_dance[:,None],swords_dance_semantic,semantic)

        # Annotation-only lines still enter every original raw consumer. Their
        # main NONPROC word has no public target; the full adapter resolves the
        # lexical target independently from this literal raw receipt.
        direct_semantic=jnp.stack((jnp.full_like(z,int(inputs.Kind.NONPROC)),semantic_sequence,
            turn,phase,absent,absent,absent,absent,z,absent,absent,z,z,z,z,z),axis=1)
        semantic=jnp.where((direct_control|toxic|no_damage_header|no_damage_facts|self_destruct|focus)[:,None],direct_semantic,semantic)
        semantic=jnp.where(heal_bell[:,None],jnp.where(bell_cure[:,None],bell_semantic,direct_semantic),semantic)
        semantic=jnp.where((toxic&toxic_callbacks.substitute_failure_raw(raw))[:,None],failed_semantic,semantic)
        toxic_existing=toxic&toxic_callbacks.existing_status_raw(raw);toxic_same=toxic_existing&(raw[:,7]==4)
        toxic_failure_payload=jnp.stack((status_enc,jnp.full_like(z,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        toxic_failure_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),toxic_failure_payload),axis=1)
        semantic=jnp.where(toxic_existing[:,None],jnp.where(toxic_same[:,None],toxic_failure_semantic,failed_semantic),semantic)
        pressure_payload=jnp.stack((status_enc,jnp.full_like(z,tables.pressure_public_ability),
            jnp.ones_like(z),z,z,z,z,z),axis=1)
        pressure_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ABILITY)),
            semantic_sequence,turn,phase,absent,absent,side,status_slot),axis=1),pressure_payload),axis=1)
        # Keep one internal semantic per receipt for exact raw-index alignment.
        # FULL projection alone enforces the source's owner-only visibility.
        semantic=jnp.where(pressure[:,None],pressure_semantic,semantic)
        trace_payload=jnp.stack((status_enc,tables.trace_public_ability[jnp.clip(raw[:,2],0,tables.trace_public_ability.shape[0]-1)],donor_enc,jnp.ones_like(z),z,z,z,z),axis=1)
        trace_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.TRACE)),semantic_sequence,
            turn,phase,side,status_slot,raw[:,5],donor_slot),axis=1),trace_payload),axis=1)
        semantic=jnp.where(trace_copy[:,None],trace_semantic,semantic)
        # The main fact of the exact two-line original FULL template is its
        # timestamp control. The preceding blank retains its own six events.
        prefix_payload=jnp.broadcast_to(jnp.array([public_toxic.TIMESTAMP,0,0,-1,0,1,0,0],jnp.int32),(b,8))
        prefix_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.TOXIC)),
            semantic_sequence,turn,phase,absent,absent,absent,absent),axis=1),prefix_payload),axis=1)
        semantic=jnp.where(turnloop_prefix[:,None],prefix_semantic,semantic)
        secondary_semantic=jnp.stack((jnp.full_like(z,int(inputs.Kind.NONPROC)),semantic_sequence,turn,phase,
            absent,absent,absent,absent,z,z,z,z,z,z,z,z),axis=1)
        semantic=jnp.where(secondary_cont[:,None],secondary_semantic,semantic)
        semantic=jnp.where((flash_ended|protect_block)[:,None],direct_semantic,semantic)
        semantic=jnp.where(sub_damage[:,None],secondary_semantic,semantic)
        protect_payload=jnp.stack((actor_enc,jnp.where((raw[:,6]&protect_callbacks.SUCCESS)!=0,public_active.STALL_SUCCESS,public_active.STALL_FAIL),z,z,z,z,z,z),axis=1)
        protect_semantic=jnp.concatenate((jnp.stack((jnp.full_like(z,int(inputs.Kind.ACTIVE)),semantic_sequence,turn,phase,absent,absent,actor_side,actor_slot),axis=1),protect_payload),axis=1)
        semantic=jnp.where(protect_attempt[:,None],protect_semantic,semantic)
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


def project_frame(state, frame, tables, *, weather_hp=None, leftovers=None,status_residual=None):
    """Device connection from an actual lifecycle result; never reads its state."""
    if type(frame) is not life.LifecycleResult:
        raise ValueError('actual lifecycle result required')
    return project_device(state, frame.callbacks, frame.before_registry, frame.after_registry, tables,
        weather_hp=weather_hp,leftovers=leftovers,status_residual=status_residual)


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
