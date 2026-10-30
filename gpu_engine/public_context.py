"""Compose public move, ability and optional item evidence.

A separate hard mask retains nonreplaceable facts. Current original-ability and
assignment-only item constraints are rebuilt on that basis, allowing corrections
and item mutations to recover variants while preserving moves, exclusions and pins.
This is incremental consumer composition, not complete raw protocol coverage.
"""
from typing import NamedTuple
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from . import public_abilities as ability, public_moves as move, public_mon_state as mon, public_items as item, public_field as field_api
from . import public_item_callbacks as item_callbacks
from .public_moves import _bool
from . import public_item_nonproc as nonproc_api,public_nonproc_protocol as nonproc_protocol,public_nonproc_refresh as nonproc_refresh
from . import public_mudshot as mudshot_api,public_mudshot_protocol as mudshot_protocol,public_mudshot_refresh as mudshot_refresh
from . import public_active as active_api
from . import public_damage as damage_api
from . import public_types as type_api
from . import public_sleep as sleep_api
from . import public_toxic as toxic_api, public_toxic_protocol as toxic_protocol
from . import public_tendency as tendency_api, public_tendency_protocol as tendency_protocol
from . import public_belief_refresh as refresh_api, public_refresh_stream as refresh_stream
from . import public_belief_lifecycle as lifecycle_api
from . import public_sleep_clause as clause_api
from . import public_switch_evidence as switch_api, public_belief as belief_api, public_belief_snapshot as snapshot_api


class ContextTables(NamedTuple):
    abilities: ability.AbilityTables
    moves: move.MoveTables
    ability_predicate: jax.Array
    items: item.ItemTables | None = None
    item_predicate: jax.Array | None = None
    item_narrowing: jax.Array | None = None
    binding: jax.Array | None = None
    field: field_api.FieldTables | None = None
    active: active_api.ActiveTables | None = None
    damage: damage_api.DamageTables | None = None
    types: type_api.TypeTables | None = None
    sleep: sleep_api.SleepTables | None = None
    toxic: toxic_api.ToxicTables | None = None
    tendency: tendency_api.TendencyTables | None = None
    refresh: refresh_api.RefreshTables | None = None
    lifecycle: lifecycle_api.LifecycleTables | None = None
    sleep_clause: clause_api.Tables | None = None
    switches: switch_api.Tables | None = None
    nonproc: nonproc_api.NonprocTables | None = None
    mudshot: mudshot_api.Tables | None = None
    mudshot_move_id: jax.Array | None = None


class ContextState(NamedTuple):
    abilities: ability.AbilityState
    moves: move.MoveState
    base_hard: jax.Array
    base_has_evidence: jax.Array
    items: item.ItemState | None = None
    binding: jax.Array | None = None
    field: field_api.FieldState | None = None
    active: active_api.ActiveState | None = None
    damage: damage_api.DamageState | None = None
    types: type_api.TypeState | None = None
    sleep: sleep_api.SleepState | None = None
    toxic: toxic_api.ToxicState | None = None
    toxic_raw_pending: jax.Array | None = None
    tendency: tendency_api.TendencyState | None = None
    tendency_raw_pending: jax.Array | None = None
    refresh: refresh_stream.StreamState | None = None
    lifecycle: lifecycle_api.LifecycleState | None = None
    sleep_clause: clause_api.State | None = None
    switches: switch_api.State | None = None
    nonproc: nonproc_api.NonprocState | None = None
    nonproc_binding: jax.Array | None = None
    mudshot: mudshot_api.State | None = None


def compile_context(source, abilities, moves, *, items=None, item_belief_narrowing=False, field_catalog=None, active_catalog=None, damage_tables=None, type_catalog=None, sleep_tables=None, toxic_tables=None, tendency_tables=None, refresh_tables=None, lifecycle_tables=None, clause_tables=None, switch_tables=None, nonproc_tables=None,mudshot_tables=None,mudshot_move_id=0):
    if lifecycle_tables is not None:
        if (lifecycle_tables.absorb_heal.shape!=(len(abilities.names),)
                or lifecycle_tables.absorb_heal.dtype!=jnp.bool_):
            raise ValueError('HP absorb semantics require the frozen B-less boolean ability namespace')
    if mudshot_tables is not None:
        if nonproc_tables is None or sleep_tables is None or refresh_tables is None:
            raise ValueError('Mud Shot requires ordered non-proc/sleep/source consumers')
        if type(mudshot_move_id) is not int or not 0<=mudshot_move_id<len(moves.names):
            raise ValueError('Mud Shot requires its frozen move ID')
        for actual,expected in ((mudshot_tables.source_binding,source.device.binding),
                (mudshot_tables.species_binding,sleep_tables.species_binding),
                (mudshot_tables.ability_binding,abilities.device.binding)):
            sleep_api._binding(actual);sleep_api._binding(expected)
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):raise ValueError('Mud Shot context namespace mismatch')
    if nonproc_tables is not None:
        if items is None or sleep_tables is None or refresh_tables is None:
            raise ValueError('non-proc evidence requires ordered item/sleep/source consumers')
        for actual,expected in ((nonproc_tables.source_binding,source.device.binding),
                (nonproc_tables.species_binding,sleep_tables.species_binding),
                (nonproc_tables.item_binding,items.device.binding),(nonproc_tables.sleep_binding,sleep_tables.binding)):
            sleep_api._binding(actual);sleep_api._binding(expected)
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):raise ValueError('non-proc context namespace mismatch')
    if switch_tables is not None:
        if sleep_tables is None or lifecycle_tables is None or refresh_tables is None:
            raise ValueError('pending switch evidence requires ordered sleep/lifecycle/refresh consumers')
        for actual,expected in ((switch_tables.species_binding,sleep_tables.species_binding),
                (switch_tables.source_binding,source.device.binding),(switch_tables.ability_binding,abilities.device.binding)):
            sleep_api._binding(actual);sleep_api._binding(expected)
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):
                raise ValueError('pending switch context namespace mismatch')
    if clause_tables is not None:
        if sleep_tables is None or lifecycle_tables is None or refresh_tables is None:
            raise ValueError('sleep clause requires ordered lifecycle/sleep source consumers')
        for actual,expected in ((clause_tables.sleep_binding,sleep_tables.binding),
                (clause_tables.species_binding,sleep_tables.species_binding),
                (clause_tables.vocabulary_binding,sleep_tables.vocabulary_binding)):
            sleep_api._binding(actual);sleep_api._binding(expected)
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):
                raise ValueError('sleep clause context namespace mismatch')
    if lifecycle_tables is not None:
        if refresh_tables is None or sleep_tables is None:raise ValueError('lifecycle composition requires ordered source refresh and sleep')
        for actual,expected in ((lifecycle_tables.sleep_binding,sleep_tables.binding),
                (lifecycle_tables.species_binding,sleep_tables.species_binding),
                (lifecycle_tables.ability_binding,abilities.device.binding),
                (lifecycle_tables.move_binding,moves.device.binding)):
            lifecycle_api._binding(actual);lifecycle_api._binding(expected)
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):raise ValueError('lifecycle context namespace mismatch')
    if refresh_tables is not None:
        if items is None or sleep_tables is None or tendency_tables is None:
            raise ValueError('ordered summary refresh requires item, sleep and final raw-frame consumers')
        refresh_stream.validate_source(source,pending_switch=switch_tables is not None)
        refresh_api._tables(refresh_tables,source.device,moves.device)
        if not np.array_equal(np.asarray(refresh_tables.source_binding),np.asarray(source.device.binding)) or not np.array_equal(np.asarray(refresh_tables.move_binding),np.asarray(moves.device.binding)):
            raise ValueError('ordered summary refresh requires exact source/move namespaces')
    if tendency_tables is not None and field_catalog is None:
        raise ValueError('tendency composition requires ordered raw/field context')
    if type_catalog is not None and field_catalog is None:
        raise ValueError('live-type composition requires the ordered field context')
    if toxic_tables is not None and field_catalog is None:
        raise ValueError('Toxic composition requires ordered raw/field context')
    if type(item_belief_narrowing) is not bool:
        raise ValueError("item narrowing must be a frozen boolean")
    if items is None and field_catalog is not None:
        raise ValueError("field composition requires the raw item-enabled context")
    if active_catalog is not None and field_catalog is None:
        raise ValueError("active composition requires the ordered field context")
    if damage_tables is not None and field_catalog is None:
        raise ValueError("damage composition requires the ordered field context")
    if sleep_tables is not None and field_catalog is None:
        raise ValueError('sleep composition requires ordered field context')
    if items is None and item_belief_narrowing:
        raise ValueError("item narrowing requires item tables")
    for binding in (abilities.device.source_binding, moves.device.source_binding):
        if not np.array_equal(np.asarray(binding), np.asarray(source.device.binding)):
            raise ValueError("public context requires the same frozen source")
    flags = np.zeros(source.device.predicates.shape[0], bool)
    for (kind, _), index in source.predicate_ids.items():
        if kind == "ability": flags[index] = True
    if items is None:
        return ContextTables(abilities.device, moves.device, jnp.asarray(flags))
    if not np.array_equal(np.asarray(items.device.source_binding), np.asarray(source.device.binding)):
        raise ValueError("public items require the same frozen source")
    item_flags = np.zeros_like(flags)
    for (kind, _), index in source.predicate_ids.items():
        if kind == "item": item_flags[index] = True
    payload = dict(schema="pokezero.public_item_context.v1", source=np.asarray(source.device.binding).tolist(),
        ability=np.asarray(abilities.device.binding).tolist(), move=np.asarray(moves.device.binding).tolist(),
        item=np.asarray(items.device.binding).tolist(), narrowing=item_belief_narrowing,
        ability_predicate=flags.tolist(), item_predicate=item_flags.tolist())
    if field_catalog is not None:
        payload.update(schema="pokezero.public_item_field_context.v1",field=np.asarray(field_catalog.device.binding).tolist())
    if active_catalog is not None:
        if not np.array_equal(np.asarray(active_catalog.device.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding)):
            raise ValueError("active and field columns require the same frozen vocabulary")
        payload.update(schema="pokezero.public_item_field_active_context.v1",active=np.asarray(active_catalog.device.binding).tolist())
    if damage_tables is not None:
        if not np.array_equal(np.asarray(damage_tables.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding)):
            raise ValueError("damage and field columns require the same frozen vocabulary")
        payload.update(schema=payload['schema']+'.damage.v1',damage=np.asarray(damage_tables.binding).tolist())
    if type_catalog is not None:
        if not np.array_equal(np.asarray(type_catalog.device.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding)):
            raise ValueError('live types and field columns require the same frozen vocabulary')
        payload.update(schema=payload['schema']+'.types.v1',types=np.asarray(type_catalog.device.binding).tolist())
    if sleep_tables is not None:
        if not np.array_equal(np.asarray(sleep_tables.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding)):
            raise ValueError('sleep and field columns require the same frozen vocabulary')
        payload.update(schema=payload['schema']+'.sleep.v1',sleep=np.asarray(sleep_tables.binding).tolist())
    if toxic_tables is not None:
        if not np.array_equal(np.asarray(toxic_tables.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding)):
            raise ValueError('Toxic and field columns require the same frozen vocabulary')
        payload.update(schema=payload['schema']+'.toxic.raw-commit.v1',toxic=np.asarray(toxic_tables.binding).tolist())
    if tendency_tables is not None:
        if (not np.array_equal(np.asarray(tendency_tables.vocabulary_binding),np.asarray(field_catalog.device.vocabulary_binding))
                or not np.array_equal(np.asarray(tendency_tables.move_binding),np.asarray(moves.device.binding))):
            raise ValueError('tendency requires the exact frozen vocabulary and move namespace')
        payload.update(schema=payload['schema']+'.tendency.raw-commit.v1',tendency=np.asarray(tendency_tables.binding).tolist())
    if refresh_tables is not None:
        payload.update(schema=payload['schema']+'.ordered-summary.v1',refresh=np.asarray(refresh_tables.binding).tolist(),
            refresh_stream_schema=refresh_stream.SCHEMA,refresh_families=refresh_stream.FAMILIES)
    if lifecycle_tables is not None:
        payload.update(schema=payload['schema']+'.lifecycle.v1',lifecycle=np.asarray(lifecycle_tables.binding).tolist())
    if clause_tables is not None:
        payload.update(schema=payload['schema']+'.sleep-clause-raw.v1',sleep_clause=np.asarray(clause_tables.binding).tolist())
    if switch_tables is not None:
        payload.update(schema=payload['schema']+'.pending-switch.v1',switches=np.asarray(switch_tables.binding).tolist())
    if nonproc_tables is not None:
        payload.update(schema=payload['schema']+'.nonproc-upkeep.v1',nonproc=np.asarray(nonproc_tables.binding).tolist())
    if mudshot_tables is not None:
        payload.update(schema=payload['schema']+'.mudshot.v1',mudshot=np.asarray(mudshot_tables.binding).tolist(),mudshot_move_id=mudshot_move_id)
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest,dtype=">u4").astype(np.uint32))
    return ContextTables(abilities.device,moves.device,jnp.asarray(flags),items.device,
        jnp.asarray(item_flags),jnp.asarray(item_belief_narrowing),binding,
        field_catalog.device if field_catalog is not None else None,
        active_catalog.device if active_catalog is not None else None,damage_tables,
        type_catalog.device if type_catalog is not None else None,sleep_tables,toxic_tables,tendency_tables,refresh_tables,lifecycle_tables,clause_tables,switch_tables,nonproc_tables,mudshot_tables,
        jnp.asarray(mudshot_move_id,jnp.int32) if mudshot_tables is not None else None)


def initialize(beliefs, tables, *, species_binding=None, toxic_complete_prefix=False, toxic_hp_visibility=(0,0), tendency_complete_prefix=False, source_tables=None, clause_complete_prefix=False):
    """Start before evidence ingestion; importing an ambiguous old mask refuses."""
    if beliefs.species.ndim != 3 or beliefs.species.shape[1:] != (2, 6):
        raise ValueError("context requires B by two observers by six encounters")
    if np.asarray(beliefs.has_evidence).any() or np.asarray(beliefs.pin_active).any():
        raise ValueError("initialize public context before evidence ingestion")
    if tables.field is not None and tables.items is None:
        raise ValueError("field context requires item tables")
    if tables.active is not None and tables.field is None:
        raise ValueError("active context requires field tables")
    if tables.damage is not None and tables.field is None:
        raise ValueError("damage context requires field tables")
    if tables.types is not None and tables.field is None:
        raise ValueError('live-type context requires field tables')
    if tables.sleep is not None and tables.field is None:
        raise ValueError('sleep context requires field tables')
    if tables.toxic is not None and tables.field is None:
        raise ValueError('Toxic context requires ordered field tables')
    if tables.tendency is not None and tables.field is None:
        raise ValueError('tendency context requires ordered field tables')
    if tables.refresh is not None and source_tables is None:
        raise ValueError('ordered summary initialization requires bound source tables')
    batch = beliefs.species.shape[0]
    return ContextState(ability.initialize(batch, tables.abilities,species_binding=species_binding), move.initialize(batch, tables.moves),
        beliefs.hard, beliefs.has_evidence,
        item.initialize(batch,tables.items,species_binding=species_binding) if tables.items is not None else None,
        jnp.broadcast_to(tables.binding,(batch,8)) if tables.binding is not None else None,
        field_api.initialize(batch,tables.field) if tables.field is not None else None,
        active_api.initialize(batch,tables.active,species_binding=species_binding) if tables.active is not None else None,
        damage_api.initialize(batch,tables.damage,species_binding=species_binding) if tables.damage is not None else None,
        type_api.initialize(batch,tables.types,species_binding=species_binding) if tables.types is not None else None,
        sleep_api.initialize(batch,tables.sleep,species_binding=species_binding) if tables.sleep is not None else None,
        toxic_api.initialize(batch,tables.toxic,species_binding=species_binding,complete_prefix=toxic_complete_prefix,
            hp_visibility=toxic_hp_visibility) if tables.toxic is not None else None,
        jnp.zeros(batch,bool) if tables.toxic is not None else None,
        tendency_api.initialize(batch,tables.tendency,species_binding=species_binding,
            complete_prefix=tendency_complete_prefix) if tables.tendency is not None else None,
        jnp.zeros(batch,bool) if tables.tendency is not None else None,
        refresh_stream.initialize(beliefs,source_tables,tables.refresh) if tables.refresh is not None else None,
        lifecycle_api.initialize(batch,tables.lifecycle) if tables.lifecycle is not None else None,
        clause_api.initialize(batch,tables.sleep_clause,complete_prefix=clause_complete_prefix) if tables.sleep_clause is not None else None,
        switch_api.initialize(batch,tables.switches) if tables.switches is not None else None,
        nonproc_api.initialize(batch,species_binding=species_binding) if tables.nonproc is not None else None,
        jnp.broadcast_to(tables.nonproc.binding,(batch,8)) if tables.nonproc is not None else None,
        mudshot_api.initialize(batch,tables.mudshot) if tables.mudshot is not None else None)


def validate_nonproc(context,mons,tables,*,pending_arrival=False):
    """Join registered ledger identity and table namespace, also on empty chunks."""
    b=mons.encounter.shape[0]
    if context.nonproc is None or tables.nonproc is None:
        return jnp.full(b,context.nonproc is None and tables.nonproc is None and context.nonproc_binding is None)
    binding=context.nonproc_binding
    if getattr(binding,'shape',None)!=(b,8) or binding.dtype!=jnp.uint32:
        raise ValueError('non-proc context binding shape/dtype')
    if tables.nonproc.binding.shape!=(8,) or tables.nonproc.binding.dtype!=jnp.uint32:
        raise ValueError('non-proc table binding shape/dtype')
    ok=nonproc_api._validate(context.nonproc,mons)&jnp.all(binding==tables.nonproc.binding,axis=-1)
    difference=context.nonproc.encounter!=mons.encounter
    active=jnp.arange(6)[None,None,:]==mons.active[:,:,None]
    pending=jnp.broadcast_to(_bool(pending_arrival),(b,))
    permitted=pending[:,None,None]&active&(context.nonproc.encounter==0)&(mons.encounter>0)
    return ok&~jnp.any(difference&~permitted,axis=(1,2))&(jnp.sum(difference,axis=(1,2))<=1)


def validate_mudshot(context,mons,source_tables,tables,effective):
    if context.mudshot is None or tables.mudshot is None:
        return jnp.full(mons.encounter.shape[0],context.mudshot is None and tables.mudshot is None and tables.mudshot_move_id is None)
    if context.refresh is None or context.sleep is None or context.nonproc is None or tables.nonproc is None:
        return jnp.zeros(mons.encounter.shape[0],bool)
    mid=tables.mudshot_move_id
    if getattr(mid,'shape',None)!=() or mid.dtype!=jnp.int32:raise ValueError('Mud Shot move ID requires scalar int32')
    result=mudshot_api.apply(context.mudshot,mons,context.abilities,source_tables,tables.mudshot,effective,context.sleep.live,
        operation=mudshot_api.RESOLVE,producer_binding=tables.mudshot.binding)
    return result.admitted&(mid>=0)&(mid<tables.moves.max_pp.shape[0])


def basis(context, beliefs):
    # Pin state and conflict receipts may have advanced during materialization;
    # only nonreplaceable reveal/exclusion facts need a separate basis.
    return beliefs._replace(hard=context.base_hard, has_evidence=context.base_has_evidence)


def rebuild(context, base, source_tables, tables):
    ids = context.abilities.original[:, ::-1, :]
    valid_ids = (ids >= 0) & (ids < tables.abilities.predicate.shape[0])
    safe = jnp.clip(ids, 0, tables.abilities.predicate.shape[0] - 1)
    predicate = tables.abilities.predicate[safe]
    species = jnp.clip(base.species, 0, source_tables.pool.shape[0] - 1)
    matched = source_tables.predicates[predicate, species]
    known = ids > 0
    ok = jnp.all(valid_ids, axis=(1, 2)) & jnp.all(
        context.abilities.binding == tables.abilities.binding, axis=-1) & jnp.all(
        context.moves.binding == tables.moves.binding, axis=-1) & jnp.all(
        tables.abilities.source_binding == source_tables.binding) & jnp.all(
        tables.moves.source_binding == source_tables.binding)
    hard=base.hard & jnp.where(known[...,None],matched,jnp.uint32(0xffffffff))
    evidence=base.has_evidence | known
    if context.items is not None:
        ids=item.matching_item(context.items,item_belief_narrowing=tables.item_narrowing)[:,::-1,:]
        valid=(ids>=0)&(ids<tables.items.predicate.shape[0])
        predicate=tables.items.predicate[jnp.clip(ids,0,tables.items.predicate.shape[0]-1)]
        hard &= jnp.where((ids>0)[...,None],source_tables.predicates[predicate,species],jnp.uint32(0xffffffff))
        evidence |= ids>0
        ok &= jnp.all(valid,axis=(1,2)) & jnp.all(context.items.binding==tables.items.binding,axis=-1)
        ok &= jnp.all(context.binding==tables.binding,axis=-1) & jnp.all(tables.items.source_binding==source_tables.binding)
    if context.field is not None:
        ok &= jnp.all(context.field.binding==tables.field.binding,axis=-1)
    if context.active is not None:
        ok &= jnp.all(context.active.binding==tables.active.binding,axis=-1)
    if context.damage is not None:
        ok &= jnp.all(context.damage.binding==tables.damage.binding,axis=-1)
    if context.sleep is not None:
        ok &= jnp.all(context.sleep.binding==tables.sleep.binding,axis=-1)
    if context.toxic is not None:
        ok &= jnp.all(context.toxic.binding==tables.toxic.binding,axis=-1)
    if context.tendency is not None:
        ok &= jnp.all(context.tendency.binding==tables.tendency.binding,axis=-1)
    return base._replace(hard=hard,has_evidence=evidence),ok



def observation_boundary_copy(context,beliefs,mons,source_tables,tables,*,valid=True):
    """Resolve pending negative evidence on a value copy for observation only.

    Python's resolved_player_view clones the persistent engine before this
    checkpoint. A later real trigger must still see the original raw queue,
    base, visible cache generations and standing-pin conflict counters.
    """
    batch=beliefs.species.shape[0];rows=jnp.arange(batch)
    if context.switches is None or tables.switches is None:
        return context,beliefs,jnp.zeros(batch,bool)
    use=ability._bool(valid)
    if use.shape not in ((),(batch,)):raise ValueError('observation boundary validity must be scalar or exact batch')
    use=jnp.broadcast_to(use,(batch,))
    summary,cache_ok=snapshot_api.read(context.refresh.controller.visible,beliefs,source_tables)
    resolved=switch_api.resolve(context.switches,tables.switches,mons,context.abilities,source_tables,
        summary.effective,context.sleep.live,arrival=False,boundary=True,confirmed_ability=0,
        producer_binding=tables.switches.binding,valid=use)
    mask=(jnp.arange(2)[None,:,None]==(1-jnp.clip(resolved.side,0,1))[:,None,None])&(
        jnp.arange(6)[None,None,:]==jnp.clip(resolved.slot,0,5)[:,None,None])&resolved.exclude[:,None,None]
    base=basis(context,beliefs)
    predicate=tables.abilities.predicate[jnp.clip(tables.switches.intimidate_id,0,tables.abilities.predicate.shape[0]-1)]
    base,evidence_ok=belief_api.add_evidence(base,source_tables,predicate,exclude=True,valid=mask)
    full,bound=rebuild(context,base,source_tables,tables)
    visible,full,refresh_ok=snapshot_api.refresh(context.refresh.controller.visible,full,source_tables,source_refresh=mask)
    controller=context.refresh.controller._replace(visible=visible)
    copied=context._replace(switches=resolved.state,base_hard=base.hard,base_has_evidence=base.has_evidence,
        refresh=context.refresh._replace(controller=controller))
    ok=resolved.admitted&bound&jnp.all(cache_ok,axis=(1,2))&jnp.all(evidence_ok,axis=(1,2))&jnp.all(refresh_ok,axis=(1,2))
    return copied,full,ok|~use


def preconsume(context,mons,tables,row,valid,*,condition_kind):
    """Read damage's precondition HP in the same transaction as the HP update."""
    if context.damage is None:return context,jnp.ones(row.shape[0],bool)
    data=row[:,8:];use=(row[:,0]==condition_kind)&(data[:,5]==1)
    damage,ok=damage_api.apply(context.damage,tables.damage,mons,operation=damage_api.DAMAGE,
        side=row[:,6],slot=row[:,7],encounter=data[:,0],new_hp=data[:,1],
        tagged=data[:,6]==1,spikes=data[:,7]==1,producer_binding=tables.damage.binding,valid=valid&use)
    return context._replace(damage=damage),jnp.where(use,ok,True)


def consume(context, base, mons, source_tables, tables, row, valid, *,
            move_kind, ability_kind, trace_kind, transform_kind, arrival, faint, raw_kind, item_kind, field_kind, turn_kind,
            active_kind, baton_kind, upkeep_kind, damage_kind, type_kind, sleep_kind, cure_kind, toxic_kind=None, action_end_kind=None, tendency_kind=None,
            boost_kind=None,copy_boost_kind=None,clear_all_kind=None,status_kind=None,condition_kind=None,switch_kind=None,nonproc_kind=None,mudshot_kind=None):
    """One resolved typed event per battle; the outer fold commits atomically."""
    k, actor_side, actor_slot, side, slot = row[:, 0], row[:, 4], row[:, 5], row[:, 6], row[:, 7]
    data = row[:, 8:]
    summary_control=data
    a, m = context.abilities, context.moves
    summary=context.refresh;life=context.lifecycle
    switches=context.switches if tables.switches is not None else None
    nonproc=context.nonproc if tables.nonproc is not None else None
    mudshot=context.mudshot if tables.mudshot is not None else None
    batch=row.shape[0];rows=jnp.arange(batch)
    def slots(seat,target,use):
        return (jnp.arange(2)[None,:,None]==(1-jnp.clip(seat,0,1))[:,None,None])&(jnp.arange(6)[None,None,:]==jnp.clip(target,0,5)[:,None,None])&use[:,None,None]
    def checkpoint(summary,base,a,mask):
        full,bound=rebuild(context._replace(abilities=a),base,source_tables,tables)
        summary,full,ok=refresh_stream.checkpoint(summary,full,source_tables,mask)
        return summary,base._replace(conflicts=full.conflicts),ok&bound
    def resolve_mudshot(summary,base,a,pending,use):
        def resolve_once(values):
            stream,b,abilities,ledger=values
            result=mudshot_refresh.resolve(context._replace(refresh=stream,abilities=abilities,mudshot=ledger,
                base_hard=b.hard,base_has_evidence=b.has_evidence),ledger,b,mons,source_tables,tables,tables.mudshot,
                producer_binding=tables.mudshot.binding,valid=use)
            return result.context.refresh,result.basis,result.context.abilities,result.state,result.admitted
        return jax.lax.cond(jnp.any(use),resolve_once,lambda values:(*values,jnp.ones(batch,bool)),(summary,base,a,pending))
    summary_ok=jnp.ones(batch,bool)
    def life_step(life,a,op,use,*,target_side=side,target_slot=slot,encounter=data[:,0],mid=0,aid=0,benched=False):
        result=lifecycle_api.apply(life,context.sleep,a,mons,tables.lifecycle,operation=op,
            side=target_side,slot=target_slot,encounter=encounter,move_id=mid,ability_id=aid,benched=benched,
            producer_binding=tables.lifecycle.binding,valid=use)
        return result.state,result.ability_claim,result.admitted
    if summary is not None:
        # These are the installed source stages. Additional public semantics
        # cannot bypass their missing source callback merely by leaving the
        # final hard mask unchanged.
        supported=(k==raw_kind)|arrival|(k==move_kind)|(k==ability_kind)|(k==trace_kind)|(k==transform_kind)|(
            k==turn_kind)|(k==action_end_kind)|(k==item_kind)|(k==field_kind)|(k==active_kind)|(
            k==damage_kind)|(k==type_kind)|(k==toxic_kind)|(k==tendency_kind)|(
            (k==sleep_kind)&(data[:,1]==sleep_api.USABLE_MOVE))|(k==boost_kind)|(k==copy_boost_kind)|(k==clear_all_kind)
        if life is not None:
            supported |= (k==status_kind)|(k==condition_kind)|faint|(k==cure_kind)|(k==sleep_kind)
        if switches is not None:supported |= k==switch_kind
        if nonproc is not None:supported |= (k==nonproc_kind)|(k==upkeep_kind)
        if mudshot is not None:supported |= k==mudshot_kind
        summary_ok &= ~valid|supported
        summary,opened=refresh_stream.open_frame(summary,data[:,3],data[:,4:],valid&(k==raw_kind),lifecycle=life is not None,clause=tables.sleep_clause is not None,item_word=data[:,0])
        summary_ok &= opened
        if nonproc is None:summary_ok &= ~(valid&(k==raw_kind))|((data[:,0]>>5)==0)
        if mudshot is None:summary_ok &= ~(valid&(k==raw_kind))|((data[:,0]>>13)==0)
        raw_proc=(k==ability_kind)&(data[:,2]==0)
        # Raw ability procs precede items; every main/common stage follows them.
        stage=(k==raw_kind)|(k==switch_kind)|(k==item_kind)|raw_proc
        summary_ok &= ~valid|stage|(summary.item_seen==summary.item_expected)
        summary_ok &= ~(valid&raw_proc&(summary.item_expected>0))|(summary.item_seen==0)
        if mudshot is not None:
            before=(k==raw_kind)|(k==switch_kind)|(k==item_kind)|raw_proc
            summary_ok &= ~(valid&~(before|(k==mudshot_kind)))|(summary.mudshot_seen==1)
            summary_ok &= ~(valid&((k==switch_kind)|(k==item_kind)|raw_proc))|(summary.mudshot_seen==0)
        if switches is not None:
            prelude=k==switch_kind
            summary_ok &= ~(valid&~((k==raw_kind)|prelude))|(summary.switch_seen==1)
            ready=(summary.family>0)&(summary.switch_seen==0)&(summary.main_seen==0)&(summary.common_seen==0)&(summary.proc_seen==0)
            ready &= data[:,0]==(summary.family==refresh_stream.ARRIVAL)
            summary_ok &= ~(valid&prelude)|ready
            use=valid&prelude&ready&summary_ok
            full,bound=rebuild(context._replace(abilities=a),base,source_tables,tables)
            cached,cache_ok=snapshot_api.read(summary.controller.visible,full,source_tables)
            result=switch_api.resolve(switches,tables.switches,mons,a,source_tables,cached.effective,context.sleep.live,
                arrival=data[:,0]==1,boundary=data[:,1]==1,confirmed_ability=data[:,2],
                confirmed_side=actor_side,confirmed_slot=actor_slot,confirmed_encounter=data[:,3],
                exact_ident=data[:,4]==1,species_ident=data[:,5],producer_binding=tables.switches.binding,valid=use)
            summary_ok &= ~use|(result.admitted&bound&jnp.all(cache_ok,axis=(1,2)))
            switches=result.state
            mask=slots(result.side,result.slot,result.exclude&summary_ok)
            predicate=tables.abilities.predicate[jnp.clip(tables.switches.intimidate_id,0,tables.abilities.predicate.shape[0]-1)]
            base,evidence_ok=belief_api.add_evidence(base,source_tables,predicate,exclude=True,valid=mask)
            summary_ok &= ~use|jnp.all(evidence_ok,axis=(1,2))
            inferred=ability.reveal(a,mons,tables.abilities,side=result.side,slot=result.slot,encounter=result.encounter,
                ability_id=tables.switches.intimidate_id,authoritative=True,valid=result.reveal&summary_ok)
            a=inferred.state;summary_ok &= ~result.reveal|inferred.admitted
            summary,base,ok=checkpoint(summary,base,a,slots(result.side,result.slot,result.checkpoint&summary_ok))
            summary_ok &= ok
            summary=summary._replace(switch_seen=jnp.where(use&summary_ok,1,summary.switch_seen))
        if mudshot is not None:
            is_fact=k==mudshot_kind
            facts,codec_ok=mudshot_protocol.unpack(data)
            op=facts.operation;family=summary.family
            ready=(family>0)&(summary.mudshot_seen==0)&(summary.mudshot_effect_seen==0)
            ready &= (summary.main_seen==0)&(summary.common_seen==0)&(summary.nonproc_seen==0)
            ready &= summary.item_seen==summary.item_expected
            if switches is not None:ready &= summary.switch_seen==1
            word=(data[:,0]+1)|(data[:,4]<<3)|(data[:,5]<<4)
            ready &= codec_ok&(word==summary.mudshot_word)
            expected=jnp.where((family==refresh_stream.MOVE)|(family==refresh_stream.UPKEEP),op==mudshot_api.RESOLVE,
                jnp.where(family==refresh_stream.FAINT,op==mudshot_api.FAINT,
                jnp.where(family==refresh_stream.HP,(op==mudshot_protocol.NOTE)|(op==mudshot_api.DAMAGE),
                jnp.where(family==refresh_stream.NOOP,(op==mudshot_protocol.NOTE)|(op==mudshot_api.UNBOOST),op==mudshot_protocol.NOTE))))
            ready &= expected&(~facts.queue|(family==refresh_stream.MOVE))
            use=valid&is_fact&ready&summary_ok
            full,bound=rebuild(context._replace(abilities=a),base,source_tables,tables)
            cached,cache_ok=snapshot_api.read(summary.controller.visible,full,source_tables)
            event=(op==mudshot_api.DAMAGE)|(op==mudshot_api.UNBOOST)|(op==mudshot_api.FAINT)
            result=mudshot_api.apply(mudshot,mons,a,source_tables,tables.mudshot,cached.effective,context.sleep.live,
                operation=jnp.where(event,op,mudshot_api.RESOLVE),side=jnp.where(event,facts.side,-1),
                slot=jnp.where(event,facts.slot,-1),encounter=jnp.where(event,facts.encounter,0),
                tagged=event&facts.tagged,producer_binding=tables.mudshot.binding,valid=use&event)
            summary_ok &= ~is_fact|(~valid|(ready&bound&jnp.all(cache_ok,axis=(1,2))&result.admitted))
            mudshot=result.state
            summary,base,a,mudshot,resolved=resolve_mudshot(summary,base,a,mudshot,use&(family==refresh_stream.MOVE)&summary_ok)
            summary_ok &= resolved
            identity=jnp.stack((facts.side,facts.slot,facts.encounter),axis=-1)
            summary=summary._replace(mudshot_seen=jnp.where(use&summary_ok,1,summary.mudshot_seen),
                mudshot_target=jnp.where((use&summary_ok)[:,None],identity,summary.mudshot_target))
            # A bookkeeping packet cannot replace its actual raw consumer.
            code=(summary.mudshot_word&7)-1
            receipt=((code==mudshot_api.DAMAGE)&(k==condition_kind))|((code==mudshot_api.UNBOOST)&(k==boost_kind))|(
                (code==mudshot_api.FAINT)&faint)|((code==mudshot_api.RESOLVE)&(
                ((summary.family==refresh_stream.MOVE)&(k==move_kind))|((summary.family==refresh_stream.UPKEEP)&(k==upkeep_kind))))
            target_receipt=(code==mudshot_api.DAMAGE)|(code==mudshot_api.UNBOOST)|(code==mudshot_api.FAINT)
            matches=jnp.all(summary.mudshot_target==jnp.stack((side,slot,data[:,0]),axis=-1),axis=-1)
            receipt_ok=(summary.mudshot_seen==1)&(summary.mudshot_effect_seen==0)&(~target_receipt|matches)
            receipt_ok &= (code!=mudshot_api.DAMAGE)|((data[:,5]==1)&((data[:,6]==1)==((summary.mudshot_word&16)!=0)))
            receipt_ok &= (code!=mudshot_api.UNBOOST)|((data[:,3]==mon.DELTA)&(data[:,4]==1))
            # Bind both directions: genuine body facts cannot be hidden by a
            # consistent forged header and NOTE packet, or invented by UNBOOST.
            body_damage=k==condition_kind
            body_boost=k==boost_kind
            summary_ok &= ~(valid&body_damage)|((code==mudshot_api.DAMAGE)==(data[:,5]==1))
            summary_ok &= ~(valid&body_boost)|((code==mudshot_api.UNBOOST)==(data[:,4]==1))
            summary_ok &= ~(valid&receipt)|receipt_ok
            summary=summary._replace(mudshot_effect_seen=jnp.where(valid&receipt&receipt_ok&summary_ok,1,summary.mudshot_effect_seen))
        if nonproc is not None:
            is_fact=k==nonproc_kind
            facts,codec_ok=nonproc_protocol.unpack(data)
            upkeep=summary.family==refresh_stream.UPKEEP
            ordinary_main=~((summary.family==refresh_stream.NOOP)|(summary.family==refresh_stream.CANT_OTHER)|(summary.family==refresh_stream.ITEM)|upkeep)
            ready=(summary.family>0)&(summary.nonproc_seen==0)&(summary.main_seen==ordinary_main.astype(jnp.int32))
            ready &= summary.item_seen==summary.item_expected
            if switches is not None:ready &= summary.switch_seen==1
            expected_op=jnp.where(upkeep,nonproc_protocol.UPKEEP,jnp.where(summary.family==refresh_stream.ARRIVAL,
                nonproc_api.ARRIVE,jnp.where(summary.family==refresh_stream.HP,nonproc_api.HP,nonproc_api.NOTE)))
            ready &= codec_ok&(facts.operation==expected_op)
            ready &= ((data[:,6]<<2)|data[:,7])==summary.nonproc_word
            ready &= facts.berry_eat==((summary.family==refresh_stream.ITEM)&summary.item_eat)
            targeted=(facts.operation==nonproc_api.ARRIVE)|(facts.operation==nonproc_api.HP)
            address=(rows,jnp.clip(facts.side,0,1),jnp.clip(facts.slot,0,5))
            ready &= ~targeted|((facts.hp==mons.hp_numerator[address])&(facts.maximum==mons.hp_denominator[address]))
            identity=jnp.stack((facts.side,facts.slot,facts.encounter),axis=-1)
            ready &= (facts.operation!=nonproc_api.HP)|jnp.all(identity==summary.common_proof[:,:3],axis=-1)
            arrival_side=summary.main_proof[:,0]
            arrival_slot=mons.active[rows,jnp.clip(arrival_side,0,1)]
            ready &= (facts.operation!=nonproc_api.ARRIVE)|((facts.side==arrival_side)&(facts.slot==arrival_slot))
            ready &= ~facts.berry_eat|jnp.all(identity==summary.item_target,axis=-1)
            ready &= ~((summary.family==refresh_stream.MOVE)|(summary.family==refresh_stream.TURN))|(facts.phase==nonproc_api.ACTION_PHASE)
            use=valid&is_fact&ready&summary_ok
            updated,recorded=nonproc_api.record(nonproc,mons,operation=jnp.where(upkeep,nonproc_api.NOTE,facts.operation),
                side=facts.side,slot=facts.slot,encounter=facts.encounter,hp=facts.hp,maximum=facts.maximum,
                phase=facts.phase,action_phase_hp=facts.action_phase_hp,residual_phase_hp=facts.residual_phase_hp,
                leftovers_heal=facts.leftovers_heal,berry_eat=facts.berry_eat,valid=use)
            summary_ok &= ~is_fact|(~valid|(ready&recorded))
            sweep_use=use&upkeep&summary_ok
            def sweep_once(values):
                stream,ledger,b=values
                bridge=nonproc_refresh.upkeep(context._replace(refresh=stream,nonproc=ledger,
                    base_hard=b.hard,base_has_evidence=b.has_evidence),ledger,b,mons,source_tables,tables,
                    tables.nonproc,producer_binding=tables.nonproc.binding,valid=sweep_use)
                return bridge.context.refresh,bridge.state,bridge.basis,bridge.admitted
            summary,nonproc,base,swept=jax.lax.cond(jnp.any(sweep_use),sweep_once,
                lambda values:(*values,jnp.ones(batch,bool)),(summary,updated,base))
            summary_ok &= swept
            if mudshot is not None:
                summary,base,a,mudshot,resolved=resolve_mudshot(summary,base,a,mudshot,sweep_use&summary_ok)
                summary_ok &= resolved
            summary=summary._replace(nonproc_seen=jnp.where(use&summary_ok,1,summary.nonproc_seen))
            # All non-upkeep facts are terminal bookkeeping before the closing
            # tendency packet; upkeep alone still owes its main clock event.
            after=summary.nonproc_seen==1
            summary_ok &= ~(valid&~((k==raw_kind)|is_fact|(k==tendency_kind)|(upkeep&(k==upkeep_kind))))|~after
            summary_ok &= ~(valid&(k==upkeep_kind))|((summary.family==refresh_stream.UPKEEP)&after)
            proof=jnp.stack((jnp.full(batch,-1,jnp.int32),jnp.full(batch,tendency_api.UPKEEP,jnp.int32),
                jnp.zeros(batch,jnp.int32),jnp.full(batch,-1,jnp.int32),jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32)),axis=-1)
            summary,upkeep_ok=refresh_stream.main(summary,refresh_stream.UPKEEP,proof,valid&(k==upkeep_kind)&summary_ok)
            summary_ok &= upkeep_ok
        # Source marks the outgoing record before the incoming upsert. The
        # public mon consumer has introduced the incoming row already, but its
        # source summary remains unrefreshed until these checkpoints.
        safe_side=jnp.clip(side,0,1);safe_slot=jnp.clip(slot,0,5)
        old=context.sleep;old_slot=jnp.clip(old.active[rows,safe_side],0,5)
        leaving=valid&arrival&old.live[rows,safe_side]&~old.literal_faint[rows,safe_side,old_slot]
        summary,base,ok=checkpoint(summary,base,a,slots(side,old_slot,leaving));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,valid&arrival));summary_ok &= ok
        # Natural Cure's source replacement needs its exit/cure-count evidence
        # producer. Do not infer it from a clean return while that producer is
        # still being integrated.
        if life is None:
            exited=jnp.where(leaving&(old_slot==slot),old.status[rows,safe_side,old_slot]!=mon.NONE,
                old.status_on_exit[rows,safe_side,safe_slot])
            summary_ok &= ~(valid&arrival&exited&(mons.status[rows,safe_side,safe_slot]==mon.NONE)&(a.original[rows,safe_side,safe_slot]==0))
        else:
            summary,common_ok=refresh_stream.common(summary,row,status_kind=status_kind,condition_kind=condition_kind,
                valid=valid&((k==status_kind)|(k==condition_kind)))
            summary_ok &= common_ok
        # Transform performs an upsert before replacing its copied state.
        summary,base,ok=checkpoint(summary,base,a,slots(actor_side,actor_slot,valid&(k==transform_kind)));summary_ok &= ok
    a, aa = ability.arrive(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & arrival)
    m, ma = move.arrive(m, tables.moves, side=side, valid=valid & arrival)
    if life is not None:
        life,claim,ok=life_step(life,a,lifecycle_api.ARRIVE,valid&arrival&summary_ok)
        summary_ok &= ok
        inferred=ability.reveal(a,mons,tables.abilities,side=side,slot=slot,encounter=data[:,0],
            ability_id=claim,authoritative=True,valid=valid&arrival&(claim>0)&summary_ok)
        a=inferred.state;summary_ok &= ~(valid&arrival&(claim>0))|inferred.admitted
    a, fa = ability.faint(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & faint)
    is_move, is_ability, is_trace, is_transform = (k == x for x in (move_kind, ability_kind, trace_kind, transform_kind))
    transformed, pressure, ca = ability.move_context(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], valid=valid & is_move)
    if summary is None:
        update = move.record_move(m, base, mons, tables.moves, source_tables,
            side=actor_side, slot=actor_slot, encounter=data[:, 0], move_id=data[:, 1],
            caller=data[:, 2], has_from=data[:, 3] == 1, transformed=transformed,
            opposing_pressure=pressure, wire_target_side=side, valid=valid & is_move & ca)
    else:
        full,bound=rebuild(context._replace(abilities=a),base,source_tables,tables)
        result=refresh_api.record_move(summary.controller,m,base,full,mons,source_tables,tables.moves,tables.refresh,
            side=actor_side,slot=actor_slot,encounter=data[:,0],move_id=data[:,1],caller=data[:,2],
            has_from=data[:,3]==1,transformed=transformed,opposing_pressure=pressure,wire_target_side=side,
            item_mutated=context.items.mutated,item_revealed=context.items.revealed>0,sleep_pending=context.sleep.pending,
            item_belief_narrowing=tables.item_narrowing,lifecycle=life,lifecycle_tables=tables.lifecycle,
            public_sleep=context.sleep,public_abilities=a,valid=valid&is_move&ca&summary_ok)
        summary=summary._replace(controller=result.refresh);update=result.move_update;life=result.lifecycle
        summary_ok &= result.admitted&bound
    m, base = update.state, update.beliefs
    if mudshot is not None:
        queue=(summary.mudshot_word&8)!=0
        expected=(tables.mudshot_move_id>0)&(data[:,1]==tables.mudshot_move_id)
        use=valid&is_move
        summary_ok &= ~use|(queue==expected)
        peer=1-jnp.clip(actor_side,0,1);target=context.sleep.active[rows,peer]
        live=context.sleep.live[rows,peer]&(target>=0)
        encounter=mons.encounter[rows,peer,jnp.clip(target,0,5)]
        address=jnp.stack((jnp.where(live,peer,-1),jnp.where(live,target,-1),jnp.where(live,encounter,0)),axis=-1)
        summary_ok &= ~(use&queue)|jnp.all(address==summary.mudshot_target,axis=-1)
        full,bound=rebuild(context._replace(abilities=a),base,source_tables,tables)
        cached,cache_ok=snapshot_api.read(summary.controller.visible,full,source_tables)
        queued=mudshot_api.apply(mudshot,mons,a,source_tables,tables.mudshot,cached.effective,context.sleep.live,
            operation=mudshot_api.QUEUE,side=summary.mudshot_target[:,0],slot=summary.mudshot_target[:,1],
            encounter=summary.mudshot_target[:,2],producer_binding=tables.mudshot.binding,valid=use&queue&summary_ok)
        mudshot=queued.state;summary_ok &= ~(use&queue)|(queued.admitted&bound&jnp.all(cache_ok,axis=(1,2)))
    if life is not None:
        # _target_belief falls back to _upsert when a side has no active
        # source record. Even an equal/borrowed proc still performs that
        # lookup refresh. Trace looks up its donor again after the guard.
        inactive=~context.sleep.live[rows,jnp.clip(side,0,1)]
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,valid&is_ability&inactive&summary_ok));summary_ok &= ok
        tracer_inactive=~context.sleep.live[rows,jnp.clip(actor_side,0,1)]
        traced=valid&is_trace&(side>=0)
        summary,base,ok=checkpoint(summary,base,a,slots(actor_side,actor_slot,traced&tracer_inactive&summary_ok));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,traced&inactive&summary_ok));summary_ok &= ok
        donor=(rows,jnp.clip(side,0,1),jnp.clip(slot,0,5))
        eligible=(actor_side!=side)|(actor_slot!=slot)
        eligible &= ~a.transformed[donor]&(a.running[donor]==0)
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,traced&inactive&eligible&summary_ok));summary_ok &= ok
        shed=valid&is_ability&((summary.raw_meta[:,0]&1)!=0)&(summary.family==refresh_stream.NOOP)
        proof=summary.raw_meta[:,1:]
        shed_ok=(summary.proc_seen==0)&jnp.all(jnp.stack((side,slot,data[:,0]),axis=-1)==proof,axis=-1)
        shed_ok &= (data[:,1]==tables.lifecycle.shed_skin)&(data[:,2]==0)
        summary_ok &= ~shed|shed_ok
        life,_,ok=life_step(life,a,lifecycle_api.SHED_SKIN_ACTIVATE,shed&shed_ok&summary_ok,aid=data[:,1])
        summary_ok &= ok
        summary=summary._replace(proc_seen=jnp.where(shed&shed_ok&summary_ok,1,summary.proc_seen))
        # HP absorb outcomes require the real proc, not a generic permission to
        # close any HP frame with OUTCOME. Bind both directions so removing the
        # header marker cannot erase the outcome while retaining its ability.
        hp_proc=valid&is_ability&(summary.family==refresh_stream.HP)&(data[:,2]==0)
        word=summary.raw_meta[:,0];marked=refresh_stream.hp_absorb(word)
        hp_side,hp_slot,hp_aid=refresh_stream.hp_absorb_address(word)
        aid=jnp.clip(data[:,1],0,tables.lifecycle.absorb_heal.shape[0]-1)
        absorb=hp_proc&tables.lifecycle.absorb_heal[aid]
        checked=hp_proc&(absorb|marked)
        proc_ok=marked&absorb&(summary.proc_seen==0)&(side==hp_side)&(slot==hp_slot)&(data[:,1]==hp_aid)
        summary_ok &= ~checked|proc_ok
        summary=summary._replace(proc_seen=jnp.where(checked&proc_ok&summary_ok,1,summary.proc_seen))
    before_ability=a
    rev = ability.reveal(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0],
        ability_id=data[:, 1], authoritative=data[:, 2] == 1, valid=valid & is_ability)
    a = rev.state
    a, ta = ability.trace_copy(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], ability_id=data[:, 1],
        donor_side=side, donor_slot=slot, donor_encounter=data[:, 2],
        parser_feature=data[:, 3] == 1, valid=valid & is_trace)
    a, tr = ability.transform(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], donor_side=side, donor_slot=slot,
        donor_encounter=data[:, 1], valid=valid & is_transform)
    if summary is not None:
        # Direct declarations refresh even when equal. A proc on a borrowed
        # ability, or an already known equal claim, performs no replacement.
        target=(rows,jnp.clip(side,0,1),jnp.clip(slot,0,5))
        proc_call=(before_ability.running[target]==0)&(before_ability.original[target]!=data[:,1])
        ability_call=valid&is_ability&((data[:,2]==1)|proc_call)
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,ability_call));summary_ok &= ok
        same=(actor_side==side)&(actor_slot==slot)
        donor_call=valid&is_trace&(side>=0)&~same&~before_ability.transformed[target]&(before_ability.running[target]==0)
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,donor_call));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(actor_side,actor_slot,valid&is_transform));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,valid&arrival));summary_ok &= ok
        controller,reset_ok=refresh_api.reset_stay(summary.controller,mons,tables.refresh,
            side=side,slot=slot,encounter=data[:,0],valid=valid&arrival)
        summary=summary._replace(controller=controller);summary_ok &= reset_ok
        # Source TURN replaces active p1 then p2, including unchanged masks.
        for seat in (0,1):
            active=context.sleep.active[:,seat];use=valid&(k==turn_kind)&context.sleep.live[:,seat]
            summary,base,ok=checkpoint(summary,base,a,slots(jnp.full(batch,seat,jnp.int32),active,use));summary_ok &= ok
        for code,use in ((refresh_stream.ARRIVAL,arrival),(refresh_stream.MOVE,is_move),
                (refresh_stream.TURN,k==turn_kind),(refresh_stream.ABILITY,is_ability&(data[:,2]==1)),
                (refresh_stream.TRACE,is_trace),(refresh_stream.TRANSFORM,is_transform)):
            source_side=jnp.where((code==refresh_stream.MOVE)|(code==refresh_stream.TRACE)|(code==refresh_stream.TRANSFORM),actor_side,
                jnp.where(code==refresh_stream.TURN,-1,side))
            source_kind=jnp.where(code==refresh_stream.ARRIVAL,jnp.where(k==baton_kind,tendency_api.SWITCH,k-(baton_kind-3)+tendency_api.SWITCH),
                jnp.where(code==refresh_stream.MOVE,tendency_api.MOVE,jnp.where(code==refresh_stream.TURN,tendency_api.TURN,tendency_api.OTHER)))
            flags=jnp.where(code==refresh_stream.ARRIVAL,jnp.where(k==baton_kind,8,0),
                jnp.where(code==refresh_stream.MOVE,jnp.where((data[:,2]==move.SLEEP_TALK)|(data[:,2]==move.RANDOM_CALLER),1,
                    jnp.where(data[:,2]==move.LOCKED_MOVE,2,0)),0))
            proof=jnp.stack((source_side,source_kind,jnp.where((code==refresh_stream.ARRIVAL)|(code==refresh_stream.MOVE),data[:,1],0),
                jnp.where(code==refresh_stream.MOVE,jnp.where(side>=0,side,1-actor_side),-1),flags,
                jnp.where(code==refresh_stream.TURN,row[:,2],0)),axis=-1)
            summary,ok=refresh_stream.main(summary,code,proof,valid&use);summary_ok &= ok
    if life is not None:
        is_sleep=k==sleep_kind;op=data[:,1]
        status=valid&is_sleep&(op==sleep_api.STATUS)
        cure=valid&is_sleep&(op==sleep_api.CURE)
        hp=valid&is_sleep&(op==sleep_api.CONDITION)
        cant=valid&is_sleep&(op==sleep_api.CANT_SLEEP)
        common_use=status|cure|hp
        summary_ok &= ~common_use|refresh_stream.common_matches(summary,row)
        summary_ok &= ~status|((summary.family==refresh_stream.STATUS)&(data[:,2]==summary.raw_meta[:,1])&
            (2*data[:,3]==(summary.raw_meta[:,0]&2)))
        summary_ok &= ~cure|((summary.family==refresh_stream.CURE)&(data[:,2]==summary.raw_meta[:,1]))
        hp_marked=refresh_stream.hp_absorb(summary.raw_meta[:,0])
        hp_side,hp_slot,_=refresh_stream.hp_absorb_address(summary.raw_meta[:,0])
        summary_ok &= ~hp|((summary.family==refresh_stream.HP)&(data[:,2]==summary.raw_meta[:,3])&
            (data[:,3]==jnp.where(hp_marked,summary.raw_meta[:,0]&3,summary.raw_meta[:,0]))&
            (~hp_marked|((side==hp_side)&(slot==hp_slot)&(summary.proc_seen==1))))
        summary_ok &= ~cant|(summary.family==refresh_stream.CANT)
        # A sleep attempt upserts before replacing its counter; both calls
        # see the same candidate-set predicates, including a standing pin.
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,cant&summary_ok));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,cure&(summary.raw_meta[:,0]==1)&summary_ok));summary_ok &= ok
        life,claim,ok=life_step(life,a,lifecycle_api.CURE,cure&summary_ok,benched=data[:,2]==1)
        summary_ok &= ok
        inferred=ability.reveal(a,mons,tables.abilities,side=side,slot=slot,encounter=data[:,0],
            ability_id=claim,authoritative=True,valid=cure&(claim>0)&summary_ok)
        a=inferred.state;summary_ok &= ~(cure&(claim>0))|inferred.admitted
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,cure&(claim>0)&summary_ok));summary_ok &= ok
        summary,base,ok=checkpoint(summary,base,a,slots(side,slot,(status|cure|hp|cant|valid&faint)&summary_ok));summary_ok &= ok
        life,_,ok=life_step(life,a,lifecycle_api.FAINT,valid&faint&summary_ok);summary_ok &= ok
        team=valid&(k==cure_kind)
        life,_,ok=life_step(life,a,lifecycle_api.CURE_TEAM,team&summary_ok,encounter=0,target_slot=-1);summary_ok &= ok
        for index in range(6):
            seat=jnp.clip(side,0,1)
            affected=~context.sleep.literal_faint[rows,seat,index]&((context.sleep.status[rows,seat,index]!=mon.NONE)|
                (context.sleep.condition_status[rows,seat,index]!=mon.NONE))
            summary,base,ok=checkpoint(summary,base,a,slots(side,jnp.full(batch,index,jnp.int32),team&affected&summary_ok));summary_ok &= ok
        for code,use in ((refresh_stream.STATUS,status),(refresh_stream.CURE,cure),(refresh_stream.HP,hp),
                (refresh_stream.CANT,cant),(refresh_stream.TEAM,team),(refresh_stream.FAINT,valid&faint)):
            closing_kind=jnp.full(batch,tendency_api.CANT if code==refresh_stream.CANT else
                tendency_api.FAINT if code==refresh_stream.FAINT else tendency_api.OTHER,jnp.int32)
            absorb_hp=(code==refresh_stream.HP)&hp_marked
            closing_kind=jnp.where(absorb_hp,tendency_api.OUTCOME,closing_kind)
            proof=jnp.stack((side,closing_kind,jnp.zeros(batch,jnp.int32),jnp.where(absorb_hp,side,-1),
                jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32)),axis=-1)
            summary,ok=refresh_stream.main(summary,code,proof,use&summary_ok);summary_ok &= ok
    accepted = jnp.where(arrival, aa & ma, jnp.where(faint, fa, jnp.where(is_move, update.admitted,
        jnp.where(is_ability, rev.admitted, jnp.where(is_trace, ta, jnp.where(is_transform, tr, True))))))
    items=context.items
    if items is not None:
        raw=k==raw_kind
        items,ra=item.begin_event(items,mons,tables.items,item_event=(data[:,0]&3)==1,trick=(data[:,0]&3)==2,
            actor_side=actor_side,actor_slot=actor_slot,actor_encounter=data[:,1],
            target_side=side,target_slot=slot,target_encounter=data[:,2],valid=valid&raw)
        if summary is None:
            items,ia=item.record(items,mons,tables.items,side=side,slot=slot,encounter=data[:,0],
                item_id=data[:,1],operation=data[:,2],item_binding=tables.items.binding,valid=valid&(k==item_kind))
        else:
            use=valid&(k==item_kind)&summary_ok
            ready=(summary.family>0)&(summary.item_seen<summary.item_expected)
            ready &= (summary.main_seen==0)&(summary.common_seen==0)
            second=summary.item_seen>0
            address=jnp.stack((side,slot,data[:,0]),axis=-1)
            ready &= ~second|jnp.all(summary.item_target==address,axis=-1)
            explicit=(data[:,2]==item.DIRECT)|(data[:,2]==item.TRICK_RECEIVE)|(data[:,2]==item.OTHER_RECEIVE)
            ready &= (summary.item_expected!=2)|jnp.where(second,explicit,data[:,2]==item.INLINE)
            summary_ok &= ~use|ready
            use &= ready
            # _target_belief upserts an inactive target before item replacement.
            lookup=use&~context.sleep.live[rows,jnp.clip(side,0,1)]
            summary,base,ok=checkpoint(summary,base,a,slots(side,slot,lookup));summary_ok &= ok
            stages=item_callbacks.project(items,mons,tables.items,side=side,slot=slot,encounter=data[:,0],
                item_id=data[:,1],operation=data[:,2],item_binding=tables.items.binding,
                is_eat=summary.item_eat,valid=use&summary_ok)
            ia=stages.admitted
            for stage,mask in ((stages.mutation,slots(side,slot,stages.mutation_callback)),
                    (stages.reveal,slots(side,slot,stages.reveal_callback)),
                    (stages.partner,stages.partner_callback[:,::-1,:])):
                full,bound=rebuild(context._replace(abilities=a,items=stage),base,source_tables,tables)
                summary,full,ok=refresh_stream.checkpoint(summary,full,source_tables,mask)
                base=base._replace(conflicts=full.conflicts);summary_ok &= bound&ok
            items=stages.partner
            first=use&ia&summary_ok&(summary.item_seen==0)
            summary=summary._replace(item_seen=jnp.where(use&ia&summary_ok,summary.item_seen+1,summary.item_seen),
                item_target=jnp.where(first[:,None],address,summary.item_target))
        accepted &= jnp.where(raw,ra,jnp.where(k==item_kind,ia,True))
    fields=context.field
    if fields is not None:
        is_turn=k==turn_kind
        fields,ff=field_api.apply(fields,tables.field,operation=jnp.where(is_turn,field_api.TURN,data[:,0]),
            side=jnp.where(is_turn,-1,side),value=jnp.where(is_turn,row[:,2],data[:,1]),
            upkeep=(k==field_kind)&(data[:,2]==1),from_ability=(k==field_kind)&(data[:,3]==1),
            producer_binding=tables.field.binding,valid=valid&((k==field_kind)|is_turn))
        accepted &= ff
    active=context.active
    if active is not None:
        is_turn=k==turn_kind;is_active=k==active_kind
        is_upkeep=(k==upkeep_kind) if tables.active.native_truant is not None else jnp.zeros_like(is_turn)
        relevant=arrival|faint|is_turn|is_active|is_upkeep
        op=jnp.where(arrival,active_api.ARRIVE,jnp.where(faint,active_api.FAINT,
            jnp.where(is_turn,active_api.TURN,jnp.where(is_upkeep,active_api.UPKEEP,data[:,1]))))
        active_identity=(side>=0)&(side<2)&(slot==mons.active[jnp.arange(side.shape[0]),jnp.clip(side,0,1)])
        active,ac=active_api.apply(active,tables.active,mons,operation=op,
            side=jnp.where(is_turn|is_upkeep,-1,side),encounter=jnp.where(is_turn|is_upkeep,0,data[:,0]),
            value=jnp.where(is_turn,row[:,2],jnp.where(is_active,data[:,2],0)),baton_pass=k==baton_kind,
            producer_binding=tables.active.binding,valid=valid&relevant&(~is_active|active_identity))
        accepted &= jnp.where(relevant,ac,True)
    damage=context.damage
    if damage is not None:
        is_turn=k==turn_kind;control=k==damage_kind
        relevant=arrival|is_move|is_turn|control
        op=jnp.where(arrival,damage_api.ARRIVE,jnp.where(is_move,damage_api.MOVE,
            jnp.where(is_turn,damage_api.TURN,data[:,0])))
        targeted=arrival|is_move
        damage,da=damage_api.apply(damage,tables.damage,mons,operation=op,
            side=jnp.where(is_turn,-1,jnp.where(is_move,actor_side,side)),
            slot=jnp.where(targeted,jnp.where(is_move,actor_slot,slot),-1),
            encounter=jnp.where(targeted,data[:,0],0),value=jnp.where(is_turn,row[:,2],0),
            producer_binding=tables.damage.binding,valid=valid&relevant)
        accepted &= jnp.where(relevant,da,True)
    types=context.types
    if types is not None:
        targeted=k==type_kind;relevant=arrival|targeted
        identity=slot==mons.active[jnp.arange(side.shape[0]),jnp.clip(side,0,1)]
        types,ty=type_api.apply(types,tables.types,mons,operation=jnp.where(arrival,type_api.ARRIVE,data[:,1]),
            side=side,encounter=data[:,0],first=jnp.where(arrival,0,data[:,2]),second=jnp.where(arrival,0,data[:,3]),
            producer_binding=tables.types.binding,valid=valid&relevant&(~targeted|identity))
        accepted &= jnp.where(relevant,ty&(~targeted|identity),True)
    sleep=context.sleep
    if sleep is not None:
        is_turn=k==turn_kind;is_upkeep=k==upkeep_kind;is_sleep=k==sleep_kind;team=k==cure_kind
        if summary is not None and nonproc is not None:
            # Source upkeep replaces each unresolved plain-sleep record with
            # nonzero skippedTime, even though its variant mask is unchanged.
            # Reapplying a standing pin at that replacement is observable.
            reset=sleep.pending&(sleep.skipped>0)&(mons.encounter>0)
            reset &= (valid&is_upkeep&summary_ok)[:,None,None]
            summary,base,ok=checkpoint(summary,base,a,reset[:,::-1,:]);summary_ok &= ok
        relevant=arrival|faint|is_turn|is_upkeep|is_sleep|team
        global_fact=is_turn|is_upkeep
        op=jnp.where(arrival,sleep_api.ARRIVE,jnp.where(faint,sleep_api.FAINT,
            jnp.where(is_turn,sleep_api.TURN,jnp.where(is_upkeep,sleep_api.UPKEEP,
            jnp.where(team,sleep_api.CURE_TEAM,data[:,1])))))
        sleep,sa=sleep_api.apply(sleep,tables.sleep,mons,operation=op,
            side=jnp.where(global_fact,-1,side),slot=jnp.where(global_fact|team,-1,slot),
            encounter=jnp.where(global_fact|team,0,data[:,0]),
            first=jnp.where(is_turn,row[:,2],jnp.where(is_sleep,data[:,2],0)),
            second=jnp.where(is_sleep,data[:,3],0),producer_binding=tables.sleep.binding,valid=valid&relevant)
        accepted &= jnp.where(relevant,sa,True)
    clauses=context.sleep_clause
    if clauses is not None and tables.sleep_clause is not None:
        status_event=(k==sleep_kind)&(data[:,1]==sleep_api.STATUS)
        cure_event=(k==sleep_kind)&(data[:,1]==sleep_api.CURE)
        team_event=k==cure_kind
        relevant=arrival|faint|status_event|cure_event|team_event
        op=jnp.where(arrival,clause_api.ARRIVE,jnp.where(faint,clause_api.FAINT,
            jnp.where(team_event,clause_api.TEAM,jnp.where(status_event,clause_api.STATUS,clause_api.CURE))))
        code=jnp.where(status_event,data[:,2],jnp.where(cure_event,summary.raw_meta[:,2],0))
        parser_match=cure_event&(summary.raw_meta[:,3]==1)|faint
        clauses,ca=clause_api.apply(clauses,tables.sleep_clause,mons,context.sleep,operation=op,
            side=side,slot=jnp.where(team_event,-1,slot),encounter=jnp.where(team_event,0,data[:,0]),status=code,
            rest=status_event&(data[:,3]==1),ability_source=status_event&((summary.raw_meta[:,0]&4)!=0),
            parser_ident_matches=parser_match,producer_binding=tables.sleep_clause.binding,valid=valid&relevant)
        accepted &= jnp.where(relevant,ca,True)
    toxic=context.toxic
    raw_pending=context.toxic_raw_pending
    if toxic is not None:
        is_turn=k==turn_kind;is_upkeep=k==upkeep_kind;is_blank=k==action_end_kind;is_team=k==cure_kind
        control=k==toxic_kind;raw=k==raw_kind
        relevant=arrival|faint|is_turn|is_upkeep|is_blank|is_team|control
        # Every raw semantic consumer, not only Toxic's closing control, must
        # belong to the open transaction. Otherwise dropping BOTH frame
        # markers could charge PP or mutate HP without visiting Toxic state.
        other_close=(k==tendency_kind) if context.tendency is not None else jnp.zeros(k.shape,bool)
        if nonproc is not None:other_close |= k==nonproc_kind
        if mudshot is not None:other_close |= k==mudshot_kind
        accepted &= other_close|jnp.where(raw,~raw_pending,raw_pending)
        # Arrival transport kinds are contiguous SWITCH, DRAG, REPLACE, BP;
        # derive from the supplied first arrival kind rather than party order.
        common_kind=jnp.where(arrival,jnp.where(k==baton_kind,toxic_api.SWITCH,k-(baton_kind-3)+toxic_api.SWITCH),
            jnp.where(faint,toxic_api.FAINT,jnp.where(is_turn,toxic_api.TURN,
            jnp.where(is_upkeep,toxic_api.UPKEEP,jnp.where(is_team,toxic_api.TEAM,toxic_api.OTHER)))))
        global_fact=is_turn|is_upkeep|is_blank
        false=jnp.zeros(k.shape,bool)
        common=toxic_api.ToxicFact(common_kind,jnp.where(global_fact,-1,side),
            jnp.where(arrival|faint,data[:,0],0),jnp.ones(k.shape,bool),arrival|faint,
            jnp.where(is_turn,row[:,2],0),jnp.where(arrival,data[:,5],-1),
            jnp.where(arrival,data[:,6],0),arrival&(data[:,7]==mon.TOXIC),false,k==baton_kind,
            is_turn|is_upkeep,arrival|faint)
        decoded,codec_ok=toxic_protocol.unpack(side,data)
        named=(decoded.kind==toxic_api.OTHER)|(decoded.kind==toxic_api.TIMESTAMP)|(decoded.kind==toxic_api.STATUS)|(
            decoded.kind==toxic_api.CURE)|(decoded.kind==toxic_api.CONDITION)|(decoded.kind==toxic_api.MOVE)|(
            decoded.kind==toxic_api.WIN)|(decoded.kind==toxic_api.TERMINAL)
        codec_ok=jnp.where(control,codec_ok&named,True)
        fact=jax.tree.map(lambda a,b:jnp.where(control,a,b),decoded,common)
        toxic,tx=toxic_api.apply(toxic,tables.toxic,fact,producer_binding=tables.toxic.binding,
            valid=valid&relevant&codec_ok&raw_pending)
        accepted &= jnp.where(raw,~raw_pending,jnp.where(relevant,codec_ok&tx&raw_pending,True))
        raw_pending=jnp.where(valid&raw,True,jnp.where(valid&relevant,False,raw_pending))
    tendency=context.tendency;pending=context.tendency_raw_pending
    if tendency is not None:
        raw=k==raw_kind;control=k==tendency_kind
        # Each raw transaction has exactly one final lexical tendency control.
        # All semantic events require its opener, even if both markers were lost.
        accepted &= jnp.where(raw,~pending,pending)
        decoded,codec_ok=tendency_protocol.unpack(data,side)
        if summary is not None:
            # Kind.BATON_PASS also carries a pending prior selection. The
            # lexical closing flag alone names only an explicit raw marker.
            # Compare the effective source arrival semantics across consumers.
            baton=(decoded.kind==tendency_api.SWITCH)&(decoded.baton|tendency.baton[jnp.arange(batch),jnp.clip(side,0,1)])
            arrival_control=(decoded.kind>=tendency_api.SWITCH)&(decoded.kind<=tendency_api.REPLACE)
            summary_control=data.at[:,5].set(jnp.where(arrival_control,jnp.where(baton,8,0),data[:,5]))
        tendency,tc=tendency_api.apply(tendency,tables.tendency,decoded,
            producer_binding=tables.tendency.binding,valid=valid&control&pending&codec_ok)
        current_species=jnp.take_along_axis(tendency.species,jnp.clip(tendency.active,0,5)[:,:,None],axis=2)[:,:,0]
        public_species=jnp.take_along_axis(mons.species,jnp.clip(mons.active,0,5)[:,:,None],axis=2)[:,:,0]
        current_species=jnp.where(tendency.active>=0,current_species,0)
        public_species=jnp.where(mons.active>=0,public_species,0)
        coherent=(tendency.turn==row[:,2])&jnp.all(current_species==public_species,axis=-1)
        accepted &= ~control|(pending&codec_ok&tc&coherent)
        pending=jnp.where(valid&raw,True,jnp.where(valid&control,False,pending))
    if summary is not None:
        full,bound=rebuild(context._replace(abilities=a,items=items),base,source_tables,tables)
        queue_arrival=(summary.family==refresh_stream.ARRIVAL)&valid&(k==tendency_kind)
        summary,ok=refresh_stream.close_frame(summary,full,source_tables,summary_control,side,valid&(k==tendency_kind),pending_switch=switches is not None,nonproc=nonproc is not None,mudshot=mudshot is not None)
        if switches is not None:
            cached,cache_ok=snapshot_api.read(summary.controller.visible,full,source_tables)
            target=jnp.clip(mons.active[rows,jnp.clip(side,0,1)],0,5)
            encounter=mons.encounter[rows,jnp.clip(side,0,1),target]
            switches,queued=switch_api.queue_arrival(switches,tables.switches,mons,a,source_tables,cached.effective,sleep.live,
                side=side,slot=target,encounter=encounter,producer_binding=tables.switches.binding,
                valid=queue_arrival&summary_ok&bound&ok)
            accepted &= ~queue_arrival|(queued&jnp.all(cache_ok,axis=(1,2)))
        accepted &= summary_ok&bound&ok
    return context._replace(refresh=summary,lifecycle=life,sleep_clause=clauses,switches=switches if tables.switches is not None else context.switches,nonproc=nonproc if tables.nonproc is not None else context.nonproc,mudshot=mudshot if tables.mudshot is not None else context.mudshot,abilities=a,moves=m,items=items,field=fields,active=active,damage=damage,types=types,sleep=sleep,
        toxic=toxic,toxic_raw_pending=raw_pending,tendency=tendency,tendency_raw_pending=pending),base,accepted | ~valid
