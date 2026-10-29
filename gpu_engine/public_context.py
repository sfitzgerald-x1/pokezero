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
from . import public_active as active_api
from . import public_damage as damage_api
from . import public_types as type_api


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


def compile_context(source, abilities, moves, *, items=None, item_belief_narrowing=False, field_catalog=None, active_catalog=None, damage_tables=None, type_catalog=None):
    if type_catalog is not None and field_catalog is None:
        raise ValueError('live-type composition requires the ordered field context')
    if type(item_belief_narrowing) is not bool:
        raise ValueError("item narrowing must be a frozen boolean")
    if items is None and field_catalog is not None:
        raise ValueError("field composition requires the raw item-enabled context")
    if active_catalog is not None and field_catalog is None:
        raise ValueError("active composition requires the ordered field context")
    if damage_tables is not None and field_catalog is None:
        raise ValueError("damage composition requires the ordered field context")
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
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest,dtype=">u4").astype(np.uint32))
    return ContextTables(abilities.device,moves.device,jnp.asarray(flags),items.device,
        jnp.asarray(item_flags),jnp.asarray(item_belief_narrowing),binding,
        field_catalog.device if field_catalog is not None else None,
        active_catalog.device if active_catalog is not None else None,damage_tables,
        type_catalog.device if type_catalog is not None else None)


def initialize(beliefs, tables, *, species_binding=None):
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
    batch = beliefs.species.shape[0]
    return ContextState(ability.initialize(batch, tables.abilities,species_binding=species_binding), move.initialize(batch, tables.moves),
        beliefs.hard, beliefs.has_evidence,
        item.initialize(batch,tables.items,species_binding=species_binding) if tables.items is not None else None,
        jnp.broadcast_to(tables.binding,(batch,8)) if tables.binding is not None else None,
        field_api.initialize(batch,tables.field) if tables.field is not None else None,
        active_api.initialize(batch,tables.active,species_binding=species_binding) if tables.active is not None else None,
        damage_api.initialize(batch,tables.damage,species_binding=species_binding) if tables.damage is not None else None,
        type_api.initialize(batch,tables.types,species_binding=species_binding) if tables.types is not None else None)


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
    return base._replace(hard=hard,has_evidence=evidence),ok


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
            active_kind, baton_kind, upkeep_kind, damage_kind, type_kind):
    """One resolved typed event per battle; the outer fold commits atomically."""
    k, actor_side, actor_slot, side, slot = row[:, 0], row[:, 4], row[:, 5], row[:, 6], row[:, 7]
    data = row[:, 8:]
    a, m = context.abilities, context.moves
    a, aa = ability.arrive(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & arrival)
    m, ma = move.arrive(m, tables.moves, side=side, valid=valid & arrival)
    a, fa = ability.faint(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & faint)
    is_move, is_ability, is_trace, is_transform = (k == x for x in (move_kind, ability_kind, trace_kind, transform_kind))
    transformed, pressure, ca = ability.move_context(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], valid=valid & is_move)
    update = move.record_move(m, base, mons, tables.moves, source_tables,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], move_id=data[:, 1],
        caller=data[:, 2], has_from=data[:, 3] == 1, transformed=transformed,
        opposing_pressure=pressure, wire_target_side=side, valid=valid & is_move & ca)
    m, base = update.state, update.beliefs
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
    accepted = jnp.where(arrival, aa & ma, jnp.where(faint, fa, jnp.where(is_move, update.admitted,
        jnp.where(is_ability, rev.admitted, jnp.where(is_trace, ta, jnp.where(is_transform, tr, True))))))
    items=context.items
    if items is not None:
        raw=k==raw_kind
        items,ra=item.begin_event(items,mons,tables.items,item_event=data[:,0]==1,trick=data[:,0]==2,
            actor_side=actor_side,actor_slot=actor_slot,actor_encounter=data[:,1],
            target_side=side,target_slot=slot,target_encounter=data[:,2],valid=valid&raw)
        items,ia=item.record(items,mons,tables.items,side=side,slot=slot,encounter=data[:,0],
            item_id=data[:,1],operation=data[:,2],item_binding=tables.items.binding,valid=valid&(k==item_kind))
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
    return context._replace(abilities=a,moves=m,items=items,field=fields,active=active,damage=damage,types=types),base,accepted | ~valid
