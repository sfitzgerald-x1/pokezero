"""Selected Gen 3 ability tables and entry effects; not full coverage.

The inherited Gen 4 runSwitch applies hazards before Ability Start. Ordinary
switches and their runSwitch callbacks resolve one at a time, not together.
Pursuit is dispatched by transition before entry. Direct Baton and phazing
reuse these selected entry/departure events; broader ability events are pending.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import weather
from . import mechanics as m
from . import lane_index as _li


class EntryTable(NamedTuple):
    weather_ability: jax.Array
    intimidate: jax.Array
    natural_cure: jax.Array
    levitate: jax.Array
    attack_drop_immune: jax.Array
    swift_swim: jax.Array
    chlorophyll: jax.Array
    flying_type: jax.Array
    ground_type: jax.Array
    spikes_move: jax.Array
    speed_boost: jax.Array
    truant: jax.Array | None = None
    trace: jax.Array | None = None
    status_cure: jax.Array | None = None
    own_tempo: jax.Array | None = None
    pressure: jax.Array | None = None
    trace_copy: jax.Array | None = None
    spikes_type_count: jax.Array | None = None
    spikes_item_count: jax.Array | None = None
    spikes_boots_item: jax.Array | None = None


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda names: jnp.asarray([name in names for name in vocab["abilities"]])
    climates = {"drizzle":weather.RAIN,"drought":weather.SUN,"sandstream":weather.SAND}
    cures={"waterveil":{m.BRN},"magmaarmor":{m.FRZ},"limber":{m.PAR},
        "immunity":{m.PSN,m.TOX},"insomnia":{m.SLP},"vitalspirit":{m.SLP}}
    return EntryTable(jnp.asarray([climates.get(name,weather.NONE) for name in vocab["abilities"]]),
        flag({"intimidate"}),flag({"naturalcure"}),flag({"levitate"}),
        flag({"clearbody","whitesmoke","hypercutter"}),flag({"swiftswim"}),flag({"chlorophyll"}),
        jnp.asarray(vocab["types"].index("flying")),jnp.asarray(vocab["types"].index("ground")),
        jnp.asarray([name == "spikes" for name in vocab["moves"]]),flag({"speedboost"}),flag({"truant"}),
        flag({"trace"}),jnp.asarray([[status in cures.get(name,set()) for status in range(7)]
            for name in vocab["abilities"]]),flag({"owntempo"}),flag({"pressure"}),flag({"sturdy","naturalcure","oblivious"}),
        jnp.asarray(len(vocab['types']),jnp.int32),jnp.asarray(len(vocab.get('items',[])),jnp.int32),
        jnp.asarray(next((i for i,name in enumerate(vocab.get('items',[])) if name=='heavydutyboots'),0),jnp.int32))


def speed_modifier(table,ability,effective_weather,speed):
    doubled = ((table.swift_swim[ability] & (effective_weather == weather.RAIN))
        | (table.chlorophyll[ability] & (effective_weather == weather.SUN)))
    return speed * jnp.where(doubled,2,1)


def depart(state,side,table,*,eligible):
    """Selected outgoing cleanup shared by voluntary, Baton and drag switches.

    Baton copies its supported fields before calling this. This does not run
    BeforeSwitchOut/Pursuit, which does not apply to a drag or direct Baton.
    """
    rows=jnp.arange(state.hp.shape[0]);active=_li.take(state.active,rows,side)
    hp=_li.take(state.hp,rows,side,active);status=_li.take(state.status,rows,side,active)
    cure=hp<=0
    if table is not None:
        cure |= table.natural_cure[_li.take(state.ability,rows,side,active)]
    # Departure consumers see the current ability first (including a copied
    # Natural Cure). Restore the immutable constructor ability afterwards.
    state=state._replace(ability=_li.put(state.ability,rows,(side,active),jnp.where(
        eligible,_li.take(state.base_ability,rows,side,active),_li.take(state.ability,rows,side,active))))
    # clearVolatile -> setSpecies(baseSpecies) restores the original types on
    # every actual departure, including direct Baton and forced displacement.
    state=state._replace(types=_li.put(state.types,rows,(side,active),jnp.where(
        eligible[:,None],_li.take(state.base_types,rows,side,active),_li.take(state.types,rows,side,active))))
    counter=_li.take(state.status_counter,rows,side,active)
    old_clock=_li.take(state.truant_loaf,rows,side,active)
    state=state._replace(truant_loaf=_li.put(state.truant_loaf,rows,(side,active),jnp.where(eligible,False,old_clock)))
    state=state._replace(status=_li.put(state.status,rows,(side,active),jnp.where(eligible & cure,m.NONE,status)),
        status_counter=_li.put(state.status_counter,rows,(side,active),jnp.where(eligible & (cure | (status==m.TOX)),0,counter)),
        sleep_skipped=_li.put(state.sleep_skipped,rows,(side,active),jnp.where(
            eligible & (cure | (status!=m.SLP)),0,_li.take(state.sleep_skipped,rows,side,active))),
        rest_origin=_li.put(state.rest_origin,rows,(side,active),jnp.where(
            eligible & (cure | (status!=m.SLP)),False,_li.take(state.rest_origin,rows,side,active))))
    for name in ("boosts","volatile","confusion_counter","protect_counter","substitute_hp","choice_move",
                 "last_move","encore_move","encore_counter","charge_move","charge_counter"):
        value=getattr(state,name);old=_li.take(value,rows,side,active)
        gate=eligible[:,None] if old.ndim==2 else eligible
        state=state._replace(**{name:_li.put(value,rows,(side,active),jnp.where(gate,0,old))})
    # Remove both ends of linked move traps. A direct Baton remaps/copies its
    # links to the recipient BEFORE departure, so those transferred links stay.
    own=_li.take(state.trap_source,rows,side,active)
    state=state._replace(trap_source=_li.put(state.trap_source,rows,(side,active),jnp.where(eligible,-1,own)))
    foe=1-side;linked=_li.take(state.trap_source,rows,foe)
    return state._replace(trap_source=_li.put(state.trap_source,rows,(foe,),jnp.where(eligible[:,None] & (linked==active[:,None]),-1,linked)))


def enter(state,side,table,*,eligible,hazards=True,startup=False,_weather_trace=None):
    """One entry event; the caller owns ordering and queued-action cancellation."""
    rows = jnp.arange(state.hp.shape[0])
    active = _li.take(state.active,rows,side)
    foe = 1-side
    foe_active = _li.take(state.active,rows,foe)
    hp = _li.take(state.hp,rows,side,active)
    ability = _li.take(state.ability,rows,side,active)
    grounded = (~jnp.any(_li.take(state.types,rows,side,active) == table.flying_type,axis=-1)
        & ~table.levitate[ability])
    layers = jnp.clip(_li.take(state.spikes,rows,side),0,3)
    if _weather_trace is not None:
        from . import engine_execution_completeness as execution_accounting
        from . import engine_entry_spikes_callbacks as entry_spikes_callbacks
        # Dispatch precedes grounded/Levitate blocking. An absent numerical HP
        # change cannot silently prove no original source output/hook ran.
        _weather_trace,_entry_spikes_pending=entry_spikes_callbacks.begin(_weather_trace,state,side,layers,table,
            requested=eligible & (hp>0) & (layers>0) & jnp.asarray(hazards))
    numerator = jnp.array([0,3,4,6],jnp.int32)[layers]
    amount = jnp.maximum(1,_li.take(state.max_hp,rows,side,active)*numerator//24)
    hit = eligible & (hp > 0) & grounded & (layers > 0) & hazards
    next_hp = jnp.where(hit,jnp.maximum(0,hp-amount),hp)
    knocked_out = hit & (next_hp == 0)
    state = state._replace(hp=_li.put(state.hp,rows,(side,active),next_hp))
    if _weather_trace is not None:
        _weather_trace=entry_spikes_callbacks.finish(_weather_trace,_entry_spikes_pending,_li.take(state.hp,rows,side,active))
    # Gen3 runs status SwitchIn even after a hazard KO, before faintMessages.
    state=restore_sleep(state,side,eligible=eligible)
    starts = eligible & (next_hp > 0)
    if _weather_trace is not None:
        from . import engine_weather_callbacks as callbacks
        _weather_trace=callbacks.unattributed_ability(_weather_trace,
            eligible=starts & (ability!=_li.take(state.base_ability,rows,side,active)))
    if _weather_trace is not None and table.pressure is not None:
        from . import engine_pressure_callbacks as pressure_callbacks
        # Effective Gen3 onStart is owner-only addSplit. Use the actual
        # pre-Trace Start ability, after hazards/status SwitchIn, never a copy.
        _weather_trace=pressure_callbacks.entry_pressure(_weather_trace,side,active,ability,
            eligible=starts & table.pressure[ability])
    if table.truant is not None:
        old_clock=_li.take(state.truant_loaf,rows,side,active)
        # Gen3 onSwitchIn: turn0 starts ACT; every later entry starts LOAF.
        # The caller supplies startup context rather than guessing from the
        # step's completed-turn counter during a first-turn action switch.
        clock=table.truant[ability] & ~jnp.asarray(startup)
        state=state._replace(truant_loaf=_li.put(state.truant_loaf,rows,(side,active),jnp.where(eligible,starts & clock,old_clock)))
    # Gen3 Trace attempts only at Ability Start, after hazards/SwitchIn. Keep
    # the pre-Trace current ability for Start handlers below: gaining weather,
    # Intimidate or Truant through Trace does not run their Start/SwitchIn.
    foe_hp = _li.take(state.hp,rows,foe,foe_active)
    if table.trace is not None:
        copied=starts & table.trace[ability] & (foe_hp>0)
        if _weather_trace is not None:
            from . import engine_trace_callbacks as trace_callbacks
            _trace_foe_ability=_li.take(state.ability,rows,foe,foe_active)
            _weather_trace,_trace_ticket=execution_accounting.open_attempt(_weather_trace,
                execution_accounting.Site.TRACE_COPY,side,active=copied)
        state=state._replace(ability=_li.put(state.ability,rows,(side,active),jnp.where(
            copied,_li.take(state.ability,rows,foe,foe_active),ability)))
        if _weather_trace is not None:
            _weather_trace=trace_callbacks.entry_trace(_weather_trace,side,active,foe_active,
                ability,_li.take(state.ability,rows,side,active),_trace_foe_ability,table,eligible=copied)
            _weather_trace=execution_accounting.capture_trace_copy(_weather_trace,_trace_ticket,requested=copied)
    current=_li.take(state.ability,rows,side,active)
    status=_li.take(state.status,rows,side,active)
    if table.status_cure is not None:
        cure=starts & table.status_cure[current,status]
        state=state._replace(status=_li.put(state.status,rows,(side,active),jnp.where(cure,m.NONE,status)),
            status_counter=_li.put(state.status_counter,rows,(side,active),jnp.where(
                cure,0,_li.take(state.status_counter,rows,side,active))),
            sleep_skipped=_li.put(state.sleep_skipped,rows,(side,active),jnp.where(
                cure,0,_li.take(state.sleep_skipped,rows,side,active))),
            rest_origin=_li.put(state.rest_origin,rows,(side,active),jnp.where(
                cure,False,_li.take(state.rest_origin,rows,side,active))))
    if table.own_tempo is not None:
        if _weather_trace is not None:
            from . import engine_kernel_frame_risk as kernel_risks
            # Original Own Tempo Update prints/removes only an existing
            # confusion volatile. Zero is absent in this state representation;
            # every nonzero counter (including malformed negatives) stays unknown.
            _weather_trace=kernel_risks.branch_on_tape(_weather_trace,kernel_risks.Branch.ENTRY_CONFUSION_CURE,
                starts & table.own_tempo[current] & (_li.take(state.confusion_counter,rows,side,active)!=0),active=eligible)
        state=state._replace(confusion_counter=_li.put(state.confusion_counter,rows,(side,active),jnp.where(
            starts & table.own_tempo[current],0,_li.take(state.confusion_counter,rows,side,active))))
    requested = table.weather_ability[ability]
    change = (starts & (requested != weather.NONE)
        & ((state.weather != requested) | (state.weather_counter > 0)))
    if _weather_trace is not None:
        old_weather,old_counter=state.weather,state.weather_counter
    state = state._replace(weather=jnp.where(change,requested,state.weather),
        weather_counter=jnp.where(change,0,state.weather_counter))
    if _weather_trace is not None:
        from . import engine_weather_callbacks as callbacks
        _weather_trace=callbacks.entry_weather(_weather_trace,side,active,old_weather,state.weather,
            old_counter,state.weather_counter,eligible=change)
    if _weather_trace is not None:
        # PRE-Trace Start ability, before foe-live/Substitute/immunity/cap
        # branches. The source can print controls with zero numeric change.
        from . import engine_intimidate_callbacks as intimidate_callbacks
        _intimidate_requested=starts & table.intimidate[ability]
        _intimidate_target_ability=_li.take(state.ability,rows,foe,foe_active)
        _intimidate_substitute=_li.take(state.substitute_hp,rows,foe,foe_active)
        _intimidate_initial=(intimidate_callbacks._constructor(_weather_trace)&(foe_hp>0)
            &(_intimidate_substitute==0)&~table.attack_drop_immune[_intimidate_target_ability]
            &(_li.take(state.boosts,rows,foe,foe_active,0)==0))
        # Ordinary/blocked attempts retain their own requested obligation.
        # Only the SAME positive write with a fresh installation can capture it.
        _weather_trace,_intimidate_ordinary_ticket=execution_accounting.open_attempt(_weather_trace,
            execution_accounting.Site.INTIMIDATE_ATTEMPT,side,
            active=_intimidate_requested&~_intimidate_initial)
        _weather_trace,_intimidate_ticket=execution_accounting.open_attempt(_weather_trace,
            execution_accounting.Site.INTIMIDATE_INITIAL,side,
            active=_intimidate_requested&_intimidate_initial)
    blocked = ((_li.take(state.substitute_hp,rows,foe,foe_active) > 0)
        | table.attack_drop_immune[_li.take(state.ability,rows,foe,foe_active)])
    drops = starts & table.intimidate[ability] & (foe_hp > 0) & ~blocked
    old = _li.take(state.boosts,rows,foe,foe_active,0)
    boosts = _li.put(state.boosts,rows,(foe,foe_active,0),jnp.where(drops,jnp.maximum(-6,old-1),old))
    if _weather_trace is not None:
        _weather_trace=intimidate_callbacks.record_device(_weather_trace,_intimidate_ticket,side,active,foe_active,
            intimidate_callbacks.Write(ability,_intimidate_target_ability,foe_hp,_intimidate_substitute,
                old,_li.take(boosts,rows,foe,foe_active,0)),table,requested=_intimidate_requested&_intimidate_initial)
        _weather_trace=intimidate_callbacks.record_ordinary_device(_weather_trace,_intimidate_ordinary_ticket,side,active,foe_active,
            intimidate_callbacks.Write(ability,_intimidate_target_ability,foe_hp,_intimidate_substitute,
                old,_li.take(boosts,rows,foe,foe_active,0)),table,requested=_intimidate_requested&~_intimidate_initial)
        return state._replace(boosts=boosts),knocked_out,_weather_trace
    return state._replace(boosts=boosts),knocked_out


def restore_sleep(state,side,*,eligible):
    """Gen3 status SwitchIn after hazards, before Ability Start."""
    rows=jnp.arange(state.hp.shape[0]);active=_li.take(state.active,rows,side)
    skipped=_li.take(state.sleep_skipped,rows,side,active)
    sleeping=_li.take(state.status,rows,side,active)==m.SLP
    counter=_li.take(state.status_counter,rows,side,active)
    refunded=counter+skipped
    refunded=jnp.where(_li.take(state.rest_origin,rows,side,active),jnp.minimum(3,refunded),refunded)
    return state._replace(status_counter=_li.put(state.status_counter,rows,(side,active),jnp.where(eligible & sleeping,refunded,counter)),
        sleep_skipped=_li.put(state.sleep_skipped,rows,(side,active),jnp.where(eligible,0,skipped)))
