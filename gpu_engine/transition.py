"""First lockstep turn integration; explicitly incomplete, not covered gameplay.

All moves are still flagged placeholders: damaging moves use static type and
power, and only the explicit major-status/self-effect subset is integrated.
Most abilities, switch effects, volatiles and
special move rules are not integrated. Do not use this kernel for strength or
the final throughput/go-no-go table. It supplies the turn/PRNG/array plumbing
needed to integrate those mechanics, with placeholder use returned explicitly.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from . import items as held
from . import weather as climate
from . import absorption as absorb
from . import entry as arrival
from . import combat as traits
from .effects import MoveEffects, from_metadata as effects_from_metadata, stall_success
from .state import FLINCH, BattleState, bench_indices, empty_state, legal_actions


class MoveTable(NamedTuple):
    power: jax.Array
    type: jax.Array
    physical: jax.Array
    accuracy: jax.Array
    always_hit: jax.Array
    priority: jax.Array
    type_num: jax.Array
    type_den: jax.Array
    struggle: jax.Array
    effects: MoveEffects | None = None
    items: held.ItemTable | None = None
    weather: climate.WeatherTable | None = None
    entry: arrival.EntryTable | None = None
    absorption: absorb.AbsorptionTable | None = None
    combat: traits.CombatTable | None = None


def tables_from_metadata(metadata: dict) -> MoveTable:
    """Load static pool tables once on the host, never inside the device step."""
    moves = metadata["moves"]
    if len(moves) != len(metadata["vocabulary"]["moves"]):
        raise ValueError("move metadata/vocabulary mismatch")
    for i, row in enumerate(moves[1:], 1):
        if row["id"] != metadata["vocabulary"]["moves"][i]:
            raise ValueError("move table is not vocabulary ordered")
    chart = np.ones((18, 18), dtype=np.float32)
    chart[1:, 1:] = np.asarray(metadata["type_chart"], dtype=np.float32)
    if not np.all(np.isin(chart, [0, .5, 1, 2])):
        raise ValueError("invalid Gen 3 type chart")
    field = lambda key, default: [default if row is None else row[key] for row in moves]
    accuracy = field("accuracy", 100)
    return MoveTable(
        jnp.asarray(field("base_power", 0), jnp.int32),
        jnp.asarray(field("type", 0), jnp.int32),
        jnp.asarray([x == "Physical" for x in field("category", "Status")]),
        jnp.asarray([100 if x is True else x for x in accuracy], jnp.int32),
        jnp.asarray([x is True for x in accuracy]),
        jnp.asarray(field("priority", 0), jnp.int32),
        jnp.asarray(np.where(chart == .5, 1, chart), jnp.int32),
        jnp.asarray(np.where(chart == .5, 2, 1), jnp.int32),
        jnp.asarray(metadata["vocabulary"]["moves"].index("struggle"), jnp.int32),
        effects_from_metadata(metadata),held.from_metadata(metadata),climate.from_metadata(metadata),
        arrival.from_metadata(metadata),absorb.from_metadata(metadata),traits.from_metadata(metadata))


def state_from_pool(pool, team_indices, *, seed=0, table=None) -> BattleState:
    """Initialize a fixed selection of two real teams per battle without redraws."""
    indices = np.asarray(team_indices)
    if (indices.ndim != 2 or indices.shape[1] != 2 or indices.shape[0] == 0
            or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(pool["species"]))):
        raise ValueError("team_indices must be in-range integers shaped [B,2]")
    state = empty_state(len(indices), seed=seed)
    values = {name: jnp.asarray(pool[name][indices], jnp.int32)
              for name in ("species", "level", "types", "stats", "moves", "pp", "item", "ability")}
    hp = values["stats"][..., 0]
    state = state._replace(**values, hp=hp, max_hp=hp, done=jnp.zeros_like(state.done),
        entry_pending=jnp.ones_like(state.entry_pending))
    return initialize(state,table) if table is not None else state


def _active(value, state, side):
    rows = jnp.arange(state.hp.shape[0])
    return value[rows, side, state.active[rows, side]]


def _set_active(value, state, side, update):
    rows = jnp.arange(state.hp.shape[0])
    return value.at[rows, side, state.active[rows, side]].set(update)


def _lum_update(state,table,eligible):
    if table.items is None:
        return state
    for side in range(2):
        result = held.consume_lum(table.items,_active(state.item,state,side),
            _active(state.status,state,side),_active(state.status_counter,state,side),
            _active(state.confusion_counter,state,side),_active(state.hp,state,side),eligible=eligible)
        for name,value in zip(("item","status","status_counter","confusion_counter"),result):
            state = state._replace(**{name:_set_active(getattr(state,name),state,side,value)})
    return state


def _speeds(state,table):
    rows,sides = jnp.arange(state.hp.shape[0])[:,None],jnp.arange(2)[None,:]
    stats = state.stats[rows,sides,state.active]
    boosts = state.boosts[rows,sides,state.active]
    status = state.status[rows,sides,state.active]
    speed = m.boosted_stat(stats[...,5],boosts[...,4])
    if table.entry is not None:
        ability = state.ability[rows,sides,state.active]
        weather = state.weather
        if table.weather is not None:
            weather = climate.effective(table.weather,weather,ability,state.hp[rows,sides,state.active])
        speed = arrival.speed_modifier(table.entry,ability,weather[:,None],speed)
    # Gen 3 paralysis is applied after the other Speed modifiers.
    live_speed = m.paralyzed_speed(speed,status == m.PAR)
    # faintMessages clears boosts; status/ability handlers no longer run for
    # fainted holders. Their pending instaswitch still has an outgoing speed.
    return jnp.where(state.hp[rows,sides,state.active] > 0,live_speed,stats[...,5])


def initialize(state,table,*,eligible=True):
    """Resolve initial entries before building the first policy observation.

    No turn, residual, hazard or ordinary-step RNG is consumed. A dedicated
    fold-in stream resolves initial speed ties, so explicit initialization and
    step's compatibility initialization produce identical ordinary trajectories.
    """
    if table.entry is None:
        return state
    rows = jnp.arange(state.hp.shape[0])
    live = ~state.done & jnp.asarray(eligible)
    speed = _speeds(state,table)
    coin = jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,314159),(),0,2))(state.keys)
    first = jnp.where(speed[:,0] != speed[:,1],speed[:,1] > speed[:,0],coin == 1).astype(jnp.int32)
    pending_battle = live & jnp.any(state.entry_pending,axis=1)
    for order in range(2):
        side = jnp.where(order == 0,first,1-first)
        pending = live & state.entry_pending[rows,side]
        state,_ = arrival.enter(state,side,table.entry,eligible=pending,hazards=False)
        state = state._replace(entry_pending=state.entry_pending.at[rows,side].set(
            state.entry_pending[rows,side] & ~pending))
    return _lum_update(state,table,pending_battle)


class StepResult(NamedTuple):
    state: BattleState
    invalid_actions: jax.Array      # [B]; invalid live requests freeze that battle
    placeholder_moves: jax.Array   # [B]; every executed move is still a placeholder


def action_mask(state,table):
    """Current request mask with selected ability traps, without cached state.

    Use this rather than state.legal_actions for table-backed battle requests.
    Computing traps here immediately reflects switches/faints and preserves the
    original state byte-for-byte for invalid and terminal requests.
    """
    mask = legal_actions(state)
    if table.combat is None:
        return mask
    rows,sides = jnp.arange(state.hp.shape[0])[:,None],jnp.arange(2)[None,:]
    hp = state.hp[rows,sides,state.active]
    trapped = traits.switch_trapped(table.combat,state.ability[rows,sides,state.active],
        state.types[rows,sides,state.active],hp)
    forced = state.forced_switch | (hp <= 0)
    return jnp.concatenate((mask[...,:4],mask[...,4:] & ~(trapped & ~forced)[...,None]),axis=-1)


def step(state: BattleState, actions: jax.Array, table: MoveTable) -> StepResult:
    """Static-shape two-action turn, with forced replacement as a pseudo-turn.

    Actions must be integer [B,2]. Chance consumes the same fixed block per live
    battle, independent of branches. Dead/invalid battles remain byte-identical.
    The ordinary turn integrates switches, priority/speed, status action gates,
    PP, confusion self-damage, Struggle recoil, static single-hit damage,
    major-status residuals and party exhaustion. Flinch uses volatile bit 0.
    Selected held items are integrated; broader switch/item/ability events remain
    incomplete. Every executed move is still flagged a placeholder.
    """
    if actions.shape != state.active.shape or not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("actions must be integer [B,2]")
    rows = jnp.arange(state.hp.shape[0])
    sides = jnp.arange(2)[None, :]
    active_hp = jnp.take_along_axis(state.hp, state.active[..., None], axis=2)[..., 0]
    forced = state.forced_switch | (active_hp <= 0)
    replacement = jnp.any(forced, axis=1)
    requested = ~replacement[:, None] | forced
    valid = jnp.take_along_axis(action_mask(state,table), jnp.clip(actions, 0, 8)[..., None], axis=2)[..., 0]
    invalid = ~state.done & jnp.any(requested & (~valid | (actions < 0) | (actions > 8)), axis=1)
    live = ~state.done & ~invalid
    splits = jax.vmap(lambda key: jax.random.split(key, 2))(state.keys)
    draws = jax.vmap(lambda key: jax.random.randint(key, (2, 12), 0,
        jnp.array([2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2], jnp.int32)))(splits[:, 1])
    work = initialize(state,table,eligible=live)._replace(keys=splits[:, 0])
    work = _lum_update(work,table,live)
    # All action speeds are frozen before ordinary switches. In Gen 3 a
    # switch-in weather change does not reorder the remaining move queue.
    speed = _speeds(work,table)
    switches = live[:, None] & requested & (actions >= 4)
    bench = bench_indices(state.active)
    target = jnp.take_along_axis(bench, jnp.clip(actions - 4, 0, 4)[..., None], axis=2)[..., 0]
    switch_first = jnp.where(speed[:,0] != speed[:,1],speed[:,1] > speed[:,0],draws[:,0,11] == 1).astype(jnp.int32)
    switch_faint = jnp.zeros_like(state.done)
    installed = jnp.zeros_like(switches)
    callback_speed = jnp.zeros_like(speed)
    for order in range(2):
        side = jnp.where(order == 0,switch_first,1-switch_first)
        old_active = work.active[rows,side]
        continues = jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        do_switch = switches[rows,side] & continues & (~switch_faint | replacement)
        installed = installed.at[rows,side].set(do_switch)
        old_hp = work.hp[rows,side,old_active]
        old_status = work.status[rows,side,old_active]
        cure = old_hp <= 0
        if table.entry is not None:
            cure |= table.entry.natural_cure[work.ability[rows,side,old_active]]
        work = work._replace(status=work.status.at[rows,side,old_active].set(
            jnp.where(do_switch & cure,m.NONE,old_status)))
        counter = work.status_counter[rows,side,old_active]
        work = work._replace(status_counter=work.status_counter.at[rows,side,old_active].set(
            jnp.where(do_switch & (cure | (old_status == m.TOX)),0,counter)))
        for name in ("boosts","volatile","confusion_counter","protect_counter","substitute_hp","choice_move"):
            value = getattr(work,name)
            old = value[rows,side,old_active]
            gate = do_switch[:,None] if old.ndim == 2 else do_switch
            work = work._replace(**{name:value.at[rows,side,old_active].set(jnp.where(gate,0,old))})
        work = work._replace(active=work.active.at[rows,side].set(jnp.where(do_switch,target[rows,side],old_active)),
            forced_switch=work.forced_switch.at[rows,side].set(jnp.where(do_switch,False,work.forced_switch[rows,side])))
        # insertChoice snapshots each incoming callback's Speed now; a later
        # Air Lock/Cloud Nine entrant must not re-sort an already queued event.
        callback_speed = callback_speed.at[rows,side].set(_speeds(work,table)[rows,side])
        if table.entry is not None:
            work,knocked_out = arrival.enter(work,side,table.entry,eligible=do_switch & ~replacement)
            switch_faint |= knocked_out
    if table.entry is not None:
        # Forced instaswitch actions(order3) install ALL entrants before any
        # runSwitch callback(order101). Ordinary switch actions(order103) do
        # not: their callbacks were resolved immediately above. Forced callback
        # order uses each incoming speed frozen at its own insertion.
        first_entry = jnp.where(callback_speed[:,0] != callback_speed[:,1],
            callback_speed[:,1] > callback_speed[:,0],draws[:,1,10] == 1).astype(jnp.int32)
        for order in range(2):
            side = jnp.where(order == 0,first_entry,1-first_entry)
            continues = jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
            eligible = replacement & installed[rows,side] & continues & ~switch_faint
            work,knocked_out = arrival.enter(work,side,table.entry,eligible=eligible)
            # A hazard KO cancels subsequent queued runSwitch callbacks too.
            switch_faint |= knocked_out
    work = _lum_update(work,table,live)
    slot = jnp.clip(actions, 0, 3)
    pp = state.pp[rows[:, None], sides, state.active]
    choice = state.choice_move[rows[:,None],sides,state.active]
    move_ids = state.moves[rows[:,None],sides,state.active]
    usable = ((pp > 0) & ~state.disabled[rows[:, None], sides, state.active]
              & ((choice[...,None] == 0) | (move_ids == choice[...,None])))
    struggle = ~jnp.any(usable, axis=-1)
    moves = state.moves[rows[:, None], sides, state.active, slot]
    moves = jnp.where(struggle, table.struggle, moves)
    priority = table.priority[moves]
    first = jnp.where(priority[:, 0] != priority[:, 1], priority[:, 1] > priority[:, 0],
        jnp.where(speed[:, 0] != speed[:, 1], speed[:, 1] > speed[:, 0], draws[:, 0, 0] == 1)).astype(jnp.int32)
    placeholders = jnp.zeros_like(state.turn)
    moves_cancelled = switch_faint
    protected = jnp.zeros_like(state.active,dtype=jnp.bool_)
    enduring = jnp.zeros_like(protected)
    # Stall lasts through the next ordinary turn unless refreshed by success.
    old_stall = work.protect_counter
    work = work._replace(protect_counter=jnp.where(
        (live & ~replacement)[:,None,None],0,work.protect_counter))
    for order in range(2):
        side = jnp.where(order == 0, first, 1 - first)
        foe = 1 - side
        move = moves[rows, side]
        chance = draws[rows, side]
        eligible = live & ~replacement & ~moves_cancelled & ~switches[rows, side] & (_active(work.hp, work, side) > 0)
        before = m.before_move(_active(work.status, work, side), _active(work.status_counter, work, side),
            _active(work.confusion_counter, work, side), thaw_roll=chance[:, 1],
            paralysis_roll=chance[:, 2], confusion_roll=chance[:, 3],
            flinched=(_active(work.volatile, work, side) & FLINCH) != 0)
        for name, value in (("status", before.status), ("status_counter", before.counter),
                            ("confusion_counter", before.confusion_counter)):
            current = getattr(work, name)
            work = work._replace(**{name: _set_active(current, work, side,
                jnp.where(eligible, value, _active(current, work, side)))})
        own_stats = _active(work.stats, work, side)
        own_boosts = _active(work.boosts, work, side)
        confusion_num,confusion_den = (1,1)
        if table.items is not None:
            confusion_num,confusion_den = held.attack_modifier(table.items,
                _active(work.item,work,side),_active(work.species,work,side),jnp.ones_like(eligible))
        guts = jnp.zeros_like(eligible)
        if table.combat is not None:
            ability = _active(work.ability,work,side)
            guts = table.combat.guts[ability]
            confusion_num,confusion_den = traits.attack_modifier(table.combat,ability,
                _active(work.status,work,side),True,item_num=confusion_num,item_den=confusion_den)
        confusion_damage = m.gen3_damage(level=_active(work.level, work, side), power=40,
            attack=own_stats[:, 1], defense=own_stats[:, 2],
            attack_stage=own_boosts[:, 0], defense_stage=own_boosts[:, 1],
            typeless=True, burned=_active(work.status, work, side) == m.BRN,
            guts=guts,
            attack_num=confusion_num,attack_den=confusion_den,
            roll=85 + chance[:, 5])
        hp = _active(work.hp, work, side)
        hp = jnp.where(eligible & before.confusion_self_hit, jnp.maximum(0, hp - confusion_damage), hp)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        # Gen 3 singles cancels all queued actions after any faint, not only
        # the fainted mon's move. A confusion self-KO therefore skips the foe.
        moves_cancelled |= eligible & (hp == 0)
        acts = eligible & before.can_act & ~moves_cancelled
        chosen_slot = slot[rows, side]
        active = work.active[rows, side]
        old_pp = work.pp[rows, side, active, chosen_slot]
        work = work._replace(pp=work.pp.at[rows, side, active, chosen_slot].set(
            jnp.where(acts & ~struggle[rows, side], jnp.maximum(0, old_pp - 1), old_pp)))
        placeholders += acts.astype(jnp.int32)
        attack_stats = _active(work.stats, work, side)
        defense_stats = _active(work.stats, work, foe)
        attack_boosts = _active(work.boosts, work, side)
        defense_boosts = _active(work.boosts, work, foe)
        physical = table.physical[move]
        attack_num,attack_den = (1,1)
        if table.items is not None:
            attack_num,attack_den = held.attack_modifier(table.items,
                _active(work.item,work,side),_active(work.species,work,side),physical)
        if table.combat is not None:
            attack_num,attack_den = traits.attack_modifier(table.combat,_active(work.ability,work,side),
                _active(work.status,work,side),physical,item_num=attack_num,item_den=attack_den)
        is_struggle = move == table.struggle
        typ = jnp.where(is_struggle, 0, table.type[move])
        own_types = _active(work.types, work, side)
        foe_types = _active(work.types, work, foe)
        num = table.type_num[typ, foe_types[:, 0]] * table.type_num[typ, foe_types[:, 1]]
        den = table.type_den[typ, foe_types[:, 0]] * table.type_den[typ, foe_types[:, 1]]
        immune = num == 0
        if table.entry is not None:
            immune |= (typ == table.entry.ground_type) & table.entry.levitate[_active(work.ability,work,foe)]
        weather_num,weather_den = (1,1)
        accuracy,always_hit = table.accuracy[move],table.always_hit[move]
        if table.weather is not None:
            active_abilities = work.ability[rows[:,None],sides,work.active]
            active_hp = work.hp[rows[:,None],sides,work.active]
            effective_weather = climate.effective(table.weather,work.weather,active_abilities,active_hp)
            weather_num,weather_den = climate.damage_modifier(table.weather,effective_weather,typ)
            accuracy,always_hit = climate.thunder_accuracy(table.weather,effective_weather,move,accuracy,always_hit)
        phase1_num,phase1_den = (1,1)
        if table.absorption is not None:
            phase1_num,phase1_den = absorb.damage_phase1(table.absorption,typ,_active(work.volatile,work,side))
        power = table.power[move]
        if table.combat is not None:
            power = traits.base_power(table.combat,_active(work.ability,work,foe),typ,power)
        damage = m.gen3_damage(level=_active(work.level, work, side), power=power,
            attack=jnp.where(physical, attack_stats[:, 1], attack_stats[:, 3]),
            defense=jnp.where(physical, defense_stats[:, 2], defense_stats[:, 4]),
            attack_stage=jnp.where(physical, attack_boosts[:, 0], attack_boosts[:, 2]),
            defense_stage=jnp.where(physical, defense_boosts[:, 1], defense_boosts[:, 3]),
            physical=physical, critical=chance[:, 4] == 0,
            attack_num=attack_num,attack_den=attack_den,
            weather_num=weather_num,weather_den=weather_den,
            phase1_num=phase1_num,phase1_den=phase1_den,
            stab=jnp.any(own_types == typ[:, None], axis=1) & (typ != 0),
            typeless=typ == 0,
            burned=_active(work.status, work, side) == m.BRN,
            guts=guts,
            effectiveness_num=num, effectiveness_den=den, immune=immune, roll=85 + chance[:, 5])
        hit = m.accuracy_hit(accuracy, attack_boosts[:, 5], defense_boosts[:, 6],
                             chance[:, 6], always_hit=always_hit)
        bypass_sub = jnp.zeros_like(acts)
        if table.effects is not None:
            hit &= ~(protected[rows,foe] & table.effects.protectable[move])
            bypass_sub = table.effects.bypass_sub[move]
        if table.absorption is not None:
            # TryHit follows accuracy/Protect but precedes Substitute. Field
            # weather and self effects do not run a Pokemon-targeted TryHit.
            targets_mon = table.power[move] > 0
            if table.effects is not None:
                targets_mon |= table.effects.status[move] != m.NONE
            absorbed = absorb.try_hit(table.absorption,_active(work.ability,work,foe),move,typ,
                _active(work.status,work,foe),foe_types,_active(work.hp,work,foe),
                _active(work.max_hp,work,foe),_active(work.substitute_hp,work,foe),
                _active(work.volatile,work,foe),eligible=acts & hit & targets_mon)
            work = work._replace(hp=_set_active(work.hp,work,foe,absorbed.hp),
                volatile=_set_active(work.volatile,work,foe,absorbed.volatile))
            hit &= ~absorbed.blocked
        hp = _active(work.hp, work, foe)
        sub_hp = _active(work.substitute_hp,work,foe)
        hits_sub = (sub_hp > 0) & ~bypass_sub
        capacity = jnp.where(hits_sub,sub_hp,jnp.where(enduring[rows,foe],jnp.maximum(0,hp-1),hp))
        dealt = jnp.where(acts & hit,jnp.minimum(capacity,damage),0)
        foe_hp = hp - jnp.where(hits_sub,0,dealt)
        work = work._replace(hp=_set_active(work.hp,work,foe,foe_hp),
            substitute_hp=_set_active(work.substitute_hp,work,foe,
                sub_hp - jnp.where(hits_sub,dealt,0)))
        if table.absorption is not None:
            # Gen 3 thaw is original-catalog Fire, not Hidden Power Fire or
            # dynamically Fire Weather Ball; a Substitute hit does not thaw.
            thawed = ((dealt > 0) & ~hits_sub & table.absorption.fire_thaws[move]
                & (_active(work.status,work,foe) == m.FRZ))
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(thawed,m.NONE,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(thawed,0,_active(work.status_counter,work,foe))))
        # Gen 3 Struggle uses round(actual damage / 4), not max HP / 4.
        recoil = jnp.maximum(1, (dealt + 2) // 4)
        hp = _active(work.hp, work, side)
        hp = jnp.where(is_struggle & (dealt > 0), jnp.maximum(0, hp - recoil), hp)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        if table.effects is not None:
            effects = table.effects
            # Protect/Endure fail if there is no queued move left (including
            # when the foe switched before move execution). A failed use or a
            # non-stalling turn lets the previous stall counter expire.
            stall = (acts & (effects.protect[move] | effects.endure[move])
                & (order == 0) & ~switches[rows,foe])
            count = _active(old_stall,work,side)
            stall &= stall_success(count,chance[:,8])
            protected = protected.at[rows,side].set(stall & effects.protect[move])
            enduring = enduring.at[rows,side].set(stall & effects.endure[move])
            work = work._replace(protect_counter=_set_active(work.protect_counter,work,side,
                jnp.where(stall,jnp.minimum(8,jnp.maximum(2,count*2)),
                          _active(work.protect_counter,work,side))))
            # Self effects use the active actor, never the opposing target.
            hp = _active(work.hp, work, side)
            max_hp = _active(work.max_hp, work, side)
            sub_hp = _active(work.substitute_hp,work,side)
            creates_sub = (acts & effects.substitute[move] & (sub_hp == 0)
                & (max_hp > 1) & (hp * 4 > max_hp))
            cost = max_hp // 4
            hp -= jnp.where(creates_sub,cost,0)
            work = work._replace(substitute_hp=_set_active(work.substitute_hp,work,side,
                jnp.where(creates_sub,cost,sub_hp)))
            restored = jnp.minimum(max_hp,hp + max_hp // 2)
            hp = jnp.where(acts & effects.heal_half[move],restored,hp)
            rest = (acts & effects.rest[move] & (hp < max_hp)
                    & (_active(work.status,work,side) != m.SLP))
            hp = jnp.where(rest,max_hp,hp)
            work = work._replace(hp=_set_active(work.hp,work,side,hp),
                status=_set_active(work.status,work,side,
                    jnp.where(rest,m.SLP,_active(work.status,work,side))),
                status_counter=_set_active(work.status_counter,work,side,
                    jnp.where(rest,3,_active(work.status_counter,work,side))))
            boosts = _active(work.boosts,work,side)
            boosts = jnp.clip(boosts + jnp.where(acts[:,None],effects.self_boosts[move],0),-6,6)
            work = work._replace(boosts=_set_active(work.boosts,work,side,boosts))
            for target_side in range(2):
                old = _active(work.boosts,work,target_side)
                work = work._replace(boosts=_set_active(work.boosts,work,target_side,
                    jnp.where((acts & effects.haze[move])[:,None],0,old)))
            status = effects.status[move]
            target_types = _active(work.types,work,foe)
            immune = jnp.any(effects.status_immunity[status[:,None],target_types],axis=1)
            immune |= effects.thunderwave[move] & jnp.any(target_types == effects.ground_type,axis=1)
            apply_status = (acts & hit & (status != m.NONE) & ~immune
                & (_active(work.hp,work,foe) > 0) & (_active(work.status,work,foe) == m.NONE)
                & ((_active(work.substitute_hp,work,foe) == 0) | effects.bypass_sub[move]))
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(apply_status,status,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(apply_status,jnp.where(status == m.SLP,2 + chance[:,7],0),
                              _active(work.status_counter,work,foe))))
        if table.items is not None:
            locked = _active(work.choice_move,work,side)
            lock = (acts & ~is_struggle & table.items.choice_band[_active(work.item,work,side)] & (locked == 0))
            work = work._replace(choice_move=_set_active(work.choice_move,work,side,jnp.where(lock,move,locked)))
        work = _lum_update(work,table,live & ~replacement)
        if table.weather is not None:
            weather,counter = climate.set_move(work.weather,work.weather_counter,
                table.weather.move_weather[move],eligible=acts)
            work = work._replace(weather=weather,weather_counter=counter)
        if table.entry is not None:
            old_layers = work.spikes[rows,foe]
            work = work._replace(spikes=work.spikes.at[rows,foe].set(jnp.where(
                acts & table.entry.spikes_move[move],jnp.minimum(3,old_layers+1),old_layers)))
        moves_cancelled |= acts & ((foe_hp == 0) | (hp == 0))
    # Residual handler speeds are snapshotted before weather expiry/upkeep.
    end_speed = _speeds(work,table)
    if table.weather is not None:
        ordinary = live & ~replacement & jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        weather,counter = climate.residual_clock(work.weather,work.weather_counter,eligible=ordinary)
        work = work._replace(weather=weather,weather_counter=counter)
        active_abilities = work.ability[rows[:,None],sides,work.active]
        active_hp = work.hp[rows[:,None],sides,work.active]
        effective_weather = climate.effective(table.weather,weather,active_abilities,active_hp)
        # The field weather handler runs Weather on BOTH actives before
        # faintMessages/checkWin. A double last-mon weather KO is a Gen 3 tie.
        for side in range(2):
            hp = climate.chip(table.weather,effective_weather,_active(work.types,work,side),
                _active(work.ability,work,side),_active(work.hp,work,side),
                _active(work.max_hp,work,side),eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,hp))
    # Both selected handlers have order10/priority0. The pinned comparator
    # sorts SPEED before suborder: a faster status precedes a slower heal.
    # At equal speed, both Leftovers(4) precede statuses(6), with independent
    # per-suborder tie coins. Absent handlers are masked no-ops.
    event_sides = jnp.array([0,1,0,1])
    event_suborder = jnp.array([4,4,6,6])
    tie = jnp.concatenate((draws[:,0,9,None] ^ sides,draws[:,1,9,None] ^ sides),axis=1)
    event_keys = -end_speed[:,event_sides]*32 + event_suborder[None,:]*2 + tie
    event_order = jnp.argsort(event_keys,axis=1)
    for order in range(4):
        event = event_order[:,order]
        side = event % 2
        heal_event = event < 2
        battle_continues = jnp.all(jnp.any(work.hp > 0, axis=2), axis=1)
        eligible = live & ~replacement & battle_continues
        hp = _active(work.hp,work,side)
        healed = hp
        if table.items is not None:
            healed = held.leftovers_heal(table.items,_active(work.item,work,side),hp,
                _active(work.max_hp,work,side),eligible=eligible & heal_event)
        residual = m.major_status_residual(hp,
            _active(work.max_hp, work, side), _active(work.status, work, side),
            _active(work.status_counter, work, side), active=eligible & ~heal_event)
        work = work._replace(hp=_set_active(work.hp, work, side,jnp.where(heal_event,healed,residual.hp)),
            status_counter=_set_active(work.status_counter, work, side, residual.counter))
    alive = jnp.any(work.hp > 0, axis=2)
    done = ~jnp.all(alive, axis=1)
    winner = jnp.where(alive[:, 0], 0, jnp.where(alive[:, 1], 1, 2))
    work = work._replace(done=done, winner=jnp.where(done, winner, -1),
        turn=state.turn + (live & ~replacement).astype(jnp.int32),
        forced_switch=(jnp.take_along_axis(work.hp, work.active[..., None], axis=2)[..., 0] <= 0) & alive)
    ordinary = live & ~replacement
    work = work._replace(volatile=jnp.where(ordinary[:, None, None],
        work.volatile & jnp.uint32(~FLINCH & 0xffffffff), work.volatile))
    work = jax.tree.map(lambda old, new: jnp.where(live.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), state, work)
    return StepResult(work, invalid, placeholders)
