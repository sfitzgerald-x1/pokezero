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
from . import pressure as pp_cost
from . import sustain as recovery
from . import damage_moves as special_hits
from . import variable_moves as variable
from . import pursuit as intercepts
from . import baton as passing
from . import phazing as dragging
from . import linked_traps as linking
from . import critical as crits
from . import secondary as secondaries
from . import sleep_talk as talking
from . import self_destruct as exploding
from . import encore as repeating
from .effects import MoveEffects, from_metadata as effects_from_metadata, stall_success
from .state import FLINCH, BattleState, bench_indices, empty_state, legal_actions, usable_moves


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
    pressure: pp_cost.PressureTable | None = None
    sustain: recovery.SustainTable | None = None
    damage_moves: special_hits.DamageMoveTable | None = None
    variable_moves: variable.VariableMoveTable | None = None
    pursuit: intercepts.PursuitTable | None = None
    baton: passing.BatonTable | None = None
    phazing: dragging.PhazingTable | None = None
    linked_traps: linking.TrapTable | None = None
    critical: crits.CriticalTable | None = None
    secondary: secondaries.SecondaryTable | None = None
    sleep_talk: talking.SleepTalkTable | None = None
    self_destruct: exploding.SelfDestructTable | None = None
    encore: repeating.EncoreTable | None = None


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
        arrival.from_metadata(metadata),absorb.from_metadata(metadata),traits.from_metadata(metadata),
        pp_cost.from_metadata(metadata),recovery.from_metadata(metadata),special_hits.from_metadata(metadata),
        variable.from_metadata(metadata),intercepts.from_metadata(metadata),passing.from_metadata(metadata),
        dragging.from_metadata(metadata),linking.from_metadata(metadata),crits.from_metadata(metadata),
        secondaries.from_metadata(metadata),talking.from_metadata(metadata),exploding.from_metadata(metadata),
        repeating.from_metadata(metadata))


def state_from_pool(pool, team_indices, *, seed=0, table=None) -> BattleState:
    """Initialize a fixed selection of two real teams per battle without redraws."""
    indices = np.asarray(team_indices)
    if (indices.ndim != 2 or indices.shape[1] != 2 or indices.shape[0] == 0
            or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(pool["species"]))):
        raise ValueError("team_indices must be in-range integers shaped [B,2]")
    state = empty_state(len(indices), seed=seed)
    values = {name: jnp.asarray(pool[name][indices], jnp.int32)
              for name in ("species", "level", "types", "stats", "moves", "pp", "item", "ability",
                           "hidden_power_type", "hidden_power_power")}
    # The pinned random-team generator returns team objects without happiness;
    # Pokemon's constructor defaults these to 255 (not Frustration text-import 0).
    if "happiness" in pool:
        values["happiness"] = jnp.asarray(pool["happiness"][indices], jnp.int32)
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
        state=state._replace(sleep_skipped=_set_active(state.sleep_skipped,state,side,
            jnp.where(result.status==m.SLP,_active(state.sleep_skipped,state,side),0)))
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
    """Static-shape request: ordinary turn, pending residual, or replacement.

    Actions must be integer [B,2]. Ordinary requests draw a fixed chance block;
    suspended-turn replacements retain it until residuals finish. Standalone
    replacements do not tick an ordinary turn. Dead/invalid states stay identical.
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
    resuming = live & state.pending_residual
    splits = jax.vmap(lambda key: jax.random.split(key, 2))(state.keys)
    draws = jax.vmap(lambda key: jax.random.randint(key, (2, 12), 0,
        jnp.array([2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2], jnp.int32)))(splits[:, 1])
    draws = jnp.where(resuming[:,None,None],state.residual_draws,draws)
    # Extend the ordinary chance block without changing its existing twelve
    # columns. One fixed roll per seat gives unbiased 1..5-bench selection.
    phazing_draws=jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,271828),(2,),0,60))(splits[:,1])
    phazing_draws=jnp.where(resuming[:,None],state.phazing_draws,phazing_draws)
    secondary_draws=jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,1618033),(2,),0,100))(splits[:,1])
    secondary_draws=jnp.where(resuming[:,None],state.secondary_draws,secondary_draws)
    sleep_talk_draws=jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,1414213),(2,),0,60))(splits[:,1])
    sleep_talk_draws=jnp.where(resuming[:,None],state.sleep_talk_draws,sleep_talk_draws)
    # Replacement requests do not consume another ordinary-turn chance block.
    # A separate counted fold-in distinguishes repeated hazard-KO entry ties.
    entry_ties = jax.vmap(lambda key,count:jax.random.randint(
        jax.random.fold_in(key,jnp.uint32(982451653)+count.astype(jnp.uint32)),(2,),0,2))(
            state.keys,state.replacement_count)
    # nextTurn increments activeTurns before an ordinary request. switchIn
    # later resets an entrant to zero, excluding its entry turn's Speed Boost.
    age = state.active_turns + ((live & ~replacement)[:,None] & (active_hp > 0)).astype(jnp.int32)
    work = initialize(state,table,eligible=live)._replace(
        keys=jnp.where(resuming[:,None],state.keys,splits[:,0]),active_turns=age)
    work = _lum_update(work,table,live)
    # All action speeds are frozen before ordinary switches. In Gen 3 a
    # switch-in weather change does not reorder the remaining move queue.
    speed = _speeds(work,table)
    switches = live[:, None] & requested & (actions >= 4)
    # An interrupted direct Baton resumes the already chosen foe slot, never
    # the ignored action supplied alongside the forced replacement request.
    saved_actor = jnp.where(live & state.pending_residual,state.queued_actor,-1)
    move_actions = jnp.where((saved_actor[:,None] == sides),state.queued_actions,actions)
    slot = jnp.clip(move_actions, 0, 3)
    struggle = ~jnp.any(usable_moves(state), axis=-1)
    moves = state.moves[rows[:, None], sides, state.active, slot]
    moves = jnp.where(struggle, table.struggle, moves)
    pursuit_used = jnp.zeros_like(switches)
    pursuit_faint = jnp.zeros_like(state.done)
    if table.pursuit is not None:
        for side in range(2):
            initial_status = _active(state.status,state,side)
            eligible = (live & ~replacement & (actions[:,side]<4) & switches[:,1-side]
                & ~struggle[:,side] & (initial_status!=m.SLP) & (initial_status!=m.FRZ))
            work,used,fainted = intercepts.intercept(work,side,slot[:,side],moves[:,side],
                draws[:,side],table,eligible=eligible)
            pursuit_used = pursuit_used.at[:,side].set(used)
            pursuit_faint |= fainted
    bench = bench_indices(state.active)
    target = jnp.take_along_axis(bench, jnp.clip(actions - 4, 0, 4)[..., None], axis=2)[..., 0]
    switch_coin = jnp.where(resuming,entry_ties[:,0],draws[:,0,11])
    switch_first = jnp.where(speed[:,0] != speed[:,1],speed[:,1] > speed[:,0],switch_coin == 1).astype(jnp.int32)
    switch_faint = jnp.zeros_like(state.done)
    installed = jnp.zeros_like(switches)
    callback_speed = jnp.zeros_like(speed)
    for order in range(2):
        side = jnp.where(order == 0,switch_first,1-switch_first)
        old_active = work.active[rows,side]
        continues = jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        do_switch = switches[rows,side] & continues & (~switch_faint | replacement)
        installed = installed.at[rows,side].set(do_switch)
        work = passing.copy_to_recipient(work,side,target[rows,side],
            eligible=do_switch & work.baton_pass[rows,side])
        work=arrival.depart(work,side,table.entry,eligible=do_switch)
        work = work._replace(active=work.active.at[rows,side].set(jnp.where(do_switch,target[rows,side],old_active)),
            baton_pass=work.baton_pass.at[rows,side].set(work.baton_pass[rows,side] & ~do_switch),
            forced_switch=work.forced_switch.at[rows,side].set(jnp.where(do_switch,False,work.forced_switch[rows,side])),
            active_turns=work.active_turns.at[rows,side].set(jnp.where(do_switch,0,work.active_turns[rows,side])))
        # insertChoice snapshots each incoming callback's Speed now; a later
        # Air Lock/Cloud Nine entrant must not re-sort an already queued event.
        callback_speed = callback_speed.at[rows,side].set(_speeds(work,table)[rows,side])
        if table.entry is not None:
            work,knocked_out = arrival.enter(work,side,table.entry,eligible=do_switch & ~replacement)
            switch_faint |= knocked_out
        else:
            work=arrival.restore_sleep(work,side,eligible=do_switch)
    if table.entry is not None:
        # Forced instaswitch actions(order3) install ALL entrants before any
        # runSwitch callback(order101). Ordinary switch actions(order103) do
        # not: their callbacks were resolved immediately above. Forced callback
        # order uses each incoming speed frozen at its own insertion.
        first_entry = jnp.where(callback_speed[:,0] != callback_speed[:,1],
            callback_speed[:,1] > callback_speed[:,0],
            jnp.where(resuming,entry_ties[:,1],draws[:,1,10]) == 1).astype(jnp.int32)
        for order in range(2):
            side = jnp.where(order == 0,first_entry,1-first_entry)
            continues = jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
            eligible = replacement & installed[rows,side] & continues & ~switch_faint
            work,knocked_out = arrival.enter(work,side,table.entry,eligible=eligible)
            # A hazard KO cancels subsequent queued runSwitch callbacks too.
            switch_faint |= knocked_out
    work = _lum_update(work,table,live)
    priority = table.priority[moves]
    first = jnp.where(priority[:, 0] != priority[:, 1], priority[:, 1] > priority[:, 0],
        jnp.where(speed[:, 0] != speed[:, 1], speed[:, 1] > speed[:, 0], draws[:, 0, 0] == 1)).astype(jnp.int32)
    placeholders = jnp.sum(pursuit_used,axis=1,dtype=jnp.int32)
    # A Pursuit KO still installs the chosen switch in Gen 3. It cancels moves,
    # not that requeued switch; separate it from ordinary hazard cancellation.
    moves_cancelled = switch_faint | pursuit_faint
    protected = jnp.zeros_like(state.active,dtype=jnp.bool_)
    enduring = jnp.zeros_like(protected)
    # Stall lasts through the next ordinary turn unless refreshed by success.
    old_stall = work.protect_counter
    saved_stall = jnp.where(state.pending_residual[:,None],state.queued_stall,
        old_stall[rows[:,None],sides,work.active])
    queued_actor = jnp.full_like(state.queued_actor,-1)
    paused = jnp.zeros_like(live)
    displaced=jnp.zeros_like(state.active,dtype=jnp.bool_)
    work = work._replace(protect_counter=jnp.where(
        (live & ~replacement)[:,None,None],0,work.protect_counter))
    for order in range(2):
        side = jnp.where(order == 0, first, 1 - first)
        foe = 1 - side
        move = moves[rows, side]
        chosen_slot=slot[rows,side]
        move,chosen_slot=repeating.override(work,side,move,chosen_slot,struggle=struggle[rows,side])
        caller_move=move
        chance = draws[rows, side]
        move_context = ~replacement | (state.pending_residual & (saved_actor == side))
        eligible = (live & move_context & ~paused & ~moves_cancelled & ~switches[rows, side]
            & ~pursuit_used[rows,side] & ~displaced[rows,side] & (_active(work.hp, work, side) > 0))
        sleep_usable=jnp.zeros_like(live);early_bird=jnp.zeros_like(live)
        if table.sleep_talk is not None:
            sleep_usable=table.sleep_talk.move[move]
            early_bird=table.sleep_talk.early_bird[_active(work.ability,work,side)]
        before = m.before_move(_active(work.status, work, side), _active(work.status_counter, work, side),
            _active(work.confusion_counter, work, side), thaw_roll=chance[:, 1],
            paralysis_roll=chance[:, 2], confusion_roll=chance[:, 3],
            flinched=(_active(work.volatile, work, side) & FLINCH) != 0,
            sleep_usable=sleep_usable,early_bird=early_bird)
        work=talking.before_move(work,side,before.status,eligible=eligible,sleep_usable=sleep_usable)
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
        active = work.active[rows, side]
        old_pp = work.pp[rows, side, active, chosen_slot]
        acts &= struggle[rows,side] | (old_pp>0)
        used_pp = 1
        if table.pressure is not None:
            used_pp = pp_cost.cost(table.pressure,move,_active(work.types,work,side),
                _active(work.ability,work,foe),_active(work.hp,work,foe))
        work = work._replace(pp=work.pp.at[rows, side, active, chosen_slot].set(
            jnp.where(acts & ~struggle[rows, side], jnp.maximum(0, old_pp - used_pp), old_pp)))
        work=work._replace(last_move=_set_active(work.last_move,work,side,
            jnp.where(acts,caller_move,_active(work.last_move,work,side))))
        placeholders += acts.astype(jnp.int32)
        caller_acts=acts
        if table.sleep_talk is not None:
            called,usable=talking.choose(table.sleep_talk,_active(work.moves,work,side),
                _active(work.pp,work,side),sleep_talk_draws[rows,side])
            executes=((_active(work.status,work,side)==m.SLP) & usable
                & (_active(work.encore_counter,work,side)==0)
                & (_active(work.choice_move,work,side)==0))
            move=jnp.where(sleep_usable,called,move)
            acts &= ~sleep_usable | executes
            placeholders += (acts & sleep_usable).astype(jnp.int32)
        explodes=jnp.zeros_like(acts)
        if table.self_destruct is not None:
            blocked=exploding.blocked(table.self_destruct,move,
                work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
            acts &= ~blocked
            explodes=acts & table.self_destruct.move[move]
        attack_stats = _active(work.stats, work, side)
        defense_stats = _active(work.stats, work, foe)
        attack_boosts = _active(work.boosts, work, side)
        defense_boosts = _active(work.boosts, work, foe)
        physical = table.physical[move]
        power,typ = table.power[move],table.type[move]
        if table.variable_moves is not None:
            power,typ,physical = variable.resolve(table.variable_moves,move,power,typ,physical,
                _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                _active(work.happiness,work,side))
        attack_num,attack_den = (1,1)
        if table.items is not None:
            attack_num,attack_den = held.attack_modifier(table.items,
                _active(work.item,work,side),_active(work.species,work,side),physical)
        if table.combat is not None:
            attack_num,attack_den = traits.attack_modifier(table.combat,_active(work.ability,work,side),
                _active(work.status,work,side),physical,item_num=attack_num,item_den=attack_den)
        is_struggle = move == table.struggle
        typ = jnp.where(is_struggle, 0, typ)
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
        if table.combat is not None:
            power = traits.base_power(table.combat,_active(work.ability,work,foe),typ,power)
        critical = chance[:,4] == 0
        if table.critical is not None:
            critical = crits.is_critical(table.critical,move,_active(work.ability,work,foe),chance[:,4])
        damage = m.gen3_damage(level=_active(work.level, work, side), power=power,
            attack=jnp.where(physical, attack_stats[:, 1], attack_stats[:, 3]),
            defense=jnp.where(physical, defense_stats[:, 2], defense_stats[:, 4]),
            attack_stage=jnp.where(physical, attack_boosts[:, 0], attack_boosts[:, 2]),
            defense_stage=jnp.where(physical, defense_boosts[:, 1], defense_boosts[:, 3]),
            physical=physical, critical=critical,halve_defense=explodes & physical,
            attack_num=attack_num,attack_den=attack_den,
            weather_num=weather_num,weather_den=weather_den,
            phase1_num=phase1_num,phase1_den=phase1_den,
            stab=jnp.any(own_types == typ[:, None], axis=1) & (typ != 0),
            typeless=typ == 0,
            burned=_active(work.status, work, side) == m.BRN,
            guts=guts,
            effectiveness_num=num, effectiveness_den=den, immune=immune, roll=85 + chance[:, 5])
        if table.damage_moves is not None:
            damage=special_hits.resolve_fixed(table.damage_moves,move,_active(work.level,work,side),
                damage,immune=immune)
        hit = m.accuracy_hit(accuracy, attack_boosts[:, 5], defense_boosts[:, 6],
                             chance[:, 6], always_hit=always_hit)
        # TryMove (including Damp) succeeds before self-KO, which occurs even
        # if the subsequent target hit misses, is protected, immune or a sub.
        # Damage inputs were resolved while the user's ability was still active.
        work=work._replace(hp=_set_active(work.hp,work,side,
            jnp.where(explodes,0,_active(work.hp,work,side))))
        bypass_sub = jnp.zeros_like(acts)
        if table.effects is not None:
            hit &= ~(protected[rows,foe] & table.effects.protectable[move])
            bypass_sub = table.effects.bypass_sub[move]
        if table.absorption is not None:
            # TryHit follows accuracy/Protect but precedes Substitute. Field
            # weather and self effects do not run a Pokemon-targeted TryHit.
            targets_mon = power > 0
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
        # Gen 3 floors recoil from actual damage, including capped sub damage.
        recoil=jnp.where(is_struggle & (dealt>0),jnp.maximum(1,dealt//4),0)
        if table.damage_moves is not None:
            recoil=special_hits.recoil_damage(table.damage_moves,move,_active(work.ability,work,side),dealt)
        hp = _active(work.hp, work, side)
        hp = jnp.maximum(0,hp-recoil)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        if table.effects is not None:
            effects = table.effects
            # Protect/Endure fail if there is no queued move left (including
            # when the foe switched before move execution). A failed use or a
            # non-stalling turn lets the previous stall counter expire.
            stall = (acts & (effects.protect[move] | effects.endure[move])
                & (order == 0) & ~state.pending_residual & ~switches[rows,foe])
            count = jnp.where(state.pending_residual,saved_stall[rows,side],_active(old_stall,work,side))
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
                    jnp.where(rest,3,_active(work.status_counter,work,side))),
                sleep_skipped=_set_active(work.sleep_skipped,work,side,
                    jnp.where(rest,0,_active(work.sleep_skipped,work,side))))
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
            if table.secondary is not None:
                immune |= ~secondaries.status_allowed(table.secondary,status,target_types,
                    _active(work.ability,work,foe))
            immune |= effects.thunderwave[move] & jnp.any(target_types == effects.ground_type,axis=1)
            apply_status = (acts & hit & (status != m.NONE) & ~immune
                & (_active(work.hp,work,foe) > 0) & (_active(work.status,work,foe) == m.NONE)
                & ((_active(work.substitute_hp,work,foe) == 0) | effects.bypass_sub[move]))
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(apply_status,status,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(apply_status,jnp.where(status == m.SLP,2 + chance[:,7],0),
                              _active(work.status_counter,work,foe))),
                sleep_skipped=_set_active(work.sleep_skipped,work,foe,
                    jnp.where(apply_status,0,_active(work.sleep_skipped,work,foe))))
        if table.sustain is not None:
            seeded = recovery.seed_target(table.sustain,move,_active(work.types,work,foe),
                _active(work.hp,work,foe),_active(work.substitute_hp,work,foe),
                _active(work.volatile,work,foe),eligible=acts & hit)
            pending = work.wish_counter[rows,side]
            work = work._replace(volatile=_set_active(work.volatile,work,foe,seeded),
                wish_counter=work.wish_counter.at[rows,side].set(jnp.where(
                    acts & table.sustain.wish[move] & (pending == 0),2,pending)))
        if table.items is not None:
            locked = _active(work.choice_move,work,side)
            lock = (caller_acts & (caller_move!=table.struggle)
                & table.items.choice_band[_active(work.item,work,side)] & (locked == 0))
            work = work._replace(choice_move=_set_active(work.choice_move,work,side,jnp.where(lock,caller_move,locked)))
        if table.linked_traps is not None:
            work=linking.apply(work,side,move,table.linked_traps,eligible=acts & hit)
        if table.encore is not None:
            # Duration uses the existing uniform [0,4) status-duration column:
            # a dispatched Encore cannot also inflict a primary sleep status.
            queued=(order==0) & ~state.pending_residual & ~switches[rows,foe] & ~displaced[rows,foe]
            work=repeating.apply(work,foe,move,chance[:,7],table.encore,eligible=acts & hit,queued=queued)
        if table.secondary is not None:
            effective_weather=work.weather
            if table.weather is not None:
                effective_weather=climate.effective(table.weather,work.weather,
                    work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
            work=secondaries.apply(work,side,move,secondary_draws[rows,side],table.secondary,
                eligible=acts & hit & (dealt>0) & ~hits_sub,effective_weather=effective_weather)
        if table.absorption is not None:
            # Gen 3 onDamagingHit follows secondaries: a frozen target cannot
            # receive this Fire hit's burn, then thaws. Original-catalog Fire
            # only (not Hidden Power Fire); Substitute damage does not thaw.
            thawed = ((dealt > 0) & ~hits_sub & table.absorption.fire_thaws[move]
                & (_active(work.status,work,foe) == m.FRZ))
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(thawed,m.NONE,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(thawed,0,_active(work.status_counter,work,foe))))
        work = _lum_update(work,table,live & move_context)
        if table.weather is not None:
            weather,counter = climate.set_move(work.weather,work.weather_counter,
                table.weather.move_weather[move],eligible=acts)
            work = work._replace(weather=weather,weather_counter=counter)
        if table.entry is not None:
            old_layers = work.spikes[rows,foe]
            work = work._replace(spikes=work.spikes.at[rows,foe].set(jnp.where(
                acts & table.entry.spikes_move[move],jnp.minimum(3,old_layers+1),old_layers)))
        moves_cancelled |= acts & ((foe_hp == 0) | (hp == 0))
        if table.phazing is not None:
            work,changed,hazard_ko=dragging.drag(work,foe,move,phazing_draws[rows,side],
                table.phazing,table.entry,eligible=acts & hit & (hp>0))
            displaced=displaced.at[rows,foe].set(displaced[rows,foe] | changed)
            moves_cancelled |= hazard_ko
            work=_lum_update(work,table,changed)
        if table.baton is not None:
            bench_alive = jnp.any(
                (work.hp[rows,side] > 0) & (jnp.arange(6)[None,:] != work.active[rows,side,None]),axis=1)
            passed = acts & table.baton.move[move] & (hp > 0) & bench_alive
            work = work._replace(baton_pass=work.baton_pass.at[rows,side].set(
                work.baton_pass[rows,side] | passed))
            # Only a first ordinary action can leave an opposing action queued.
            # A resumed Baton is already the final original action.
            save_foe = passed & (order == 0) & ~state.pending_residual & ~switches[rows,foe]
            queued_actor = jnp.where(save_foe,foe,queued_actor)
            paused |= passed
    # Gen 3 action-phase faints request replacement BEFORE the residual action.
    # A KO cancels queued moves; direct Baton retains at most one selected foe
    # move. Both pause residuals, including hazard-KO recovery chains.
    # Post-residual replacement requests stay separate.
    turn_context = live & (~replacement | state.pending_residual)
    parties_alive = jnp.all(jnp.any(work.hp>0,axis=2),axis=1)
    active_alive = jnp.all(work.hp[rows[:,None],sides,work.active]>0,axis=1)
    defer_residual = turn_context & parties_alive & (~active_alive | paused)
    residual_ready = turn_context & parties_alive & active_alive & ~paused
    # Snapshot before Wish(order7), weather(order8), and order10 handlers.
    end_speed = _speeds(work,table)
    if table.sustain is not None:
        ordinary = residual_ready & jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        for side in range(2):
            wish = recovery.wish_residual(_active(work.hp,work,side),_active(work.max_hp,work,side),
                work.wish_counter[:,side],eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,wish.hp),
                wish_counter=work.wish_counter.at[:,side].set(wish.counter))
    if table.weather is not None:
        ordinary = residual_ready & jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
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
    # Selected order10/priority0 handlers snapshot SPEED before suborder.
    # Equal-speed Speed Boost(3), Leftovers(4), seed(5), status(6) use independent ties.
    # Ability boosts do not re-sort the already queued residual handlers.
    event_sides = jnp.array([0,1,0,1,0,1,0,1,0,1])
    event_suborder = jnp.array([3,3,4,4,5,5,6,6,14,14])
    tie = jnp.concatenate((draws[:,1,0,None] ^ sides,
        draws[:,0,9,None] ^ sides,draws[:,0,10,None] ^ sides,draws[:,1,9,None] ^ sides,
        draws[:,1,11,None] ^ sides),axis=1)
    event_keys = -end_speed[:,event_sides]*32 + event_suborder[None,:]*2 + tie
    event_order = jnp.argsort(event_keys,axis=1)
    for order in range(10):
        event = event_order[:,order]
        side = event % 2
        ability_event = event < 2
        heal_event = (event >= 2) & (event < 4)
        seed_event = (event >= 4) & (event < 6)
        status_event = (event >= 6) & (event < 8)
        battle_continues = jnp.all(jnp.any(work.hp > 0, axis=2), axis=1)
        eligible = residual_ready & battle_continues
        hp = _active(work.hp,work,side)
        if table.entry is not None:
            boosts = _active(work.boosts,work,side)
            speed_boost = (eligible & ability_event & (hp > 0)
                & (work.active_turns[rows,side] > 0)
                & table.entry.speed_boost[_active(work.ability,work,side)])
            boosts = boosts.at[:,4].set(jnp.where(speed_boost,jnp.minimum(6,boosts[:,4]+1),boosts[:,4]))
            work = work._replace(boosts=_set_active(work.boosts,work,side,boosts))
        healed = hp
        if table.items is not None:
            healed = held.leftovers_heal(table.items,_active(work.item,work,side),hp,
                _active(work.max_hp,work,side),eligible=eligible & heal_event)
        residual = m.major_status_residual(hp,
            _active(work.max_hp, work, side), _active(work.status, work, side),
            _active(work.status_counter, work, side), active=eligible & status_event)
        work = work._replace(hp=_set_active(work.hp, work, side,jnp.where(heal_event,healed,residual.hp)),
            status_counter=_set_active(work.status_counter, work, side, residual.counter))
        if table.sustain is not None:
            foe = 1-side
            drained = recovery.leech(_active(work.hp,work,side),_active(work.max_hp,work,side),
                _active(work.volatile,work,side),_active(work.hp,work,foe),_active(work.max_hp,work,foe),
                eligible=eligible & seed_event)
            work = work._replace(hp=_set_active(work.hp,work,side,drained.hp))
            work = work._replace(hp=_set_active(work.hp,work,foe,drained.recipient_hp))
        if table.encore is not None:
            work=repeating.residual(work,side,eligible=eligible & (event>=8))
    work=linking.prune(work)
    work=work._replace(**{name:jnp.where(work.hp>0,getattr(work,name),0)
        for name in ("last_move","encore_move","encore_counter")})
    alive = jnp.any(work.hp > 0, axis=2)
    done = ~jnp.all(alive, axis=1)
    winner = jnp.where(alive[:, 0], 0, jnp.where(alive[:, 1], 1, 2))
    work = work._replace(done=done, winner=jnp.where(done, winner, -1),
        turn=state.turn + (live & ~replacement).astype(jnp.int32),
        pending_residual=defer_residual,
        queued_actor=jnp.where(defer_residual,queued_actor,-1),
        queued_actions=jnp.where((defer_residual & (queued_actor >= 0))[:,None],
            jnp.where(resuming[:,None],state.queued_actions,actions),0),
        queued_stall=jnp.where((defer_residual & (queued_actor >= 0))[:,None],saved_stall,0),
        residual_draws=jnp.where(defer_residual[:,None,None],draws,jnp.zeros_like(draws)),
        phazing_draws=jnp.where(defer_residual[:,None],phazing_draws,0),
        secondary_draws=jnp.where(defer_residual[:,None],secondary_draws,0),
        sleep_talk_draws=jnp.where(defer_residual[:,None],sleep_talk_draws,0),
        replacement_count=jnp.where(defer_residual,
            jnp.where(resuming,state.replacement_count+1,0),0),
        forced_switch=((jnp.take_along_axis(work.hp, work.active[..., None], axis=2)[..., 0] <= 0)
            | work.baton_pass) & alive,
        baton_pass=work.baton_pass & ~done[:,None])
    ordinary = turn_context & ~defer_residual
    work = work._replace(volatile=jnp.where(ordinary[:, None, None],
        work.volatile & jnp.uint32(~FLINCH & 0xffffffff), work.volatile))
    work = jax.tree.map(lambda old, new: jnp.where(live.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), state, work)
    return StepResult(work, invalid, placeholders)
