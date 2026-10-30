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
from . import action_history as history
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
from . import solar_beam as charging
from . import focus_punch as focusing
from .effects import MoveEffects, from_metadata as effects_from_metadata, stall_success, cure_team, weather_healing
from .state import FLINCH, FOCUS_PUNCH, FOCUS_LOST, BattleState, bench_indices, empty_state, legal_actions, usable_moves


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
    focus_punch: focusing.FocusTable | None = None
    solar_beam: charging.SolarTable | None = None


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
    if any(x is not True and (type(x) is not int or not 0 <= x <= 100) for x in accuracy):
        raise ValueError('ACCURACY_DOMAIN: integer accuracy0..100 or always-hit True required')
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
        repeating.from_metadata(metadata),focusing.from_metadata(metadata),charging.from_metadata(metadata))


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
    state = state._replace(**values, base_ability=values["ability"], hp=hp, max_hp=hp, done=jnp.zeros_like(state.done),
        entry_pending=jnp.ones_like(state.entry_pending))
    return initialize(state,table) if table is not None else state


def _active(value, state, side):
    rows = jnp.arange(state.hp.shape[0])
    return value[rows, side, state.active[rows, side]]


def _set_active(value, state, side, update):
    rows = jnp.arange(state.hp.shape[0])
    return value.at[rows, side, state.active[rows, side]].set(update)


def _lum_update(state,table,eligible,_status_trace=None):
    if table.items is None:
        if _status_trace is not None:return state,_status_trace
        return state
    for side in range(2):
        result = held.consume_lum(table.items,_active(state.item,state,side),
            _active(state.status,state,side),_active(state.status_counter,state,side),
            _active(state.confusion_counter,state,side),_active(state.hp,state,side),eligible=eligible)
        if _status_trace is not None:
            from . import engine_item_callbacks as item_callbacks
            old_item=_active(state.item,state,side)
            _status_trace=item_callbacks.lum_consumed(_status_trace,side,state.active[:,side],
                old_item,_active(state.status,state,side),result.status,
                (_active(state.confusion_counter,state,side)>0) & (result.confusion==0),
                eligible=result.item!=old_item)
        for name,value in zip(("item","status","status_counter","confusion_counter"),result):
            state = state._replace(**{name:_set_active(getattr(state,name),state,side,value)})
        state=state._replace(sleep_skipped=_set_active(state.sleep_skipped,state,side,
            jnp.where(result.status==m.SLP,_active(state.sleep_skipped,state,side),0)),
            rest_origin=_set_active(state.rest_origin,state,side,
                _active(state.rest_origin,state,side) & (result.status==m.SLP)))
    if _status_trace is not None:return state,_status_trace
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


def initialize(state,table,*,eligible=True,tie_coin=None,_weather_trace=None):
    """Resolve initial entries before building the first policy observation.

    No turn, residual, hazard or ordinary-step RNG is consumed. A dedicated
    fold-in stream resolves initial speed ties, so explicit initialization and
    step's compatibility initialization produce identical ordinary trajectories.
    """
    if table.entry is None:
        if _weather_trace is not None:return state,_weather_trace
        return state
    rows = jnp.arange(state.hp.shape[0])
    live = ~state.done & jnp.asarray(eligible)
    speed = _speeds(state,table)
    coin = (jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,314159),(),0,2))(state.keys)
        if tie_coin is None else tie_coin)
    first = jnp.where(speed[:,0] != speed[:,1],speed[:,1] > speed[:,0],coin == 1).astype(jnp.int32)
    pending_battle = live & jnp.any(state.entry_pending,axis=1)
    for order in range(2):
        side = jnp.where(order == 0,first,1-first)
        pending = live & state.entry_pending[rows,side]
        if _weather_trace is not None:
            state,_,_weather_trace = arrival.enter(state,side,table.entry,eligible=pending,
                hazards=False,startup=state.turn==0,_weather_trace=_weather_trace)
        else:
            state,_ = arrival.enter(state,side,table.entry,eligible=pending,hazards=False,startup=state.turn==0)
        state = state._replace(entry_pending=state.entry_pending.at[rows,side].set(
            state.entry_pending[rows,side] & ~pending))
    if _weather_trace is not None:return _lum_update(state,table,pending_battle),_weather_trace
    return _lum_update(state,table,pending_battle)


class StepResult(NamedTuple):
    state: BattleState
    invalid_actions: jax.Array      # [B]; invalid live requests freeze that battle
    placeholder_moves: jax.Array   # [B]; every executed move is still a placeholder


class WeatherProfileTracedStep(NamedTuple):
    """Private owning-profile path only; legacy StepResult/TracedStep unchanged."""
    step: StepResult
    callbacks: object
    weather_profile: object


class ChanceInputs(NamedTuple):
    """Finite semantic inputs, not a claim of independent PRNG-key outcomes.

    ``draws`` bounds by column are 2,100,100,100,16,16,100,4,8,2,2,2.
    Auxiliary bounds are phazing/Static 60, secondary 100, Sleep Talk 60 and ties 2.
    Phazing and contact cannot occur on the same dispatched Gen3 move.
    Saved ordinary-turn draws override these during suspended replacements.
    Reused columns remain shared; enumeration must not break that correlation.
    """
    draws: jax.Array               # [B,2,12]
    phazing: jax.Array             # [B,2]
    secondary: jax.Array           # [B,2]
    sleep_talk: jax.Array          # [B,2]
    entry_ties: jax.Array          # [B,2]
    initial_tie: jax.Array         # [B]


def sample_chance_inputs(state: BattleState) -> ChanceInputs:
    """The original streams, unchanged; the sampled API owns randomness here."""
    keys = jax.vmap(lambda key:jax.random.split(key,2))(state.keys)[:,1]
    draw = lambda salt,limit:jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,salt),(2,),0,limit))(keys)
    return ChanceInputs(
        jax.vmap(lambda key:jax.random.randint(key,(2,12),0,
            jnp.array([2,100,100,100,16,16,100,4,8,2,2,2],jnp.int32)))(keys),
        draw(271828,60),draw(1618033,100),draw(1414213,60),
        jax.vmap(lambda key,count:jax.random.randint(jax.random.fold_in(
            key,jnp.uint32(982451653)+count.astype(jnp.uint32)),(2,),0,2))(
                state.keys,state.replacement_count),
        jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,314159),(),0,2))(state.keys))


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
    return step_with_chance(state,actions,table,sample_chance_inputs(state))


def step_with_chance(state: BattleState, actions: jax.Array, table: MoveTable,
                     chance_inputs: ChanceInputs, *, _public_trace=None, _lifecycle_trace=False,
                     _weather_hp_context=None, _leftovers_context=None, _status_residual_context=None) -> StepResult:
    """Same transition body with supplied finite chance inputs for the oracle.

    Callers must supply in-range integers under ``ChanceInputs``' finite law.
    This seam does not itself enumerate or certify probabilities. PRNG keys
    advance identically to the sampled API but do not choose semantic outcomes.
    """
    if _lifecycle_trace and _public_trace is None:
        raise ValueError('lifecycle mode requires its actual registered callback tape')
    if _status_residual_context is not None:
        from . import engine_lifecycle_callbacks as lifecycle_callbacks
        from . import engine_status_residual_callbacks as status_residual_callbacks
        _status_registered=type(_status_residual_context) is status_residual_callbacks.RegisteredStatusResidualContext
        if (not _lifecycle_trace or type(_status_residual_context) not in
                (lifecycle_callbacks.StatusResidualContext,status_residual_callbacks.RegisteredStatusResidualContext)
                or type(_status_residual_context.data) is not status_residual_callbacks.Data):
            raise ValueError('typed owning lifecycle status residual context required')
        if _status_registered:
            from . import engine_weather_speed_profile as weather_speed_profile
            if (type(_status_residual_context.weather_profile_data) is not weather_speed_profile.ProfileData
                    or type(_weather_hp_context) is not tuple or len(_weather_hp_context)!=2
                    or type(_weather_hp_context[0]) is not weather_speed_profile.BirthRecord
                    or _weather_hp_context[1] is not _status_residual_context.weather_profile_data):
                raise ValueError('registered status residual requires actual birth and same original profile')
            if _status_residual_context.hook_context is not None:
                raise ValueError('kernel-event observer does not accept original hook/history context')
            # This private registration observes the ACTUAL kernel write.
            # Original Showdown hook differences belong to physics fidelity,
            # never to an invented source event or a caller proof boolean.
    if _leftovers_context is not None:
        from . import engine_lifecycle_callbacks as lifecycle_callbacks
        from . import engine_leftovers_callbacks as leftovers_callbacks
        _leftovers_registered=type(_leftovers_context) is leftovers_callbacks.RegisteredLeftoversContext
        if (not _lifecycle_trace or type(_leftovers_context) not in
                (lifecycle_callbacks.LeftoversContext,leftovers_callbacks.RegisteredLeftoversContext)
                or type(_leftovers_context.data) is not leftovers_callbacks.Data):
            raise ValueError('typed owning lifecycle Leftovers context required')
        if _leftovers_registered:
            if _leftovers_context.no_output_tables is not None:
                from . import engine_leftovers_no_output as _leftovers_no_output
                if (type(_leftovers_context.no_output_tables) is not _leftovers_no_output.Tables
                        or _leftovers_context.no_output_tables.leftovers is not _leftovers_context.data
                        or _leftovers_context.no_output_tables.profile is not _leftovers_context.weather_profile_data
                        or getattr(_public_trace,'execution',None) is None):
                    raise ValueError('original no-output metadata and actual owning accounting required')
            from . import engine_weather_speed_profile as weather_speed_profile
            if (type(_leftovers_context.weather_profile_data) is not weather_speed_profile.ProfileData
                    or type(_weather_hp_context) is not tuple or len(_weather_hp_context)!=2
                    or type(_weather_hp_context[0]) is not weather_speed_profile.BirthRecord
                    or _weather_hp_context[1] is not _leftovers_context.weather_profile_data):
                raise ValueError('registered Leftovers requires actual owning birth and same original profile')
    if _lifecycle_trace:
        from . import engine_execution_completeness as execution_accounting
        from .state import LEECH_SEED as _execution_leech_seed
        from . import engine_status_callbacks as status_callbacks
        from . import engine_weather_effect_callbacks as weather_effect_callbacks
        from . import engine_weather_order_callbacks as weather_order_callbacks
        if _weather_hp_context is not None:
            from . import engine_weather_hp_segment as weather_hp_segment
            from . import engine_weather_speed_profile as weather_speed_profile
            if (type(_weather_hp_context) is not tuple or len(_weather_hp_context)!=2
                    or type(_weather_hp_context[0]) is not weather_speed_profile.BirthRecord
                    or type(_weather_hp_context[1]) is not weather_speed_profile.ProfileData):
                raise ValueError('owning birth record and original static weather profile required')
            _weather_hp_record=_weather_hp_context[0]
    batch = state.hp.shape[0]
    shapes = ((batch,2,12),(batch,2),(batch,2),(batch,2),(batch,2),(batch,))
    for value,shape in zip(chance_inputs,shapes,strict=True):
        if value.shape != shape or not jnp.issubdtype(value.dtype,jnp.integer):
            raise ValueError('chance inputs must have declared integer batch shapes')
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
    if _public_trace is not None:
        from . import engine_public_events as public_callbacks
        # Diagnostic receipts only. Uninstrumented consumers keep every trace
        # partial; this must never authorize a full observation or B4 gate.
        if _lifecycle_trace:
            from . import engine_lifecycle_callbacks as lifecycle_callbacks
            _public_trace=lifecycle_callbacks.begin(_public_trace,state,live,invalid)
        _public_trace = _public_trace._replace(
            turn=_public_trace.turn if _lifecycle_trace else state.turn+(live & ~replacement).astype(jnp.int32),
            refusal=_public_trace.refusal | jnp.where(live,public_callbacks.UNINSTRUMENTED,0)
                | jnp.where(invalid,public_callbacks.INVALID_ACTION,0))
    resuming = live & state.pending_residual
    splits = jax.vmap(lambda key: jax.random.split(key, 2))(state.keys)
    draws = chance_inputs.draws
    draws = jnp.where(resuming[:,None,None],state.residual_draws,draws)
    # Extend the ordinary chance block without changing its existing twelve
    # columns. One fixed roll per seat gives unbiased 1..5-bench selection.
    phazing_draws=chance_inputs.phazing
    phazing_draws=jnp.where(resuming[:,None],state.phazing_draws,phazing_draws)
    secondary_draws=chance_inputs.secondary
    secondary_draws=jnp.where(resuming[:,None],state.secondary_draws,secondary_draws)
    sleep_talk_draws=chance_inputs.sleep_talk
    sleep_talk_draws=jnp.where(resuming[:,None],state.sleep_talk_draws,sleep_talk_draws)
    # Replacement requests do not consume another ordinary-turn chance block.
    # A separate counted fold-in distinguishes repeated hazard-KO entry ties.
    entry_ties = chance_inputs.entry_ties
    # nextTurn increments activeTurns before an ordinary request. switchIn
    # later resets an entrant to zero, excluding its entry turn's Speed Boost.
    age = state.active_turns + ((live & ~replacement)[:,None] & (active_hp > 0)).astype(jnp.int32)
    if _lifecycle_trace:
        _public_trace=status_callbacks.initial_context(_public_trace,state,table,live)
    if _lifecycle_trace:
        work,_public_trace=initialize(state,table,eligible=live,tie_coin=chance_inputs.initial_tie,
            _weather_trace=_public_trace)
        work=work._replace(keys=jnp.where(resuming[:,None],state.keys,splits[:,0]),active_turns=age)
    else:
        work = initialize(state,table,eligible=live,tie_coin=chance_inputs.initial_tie)._replace(
            keys=jnp.where(resuming[:,None],state.keys,splits[:,0]),active_turns=age)
    if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live,_public_trace)
    else:work = _lum_update(work,table,live)
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
    charge_moves=state.charge_move[rows[:,None],sides,state.active]
    moves=jnp.where(charge_moves>0,charge_moves,moves)
    moves = jnp.where(struggle, table.struggle, moves)
    if table.focus_punch is not None:
        work=focusing.prepare(work,moves,table.focus_punch,ordinary=live & ~replacement,
            move_requests=(actions<4) & ~struggle)
    pursuit_used = jnp.zeros_like(switches)
    pursuit_faint = jnp.zeros_like(state.done)
    if table.pursuit is not None:
        for side in range(2):
            initial_status = _active(state.status,state,side)
            initial_loaf = jnp.zeros_like(live)
            if table.entry is not None and table.entry.truant is not None:
                initial_loaf = (table.entry.truant[_active(state.ability,state,side)]
                    & _active(state.truant_loaf,state,side))
            eligible = (live & ~replacement & (actions[:,side]<4) & switches[:,1-side]
                & ~struggle[:,side] & (initial_status!=m.SLP) & (initial_status!=m.FRZ)
                & ~initial_loaf)
            work,used,fainted = intercepts.intercept(work,side,slot[:,side],moves[:,side],
                draws[:,side],table,eligible=eligible,contact_roll=phazing_draws[:,side])
            if _lifecycle_trace:
                _public_trace=status_callbacks.unknown(_public_trace,used)
            if _lifecycle_trace:work,_public_trace=_lum_update(work,table,used,_public_trace)
            else:work = _lum_update(work,table,used)
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
        if _lifecycle_trace:
            switch_baton=work.baton_pass[rows,side]
            # Refuse live outgoing Natural Cure and a possible incoming cure
            # before departure/Ability Start can erase their source status.
            departure=jnp.zeros_like(do_switch)
            if table.entry is not None:
                departure=(_active(work.status,work,side)!=m.NONE) & (_active(work.hp,work,side)>0) & table.entry.natural_cure[_active(work.ability,work,side)]
            incoming=status_callbacks.entry_cure_possible(table,work.status[rows,side,target[rows,side]])
            _public_trace=status_callbacks.unknown(_public_trace,do_switch & (departure | incoming))
        installed = installed.at[rows,side].set(do_switch)
        work = passing.copy_to_recipient(work,side,target[rows,side],
            eligible=do_switch & work.baton_pass[rows,side])
        work=arrival.depart(work,side,table.entry,eligible=do_switch)
        work = work._replace(active=work.active.at[rows,side].set(jnp.where(do_switch,target[rows,side],old_active)),
            baton_pass=work.baton_pass.at[rows,side].set(work.baton_pass[rows,side] & ~do_switch),
            forced_switch=work.forced_switch.at[rows,side].set(jnp.where(do_switch,False,work.forced_switch[rows,side])),
            active_turns=work.active_turns.at[rows,side].set(jnp.where(do_switch,0,work.active_turns[rows,side])))
        if _lifecycle_trace:
            _public_trace=lifecycle_callbacks.installed(_public_trace,work,side,old_active,target[rows,side],
                eligible=do_switch,replacement=replacement,baton=switch_baton)
        work=history.record_switch(work,side,target[rows,side],eligible=do_switch)
        # insertChoice snapshots each incoming callback's Speed now; a later
        # Air Lock/Cloud Nine entrant must not re-sort an already queued event.
        callback_speed = callback_speed.at[rows,side].set(_speeds(work,table)[rows,side])
        if table.entry is not None:
            if _lifecycle_trace:
                work,knocked_out,_public_trace = arrival.enter(work,side,table.entry,
                    eligible=do_switch & ~replacement,_weather_trace=_public_trace)
            else:
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
            if _lifecycle_trace:
                work,knocked_out,_public_trace = arrival.enter(work,side,table.entry,
                    eligible=eligible,_weather_trace=_public_trace)
            else:
                work,knocked_out = arrival.enter(work,side,table.entry,eligible=eligible)
            # A hazard KO cancels subsequent queued runSwitch callbacks too.
            switch_faint |= knocked_out
    if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live,_public_trace)
    else:work = _lum_update(work,table,live)
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
        charge_locked=_active(work.charge_move,work,side)>0
        chance = draws[rows, side]
        move_context = ~replacement | (state.pending_residual & (saved_actor == side))
        eligible = (live & move_context & ~paused & ~moves_cancelled & ~switches[rows, side]
            & ~pursuit_used[rows,side] & ~displaced[rows,side] & (_active(work.hp, work, side) > 0))
        sleep_usable=jnp.zeros_like(live);early_bird=jnp.zeros_like(live)
        if table.sleep_talk is not None:
            sleep_usable=table.sleep_talk.move[move]
            early_bird=table.sleep_talk.early_bird[_active(work.ability,work,side)]
        loafing=jnp.zeros_like(live)
        if table.entry is not None and table.entry.truant is not None:
            loafing=table.entry.truant[_active(work.ability,work,side)] & _active(work.truant_loaf,work,side)
        before = m.before_move(_active(work.status, work, side), _active(work.status_counter, work, side),
            _active(work.confusion_counter, work, side), thaw_roll=chance[:, 1],
            paralysis_roll=chance[:, 2], confusion_roll=chance[:, 3],
            flinched=(_active(work.volatile, work, side) & FLINCH) != 0,
            sleep_usable=sleep_usable,early_bird=early_bird,loafing=loafing)
        if _lifecycle_trace:
            old_status=_active(work.status,work,side)
            _public_trace=status_callbacks.changed(_public_trace,side,work.active[rows,side],old_status,
                before.status,jnp.where(old_status==m.SLP,status_callbacks.WAKE,status_callbacks.THAW),
                eligible=eligible & (old_status!=m.NONE) & (before.status==m.NONE))
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
        hustle = jnp.zeros_like(eligible)
        if table.combat is not None:
            ability = _active(work.ability,work,side)
            guts = table.combat.guts[ability]
            hustle = table.combat.hustle[ability]
            confusion_num,confusion_den = traits.attack_modifier(table.combat,ability,
                _active(work.status,work,side),True,item_num=confusion_num,item_den=confusion_den)
        confusion_damage = m.gen3_damage(level=_active(work.level, work, side), power=40,
            attack=own_stats[:, 1], defense=own_stats[:, 2],
            attack_stage=own_boosts[:, 0], defense_stage=own_boosts[:, 1],
            typeless=True, burned=_active(work.status, work, side) == m.BRN,
            guts=guts,hustle=hustle,
            attack_num=confusion_num,attack_den=confusion_den,
            roll=85 + chance[:, 5])
        hp = _active(work.hp, work, side)
        hp = jnp.where(eligible & before.confusion_self_hit, jnp.maximum(0, hp - confusion_damage), hp)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        # Gen 3 singles cancels all queued actions after any faint, not only
        # the fainted mon's move. A confusion self-KO therefore skips the foe.
        moves_cancelled |= eligible & (hp == 0)
        if table.solar_beam is not None:
            work=charging.clear(work,side,eligible=eligible & ~before.can_act)
        acts = eligible & before.can_act & ~moves_cancelled
        active = work.active[rows, side]
        old_pp = work.pp[rows, side, active, chosen_slot]
        acts &= charge_locked | struggle[rows,side] | (old_pp>0)
        used_pp = 1
        if table.pressure is not None:
            used_pp = pp_cost.cost(table.pressure,move,_active(work.types,work,side),
                _active(work.ability,work,foe),_active(work.hp,work,foe))
        work = work._replace(pp=work.pp.at[rows, side, active, chosen_slot].set(
            jnp.where(acts & ~charge_locked & ~struggle[rows, side], jnp.maximum(0, old_pp - used_pp), old_pp)))
        work=work._replace(last_move=_set_active(work.last_move,work,side,
            jnp.where(acts,caller_move,_active(work.last_move,work,side))))
        work=history.record_move(work,side,chosen_slot,eligible=acts,struggle=struggle[rows,side])
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
        if table.focus_punch is not None:
            acts &= ~focusing.failed(work,side,move,table.focus_punch)
        effective_weather=work.weather
        if table.weather is not None:
            effective_weather=climate.effective(table.weather,work.weather,
                work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
        if table.solar_beam is not None:
            work,starts=charging.try_move(work,side,move,table.solar_beam,effective_weather,eligible=acts)
            acts &= ~starts
        if table.self_destruct is not None:
            blocked=exploding.blocked(table.self_destruct,move,
                work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
            acts &= ~blocked
            explodes=acts & table.self_destruct.move[move]
        if _public_trace is not None:
            # A genuine ordinary declaration is known here, after the actual
            # suppression/preparation gates and before any main-hit callback.
            _public_trace = public_callbacks.move(_public_trace,side,work.active[rows,side],
                work.active[rows,foe],caller_move,eligible=caller_acts,
                unsupported=charge_locked | sleep_usable | ~acts)
        attack_stats = _active(work.stats, work, side)
        defense_stats = _active(work.stats, work, foe)
        attack_boosts = _active(work.boosts, work, side)
        defense_boosts = _active(work.boosts, work, foe)
        physical = table.physical[move]
        power,typ = table.power[move],table.type[move]
        if table.solar_beam is not None:
            power=charging.base_power(table.solar_beam,move,power,effective_weather)
        if table.variable_moves is not None:
            power,typ,physical = variable.resolve(table.variable_moves,move,power,typ,physical,
                _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                _active(work.happiness,work,side),_active(work.hp,work,side),_active(work.max_hp,work,side))
        attack_num,attack_den = (1,1)
        if table.items is not None:
            attack_num,attack_den = held.attack_modifier(table.items,
                _active(work.item,work,side),_active(work.species,work,side),physical)
        defense_num,defense_den = (1,1)
        if table.items is not None:
            defense_num,defense_den = held.defense_modifier(table.items,
                _active(work.item,work,foe),_active(work.species,work,foe),physical)
        if table.combat is not None:
            hustle = table.combat.hustle[_active(work.ability,work,side)]
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
            physical=physical, critical=critical,halve_defense=explodes & physical,hustle=hustle,
            attack_num=attack_num,attack_den=attack_den,
            defense_num=defense_num,defense_den=defense_den,
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
        # Gen3's Hustle accuracy handler checks TYPE, including Status moves.
        # Hidden Power's actor-owned type has already been resolved above.
        hustle_accuracy = (hustle & table.combat.physical_type[typ]
            if table.combat is not None else jnp.zeros_like(acts))
        compound_eyes = (table.combat.compound_eyes[_active(work.ability,work,side)]
            if table.combat is not None else jnp.zeros_like(acts))
        sand_veil = (table.weather.sand_veil[_active(work.ability,work,foe)]
            & (effective_weather == climate.SAND) if table.weather is not None else jnp.zeros_like(acts))
        hit = m.accuracy_hit(accuracy, attack_boosts[:, 5], defense_boosts[:, 6],
            chance[:,6],always_hit=always_hit,hustle=hustle_accuracy,
            compound_eyes=compound_eyes,sand_veil=sand_veil)
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
        if _public_trace is not None:
            _public_trace = public_callbacks.condition(_public_trace,foe,work.active[rows,foe],
                foe_hp,_active(work.max_hp,work,foe),_active(work.status,work,foe),
                eligible=acts & hit & ~hits_sub & (dealt>0))
        if table.focus_punch is not None:
            work=focusing.hit(work,side,move,table.focus_punch,eligible=acts & hit & ~immune & ~hits_sub)
        # Gen 3 floors recoil from actual damage, including capped sub damage.
        recoil=jnp.where(is_struggle & (dealt>0),jnp.maximum(1,dealt//4),0)
        if table.damage_moves is not None:
            recoil=special_hits.recoil_damage(table.damage_moves,move,_active(work.ability,work,side),dealt)
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
            heal_weather=work.weather
            if table.weather is not None:
                heal_weather=climate.effective(table.weather,work.weather,
                    work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
            restored=jnp.minimum(max_hp,hp+weather_healing(max_hp,heal_weather))
            hp=jnp.where(acts & effects.weather_heal[move] & (hp>0),restored,hp)
            rest = (acts & effects.rest[move] & (hp < max_hp)
                    & (_active(work.status,work,side) != m.SLP))
            if _lifecycle_trace:
                _public_trace=status_callbacks.unknown(_public_trace,rest)
            hp = jnp.where(rest,max_hp,hp)
            work = work._replace(hp=_set_active(work.hp,work,side,hp),
                status=_set_active(work.status,work,side,
                    jnp.where(rest,m.SLP,_active(work.status,work,side))),
                status_counter=_set_active(work.status_counter,work,side,
                    jnp.where(rest,3,_active(work.status_counter,work,side))),
                sleep_skipped=_set_active(work.sleep_skipped,work,side,
                    jnp.where(rest,0,_active(work.sleep_skipped,work,side))),
                rest_origin=_set_active(work.rest_origin,work,side,
                    jnp.where(rest,True,_active(work.rest_origin,work,side))))
            # Pain Split assigns HP rather than dealing ordinary damage or
            # healing through callbacks. Read the CURRENT opposing active
            # after switches/earlier moves; floor the shared average once,
            # then clamp independently to each mon's own maximum HP. Ghost
            # typing does not block it, but Protect and target Substitute do.
            target_hp=_active(work.hp,work,foe)
            split=(acts & hit & effects.pain_split[move] & (hp>0) & (target_hp>0)
                & (_active(work.substitute_hp,work,foe)==0))
            average=jnp.maximum(1,(hp+target_hp)//2)
            hp=jnp.where(split,jnp.minimum(max_hp,average),hp)
            target_hp=jnp.where(split,jnp.minimum(_active(work.max_hp,work,foe),average),target_hp)
            work=work._replace(hp=_set_active(work.hp,work,side,hp))
            work=work._replace(hp=_set_active(work.hp,work,foe,target_hp))
            boosts = _active(work.boosts,work,side)
            # Gen3 Belly Drum reads live HP after earlier moves, fails at
            # half HP or +6 Attack, and pays a floored direct HP cost. The
            # +12 boost reaches +6 even from -6; other stages are untouched.
            drum=(acts & effects.belly_drum[move] & (max_hp>1)
                & (hp*2>max_hp) & (boosts[:,0]<6))
            hp=hp-jnp.where(drum,max_hp//2,0)
            boosts=boosts.at[:,0].set(jnp.where(drum,6,boosts[:,0]))
            work=work._replace(hp=_set_active(work.hp,work,side,hp))
            boosts = jnp.clip(boosts + jnp.where(acts[:,None],effects.self_boosts[move],0),-6,6)
            # These damaging moves lower their own SpA on a successful hit,
            # including Substitute/KO, but not miss, Protect, immunity or
            # absorption. Unlike target secondaries this is not Shield Dust
            # or Clear Body gated. White Herb handling remains a separate gap.
            drop = acts & hit & ~immune & effects.hit_spa_drop[move]
            boosts = boosts.at[:,2].set(jnp.where(drop,jnp.maximum(-6,boosts[:,2]-2),boosts[:,2]))
            work = work._replace(boosts=_set_active(work.boosts,work,side,boosts))
            if _lifecycle_trace:
                team_cured=(acts[:,None] & effects.heal_bell[move,None] & (work.hp[rows,side]>0)
                    & (work.status[rows,side]!=m.NONE) & ~effects.soundproof[work.ability[rows,side]])
                _public_trace=status_callbacks.unknown(_public_trace,jnp.any(team_cured,axis=1))
            work=cure_team(work,side,effects,eligible=acts & effects.heal_bell[move])
            for target_side in range(2):
                old = _active(work.boosts,work,target_side)
                work = work._replace(boosts=_set_active(work.boosts,work,target_side,
                    jnp.where((acts & effects.haze[move])[:,None],0,old)))
            status = effects.status[move]
            target_types = _active(work.types,work,foe)
            status_immune = jnp.any(effects.status_immunity[status[:,None],target_types],axis=1)
            if table.secondary is not None:
                status_immune |= ~secondaries.status_allowed(table.secondary,status,target_types,
                    _active(work.ability,work,foe))
            status_immune |= effects.thunderwave[move] & jnp.any(target_types == effects.ground_type,axis=1)
            apply_status = (acts & hit & (status != m.NONE) & ~status_immune
                & (_active(work.hp,work,foe) > 0) & (_active(work.status,work,foe) == m.NONE)
                & ((_active(work.substitute_hp,work,foe) == 0) | effects.bypass_sub[move]))
            if _lifecycle_trace:
                _public_trace=status_callbacks.changed(_public_trace,foe,work.active[rows,foe],
                    _active(work.status,work,foe),status,status_callbacks.PRIMARY,eligible=apply_status)
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(apply_status,status,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(apply_status,jnp.where(status == m.SLP,2 + chance[:,7],0),
                              _active(work.status_counter,work,foe))),
                sleep_skipped=_set_active(work.sleep_skipped,work,foe,
                    jnp.where(apply_status,0,_active(work.sleep_skipped,work,foe))),
                rest_origin=_set_active(work.rest_origin,work,foe,
                    jnp.where(apply_status,False,_active(work.rest_origin,work,foe))))
            if table.secondary is not None:
                if _lifecycle_trace:
                    work,_public_trace=secondaries.after_status_event(work,side,foe,status,table.secondary,
                        applied=apply_status,item_table=table.items,_status_trace=_public_trace)
                else:
                    work=secondaries.after_status_event(work,side,foe,status,table.secondary,
                        applied=apply_status,item_table=table.items)
        if table.sustain is not None:
            seeded = recovery.seed_target(table.sustain,move,_active(work.types,work,foe),
                _active(work.hp,work,foe),_active(work.substitute_hp,work,foe),
                _active(work.volatile,work,foe),eligible=acts & hit)
            pending = work.wish_counter[rows,side]
            starts_wish = acts & table.sustain.wish[move] & (pending == 0)
            work = work._replace(volatile=_set_active(work.volatile,work,foe,seeded),
                wish_counter=work.wish_counter.at[rows,side].set(jnp.where(
                    starts_wish,2,pending)),
                wish_amount=work.wish_amount.at[rows,side].set(jnp.where(
                    starts_wish,_active(work.max_hp,work,side)//2,work.wish_amount[rows,side])))
            # Gen3 clears even through an unbroken Substitute or zero actual
            # Endure damage, but not on a miss, Protect or type immunity.
            work=recovery.spin_cleanup(work,side,
                eligible=acts & hit & ~immune & table.sustain.spin[move])
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
            if _lifecycle_trace:
                work,_public_trace=secondaries.apply(work,side,move,secondary_draws[rows,side],table.secondary,
                    eligible=acts & hit & ~immune,effective_weather=effective_weather,
                    substitute_hit=hits_sub,item_table=table.items,_status_trace=_public_trace)
            else:
                work=secondaries.apply(work,side,move,secondary_draws[rows,side],table.secondary,
                    eligible=acts & hit & ~immune,effective_weather=effective_weather,
                    substitute_hit=hits_sub,item_table=table.items)
        if table.absorption is not None:
            # Gen 3 onDamagingHit follows secondaries: a frozen target cannot
            # receive this Fire hit's burn, then thaws. Original-catalog Fire
            # only (not Hidden Power Fire); Substitute damage does not thaw.
            thawed = ((dealt > 0) & ~hits_sub & table.absorption.fire_thaws[move]
                & (_active(work.status,work,foe) == m.FRZ))
            if _lifecycle_trace:
                _public_trace=status_callbacks.unknown(_public_trace,thawed)
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(thawed,m.NONE,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(thawed,0,_active(work.status_counter,work,foe))))
        if table.secondary is not None:
            if _lifecycle_trace:
                work,_public_trace = secondaries.apply_contact(work,side,move,phazing_draws[rows,side],table.secondary,
                    eligible=acts & hit & (dealt>0) & ~hits_sub,item_table=table.items,_status_trace=_public_trace)
            else:
                work = secondaries.apply_contact(work,side,move,phazing_draws[rows,side],table.secondary,
                    eligible=acts & hit & (dealt>0) & ~hits_sub,item_table=table.items)
        if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live & move_context,_public_trace)
        else:work = _lum_update(work,table,live & move_context)
        # Gen3 applies contact DamagingHit (and immediate Lum cure) before
        # recoil. A source about to faint can still consume its berry first.
        hp = jnp.maximum(0,_active(work.hp,work,side)-recoil)
        work = work._replace(hp=_set_active(work.hp,work,side,hp))
        if table.weather is not None:
            if _lifecycle_trace:
                weather,counter,_public_trace = climate.set_move(work.weather,work.weather_counter,
                    table.weather.move_weather[move],eligible=acts,_weather_trace=_public_trace,
                    _actor_side=side,_actor_physical=work.active[rows,side],_move_id=move)
            else:
                weather,counter = climate.set_move(work.weather,work.weather_counter,
                    table.weather.move_weather[move],eligible=acts)
            work = work._replace(weather=weather,weather_counter=counter)
        if table.entry is not None:
            old_layers = work.spikes[rows,foe]
            work = work._replace(spikes=work.spikes.at[rows,foe].set(jnp.where(
                acts & table.entry.spikes_move[move],jnp.minimum(3,old_layers+1),old_layers)))
        moves_cancelled |= acts & ((foe_hp == 0) | (hp == 0))
        if _public_trace is not None:
            _public_trace = public_callbacks.faint(_public_trace,foe,work.active[rows,foe],
                eligible=acts & hit & ~hits_sub & (dealt>0) & (foe_hp==0))
        if table.phazing is not None:
            work,changed,hazard_ko=dragging.drag(work,foe,move,phazing_draws[rows,side],
                table.phazing,table.entry,eligible=acts & hit & (hp>0))
            displaced=displaced.at[rows,foe].set(displaced[rows,foe] | changed)
            moves_cancelled |= hazard_ko
            if _lifecycle_trace:work,_public_trace=_lum_update(work,table,changed,_public_trace)
            else:work=_lum_update(work,table,changed)
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
    if _lifecycle_trace:
        # Source runAction('residual') adds its blank separator before handlers.
        _public_trace=lifecycle_callbacks.action_end(_public_trace,residual_ready)
    # Snapshot before Wish(order7), weather(order8), and order10 handlers.
    end_speed = _speeds(work,table)
    if _lifecycle_trace:_weather_pre_wish=work
    if _status_residual_context is not None:
        if _status_registered:
            _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                work,_public_trace.stream_id,_status_residual_context.weather_profile_data,eligible=residual_ready)
            _status_cached_speed_ok=(jnp.all((end_speed>0)&(end_speed<=10000),axis=1)
                & (end_speed[:,0]!=end_speed[:,1])
                & jnp.all(_status_residual_context.data.engine_binding==_status_residual_context.weather_profile_data.engine_binding)
                & jnp.all(_status_residual_context.data.source_binding==_status_residual_context.weather_profile_data.source_binding))
    if _leftovers_context is not None:
        if _leftovers_registered:
            _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                work,_public_trace.stream_id,_leftovers_context.weather_profile_data,eligible=residual_ready)
            # Actual source cache, not a resample after Speed Boost/pinch.
            # The cap guard keeps original getStat's 10000 clamp inactive even
            # when an installed weather ability doubles a bounded live Speed.
            _leftovers_cached_speed_ok=(jnp.all((end_speed>0)&(end_speed<=10000),axis=1)
                & (end_speed[:,0]!=end_speed[:,1])
                & jnp.all(_leftovers_context.data.engine_binding==_leftovers_context.weather_profile_data.engine_binding)
                & jnp.all(_leftovers_context.data.source_binding==_leftovers_context.weather_profile_data.source_binding))
    if table.sustain is not None:
        ordinary = residual_ready & jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        for side in range(2):
            if _lifecycle_trace:
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.WISH,side,
                    active=ordinary & (work.wish_counter[:,side]>0))
            wish = recovery.wish_residual(_active(work.hp,work,side),_active(work.max_hp,work,side),
                work.wish_counter[:,side],eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,wish.hp),
                wish_counter=work.wish_counter.at[:,side].set(wish.counter))
    if table.weather is not None:
        ordinary = residual_ready & jnp.all(jnp.any(work.hp > 0,axis=2),axis=1)
        if _lifecycle_trace:
            weather,counter,_public_trace = climate.residual_clock(work.weather,work.weather_counter,
                eligible=ordinary,_weather_trace=_public_trace)
        else:
            weather,counter = climate.residual_clock(work.weather,work.weather_counter,eligible=ordinary)
        work = work._replace(weather=weather,weather_counter=counter)
        active_abilities = work.ability[rows[:,None],sides,work.active]
        active_hp = work.hp[rows[:,None],sides,work.active]
        effective_weather = climate.effective(table.weather,weather,active_abilities,active_hp)
        if _lifecycle_trace:
            _weather_segment_before_count=_public_trace.count
            _weather_pre_writes=work  # actual AFTER Wish/clock, BEFORE either HP write
        # The field weather handler runs Weather on BOTH actives before
        # faintMessages/checkWin. A double last-mon weather KO is a Gen 3 tie.
        for side in range(2):
            if _lifecycle_trace:_weather_old_hp=_active(work.hp,work,side)
            hp = climate.chip(table.weather,effective_weather,_active(work.types,work,side),
                _active(work.ability,work,side),_active(work.hp,work,side),
                _active(work.max_hp,work,side),eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,hp))
            if _lifecycle_trace:
                _weather_before_count=_public_trace.count
                _public_trace=weather_effect_callbacks.weather_chip(_public_trace,side,work.active[rows,side],
                    _weather_old_hp,hp,_active(work.max_hp,work,side),_active(work.status,work,side),weather,
                    eligible=ordinary&(hp<_weather_old_hp),_defer_own_unknown=_weather_hp_context is not None)
                _public_trace=weather_order_callbacks.annotate_weather_chip(_public_trace,_weather_before_count,
                    side,work.active[rows,side],end_speed[:,0],end_speed[:,1],
                    eligible=ordinary&(hp<_weather_old_hp),_defer_own_unknown=_weather_hp_context is not None)
        if _lifecycle_trace:
            if _weather_hp_context is not None:
                _weather_segment=weather_hp_segment.order_segment_device(_public_trace,
                    _weather_segment_before_count,_weather_pre_wish,work,end_speed,
                    _weather_hp_context[0],_weather_hp_context[1],eligible=ordinary,
                    pre_writes_state=_weather_pre_writes)
                _public_trace=_weather_segment.tape
                _weather_hp_record=_weather_segment.record
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
        if _status_residual_context is not None:
            if _status_registered:
                _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,eligible=eligible)
        if table.entry is not None:
            boosts = _active(work.boosts,work,side)
            speed_boost = (eligible & ability_event & (hp > 0)
                & (work.active_turns[rows,side] > 0)
                & table.entry.speed_boost[_active(work.ability,work,side)])
            if _lifecycle_trace:
                # The attempt exists at stage+6 too, even if min leaves no write.
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.SPEED_BOOST,side,active=speed_boost)
            boosts = boosts.at[:,4].set(jnp.where(speed_boost,jnp.minimum(6,boosts[:,4]+1),boosts[:,4]))
            work = work._replace(boosts=_set_active(work.boosts,work,side,boosts))
        healed = hp
        if _leftovers_context is not None:
            # Literal BEFORE values precede both the numerical heal and item
            # consumption. No final item/state reconstruction or amount formula.
            _leftovers_physical=work.active[rows,side]
            _leftovers_item=_active(work.item,work,side)
            _leftovers_old_hp=hp
            _leftovers_maximum=_active(work.max_hp,work,side)
            _leftovers_status=_active(work.status,work,side)
            if _leftovers_registered:
                _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_leftovers_context.weather_profile_data,eligible=eligible)
        if table.items is not None:
            if _lifecycle_trace:
                _execution_item=_active(work.item,work,side)
                _execution_leftovers_requested=(eligible & heal_event & (hp>0)
                    & table.items.leftovers[_execution_item])
                _execution_old_heal_hp=hp
                _public_trace,_execution_leftovers_ticket=execution_accounting.open_attempt(
                    _public_trace,execution_accounting.Site.LEFTOVERS,side,
                    active=_execution_leftovers_requested)
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.PINCH_ITEM,side,
                    active=eligible & heal_event & (hp>0)
                    & (hp*4<=_active(work.max_hp,work,side)) & (table.items.pinch_stat[_execution_item]>=0))
                if _leftovers_context is not None:
                    if _leftovers_registered and _leftovers_context.no_output_tables is not None:
                        _execution_full_hp_before=_leftovers_no_output.capture_before_device(
                            _public_trace,_execution_leftovers_ticket,work,event,_weather_hp_record,
                            _leftovers_context.no_output_tables,
                            requested=_execution_leftovers_requested&(hp==_active(work.max_hp,work,side)))
            healed = held.leftovers_heal(table.items,_active(work.item,work,side),hp,
                _active(work.max_hp,work,side),eligible=eligible & heal_event)
            item,boosts=held.consume_pinch(table.items,_active(work.item,work,side),
                _active(work.boosts,work,side),hp,_active(work.max_hp,work,side),
                eligible=eligible & heal_event)
            work=work._replace(item=_set_active(work.item,work,side,item),
                boosts=_set_active(work.boosts,work,side,boosts))
            if _leftovers_context is not None:
                if _leftovers_registered:
                    _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                        work,_public_trace.stream_id,_leftovers_context.weather_profile_data,
                        eligible=eligible & heal_event)
        if _lifecycle_trace:
            _execution_status=_active(work.status,work,side)
            _execution_status_requested=(eligible & status_event & (hp>0)
                & ((_execution_status==m.BRN)|(_execution_status==m.PSN)|(_execution_status==m.TOX)))
            if _status_residual_context is not None and _status_registered:
                _public_trace,_execution_status_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.STATUS_RESIDUAL,side,active=_execution_status_requested)
            else:
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.STATUS_RESIDUAL,side,active=_execution_status_requested)
        if _status_residual_context is not None:
            # Literal BEFORE values at this exact numerical assignment. The
            # kernel-site ticket may capture its exact appended span later;
            # no ticket is consumed or physically qualified by this write.
            _status_physical=work.active[rows,side]
            _status_old_hp=_active(work.hp,work,side)
            _status_maximum=_active(work.max_hp,work,side)
            _status_old_status=_active(work.status,work,side)
            _status_old_counter=_active(work.status_counter,work,side)
            if _status_registered:
                _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,
                    eligible=_execution_status_requested)
        residual = m.major_status_residual(hp,
            _active(work.max_hp, work, side), _active(work.status, work, side),
            _active(work.status_counter, work, side), active=eligible & status_event)
        work = work._replace(hp=_set_active(work.hp, work, side,jnp.where(heal_event,healed,residual.hp)),
            status_counter=_set_active(work.status_counter, work, side, residual.counter))
        if _leftovers_context is not None:
            if _leftovers_registered and _leftovers_context.no_output_tables is not None:
                if table.items is not None:
                    _execution_full_hp_after=_leftovers_no_output.finish_device(_execution_full_hp_before,
                        _public_trace,work,_weather_hp_record,_leftovers_context.no_output_tables)
                    _public_trace=execution_accounting.capture_leftovers_full_hp(_public_trace,
                        _execution_full_hp_after,no_output_tables=_leftovers_context.no_output_tables)
                    # Only this exact entered full-HP branch uses its typed
                    # no-output receipt; failed proofs remain sticky/unhandled.
                    _execution_leftovers_requested &= ~_execution_full_hp_before.requested
        if _status_residual_context is not None:
            # Read AFTER the SAME HP/counter write, before later handlers. No
            # damage/stage formula, final-state reconstruction or faint event.
            _status_new_hp=_active(work.hp,work,side)
            _status_defer=False
            if _status_registered:
                _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,
                    eligible=_execution_status_requested)
                # Selected kernel-event coverage only. Original-hook/history
                # equivalence is deliberately not inferred from speed membership.
                _status_guard_valid=(_weather_hp_record.qualified_birth & _status_cached_speed_ok
                    & ((_public_trace.refusal&~public_callbacks.UNINSTRUMENTED)==0))
                _status_defer=_status_guard_valid
            _public_trace=status_residual_callbacks.record_status_residual_device(_public_trace,
                side,_status_physical,_status_old_hp,_active(work.hp,work,side),_status_maximum,
                _status_old_status,_status_old_counter,_active(work.status_counter,work,side),
                _status_residual_context.data,eligible=eligible & status_event & (_status_old_hp>0)
                & ((_status_old_status==m.BRN)|(_status_old_status==m.PSN)|(_status_old_status==m.TOX)),
                _defer_own_unknown=_status_defer)
            if _status_registered:
                _status_execution=getattr(_public_trace,'execution',None)
                if _status_execution is not None:
                    _status_ledger=execution_accounting.unhandled(_status_execution.ledger,
                        _execution_status_ticket,active=_execution_status_requested & ~_status_guard_valid,
                        tables=_status_execution.tables)
                    _public_trace=execution_accounting._put_context(_public_trace,_status_ledger,_status_execution)
                _public_trace=execution_accounting.capture_status_residual_write(_public_trace,
                    _execution_status_ticket,requested=_execution_status_requested & _status_guard_valid,
                    positive_write=(_status_new_hp<_status_old_hp)&(_status_new_hp>0))
        if _leftovers_context is not None:
            # Record immediately AFTER this exact HP assignment. The immutable
            # Diagnostic mode retains UNKNOWN. The registered actual context
            # defers ONLY this new emitter latch; all prior refusals survive.
            _leftovers_new_hp=_active(work.hp,work,side)
            _leftovers_defer=False
            if _leftovers_registered:
                _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_leftovers_context.weather_profile_data,
                    eligible=eligible & heal_event)
                _leftovers_defer=_weather_hp_record.qualified_birth & _leftovers_cached_speed_ok
            _public_trace=leftovers_callbacks.record_leftovers_heal_device(_public_trace,
                side,_leftovers_physical,_leftovers_item,_leftovers_old_hp,_leftovers_new_hp,
                _leftovers_maximum,_leftovers_status,_leftovers_context.data,
                eligible=eligible & heal_event & (_leftovers_new_hp>_leftovers_old_hp),
                _defer_own_unknown=_leftovers_defer)
        if _lifecycle_trace:
            if table.items is not None:
                _public_trace=execution_accounting.capture_leftovers_write(_public_trace,
                    _execution_leftovers_ticket,requested=_execution_leftovers_requested,
                    positive_write=_active(work.hp,work,side)>_execution_old_heal_hp)
        if table.sustain is not None:
            foe = 1-side
            if _lifecycle_trace:
                _execution_seed_attempt=(eligible & seed_event & (_active(work.hp,work,side)>0)
                    & ((_active(work.volatile,work,side)&_execution_leech_seed)!=0))
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.LEECH_DRAIN,side,active=_execution_seed_attempt)
                # This is the numerical handler's recipient-live dispatch
                # guard, not a reconstructed drain/healing delta. Source-slot
                # provenance and cancellation hooks remain unproved/UNKNOWN.
                _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                    execution_accounting.Site.LEECH_HEAL,foe,
                    active=_execution_seed_attempt & (_active(work.hp,work,foe)>0))
            drained = recovery.leech(_active(work.hp,work,side),_active(work.max_hp,work,side),
                _active(work.volatile,work,side),_active(work.hp,work,foe),_active(work.max_hp,work,foe),
                eligible=eligible & seed_event)
            work = work._replace(hp=_set_active(work.hp,work,side,drained.hp))
            work = work._replace(hp=_set_active(work.hp,work,foe,drained.recipient_hp))
        if table.encore is not None:
            work=repeating.residual(work,side,eligible=eligible & (event>=8))
    work=linking.prune(work)
    work=work._replace(volatile=jnp.where(work.hp>0,work.volatile,
        work.volatile & jnp.uint32(~(FOCUS_PUNCH|FOCUS_LOST)&0xffffffff)))
    work=work._replace(**{name:jnp.where(work.hp>0,getattr(work,name),0)
        for name in ("last_move","encore_move","encore_counter","charge_move","charge_counter")})
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
    if table.entry is not None and table.entry.truant is not None:
        # Pinned Gen3 ability residual order27. It flips even when no move
        # executed (sleep/paralysis/etc), and never on an unfinished residual.
        # An end-of-turn faint replacement enters later and misses this flip.
        for side in range(2):
            active=work.active[:,side]
            old=work.truant_loaf[rows,side,active]
            holder=table.entry.truant[_active(work.ability,work,side)]
            tick=ordinary & ~work.done & (_active(work.hp,work,side)>0) & holder
            work=work._replace(truant_loaf=work.truant_loaf.at[rows,side,active].set(jnp.where(tick,~old,old)))
    if table.solar_beam is not None:
        work=charging.residual(work,eligible=ordinary)
    if _lifecycle_trace:
        _public_trace=lifecycle_callbacks.finish(_public_trace,work,live=live,replacement=replacement,
            pending_before=state.pending_residual,residual_ready=residual_ready,
            defer_residual=defer_residual,turn_context=turn_context)
    work = work._replace(volatile=jnp.where(ordinary[:, None, None],
        work.volatile & jnp.uint32(~(FLINCH|FOCUS_PUNCH|FOCUS_LOST) & 0xffffffff), work.volatile))
    work = jax.tree.map(lambda old, new: jnp.where(live.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), state, work)
    result = StepResult(work, invalid, placeholders)
    if _lifecycle_trace:
        if _weather_hp_context is not None:
            return WeatherProfileTracedStep(result,_public_trace,_weather_hp_record)
    return result if _public_trace is None else public_callbacks.TracedStep(result,_public_trace)
