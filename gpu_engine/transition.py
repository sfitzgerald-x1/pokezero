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
from . import engine_device_public_hooks as device_public
from . import engine_device_public_port as public_port

import jax
import jax.numpy as jnp
from . import engine_pinch_item_callbacks as pinch_callbacks
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
from .state import DESTINY_BOND


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
    public_port: object = None


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
        repeating.from_metadata(metadata),focusing.from_metadata(metadata),charging.from_metadata(metadata),public_port.compile_tables(metadata))


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
    state = state._replace(**values, base_types=values["types"], base_ability=values["ability"], hp=hp, max_hp=hp, done=jnp.zeros_like(state.done),
        entry_pending=jnp.ones_like(state.entry_pending))
    return initialize(state,table) if table is not None else state


def _active(value, state, side):
    rows = jnp.arange(state.hp.shape[0])
    return value[rows, side, state.active[rows, side]]


def _set_active(value, state, side, update):
    rows = jnp.arange(state.hp.shape[0])
    return value.at[rows, side, state.active[rows, side]].set(update)


def _lum_update(state,table,eligible,_status_trace=None,*,_device_public=None):
    if _device_public is not None:_device_before=state
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
    if _device_public is not None:
        device_public.item_changes(_device_public,_device_before,state,jnp.broadcast_to(jnp.asarray(eligible),(state.hp.shape[0],)))
    if _status_trace is not None:return state,_status_trace
    return state


def _clear_fainted_boosts(state,before_hp,*,eligible):
    """Gen3 faintMessages -> clearVolatile(false), seven boost stages only.

    A faint queue is resolved at the caller's original action/handler boundary.
    Previously dead slots and ineligible rows are not new faint transactions.
    """
    fainted=jnp.asarray(eligible)[:,None,None] & (before_hp>0) & (state.hp<=0)
    return state._replace(boosts=jnp.where(fainted[...,None],0,state.boosts))


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


def initialize(state,table,*,eligible=True,tie_coin=None,_weather_trace=None,_device_public=None):
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
        if _device_public is not None:_device_entry_before=state
        if _weather_trace is not None:
            state,_,_weather_trace = arrival.enter(state,side,table.entry,eligible=pending,
                hazards=False,startup=state.turn==0,_weather_trace=_weather_trace)
        else:
            state,_ = arrival.enter(state,side,table.entry,eligible=pending,hazards=False,startup=state.turn==0)
        if _device_public is not None:device_public.entry(_device_public,_device_entry_before,state,side,table.entry,pending)
        state = state._replace(entry_pending=state.entry_pending.at[rows,side].set(
            state.entry_pending[rows,side] & ~pending))
    if _weather_trace is not None:return _lum_update(state,table,pending_battle,_device_public=_device_public),_weather_trace
    return _lum_update(state,table,pending_battle,_device_public=_device_public)


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
    ``effect`` (fidelity patch R, design 2.1) is an appended, independent
    stream with bounds 16,16,4,2; None means "the value sample_chance_inputs
    draws", so six-field explicit callers keep their exact trajectories.
    """
    draws: jax.Array               # [B,2,12]
    phazing: jax.Array             # [B,2]
    secondary: jax.Array           # [B,2]
    sleep_talk: jax.Array          # [B,2]
    entry_ties: jax.Array          # [B,2]
    initial_tie: jax.Array         # [B]
    effect: jax.Array | None = None  # [B,2,4]; appended last, so positional six-field constructors still work


# Fidelity patch R (design 2.1): ChanceInputs.effect is fold_in(step key, EFFECT_SALT), drawn from the same
# split key as `draws` but through its own fold_in, so no existing field and no key advance reads it. The salt
# differs from 271828, 1618033, 1414213, 314159 and 982451653+n. Columns: [s][0] a second hit's critical roll,
# [s][1] a second hit's damage roll (85+d) (Bonemerang), [s][2] the residual Yawn sleep duration (2+d), and the
# residual pair ties [0][3] (partial trap) and [1][3] (Perish); M2 adds the last three consumers.
# The hand-written CUDA chance kernel (cuda/v2e_ffi.cu) does not draw this field yet: it is deferred to the
# CUDA option-A work (design 13 item 6), so `v2e_diff chance` and v2e_cuda_check p2 report it as differing.
EFFECT_SALT = 1732050
EFFECT_BOUNDS = (16, 16, 4, 2)


def effect_chance(keys):
    """ChanceInputs.effect from the step's split keys (split(state.keys, 2)[:, 1], as sample_chance_inputs uses)."""
    return jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,EFFECT_SALT),(2,4),0,
        jnp.array(EFFECT_BOUNDS,jnp.int32)))(keys)


def sample_chance_inputs(state: BattleState) -> ChanceInputs:
    """The original streams, unchanged; the sampled API owns randomness here.

    The appended `effect` stream (design 2.1) is computed from the same split
    keys by its own fold_in; the six original expressions are untouched.
    """
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
        jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,314159),(),0,2))(state.keys),
        effect=effect_chance(keys))


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
                     _weather_hp_context=None, _leftovers_context=None, _status_residual_context=None,
                     _direct_hit_context=None, _substitute_context=None, _toxic_context=None, _recoil_context=None, _calm_mind_context=None, _swords_dance_context=None, _spikes_context=None, _thunder_wave_context=None, _sleep_powder_context=None, _hypothetical_profile=None, _device_public=None) -> StepResult:
    """Same transition body with supplied finite chance inputs for the oracle.

    Callers must supply in-range integers under ``ChanceInputs``' finite law.
    This seam does not itself enumerate or certify probabilities. PRNG keys
    advance identically to the sampled API but do not choose semantic outcomes.
    """
    if _device_public is not None:
        if type(_device_public) is not device_public.Context or _public_trace is not None or _lifecycle_trace:
            raise ValueError('device public wrapper is independent of legacy capture')
    _hypothetical = _hypothetical_profile is not None
    _weather_observer_context = _weather_hp_context
    if _hypothetical:
        from . import v5_hypothetical_profile as hypothesis_profile
        if not _lifecycle_trace or _weather_hp_context is not None:
            raise ValueError('hypothetical observer is separate from ordinary birth context')
        hypothesis_profile.check_context(_hypothetical_profile,state.hp.shape[0])
        if (_leftovers_context is not None and type(_leftovers_context) is not hypothesis_profile.LeftoversContext
                or _status_residual_context is not None and type(_status_residual_context) is not hypothesis_profile.StatusContext):
            raise ValueError('hypothetical residual statics require their separate context types')
        _weather_observer_context=(_hypothetical_profile.record,_hypothetical_profile.data)
    _profile_qualified=(lambda record:record.qualified_origin) if _hypothetical else (lambda record:record.qualified_birth)
    if _lifecycle_trace and _public_trace is None:
        raise ValueError('lifecycle mode requires its actual registered callback tape')
    if _calm_mind_context is not None:
        from . import engine_calm_mind_callbacks as calm_mind_callbacks
        if not _lifecycle_trace or type(_calm_mind_context) is not calm_mind_callbacks.Context:
            raise ValueError('typed actual registered Calm Mind context required')
    if _swords_dance_context is not None:
        from . import engine_swords_dance_callbacks as swords_dance_callbacks
        if not _lifecycle_trace or type(_swords_dance_context) is not swords_dance_callbacks.Context:
            raise ValueError('typed actual registered Swords Dance context required')
    if _spikes_context is not None:
        from . import engine_spikes_callbacks as spikes_callbacks
        if not _lifecycle_trace or type(_spikes_context) is not spikes_callbacks.Context:
            raise ValueError('typed original registered Spikes context required')
    if _thunder_wave_context is not None:
        from . import engine_thunder_wave_callbacks as thunder_callbacks
        if not _lifecycle_trace or type(_thunder_wave_context) is not thunder_callbacks.Context:
            raise ValueError('typed actual registered Thunder Wave context required')
    if _sleep_powder_context is not None:
        from . import engine_sleep_powder_callbacks as powder_callbacks
        if not _lifecycle_trace or type(_sleep_powder_context) is not powder_callbacks.Context:
            raise ValueError('typed actual registered Sleep Powder context required')
    if _toxic_context is not None:
        from . import engine_toxic_callbacks as toxic_callbacks
        _toxic_registered=type(_toxic_context) is toxic_callbacks.RegisteredContext
        if (not _lifecycle_trace or type(_toxic_context) not in (toxic_callbacks.Context,toxic_callbacks.RegisteredContext)
                or type(_toxic_context.data) is not toxic_callbacks.Data):
            raise ValueError('typed actual lifecycle Toxic context required')
    if _substitute_context is not None:
        from . import engine_substitute_callbacks as substitute_callbacks
        _substitute_registered=type(_substitute_context) is substitute_callbacks.RegisteredContext
        if (not _lifecycle_trace or type(_substitute_context) not in
                (substitute_callbacks.Context,substitute_callbacks.RegisteredContext)
                or type(_substitute_context.data) is not substitute_callbacks.Data):
            raise ValueError('typed actual lifecycle Substitute context required')
    if _recoil_context is not None:
        from . import engine_recoil_callbacks as recoil_callbacks
        from . import engine_direct_hit_controls as direct_hit_controls
        if (not _lifecycle_trace or type(_recoil_context) is not recoil_callbacks.Context
            or _direct_hit_context is not _recoil_context.direct
            or type(_direct_hit_context) is not direct_hit_controls.RegisteredContext):
            raise ValueError('typed SAME original registered recoil/direct-hit context required')
    if _lifecycle_trace:
        _direct_hit_registered=False
    if _direct_hit_context is not None:
        from . import engine_direct_hit_controls as direct_hit_controls
        _direct_hit_registered=type(_direct_hit_context) is direct_hit_controls.RegisteredContext
        if _direct_hit_registered:from . import engine_direct_ko_callbacks as direct_ko_callbacks
        if (not _lifecycle_trace or type(_direct_hit_context) not in
                (direct_hit_controls.Context,direct_hit_controls.RegisteredContext)
                or type(_direct_hit_context.data) is not direct_hit_controls.Data):
            raise ValueError('typed owning lifecycle direct-hit context required')
    if _status_residual_context is not None:
        from . import engine_lifecycle_callbacks as lifecycle_callbacks
        from . import engine_status_residual_callbacks as status_residual_callbacks
        _status_registered=(type(_status_residual_context) is status_residual_callbacks.RegisteredStatusResidualContext
            or (_hypothetical and type(_status_residual_context) is hypothesis_profile.StatusContext))
        if (not _lifecycle_trace or type(_status_residual_context) not in
                (lifecycle_callbacks.StatusResidualContext,status_residual_callbacks.RegisteredStatusResidualContext)
                    + ((hypothesis_profile.StatusContext,) if _hypothetical else ())
                or type(_status_residual_context.data) is not status_residual_callbacks.Data):
            raise ValueError('typed owning lifecycle status residual context required')
        if _status_registered:
            from . import engine_weather_speed_profile as weather_speed_profile
            if (type(_status_residual_context.weather_profile_data) is not weather_speed_profile.ProfileData
                    or type(_weather_observer_context) is not tuple or len(_weather_observer_context)!=2
                    or (not _hypothetical and type(_weather_observer_context[0]) is not weather_speed_profile.BirthRecord)
                    or _weather_observer_context[1] is not _status_residual_context.weather_profile_data):
                raise ValueError('registered status residual requires actual birth and same original profile')
            if _status_residual_context.hook_context is not None:
                raise ValueError('kernel-event observer does not accept original hook/history context')
            # This private registration observes the ACTUAL kernel write.
            # Original Showdown hook differences belong to physics fidelity,
            # never to an invented source event or a caller proof boolean.
    if _leftovers_context is not None:
        from . import engine_lifecycle_callbacks as lifecycle_callbacks
        from . import engine_leftovers_callbacks as leftovers_callbacks
        _leftovers_registered=(type(_leftovers_context) is leftovers_callbacks.RegisteredLeftoversContext
            or (_hypothetical and type(_leftovers_context) is hypothesis_profile.LeftoversContext))
        if (not _lifecycle_trace or type(_leftovers_context) not in
                (lifecycle_callbacks.LeftoversContext,leftovers_callbacks.RegisteredLeftoversContext)
                    + ((hypothesis_profile.LeftoversContext,) if _hypothetical else ())
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
                    or type(_weather_observer_context) is not tuple or len(_weather_observer_context)!=2
                    or (not _hypothetical and type(_weather_observer_context[0]) is not weather_speed_profile.BirthRecord)
                    or _weather_observer_context[1] is not _leftovers_context.weather_profile_data):
                raise ValueError('registered Leftovers requires actual owning birth and same original profile')
    if _lifecycle_trace:
        from . import engine_execution_completeness as execution_accounting
        from .state import LEECH_SEED as _execution_leech_seed
        from . import engine_status_callbacks as status_callbacks
        from . import engine_weather_effect_callbacks as weather_effect_callbacks
        from . import engine_weather_order_callbacks as weather_order_callbacks
        if _weather_observer_context is not None:
            from . import engine_weather_hp_segment as weather_hp_segment
            from . import engine_weather_speed_profile as weather_speed_profile
            if (type(_weather_observer_context) is not tuple or len(_weather_observer_context)!=2
                    or (not _hypothetical and type(_weather_observer_context[0]) is not weather_speed_profile.BirthRecord)
                    or type(_weather_observer_context[1]) is not weather_speed_profile.ProfileData):
                raise ValueError('owning birth record and original static weather profile required')
            _weather_hp_record=_weather_observer_context[0]
    batch = state.hp.shape[0]
    shapes = ((batch,2,12),(batch,2),(batch,2),(batch,2),(batch,2),(batch,),(batch,2,4))
    for value,shape in zip(chance_inputs,shapes,strict=True):
        if value is None and shape == (batch,2,4):continue   # effect omitted: derived from the keys below
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
            from . import engine_kernel_frame_risk as kernel_risks
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
    # Fidelity patch R (design 2.1): the appended effect stream. A caller without it gets exactly what
    # sample_chance_inputs draws from these keys; a resumed suspended turn replays its saved block.
    effect_draws=chance_inputs.effect
    if effect_draws is None:effect_draws=effect_chance(splits[:,1])
    effect_draws=jnp.where(resuming[:,None,None],state.effect_draws,effect_draws)
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
            _weather_trace=_public_trace,_device_public=_device_public)
        work=work._replace(keys=jnp.where(resuming[:,None],state.keys,splits[:,0]),active_turns=age)
    else:
        work = initialize(state,table,eligible=live,tie_coin=chance_inputs.initial_tie,_device_public=_device_public)._replace(
            keys=jnp.where(resuming[:,None],state.keys,splits[:,0]),active_turns=age)
    if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live,_public_trace,_device_public=_device_public)
    else:work = _lum_update(work,table,live,_device_public=_device_public)
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
    # Fidelity M1: a mustrecharge holder's queued action is the pseudo-move 'recharge' (sim/side.ts
    # chooseMove, getLockedMove). It is no move (dex.moves.get misses): priority 0 and the user's
    # Speed order it, and no Focus Punch preparation or Pursuit can fire. Move id 0 has exactly that.
    _recharging = state.recharge_counter[rows[:, None], sides, state.active] > 0
    moves = jnp.where(_recharging, 0, moves)
    if _lifecycle_trace:_focus_capture=None
    if table.focus_punch is not None:
        if _lifecycle_trace:
            from . import engine_focus_punch_callbacks as focus_callbacks
            _public_trace,_focus_pending=focus_callbacks.begin(_public_trace,work,moves,slot,
                _direct_hit_context if _direct_hit_registered else None,table.focus_punch,
                ordinary=live & ~replacement,move_requests=(actions<4) & ~struggle)
        work=focusing.prepare(work,moves,table.focus_punch,ordinary=live & ~replacement,
            move_requests=(actions<4) & ~struggle)
        if _device_public is not None:
            for _device_side in range(2):
                device_public.move(_device_public,work,_device_side,moves[:,_device_side],
                    live & ~replacement & ((_active(work.volatile,work,_device_side)&FOCUS_PUNCH)!=0),
                    pp_use=False,count_turn=False,preparation=True)
        if _lifecycle_trace:
            _public_trace,_focus_capture=focus_callbacks.finish(_public_trace,_focus_pending,work)
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
            _faint_before_hp=work.hp
            if _device_public is not None:_device_pursuit_before=work
            work,used,fainted = intercepts.intercept(work,side,slot[:,side],moves[:,side],
                draws[:,side],table,eligible=eligible,contact_roll=phazing_draws[:,side])
            if _lifecycle_trace:
                _public_trace=status_callbacks.unknown(_public_trace,used)
            if _lifecycle_trace:work,_public_trace=_lum_update(work,table,used,_public_trace,_device_public=_device_public)
            else:work = _lum_update(work,table,used,_device_public=_device_public)
            # Destiny Bond holds through a switch until switch-out (gen4 pursuit's onBeforeSwitchOut,
            # inherited by Gen 3, does not remove it, unlike later gens): a Pursuit KO of the bonded
            # switcher faints the pursuer at faintMessages. useMove runs no BeforeMove, so the
            # pursuer's own bond is untouched (B2 still expires its bag on the Pursuit move line).
            _pursuit_bond=(fainted & ((_active(work.volatile,work,1-side)&jnp.uint32(DESTINY_BOND))!=0)
                & (_active(work.hp,work,side)>0))
            work=work._replace(hp=_set_active(work.hp,work,side,jnp.where(_pursuit_bond,0,_active(work.hp,work,side))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_pursuit_bond)
            if _device_public is not None:
                device_public.observe(_device_public,work,_pursuit_bond,record_damage=False)
                device_public.volatile(_device_public,work,side,'destinybond',jnp.zeros_like(used),used)
            work=_clear_fainted_boosts(work,_faint_before_hp,eligible=used)
            if _device_public is not None:
                device_public.move(_device_public,work,side,moves[:,side],used,pp_use=True)
                device_public.port_count(_device_public,side,used,'pursuit')
                device_public.item_changes(_device_public,_device_pursuit_before,work,used)
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
            # Retain outgoing Natural Cure before departure for the exact same-call
            # receipt below; unported incoming cures still refuse immediately.
            departure=jnp.zeros_like(do_switch)
            if table.entry is not None:
                departure=(_active(work.status,work,side)!=m.NONE) & (_active(work.hp,work,side)>0) & table.entry.natural_cure[_active(work.ability,work,side)]
            incoming=status_callbacks.entry_cure_possible(table,work.status[rows,side,target[rows,side]])
            _public_trace=status_callbacks.unknown(_public_trace,do_switch & incoming)
        installed = installed.at[rows,side].set(do_switch)
        if _lifecycle_trace:
            from . import engine_baton_pass_callbacks as baton_callbacks
            _public_trace,_baton_copy=baton_callbacks.before_copy(_public_trace,work,side,target[rows,side],
                _direct_hit_context if _direct_hit_registered else None,eligible=do_switch)
        work = passing.copy_to_recipient(work,side,target[rows,side],
            eligible=do_switch & work.baton_pass[rows,side])
        if _lifecycle_trace:
            _public_trace=baton_callbacks.after_copy(_public_trace,_baton_copy,work)
        if _lifecycle_trace:
            from . import engine_flash_fire_end_callbacks as _flash_end
            _public_trace,_flash_pending=_flash_end.begin(_public_trace,work,side,
                eligible=do_switch,baton=switch_baton,
                context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None)
        if _lifecycle_trace:
            from . import engine_natural_cure_callbacks as natural_cure_callbacks
            _public_trace,_natural_cure_pending=natural_cure_callbacks.begin(_public_trace,work,side,target[rows,side],table,
                _direct_hit_context if _direct_hit_registered else None,eligible=do_switch,baton=switch_baton)
            _public_trace=status_callbacks.unknown(_public_trace,do_switch & departure & ~natural_cure_callbacks.prospective(_natural_cure_pending))
        if _device_public is not None:
            _device_depart_before=work
            _device_baton=work.baton_pass[rows,side]
            device_public.switch_decision(_device_public,work,side,do_switch & ~replacement,baton=_device_baton)
        work=arrival.depart(work,side,table.entry,eligible=do_switch)
        if _device_public is not None:device_public.depart(_device_public,_device_depart_before,work,side,table.entry,do_switch)
        if _lifecycle_trace:
            _public_trace=natural_cure_callbacks.finish(_public_trace,_natural_cure_pending,work)
        if _lifecycle_trace:
            _public_trace=_flash_end.finish(_public_trace,_flash_pending,work)
        work = work._replace(active=work.active.at[rows,side].set(jnp.where(do_switch,target[rows,side],old_active)),
            baton_pass=work.baton_pass.at[rows,side].set(work.baton_pass[rows,side] & ~do_switch),
            forced_switch=work.forced_switch.at[rows,side].set(jnp.where(do_switch,False,work.forced_switch[rows,side])),
            active_turns=work.active_turns.at[rows,side].set(jnp.where(do_switch,0,work.active_turns[rows,side])))
        if _lifecycle_trace:
            _public_trace=lifecycle_callbacks.installed(_public_trace,work,side,old_active,target[rows,side],
                eligible=do_switch,replacement=replacement,baton=switch_baton)
            _public_trace=baton_callbacks.finish_install(_public_trace,work,side,eligible=do_switch)
        if _device_public is not None:
            device_public.install(_device_public,work,side,do_switch)
            device_public.baton_copy(_device_public,work,side,old_active,do_switch & _device_baton)
            device_public.withdraw(_device_public,work,side,old_active,do_switch)
        work=history.record_switch(work,side,target[rows,side],eligible=do_switch)
        # insertChoice snapshots each incoming callback's Speed now; a later
        # Air Lock/Cloud Nine entrant must not re-sort an already queued event.
        callback_speed = callback_speed.at[rows,side].set(_speeds(work,table)[rows,side])
        if table.entry is not None:
            if _device_public is not None:_device_entry_before=work
            _faint_before_hp=work.hp
            if _lifecycle_trace:
                work,knocked_out,_public_trace = arrival.enter(work,side,table.entry,
                    eligible=do_switch & ~replacement,_weather_trace=_public_trace)
            else:
                work,knocked_out = arrival.enter(work,side,table.entry,eligible=do_switch & ~replacement)
            if _device_public is not None:device_public.entry(_device_public,_device_entry_before,work,side,table.entry,do_switch & ~replacement)
            work=_clear_fainted_boosts(work,_faint_before_hp,eligible=do_switch & ~replacement)
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
            if _device_public is not None:_device_entry_before=work
            _faint_before_hp=work.hp
            if _lifecycle_trace:
                work,knocked_out,_public_trace = arrival.enter(work,side,table.entry,
                    eligible=eligible,_weather_trace=_public_trace)
            else:
                work,knocked_out = arrival.enter(work,side,table.entry,eligible=eligible)
            if _device_public is not None:device_public.entry(_device_public,_device_entry_before,work,side,table.entry,eligible)
            work=_clear_fainted_boosts(work,_faint_before_hp,eligible=eligible)
            # A hazard KO cancels subsequent queued runSwitch callbacks too.
            switch_faint |= knocked_out
    if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live,_public_trace,_device_public=_device_public)
    else:work = _lum_update(work,table,live,_device_public=_device_public)
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
    # Counter/Mirror Coat (design 3.A2.3): the 1-turn volatile's record of the last foe Move hit
    # on the Pokemon this turn. Priority -5 and Gen 3's faint cancellation keep the hit and the
    # reply in one step (a suspension follows only a Baton Pass, which deals no damage).
    _taken_amount=jnp.zeros_like(state.active)
    _taken_kind=jnp.zeros_like(state.active)    # 0 none, 1 Counter-able, 2 Mirror Coat-able
    for order in range(2):
        _faint_before_hp=work.hp
        side = jnp.where(order == 0, first, 1 - first)
        foe = 1 - side
        _bond_foe_hp=_active(work.hp,work,foe)
        move = moves[rows, side]
        chosen_slot=slot[rows,side]
        move,chosen_slot=repeating.override(work,side,move,chosen_slot,struggle=struggle[rows,side])
        caller_move=move
        charge_locked=_active(work.charge_move,work,side)>0
        chance = draws[rows, side]
        move_context = ~replacement | (state.pending_residual & (saved_actor == side))
        eligible = (live & move_context & ~paused & ~moves_cancelled & ~switches[rows, side]
            & ~pursuit_used[rows,side] & ~displaced[rows,side] & (_active(work.hp, work, side) > 0))
        # Destiny Bond ends at the holder's next move attempt: its onBeforeMove (priority -1), or
        # MoveAborted when sleep, freeze, flinch, paralysis, a confusion self-hit or Truant stops the
        # move (data/moves.ts destinybond condition). A Destiny Bond executed below sets it again
        # (gen6 onPrepareHit, inherited by Gen 3: consecutive uses succeed).
        work=work._replace(volatile=_set_active(work.volatile,work,side,jnp.where(eligible,
            _active(work.volatile,work,side)&jnp.uint32(~DESTINY_BOND&0xffffffff),_active(work.volatile,work,side))))
        if _device_public is not None:device_public.destiny_bond(_device_public,work,side,eligible)
        # Fidelity M1: mustrecharge's onBeforeMove (data/conditions.ts, priority 11) runs before sleep (10),
        # freeze, flinch, confusion and Gen 3 Truant: cant|X|recharge, the volatile ends, the move stops
        # (MoveAborted, so Destiny Bond above still ended). No sleep or confusion tick, Truant not shown.
        _recharge_turn=eligible & (_active(work.recharge_counter,work,side)>0)
        work=work._replace(recharge_counter=_set_active(work.recharge_counter,work,side,
            jnp.where(_recharge_turn,0,_active(work.recharge_counter,work,side))))
        if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_recharge_turn)
        if _device_public is not None:device_public.recharge_turn(_device_public,work,side,_recharge_turn)
        eligible=eligible & ~_recharge_turn
        sleep_usable=jnp.zeros_like(live);early_bird=jnp.zeros_like(live)
        if table.sleep_talk is not None:
            sleep_usable=table.sleep_talk.move[move]
            early_bird=table.sleep_talk.early_bird[_active(work.ability,work,side)]
        loafing=jnp.zeros_like(live)
        if table.entry is not None and table.entry.truant is not None:
            loafing=table.entry.truant[_active(work.ability,work,side)] & _active(work.truant_loaf,work,side)
        if _lifecycle_trace:
            from . import engine_sleep_blocked_callbacks as sleep_blocked_callbacks
            _sleep_blocked_old_work=work
            _public_trace,_sleep_blocked_pending=sleep_blocked_callbacks.begin(_public_trace,work,side,move,
                _direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,table.sleep_talk,
                eligible=eligible,sleep_usable=sleep_usable,early_bird=early_bird,loafing=loafing)
        if _lifecycle_trace:
            from . import engine_before_move_blocked_callbacks as before_move_blocked_callbacks
            _public_trace,_before_move_blocked_pending=before_move_blocked_callbacks.begin(_public_trace,work,side,move,caller_move,chance[:,1:4],
                _direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,
                eligible=eligible,sleep_usable=sleep_usable,loafing=loafing)
        if _lifecycle_trace:
            from . import engine_sleep_talk_callbacks as sleep_talk_callbacks
            _public_trace,_sleep_talk_pending=sleep_talk_callbacks.begin(_public_trace,work,side,caller_move,chosen_slot,
                _direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,table.sleep_talk,
                choice_roll=sleep_talk_draws[rows,side],eligible=eligible,sleep_usable=sleep_usable,early_bird=early_bird,loafing=loafing)
        if _device_public is not None:_device_before_move=work
        before = m.before_move(_active(work.status, work, side), _active(work.status_counter, work, side),
            _active(work.confusion_counter, work, side), thaw_roll=chance[:, 1],
            paralysis_roll=chance[:, 2], confusion_roll=chance[:, 3],
            flinched=(_active(work.volatile, work, side) & FLINCH) != 0,
            sleep_usable=sleep_usable,early_bird=early_bird,loafing=loafing)
        if _device_public is not None:device_public.before_move(_device_public,work,side,before,eligible,loafing=loafing,sleep_usable=sleep_usable)
        if _lifecycle_trace:
            from . import engine_outcome_diagnostics as _before_move_diagnostics
            _public_trace=_before_move_diagnostics.record_before_move_on_tape(_public_trace,before,side,
                work.active[rows,side],move,caller_move,_active(work.status,work,side),
                _active(work.status_counter,work,side),_active(work.confusion_counter,work,side),
                hp=_active(work.hp,work,side),maximum=_active(work.max_hp,work,side),ability=_active(work.ability,work,side),
                thaw_roll=chance[:,1],paralysis_roll=chance[:,2],confusion_roll=chance[:,3],
                flinched=(_active(work.volatile,work,side)&FLINCH)!=0,
                sleep_usable=sleep_usable,early_bird=early_bird,loafing=loafing,eligible=eligible,active=live)
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
        if _lifecycle_trace:
            _public_trace,_sleep_blocked_capture=sleep_blocked_callbacks.finish(_public_trace,_sleep_blocked_pending,work,before)
            _public_trace,_before_move_blocked_capture=before_move_blocked_callbacks.finish(_public_trace,_before_move_blocked_pending,work,before)
            _public_trace=kernel_risks.pre_move_on_tape(_public_trace,_sleep_blocked_old_work,side,before,
                eligible=eligible,active=live,captured_sleep=_sleep_blocked_capture,captured_before_move=_before_move_blocked_capture)
        if _device_public is not None:device_public.observe(_device_public,work,eligible)
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
        if _lifecycle_trace:
            _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.CONFUSION_HP,
                eligible & before.confusion_self_hit,active=live)
        hp = jnp.where(eligible & before.confusion_self_hit, jnp.maximum(0, hp - confusion_damage), hp)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        # Fidelity patch B: the self-hit's -damage|X|..|[from] confusion is an HP line; B2's absent-item evidence
        # takes it as action-phase HP in either phase (pokezero.belief _ACTION_PHASE_ONLY_HP_TAGS).
        if _device_public is not None:device_public.hp_line(_device_public,work,side,eligible & before.confusion_self_hit,source_class=device_public.HP_ACTION)
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
        if _device_public is not None:_device_move_before=work
        if _device_public is not None:
            device_public.observe(_device_public,work,eligible)
            device_public.move(_device_public,work,side,caller_move,caller_acts,
                pp_use=~charge_locked & ~struggle[rows,side],original=not False)
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
        if table.damage_moves is not None:
            # Counter/Mirror Coat onTry (gen3 moves.ts): no volatile or no qualifying hit -> silent
            # failure after PP, before accuracy. beforeTurnCallback adds the volatile only for the
            # move queued at the start of the turn, so a Sleep Talk or same-turn Encore call fails.
            _counter_kind=table.damage_moves.counter_kind[move]
            acts &= ~((_counter_kind>0) & ((_taken_kind[rows,side]!=_counter_kind) | (moves[rows,side]!=move)))
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
            if _device_public is not None:
                for _device_side in range(2):
                    device_public.reveal_ability(_device_public,work,_device_side,
                        acts & blocked & table.self_destruct.damp[_active(work.ability,work,_device_side)] & (_active(work.hp,work,_device_side)>0))
            acts &= ~blocked
            explodes=acts & table.self_destruct.move[move]
        if _device_public is not None:
            device_public.move(_device_public,work,side,move,acts & (move!=caller_move),pp_use=False,count_turn=False)
        if _lifecycle_trace:
            from . import engine_self_move_callbacks as self_move_callbacks
            _self_move_context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None
            _sleep_failure=self_move_callbacks.awake_failure(_public_trace,work,side,caller_move,_self_move_context,eligible=caller_acts,acts=acts)
        if _lifecycle_trace:
            _public_trace,_sleep_talk_capture=sleep_talk_callbacks.finish(_public_trace,_sleep_talk_pending,work,before,move,
                usable if table.sleep_talk is not None else jnp.zeros_like(acts),sleep_talk_draws[rows,side],acts=acts,caller_acts=caller_acts)
            _sleep_talk_supported=sleep_talk_callbacks.captured_matches(_public_trace,_sleep_talk_capture)
        if _public_trace is not None:
            # A genuine ordinary declaration is known here, after the actual
            # suppression/preparation gates and before any main-hit callback.
            _public_trace = public_callbacks.move(_public_trace,side,work.active[rows,side],
                work.active[rows,foe],caller_move,eligible=caller_acts & ~(_sleep_talk_supported if _lifecycle_trace else jnp.zeros_like(acts)),
                unsupported=charge_locked | ((sleep_usable | ~acts) & ~(_sleep_failure if _lifecycle_trace else jnp.zeros_like(acts))))
        if _lifecycle_trace:
            from . import engine_protect_callbacks as protect_callbacks
            _protect_pending=None;_protect_block_capture=None;_protect_block_mask=jnp.zeros_like(acts)
            if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.protect is not None:
                _protect_count=jnp.where(state.pending_residual,saved_stall[rows,side],_active(old_stall,work,side))
                _public_trace,_protect_pending=protect_callbacks.begin(_public_trace,work,side,move,_protect_count,chance[:,8],
                    jnp.full_like(acts,order==0),~state.pending_residual,~switches[rows,foe],
                    _direct_hit_context,table.effects,eligible=acts)
            _public_trace,_sleep_pending=self_move_callbacks.begin(_public_trace,work,side,caller_move,_self_move_context,code=self_move_callbacks.SLEEP_FAIL,eligible=_sleep_failure)
            _public_trace,_sleep_capture=self_move_callbacks.finish(_public_trace,_sleep_pending,work)
            _public_trace=kernel_risks.selected_on_tape(_public_trace,jnp.where(_sleep_failure,caller_move,move),
                eligible=caller_acts,active=live,captured_focus=_focus_capture,state=work,side=side,caller_move=caller_move)
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
        if table.damage_moves is not None:
            # damageCallback: twice the recorded damage, no crit or roll; type immunity still applies
            # (Ghost to Counter, Dark to Mirror Coat).
            damage=jnp.where(table.damage_moves.counter_kind[move]>0,
                jnp.where(immune,0,special_hits.counter_damage(_taken_amount[rows,side])),damage)
        # Bonemerang's second hit (design R.2): every hit of Gen 3's multihit loop is its own moveHit, so it
        # runs its own getDamage: a critical roll and a damage roll, here from the appended effect stream
        # (effect[s][0], 85+effect[s][1]). Same attacker, target and modifiers as the first hit.
        _effect=effect_draws[rows,side]
        critical2=_effect[:,0]==0
        if table.critical is not None:
            critical2=crits.is_critical(table.critical,move,_active(work.ability,work,foe),_effect[:,0])
        damage2=m.gen3_damage(level=_active(work.level, work, side), power=power,
            attack=jnp.where(physical, attack_stats[:, 1], attack_stats[:, 3]),
            defense=jnp.where(physical, defense_stats[:, 2], defense_stats[:, 4]),
            attack_stage=jnp.where(physical, attack_boosts[:, 0], attack_boosts[:, 2]),
            defense_stage=jnp.where(physical, defense_boosts[:, 1], defense_boosts[:, 3]),
            physical=physical, critical=critical2,halve_defense=explodes & physical,hustle=hustle,
            attack_num=attack_num,attack_den=attack_den,
            defense_num=defense_num,defense_den=defense_den,
            weather_num=weather_num,weather_den=weather_den,
            phase1_num=phase1_num,phase1_den=phase1_den,
            stab=jnp.any(own_types == typ[:, None], axis=1) & (typ != 0),
            typeless=typ == 0,
            burned=_active(work.status, work, side) == m.BRN,
            guts=guts,
            effectiveness_num=num, effectiveness_den=den, immune=immune, roll=85 + _effect[:, 1])
        # The same second hit after a Color Change on hit 1 (gen4 colorchange onDamagingHit, per hit): the target
        # is then the move's type alone, so hit 2's effectiveness is the move's type against [type, none].
        _num_cc=table.type_num[typ,typ]*table.type_num[typ,jnp.zeros_like(typ)]
        _den_cc=table.type_den[typ,typ]*table.type_den[typ,jnp.zeros_like(typ)]
        damage2_cc=m.gen3_damage(level=_active(work.level, work, side), power=power,
            attack=jnp.where(physical, attack_stats[:, 1], attack_stats[:, 3]),
            defense=jnp.where(physical, defense_stats[:, 2], defense_stats[:, 4]),
            attack_stage=jnp.where(physical, attack_boosts[:, 0], attack_boosts[:, 2]),
            defense_stage=jnp.where(physical, defense_boosts[:, 1], defense_boosts[:, 3]),
            physical=physical, critical=critical2,halve_defense=explodes & physical,hustle=hustle,
            attack_num=attack_num,attack_den=attack_den,
            defense_num=defense_num,defense_den=defense_den,
            weather_num=weather_num,weather_den=weather_den,
            phase1_num=phase1_num,phase1_den=phase1_den,
            stab=jnp.any(own_types == typ[:, None], axis=1) & (typ != 0),
            typeless=typ == 0,
            burned=_active(work.status, work, side) == m.BRN,
            guts=guts,
            effectiveness_num=_num_cc, effectiveness_den=_den_cc, immune=immune, roll=85 + _effect[:, 1])
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
        if _direct_hit_context is not None and _direct_hit_registered:
            _direct_accuracy_hit=hit
        if _thunder_wave_context is not None:
            _thunder_wave_accuracy_hit=hit
        if _sleep_powder_context is not None:
            _sleep_powder_accuracy_hit=hit
        if _toxic_context is not None:
            _toxic_accuracy_hit=hit  # Literal accuracy result BEFORE Protect/absorption.
        # TryMove (including Damp) succeeds before self-KO, which occurs even
        # if the subsequent target hit misses, is protected, immune or a sub.
        # Damage inputs were resolved while the user's ability was still active.
        if _lifecycle_trace:
            from . import engine_self_destruct_callbacks as self_destruct_callbacks
            _public_trace,_self_destruct_pending=self_destruct_callbacks.begin(_public_trace,work,side,move,caller_move,
                _direct_hit_context if _direct_hit_registered else None,requested=explodes)
        work=work._replace(hp=_set_active(work.hp,work,side,
            jnp.where(explodes,0,_active(work.hp,work,side))))
        if _device_public is not None:
            # Self-Destruct emits faint, not a damage condition for its user.
            device_public.observe(_device_public,work,explodes,record_damage=False)
        if _lifecycle_trace:
            _public_trace,_self_destruct_pending=self_destruct_callbacks.self_write(_public_trace,_self_destruct_pending,work)
            _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.SELF_DESTRUCT,
                explodes,active=live)
        bypass_sub = jnp.zeros_like(acts)
        if _lifecycle_trace:
            _risk_protected=jnp.zeros_like(acts)
            _risk_absorbed=jnp.zeros_like(acts)
        if table.effects is not None:
            if _lifecycle_trace:
                _risk_protected=protected[rows,foe] & table.effects.protectable[move]
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.protect is not None:
                    _public_trace,_protect_block_pending=protect_callbacks.begin_block(_public_trace,work,side,move,hit,
                        protected[rows,foe],table.effects.protectable[move],_direct_hit_context,eligible=acts)
            if _device_public is not None:
                device_public.port_protect(_device_public,side,acts,hit,protected[rows,foe],table.effects.protectable[move])
            hit &= ~(protected[rows,foe] & table.effects.protectable[move])
            if _lifecycle_trace and _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.protect is not None:
                _public_trace,_protect_block_capture=protect_callbacks.finish_block(_public_trace,_protect_block_pending,hit)
                _protect_block_mask=protect_callbacks.captured_block_matches(_public_trace,_protect_block_capture)
            bypass_sub = table.effects.bypass_sub[move]
        # Gen3 emits a generic -immune for ordinary Ground/Levitate immunity.
        # B2 does not mark an ability revealed without an attribution line;
        # source possibilities can still imply Levitate without that flag.
        if _direct_hit_context is not None and _direct_hit_registered:
            from . import engine_no_damage_callbacks as no_damage_callbacks
            from . import engine_substitute_damage_callbacks as substitute_damage_callbacks
            _sub_damage_requested=substitute_damage_callbacks.route(_direct_hit_context,move,
                acts=acts,power=power,hit=hit,immune=immune,protected=_risk_protected,
                enduring=enduring[rows,foe],bypass=bypass_sub,absorbed=_risk_absorbed,
                sub_hp=_active(work.substitute_hp,work,foe),actor_happiness=_active(work.happiness,work,side),caller_move=caller_move,actor_level=_active(work.level,work,side),actor_hp_type=_active(work.hidden_power_type,work,side),actor_hp_power=_active(work.hidden_power_power,work,side),resolved_type=typ)
            _public_trace,_sub_damage_ticket=execution_accounting.open_attempt(_public_trace,
                execution_accounting.Site.SUBSTITUTE_DAMAGE,foe,active=_sub_damage_requested)
            _sub_damage_pending=substitute_damage_callbacks.Pending(_direct_hit_context,_sub_damage_ticket,
                side,work.active[rows,side],work.active[rows,foe],move,_sub_damage_requested,_active(work.happiness,work,side),caller_move,_active(work.level,work,side),_active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),typ)
            _direct_requested=acts & ((power>0)|direct_hit_controls.fixed_level_move_device(_direct_hit_context.data,move)) & ~_sub_damage_requested & ~_protect_block_mask
            if _lifecycle_trace and _self_destruct_pending is not None:_direct_requested &= ~_self_destruct_pending.requested
            _public_trace,_direct_ticket=execution_accounting.open_attempt(_public_trace,
                execution_accounting.Site.DIRECT_HIT,foe,active=_direct_requested)
            _direct_ko_actor_before=_active(work.hp,work,side)
            _direct_ko_party_before=work.hp[rows,foe]
            _no_damage_before=no_damage_callbacks.Before(_direct_accuracy_hit,typ,_active(work.ability,work,foe),
                _active(work.status,work,foe),_active(work.hp,work,foe),_active(work.max_hp,work,foe),
                _active(work.volatile,work,foe),_active(work.substitute_hp,work,foe),_active(work.hp,work,side),explodes,
                _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                _active(work.types,work,foe),_active(work.item,work,foe))
            _no_damage_pending=no_damage_callbacks.prepare(_public_trace,_direct_ticket,side,work.active[rows,side],
                work.active[rows,foe],move,_no_damage_before,_direct_hit_context,requested=_direct_requested,caller_move=caller_move)
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
            if _lifecycle_trace:
                _risk_absorbed=absorbed.blocked
            work = work._replace(hp=_set_active(work.hp,work,foe,absorbed.hp),
                volatile=_set_active(work.volatile,work,foe,absorbed.volatile))
            if _direct_hit_context is not None and _direct_hit_registered:
                _no_damage_pending=no_damage_callbacks.observed_absorption(_no_damage_pending,absorbed)
            if _device_public is not None:
                device_public.reveal_ability(_device_public,work,foe,absorbed.blocked)
                device_public.observe(_device_public,work,acts)
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
        if _device_public is not None:
            device_public.damage(_device_public,work,side,jnp.where(hits_sub,0,dealt),jnp.where(hits_sub,0,dealt),acts)
            device_public.negative_hp_line(_device_public,work,foe,acts & ~hits_sub & (dealt>0),source_class=device_public.negative_items.HP_ACTION)
            device_public.observe(_device_public,work,acts)
        if table.damage_moves is not None:
            # Counter/Mirror Coat's onDamage record (gen3 moves.ts, priority -101, so after Endure):
            # a foe Move whose damage reaches the Pokemon; the last hit wins. Substitute hits never
            # reach onDamage. Counter counts physical moves and Hidden Power, Mirror Coat the rest.
            _taken=acts & hit & ~immune & ~hits_sub & (damage>0)
            _taken_amount=_taken_amount.at[rows,foe].set(jnp.where(_taken,dealt,_taken_amount[rows,foe]))
            _taken_kind=_taken_kind.at[rows,foe].set(jnp.where(_taken,
                jnp.where(physical | table.damage_moves.hidden_power[move],1,2),_taken_kind[rows,foe]))
        if table.damage_moves is not None:
            # Giga Drain (drain [1,2]) heals the user from the damage dealt, also through a
            # Substitute; at full HP the heal is silent. Liquid Ooze (gen4 abilities.ts, the target's
            # onSourceTryHeal) instead damages the drainer by the same amount, even at full HP, and
            # an Ooze KO faints the drainer first (instafaint).
            _drained=acts & hit & ~immune & (dealt>0) & (table.damage_moves.drain_den[move]>0)
            _ooze=_drained & table.damage_moves.liquid_ooze[_active(work.ability,work,foe)]
            _drain=special_hits.drain_amount(dealt,hits_sub)
            _drain_old=_active(work.hp,work,side)
            _drain_new=jnp.where(_ooze,jnp.maximum(0,_drain_old-_drain),
                jnp.minimum(_active(work.max_hp,work,side),_drain_old+_drain))
            work=work._replace(hp=_set_active(work.hp,work,side,jnp.where(_drained,_drain_new,_drain_old)))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_drained)
            if _device_public is not None:
                device_public.reveal_ability(_device_public,work,foe,_ooze)
                device_public.hp_line(_device_public,work,side,_drained & (_drain_new!=_drain_old),
                    source_class=device_public.HP_ACTION)
        if _direct_hit_context is not None and _direct_hit_registered:
            _sub_damage_write=substitute_damage_callbacks.Write(power,hit,immune,_risk_protected,
                enduring[rows,foe],bypass_sub,_risk_absorbed,critical,num,den,sub_hp,
                _active(work.substitute_hp,work,foe),dealt,damage,hp,foe_hp)
            _public_trace=substitute_damage_callbacks.record(_public_trace,_sub_damage_pending,_sub_damage_write)
        if _direct_hit_context is not None:
            _direct_write=direct_hit_controls.HitWrite(power,critical,hit,immune,num,den,hits_sub,
                _risk_protected,enduring[rows,foe],hp,foe_hp,_active(work.max_hp,work,foe))
            if _lifecycle_trace:
                _public_trace=self_destruct_callbacks.target_write(_public_trace,_self_destruct_pending,_direct_write,secondary=acts & hit & ~immune)
            if _direct_hit_registered:
                _public_trace=direct_hit_controls.record_registered_direct_hit_controls_device(
                    _public_trace,side,work.active[rows,side],work.active[rows,foe],move,
                    _direct_write,_direct_hit_context,eligible=_direct_requested & (_direct_write.new_hp!=0) & ~no_damage_callbacks.scheduled(_no_damage_pending,_direct_write))
                _public_trace,_direct_ko_pending=direct_ko_callbacks.prepare_controls_device(
                    _public_trace,_direct_ticket,side,work.active[rows,side],work.active[rows,foe],move,
                    _direct_write,_direct_hit_context,_direct_ko_actor_before,_direct_ko_party_before,
                    eligible=_direct_requested,actor_happiness=_active(work.happiness,work,side),
                    recoil_before=(recoil_callbacks.KOBefore(_recoil_context,caller_move,_active(work.ability,work,side),
                        _active(work.max_hp,work,side),_active(work.status,work,side),jnp.full_like(move,-1))if _recoil_context is not None else None))
            else:
                _public_trace=direct_hit_controls.record_direct_hit_controls_device(
                    _public_trace,side,work.active[rows,side],work.active[rows,foe],move,
                    _direct_write,_direct_hit_context.data,
                    eligible=acts & ((power>0)|direct_hit_controls.fixed_level_move_device(_direct_hit_context.data,move)))
        if _public_trace is not None:
            _public_trace = public_callbacks.condition(_public_trace,foe,work.active[rows,foe],
                foe_hp,_active(work.max_hp,work,foe),_active(work.status,work,foe),
                eligible=acts & hit & ~hits_sub & (dealt>0))
        if _direct_hit_context is not None and _direct_hit_registered:
            _public_trace=execution_accounting.capture_direct_hit_write(_public_trace,_direct_ticket,
                side,work.active[rows,side],work.active[rows,foe],move,_direct_write,
                requested=_direct_requested & (_direct_write.new_hp!=0) & ~no_damage_callbacks.scheduled(_no_damage_pending,_direct_write),data=_direct_hit_context.data)
        if table.damage_moves is not None:
            # Bonemerang (data/moves.ts multihit:2; design R.2). Gen 3's tryMoveHit (data/mods/gen3/scripts.ts)
            # rolls accuracy once, then hits while `i < hits && target.hp && pokemon.hp`: a first-hit KO ends the
            # move. Each hit is a full moveHit: the Substitute check (gen4 substitute onTryPrimaryHit), its own
            # crit and damage roll (above), Endure. When the first hit breaks the Substitute the second hits the
            # Pokemon (gen4's substitutebroken volatile has no handler). Counter/Mirror Coat's onDamage records
            # every hit that reaches the Pokemon, so the last one wins; Endure's 0 at 1 HP stops the Damage event
            # before it (runEvent breaks on a falsy relay), keeping the earlier record. B2 sees one -damage line per
            # hit and a -hitcount (no tracker reads it). Downstream (Focus Punch, the end-of-move Color Change for a
            # hit 2 that is the first to reach the Pokemon, the faint cancellation) reads the final hit, which reaches
            # the Pokemon exactly when either hit did.
            _second=(acts & hit & ~immune & (table.damage_moves.hits[move]>1)
                & (_active(work.hp,work,foe)>0) & (_active(work.hp,work,side)>0))
            # DamagingHit runs after each hit (sim/battle-actions.ts spreadMoveHit): Color Change retypes a target
            # that hit 1 damaged and left standing, before hit 2's getDamage reads the type (Kecleon after Rock Slide:
            # Ground hit 1 is 2x, hit 2 against the new Ground type 1x). A later end-of-move call then finds it done.
            if table.secondary is not None:
                _types1=_active(work.types,work,foe)
                work=secondaries.apply_color_change(work,side,typ,table.secondary,
                    eligible=_second & ~hits_sub & (dealt>0),_device_public=_device_public)
                damage2=jnp.where(jnp.any(_active(work.types,work,foe)!=_types1,axis=1),damage2_cc,damage2)
            _hp1=_active(work.hp,work,foe);_sub1=_active(work.substitute_hp,work,foe)
            _sub2=(_sub1>0) & ~bypass_sub
            _capacity2=jnp.where(_sub2,_sub1,jnp.where(enduring[rows,foe],jnp.maximum(0,_hp1-1),_hp1))
            _dealt2=jnp.where(_second,jnp.minimum(_capacity2,damage2),0)
            _foe_hp2=_hp1-jnp.where(_sub2,0,_dealt2)
            work=work._replace(hp=_set_active(work.hp,work,foe,_foe_hp2),
                substitute_hp=_set_active(work.substitute_hp,work,foe,_sub1-jnp.where(_sub2,_dealt2,0)))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_second)
            if _device_public is not None:
                device_public.damage(_device_public,work,side,jnp.where(_sub2,0,_dealt2),jnp.where(_sub2,0,_dealt2),_second)
                device_public.negative_hp_line(_device_public,work,foe,_second & ~_sub2 & (_dealt2>0),source_class=device_public.negative_items.HP_ACTION)
                device_public.observe(_device_public,work,_second)
            _taken2=_second & ~_sub2 & (_dealt2>0)
            _taken_amount=_taken_amount.at[rows,foe].set(jnp.where(_taken2,_dealt2,_taken_amount[rows,foe]))
            _taken_kind=_taken_kind.at[rows,foe].set(jnp.where(_taken2,
                jnp.where(physical | table.damage_moves.hidden_power[move],1,2),_taken_kind[rows,foe]))
            dealt=jnp.where(_second,_dealt2,dealt)
            hits_sub=jnp.where(_second,_sub2,hits_sub)
            foe_hp=jnp.where(_second,_foe_hp2,foe_hp)
        if table.focus_punch is not None:
            work=focusing.hit(work,side,move,table.focus_punch,eligible=acts & hit & ~immune & ~hits_sub)
        # Gen 3 floors recoil from actual damage, including capped sub damage.
        recoil=jnp.where(is_struggle & (dealt>0),jnp.maximum(1,dealt//4),0)
        if table.damage_moves is not None:
            recoil=special_hits.recoil_damage(table.damage_moves,move,_active(work.ability,work,side),dealt)
        if _recoil_context is not None:
            _direct_ko_pending=recoil_callbacks.with_ko_amount(_direct_ko_pending,recoil)
            _recoil_captured=kernel_risks.CapturedDirectHit(_direct_hit_context,_direct_ticket,side,
                work.active[rows,side],work.active[rows,foe],move,_direct_write)
        if _lifecycle_trace:
            _literal_outcome=kernel_risks.Outcome(power,critical,hit,immune,num,den,_risk_absorbed,
                hits_sub,_risk_protected,enduring[rows,foe],recoil,explodes)
            if _direct_hit_context is not None and _direct_hit_registered:
                _public_trace=no_damage_callbacks.finish(_public_trace,_no_damage_pending,_direct_write,_literal_outcome,
                    _active(work.hp,work,side),_active(work.volatile,work,foe),_active(work.substitute_hp,work,foe),
                    _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                    target_types=_active(work.types,work,foe),item=_active(work.item,work,foe),ability=_active(work.ability,work,foe))
            if _recoil_context is not None:
                # Open Site14 only after the actual no-damage Site10 pair. On
                # positive/KO paths finish above emits nothing: same boundary.
                _recoil_no_damage=no_damage_callbacks.Captured(_no_damage_pending,_direct_write,_active(work.hp,work,side),
                    _active(work.volatile,work,foe),_active(work.substitute_hp,work,foe),
                    _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                    _active(work.types,work,foe),_active(work.item,work,foe),_active(work.ability,work,foe))
                _public_trace,_recoil_pending=recoil_callbacks.prepare_device(_public_trace,_recoil_captured,recoil,
                    _active(work.ability,work,side),_active(work.hp,work,side),_active(work.max_hp,work,side),
                    _active(work.status,work,side),_recoil_context,eligible=acts,target_ko=_direct_ko_pending,caller_move=caller_move,
                    no_damage=_recoil_no_damage,no_damage_outcome=_literal_outcome,
                    substitute_damage=substitute_damage_callbacks.Captured(_sub_damage_pending,_sub_damage_write))
            from . import engine_outcome_diagnostics as outcome_diagnostics
            _public_trace=outcome_diagnostics.record_on_tape(_public_trace,_literal_outcome,move,side,
                work.active[rows,side],work.active[rows,foe],eligible=acts,active=live)
            _public_trace=kernel_risks.outcome_on_tape(_public_trace,_literal_outcome,
                eligible=acts,active=live,captured_direct=(kernel_risks.CapturedDirectHit(
                    _direct_hit_context,_direct_ticket,side,work.active[rows,side],work.active[rows,foe],move,_direct_write)
                    if _direct_hit_context is not None and _direct_hit_registered else None),
                captured_substitute_damage=(substitute_damage_callbacks.Captured(_sub_damage_pending,_sub_damage_write)
                    if _direct_hit_context is not None and _direct_hit_registered else None),
                pending_recoil=_recoil_pending if _recoil_context is not None else None,
                pending_ko=_direct_ko_pending if _direct_hit_context is not None and _direct_hit_registered else None,
                captured_protect_block=_protect_block_capture,
                captured_no_damage=(no_damage_callbacks.Captured(_no_damage_pending,_direct_write,_active(work.hp,work,side),
                    _active(work.volatile,work,foe),_active(work.substitute_hp,work,foe),
                    _active(work.hidden_power_type,work,side),_active(work.hidden_power_power,work,side),
                    _active(work.types,work,foe),_active(work.item,work,foe),_active(work.ability,work,foe))
                    if _direct_hit_context is not None and _direct_hit_registered else None))
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
            if _device_public is not None:
                device_public.feature(_device_public,side,work.active[rows,side],'stall_counter',jnp.ones_like(move),stall,add=True)
                device_public.feature(_device_public,side,work.active[rows,side],'stall_counter',jnp.zeros_like(move),acts & ~stall)
            if _lifecycle_trace:
                _public_trace=protect_callbacks.finish(_public_trace,_protect_pending,work,stall,protected[rows,side],acts&hit&~immune)
            # Self effects use the active actor, never the opposing target.
            hp = _active(work.hp, work, side)
            max_hp = _active(work.max_hp, work, side)
            if _device_public is not None:_device_self_effect_hp=hp
            sub_hp = _active(work.substitute_hp,work,side)
            creates_sub = (acts & effects.substitute[move] & (sub_hp == 0)
                & (max_hp > 1) & (hp * 4 > max_hp))
            if _lifecycle_trace:
                _public_trace,_rest_pending=self_move_callbacks.begin(_public_trace,work,side,move,_self_move_context,code=self_move_callbacks.REST_FAIL,eligible=acts&caller_acts,secondary=acts&hit&~immune,caller_move=caller_move)
                from . import engine_self_heal_callbacks as self_heal_callbacks
                _public_trace,_self_heal_pending=self_heal_callbacks.begin(_public_trace,work,side,move,
                    _direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,table,eligible=acts&caller_acts&(caller_move==move))
            if _substitute_context is not None:
                _substitute_old_hp=hp
                _substitute_requested=acts & effects.substitute[move]
                if _substitute_registered:
                    _public_trace,_substitute_ticket=execution_accounting.open_attempt(_public_trace,
                        execution_accounting.Site.SUBSTITUTE_SELF,side,active=_substitute_requested)
            cost = max_hp // 4
            hp -= jnp.where(creates_sub,cost,0)
            work = work._replace(substitute_hp=_set_active(work.substitute_hp,work,side,
                jnp.where(creates_sub,cost,sub_hp)))
            # Fidelity M2: the Substitute condition's onStart (data/moves.ts:18343-18346; Gen 3 inherits it) ends the
            # user's partial trap: -end|X|Wrap|[partiallytrapped]|[silent].
            _sub_freed=creates_sub & (_active(work.partial_trap_counter,work,side)>0)
            work=work._replace(partial_trap_counter=_set_active(work.partial_trap_counter,work,side,
                jnp.where(_sub_freed,0,_active(work.partial_trap_counter,work,side))),
                partial_trap_source=_set_active(work.partial_trap_source,work,side,
                    jnp.where(_sub_freed,-1,_active(work.partial_trap_source,work,side))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_sub_freed)
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
            # Rest's onTry (data/moves.ts Rest) fails an Insomnia/Vital Spirit user after
            # its asleep and full-HP checks, with -fail [from] ability (revealed below).
            _rest_immune=jnp.zeros_like(rest)
            if table.secondary is not None:
                _rest_immune=rest & table.secondary.ability_immunity[_active(work.ability,work,side),m.SLP]
            rest &= ~_rest_immune
            if _lifecycle_trace:
                _rest_write_ready=self_move_callbacks.rest_write_supported(_public_trace,_rest_pending,rest)
                _public_trace=status_callbacks.unknown(_public_trace,rest&~_rest_write_ready)
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
            # Refresh (data/moves.ts:14924-14927) fails on no status, slp or frz; otherwise
            # cureStatus() clears par/brn/psn/tox with its toxic stage. Published by observe below.
            _refresh_status=_active(work.status,work,side)
            _refreshed=(acts & effects.refresh[move] & (_active(work.hp,work,side)>0)
                & ((_refresh_status==m.PAR)|(_refresh_status==m.BRN)|(_refresh_status==m.PSN)|(_refresh_status==m.TOX)))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_refreshed)
            work=work._replace(**{name:_set_active(getattr(work,name),work,side,jnp.where(_refreshed,
                jnp.zeros_like(_active(getattr(work,name),work,side)),_active(getattr(work,name),work,side)))
                for name in ("status","status_counter","sleep_skipped","rest_origin")})
            if _device_public is not None:
                # Successful self-heals and Substitute publish a condition
                # before residual items. Replace the pre-item HP snapshot at
                # that write, rather than retaining an earlier damaged HP.
                device_public.negative_hp_line(_device_public,work,side,
                    acts & (hp!=_device_self_effect_hp) & (effects.heal_half[move]
                        | effects.weather_heal[move] | effects.rest[move] | creates_sub),
                    source_class=device_public.HP_ACTION)
                device_public.observe(_device_public,work,acts,rest_side=side,rest_mask=rest)
                device_public.feature(_device_public,side,work.active[rows,side],'rest_sleep',jnp.ones_like(move),rest)
            if _device_public is not None:device_public.reveal_ability(_device_public,work,side,_rest_immune)
            if _lifecycle_trace:
                _public_trace,_rest_capture=self_move_callbacks.finish(_public_trace,_rest_pending,work)
                _public_trace,_self_heal_capture=self_heal_callbacks.finish(_public_trace,_self_heal_pending,work,heal_weather,acts&hit&~immune)
                _public_trace=kernel_risks.self_heal_on_tape(_public_trace,
                    acts & (effects.heal_half[move] | effects.weather_heal[move] | effects.rest[move]),active=live,captured=_self_heal_capture,captured_self_move=_rest_capture)
            if _substitute_context is not None:
                _substitute_write=substitute_callbacks.Write(_substitute_old_hp,_active(work.hp,work,side),
                    max_hp,_active(work.status,work,side),sub_hp,_active(work.substitute_hp,work,side))
                _public_trace=substitute_callbacks.record_device(_public_trace,side,work.active[rows,side],
                    move,_substitute_write,_substitute_context,eligible=_substitute_requested,caller_move=caller_move)
                if _substitute_registered:
                    _public_trace=execution_accounting.capture_substitute_write(_public_trace,_substitute_ticket,
                        side,work.active[rows,side],move,_substitute_write,requested=_substitute_requested)
            if _lifecycle_trace:
                _public_trace=kernel_risks.substitute_on_tape(_public_trace,acts & effects.substitute[move],active=live,
                    captured=(kernel_risks.CapturedSubstitute(_substitute_context,_substitute_ticket,
                        side,work.active[rows,side],move,_substitute_write)
                        if _substitute_context is not None and _substitute_registered else None))
            # Pain Split assigns HP rather than dealing ordinary damage or
            # healing through callbacks. Read the CURRENT opposing active
            # after switches/earlier moves; floor the shared average once,
            # then clamp independently to each mon's own maximum HP. Ghost
            # typing does not block it, but Protect and target Substitute do.
            target_hp=_active(work.hp,work,foe)
            split=(acts & hit & effects.pain_split[move] & (hp>0) & (target_hp>0)
                & (_active(work.substitute_hp,work,foe)==0))
            if _lifecycle_trace:
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.PAIN_SPLIT,
                    acts & hit & effects.pain_split[move],active=live)
            average=jnp.maximum(1,(hp+target_hp)//2)
            hp=jnp.where(split,jnp.minimum(max_hp,average),hp)
            target_hp=jnp.where(split,jnp.minimum(_active(work.max_hp,work,foe),average),target_hp)
            work=work._replace(hp=_set_active(work.hp,work,side,hp))
            work=work._replace(hp=_set_active(work.hp,work,foe,target_hp))
            if _device_public is not None:
                # Original Pain Split uses -sethp, excluded by damage totals.
                device_public.hp_line(_device_public,work,side,split,
                    source_class=device_public.HP_ACTION,record_damage=False)
                device_public.hp_line(_device_public,work,foe,split,
                    source_class=device_public.HP_ACTION,record_damage=False)
            boosts = _active(work.boosts,work,side)
            # Gen3 Belly Drum reads live HP after earlier moves, fails at
            # half HP or +6 Attack, and pays a floored direct HP cost. The
            # +12 boost reaches +6 even from -6; other stages are untouched.
            drum=(acts & effects.belly_drum[move] & (max_hp>1)
                & (hp*2>max_hp) & (boosts[:,0]<6))
            if _lifecycle_trace:
                from . import engine_belly_drum_callbacks as belly_callbacks
                _belly_pending=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.belly_drum is not None:
                    _public_trace,_belly_pending=belly_callbacks.begin(_public_trace,work,side,move,hp,boosts,
                        _direct_hit_context,effects,eligible=caller_acts & (caller_move==move))
            hp=hp-jnp.where(drum,max_hp//2,0)
            boosts=boosts.at[:,0].set(jnp.where(drum,6,boosts[:,0]))
            work=work._replace(hp=_set_active(work.hp,work,side,hp))
            if _calm_mind_context is not None:
                _calm_mind_old_spa,_calm_mind_old_spd=boosts[:,2],boosts[:,3]
                _calm_mind_requested=caller_acts & (caller_move==_calm_mind_context.data.move_id)
                _public_trace,_calm_mind_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.CALM_MIND,side,active=_calm_mind_requested)
            if _swords_dance_context is not None:
                _swords_dance_old_atk=boosts[:,0]
                _swords_dance_requested=caller_acts & (caller_move==_swords_dance_context.data.move_id)
                _public_trace,_swords_dance_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.SWORDS_DANCE,side,active=_swords_dance_requested)
            if _lifecycle_trace:
                from . import engine_bulk_up_callbacks as bulk_up_callbacks
                _bulk_up_pending=None;_bulk_up_capture=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.bulk_up is not None:
                    _public_trace,_bulk_up_pending=bulk_up_callbacks.begin(_public_trace,work,side,move,boosts,
                        _direct_hit_context,effects,eligible=caller_acts)
            if _lifecycle_trace:
                from . import engine_self_boost_callbacks as self_boost_callbacks
                _public_trace,_self_boost_pending=self_boost_callbacks.begin(_public_trace,work,side,move,caller_move,boosts,
                    _direct_hit_context if _direct_hit_registered else None,effects,eligible=caller_acts)
            # Curse's table row is the non-Ghost form (gen4 onModifyMove, inherited by Gen 3).
            # A Ghost-type user would take the unported Ghost form: keep its stages (placeholder).
            _curse_ghost=(acts & effects.curse[move]
                & jnp.any(_active(work.types,work,side)==effects.ghost_type,axis=1))
            _curse_kept=boosts
            boosts = jnp.clip(boosts + jnp.where(acts[:,None],effects.self_boosts[move],0),-6,6)
            boosts = jnp.where(_curse_ghost[:,None],_curse_kept,boosts)
            if _lifecycle_trace:
                _public_trace,_self_boost_capture=self_boost_callbacks.finish(_public_trace,_self_boost_pending,boosts,acts&hit&~immune)
            if _lifecycle_trace:
                _public_trace,_bulk_up_capture=bulk_up_callbacks.finish(_public_trace,_bulk_up_pending,boosts,acts&hit&~immune)
            if _calm_mind_context is not None:
                _calm_mind_write=calm_mind_callbacks.Write(_calm_mind_old_spa,boosts[:,2],
                    _calm_mind_old_spd,boosts[:,3],acts & hit & ~immune)
                _public_trace=calm_mind_callbacks.record_device(_public_trace,_calm_mind_ticket,side,
                    work.active[rows,side],move,_calm_mind_write,_calm_mind_context,eligible=_calm_mind_requested)
            if _swords_dance_context is not None:
                _swords_dance_write=swords_dance_callbacks.Write(_swords_dance_old_atk,boosts[:,0],acts & hit & ~immune)
                _public_trace=swords_dance_callbacks.record_device(_public_trace,_swords_dance_ticket,side,
                    work.active[rows,side],move,_swords_dance_write,_swords_dance_context,eligible=_swords_dance_requested)
            if _lifecycle_trace:
                _public_trace=kernel_risks.self_boost_on_tape(_public_trace,
                    acts & jnp.any(effects.self_boosts[move]!=0,axis=1),active=live,
                    captured=(kernel_risks.CapturedCalmMind(_calm_mind_context,_calm_mind_ticket,
                        side,work.active[rows,side],move,_calm_mind_write)
                        if _calm_mind_context is not None else None),
                    captured_bulk_up=_bulk_up_capture,captured_self_boost=_self_boost_capture,
                    captured_swords=(kernel_risks.CapturedSwordsDance(_swords_dance_context,_swords_dance_ticket,
                        side,work.active[rows,side],move,_swords_dance_write)
                        if _swords_dance_context is not None else None))
            # These damaging moves lower their own SpA on a successful hit,
            # including Substitute/KO, but not miss, Protect, immunity or
            # absorption. Unlike target secondaries this is not Shield Dust
            # or Clear Body gated. White Herb handling remains a separate gap.
            drop = acts & hit & ~immune & effects.hit_spa_drop[move]
            if _lifecycle_trace:
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.HIT_SPA_DROP,
                    drop,active=live)
            boosts = boosts.at[:,2].set(jnp.where(drop,jnp.maximum(-6,boosts[:,2]-2),boosts[:,2]))
            # Superpower's self boosts: selfDrops runs after the hit (sim/battle-actions.ts:1117,
            # 1344-1350), on a Substitute hit and a KO too; same gate as the SpA drop. -6 is silent.
            boosts = jnp.clip(boosts + jnp.where((acts & hit & ~immune)[:,None],effects.hit_self_boosts[move],0),-6,6)
            work = work._replace(boosts=_set_active(work.boosts,work,side,boosts))
            # Fidelity M1: Hyper Beam's self volatileStatus mustrecharge (data/moves.ts hyperbeam) comes from the
            # same selfDrops: on a hit, a Substitute hit and a KO, never on a miss, Protect or an immunity.
            # Duration 2; -mustrecharge is published by public_move_state (0 -> 2).
            _recharge_set=acts & hit & ~immune & effects.recharge[move]
            work=work._replace(recharge_counter=_set_active(work.recharge_counter,work,side,
                jnp.where(_recharge_set,2,_active(work.recharge_counter,work,side))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_recharge_set)
            if _lifecycle_trace:
                _public_trace,_belly_capture=belly_callbacks.finish(_public_trace,_belly_pending,work,acts&hit&~immune)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.BELLY_DRUM,
                    acts & effects.belly_drum[move],active=live,captured_belly_drum=_belly_capture)
            if _device_public is not None:_device_party_before=work
            if _lifecycle_trace:
                from . import engine_heal_bell_callbacks as heal_bell_callbacks
                _public_trace,_heal_bell_pending=heal_bell_callbacks.begin(_public_trace,work,side,move,
                    _direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,table,
                    eligible=acts,primary=caller_acts&(caller_move==move),secondary=acts&hit&~immune)
                team_cured=(acts[:,None] & effects.heal_bell[move,None] & (work.hp[rows,side]>0)
                    & (work.status[rows,side]!=m.NONE) & ~effects.soundproof[work.ability[rows,side]])
                _public_trace=status_callbacks.unknown(_public_trace,jnp.any(team_cured,axis=1)&~heal_bell_callbacks.prospective(_heal_bell_pending))
                work,_heal_bell_mask=cure_team(work,side,effects,eligible=acts & effects.heal_bell[move],_return_write_mask=True)
                _public_trace,_heal_bell_capture=heal_bell_callbacks.finish(_public_trace,_heal_bell_pending,work,_heal_bell_mask)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.HEAL_BELL,
                    acts & effects.heal_bell[move],active=live,captured_heal_bell=_heal_bell_capture)
            else:
                work=cure_team(work,side,effects,eligible=acts & effects.heal_bell[move])
            # Aromatherapy: -cureteam, every living statused member of the user's side,
            # Soundproof holders included (gen4 onHit inherited by Gen 3); never fails.
            from .effects import cure_party
            _aromatherapy=acts & effects.aromatherapy[move]
            if _lifecycle_trace:
                _public_trace=status_callbacks.unknown(_public_trace,_aromatherapy
                    & jnp.any((work.hp[rows,side]>0) & (work.status[rows,side]!=m.NONE),axis=1))
            work=cure_party(work,side,effects,eligible=_aromatherapy)
            if _device_public is not None:
                device_public.party_cure(_device_public,_device_party_before,work,side,_aromatherapy)
            if _device_public is not None:
                device_public.party_cure(_device_public,_device_party_before,work,side,acts & effects.heal_bell[move],effects)
                device_public.observe(_device_public,work,acts)
            if _lifecycle_trace:
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.HAZE,
                    acts & effects.haze[move],active=live)
            for target_side in range(2):
                old = _active(work.boosts,work,target_side)
                work = work._replace(boosts=_set_active(work.boosts,work,target_side,
                    jnp.where((acts & effects.haze[move])[:,None],0,old)))
            if table.items is not None:
                # Fidelity M1: Trick (data/moves.ts, Gen 3 = base). Accuracy 100 and Protect are in `hit`; a Status
                # move ignores type immunity (Dark is hit). Sticky Hold is its onTryImmunity (-immune, no ability
                # line); a Substitute fails it (TryPrimaryHit, no bypasssub). onHit: takeItem fails when either
                # mon has itemKnockedOff (sim/pokemon.ts, gen<=4) or neither holds an item; otherwise the items
                # swap, an identical pair included (unchanged state; the device still sees the -item lines).
                # The Choice lock follows the current item: the lock site below and the end-of-turn unlock.
                _trick=(acts & hit & table.items.trick[move] & (_active(work.hp,work,side)>0) & (_active(work.hp,work,foe)>0)
                    & ((_active(work.substitute_hp,work,foe)==0) | effects.bypass_sub[move])
                    & ~table.items.sticky_hold[_active(work.ability,work,foe)]
                    & ~_active(work.item_knocked,work,side) & ~_active(work.item_knocked,work,foe)
                    & ((_active(work.item,work,side)>0) | (_active(work.item,work,foe)>0)))
                if _device_public is not None:_device_trick_before=work
                _trick_own=_active(work.item,work,side);_trick_foe=_active(work.item,work,foe)
                work=work._replace(item=_set_active(work.item,work,side,jnp.where(_trick,_trick_foe,_trick_own)))
                work=work._replace(item=_set_active(work.item,work,foe,jnp.where(_trick,_trick_own,_trick_foe)))
                if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_trick)
                if _device_public is not None:
                    device_public.trick(_device_public,_device_trick_before,work,side,_trick,table.items.choice_band)
            # Charm's primary boosts (moveHit after TryHit): Protect is already in `hit`; a
            # Substitute fails it (TryPrimaryHit); status moves ignore type immunity
            # (tryMoveHit sets ignoreImmunity for Status). The drop is capped first, then
            # TryBoost: Clear Body/White Smoke delete every negative stage, Hyper Cutter
            # Attack's, printing -fail [from] ability only for a nonzero capped drop.
            _drop=effects.target_boosts[move]
            _drop_old=_active(work.boosts,work,foe)
            _drop_blocked=effects.drop_block[_active(work.ability,work,foe)] & (_drop<0)
            _drop_lands=(acts & hit & jnp.any(_drop!=0,axis=1) & (_active(work.hp,work,foe)>0)
                & ((_active(work.substitute_hp,work,foe)==0) | effects.bypass_sub[move]))
            work=work._replace(boosts=_set_active(work.boosts,work,foe,jnp.where(_drop_lands[:,None],
                jnp.clip(_drop_old+jnp.where(_drop_blocked,0,_drop),-6,6),_drop_old)))
            if _device_public is not None:
                device_public.reveal_ability(_device_public,work,foe,_drop_lands
                    & jnp.any(_drop_blocked & (jnp.clip(_drop_old+_drop,-6,6)!=_drop_old),axis=1))
            # Destiny Bond: volatileStatus destinybond with -singlemove (accuracy true, targets the
            # user, bypasses a Substitute; Gen 3 reuse never fails).
            _bond=acts & effects.destinybond[move]
            work=work._replace(volatile=_set_active(work.volatile,work,side,jnp.where(_bond,
                _active(work.volatile,work,side)|jnp.uint32(DESTINY_BOND),_active(work.volatile,work,side))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_bond)
            if _device_public is not None:device_public.destiny_bond(_device_public,work,side,_bond)
            # Fidelity M2: Perish Song (data/moves.ts:13241-13270 onHitField; data/mods/gen4/moves.ts:993-999 order 12).
            # Accuracy true and target 'all': no accuracy roll, and neither Protect nor a Substitute blocks it. For each
            # active: Soundproof's onTryHit returns null (-immune [from] ability: Soundproof; data/mods/gen7/abilities.ts:
            # 98-106 has no target !== source check, so a Soundproof user is blocked too); otherwise one without the volatile
            # gets duration 4 (-start|X|perish3|[silent]). Nobody getting it is a -fail (no state).
            _perish_used=acts & effects.perishsong[move]
            for _perish_side in range(2):
                _perish_alive=_perish_used & (_active(work.hp,work,_perish_side)>0)
                _perish_sound=_perish_alive & effects.soundproof[_active(work.ability,work,_perish_side)]
                _perish_gets=_perish_alive & ~_perish_sound & (_active(work.perish_counter,work,_perish_side)==0)
                work=work._replace(perish_counter=_set_active(work.perish_counter,work,_perish_side,
                    jnp.where(_perish_gets,4,_active(work.perish_counter,work,_perish_side))))
                if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_perish_alive)
                if _device_public is not None:device_public.reveal_ability(_device_public,work,_perish_side,_perish_sound)
            # Fidelity M2: Yawn (data/moves.ts:21141-21172; data/mods/gen4/moves.ts:1527-1533 order 10, subOrder 19).
            # Accuracy true; Protect is in `hit`; a Status move ignores type immunity. Its onTryHit fails a statused target,
            # a Substitute fails it (TryPrimaryHit), addVolatile fails on an existing yawn (no onRestart), then Insomnia
            # and Vital Spirit's onTryAddVolatile print -immune [from] ability. Otherwise duration 2 (-start|T|move: Yawn).
            _yawn_try=(acts & hit & effects.yawn[move] & (_active(work.hp,work,foe)>0)
                & (_active(work.status,work,foe)==m.NONE)
                & ((_active(work.substitute_hp,work,foe)==0) | effects.bypass_sub[move])
                & (_active(work.yawn_counter,work,foe)==0))
            _yawn_immune=jnp.zeros_like(_yawn_try)
            if table.secondary is not None:
                _yawn_immune=_yawn_try & table.secondary.ability_immunity[_active(work.ability,work,foe),m.SLP]
            work=work._replace(yawn_counter=_set_active(work.yawn_counter,work,foe,
                jnp.where(_yawn_try & ~_yawn_immune,2,_active(work.yawn_counter,work,foe))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_yawn_try)
            if _device_public is not None:device_public.reveal_ability(_device_public,work,foe,_yawn_immune)
            status = effects.status[move]
            target_types = _active(work.types,work,foe)
            status_immune = jnp.any(effects.status_immunity[status[:,None],target_types],axis=1)
            if _thunder_wave_context is not None:
                _thunder_wave_type_immune=status_immune
                _thunder_wave_ability_immune=jnp.zeros_like(acts)
                if table.secondary is not None:
                    _thunder_wave_ability_immune=table.secondary.ability_immunity[_active(work.ability,work,foe),status]
            if _sleep_powder_context is not None:
                _sleep_powder_type_immune=status_immune
                _sleep_powder_ability_immune=jnp.zeros_like(acts)
                if table.secondary is not None:
                    _sleep_powder_ability_immune=table.secondary.ability_immunity[_active(work.ability,work,foe),status]
            if _toxic_context is not None:
                _toxic_type_immune=status_immune
                _toxic_ability_immune=jnp.zeros_like(acts)
                if table.secondary is not None:
                    _toxic_ability_immune=table.secondary.ability_immunity[_active(work.ability,work,foe),status]
            if table.secondary is not None:
                status_immune |= ~secondaries.status_allowed(table.secondary,status,target_types,
                    _active(work.ability,work,foe))
            status_immune |= effects.thunderwave[move] & jnp.any(target_types == effects.ground_type,axis=1)
            # Sleep Clause Mod (data/rulesets.ts:1386-1401): onSetStatus refuses a foe's
            # sleep while another living member of the target's side sleeps from a foe's
            # move; Rest's sleep does not count. Checked after accuracy, Protect, the
            # Substitute and an existing status; the duration roll is then never drawn.
            from .effects import sleep_clause_blocked
            _sleep_clause=(status==m.SLP) & sleep_clause_blocked(work,foe)
            apply_status = (acts & hit & (status != m.NONE) & ~status_immune
                & (_active(work.hp,work,foe) > 0) & (_active(work.status,work,foe) == m.NONE)
                & ((_active(work.substitute_hp,work,foe) == 0) | effects.bypass_sub[move]))
            apply_status &= ~_sleep_clause
            if _toxic_context is not None:
                _toxic_requested=acts & (move==_toxic_context.data.move_id)
                if _lifecycle_trace:
                    _toxic_requested &= ~_protect_block_mask
                _toxic_old_status=_active(work.status,work,foe)
                _toxic_old_counter=_active(work.status_counter,work,foe)
                _toxic_live=_active(work.hp,work,foe)>0
                _toxic_sub_block=(_active(work.substitute_hp,work,foe)>0)&~effects.bypass_sub[move]
                if _toxic_registered:
                    _public_trace,_toxic_ticket=execution_accounting.open_attempt(_public_trace,
                        execution_accounting.Site.TOXIC_ATTEMPT,foe,active=_toxic_requested)
            if _thunder_wave_context is not None:
                _thunder_wave_requested=acts & (move==_thunder_wave_context.data.move_id)
                if _lifecycle_trace:
                    _thunder_wave_requested &= ~_protect_block_mask
                _thunder_wave_old_status=_active(work.status,work,foe)
                _thunder_wave_old_counter=_active(work.status_counter,work,foe)
                _thunder_wave_live=_active(work.hp,work,foe)>0
                _thunder_wave_sub_block=(_active(work.substitute_hp,work,foe)>0)&~effects.bypass_sub[move]
                _public_trace,_thunder_wave_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.PRIMARY_PARALYSIS,foe,active=_thunder_wave_requested)
                _thunder_wave_ground_pending=thunder_callbacks.begin_ground(_public_trace,work,_thunder_wave_ticket,
                    side,move,caller_move,chosen_slot,_thunder_wave_context,requested=_thunder_wave_requested,
                    resolved_type=typ,power=power,accuracy=_thunder_wave_accuracy_hit,hit=hit,immune=immune,
                    numerator=num,denominator=den,protected=_risk_protected,absorbed=_risk_absorbed)
            if _sleep_powder_context is not None:
                _sleep_powder_requested=acts & powder_callbacks.selected_device(_sleep_powder_context.data,move)
                if _lifecycle_trace:
                    _sleep_powder_requested &= ~_protect_block_mask
                _sleep_powder_old_status=_active(work.status,work,foe)
                _sleep_powder_old_counter=_active(work.status_counter,work,foe)
                _sleep_powder_live=_active(work.hp,work,foe)>0
                _sleep_powder_sub_block=(_active(work.substitute_hp,work,foe)>0)&~effects.bypass_sub[move]
                _public_trace,_sleep_powder_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.PRIMARY_SLEEP,foe,active=_sleep_powder_requested)
            if _lifecycle_trace:
                from . import engine_primary_plain_status_callbacks as plain_primary_callbacks
                _primary_plain_pending=None;_primary_plain_capture=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.primary_plain_status is not None:
                    _public_trace,_primary_plain_pending=plain_primary_callbacks.begin(_public_trace,work,side,move,
                        _direct_hit_context,table,eligible=acts&~_protect_block_mask,caller_move=caller_move)
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
            if _device_public is not None:
                device_public.observe(_device_public,work,acts)
                if table.secondary is not None:
                    device_public.reveal_ability(_device_public,work,foe,
                        acts & hit & (status!=m.NONE) & (_active(work.hp,work,foe)>0)
                        & (_active(work.status,work,foe)==m.NONE)
                        & ((_active(work.substitute_hp,work,foe)==0)|effects.bypass_sub[move])
                        & table.secondary.ability_immunity[_active(work.ability,work,foe),status])
            if _lifecycle_trace and _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.primary_plain_status is not None:
                _plain_type_immune=jnp.any(effects.status_immunity[status[:,None],target_types],axis=1)
                _plain_ability_immune=table.secondary.ability_immunity[_active(work.ability,work,foe),status]
                _plain_attempt=plain_primary_callbacks.Attempt(_direct_accuracy_hit,hit,_plain_type_immune,_plain_ability_immune,
                    (_active(work.substitute_hp,work,foe)>0)&~effects.bypass_sub[move],_active(work.hp,work,foe)>0,
                    apply_status,_risk_protected,_risk_absorbed,immune,acts&hit&~immune,
                    _primary_plain_pending.old_status,_active(work.status,work,foe),_primary_plain_pending.old_counter,
                    _active(work.status_counter,work,foe))
                _public_trace,_primary_plain_capture=plain_primary_callbacks.finish(_public_trace,_primary_plain_pending,_plain_attempt,state=work)
            if _toxic_context is not None:
                _toxic_attempt=toxic_callbacks.Attempt(_toxic_accuracy_hit,hit,_toxic_type_immune,
                    _toxic_ability_immune,_toxic_sub_block,_toxic_live,apply_status,_risk_protected,_risk_absorbed,
                    immune,acts & hit & ~immune,_toxic_old_status,_active(work.status,work,foe),
                    _toxic_old_counter,_active(work.status_counter,work,foe))
                _public_trace=toxic_callbacks.record_device(_public_trace,side,work.active[rows,side],
                    work.active[rows,foe],move,_toxic_attempt,_toxic_context,eligible=_toxic_requested,caller_move=caller_move)
                if _toxic_registered:
                    _public_trace=execution_accounting.capture_toxic_attempt(_public_trace,_toxic_ticket,
                        side,work.active[rows,side],work.active[rows,foe],move,_toxic_attempt,requested=_toxic_requested)
            if _thunder_wave_context is not None:
                _thunder_wave_attempt=thunder_callbacks.Attempt(_thunder_wave_accuracy_hit,hit,_thunder_wave_type_immune,
                    _thunder_wave_ability_immune,_thunder_wave_sub_block,_thunder_wave_live,apply_status,_risk_protected,_risk_absorbed,
                    immune,acts & hit & ~immune,_thunder_wave_old_status,_active(work.status,work,foe),
                    _thunder_wave_old_counter,_active(work.status_counter,work,foe))
                _public_trace=thunder_callbacks.record_device(_public_trace,_thunder_wave_ticket,side,work.active[rows,side],
                    work.active[rows,foe],move,_thunder_wave_attempt,_thunder_wave_context,requested=_thunder_wave_requested,
                    ground_pending=_thunder_wave_ground_pending,state=work)
                _public_trace=thunder_callbacks.exclude_cure_device(_public_trace,_thunder_wave_requested,apply_status,
                    _active(work.item,work,foe),table.items)
            if _sleep_powder_context is not None:
                _sleep_powder_attempt=powder_callbacks.Attempt(_sleep_powder_accuracy_hit,hit,_sleep_powder_type_immune,
                    _sleep_powder_ability_immune,_sleep_powder_sub_block,_sleep_powder_live,apply_status,_risk_protected,_risk_absorbed,
                    immune,acts & hit & ~immune,_sleep_powder_old_status,_active(work.status,work,foe),
                    _sleep_powder_old_counter,_active(work.status_counter,work,foe))
                _public_trace=powder_callbacks.record_device(_public_trace,_sleep_powder_ticket,side,work.active[rows,side],
                    work.active[rows,foe],move,_sleep_powder_attempt,_sleep_powder_context,requested=_sleep_powder_requested)
                _public_trace=powder_callbacks.exclude_cure_device(_public_trace,_sleep_powder_requested,apply_status,
                    _active(work.item,work,foe),table.items)
            if _lifecycle_trace:
                _public_trace=kernel_risks.primary_status_on_tape(_public_trace,acts & (status!=m.NONE),active=live,
                    captured=(kernel_risks.CapturedToxic(_toxic_context,_toxic_ticket,side,
                        work.active[rows,side],work.active[rows,foe],move,_toxic_attempt)
                        if _toxic_context is not None and _toxic_registered else None),
                    captured_thunder_wave=(kernel_risks.CapturedThunderWave(_thunder_wave_context,_thunder_wave_ticket,side,
                        work.active[rows,side],work.active[rows,foe],move,_thunder_wave_attempt)
                        if _thunder_wave_context is not None else None),
                    captured_protect_block=_protect_block_capture,captured_primary_plain=_primary_plain_capture,
                    captured_sleep_powder=(kernel_risks.CapturedSleepPowder(_sleep_powder_context,_sleep_powder_ticket,side,
                        work.active[rows,side],work.active[rows,foe],move,_sleep_powder_attempt)
                        if _sleep_powder_context is not None else None))
            if table.secondary is not None:
                if _lifecycle_trace:
                    work,_public_trace=secondaries.after_status_event(work,side,foe,status,table.secondary,
                        applied=apply_status,item_table=table.items,_status_trace=_public_trace,_device_public=_device_public)
                else:
                    work=secondaries.after_status_event(work,side,foe,status,table.secondary,
                        applied=apply_status,item_table=table.items,_device_public=_device_public)
        if table.sustain is not None:
            if _lifecycle_trace:
                from . import engine_leech_seed_callbacks as leech_callbacks
                _seed_pending=None;_seed_capture=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.leech_seed is not None:
                    _public_trace,_seed_pending=leech_callbacks.begin_seed(_public_trace,work,side,move,
                        _direct_hit_context,table.sustain,eligible=acts & hit & (caller_move==move),secondary=acts & hit & ~immune)
                _public_trace,_wish_pending=self_move_callbacks.begin(_public_trace,work,side,caller_move,_self_move_context,code=self_move_callbacks.WISH_START,eligible=acts&caller_acts&(caller_move==move),secondary=acts&hit&~immune)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.SPIN_CLEAR,
                    acts & hit & ~immune & table.sustain.spin[move],active=live)
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
            if _device_public is not None:device_public.wish(_device_public,side,starts_wish,start=True)
            if _lifecycle_trace:
                _public_trace,_wish_capture=self_move_callbacks.finish(_public_trace,_wish_pending,work)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.WISH_START,acts & table.sustain.wish[move],active=live,captured_self_move=_wish_capture)
                _public_trace,_seed_capture=leech_callbacks.finish_seed(_public_trace,_seed_pending,work)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.SEED_START,
                    acts & hit & table.sustain.seed[move],active=live,captured_leech=_seed_capture)
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
            if _lifecycle_trace:
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.LINK_TRAP,
                    acts & hit & table.linked_traps.move[move],active=live)
            work=linking.apply(work,side,move,table.linked_traps,eligible=acts & hit)
        if table.effects is not None:
            # Fidelity M2: Wrap's volatileStatus partiallytrapped (data/moves.ts:21091-21104, accuracy 85 from
            # data/mods/gen4/moves.ts:1515-1518), applied by runMoveEffects after the hit: never on a miss, Protect, an
            # immunity or a Substitute hit, and addVolatile fails on a fainted target and on one already trapped (no
            # onRestart: no new duration roll). Duration random(3,7) (data/mods/gen4/conditions.ts:110-118) = 3+draws[s][7],
            # the attacker's column: Wrap never also uses primary sleep, Encore or Signal Beam. -activate|T|move: Wrap.
            _wrap_set=(acts & hit & ~immune & ~hits_sub & table.effects.wrap[move]
                & (_active(work.hp,work,foe)>0) & (_active(work.partial_trap_counter,work,foe)==0))
            work=work._replace(partial_trap_counter=_set_active(work.partial_trap_counter,work,foe,
                jnp.where(_wrap_set,3+chance[:,7],_active(work.partial_trap_counter,work,foe))),
                partial_trap_source=_set_active(work.partial_trap_source,work,foe,
                    jnp.where(_wrap_set,work.active[rows,side],_active(work.partial_trap_source,work,foe))))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_wrap_set)
        if table.encore is not None:
            # Duration uses the existing uniform [0,4) status-duration column:
            # a dispatched Encore cannot also inflict a primary sleep status.
            queued=(order==0) & ~state.pending_residual & ~switches[rows,foe] & ~displaced[rows,foe]
            if _lifecycle_trace:
                from . import engine_encore_callbacks as encore_callbacks
                _encore_pending=None;_encore_capture=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.encore is not None:
                    _public_trace,_encore_pending=encore_callbacks.begin(_public_trace,work,foe,move,_direct_hit_context,
                        table.encore,eligible=acts & hit & (caller_move==move),secondary=acts & hit & ~immune,roll=chance[:,7],queued=queued)
            work=repeating.apply(work,foe,move,chance[:,7],table.encore,eligible=acts & hit,queued=queued)
            if _lifecycle_trace:
                _public_trace,_encore_capture=encore_callbacks.finish(_public_trace,_encore_pending,work)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.ENCORE,
                    acts & hit & table.encore.move[move],active=live,captured_encore=_encore_capture)
        if table.secondary is not None:
            effective_weather=work.weather
            if table.weather is not None:
                effective_weather=climate.effective(table.weather,work.weather,
                    work.ability[rows[:,None],sides,work.active],work.hp[rows[:,None],sides,work.active])
            if _lifecycle_trace:
                work,_public_trace=secondaries.apply(work,side,move,secondary_draws[rows,side],table.secondary,
                    eligible=acts & hit & ~immune,effective_weather=effective_weather,
                    substitute_hit=hits_sub,item_table=table.items,_status_trace=_public_trace,
                    _secondary_status_context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,_device_public=_device_public)
            else:
                work=secondaries.apply(work,side,move,secondary_draws[rows,side],table.secondary,
                    eligible=acts & hit & ~immune,effective_weather=effective_weather,
                    substitute_hit=hits_sub,item_table=table.items,_device_public=_device_public)
        if table.secondary is not None:
            # Fidelity patch B: Iron Tail's Def drop, Steel Wing's and Meteor Mash's self boost and Signal Beam's
            # confusion, on the same roll (one secondary entry per move); confusion lasts 2+draws[s][7].
            work,_chance_secondary=secondaries.apply_chance_effects(work,side,move,secondary_draws[rows,side],chance[:,7],
                table.secondary,eligible=acts & hit & ~immune,substitute_hit=hits_sub,_device_public=_device_public)
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_chance_secondary)
        if table.items is not None:
            # Knock Off (gen4 moves.ts onAfterHit, inherited by Gen 3): AfterHit runs after the
            # secondaries and, in Gen 3, before DamagingHit; only for a damaged non-Substitute target,
            # with the attacker still up, also when the hit KOs. TakeItem: Sticky Hold blocks with
            # -activate (gen4 abilities.ts, no HP check). The item is gone for good (itemKnockedOff).
            _knock=(acts & hit & ~immune & ~hits_sub & table.items.knock_off[move]
                & (_active(work.item,work,foe)>0) & (_active(work.hp,work,side)>0))
            _sticky=_knock & table.items.sticky_hold[_active(work.ability,work,foe)]
            _knocked=_knock & ~_sticky
            if _device_public is not None:_device_knock_before=work
            work=work._replace(item=_set_active(work.item,work,foe,jnp.where(_knocked,0,_active(work.item,work,foe))),
                item_knocked=_set_active(work.item_knocked,work,foe,_active(work.item_knocked,work,foe) | _knocked))
            if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_knock)
            if _device_public is not None:
                device_public.reveal_ability(_device_public,work,foe,_sticky)
                device_public.knock_off(_device_public,_device_knock_before,work,foe,_knocked)
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
                    eligible=acts & hit & (dealt>0) & ~hits_sub,item_table=table.items,_status_trace=_public_trace,
                    _rough_skin_context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,
                    _rough_skin_capture=(kernel_risks.CapturedDirectHit(_direct_hit_context,_direct_ticket,side,
                        work.active[rows,side],work.active[rows,foe],move,_direct_write)
                        if _direct_hit_context is not None and _direct_hit_registered else None),_device_public=_device_public)
            else:
                work = secondaries.apply_contact(work,side,move,phazing_draws[rows,side],table.secondary,
                    eligible=acts & hit & (dealt>0) & ~hits_sub,item_table=table.items,_device_public=_device_public)
        if table.secondary is not None:
            if _lifecycle_trace:
                work,_public_trace=secondaries.apply_color_change(work,side,typ,table.secondary,
                    eligible=acts & hit & (dealt>0) & ~hits_sub,_status_trace=_public_trace,
                    _direct_hit_context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None,move=move,_device_public=_device_public)
            else:
                work=secondaries.apply_color_change(work,side,typ,table.secondary,
                    eligible=acts & hit & (dealt>0) & ~hits_sub,_device_public=_device_public)
        if _lifecycle_trace:work,_public_trace=_lum_update(work,table,live & move_context,_public_trace,_device_public=_device_public)
        else:work = _lum_update(work,table,live & move_context,_device_public=_device_public)
        # Gen3 applies contact DamagingHit (and immediate Lum cure) before
        # recoil. A source about to faint can still consume its berry first.
        if _recoil_context is not None:
            _recoil_before=recoil_callbacks.Before(work.active[rows,side],work.active[rows,foe],
                _active(work.ability,work,side),_active(work.hp,work,side),_active(work.max_hp,work,side),_active(work.status,work,side))
            _public_trace=kernel_risks.recoil_on_tape(_public_trace,recoil,active=live,
                pending=_recoil_pending,before=_recoil_before)
        elif _lifecycle_trace:
            _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.RECOIL,
                recoil>0,active=live)
        hp = jnp.maximum(0,_active(work.hp,work,side)-recoil)
        work = work._replace(hp=_set_active(work.hp,work,side,hp))
        if _device_public is not None:
            device_public.hp_line(_device_public,work,side,recoil>0,
                source_class=device_public.HP_ACTION)
        if _recoil_context is not None:
            _public_trace=recoil_callbacks.record_device(_public_trace,_recoil_pending,recoil_callbacks.Write(
                _recoil_before.actor_physical,_recoil_before.target_physical,_recoil_before.ability,
                _recoil_before.old_hp,_active(work.hp,work,side),_recoil_before.maximum,_recoil_before.status))
        # Destiny Bond onFaint: the bonded foe fainted from this actor's move (every foe HP loss in
        # the actor's action is its move's), so the attacker faints after it at faintMessages.
        # Residual, recoil, confusion or Liquid Ooze faints are not Move effects and never trigger.
        _bond_trigger=(acts & (_bond_foe_hp>0) & (_active(work.hp,work,foe)==0)
            & ((_active(work.volatile,work,foe)&jnp.uint32(DESTINY_BOND))!=0) & (_active(work.hp,work,side)>0))
        work=work._replace(hp=_set_active(work.hp,work,side,jnp.where(_bond_trigger,0,_active(work.hp,work,side))))
        hp=jnp.where(_bond_trigger,0,hp)
        if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_bond_trigger)
        if _device_public is not None:device_public.observe(_device_public,work,_bond_trigger,record_damage=False)
        if _device_public is not None:_device_weather_before=work
        if table.weather is not None:
            if _lifecycle_trace:
                weather,counter,_public_trace = climate.set_move(work.weather,work.weather_counter,
                    table.weather.move_weather[move],eligible=acts,_weather_trace=_public_trace,
                    _actor_side=side,_actor_physical=work.active[rows,side],_move_id=move)
            else:
                weather,counter = climate.set_move(work.weather,work.weather_counter,
                    table.weather.move_weather[move],eligible=acts)
            work = work._replace(weather=weather,weather_counter=counter)
            if _device_public is not None:device_public.weather(_device_public,_device_weather_before,work,acts,side=side)
        if table.entry is not None:
            old_layers = work.spikes[rows,foe]
            if _spikes_context is not None:
                _spikes_requested=acts & table.entry.spikes_move[move]
                _public_trace,_spikes_ticket=execution_accounting.open_attempt(_public_trace,
                    execution_accounting.Site.SPIKES_WRITE,side,active=_spikes_requested)
            work = work._replace(spikes=work.spikes.at[rows,foe].set(jnp.where(
                acts & table.entry.spikes_move[move],jnp.minimum(3,old_layers+1),old_layers)))
            if _spikes_context is not None:
                _spikes_write=spikes_callbacks.Write(foe,old_layers,
                    work.spikes[rows,foe],acts & hit & ~immune)
                _public_trace=spikes_callbacks.record_device(_public_trace,_spikes_ticket,side,
                    work.active[rows,side],move,_spikes_write,_spikes_context,requested=_spikes_requested)
            if _lifecycle_trace:
                _public_trace=kernel_risks.spikes_on_tape(_public_trace,
                    acts & table.entry.spikes_move[move],active=live,
                    captured=(kernel_risks.CapturedSpikes(_spikes_context,_spikes_ticket,side,
                        work.active[rows,side],move,_spikes_write) if _spikes_context is not None else None))
        moves_cancelled |= acts & ((foe_hp == 0) | (hp == 0))
        if _lifecycle_trace:
            _public_trace=self_destruct_callbacks.actor_faint(_public_trace,work)
        if _public_trace is not None:
            _public_trace = public_callbacks.faint(_public_trace,foe,work.active[rows,foe],
                eligible=acts & hit & ~hits_sub & (dealt>0) & (foe_hp==0))
        if _direct_hit_context is not None and _direct_hit_registered:
            _public_trace=direct_ko_callbacks.finish_device(_public_trace,_direct_ko_pending,
                _active(work.hp,work,side),work.hp[rows,foe],work.active[rows,side],work.active[rows,foe],
                recoil_after=(recoil_callbacks.Write(work.active[rows,side],work.active[rows,foe],_active(work.ability,work,side),
                    _recoil_before.old_hp,_active(work.hp,work,side),_active(work.max_hp,work,side),_active(work.status,work,side))if _recoil_context is not None else None))

        if _lifecycle_trace:
            _public_trace=self_destruct_callbacks.finish(_public_trace,work)

        if _device_public is not None:
            device_public.public_move_state(_device_public,_device_move_before,work,side,move,caller_acts)
            _device_drag_before=work
        if table.phazing is not None:
            _whirlwind_pending=None
            if _lifecycle_trace and _direct_hit_registered and _direct_hit_context.data.whirlwind is not None:
                from . import engine_whirlwind_callbacks as whirlwind_callbacks
                _public_trace,_whirlwind_pending=whirlwind_callbacks.begin(_public_trace,work,side,move,caller_move,chosen_slot,phazing_draws[rows,side],_direct_hit_context,table,
                    eligible=acts,hit=hit,immune=immune,protected=_risk_protected,absorbed=_risk_absorbed)
            if _whirlwind_pending is not None:
                work,changed,hazard_ko,_public_trace=dragging.drag(work,foe,move,phazing_draws[rows,side],
                    table.phazing,table.entry,eligible=acts & hit & (hp>0),_public_trace=_public_trace,_whirlwind_pending=_whirlwind_pending)
            else:
                work,changed,hazard_ko=dragging.drag(work,foe,move,phazing_draws[rows,side],
                    table.phazing,table.entry,eligible=acts & hit & (hp>0))
            if _device_public is not None:device_public.drag(_device_public,_device_drag_before,work,foe,table.entry,changed)
            displaced=displaced.at[rows,foe].set(displaced[rows,foe] | changed)
            moves_cancelled |= hazard_ko
            if _lifecycle_trace:work,_public_trace=_lum_update(work,table,changed,_public_trace,_device_public=_device_public)
            else:work=_lum_update(work,table,changed,_device_public=_device_public)
        if table.baton is not None:
            bench_alive = jnp.any(
                (work.hp[rows,side] > 0) & (jnp.arange(6)[None,:] != work.active[rows,side,None]),axis=1)
            passed = acts & table.baton.move[move] & (hp > 0) & bench_alive
            if _lifecycle_trace:
                from . import engine_baton_pass_callbacks as baton_callbacks
                _public_trace,_baton_start=baton_callbacks.begin_start(_public_trace,work,side,move,caller_move,
                    _direct_hit_context if _direct_hit_registered else None,acts=acts,eligible=caller_acts)
            work = work._replace(baton_pass=work.baton_pass.at[rows,side].set(
                work.baton_pass[rows,side] | passed))
            if _lifecycle_trace:
                _public_trace=baton_callbacks.finish_start(_public_trace,_baton_start,work,passed,acts&hit&~immune)
            # Only a first ordinary action can leave an opposing action queued.
            # A resumed Baton is already the final original action.
            save_foe = passed & (order == 0) & ~state.pending_residual & ~switches[rows,foe]
            queued_actor = jnp.where(save_foe,foe,queued_actor)
            paused |= passed
        # All move effects, including self-KO/recoil and forced-entry hazards,
        # finish before the original runAction faint queue is processed.
        work=_clear_fainted_boosts(work,_faint_before_hp,eligible=live & move_context)
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
    if _lifecycle_trace and _weather_hp_context is not None:
        _weather_hp_record=weather_speed_profile.retain_frame_device(_weather_hp_record,work,
            _public_trace.stream_id,_weather_hp_context[1],_public_trace)
    # Snapshot before Wish(order7), weather(order8), and order10 handlers.
    end_speed = _speeds(work,table)
    if _lifecycle_trace:_weather_pre_wish=work
    if _status_residual_context is not None:
        if _status_registered:
            _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                work,_public_trace.stream_id,_status_residual_context.weather_profile_data,hypothetical=_hypothetical,eligible=residual_ready)
            _status_cached_speed_ok=(jnp.all((end_speed>0)&(end_speed<=10000),axis=1)
                & (end_speed[:,0]!=end_speed[:,1])
                & jnp.all(_status_residual_context.data.engine_binding==_status_residual_context.weather_profile_data.engine_binding)
                & jnp.all(_status_residual_context.data.source_binding==_status_residual_context.weather_profile_data.source_binding))
            _status_queue=status_residual_callbacks.capture_queue_device(_public_trace,
                _status_residual_context.data,work.hp[rows[:,None],sides,work.active],
                work.status[rows[:,None],sides,work.active],end_speed,eligible=residual_ready)
    if _leftovers_context is not None:
        if _leftovers_registered:
            _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                work,_public_trace.stream_id,_leftovers_context.weather_profile_data,hypothetical=_hypothetical,eligible=residual_ready)
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
                from . import engine_self_move_callbacks as self_move_callbacks
                _wish_context=_direct_hit_context if _direct_hit_context is not None and _direct_hit_registered else None
                if _wish_context is not None and _wish_context.data.self_move is not None:
                    _public_trace,_wish_tick_pending=self_move_callbacks.begin(_public_trace,work,side,_wish_context.data.self_move.move_ids[2],_wish_context,code=self_move_callbacks.WISH_TICK,eligible=ordinary)
                else:
                    _wish_tick_pending=None
                    _public_trace=execution_accounting.unhandled_attempt(_public_trace,execution_accounting.Site.WISH,side,active=ordinary & (work.wish_counter[:,side]>0))
            if _device_public is not None:_device_wish_before=work
            wish = recovery.wish_residual(_active(work.hp,work,side),_active(work.max_hp,work,side),
                work.wish_counter[:,side],eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,wish.hp),
                wish_counter=work.wish_counter.at[:,side].set(wish.counter))
            if _device_public is not None:device_public.wish(_device_public,side,ordinary,
                resolved=ordinary & (wish.hp>_active(_device_wish_before.hp,_device_wish_before,side)))
            if _device_public is not None:
                device_public.hp_line(_device_public,work,side,
                    ordinary & (wish.hp>_active(_device_wish_before.hp,_device_wish_before,side)),
                    source_class=device_public.HP_PRE_ITEM)
            if _lifecycle_trace:
                _public_trace,_wish_tick_capture=self_move_callbacks.finish(_public_trace,_wish_tick_pending,work)
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
        _faint_before_hp=work.hp
        for side in range(2):
            if _lifecycle_trace:_weather_old_hp=_active(work.hp,work,side)
            if _device_public is not None:_device_weather_old_hp=_active(work.hp,work,side)
            hp = climate.chip(table.weather,effective_weather,_active(work.types,work,side),
                _active(work.ability,work,side),_active(work.hp,work,side),
                _active(work.max_hp,work,side),eligible=ordinary)
            work = work._replace(hp=_set_active(work.hp,work,side,hp))
            if _device_public is not None:
                device_public.hp_line(_device_public,work,side,ordinary & (hp<_device_weather_old_hp),
                    source_class=device_public.HP_PRE_ITEM)
            if _lifecycle_trace:
                _weather_before_count=_public_trace.count
                _public_trace=weather_effect_callbacks.weather_chip(_public_trace,side,work.active[rows,side],
                    _weather_old_hp,hp,_active(work.max_hp,work,side),_active(work.status,work,side),weather,
                    eligible=ordinary&(hp<_weather_old_hp),_defer_own_unknown=_weather_observer_context is not None)
                _public_trace=weather_order_callbacks.annotate_weather_chip(_public_trace,_weather_before_count,
                    side,work.active[rows,side],end_speed[:,0],end_speed[:,1],
                    eligible=ordinary&(hp<_weather_old_hp),_defer_own_unknown=_weather_observer_context is not None)
        if _lifecycle_trace:
            if _weather_observer_context is not None:
                _weather_segment=weather_hp_segment.order_segment_device(_public_trace,
                    _weather_segment_before_count,_weather_pre_wish,work,end_speed,
                    _weather_hp_record,_weather_observer_context[1],eligible=ordinary,
                    pre_writes_state=_weather_pre_writes,hypothetical=_hypothetical)
                _public_trace=_weather_segment.tape
                _weather_hp_record=_weather_segment.record
        work=_clear_fainted_boosts(work,_faint_before_hp,eligible=ordinary)
    # Selected order10/priority0 handlers snapshot SPEED before suborder.
    # Equal-speed Speed Boost(3), Leftovers(4), seed(5), status(6) use independent ties.
    # Ability boosts do not re-sort the already queued residual handlers.
    # Fidelity M2 (design 2.3): partiallytrapped (subOrder 9, events 8-9) and Yawn (subOrder 19, events 12-13) join the
    # list (14 events; Encore moves to 10-11). Showdown's comparePriority sorts by speed, then subOrder (sim/battle.ts:
    # 404-411); the key multiplier is 64 because 2*19+1 > 31. The old events keep their relative order: their
    # subOrder*2+tie is at most 29 < 32 < 64, and keys stay distinct. An equal-Speed Wrap pair takes effect[0][3]; the
    # Yawn pair is inert (each handler touches only its own holder) and takes the fixed tie `sides`.
    event_sides = jnp.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1])
    event_suborder = jnp.array([3,3,4,4,5,5,6,6,9,9,14,14,19,19])
    tie = jnp.concatenate((draws[:,1,0,None] ^ sides,
        draws[:,0,9,None] ^ sides,draws[:,0,10,None] ^ sides,draws[:,1,9,None] ^ sides,
        effect_draws[:,0,3,None] ^ sides,draws[:,1,11,None] ^ sides,
        jnp.broadcast_to(sides,(draws.shape[0],2))),axis=1)
    event_keys = -end_speed[:,event_sides]*64 + event_suborder[None,:]*2 + tie
    event_order = jnp.argsort(event_keys,axis=1)
    for order in range(14):
        _faint_before_hp=work.hp
        event = event_order[:,order]
        side = event % 2
        ability_event = event < 2
        heal_event = (event >= 2) & (event < 4)
        seed_event = (event >= 4) & (event < 6)
        status_event = (event >= 6) & (event < 8)
        wrap_event = (event >= 8) & (event < 10)
        encore_event = (event >= 10) & (event < 12)
        yawn_event = event >= 12
        battle_continues = jnp.all(jnp.any(work.hp > 0, axis=2), axis=1)
        eligible = residual_ready & battle_continues
        if _device_public is not None:_device_residual_before=work
        hp = _active(work.hp,work,side)
        if _status_residual_context is not None:
            if _status_registered:
                _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,hypothetical=_hypothetical,eligible=eligible)
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
        if _device_public is not None and table.entry is not None:
            device_public.reveal_ability(_device_public,work,side,speed_boost & (_active(work.boosts,work,side)[:,4]!=_active(_device_residual_before.boosts,_device_residual_before,side)[:,4]))
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
                    work,_public_trace.stream_id,_leftovers_context.weather_profile_data,hypothetical=_hypothetical,eligible=eligible)
        if table.items is not None:
            if _lifecycle_trace:
                _execution_item=_active(work.item,work,side)
                _execution_leftovers_requested=(eligible & heal_event & (hp>0)
                    & table.items.leftovers[_execution_item])
                _execution_old_heal_hp=hp
                _public_trace,_execution_leftovers_ticket=execution_accounting.open_attempt(
                    _public_trace,execution_accounting.Site.LEFTOVERS,side,
                    active=_execution_leftovers_requested)
                _pinch_pending=None
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.pinch_item is not None:
                    _public_trace,_pinch_pending=pinch_callbacks.begin(_public_trace,work,side,
                        _execution_item,_active(work.boosts,work,side),hp,_active(work.max_hp,work,side),
                        table.items,_direct_hit_context,eligible=eligible & heal_event)
                else:
                    _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                        execution_accounting.Site.PINCH_ITEM,side,
                        active=eligible & heal_event & (hp>0)
                        & (hp*4<=_active(work.max_hp,work,side)) & (table.items.pinch_stat[_execution_item]>=0))
                if _leftovers_context is not None:
                    if _leftovers_registered and _leftovers_context.no_output_tables is not None:
                        _execution_full_hp_before=_leftovers_no_output.capture_before_device(
                            _public_trace,_execution_leftovers_ticket,work,event,_weather_hp_record,
                            _leftovers_context.no_output_tables,
                            requested=_execution_leftovers_requested&(hp==_active(work.max_hp,work,side)),hypothetical=_hypothetical)
            healed = held.leftovers_heal(table.items,_active(work.item,work,side),hp,
                _active(work.max_hp,work,side),eligible=eligible & heal_event)
            if _device_public is not None:
                device_public.reveal_item(_device_public,work,side,_active(work.item,work,side),eligible & heal_event & (healed>hp))
            item,boosts=held.consume_pinch(table.items,_active(work.item,work,side),
                _active(work.boosts,work,side),hp,_active(work.max_hp,work,side),
                eligible=eligible & heal_event)
            work=work._replace(item=_set_active(work.item,work,side,item),
                boosts=_set_active(work.boosts,work,side,boosts))
            if _device_public is not None:device_public.item_changes(_device_public,_device_residual_before,work,eligible & heal_event)
            if _lifecycle_trace:
                _public_trace=pinch_callbacks.finish(_public_trace,_pinch_pending,item,boosts)
            if _leftovers_context is not None:
                if _leftovers_registered:
                    _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                        work,_public_trace.stream_id,_leftovers_context.weather_profile_data,hypothetical=_hypothetical,
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
                _status_lethal_pending=status_residual_callbacks.begin_lethal_device(_public_trace,work,
                    _execution_status_ticket,side,_status_residual_context.data,requested=_execution_status_requested)
            if _status_registered:
                _weather_hp_record=status_residual_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,hypothetical=_hypothetical,
                    eligible=_execution_status_requested)
        residual = m.major_status_residual(hp,
            _active(work.max_hp, work, side), _active(work.status, work, side),
            _active(work.status_counter, work, side), active=eligible & status_event)
        work = work._replace(hp=_set_active(work.hp, work, side,jnp.where(heal_event,healed,residual.hp)),
            status_counter=_set_active(work.status_counter, work, side, residual.counter))
        if _device_public is not None:
            # Leftovers has its own source-local notification. This baseline
            # update prevents a later status loss being netted against healing.
            device_public.observe(_device_public,work,eligible & heal_event)
            device_public.hp_line(_device_public,work,side,
                eligible & status_event & (residual.hp<hp),source_class=device_public.HP_POST_ITEM)
        if _leftovers_context is not None:
            if _leftovers_registered and _leftovers_context.no_output_tables is not None:
                if table.items is not None:
                    _execution_full_hp_after=_leftovers_no_output.finish_device(_execution_full_hp_before,
                        _public_trace,work,_weather_hp_record,_leftovers_context.no_output_tables,hypothetical=_hypothetical)
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
                    work,_public_trace.stream_id,_status_residual_context.weather_profile_data,hypothetical=_hypothetical,
                    eligible=_execution_status_requested)
                # Selected kernel-event coverage only. Original-hook/history
                # equivalence is deliberately not inferred from speed membership.
                _status_singleton_ok=status_residual_callbacks.singleton_queue_matches(_public_trace,
                    _status_residual_context.data,_status_queue,side,_status_physical,_status_old_hp,_active(work.status,work,side))
                _status_guard_valid=(_profile_qualified(_weather_hp_record) & (_status_cached_speed_ok|_status_singleton_ok)
                    & ((_public_trace.refusal&~public_callbacks.UNINSTRUMENTED)==0))
                from . import engine_outcome_diagnostics as _residual_diagnostics
                _public_trace=_residual_diagnostics.record_residual_on_tape(_public_trace,
                    side,_status_physical,_profile_qualified(_weather_hp_record),_status_cached_speed_ok,
                    end_speed,_status_guard_valid,active=_execution_status_requested)
                _status_defer=_status_guard_valid
                _status_lethal_write=status_residual_callbacks.LethalWrite(_status_lethal_pending,work,_status_guard_valid)
            _public_trace=status_residual_callbacks.record_status_residual_device(_public_trace,
                side,_status_physical,_status_old_hp,_active(work.hp,work,side),_status_maximum,
                _status_old_status,_status_old_counter,_active(work.status_counter,work,side),
                _status_residual_context.data,eligible=eligible & status_event & (_status_old_hp>0)
                & ((_status_old_status==m.BRN)|(_status_old_status==m.PSN)|(_status_old_status==m.TOX)),
                _defer_own_unknown=_status_defer,_lethal_write=_status_lethal_write if _status_registered else None)
            if _status_registered:
                _status_lethal_requested=_status_lethal_pending.requested&(_status_new_hp==0)
                _public_trace=status_residual_callbacks.finish_lethal_device(_public_trace,_status_lethal_write)
                _status_execution=getattr(_public_trace,'execution',None)
                if _status_execution is not None:
                    _status_ledger=execution_accounting.unhandled(_status_execution.ledger,
                        _execution_status_ticket,active=_execution_status_requested & ~_status_guard_valid,
                        tables=_status_execution.tables)
                    _public_trace=execution_accounting._put_context(_public_trace,_status_ledger,_status_execution)
                _public_trace=execution_accounting.capture_status_residual_write(_public_trace,
                    _execution_status_ticket,requested=_execution_status_requested & _status_guard_valid & ~_status_lethal_requested,
                    positive_write=(_status_new_hp<_status_old_hp)&(_status_new_hp>0))
        if _leftovers_context is not None:
            # Record immediately AFTER this exact HP assignment. The immutable
            # Diagnostic mode retains UNKNOWN. The registered actual context
            # defers ONLY this new emitter latch; all prior refusals survive.
            _leftovers_new_hp=_active(work.hp,work,side)
            _leftovers_defer=False
            if _leftovers_registered:
                _weather_hp_record=leftovers_callbacks.retain_profile_device(_weather_hp_record,
                    work,_public_trace.stream_id,_leftovers_context.weather_profile_data,hypothetical=_hypothetical,
                    eligible=eligible & heal_event)
                _leftovers_defer=_profile_qualified(_weather_hp_record) & _leftovers_cached_speed_ok
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
                from . import engine_leech_seed_callbacks as leech_callbacks
                _leech_pending=None;_leech_capture=None;_leech_heal_pending=None
                _leech_registered=(_direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.leech_seed is not None)
                if _leech_registered:
                    _public_trace,_leech_pending=leech_callbacks.begin_drain(_public_trace,work,side,
                        _direct_hit_context,eligible=eligible & seed_event)
                else:
                    _execution_seed_attempt=(eligible & seed_event & (_active(work.hp,work,side)>0)
                        & ((_active(work.volatile,work,side)&_execution_leech_seed)!=0))
                    _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                        execution_accounting.Site.LEECH_DRAIN,side,active=_execution_seed_attempt)
                    _public_trace=execution_accounting.unhandled_attempt(_public_trace,
                        execution_accounting.Site.LEECH_HEAL,foe,
                        active=_execution_seed_attempt & (_active(work.hp,work,foe)>0))
            drained = recovery.leech(_active(work.hp,work,side),_active(work.max_hp,work,side),
                _active(work.volatile,work,side),_active(work.hp,work,foe),_active(work.max_hp,work,foe),
                eligible=eligible & seed_event)
            if table.damage_moves is not None:
                # Liquid Ooze on Leech Seed (gen4 abilities.ts liquidooze onSourceTryHeal, inherited by Gen 3:
                # 'leechseed' is in canOoze): the seeder takes the drained amount instead of healing it, also at full
                # HP (TryHeal runs before heal's full-HP check), and the seeded Ooze holder is revealed.
                _seed_amount=_active(work.hp,work,side)-drained.hp
                _seed_ooze=(_seed_amount>0) & table.damage_moves.liquid_ooze[_active(work.ability,work,side)]
                drained=drained._replace(recipient_hp=jnp.where(_seed_ooze,
                    jnp.maximum(0,_active(work.hp,work,foe)-_seed_amount),drained.recipient_hp))
                if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_seed_ooze)
            work = work._replace(hp=_set_active(work.hp,work,side,drained.hp))
            if _lifecycle_trace:
                _public_trace,_leech_capture=leech_callbacks.finish_drain(_public_trace,_leech_pending,work,drained)
                _public_trace,_leech_heal_pending=leech_callbacks.begin_heal(_public_trace,work,_leech_capture)
            work = work._replace(hp=_set_active(work.hp,work,foe,drained.recipient_hp))
            if _device_public is not None:
                device_public.hp_line(_device_public,work,side,
                    eligible & seed_event & (drained.hp<_active(_device_residual_before.hp,_device_residual_before,side)),
                    source_class=device_public.HP_UNKNOWN)
                device_public.hp_line(_device_public,work,foe,
                    eligible & seed_event & (drained.recipient_hp!=_active(_device_residual_before.hp,_device_residual_before,foe)),
                    source_class=device_public.HP_UNKNOWN)
                # -damage|seeder|..|[from] ability: Liquid Ooze|[of] holder (B2 attributes the ability to the holder).
                if table.damage_moves is not None:device_public.reveal_ability(_device_public,work,side,_seed_ooze)
            if _lifecycle_trace:
                _public_trace,_leech_heal_capture=leech_callbacks.finish_heal(_public_trace,_leech_heal_pending,work)
        # Fidelity M2: partiallytrapped's residual (data/mods/gen4/conditions.ts:110-118, data/mods/gen5/conditions.ts:
        # 8-23). fieldEvent lowers the duration first; at 0 its onEnd ends it (-end|X|Wrap|[partiallytrapped], no damage)
        # (sim/battle.ts:515-523). Otherwise a trapper that is not the foe's active, has fainted or has activeTurns 0
        # releases it silently; else the holder takes baseMaxhp/16, at least 1 (spreadDamage's clamp). A duration d thus
        # gives d-1 ticks. A fainted holder's handler is skipped.
        _wrap_counter=_active(work.partial_trap_counter,work,side);_wrap_source=_active(work.partial_trap_source,work,side)
        _wrap_holder=eligible & wrap_event & (_active(work.hp,work,side)>0) & (_wrap_counter>0)
        _wrap_held=((_wrap_source==work.active[rows,1-side]) & (_active(work.hp,work,1-side)>0)
            & (work.active_turns[rows,1-side]>0))
        _wrap_tick=_wrap_holder & (_wrap_counter>1) & _wrap_held
        _wrap_end=_wrap_holder & ~_wrap_tick
        _wrap_hp=_active(work.hp,work,side)
        _wrap_damage=jnp.minimum(_wrap_hp,jnp.maximum(1,_active(work.max_hp,work,side)//16))
        work=work._replace(hp=_set_active(work.hp,work,side,jnp.where(_wrap_tick,_wrap_hp-_wrap_damage,_wrap_hp)),
            partial_trap_counter=_set_active(work.partial_trap_counter,work,side,
                jnp.where(_wrap_tick,_wrap_counter-1,jnp.where(_wrap_end,0,_wrap_counter))),
            partial_trap_source=_set_active(work.partial_trap_source,work,side,jnp.where(_wrap_end,-1,_wrap_source)))
        if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_wrap_holder)
        if _device_public is not None:
            device_public.hp_line(_device_public,work,side,_wrap_tick,source_class=device_public.HP_POST_ITEM)
        # Fidelity M2: Yawn's countdown (data/moves.ts:21156-21167; no onResidual, a duration only). At 0 its onEnd
        # prints -end|X|move: Yawn|[silent] and calls trySetStatus('slp', source): a statused holder stays as it is
        # (setStatus of its own status fails silently), Sleep Clause Mod refuses it while another living member sleeps
        # from a foe (the source is the foe; data/rulesets.ts:1386-1401), Insomnia and Vital Spirit refuse it (onSetStatus),
        # and Safeguard is not consulted. slp's onStart (data/mods/gen3/conditions.ts) rolls random(2,6) with no active
        # move: 2+effect[side][2], the drowsy mon's side. A Lum Berry cures it at once (AfterSetStatus).
        _yawn_counter=_active(work.yawn_counter,work,side)
        _yawn_holder=eligible & yawn_event & (_active(work.hp,work,side)>0) & (_yawn_counter>0)
        work=work._replace(yawn_counter=_set_active(work.yawn_counter,work,side,jnp.where(_yawn_holder,_yawn_counter-1,_yawn_counter)))
        from .effects import sleep_clause_blocked as _yawn_clause
        _yawn_sleep=(_yawn_holder & (_yawn_counter==1) & (_active(work.status,work,side)==m.NONE)
            & ~_yawn_clause(work,side))
        if table.secondary is not None:
            _yawn_sleep&=~table.secondary.ability_immunity[_active(work.ability,work,side),m.SLP]
        work=work._replace(status=_set_active(work.status,work,side,jnp.where(_yawn_sleep,m.SLP,_active(work.status,work,side))),
            status_counter=_set_active(work.status_counter,work,side,
                jnp.where(_yawn_sleep,2+effect_draws[rows,side,2],_active(work.status_counter,work,side))),
            sleep_skipped=_set_active(work.sleep_skipped,work,side,jnp.where(_yawn_sleep,0,_active(work.sleep_skipped,work,side))),
            rest_origin=_set_active(work.rest_origin,work,side,_active(work.rest_origin,work,side) & ~_yawn_sleep))
        if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_yawn_holder)
        if table.secondary is not None:
            if _lifecycle_trace:
                work,_public_trace=secondaries.after_status_event(work,1-side,side,jnp.full_like(side,m.SLP),table.secondary,
                    applied=_yawn_sleep,item_table=table.items,_status_trace=_public_trace,_device_public=_device_public)
            else:
                work=secondaries.after_status_event(work,1-side,side,jnp.full_like(side,m.SLP),table.secondary,
                    applied=_yawn_sleep,item_table=table.items,_device_public=_device_public)
        elif _device_public is not None:device_public.observe(_device_public,work,_yawn_sleep)
        if table.encore is not None:
            if _lifecycle_trace:
                from . import engine_encore_callbacks as encore_callbacks
                _encore_residual_pending=None;_encore_residual_capture=None
                _encore_attempt=eligible & encore_event & (_active(work.hp,work,side)>0) & (_active(work.encore_counter,work,side)>0)
                if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.encore is not None:
                    _public_trace,_encore_residual_pending=encore_callbacks.begin(_public_trace,work,side,
                        jnp.zeros_like(side),_direct_hit_context,table.encore,eligible=_encore_attempt,residual=True)
            work=repeating.residual(work,side,eligible=eligible & encore_event)
            if _lifecycle_trace:
                _public_trace,_encore_residual_capture=encore_callbacks.finish(_public_trace,_encore_residual_pending,work)
                _public_trace=kernel_risks.branch_on_tape(_public_trace,kernel_risks.Branch.ENCORE,
                    _encore_attempt,active=live,captured_encore=_encore_residual_capture)
        work=_clear_fainted_boosts(work,_faint_before_hp,eligible=eligible)
    # Fidelity M2 (design 2.3): Perish Song's residual (data/mods/gen4/moves.ts:993-999, order 12), after the order-10
    # list. Faster first; an equal-Speed pair takes effect[1][3] (that side first). The duration falls; at 0 its onEnd
    # prints perish0 and faints the holder (faint() queues it), else -start|X|perishN. A fainted holder is skipped. The end
    # path runs no faintMessages before the next handler (sim/battle.ts:515-523), so both holders resolve even when the
    # first faint empties a side (Gen 3 checkWin with both sides empty is a tie): the pass is gated once, at its start.
    _perish_continues=residual_ready & jnp.all(jnp.any(work.hp>0,axis=2),axis=1)
    _perish_first=jnp.where(end_speed[:,0]!=end_speed[:,1],(end_speed[:,1]>end_speed[:,0]).astype(jnp.int32),effect_draws[:,1,3])
    for order in range(2):
        side=jnp.where(order==0,_perish_first,1-_perish_first)
        _faint_before_hp=work.hp
        _perish_counter=_active(work.perish_counter,work,side)
        _perish_holder=_perish_continues & (_active(work.hp,work,side)>0) & (_perish_counter>0)
        _perish_faint=_perish_holder & (_perish_counter==1)
        work=work._replace(perish_counter=_set_active(work.perish_counter,work,side,jnp.where(_perish_holder,_perish_counter-1,_perish_counter)),
            hp=_set_active(work.hp,work,side,jnp.where(_perish_faint,0,_active(work.hp,work,side))))
        if _lifecycle_trace:_public_trace=status_callbacks.unknown(_public_trace,_perish_holder)
        # perish0 then faint: a faint line, no -damage (as Destiny Bond's and Self-Destruct's faints).
        if _device_public is not None:device_public.observe(_device_public,work,_perish_faint,record_damage=False)
        work=_clear_fainted_boosts(work,_faint_before_hp,eligible=_perish_continues)
    # Fidelity M1 (design 2.3): mustrecharge has a duration and no residual handler, so the Residual event
    # still decrements it, last (order false sorts after every numbered handler, sim/battle.ts). At 0 it ends
    # silently: a recharge action cancelled by a Gen 3 faint expires here. Fainted holders are cleared below;
    # a residual faint that ends the battle stops the Residual event first (checkWin, `if (this.ended) return`).
    _recharge_tick=residual_ready & jnp.all(jnp.any(work.hp>0,axis=2),axis=1)
    work=work._replace(recharge_counter=jnp.where(_recharge_tick[:,None,None] & (work.recharge_counter>0),
        work.recharge_counter-1,work.recharge_counter))
    work=linking.prune(work)
    work=work._replace(volatile=jnp.where(work.hp>0,work.volatile,
        work.volatile & jnp.uint32(~(FOCUS_PUNCH|FOCUS_LOST)&0xffffffff)))
    work=work._replace(**{name:jnp.where(work.hp>0,getattr(work,name),0)
        for name in ("last_move","encore_move","encore_counter","charge_move","charge_counter")})
    # faintMessages -> clearVolatile(false) empties the volatiles, the Substitute too
    # (sim/battle.ts faintMessages, sim/pokemon.ts clearVolatile). The fainted holder
    # stays the active token until replaced; its replacement request sees no Substitute.
    work=work._replace(substitute_hp=jnp.where(work.hp>0,work.substitute_hp,0))
    # Fidelity B: the same clearVolatile ends confusion (reachable since Signal Beam starts it) and the Choice
    # lock (choicelock), so a forced-replacement request sees neither; departure clears both later anyway.
    work=work._replace(**{name:jnp.where(work.hp>0,getattr(work,name),0) for name in ("confusion_counter","choice_move")})
    # clearVolatile at the faint also ends Destiny Bond (design 2.4).
    work=work._replace(volatile=jnp.where(work.hp>0,work.volatile,work.volatile & jnp.uint32(~DESTINY_BOND&0xffffffff)))
    # Fidelity M1: and mustrecharge (design 2.4).
    work=work._replace(recharge_counter=jnp.where(work.hp>0,work.recharge_counter,0))
    # Fidelity M2: and perishsong, yawn and partiallytrapped (design 2.4). A trapper's faint leaves its target's trap
    # until that target's next residual tick releases it silently (data/mods/gen5/conditions.ts:14-20).
    work=work._replace(**{name:jnp.where(work.hp>0,getattr(work,name),0) for name in ("perish_counter","yawn_counter","partial_trap_counter")},
        partial_trap_source=jnp.where(work.hp>0,work.partial_trap_source,-1))
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
    # The effect stream is saved with the other blocks for a suspended turn (design 2.1), zero otherwise.
    work = work._replace(effect_draws=jnp.where(defer_residual[:,None,None],effect_draws,0))
    if _lifecycle_trace:
        from . import engine_baton_pass_callbacks as baton_callbacks
        _public_trace=baton_callbacks.finalize_request(_public_trace,work)
        from . import engine_terminal_callbacks as terminal_callbacks
        _public_trace=terminal_callbacks.capture_final_on_tape(_public_trace,work,active=live)
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
        _public_trace=kernel_risks.end_on_tape(_public_trace,work,active=live)
    work = work._replace(volatile=jnp.where(ordinary[:, None, None],
        work.volatile & jnp.uint32(~(FLINCH|FOCUS_PUNCH|FOCUS_LOST) & 0xffffffff), work.volatile))
    if table.items is not None:
        # endTurn's DisableMove: choicelock's onDisableMove (data/conditions.ts) removes a lock whose
        # holder no longer holds a Choice item (a Knock Off). endTurn runs when the turn's queue
        # empties with no replacement request pending: exactly when a TURN line follows (an
        # end-of-turn faint defers it to the replacement step). Identity otherwise: a lock is only
        # set while the holder has Choice Band, and departure clears it.
        _end_turn=~work.done & ~work.pending_residual & ~jnp.any(work.forced_switch,axis=1)
        work=work._replace(choice_move=jnp.where(_end_turn[:,None,None] & ~table.items.choice_band[work.item],
            0,work.choice_move))
    work = jax.tree.map(lambda old, new: jnp.where(live.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), state, work)
    if _device_public is not None:
        device_public.turn_end(_device_public,state,work,ordinary,upkeep=residual_ready & ~work.done,
            leftovers=None if table.items is None else table.items.leftovers)
    result = StepResult(work, invalid, placeholders)
    if _device_public is not None:return device_public.PublicStepResult(result,_device_public.trackers)
    if _lifecycle_trace:
        if _weather_observer_context is not None:
            return WeatherProfileTracedStep(result,_public_trace,_weather_hp_record)
    return result if _public_trace is None else public_callbacks.TracedStep(result,_public_trace)


def initialize_with_public(state,table,*,belief_tables=None,eligible=True,tie_coin=None):
    """Initialize actual entries and their public knowledge in one transaction."""
    context=device_public.initialize(state,belief_tables,port_tables=getattr(table,'public_port',None))
    result=initialize(state,table,eligible=eligible,tie_coin=tie_coin,_device_public=context)
    return device_public.Initialized(result,context.trackers)


def step_with_public(state,actions,table,trackers,*,belief_tables=None,chance_inputs=None):
    """Actual numerical step plus public carry; explicit chance is never redrawn."""
    if trackers.mon_features is None or trackers.observed_moves is None:
        raise ValueError('initialize_with_public tracker carry required')
    context=device_public.Context(trackers,belief_tables,port_tables=getattr(table,'public_port',None))
    chance=sample_chance_inputs(state) if chance_inputs is None else chance_inputs
    return step_with_chance(state,actions,table,chance,_device_public=context)
