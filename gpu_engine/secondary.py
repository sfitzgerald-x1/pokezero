"""Selected Gen 3 self/target secondaries and Static, not full event coverage.

The caller supplies one fixed [0,100) roll and the original Substitute-hit
flag. Self effects survive Substitute damage and a target KO; target effects
do not. No chance is drawn conditionally here.
"""
from typing import NamedTuple
from . import engine_device_public_hooks as device_public

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from .effects import from_metadata as effects_from_metadata
from .items import consume_lum
from .state import FLINCH,FOCUS_PUNCH
from .weather import SUN


# chance, major status, SpD drop, flinch; pinned Gen 3 overrides matter (Crunch).
EFFECTS = {
    "rockslide": (30, m.NONE, False, True),
    "icebeam": (10, m.FRZ, False, False),
    "icepunch": (10, m.FRZ, False, False),
    "thunderbolt": (10, m.PAR, False, False),
    "thunderpunch": (10, m.PAR, False, False),
    "thunder": (30, m.PAR, False, False),
    "bodyslam": (30, m.PAR, False, False),
    "sludgebomb": (30, m.PSN, False, False),
    "fireblast": (10, m.BRN, False, False),
    "flamethrower": (10, m.BRN, False, False),
    "firepunch": (10, m.BRN, False, False),
    "shadowball": (20, m.NONE, True, False),
    "crunch": (20, m.NONE, True, False),
    "psychic": (10, m.NONE, True, False),
    "sacredfire": (50, m.BRN, False, False),
    # Fidelity patch B: secondary {chance:30, volatileStatus:'flinch'} (data/moves.ts headbutt,
    # needlearm; gen3 changes only Needle Arm's flags). The flinch path below, unchanged.
    "headbutt": (30, m.NONE, False, True),
    "needlearm": (30, m.NONE, False, True),
}

EXTRA_CHANCES = {"silverwind": 10, "ancientpower": 10, "mudshot": 100,
    # Fidelity patch B (apply_chance_effects): data/moves.ts secondary chances, no gen3-8 override.
    "irontail": 30, "steelwing": 10, "meteormash": 20, "signalbeam": 10}

# Fidelity patch B. secondary.boosts on the target (data/moves.ts irontail: def -1), secondary.self.boosts
# on the user (steelwing: def +1; meteormash: atk +1, gen5's 100 BP/85% inherited), and secondary
# volatileStatus confusion (signalbeam; Bug is a Physical type in Gen 3, data/mods/gen3/scripts.ts:5-13).
# Stage order: atk def spa spd spe acc eva.
TARGET_DROPS = {"irontail": [0, -1, 0, 0, 0, 0, 0]}
SELF_BOOSTS = {"steelwing": [0, 1, 0, 0, 0, 0, 0], "meteormash": [1, 0, 0, 0, 0, 0, 0]}
CONFUSION_MOVES = ("signalbeam",)


class SecondaryTable(NamedTuple):
    chance: jax.Array
    status: jax.Array
    spd_drop: jax.Array
    flinch: jax.Array
    grace: jax.Array
    dust: jax.Array
    clear_body: jax.Array
    inner_focus: jax.Array
    ability_immunity: jax.Array   # [A,7], selected status-admission abilities
    type_immunity: jax.Array
    contact: jax.Array           # [M], pinned catalog flag, not damage category
    static: jax.Array            # [A], Gen3 one-third contact paralysis
    speed_drop: jax.Array        # [M], selected target Speed secondary
    self_omni: jax.Array         # [M], five user stages; not target Dust/Sub gated
    synchronize: jax.Array       # [A], deterministic reflection of fresh status
    rough_skin: jax.Array        # [A], Gen3 deterministic contact HP damage
    color_change: jax.Array      # [A], original Gen3 inherited DamagingHit
    # Fidelity patch B (appended; every field above keeps its position and value).
    target_drops: jax.Array      # [M,7], secondary.boosts on the target (Iron Tail)
    self_boosts: jax.Array       # [M,7], secondary.self.boosts on the user (Steel Wing, Meteor Mash)
    confusion: jax.Array         # [M], secondary volatileStatus confusion (Signal Beam)
    own_tempo: jax.Array         # [A], onTryAddVolatile refuses confusion


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    rows = [EFFECTS.get(name, (EXTRA_CHANCES.get(name, 0), m.NONE, False, False)) for name in vocab["moves"]]
    flags = lambda names: jnp.asarray([name in names for name in vocab["abilities"]])
    immunity = np.zeros((len(vocab["abilities"]), 7), bool)
    # Insomnia/Vital Spirit onSetStatus refuse slp (data/abilities.ts:2129-2135,
    # 5283-5289; gen3 inherits them): sleep moves fail with -immune [from] ability.
    blocked = {"limber": (m.PAR,), "magmaarmor": (m.FRZ,),
        "waterveil": (m.BRN,), "immunity": (m.PSN, m.TOX),
        "insomnia": (m.SLP,), "vitalspirit": (m.SLP,)}
    for index, name in enumerate(vocab["abilities"]):
        for status in blocked.get(name, ()): immunity[index, status] = True
    move_rows = metadata.get('moves')
    if move_rows is not None and len(move_rows) != len(vocab['moves']):
        raise ValueError('move metadata/vocabulary mismatch')
    contact = [bool(row and row.get('flags', {}).get('contact')) for row in move_rows] \
        if move_rows is not None else [False]*len(vocab['moves'])
    stages = lambda chosen: jnp.asarray([chosen.get(name, [0]*7) for name in vocab['moves']], jnp.int32).reshape(len(vocab['moves']), 7)
    return SecondaryTable(jnp.asarray([row[0] for row in rows], jnp.int32),
        jnp.asarray([row[1] for row in rows], jnp.int32),
        jnp.asarray([row[2] for row in rows]), jnp.asarray([row[3] for row in rows]),
        flags({"serenegrace"}), flags({"shielddust"}), flags({"clearbody", "whitesmoke"}),
        flags({"innerfocus"}), jnp.asarray(immunity), effects_from_metadata(metadata).status_immunity,
        jnp.asarray(contact), flags({'static'}),
        jnp.asarray([name == 'mudshot' for name in vocab['moves']]),
        jnp.asarray([name in {'silverwind', 'ancientpower'} for name in vocab['moves']]),
        flags({'synchronize'}),flags({'roughskin'}),flags({'colorchange'}),
        stages(TARGET_DROPS),stages(SELF_BOOSTS),
        jnp.asarray([name in CONFUSION_MOVES for name in vocab['moves']]),flags({'owntempo'}))


def apply_color_change(state,side,move_type,table,*,eligible,_status_trace=None,_direct_hit_context=None,move=None,_device_public=None):
    """Original Gen3 DamagingHit, after secondaries and before recoil.

    The caller supplies positive literal target damage and the SAME resolved
    attack type (including Hidden Power). No additional chance is consumed.
    """
    rows=jnp.arange(state.hp.shape[0]);target=1-side;physical=state.active[rows,target]
    old=state.types[rows,target,physical];ability=state.ability[rows,target,physical]
    changes=(eligible & (state.hp[rows,target,physical]>0) & table.color_change[ability]
        & (move_type>0) & ~jnp.any(old==move_type[:,None],axis=1))
    pending=None
    if _status_trace is not None:
        from . import engine_color_change_callbacks as callbacks
        _status_trace,pending=callbacks.begin(_status_trace,state,side,move,move_type,
            _direct_hit_context,table,eligible=changes)
    new=jnp.stack((move_type,jnp.zeros_like(move_type)),axis=1)
    if _device_public is not None:_device_color_before=state
    state=state._replace(types=state.types.at[rows,target,physical].set(jnp.where(changes[:,None],new,old)))
    if _device_public is not None:device_public.color_change(_device_public,_device_color_before,state,target,changes)
    if _status_trace is not None:
        _status_trace=callbacks.finish(_status_trace,pending,state)
        return state,_status_trace
    return state


def after_status_event(state, source, afflicted, status, table, *, applied, item_table=None, _status_trace=None, _device_public=None):
    """Gen3 AfterSetStatus: reflect first, then consume each event's Lum.

    The caller has just installed a fresh status and supplies that exact mask.
    Synchronize uses no new chance and bypasses the source's Substitute. Toxic
    reflects as ordinary poison; sleep/freeze and self-infliction do not reflect.
    Existing type/ability immunity and a living, unstatused source still apply.
    """
    rows=jnp.arange(state.hp.shape[0])
    own,target=state.active[rows,source],state.active[rows,afflicted]
    reflected=jnp.where(status==m.TOX,m.PSN,status)
    reflects=(jnp.asarray(applied) & (source!=afflicted)
        & table.synchronize[state.ability[rows,afflicted,target]]
        & ((status==m.PAR)|(status==m.BRN)|(status==m.PSN)|(status==m.TOX))
        & (state.hp[rows,source,own]>0) & (state.status[rows,source,own]==m.NONE)
        & status_allowed(table,reflected,state.types[rows,source,own],state.ability[rows,source,own]))
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        _status_trace=callbacks.unknown(_status_trace,reflects)
    updates={}
    for name,value in [('status',reflected),('status_counter',0),('sleep_skipped',0),('rest_origin',False)]:
        old=getattr(state,name)
        updates[name]=old.at[rows,source,own].set(jnp.where(reflects,value,old[rows,source,own]))
    state=state._replace(**updates)
    if _device_public is not None:
        device_public.reveal_ability(_device_public,state,afflicted,reflects)
        device_public.observe(_device_public,state,jnp.asarray(applied)|reflects)
    if item_table is not None:
        # A nested reflection finishes before the original holder's Lum event.
        # Cure now, before a later contact event can inflict/reflect again.
        for side,changed in ((source,reflects),(afflicted,jnp.asarray(applied))):
            active=state.active[rows,side]
            get=lambda a:a[rows,side,active]
            if _device_public is not None:_device_item_before=state
            old_item=get(state.item)
            result=consume_lum(item_table,old_item,get(state.status),get(state.status_counter),
                get(state.confusion_counter),get(state.hp),eligible=changed)
            consumed=result.item!=old_item
            if _status_trace is not None:
                from . import engine_item_callbacks as item_callbacks
                _status_trace=item_callbacks.lum_consumed(_status_trace,side,active,old_item,
                    get(state.status),result.status,
                    (get(state.confusion_counter)>0) & (result.confusion==0),eligible=consumed)
            updates={}
            for name,value in zip(('item','status','status_counter','confusion_counter'),result):
                updates[name]=getattr(state,name).at[rows,side,active].set(value)
            for name in ('sleep_skipped','rest_origin'):
                old=getattr(state,name)
                updates[name]=old.at[rows,side,active].set(jnp.where(consumed,jnp.zeros_like(get(old)),get(old)))
            state=state._replace(**updates)
            if _device_public is not None:device_public.item_changes(_device_public,_device_item_before,state,changed)
    if _status_trace is not None:return state,_status_trace
    return state


def apply_contact(state, side, move, roll, table, *, eligible, item_table=None, _status_trace=None,
                  _rough_skin_context=None, _rough_skin_capture=None, _device_public=None):
    """Gen3 Static and Rough Skin onDamagingHit, before Lum/recoil.

    The existing uniform [0,60) auxiliary draw gives exactly 20/60 = 1/3.
    It is shared with phazing only across mutually exclusive dispatched moves:
    the pinned Gen3 Roar/Whirlwind have no contact flag. A source plan rejects
    a table claiming both consumers for one move. Saved phazing draws already
    survive interrupted turns. No percentile approximation or extra key draw.

    Caller excludes original Substitute damage (even if it broke), misses,
    immunity, Protect and zero damage. A KO of the Static holder still triggers;
    a fainted attacking source cannot receive status. Shield Dust and Serene
    Grace do not affect this ability event; Limber and existing status do.
    """
    rows = jnp.arange(state.hp.shape[0]); foe = 1-side
    own, target = state.active[rows, side], state.active[rows, foe]
    ability = state.ability[rows, side, own]
    proc = (jnp.asarray(eligible) & table.contact[move]
        & table.static[state.ability[rows, foe, target]] & (jnp.asarray(roll) < 20)
        & (state.hp[rows, side, own] > 0) & (state.status[rows, side, own] == m.NONE)
        & ~table.ability_immunity[ability, m.PAR])
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        _status_trace=callbacks.unknown(_status_trace,proc)
    state=state._replace(
        status=state.status.at[rows, side, own].set(jnp.where(proc, m.PAR, state.status[rows, side, own])),
        status_counter=state.status_counter.at[rows, side, own].set(
            jnp.where(proc, 0, state.status_counter[rows, side, own])))
    if _device_public is not None:
        device_public.reveal_ability(_device_public,state,foe,proc)
        device_public.observe(_device_public,state,proc)
    if _status_trace is not None:
        state,_status_trace=after_status_event(state,foe,side,jnp.full(state.hp.shape[0],m.PAR),table,
            applied=proc,item_table=item_table,_status_trace=_status_trace,_device_public=_device_public)
    else:
        state=after_status_event(state,foe,side,jnp.full(state.hp.shape[0],m.PAR),table,
            applied=proc,item_table=item_table,_device_public=_device_public)
    # Original Gen3 roughskin.onDamagingHit: Battle.damage floors and clamps
    # baseMaxhp/16 to at least one, capped by living source HP. No RNG draw.
    rough=(jnp.asarray(eligible)&table.contact[move]
        &table.rough_skin[state.ability[rows,foe,target]]&(state.hp[rows,side,own]>0))
    if _device_public is not None:device_public.reveal_ability(_device_public,state,foe,rough)
    old_hp=state.hp[rows,side,own]
    if _status_trace is not None:
        from . import engine_rough_skin_callbacks as rough_callbacks
        _status_trace,pending=rough_callbacks.begin(_status_trace,_rough_skin_context,
            _rough_skin_capture,state,side,move,table,requested=rough)
    hp=jnp.maximum(0,old_hp-jnp.maximum(1,state.max_hp[rows,side,own]//16))
    state=state._replace(hp=state.hp.at[rows,side,own].set(jnp.where(rough,hp,old_hp)))
    if _status_trace is not None:
        _status_trace=rough_callbacks.record(_status_trace,pending,state)
        return state,_status_trace
    return state


def status_allowed(table, status, types, ability, *, effective_weather=0):
    immune = jnp.any(table.type_immunity[status[:, None], types], axis=1)
    immune |= table.ability_immunity[ability, status]
    immune |= (status == m.FRZ) & (jnp.asarray(effective_weather) == SUN)
    return ~immune


def apply(state, side, move, roll, table, *, eligible, effective_weather=0, substitute_hit=False,
          item_table=None, _status_trace=None, _secondary_status_context=None, _device_public=None):
    rows = jnp.arange(state.hp.shape[0]); foe = 1 - side
    own, target = state.active[rows, side], state.active[rows, foe]
    get = lambda value: value[rows, foe, target]
    ability = get(state.ability)
    chance = table.chance[move] * jnp.where(table.grace[state.ability[rows, side, own]], 2, 1)
    trigger = jnp.asarray(eligible) & (jnp.asarray(roll) < chance)
    proc = (trigger & (get(state.hp) > 0) & ~table.dust[ability]
        & ~jnp.asarray(substitute_hit))
    status = table.status[move]
    changes = (proc & (status != m.NONE) & (get(state.status) == m.NONE)
        & status_allowed(table, status, get(state.types), ability, effective_weather=effective_weather))
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        if _secondary_status_context is not None:
            from . import engine_secondary_status_callbacks as secondary_status
            _secondary_write=secondary_status.Write(jnp.asarray(eligible),trigger,proc,changes,get(state.hp)>0,
                jnp.broadcast_to(jnp.asarray(substitute_hit),(state.hp.shape[0],)),table.dust[ability],
                get(state.status),get(state.status),get(state.status_counter),get(state.status_counter),
                state.ability[rows,side,own],ability,get(state.item))
            _status_trace,_secondary_ticket=secondary_status.begin(_status_trace,
                _secondary_status_context,table,side,own,target,move,_secondary_write)
        else:
            _status_trace=callbacks.unknown(_status_trace,changes)
        from . import engine_kernel_frame_risk as kernel_risks
    boosts = get(state.boosts)
    drops = proc & table.spd_drop[move] & ~table.clear_body[ability]
    if _status_trace is not None:
        _status_trace=kernel_risks.branch_on_tape(_status_trace,kernel_risks.Branch.SECONDARY_STATS,
            proc & table.spd_drop[move],active=jnp.asarray(eligible))
    boosts = boosts.at[:, 3].set(jnp.where(drops, jnp.maximum(-6, boosts[:, 3] - 1), boosts[:, 3]))
    speed = proc & table.speed_drop[move] & ~table.clear_body[ability]
    if _status_trace is not None:
        _status_trace=kernel_risks.branch_on_tape(_status_trace,kernel_risks.Branch.SECONDARY_STATS,
            proc & table.speed_drop[move],active=jnp.asarray(eligible))
    boosts = boosts.at[:, 4].set(jnp.where(speed, jnp.maximum(-6, boosts[:, 4] - 1), boosts[:, 4]))
    self_proc = trigger & table.self_omni[move] & (state.hp[rows, side, own] > 0)
    if _status_trace is not None:
        _status_trace=kernel_risks.branch_on_tape(_status_trace,kernel_risks.Branch.SECONDARY_STATS,
            self_proc,active=jnp.asarray(eligible))
    own_boosts = state.boosts[rows, side, own]
    own_boosts = own_boosts.at[:, :5].set(jnp.where(self_proc[:, None],
        jnp.clip(own_boosts[:, :5] + 1, -6, 6), own_boosts[:, :5]))
    flinch = (proc & table.flinch[move] & ~table.inner_focus[ability]
        & ((get(state.volatile) & FOCUS_PUNCH)==0))
    if _status_trace is not None:
        _status_trace=kernel_risks.branch_on_tape(_status_trace,kernel_risks.Branch.SECONDARY_FLINCH,
            proc & table.flinch[move],active=jnp.asarray(eligible))
    state=state._replace(
        status=state.status.at[rows, foe, target].set(jnp.where(changes, status, get(state.status))),
        status_counter=state.status_counter.at[rows, foe, target].set(
            jnp.where(changes, 0, get(state.status_counter))),
        boosts=state.boosts.at[rows, foe, target].set(boosts).at[rows, side, own].set(own_boosts),
        volatile=state.volatile.at[rows, foe, target].set(
            get(state.volatile) | jnp.where(flinch, jnp.uint32(FLINCH), jnp.uint32(0))))
    if _device_public is not None:device_public.observe(_device_public,state,jnp.asarray(eligible))
    if _status_trace is not None:
        if _secondary_status_context is not None:
            _secondary_write=_secondary_write._replace(new_status=state.status[rows,foe,target],
                new_counter=state.status_counter[rows,foe,target])
            _status_trace=secondary_status.finish(_status_trace,_secondary_ticket,side,own,target,move,
                _secondary_write,_secondary_status_context,table)
        return after_status_event(state,side,foe,status,table,applied=changes,item_table=item_table,_status_trace=_status_trace,_device_public=_device_public)
    return after_status_event(state,side,foe,status,table,applied=changes,item_table=item_table,_device_public=_device_public)


def apply_chance_effects(state, side, move, roll, duration_roll, table, *, eligible, substitute_hit=False,
                         _device_public=None):
    """Fidelity patch B: Iron Tail, Steel Wing, Meteor Mash and Signal Beam's chance secondaries.

    Called right after apply() (whose AST-pinned call is unchanged) with the same eligibility (a hit
    that was not missed, protected or type-immune) and the same side's [0,100) roll: each of these
    moves has ONE secondary entry, and secondaries() (sim/battle-actions.ts:1357-1373) rolls
    random(100) once per entry, so the shared draw maps one to one. Serene Grace doubles the chance
    (gen4 abilities.ts onModifyMove, inherited by Gen 3). No chance is drawn here: the confusion
    duration reads the caller's duration roll, the attacker's draws[s][7] (2+d), which Signal Beam
    never shares with a primary sleep or Encore. Headbutt and Needle Arm flinch through apply().

    Target entries (Iron Tail's Def -1, Signal Beam's confusion) need a living target that is not
    behind the Substitute this hit met (spreadMoveHit nulls that target before secondaries) and has
    no Shield Dust (onModifySecondaries keeps only self entries). Iron Tail's drop goes through
    boost(): capped at -6 (a no-op), Clear Body and White Smoke delete it with no line because the
    effect carries secondaries (data/abilities.ts:513-531, 5434-5452); Hyper Cutter and Keen Eye
    guard stats no row here lowers. Signal Beam's addVolatile('confusion') fails silently on an
    already confused target (no onRestart) and on Own Tempo (onTryAddVolatile returns null; its onHit
    -immune line needs a primary volatileStatus); otherwise it starts at random(2,6), 2..5
    (data/conditions.ts:165-174).

    Self entries (Steel Wing's Def +1, Meteor Mash's Atk +1) survive Shield Dust, the Substitute and a
    target KO (selfDrops moves the user itself) and need only a living user; +6 is a no-op, and the
    user's own Clear Body never blocks its own boost (source === target).

    Returns the state and the [B] mask of attempted writes (for the lifecycle trace's refusal).
    """
    rows = jnp.arange(state.hp.shape[0]); foe = 1 - side
    own, target = state.active[rows, side], state.active[rows, foe]
    get = lambda value: value[rows, foe, target]
    ability = get(state.ability)
    chance = table.chance[move] * jnp.where(table.grace[state.ability[rows, side, own]], 2, 1)
    trigger = jnp.asarray(eligible) & (jnp.asarray(roll) < chance)
    proc = (trigger & (get(state.hp) > 0) & ~table.dust[ability]
        & ~jnp.asarray(substitute_hit))
    drop = table.target_drops[move]
    dropped = proc & jnp.any(drop != 0, axis=1)
    old = get(state.boosts)
    lowered = jnp.clip(old + jnp.where(table.clear_body[ability][:, None] & (drop < 0), 0, drop), -6, 6)
    raise_ = table.self_boosts[move]
    raised = trigger & jnp.any(raise_ != 0, axis=1) & (state.hp[rows, side, own] > 0)
    own_old = state.boosts[rows, side, own]
    confusing = proc & table.confusion[move]
    confused = confusing & ~table.own_tempo[ability] & (get(state.confusion_counter) == 0)
    state = state._replace(
        boosts=state.boosts.at[rows, foe, target].set(jnp.where(dropped[:, None], lowered, old))
            .at[rows, side, own].set(jnp.where(raised[:, None], jnp.clip(own_old + raise_, -6, 6), own_old)),
        confusion_counter=state.confusion_counter.at[rows, foe, target].set(
            jnp.where(confused, 2 + jnp.asarray(duration_roll, jnp.int32), get(state.confusion_counter))))
    attempted = dropped | raised | confusing
    # Publish the stages (the confusion bag is public_move_state's, at the end of the move).
    if _device_public is not None:device_public.observe(_device_public, state, attempted)
    return state, attempted
