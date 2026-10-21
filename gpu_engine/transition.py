"""First lockstep turn integration; explicitly incomplete, not covered gameplay.

All moves are still flagged placeholders: damaging moves use static type and
power, and only the explicit major-status/self-effect subset is integrated.
Items, abilities, weather, most volatiles and
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
from .effects import MoveEffects, from_metadata as effects_from_metadata
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
        effects_from_metadata(metadata))


def state_from_pool(pool, team_indices, *, seed=0) -> BattleState:
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
    return state._replace(**values, hp=hp, max_hp=hp, done=jnp.zeros_like(state.done))


def _active(value, state, side):
    rows = jnp.arange(state.hp.shape[0])
    return value[rows, side, state.active[rows, side]]


def _set_active(value, state, side, update):
    rows = jnp.arange(state.hp.shape[0])
    return value.at[rows, side, state.active[rows, side]].set(update)


class StepResult(NamedTuple):
    state: BattleState
    invalid_actions: jax.Array      # [B]; invalid live requests freeze that battle
    placeholder_moves: jax.Array   # [B]; every executed move is still a placeholder


def step(state: BattleState, actions: jax.Array, table: MoveTable) -> StepResult:
    """Static-shape two-action turn, with forced replacement as a pseudo-turn.

    Actions must be integer [B,2]. Chance consumes the same fixed block per live
    battle, independent of branches. Dead/invalid battles remain byte-identical.
    The ordinary turn integrates switches, priority/speed, status action gates,
    PP, confusion self-damage, Struggle recoil, static single-hit damage,
    major-status residuals and party exhaustion. Flinch uses volatile bit 0.
    Switch events and all move effects remain placeholders, not fidelity claims.
    """
    if actions.shape != state.active.shape or not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("actions must be integer [B,2]")
    rows = jnp.arange(state.hp.shape[0])
    sides = jnp.arange(2)[None, :]
    active_hp = jnp.take_along_axis(state.hp, state.active[..., None], axis=2)[..., 0]
    forced = state.forced_switch | (active_hp <= 0)
    replacement = jnp.any(forced, axis=1)
    requested = ~replacement[:, None] | forced
    valid = jnp.take_along_axis(legal_actions(state), jnp.clip(actions, 0, 8)[..., None], axis=2)[..., 0]
    invalid = ~state.done & jnp.any(requested & (~valid | (actions < 0) | (actions > 8)), axis=1)
    live = ~state.done & ~invalid
    splits = jax.vmap(lambda key: jax.random.split(key, 2))(state.keys)
    draws = jax.vmap(lambda key: jax.random.randint(key, (2, 8), 0,
        jnp.array([2, 100, 100, 100, 16, 16, 100, 4], jnp.int32)))(splits[:, 1])
    work = state._replace(keys=splits[:, 0])
    switches = live[:, None] & requested & (actions >= 4)
    bench = bench_indices(state.active)
    target = jnp.take_along_axis(bench, jnp.clip(actions - 4, 0, 4)[..., None], axis=2)[..., 0]
    work = work._replace(active=jnp.where(switches, target, state.active),
                         forced_switch=jnp.where(switches, False, state.forced_switch))
    # Ordinary switching drops boosts and toxic escalation on the departing mon.
    old_boosts = state.boosts[rows[:, None], sides, state.active]
    boosts = state.boosts.at[rows[:, None], sides, state.active].set(
        jnp.where(switches[..., None], 0, old_boosts))
    old_counter = state.status_counter[rows[:, None], sides, state.active]
    toxic = state.status[rows[:, None], sides, state.active] == m.TOX
    counters = state.status_counter.at[rows[:, None], sides, state.active].set(
        jnp.where(switches & toxic, 0, old_counter))
    work = work._replace(boosts=boosts, status_counter=counters)
    for name in ("volatile", "confusion_counter", "protect_counter", "substitute_hp"):
        value = getattr(work, name)
        old = value[rows[:, None], sides, state.active]
        work = work._replace(**{name: value.at[rows[:, None], sides, state.active].set(
            jnp.where(switches, 0, old))})
    slot = jnp.clip(actions, 0, 3)
    pp = state.pp[rows[:, None], sides, state.active]
    usable = (pp > 0) & ~state.disabled[rows[:, None], sides, state.active]
    struggle = ~jnp.any(usable, axis=-1)
    moves = state.moves[rows[:, None], sides, state.active, slot]
    moves = jnp.where(struggle, table.struggle, moves)
    stats = state.stats[rows[:, None], sides, state.active]
    speed = m.boosted_stat(stats[..., 5], old_boosts[..., 4])
    speed = m.paralyzed_speed(speed, state.status[rows[:, None], sides, state.active] == m.PAR)
    priority = table.priority[moves]
    first = jnp.where(priority[:, 0] != priority[:, 1], priority[:, 1] > priority[:, 0],
        jnp.where(speed[:, 0] != speed[:, 1], speed[:, 1] > speed[:, 0], draws[:, 0, 0] == 1)).astype(jnp.int32)
    placeholders = jnp.zeros_like(state.turn)
    moves_cancelled = jnp.zeros_like(state.done)
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
        confusion_damage = m.gen3_damage(level=_active(work.level, work, side), power=40,
            attack=own_stats[:, 1], defense=own_stats[:, 2],
            attack_stage=own_boosts[:, 0], defense_stage=own_boosts[:, 1],
            typeless=True, burned=_active(work.status, work, side) == m.BRN,
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
        is_struggle = move == table.struggle
        typ = jnp.where(is_struggle, 0, table.type[move])
        own_types = _active(work.types, work, side)
        foe_types = _active(work.types, work, foe)
        num = table.type_num[typ, foe_types[:, 0]] * table.type_num[typ, foe_types[:, 1]]
        den = table.type_den[typ, foe_types[:, 0]] * table.type_den[typ, foe_types[:, 1]]
        damage = m.gen3_damage(level=_active(work.level, work, side), power=table.power[move],
            attack=jnp.where(physical, attack_stats[:, 1], attack_stats[:, 3]),
            defense=jnp.where(physical, defense_stats[:, 2], defense_stats[:, 4]),
            attack_stage=jnp.where(physical, attack_boosts[:, 0], attack_boosts[:, 2]),
            defense_stage=jnp.where(physical, defense_boosts[:, 1], defense_boosts[:, 3]),
            physical=physical, critical=chance[:, 4] == 0,
            stab=jnp.any(own_types == typ[:, None], axis=1) & (typ != 0),
            typeless=typ == 0,
            burned=_active(work.status, work, side) == m.BRN,
            effectiveness_num=num, effectiveness_den=den, immune=num == 0, roll=85 + chance[:, 5])
        hit = m.accuracy_hit(table.accuracy[move], attack_boosts[:, 5], defense_boosts[:, 6],
                             chance[:, 6], always_hit=table.always_hit[move])
        hp = _active(work.hp, work, foe)
        dealt = jnp.where(acts & hit, jnp.minimum(hp, damage), 0)
        foe_hp = hp - dealt
        work = work._replace(hp=_set_active(work.hp, work, foe, foe_hp))
        # Gen 3 Struggle uses round(actual damage / 4), not max HP / 4.
        recoil = jnp.maximum(1, (dealt + 2) // 4)
        hp = _active(work.hp, work, side)
        hp = jnp.where(is_struggle & (dealt > 0), jnp.maximum(0, hp - recoil), hp)
        work = work._replace(hp=_set_active(work.hp, work, side, hp))
        if table.effects is not None:
            effects = table.effects
            # Self effects use the active actor, never the opposing target.
            hp = _active(work.hp, work, side)
            max_hp = _active(work.max_hp, work, side)
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
                & (_active(work.substitute_hp,work,foe) == 0))
            work = work._replace(status=_set_active(work.status,work,foe,
                jnp.where(apply_status,status,_active(work.status,work,foe))),
                status_counter=_set_active(work.status_counter,work,foe,
                    jnp.where(apply_status,jnp.where(status == m.SLP,2 + chance[:,7],0),
                              _active(work.status_counter,work,foe))))
        moves_cancelled |= acts & ((foe_hp == 0) | (hp == 0))
    # Residual scheduling beyond major statuses is not yet implemented.
    for side in range(2):
        battle_continues = jnp.all(jnp.any(work.hp > 0, axis=2), axis=1)
        residual = m.major_status_residual(_active(work.hp, work, side),
            _active(work.max_hp, work, side), _active(work.status, work, side),
            _active(work.status_counter, work, side), active=live & ~replacement & battle_continues)
        work = work._replace(hp=_set_active(work.hp, work, side, residual.hp),
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
