"""First lockstep turn integration; explicitly incomplete, not covered gameplay.

All move effects are still placeholders: damaging moves use their static type
and power; status moves are no-ops. Items, abilities, weather, volatiles and
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
from .state import BattleState, bench_indices, empty_state, legal_actions


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
        jnp.asarray(metadata["vocabulary"]["moves"].index("struggle"), jnp.int32))


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
    PP, static single-hit damage, major-status residuals and party exhaustion.
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
    draws = jax.vmap(lambda key: jax.random.randint(key, (2, 7), 0,
        jnp.array([2, 100, 100, 100, 16, 16, 100], jnp.int32)))(splits[:, 1])
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
    for order in range(2):
        side = jnp.where(order == 0, first, 1 - first)
        foe = 1 - side
        move = moves[rows, side]
        chance = draws[rows, side]
        eligible = live & ~replacement & ~switches[rows, side] & (_active(work.hp, work, side) > 0)
        before = m.before_move(_active(work.status, work, side), _active(work.status_counter, work, side),
            _active(work.confusion_counter, work, side), thaw_roll=chance[:, 1],
            paralysis_roll=chance[:, 2], confusion_roll=chance[:, 3])
        for name, value in (("status", before.status), ("status_counter", before.counter),
                            ("confusion_counter", before.confusion_counter)):
            current = getattr(work, name)
            work = work._replace(**{name: _set_active(current, work, side,
                jnp.where(eligible, value, _active(current, work, side)))})
        acts = eligible & before.can_act
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
        typ = table.type[move]
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
            burned=_active(work.status, work, side) == m.BRN,
            effectiveness_num=num, effectiveness_den=den, immune=num == 0, roll=85 + chance[:, 5])
        hit = m.accuracy_hit(table.accuracy[move], attack_boosts[:, 5], defense_boosts[:, 6],
                             chance[:, 6], always_hit=table.always_hit[move])
        hp = _active(work.hp, work, foe)
        hp = jnp.where(acts & hit, jnp.maximum(0, hp - damage), hp)
        work = work._replace(hp=_set_active(work.hp, work, foe, hp))
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
    work = jax.tree.map(lambda old, new: jnp.where(live.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), state, work)
    return StepResult(work, invalid, placeholders)
