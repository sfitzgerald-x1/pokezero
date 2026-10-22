"""Batched structure-of-arrays state and nine-action layout for the spike.

No battle transition is implemented here. Empty states are storage fixtures,
not generated teams, and cannot supply games-per-second measurements.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

FLINCH = 1  # volatile bit 0; cleared at the end of each ordinary turn
LEECH_SEED = 4  # volatile bit 2; opposite singles slot receives the drain
ROOTED = 8  # represented Ingrain condition; the move itself remains pending


class BattleState(NamedTuple):
    species: jax.Array             # [B,2,6]
    level: jax.Array               # [B,2,6]
    hidden_power_type: jax.Array   # [B,2,6], Pokemon's derived type, not alias
    hidden_power_power: jax.Array  # [B,2,6], derived from accepted pool IVs
    happiness: jax.Array           # [B,2,6], Showdown team-object default 255
    types: jax.Array               # [B,2,6,2]; type IDs require the pool vocabulary
    stats: jax.Array               # [B,2,6,6]: HP,Atk,Def,SpA,SpD,Spe
    hp: jax.Array                  # [B,2,6]
    max_hp: jax.Array              # [B,2,6]
    status: jax.Array              # [B,2,6]
    status_counter: jax.Array      # [B,2,6]
    sleep_skipped: jax.Array       # [B,2,6], Gen3 Sleep Talk turns restored on entry
    moves: jax.Array               # [B,2,6,4]
    pp: jax.Array                  # [B,2,6,4]
    disabled: jax.Array            # [B,2,6,4]
    choice_move: jax.Array         # [B,2,6], locked move ID; zero means unlocked
    item: jax.Array                # [B,2,6]
    ability: jax.Array             # [B,2,6]
    boosts: jax.Array              # [B,2,6,7]: Atk,Def,SpA,SpD,Spe,Acc,Eva
    volatile: jax.Array            # [B,2,6], uint32 bitmask
    confusion_counter: jax.Array   # [B,2,6]
    protect_counter: jax.Array     # [B,2,6]
    substitute_hp: jax.Array       # [B,2,6]
    active: jax.Array              # [B,2]
    entry_pending: jax.Array       # [B,2], initial Ability Start not yet resolved
    active_turns: jax.Array        # [B,2], ordinary turns begun since last entry
    trapped: jax.Array             # [B,2], resolved trap state, not ability logic
    trap_source: jax.Array         # [B,2,6], opposing party index; -1 unlinked
    forced_switch: jax.Array       # [B,2]
    baton_pass: jax.Array          # [B,2], living direct-Baton switch cause
    queued_actor: jax.Array        # [B], saved opposing action; -1 means none
    queued_actions: jax.Array      # [B,2], original ordinary move-slot requests
    queued_stall: jax.Array        # [B,2], old counters for the saved action
    pending_residual: jax.Array    # [B], suspended action turn waits for entry
    residual_draws: jax.Array      # [B,2,12], original ordinary-turn chance block
    phazing_draws: jax.Array       # [B,2], fixed uniform [0,60) bench-choice rolls
    secondary_draws: jax.Array     # [B,2], fixed uniform [0,100) target-secondary rolls
    sleep_talk_draws: jax.Array    # [B,2], fixed uniform [0,60) called-slot rolls
    replacement_count: jax.Array   # [B], tie stream within a suspended turn
    spikes: jax.Array              # [B,2]
    wish_counter: jax.Array        # [B,2], slot condition; persists across switches
    weather: jax.Array             # [B]
    weather_counter: jax.Array     # [B]
    turn: jax.Array                # [B]
    keys: jax.Array                # [B,2] legacy uint32 PRNG keys, one per battle
    done: jax.Array                # [B]
    winner: jax.Array              # [B], -1 undecided,0/1 seat,2 tie


def empty_state(batch_size: int, *, seed: int = 0) -> BattleState:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    b = batch_size
    mon = lambda: jnp.zeros((b, 2, 6), dtype=jnp.int32)
    side = lambda: jnp.zeros((b, 2), dtype=jnp.int32)
    battle = lambda: jnp.zeros((b,), dtype=jnp.int32)
    # fold_in keeps each battle's stream stable when the batch size changes.
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        jax.random.PRNGKey(seed), jnp.arange(b, dtype=jnp.uint32))
    return BattleState(
        species=mon(), level=mon(), hidden_power_type=mon(),
        hidden_power_power=jnp.full((b,2,6),70,jnp.int32),
        happiness=jnp.full((b,2,6),255,jnp.int32), types=jnp.zeros((b, 2, 6, 2), jnp.int32),
        stats=jnp.zeros((b, 2, 6, 6), jnp.int32), hp=mon(), max_hp=mon(),
        status=mon(), status_counter=mon(), sleep_skipped=mon(), moves=jnp.zeros((b, 2, 6, 4), jnp.int32),
        pp=jnp.zeros((b, 2, 6, 4), jnp.int32), disabled=jnp.zeros((b, 2, 6, 4), jnp.bool_),
        choice_move=mon(),
        item=mon(), ability=mon(), boosts=jnp.zeros((b, 2, 6, 7), jnp.int32),
        volatile=jnp.zeros((b, 2, 6), jnp.uint32), confusion_counter=mon(),
        protect_counter=mon(), substitute_hp=mon(), active=side(),
        entry_pending=jnp.zeros((b,2),jnp.bool_),
        active_turns=side(),
        trapped=jnp.zeros((b, 2), jnp.bool_), forced_switch=jnp.zeros((b, 2), jnp.bool_),
        trap_source=jnp.full((b,2,6),-1,jnp.int32),
        baton_pass=jnp.zeros((b,2),jnp.bool_),queued_actor=jnp.full((b,),-1,jnp.int32),
        queued_actions=side(),queued_stall=side(),
        pending_residual=jnp.zeros((b,),jnp.bool_),
        residual_draws=jnp.zeros((b,2,12),jnp.int32),replacement_count=battle(),
        phazing_draws=side(), secondary_draws=side(), sleep_talk_draws=side(),
        spikes=side(), wish_counter=side(),weather=battle(), weather_counter=battle(), turn=battle(),
        keys=keys, done=jnp.ones((b,), jnp.bool_), winner=jnp.full((b,), -1, jnp.int32))


def bench_indices(active: jax.Array) -> jax.Array:
    """Action4..8 enumerate party slots in order, omitting the active slot."""
    slots = jnp.arange(5, dtype=jnp.int32)
    return slots + (slots >= active[..., None])


def legal_actions(state: BattleState) -> jax.Array:
    """PP/disabled/resolved-trap/faint mask; other move restrictions are pending.

    If no move is usable, move-slot0 means synthetic Struggle. A transition must
    distinguish that case from selecting the ordinary move in slot0.
    """
    active_hp = jnp.take_along_axis(state.hp, state.active[..., None], axis=2)[..., 0]
    pp = jnp.take_along_axis(state.pp, state.active[..., None, None], axis=2)[..., 0, :]
    disabled = jnp.take_along_axis(state.disabled, state.active[..., None, None], axis=2)[..., 0, :]
    move_ids = jnp.take_along_axis(state.moves,state.active[...,None,None],axis=2)[...,0,:]
    choice = jnp.take_along_axis(state.choice_move,state.active[...,None],axis=2)[...,0]
    moves = (pp > 0) & ~disabled & ((choice[...,None] == 0) | (move_ids == choice[...,None]))
    moves = moves.at[..., 0].set(moves[..., 0] | ~jnp.any(moves, axis=-1))
    forced = state.forced_switch | (active_hp <= 0)
    moves &= ~forced[..., None]
    bench_hp = jnp.take_along_axis(state.hp, bench_indices(state.active), axis=2)
    active_volatile=jnp.take_along_axis(state.volatile,state.active[...,None],axis=2)[...,0]
    rooted=(active_volatile & jnp.uint32(ROOTED))!=0
    source=jnp.take_along_axis(state.trap_source,state.active[...,None],axis=2)[...,0]
    linked=(source>=0) & (source==state.active[:,::-1]) & (active_hp[:,::-1]>0)
    # Gen 3 has no later-generation Ghost exemption for these move traps.
    switches = (bench_hp > 0) & (~(state.trapped | rooted | linked) | forced)[..., None]
    mask = jnp.concatenate((moves, switches), axis=-1)
    return mask & ~state.done[:, None, None]
