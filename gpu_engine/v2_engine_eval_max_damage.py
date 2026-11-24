"""Device form of the PROGRESS MaxDamage heuristic."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import recharging, usable_moves


class Tables(NamedTuple):
    power: jax.Array
    accuracy: jax.Array
    priority: jax.Array
    recoil: jax.Array
    selfdestruct: jax.Array
    stab_twice: jax.Array
    effectiveness_fourfold: jax.Array


def compile_tables(vocabulary, dex):
    """Use the same static dex facts as the Showdown panel's policy."""
    species = [dex.species_info(name) for name in vocabulary['species']]
    types = [mon.types if mon is not None else () for mon in species]
    # The Showdown policy scores request IDs. All typed Hidden Power moves
    # appear there as the generic move, even when the engine knows their type.
    moves = [dex.move_info('hiddenpower' if name.startswith('hiddenpower') else name)
             for name in vocabulary['moves']]
    move_types = {move.type for move in moves if move is not None}
    # Extra integer precision keeps genuine ties exact even after recoil and
    # self-KO adjustments. Gen 3 dual typing has denominators of at most four.
    stab = {name: [3 if name in pair else 2 for pair in types] for name in move_types}
    effects = {name: [round(4 * dex.effectiveness(name, pair)) for pair in types]
               for name in move_types}
    neutral_stab, neutral_effect = [2] * len(types), [4] * len(types)
    return Tables(
        jnp.asarray([move.base_power if move and move.gen3_category != 'Status' else 0 for move in moves], jnp.int32),
        jnp.asarray([max(50, min(100, round(move.accuracy or 100))) if move else 100 for move in moves], jnp.int32),
        jnp.asarray([max(move.priority, 0) if move else 0 for move in moves], jnp.int32),
        jnp.asarray([bool(move and move.recoil) for move in moves]),
        jnp.asarray([bool(move and move.selfdestruct) for move in moves]),
        jnp.asarray([stab[move.type] if move else neutral_stab for move in moves], jnp.int32),
        jnp.asarray([effects[move.type] if move else neutral_effect for move in moves], jnp.int32),
    )


def scores(tables, moves, owner_species, foe_species, hp, max_hp):
    """Return heuristic move scores in units of 1/80,000."""
    power = tables.power[moves]
    effectiveness = tables.effectiveness_fourfold[moves, foe_species[:, None]]
    stab = tables.stab_twice[moves, owner_species[:, None]]
    result = power * effectiveness * stab * tables.accuracy[moves]
    result += tables.priority[moves] * (8 * 800)
    below = (hp * 20 < max_hp * 7)[:, None]
    above = (hp * 20 > max_hp * 7)[:, None]
    result -= jnp.where(tables.recoil[moves] & below, 15 * 800, 0)
    result *= jnp.where(tables.selfdestruct[moves] & above, 35, 100)
    return jnp.where((power > 0) & (effectiveness > 0), result, 0)


def opponent(tables, struggle):
    def choose(params, state, public, sides, legal, seeds, requests):
        rows = jnp.arange(len(sides))
        active = state.active[rows, sides]
        foe_active = state.active[rows, 1 - sides]
        moves = state.moves[rows, sides, active]
        charge = state.charge_move[rows, sides, active, None]
        usable = jnp.any(usable_moves(state)[rows, sides], axis=-1, keepdims=True)
        moves = jnp.where(charge > 0, charge, jnp.where(usable, moves, struggle))
        move_scores = scores(tables, moves, state.species[rows, sides, active],
                             state.species[rows, 1 - sides, foe_active],
                             state.hp[rows, sides, active], state.max_hp[rows, sides, active])
        move_scores = jnp.where(recharging(state)[rows, sides, None], 80_000, move_scores)
        legal = legal[rows, sides]
        legal_moves = legal[:, :4]
        best = jnp.max(jnp.where(legal_moves, move_scores, jnp.iinfo(jnp.int32).min), axis=1)
        tied = jnp.concatenate((legal_moves & (move_scores == best[:, None]),
                                legal[:, 4:] & ~jnp.any(legal_moves, axis=1, keepdims=True)), axis=1)
        valid = jnp.any(tied, axis=1)

        def sample(seed, side, request, mask):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), side), request)
            return jax.random.categorical(key, jnp.where(mask, 0., -jnp.inf))

        actions = jax.vmap(sample)(seeds, sides, requests, tied | ~valid[:, None])
        return actions.astype(jnp.int32), valid

    return choose
