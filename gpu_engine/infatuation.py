"""Gen 3 Attract and Cute Charm (Showdown f76228a1).

Singles source identity is the opposing active. Departure clears the relation
before installing a replacement; Baton Pass never copies ATTRACT.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import ATTRACT


class InfatuationTable(NamedTuple):
    attract: jax.Array
    cute_charm: jax.Array
    oblivious: jax.Array
    mental_herb: jax.Array


def from_metadata(metadata):
    vocab = metadata['vocabulary']
    flags = lambda kind, name: jnp.asarray([value == name for value in vocab[kind]])
    return InfatuationTable(flags('moves', 'attract'), flags('abilities', 'cutecharm'),
        flags('abilities', 'oblivious'), flags('items', 'mentalherb'))


def apply(state, target, table, *, eligible):
    """addVolatile: living recipient, absent volatile, immunity, opposite M/F.

    The caller owns the hit/chance gate. The source may already have zero HP:
    DamagingHit precedes faintMessages, so that activation still reveals its
    ability before Update removes the relation.
    """
    rows = jnp.arange(state.hp.shape[0]); source = 1-target
    own, foe = state.active[rows, target], state.active[rows, source]
    old = state.volatile[rows, target, own]
    gender, other = state.gender[rows, target, own], state.gender[rows, source, foe]
    opposite = ((gender == 1) & (other == 2)) | ((gender == 2) & (other == 1))
    applied = (jnp.asarray(eligible) & (state.hp[rows, target, own] > 0)
        & ((old & jnp.uint32(ATTRACT)) == 0) & opposite
        & ~table.oblivious[state.ability[rows, target, own]])
    return state._replace(volatile=state.volatile.at[rows, target, own].set(
        jnp.where(applied, old | jnp.uint32(ATTRACT), old))), applied


def source_departed(state, source, *, eligible):
    rows = jnp.arange(state.hp.shape[0]); target = 1-source
    own = state.active[rows, target]; old = state.volatile[rows, target, own]
    return state._replace(volatile=state.volatile.at[rows, target, own].set(
        jnp.where(eligible, old & ~jnp.uint32(ATTRACT), old)))


def clear_fainted(state, *, eligible):
    # Gen3's replacement request (and terminal return) precedes Update. Keep
    # the survivor's bit at that boundary, even when its source just fainted;
    # source_departed clears it when the replacement actually installs.
    clear = jnp.asarray(eligible)[:, None, None] & (state.hp <= 0)
    return state._replace(volatile=jnp.where(clear,
        state.volatile & ~jnp.uint32(ATTRACT), state.volatile))


def update(state, table, *, eligible, _device_public=None):
    """Oblivious and Mental Herb's inherited Gen 3 Update cures; never Lum."""
    from . import engine_device_public_hooks as public
    rows = jnp.arange(state.hp.shape[0])
    for side in range(2):
        own = state.active[rows, side]; old = state.volatile[rows, side, own]
        held = state.item[rows, side, own]
        active = (jnp.asarray(eligible) & (state.hp[rows, side, own] > 0)
            & ((old & jnp.uint32(ATTRACT)) != 0))
        ability_cure = active & table.oblivious[state.ability[rows, side, own]]
        # Ability Update runs before the item's Update; don't consume a herb
        # when Oblivious has already removed the volatile.
        herb = active & ~ability_cure & table.mental_herb[held]
        before = state
        state = state._replace(
            volatile=state.volatile.at[rows, side, own].set(jnp.where(
                ability_cure | herb, old & ~jnp.uint32(ATTRACT), old)),
            item=state.item.at[rows, side, own].set(jnp.where(herb, 0, held)))
        if _device_public is not None:
            public.reveal_ability(_device_public, state, side, ability_cure)
            public.item_changes(_device_public, before, state, herb)
    return state
