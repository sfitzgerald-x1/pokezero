"""Compose public move and ability consumers without losing replaceable facts.

The non-ability hard mask is retained separately. Authoritative original ability
replacements reapply the *current* ability to that basis, allowing a corrected
claim to recover variants while preserving moves, exclusions and standing pins.
This is incremental consumer composition, not complete raw protocol coverage.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import public_abilities as ability, public_moves as move, public_mon_state as mon


class ContextTables(NamedTuple):
    abilities: ability.AbilityTables
    moves: move.MoveTables
    ability_predicate: jax.Array


class ContextState(NamedTuple):
    abilities: ability.AbilityState
    moves: move.MoveState
    base_hard: jax.Array
    base_has_evidence: jax.Array


def compile_context(source, abilities, moves):
    for binding in (abilities.device.source_binding, moves.device.source_binding):
        if not np.array_equal(np.asarray(binding), np.asarray(source.device.binding)):
            raise ValueError("public context requires the same frozen source")
    flags = np.zeros(source.device.predicates.shape[0], bool)
    for (kind, _), index in source.predicate_ids.items():
        if kind == "ability": flags[index] = True
    return ContextTables(abilities.device, moves.device, jnp.asarray(flags))


def initialize(beliefs, tables, *, species_binding=None):
    """Start before evidence ingestion; importing an ambiguous old mask refuses."""
    if beliefs.species.ndim != 3 or beliefs.species.shape[1:] != (2, 6):
        raise ValueError("context requires B by two observers by six encounters")
    if np.asarray(beliefs.has_evidence).any() or np.asarray(beliefs.pin_active).any():
        raise ValueError("initialize public context before evidence ingestion")
    batch = beliefs.species.shape[0]
    return ContextState(ability.initialize(batch, tables.abilities,species_binding=species_binding), move.initialize(batch, tables.moves),
        beliefs.hard, beliefs.has_evidence)


def basis(context, beliefs):
    # Pin state and conflict receipts may have advanced during materialization;
    # only hard reveal/exclusion facts need a separate replaceable-ability basis.
    return beliefs._replace(hard=context.base_hard, has_evidence=context.base_has_evidence)


def rebuild(context, base, source_tables, tables):
    ids = context.abilities.original[:, ::-1, :]
    valid_ids = (ids >= 0) & (ids < tables.abilities.predicate.shape[0])
    safe = jnp.clip(ids, 0, tables.abilities.predicate.shape[0] - 1)
    predicate = tables.abilities.predicate[safe]
    species = jnp.clip(base.species, 0, source_tables.pool.shape[0] - 1)
    matched = source_tables.predicates[predicate, species]
    known = ids > 0
    ok = jnp.all(valid_ids, axis=(1, 2)) & jnp.all(
        context.abilities.binding == tables.abilities.binding, axis=-1) & jnp.all(
        context.moves.binding == tables.moves.binding, axis=-1) & jnp.all(
        tables.abilities.source_binding == source_tables.binding) & jnp.all(
        tables.moves.source_binding == source_tables.binding)
    return base._replace(hard=base.hard & jnp.where(known[..., None], matched, jnp.uint32(0xffffffff)),
        has_evidence=base.has_evidence | known), ok


def consume(context, base, mons, source_tables, tables, row, valid, *,
            move_kind, ability_kind, trace_kind, transform_kind, arrival, faint):
    """One resolved typed event per battle; the outer fold commits atomically."""
    k, actor_side, actor_slot, side, slot = row[:, 0], row[:, 4], row[:, 5], row[:, 6], row[:, 7]
    data = row[:, 8:]
    a, m = context.abilities, context.moves
    a, aa = ability.arrive(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & arrival)
    m, ma = move.arrive(m, tables.moves, side=side, valid=valid & arrival)
    a, fa = ability.faint(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0], valid=valid & faint)
    is_move, is_ability, is_trace, is_transform = (k == x for x in (move_kind, ability_kind, trace_kind, transform_kind))
    transformed, pressure, ca = ability.move_context(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], valid=valid & is_move)
    update = move.record_move(m, base, mons, tables.moves, source_tables,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], move_id=data[:, 1],
        caller=data[:, 2], has_from=data[:, 3] == 1, transformed=transformed,
        opposing_pressure=pressure, wire_target_side=side, valid=valid & is_move & ca)
    m, base = update.state, update.beliefs
    rev = ability.reveal(a, mons, tables.abilities, side=side, slot=slot, encounter=data[:, 0],
        ability_id=data[:, 1], authoritative=data[:, 2] == 1, valid=valid & is_ability)
    a = rev.state
    a, ta = ability.trace_copy(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], ability_id=data[:, 1],
        donor_side=side, donor_slot=slot, donor_encounter=data[:, 2],
        parser_feature=data[:, 3] == 1, valid=valid & is_trace)
    a, tr = ability.transform(a, mons, tables.abilities,
        side=actor_side, slot=actor_slot, encounter=data[:, 0], donor_side=side, donor_slot=slot,
        donor_encounter=data[:, 1], valid=valid & is_transform)
    accepted = jnp.where(arrival, aa & ma, jnp.where(faint, fa, jnp.where(is_move, update.admitted,
        jnp.where(is_ability, rev.admitted, jnp.where(is_trace, ta, jnp.where(is_transform, tr, True))))))
    return context._replace(abilities=a, moves=m), base, accepted | ~valid
