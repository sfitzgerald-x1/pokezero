"""Public ability/Transform context for device move and observation consumers.

Original-set reveals, currently borrowed abilities and the parser's active
Trace feature are distinct ledgers. This consumes resolved public addresses,
never private requests or a sampled engine world. Raw attribution, other
ability-triggered inference and complete event-fold integration remain required.
Original ability replacement must rebuild belief masks from current facts;
intersecting an old ability mask would incorrectly make replacement permanent.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from . import public_mon_state as mon
from .public_moves import _bool, _static_category


SCHEMA = "pokezero.public_abilities.v1"


class AbilityTables(NamedTuple):
    predicate: jax.Array
    category: jax.Array
    pressure_id: jax.Array
    source_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class AbilityCatalog:
    names: tuple[str, ...]
    ids: Mapping[str, int]
    device: AbilityTables

    def ability_id(self, name):
        key = normalize_id(name)
        if not key or key not in self.ids:
            raise ValueError("ability outside frozen public namespace")
        return self.ids[key]


def compile_abilities(source, vocabulary, *, ability_names=()):
    """Bind public ability IDs separately from set predicates and policy rows.

    Supply static-dex extra names for off-source abilities. Unknown nonempty
    wire names refuse lookup rather than silently aliasing the unknown row.
    """
    names = ("",) + tuple(sorted(({normalize_id(n) for n in ability_names} |
        {normalize_id(v.ability) for rows in source.variants for v in rows if v.ability} |
        {"pressure", "trace"}) - {""}))
    ids = {name: i for i, name in enumerate(names) if i}
    predicate = [source.predicate_id("ability", name) if name else 0 for name in names]
    category = [_static_category(vocabulary, f"ability:{name}") if name else 0 for name in names]
    payload = dict(schema=SCHEMA, names=names, predicate=predicate, category=category,
        source_binding=np.asarray(source.device.binding).tolist(),
        vocabulary_tokens=vocabulary.tokens, vocabulary_aliases=dict(vocabulary.aliases),
        oov_buckets=vocabulary.oov_buckets)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest, dtype=">u4").astype(np.uint32))
    return AbilityCatalog(names, MappingProxyType(ids), AbilityTables(jnp.asarray(predicate, jnp.int32),
        jnp.asarray(category, jnp.int32), jnp.asarray(ids["pressure"], jnp.int32), source.device.binding, binding))


class AbilityState(NamedTuple):
    binding: jax.Array             # [B,8]
    original: jax.Array            # [B,2,6], original-set public reveal only
    running: jax.Array             # [B,2,6], borrowed override, 0 absent
    transformed: jax.Array         # [B,2,6]
    transform_species: jax.Array   # [B,2,6], original public donor dex identity
    traced: jax.Array              # [B,2], parser feature, independently cleared


class RevealUpdate(NamedTuple):
    state: AbilityState
    admitted: jax.Array
    changed: jax.Array
    conflict: jax.Array


def initialize(batch, tables):
    if type(batch) is not int or batch <= 0:
        raise ValueError("positive public ability batch required")
    shape = (batch, 2, 6)
    z = lambda: jnp.zeros(shape, jnp.int32)
    return AbilityState(jnp.broadcast_to(tables.binding, (batch, 8)), z(), z(),
        jnp.zeros(shape, jnp.bool_), z(), jnp.zeros((batch, 2), jnp.int32))


def _address(state, mons, tables, side, slot, encounter, valid):
    row, seat, target, ok = mon._address(mons, side, slot)
    encounter = mon._int32(encounter)
    ok &= (encounter > 0) & (mons.encounter[row, seat, target] == encounter)
    ok &= jnp.all(state.binding == tables.binding, axis=-1) & _bool(valid)
    return (row, seat, target), ok


def reveal(state, mons, tables, *, side, slot, encounter, ability_id, authoritative=False, valid=True):
    """Direct ability lines overwrite; proc claims preserve an earlier original.

    Proc claims while borrowing confirm current mechanics and do not reveal an
    original ability. Return conflict bits for the caller's evidence receipt.
    """
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    ability_id = mon._int32(ability_id)
    authoritative = _bool(authoritative)
    ok &= (ability_id > 0) & (ability_id < tables.predicate.shape[0])
    old = state.original[address]
    ignored = ~authoritative & (state.running[address] > 0)
    conflict = ok & ~authoritative & ~ignored & (old > 0) & (old != ability_id)
    write = ok & ~ignored & (authoritative | (old == 0))
    result = state._replace(original=mon._write(state.original, address, ability_id, write))
    return RevealUpdate(result, ok, write & (old != ability_id), conflict)


def trace_copy(state, mons, tables, *, side, slot, encounter, ability_id,
               donor_side=-1, donor_slot=-1, donor_encounter=0, parser_feature=True, valid=True):
    """Consume a public Trace announcement with an explicitly resolved [of].

    Missing [of] changes only the parser Trace feature. A present donor receives
    an authoritative original reveal unless it is itself borrowing/Transformed.
    The tracer receives the running override, never an original-set reveal.
    """
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    ability_id = mon._int32(ability_id)
    donor_side, donor_slot, donor_encounter = (mon._int32(x) for x in (donor_side, donor_slot, donor_encounter))
    missing = (donor_side == -1) & (donor_slot == -1) & (donor_encounter == 0)
    donor, donor_ok = _address(state, mons, tables, donor_side, donor_slot, donor_encounter, True)
    ok &= (ability_id > 0) & (ability_id < tables.predicate.shape[0]) & (missing | donor_ok)
    row, seat, _ = address
    active = mons.active[row, seat] == address[2]
    # The parser's feature is keyed by active side, not persistent encounter.
    feature = ok & active & _bool(parser_feature)
    copying = ok & ~missing
    # Python installs the tracer's override before checking the donor. Even a
    # malformed self-[of] line therefore cannot reveal its original ability.
    same_encounter = (address[1] == donor[1]) & (address[2] == donor[2])
    original = copying & ~same_encounter & ~state.transformed[donor] & (state.running[donor] == 0)
    return state._replace(
        original=mon._write(state.original, donor, ability_id, original),
        running=mon._write(state.running, address, ability_id, copying),
        traced=mon._write(state.traced, (row, seat), ability_id, feature)), ok


def transform(state, mons, tables, *, side, slot, encounter,
              donor_side, donor_slot, donor_encounter, valid=True):
    """Copy a publicly tracked donor's current ability and original species.

    The raw adapter must resolve the source parser's nickname/forme rule first.
    Unknown donor ability retains an existing running override, as Python does.
    """
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    donor, donor_ok = _address(state, mons, tables, donor_side, donor_slot, donor_encounter, True)
    ok &= donor_ok & (mons.species[donor] > 0)
    copied = jnp.where(state.running[donor] > 0, state.running[donor], state.original[donor])
    return state._replace(
        running=mon._write(state.running, address, copied, ok & (copied > 0)),
        transformed=mon._write(state.transformed, address, True, ok),
        transform_species=mon._write(state.transform_species, address, mons.species[donor], ok)), ok


def arrive(state, mons, tables, *, side, slot, encounter, valid=True):
    """Clear all side Transform marks and only the incoming running override.

    Call after the mon arrival consumer commits the incoming public encounter.
    Python clears side Transform on switch-out, but retains inactive borrowed
    overrides until that particular encounter returns. Trace feature is active.
    """
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    row, seat, target = address
    ok &= mons.active[row, seat] == target
    return state._replace(
        running=mon._write(state.running, address, 0, ok),
        transformed=mon._write(state.transformed, (row, seat), False, ok),
        transform_species=mon._write(state.transform_species, (row, seat), 0, ok),
        traced=mon._write(state.traced, (row, seat), 0, ok)), ok


def faint(state, mons, tables, *, side, slot, encounter, valid=True):
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    return state._replace(transformed=mon._write(state.transformed, address, False, ok),
        transform_species=mon._write(state.transform_species, address, 0, ok)), ok


def move_context(state, mons, tables, *, side, slot, encounter, valid=True):
    """Return actor Transform and opposite active public Pressure, plus admission."""
    address, ok = _address(state, mons, tables, side, slot, encounter, valid)
    row, seat, target = address
    other = 1 - seat
    other_active = mons.active[row, other]
    safe_active = jnp.clip(other_active, 0, 5)
    foe = row, other, safe_active
    ability = jnp.where(state.running[foe] > 0, state.running[foe], state.original[foe])
    live = (other_active >= 0) & ~mons.fainted[foe]
    ok &= (mons.active[row, seat] == target) & ~mons.fainted[address]
    return state.transformed[address] & ok, live & (ability == tables.pressure_id) & ok, ok


def traced_categories(state, tables):
    ids = state.traced
    ok = jnp.all(state.binding == tables.binding, axis=-1) & jnp.all(
        (ids >= 0) & (ids < tables.category.shape[0]), axis=-1)
    return jnp.where(ok[:, None], tables.category[jnp.clip(ids, 0, tables.category.shape[0] - 1)], 0), ok
