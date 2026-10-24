"""Prospective constructor field, not recovery of discarded historical gender.

Pinned Showdown prioritizes explicit M/F/N, then fixed species sex, then a fair
M/F draw (not genderRatio). Policy provenance and a dedicated seed/domain/root
address are bound by a fresh entry. All prior tensors/keys remain unchanged.
This does not implement Attract or Cute Charm, or grant correctness credit.
"""
from __future__ import annotations

import numpy as np

UNKNOWN, GENDERLESS, MALE, FEMALE, MIXED = -1, 0, 1, 2, 3
SCHEMA = "pokezero.gen3-gender-policy.v1"


def validate_policy(policy, species_vocabulary):
    if (not isinstance(species_vocabulary, (list, tuple))
            or any(not isinstance(v, str) or not v for v in species_vocabulary)
            or not isinstance(policy, dict) or policy.get("schema") != SCHEMA
            or policy.get("species") != list(species_vocabulary)
            or not species_vocabulary or species_vocabulary[0] != "_none"
            or len(set(species_vocabulary)) != len(species_vocabulary)):
        raise ValueError("gender policy requires the exact bound species vocabulary")
    codes = policy.get("codes")
    if (not isinstance(codes, list) or len(codes) != len(species_vocabulary)
            or codes[0] != UNKNOWN
            or any(type(v) is not int or v not in (-1, 0, 1, 2, 3) for v in codes)):
        raise ValueError("gender policy has missing/invalid species codes")
    return np.asarray(codes, np.int32)


def initialize_genders(state, policy, species_vocabulary, *, seed, domain,
                       root_ids, explicit_genders=None):
    """Initialize ONCE by immutable root occurrence/seat/party addresses.

    Legacy states stay UNKNOWN unless a caller explicitly invokes this fresh
    constructor. Explicit retained set overrides take precedence even for a
    fixed-sex species, matching Showdown. Unresolved selected forms refuse.
    No mapping, switch, phaze, Baton or resumed request may call this again.
    """
    import jax
    import jax.numpy as jnp
    if any(type(v) is not int or not 0 <= v < 2**32 for v in (seed, domain)):
        raise ValueError("explicit uint32 gender seed and separate domain required")
    codes = validate_policy(policy, species_vocabulary)
    species = np.asarray(state.species)
    old = np.asarray(state.gender)
    if (species.ndim != 3 or species.shape[1:] != (2, 6) or old.shape != species.shape
            or species.dtype.kind not in "iu" or old.dtype.kind not in "iu"
            or np.any(old != UNKNOWN) or np.any(species <= 0) or np.any(species >= len(codes))):
        raise ValueError("uninitialized complete gender state and valid species IDs required")
    roots = np.asarray(root_ids)
    if (roots.shape != (len(species),) or roots.dtype.kind not in "iu"
            or any(not 0 <= int(v) < 2**32 for v in roots)
            or len(np.unique(roots)) != len(roots)):
        raise ValueError("distinct explicit uint32 root occurrence IDs required")
    explicit = np.full(species.shape, UNKNOWN, np.int32) if explicit_genders is None else np.asarray(explicit_genders)
    if (explicit.shape != species.shape or explicit.dtype.kind not in "iu"
            or not np.isin(explicit, [-1, 0, 1, 2]).all()):
        raise ValueError("retained explicit genders must have shape[B,2,6] and codes-1/0/1/2")
    fixed = codes[species]
    if np.any((fixed == UNKNOWN) & (explicit == UNKNOWN)):
        raise ValueError("selected species/form gender is unresolved; never infer or redraw")
    base = jax.random.fold_in(jax.random.PRNGKey(seed), domain)
    def one(root):
        key = jax.random.fold_in(base, root)
        return jnp.stack([jnp.stack([jax.random.randint(jax.random.fold_in(
            jax.random.fold_in(key, side), slot), (), 1, 3, dtype=jnp.int32)
            for slot in range(6)]) for side in range(2)])
    drawn = jax.vmap(one)(jnp.asarray(roots, jnp.uint32))
    values = jnp.where(jnp.asarray(fixed) == MIXED, drawn, jnp.asarray(fixed))
    values = jnp.where(jnp.asarray(explicit) != UNKNOWN, jnp.asarray(explicit), values)
    return state._replace(gender=values.astype(jnp.int32))
