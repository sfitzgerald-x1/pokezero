"""Public candidate-set inputs for the device observation encoder.

This is the set-source part of B2, not a complete observation encoder. Static
tables are compiled from the Python reference's source; device updates accept
only public reveal/exclusion predicates or explicitly supplied evidence pins.
No physical BattleState, private opponent request, or sampled world is read.
"""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.belief import variant_identity
from pokezero.randbat import canonical_gen3_randbat_species_id, _normalize_id


SCHEMA = "pokezero.public_belief.v1"
RESOLVED, UNKNOWN_SPECIES, INCONSISTENT_REVEALS, NO_SOURCE = range(4)
FACT_KINDS = ("move", "ability", "item")
WORDS = 2


class DeviceTables(NamedTuple):
    pool: jax.Array                 # [species including unknown, 2] uint32
    predicates: jax.Array           # [predicate including unknown, species, 2]
    variant_valid: jax.Array        # [species, maximum variants] bool
    binding: jax.Array              # [8] uint32 source/table contract SHA256


class BeliefState(NamedTuple):
    species: jax.Array              # arbitrary leading shape (normally B,2,6)
    source_available: jax.Array     # unavailable source != unknown species
    hard: jax.Array                 # [..., 2] raw reveal/exclusion intersection
    pin: jax.Array                  # [..., 2] separately retained damage evidence
    pin_active: jax.Array
    has_evidence: jax.Array
    conflicts: jax.Array


class BeliefSummary(NamedTuple):
    effective: jax.Array            # [..., 2]; fallback is distinct from hard
    count: jax.Array
    uncertainty: jax.Array
    reason: jax.Array
    pin_conflict: jax.Array


@dataclass(frozen=True)
class SourceTables:
    """Host contract and source-order identities accompanying a device table.

    Predicate IDs have their own namespace, unrelated to engine IDs or policy
    vocabulary rows. Zero means an unknown *nonempty* fact, never a no-op.
    Empty facts must not be submitted. Species zero means unknown species;
    source availability is a separate state field.
    """

    source_hash: str
    species_names: tuple[str, ...]
    species_ids: Mapping[str, int]
    predicate_ids: Mapping[tuple[str, str], int]
    variants: tuple[tuple, ...]
    device: DeviceTables

    def species_id(self, species: str) -> int:
        return self.species_ids.get(canonical_gen3_randbat_species_id(species), 0)

    def predicate_id(self, kind: str, value: str) -> int:
        if kind not in FACT_KINDS:
            raise ValueError(f"unknown public fact namespace: {kind}")
        normalized = _normalize_id(value)
        if not normalized:
            raise ValueError("empty public fact is not reveal evidence")
        return self.predicate_ids.get((kind, normalized), 0)

    def pin_mask(self, species: str, survivors) -> np.ndarray:
        """Map exact Python variant identities, including duplicate identities.

        This consumes a public evidence producer's survivor mappings, not a
        hidden team's true set. Nonempty unmappable identities fail admission:
        Python can retain those identities, but this fixed source table cannot.
        Silently dropping them would change later standing-pin intersections.
        """
        identities = {variant_identity(v) for v in survivors}
        rows = self.variants[self.species_id(species)]
        known = {variant_identity(v.to_summary()) for v in rows}
        if identities - known:
            raise ValueError("evidence pin has identities outside the bound public source")
        return _pack([variant_identity(v.to_summary()) in identities for v in rows])


def _pack(bits) -> np.ndarray:
    result = np.zeros(WORDS, np.uint32)
    for index, bit in enumerate(bits):
        if bit:
            result[index // 32] |= np.uint32(1 << (index % 32))
    return result


def compile_source(source) -> SourceTables:
    """Freeze predicates using Gen3RandbatVariant.matches, including Hidden Power.

    There is no independent second implementation of source normalization or
    reveal matching here. More than 64 variants is a new shape admission, not
    lossy truncation. Variant order within each species is left untouched.
    """
    source_hash = source.metadata.source_hash
    if not source_hash:
        raise ValueError("public belief tables require a source hash")
    universes = tuple(source.universes.values())
    names = ("",) + tuple(u.species for u in universes)
    species_ids = {canonical_gen3_randbat_species_id(n): i for i, n in enumerate(names) if i}
    if len(species_ids) != len(universes):
        raise ValueError("duplicate normalized species in public source")
    variants = ((),) + tuple(u.variants for u in universes)
    maximum = max((len(v) for v in variants), default=0)
    if not 0 < maximum <= WORDS * 32:
        raise ValueError("public source needs between 1 and 64 variants per species")
    keys = {("move", "hiddenpower")}
    for rows in variants:
        for v in rows:
            keys.update(("move", _normalize_id(m)) for m in v.moves if _normalize_id(m))
            if v.ability:
                keys.add(("ability", _normalize_id(v.ability)))
            if v.item:
                keys.add(("item", _normalize_id(v.item)))
    ordered = tuple(sorted(keys))
    predicate_ids = {key: index + 1 for index, key in enumerate(ordered)}
    pool = np.stack([_pack([True] * len(rows)) for rows in variants])
    predicates = np.zeros((len(ordered) + 1, len(names), WORDS), np.uint32)
    kwargs = {"move": "revealed_moves", "ability": "revealed_ability", "item": "revealed_item"}
    for key, predicate_id in predicate_ids.items():
        kind, value = key
        arguments = {kwargs[kind]: (value,) if kind == "move" else value}
        for species_id, rows in enumerate(variants):
            predicates[predicate_id, species_id] = _pack([v.matches(**arguments) for v in rows])
    valid = np.arange(maximum)[None, :] < np.asarray([len(v) for v in variants])[:, None]
    contract = json.dumps({"schema": SCHEMA, "source_hash": source_hash, "species": names,
        "predicates": ordered, "variants": [[v.to_summary() for v in rows] for rows in variants]},
        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(contract)
    # Bind the generated predicates too: a matching-rule implementation can
    # change table contents without changing the variant summaries themselves.
    for name, array in (("pool", pool), ("predicates", predicates), ("valid", valid)):
        canonical = array.astype(np.uint8 if array.dtype == np.bool_ else "<u4")
        digest.update(json.dumps([name, canonical.shape, canonical.dtype.str]).encode())
        digest.update(canonical.tobytes(order="C"))
    binding = np.frombuffer(digest.digest(), dtype=">u4").astype(np.uint32)
    return SourceTables(source_hash, names, MappingProxyType(species_ids),
                        MappingProxyType(predicate_ids), variants,
                        DeviceTables(jnp.asarray(pool), jnp.asarray(predicates), jnp.asarray(valid), jnp.asarray(binding)))


def initialize(species, tables: DeviceTables, *, source_available=True) -> BeliefState:
    """Start each encounter from its public species pool; zero stays unresolved."""
    species = jnp.asarray(species, jnp.int32)
    # Invalid source IDs are unresolved, never clipped to a different species.
    species = jnp.where((species > 0) & (species < tables.pool.shape[0]), species, 0)
    available = jnp.broadcast_to(jnp.asarray(source_available, jnp.bool_), species.shape)
    mask = jnp.where(available[..., None], tables.pool[species], 0)
    return BeliefState(species, available, mask, jnp.zeros_like(mask),
                       jnp.zeros(species.shape, jnp.bool_), jnp.zeros(species.shape, jnp.bool_),
                       jnp.zeros(species.shape, jnp.int32))


def add_evidence(state: BeliefState, tables: DeviceTables, predicate_id, *, exclude=False, valid=True):
    """Apply one typed public fact per slot, returning state and admission bits.

    Predicate 0 is an off-source fact: a reveal rejects every variant, whereas
    excluding that fact rejects none. Out-of-range IDs are malformed transport
    and leave state intact, returning False. An absent event uses valid=False.
    """
    predicate_id = jnp.asarray(predicate_id, jnp.int32)
    admitted = (predicate_id >= 0) & (predicate_id < tables.predicates.shape[0])
    matched = tables.predicates[jnp.clip(predicate_id, 0, tables.predicates.shape[0] - 1), state.species]
    accepted = jnp.where(jnp.asarray(exclude)[..., None], ~matched, matched)
    apply = admitted & jnp.asarray(valid)
    return state._replace(
        hard=jnp.where(apply[..., None], state.hard & accepted, state.hard),
        has_evidence=state.has_evidence | apply), admitted | ~jnp.asarray(valid)


def narrow_pin(state: BeliefState, incoming, *, valid=True):
    """Match narrow_candidate_variants without materializing a source summary.

    Empty pins are declined without conflict. A disjoint nonempty pin cannot
    erase the standing pin. Returns (state, changed) for the caller's receipt.
    Callers must obtain incoming from the source-bound identity table above.
    """
    incoming = jnp.asarray(incoming, jnp.uint32)
    nonempty = jnp.any(incoming != 0, axis=-1)
    merged = jnp.where(state.pin_active[..., None], state.pin & incoming, incoming)
    disjoint = state.pin_active & nonempty & ~jnp.any(merged != 0, axis=-1)
    accept = jnp.asarray(valid) & nonempty & ~disjoint
    changed = accept & (~state.pin_active | jnp.any(merged != state.pin, axis=-1))
    return state._replace(
        pin=jnp.where(accept[..., None], merged, state.pin),
        pin_active=state.pin_active | accept,
        conflicts=state.conflicts + (disjoint & jnp.asarray(valid)).astype(jnp.int32)), changed


def materialize(state: BeliefState, tables: DeviceTables):
    """Resolve fallback and standing pin exactly once, as Python snapshot does.

    Like _apply_variant_pin, each materialization counts a disjoint pin conflict.
    Consumers should reuse the returned summary rather than materialize again.
    Raw hard/pin masks are never widened by a fallback or conflicting summary.
    """
    pool = jnp.where(state.source_available[..., None], tables.pool[state.species], 0)
    known = jnp.any(pool != 0, axis=-1)
    inconsistent = known & state.has_evidence & ~jnp.any(state.hard != 0, axis=-1)
    base = jnp.where(inconsistent[..., None], pool, state.hard)
    intersect = base & state.pin
    pin_conflict = known & ~inconsistent & state.pin_active & ~jnp.any(intersect != 0, axis=-1)
    use_pin = known & ~inconsistent & state.pin_active & ~pin_conflict
    effective = jnp.where(use_pin[..., None], intersect, base)
    count = jnp.sum(jax.lax.population_count(effective), axis=-1).astype(jnp.int32)
    total = jnp.sum(jax.lax.population_count(pool), axis=-1)
    uncertainty = jnp.where(known & ~inconsistent, count / jnp.maximum(total, 1), 1).astype(jnp.float32)
    reason = jnp.where(~state.source_available, NO_SOURCE,
        jnp.where(~known, UNKNOWN_SPECIES, jnp.where(inconsistent, INCONSISTENT_REVEALS, RESOLVED)))
    return state._replace(conflicts=state.conflicts + pin_conflict.astype(jnp.int32)), BeliefSummary(
        effective, count, uncertainty, reason, pin_conflict)


def variant_weights(summary: BeliefSummary, tables: DeviceTables):
    """Uniform current-sampler weights, with source-order variant axis intact.

    Unknown species has zero mass. These are not inferred generator frequencies.
    No RNG or private physical state enters the posterior representation.
    """
    index = jnp.arange(tables.variant_valid.shape[-1], dtype=jnp.uint32)
    selected = (jnp.take(summary.effective, index // 32, axis=-1) >> (index % 32)) & 1
    return selected.astype(jnp.float32) / jnp.maximum(summary.count[..., None], 1)
