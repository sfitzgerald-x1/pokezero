"""Device projection of the reviewed PARTIAL engine public callback subset.

Host compilation/registration and holder checks are deliberately separate from
``project_device``. That primitive is trusted: its tape must be the real core's
tape and its registry/context must be attached by the same source lifecycle.
Namespace tags do not attest a runtime tape's origin. No BattleState is accepted
and no public slot/encounter is inferred from a private physical party index.

Only MOVE, direct main-hit CONDITION and direct-hit FAINT are supported. Unknown
callback families, overflow and malformed rows refuse the entire battle row.
Every result is partial and cannot admit an encoder, a decision or a B4 gate.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_public_events as source
from .public_inputs import CHUNK_SIZE, EVENT_WORDS, Error, EventChunk, Kind, Phase

SCHEMA = 'pokezero.engine_public_callback_device_subset.v1'


class DeviceTables(NamedTuple):
    public_ids: object
    targets: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    binding: object


@dataclass(frozen=True, eq=False)
class CompiledProjector:
    """Original host handle; ``device`` alone is a trusted primitive argument."""
    catalog: object
    device: DeviceTables


_COMPILED = weakref.WeakKeyDictionary()


def compile_projector(catalog, *, expected_binding, expected_engine_binding):
    """Host-only: retain the reviewed catalog and immutable original edges."""
    source.check_catalog(catalog, catalog.engine_table, expected_binding=expected_binding,
        expected_engine_binding=expected_engine_binding)
    payload = dict(schema=SCHEMA, producer=catalog.binding, ids=catalog.ids, targets=catalog.targets,
        chunk=[CHUNK_SIZE, EVENT_WORDS], missing='atomic_row_refusal', admission='partial_only')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    words = lambda value: jnp.asarray(np.frombuffer(value, dtype='>u4').astype(np.uint32))
    device = DeviceTables(jnp.asarray(catalog.ids, jnp.int32), jnp.asarray(catalog.targets, jnp.int32),
        jnp.asarray(catalog.engine_binding, jnp.uint32), jnp.asarray(catalog.source_binding, jnp.uint32),
        jnp.asarray(catalog.move_binding, jnp.uint32), words(bytes.fromhex(catalog.binding)), words(digest))
    compiled = CompiledProjector(catalog, device)
    _COMPILED[compiled] = (catalog, device, tuple(device), catalog.binding,
        device.engine_binding)
    return compiled


def check_projector(compiled):
    """Host-only guard, outside JIT; run before using a compiled session.

    Recheck after any upstream holder replacement, including after a cached JIT
    call. A compiled closure does not rerun Python host guards on cache hits.
    """
    record = _COMPILED.get(compiled) if type(compiled) is CompiledProjector else None
    if (record is None or compiled.catalog is not record[0] or compiled.device is not record[1]
            or type(compiled.device) is not DeviceTables
            or any(not isinstance(a, jax.Array) or isinstance(a, jax.core.Tracer)
                   or a is not b for a, b in zip(compiled.device, record[2]))):
        raise ValueError('original compiled device projector and immutable array edges required')
    source.check_catalog(compiled.catalog, compiled.catalog.engine_table,
        expected_binding=record[3], expected_engine_binding=record[4])
    return compiled.device


class PublicRegistry(NamedTuple):
    """INTERNAL prior public arrival evidence, never a model input.

    All [B,2] side fields are the BEFORE-frame established public arrival
    snapshot. Never use post-step active indices to project earlier declarations.
    Generation counts installed switches per side and must be supplied by the
    source lifecycle at that same before-frame boundary. Intra-frame lifecycle
    callbacks remain unsupported until causal registry updates are reviewed.
    Physical indices are used solely to join real callbacks.
    revealed_species_id is the currently observed/apparent species in the
    bound ENGINE species namespace, set only by actual public arrival evidence.
    A future Transform/forme change requires an actual public update; otherwise
    the registry must become unknown. It is not read into this subset's words.
    """
    stream_id: object              # [B] unique observer stream, int32
    engine_binding: object         # [B,8] uint32
    generation: object             # [B,2] int32
    physical_active: object        # [B,2] int32
    public_slot: object            # [B,2] int32
    encounter: object              # [B,2] int32
    revealed_species_id: object    # [B,2] int32, public observation / engine namespace
    established: object            # [B,2] bool


class ObserverContext(NamedTuple):
    """Public stream header and same-lifecycle generation, all device arrays."""
    stream_id: object              # [B] int32
    owner: object                  # [B] int32, selected before HP projection
    next_sequence: object          # [B] int32
    generation: object             # [B,2] int32, independently sourced lifecycle


def register_registry(*, stream_id, engine_binding, generation, physical_active,
                      public_slot, encounter, revealed_species_id, established):
    """Host-only explicit registration of before-frame source arrival evidence.

    This validates metadata, not the truth/provenance of an arrival. An absent
    side uses (-1 physical, -1 public slot, 0 encounter/species, established=False).
    Registration does not initialize a battle or fabricate an initial switch.
    """
    registry = PublicRegistry(stream_id, engine_binding, generation, physical_active,
        public_slot, encounter, revealed_species_id, established)
    if getattr(stream_id, 'ndim', None) != 1 or stream_id.shape[0] <= 0:
        raise ValueError('nonempty vector of public stream identities required')
    batch = stream_id.shape[0]
    _registry_shapes(registry, batch)
    if not all(isinstance(a, jax.Array) and not isinstance(a, jax.core.Tracer) for a in registry):
        raise ValueError('host registration requires immutable JAX arrays')
    ids = np.asarray(stream_id)
    if len(set(ids.tolist())) != batch or not bool(np.all(np.asarray(_registry_valid(registry)))):
        raise ValueError('unique streams and explicit valid public active evidence required')
    return registry


class DeviceProjection(NamedTuple):
    chunk: EventChunk
    error: object
    refusal: object
    next_sequence: object
    component_admitted: object
    full_line_consumers_complete: object
    full_encoder_admitted: object
    B4_qualified: object


def _shape(value, shape, dtype):
    if getattr(value, 'shape', None) != shape or getattr(value, 'dtype', None) != jnp.dtype(dtype):
        raise ValueError('exact device projection shapes/dtypes required')


def _registry_shapes(registry, batch):
    if type(registry) is not PublicRegistry:
        raise ValueError('explicit established public registry required')
    _shape(registry.stream_id, (batch,), jnp.int32)
    _shape(registry.engine_binding, (batch, 8), jnp.uint32)
    _shape(registry.generation, (batch, 2), jnp.int32)
    _shape(registry.physical_active, (batch, 2), jnp.int32)
    _shape(registry.public_slot, (batch, 2), jnp.int32)
    _shape(registry.encounter, (batch, 2), jnp.int32)
    _shape(registry.revealed_species_id, (batch, 2), jnp.int32)
    _shape(registry.established, (batch, 2), jnp.bool_)


def _registry_valid(registry):
    present = ((registry.physical_active >= 0) & (registry.physical_active < 6)
        & (registry.public_slot >= 0) & (registry.public_slot < 6) & (registry.encounter > 0)
        & (registry.revealed_species_id > 0))
    absent = ((registry.physical_active == -1) & (registry.public_slot == -1)
        & (registry.encounter == 0) & (registry.revealed_species_id == 0))
    return ((registry.stream_id > 0) & jnp.all(registry.generation >= 0, axis=1)
        & jnp.all(jnp.where(registry.established, present, absent), axis=1))


def _resolve(registry, side, physical):
    safe = jnp.clip(side, 0, 1)
    active = jnp.take_along_axis(registry.physical_active, safe, axis=1)
    slot = jnp.take_along_axis(registry.public_slot, safe, axis=1)
    encounter = jnp.take_along_axis(registry.encounter, safe, axis=1)
    established = jnp.take_along_axis(registry.established, safe, axis=1)
    ok = ((side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6)
        & (physical == active) & established & (slot >= 0) & (slot < 6) & (encounter > 0))
    return slot, encounter, ok


def project_device(tape, tables, registry, observer):
    """Trusted vectorized JAX primitive; no host reads, loops or strings.

    Call ``check_projector`` on the original compiled holder outside JIT. The
    caller must attach registry/observer evidence from the same tape lifecycle;
    tape v1 has no stream/generation attestation of its own. Unknown callback
    kinds are errors, never filtered out. No callback boundary is invented.
    Returned error rejects all words for that batch row, including valid prefix
    facts before an unsupported callback. ``more`` is always False: overflow
    refuses rather than offering a truncated chunk or a drain continuation.
    Sequence capacity must leave a representable next_sequence; a saturated
    stream refuses instead of wrapping an int32 continuation header.
    """
    if type(tape) is not source.CallbackTape or type(tables) is not DeviceTables or type(observer) is not ObserverContext:
        raise ValueError('typed callback tape, device tables and observer context required')
    batch, capacity, width = tape.words.shape
    if batch <= 0 or not 1 <= capacity <= CHUNK_SIZE or width != source.CALLBACK_WORDS:
        raise ValueError('bounded nonempty callback tape required')
    _shape(tape.words, (batch, capacity, source.CALLBACK_WORDS), jnp.int32)
    _shape(tape.count, (batch,), jnp.int32)
    _shape(tape.refusal, (batch,), jnp.int32)
    _shape(tape.turn, (batch,), jnp.int32)
    _shape(tape.engine_binding, (8,), jnp.uint32)
    _shape(tape.source_binding, (8,), jnp.uint32)
    _shape(tape.move_binding, (8,), jnp.uint32)
    _shape(tape.producer_binding, (8,), jnp.uint32)
    _shape(tables.public_ids, tables.targets.shape, jnp.int32)
    if tables.public_ids.ndim != 1 or tables.public_ids.shape[0] <= 0:
        raise ValueError('static compiled move map required')
    _shape(tables.targets, tables.public_ids.shape, jnp.int32)
    _shape(tables.engine_binding, (8,), jnp.uint32)
    _shape(tables.source_binding, (8,), jnp.uint32)
    _shape(tables.move_binding, (8,), jnp.uint32)
    _shape(tables.producer_binding, (8,), jnp.uint32)
    _shape(tables.binding, (8,), jnp.uint32)
    _registry_shapes(registry, batch)
    _shape(observer.stream_id, (batch,), jnp.int32)
    _shape(observer.owner, (batch,), jnp.int32)
    _shape(observer.next_sequence, (batch,), jnp.int32)
    _shape(observer.generation, (batch, 2), jnp.int32)

    index = jnp.arange(capacity)[None, :]
    present = index < tape.count[:, None]
    raw = tape.words
    kind, actor_side, actor_physical = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
    target_side, target_physical = raw[:, :, 3], raw[:, :, 4]
    a, b, c = raw[:, :, 5], raw[:, :, 6], raw[:, :, 7]
    move = kind == source.MOVE
    condition = kind == source.MAIN_HP
    faint = kind == source.DIRECT_FAINT
    move_index = jnp.clip(a, 0, tables.public_ids.shape[0] - 1)
    public_id, target_class = tables.public_ids[move_index], tables.targets[move_index]
    self_target = move & (target_class == 0)
    resolved_side = jnp.where(self_target, actor_side, target_side)
    resolved_physical = jnp.where(self_target, actor_physical, target_physical)
    actor_slot, actor_encounter, actor_ok = _resolve(registry, actor_side, actor_physical)
    target_slot, target_encounter, target_ok = _resolve(registry, resolved_side, resolved_physical)
    identity_ok = target_ok & (~move | actor_ok)
    ordinary_move = ((a > 0) & (a < tables.public_ids.shape[0]) & (public_id > 0)
        & ((target_class == 0) | (target_class == 1)) & (b == 0) & (c == 0))
    hp_ok = (a >= 0) & (a <= b) & (b > 0) & (b <= 1_000_000) & (c >= 0) & (c <= 6)
    no_actor = (actor_side == -1) & (actor_physical == -1)
    payload_ok = (jnp.all(raw[:, :, 8:] == 0, axis=2)
        & jnp.where(move, ordinary_move, no_actor)
        & (move | (condition & hp_ok) | (faint & (a == 0) & (b == 0) & (c == 0))))

    owner_exact = observer.owner[:, None] == resolved_side
    percentage = (100*a + jnp.maximum(1, b) - 1) // jnp.maximum(1, b)
    percentage = jnp.where((a < b) & (percentage == 100), 99, percentage)
    hp = jnp.where(a == 0, 0, jnp.where(owner_exact, a, percentage))
    maximum = jnp.where(a == 0, 0, jnp.where(owner_exact, b, 100))
    status = jnp.where(a == 0, 0, jnp.asarray([0, 2, 3, 5, 6, 1, 4], jnp.int32)[jnp.clip(c, 0, 6)])
    z = jnp.zeros_like(kind)
    absent = jnp.full_like(kind, -1)
    payload0 = jnp.where(move, actor_encounter, target_encounter)
    payload1 = jnp.where(move, public_id, jnp.where(condition, hp, 0))
    payload2 = jnp.where(condition, maximum, 0)
    payload3 = jnp.where(condition, status, 0)
    payload4 = (condition & (a == 0)).astype(jnp.int32)
    payload5 = condition.astype(jnp.int32)  # exact reviewed direct-damage metadata
    event_kind = jnp.where(move, int(Kind.MOVE), jnp.where(condition, int(Kind.CONDITION), int(Kind.FAINT)))
    sequence = observer.next_sequence[:, None] + index
    words = jnp.stack((event_kind, sequence, jnp.broadcast_to(tape.turn[:, None], kind.shape),
        jnp.full_like(kind, int(Phase.ACTION)), jnp.where(move, actor_side, absent),
        jnp.where(move, actor_slot, absent), resolved_side, target_slot,
        payload0, payload1, payload2, payload3, payload4, payload5, z, z), axis=2)

    known_flags = source.UNINSTRUMENTED | source.OVERFLOW | source.INVALID_ACTION
    header_ok = ((tape.count >= 0) & (tape.count <= capacity) & (tape.turn >= 0)
        & (tape.refusal >= 0) & ((tape.refusal & ~known_flags) == 0))
    no_overflow = (tape.refusal & source.OVERFLOW) == 0
    namespace_ok = (jnp.all(tape.engine_binding == tables.engine_binding)
        & jnp.all(tape.source_binding == tables.source_binding)
        & jnp.all(tape.move_binding == tables.move_binding)
        & jnp.all(tape.producer_binding == tables.producer_binding)
        & jnp.all(registry.engine_binding == tables.engine_binding[None, :], axis=1))
    sorted_ids = jnp.sort(observer.stream_id)
    stream_ok = ((observer.stream_id > 0) & (observer.stream_id == registry.stream_id)
        & ~jnp.any(sorted_ids[1:] == sorted_ids[:-1]))
    observer_ok = ((observer.owner >= 0) & (observer.owner <= 1)
        & (observer.next_sequence > 0)
        & (observer.next_sequence <= 2**31 - 1 - jnp.clip(tape.count, 0, capacity)))
    registry_ok = (_registry_valid(registry)
        & jnp.all(observer.generation == registry.generation, axis=1))
    error = jnp.where(header_ok, int(Error.NONE), int(Error.HEADER))
    error = jnp.where(no_overflow & jnp.all(~present | payload_ok, axis=1), error, int(Error.PAYLOAD))
    error = jnp.where(registry_ok & jnp.all(~present | identity_ok, axis=1), error, int(Error.IDENTITY))
    error = jnp.where(observer_ok, error, int(Error.SEQUENCE))
    error = jnp.where(stream_ok, error, int(Error.STREAM))
    error = jnp.where(namespace_ok, error, int(Error.SOURCE)).astype(jnp.int32)
    admitted = error == Error.NONE
    valid = present & admitted[:, None]
    words = jnp.where(valid[:, :, None], words, 0)
    words = jnp.pad(words, ((0, 0), (0, CHUNK_SIZE-capacity), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, CHUNK_SIZE-capacity)))
    no = jnp.zeros((batch,), jnp.bool_)
    chunk = EventChunk(observer.stream_id, words, valid, no, tables.source_binding, move_binding=tables.move_binding)
    next_sequence = jnp.where(admitted, observer.next_sequence + tape.count, observer.next_sequence)
    return DeviceProjection(chunk, error, tape.refusal | source.UNINSTRUMENTED, next_sequence,
        admitted, no, no, no)
