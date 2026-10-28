"""Typed, ordered transport for the public device encoder (incremental B2).

The first fold handles boundaries and source-bound set evidence. Other semantic
events are refused explicitly until their public-state consumers are ported.
It cannot encode a production observation yet. In particular a move declaration
is not automatically a set reveal: called/copied moves require the Python
belief engine's attribution rules before emitting SET_REVEAL.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import public_belief


SCHEMA = "pokezero.public_events.v1"
CHUNK_SIZE, EVENT_WORDS = 64, 16


class Kind(IntEnum):
    TURN = 1
    ACTION_CHUNK_END = 2
    UPKEEP = 3
    SET_REVEAL = 4
    SET_EXCLUDE = 5


class Phase(IntEnum):
    ACTION = 0
    RESIDUAL = 1


class Error(IntEnum):
    NONE = 0
    STREAM = 1
    MASK = 2
    SEQUENCE = 3
    UNSUPPORTED_KIND = 4
    HEADER = 5
    PAYLOAD = 6
    SOURCE = 7


@dataclass(frozen=True)
class PublicEvent:
    kind: Kind
    sequence: int
    turn: int
    phase: Phase
    actor_side: int = -1
    actor_slot: int = -1
    target_side: int = -1
    target_slot: int = -1
    payload: tuple[int, ...] = (0,) * 8

    def words(self) -> np.ndarray:
        """Validate exact integer transport; never wrap or truncate a payload."""
        row = (self.kind, self.sequence, self.turn, self.phase, self.actor_side,
               self.actor_slot, self.target_side, self.target_slot, *self.payload)
        if len(row) != EVENT_WORDS:
            raise ValueError("public event requires exactly eight payload words")
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in row):
            raise ValueError("public event words must be integers")
        if any(x < -(2**31) or x >= 2**31 for x in row):
            raise ValueError("public event word exceeds int32")
        if self.kind not in set(Kind):
            raise ValueError("unsupported public event kind")
        if self.phase not in set(Phase) or self.sequence < 1 or self.turn < 0:
            raise ValueError("invalid public event header")
        boundary = self.kind in (Kind.TURN, Kind.ACTION_CHUNK_END, Kind.UPKEEP)
        if boundary:
            if any(x != -1 for x in row[4:8]) or any(self.payload):
                raise ValueError("boundary event cannot carry a party identity or payload")
        else:
            if (self.actor_side, self.actor_slot) != (-1, -1):
                raise ValueError("set evidence uses target identity only")
            if self.target_side not in (0, 1) or not 0 <= self.target_slot < 6:
                raise ValueError("set evidence requires a resolved public encounter identity")
            if self.payload[0] < 0 or any(self.payload[1:]):
                raise ValueError("set evidence payload is one nonnegative predicate ID")
        expected_phase = Phase.RESIDUAL if self.kind == Kind.UPKEEP else Phase.ACTION
        # Set evidence and blank separators can occur in either phase.
        if self.kind in (Kind.TURN, Kind.UPKEEP) and self.phase != expected_phase:
            raise ValueError("boundary phase disagrees with event kind")
        return np.asarray(row, np.int32)


class EventChunk(NamedTuple):
    stream_id: jax.Array            # [B], host-registered battle identity
    words: jax.Array                # [B,64,16] int32
    valid: jax.Array                # [B,64] bool, a contiguous prefix
    more: jax.Array                 # [B] bool, drain before decision encoding
    source_binding: jax.Array       # [8] uint32, compiled source/variant table


class StreamCursor(NamedTuple):
    stream_id: jax.Array
    source_binding: jax.Array       # [B,8], immutable across stream continuation
    sequence: jax.Array
    turn: jax.Array
    phase: jax.Array
    action_chunks: jax.Array
    upkeeps: jax.Array
    needs_more: jax.Array


class FoldResult(NamedTuple):
    cursor: StreamCursor
    beliefs: public_belief.BeliefState
    error: jax.Array                # [B], nonzero rejects the entire row's chunk


def initialize(stream_ids, *, source_binding) -> StreamCursor:
    ids = jnp.asarray(stream_ids)
    if ids.ndim != 1 or ids.dtype != jnp.int32:
        raise ValueError("stream IDs must be an int32 vector")
    binding = jnp.asarray(source_binding)
    if binding.shape != (8,) or binding.dtype != jnp.uint32:
        raise ValueError("stream requires a uint32 SHA256 source-table binding")
    zeros = jnp.zeros(ids.shape, jnp.int32)
    return StreamCursor(ids, jnp.broadcast_to(binding, (ids.shape[0], 8)),
                        zeros, zeros, zeros, zeros, zeros, jnp.zeros(ids.shape, bool))


def pack_chunks(stream_ids, events, *, source_binding):
    """Yield every event without truncation, in fixed chunks for a batch.

    The host registry must bind each positive stream ID to one full battle ID
    and source-table hash for its lifetime. No species-based identity matching
    or private party-order inference occurs in this transport layer.
    """
    ids = np.asarray(stream_ids)
    if ids.ndim != 1 or ids.dtype != np.int32 or np.any(ids <= 0) or len(set(ids.tolist())) != len(ids):
        raise ValueError("batch needs unique positive int32 stream IDs")
    if len(events) != len(ids):
        raise ValueError("one ordered event sequence is required per stream")
    binding = np.asarray(source_binding)
    if binding.shape != (8,) or binding.dtype != np.uint32:
        raise ValueError("public events require a uint32 SHA256 source-table binding")
    rows = [[event.words() for event in stream] for stream in events]
    count = max((len(stream) for stream in rows), default=0)
    for start in range(0, max(count, 1), CHUNK_SIZE):
        words = np.zeros((len(ids), CHUNK_SIZE, EVENT_WORDS), np.int32)
        valid = np.zeros((len(ids), CHUNK_SIZE), bool)
        more = np.zeros(len(ids), bool)
        for battle, stream in enumerate(rows):
            part = stream[start:start + CHUNK_SIZE]
            if part:
                words[battle, :len(part)] = part
                valid[battle, :len(part)] = True
            more[battle] = len(stream) > start + CHUNK_SIZE
        yield EventChunk(jnp.asarray(ids), jnp.asarray(words), jnp.asarray(valid), jnp.asarray(more), jnp.asarray(binding))


def protocol_boundary(line: str, *, sequence: int, turn: int, phase=Phase.ACTION) -> PublicEvent:
    """Adapt the two exact acting-window delimiters and a turn declaration.

    Unknown kinds and annotated variants are explicit unsupported inputs. This
    helper must not be used to filter a complete protocol down to boundaries.
    The full adapter will emit or refuse each remaining semantic line.
    """
    if line == "|":
        return PublicEvent(Kind.ACTION_CHUNK_END, sequence, turn, phase)
    if line == "|upkeep":
        return PublicEvent(Kind.UPKEEP, sequence, turn, Phase.RESIDUAL)
    parts = line.split("|")
    if len(parts) == 3 and parts[:2] == ["", "turn"] and parts[2].isascii() and parts[2].isdigit():
        return PublicEvent(Kind.TURN, sequence, int(parts[2]), Phase.ACTION)
    raise ValueError(f"unsupported protocol input: {line.split('|')[1:2]}")


def fold_chunk(cursor: StreamCursor, beliefs: public_belief.BeliefState,
               tables: public_belief.DeviceTables, chunk: EventChunk) -> FoldResult:
    """Device-only ordered fold, with atomic rejection for each batch row.

    A repeated, malformed, unimplemented, or wrong-battle chunk cannot leave a
    valid-looking partial prefix applied. The caller retains raw protocol and
    must handle the error before encoding or advancing the source cursor.
    """
    batch = cursor.sequence.shape[0]
    if chunk.words.shape != (batch, CHUNK_SIZE, EVENT_WORDS) or chunk.words.dtype != jnp.int32:
        raise ValueError("expected B×64×16 int32 public events")
    if chunk.valid.shape != (batch, CHUNK_SIZE) or chunk.valid.dtype != jnp.bool_:
        raise ValueError("expected B×64 boolean event validity")
    if chunk.stream_id.shape != (batch,) or chunk.stream_id.dtype != jnp.int32:
        raise ValueError("expected B int32 stream IDs")
    if chunk.more.shape != (batch,) or chunk.more.dtype != jnp.bool_:
        raise ValueError("expected B boolean continuation bits")
    if chunk.source_binding.shape != (8,) or chunk.source_binding.dtype != jnp.uint32:
        raise ValueError("expected eight uint32 source-table binding words")
    if beliefs.species.shape != (batch, 2, 6):
        raise ValueError("public beliefs must have B×2 observers×6 encounter slots")
    error = jnp.where((chunk.stream_id == cursor.stream_id) & (cursor.stream_id > 0),
                      Error.NONE, Error.STREAM).astype(jnp.int32)
    source_ok = jnp.all(chunk.source_binding == tables.binding) & jnp.all(
        cursor.source_binding == tables.binding, axis=-1)
    error = jnp.where((error == 0) & ~source_ok, Error.SOURCE, error)
    prefix = ~jnp.any(chunk.valid[:, 1:] & ~chunk.valid[:, :-1], axis=1)
    # An empty chunk cannot promise more forever, nor complete an undrained stream.
    prefix &= jnp.any(chunk.valid, axis=1) | ~(chunk.more | cursor.needs_more)
    error = jnp.where((error == 0) & ~prefix, Error.MASK, error)

    def step(carry, event_and_valid):
        current, state, error = carry
        row, valid = event_and_valid
        kind, sequence, turn, phase = (row[:, i] for i in range(4))
        boundary = (kind == Kind.TURN) | (kind == Kind.ACTION_CHUNK_END) | (kind == Kind.UPKEEP)
        fact = (kind == Kind.SET_REVEAL) | (kind == Kind.SET_EXCLUDE)
        sequence_ok = (sequence == current.sequence + 1) & (current.sequence < 2**31 - 1)
        kind_ok = boundary | fact
        phase_ok = (phase == Phase.ACTION) | (phase == Phase.RESIDUAL)
        # Only TURN can advance turn or leave residual phase. Other events keep
        # the committed phase, except UPKEEP which enters the residual phase.
        turn_ok = (turn >= 0) & jnp.where(kind == Kind.TURN,
            (turn == current.turn + 1) & (current.turn < 2**31 - 1), turn == current.turn)
        phase_ok &= jnp.where(kind == Kind.TURN, phase == Phase.ACTION,
                              jnp.where(kind == Kind.UPKEEP, phase == Phase.RESIDUAL, phase == current.phase))
        boundary_identity = jnp.all(row[:, 4:8] == -1, axis=1)
        fact_identity = jnp.all(row[:, 4:6] == -1, axis=1) & (row[:, 6] >= 0) & (row[:, 6] <= 1) & (row[:, 7] >= 0) & (row[:, 7] < 6)
        header_ok = turn_ok & phase_ok & jnp.where(boundary, boundary_identity, fact_identity)
        predicate = row[:, 8]
        payload_ok = jnp.where(boundary, jnp.all(row[:, 8:] == 0, axis=1),
            (predicate >= 0) & (predicate < tables.predicates.shape[0]) & jnp.all(row[:, 9:] == 0, axis=1))
        event_error = jnp.where(~sequence_ok, Error.SEQUENCE,
            jnp.where(~kind_ok, Error.UNSUPPORTED_KIND,
            jnp.where(~header_ok, Error.HEADER, jnp.where(~payload_ok, Error.PAYLOAD, Error.NONE))))
        error = jnp.where(valid & (error == 0), event_error, error)
        apply = valid & (error == 0)
        # A public fact about side s narrows the other observer's belief about s.
        observer = jnp.clip(1 - row[:, 6], 0, 1)
        slot = jnp.clip(row[:, 7], 0, 5)
        fact_slots = (jnp.arange(2)[None, :, None] == observer[:, None, None]) & (
            jnp.arange(6)[None, None, :] == slot[:, None, None])
        state, _ = public_belief.add_evidence(state, tables, predicate[:, None, None],
            exclude=(kind == Kind.SET_EXCLUDE)[:, None, None],
            valid=fact_slots & (apply & fact)[:, None, None])
        current = current._replace(
            sequence=jnp.where(apply, sequence, current.sequence),
            turn=jnp.where(apply, turn, current.turn),
            phase=jnp.where(apply, phase, current.phase),
            action_chunks=current.action_chunks + (apply & ((kind == Kind.ACTION_CHUNK_END) | (kind == Kind.UPKEEP))).astype(jnp.int32),
            upkeeps=current.upkeeps + (apply & (kind == Kind.UPKEEP)).astype(jnp.int32))
        return (current, state, error), None

    (candidate, state, error), _ = jax.lax.scan(step, (cursor, beliefs, error),
                                              (jnp.swapaxes(chunk.words, 0, 1), chunk.valid.T))
    candidate = candidate._replace(needs_more=chunk.more)
    def commit(before, after):
        shape = (batch,) + (1,) * (before.ndim - 1)
        return jnp.where((error == 0).reshape(shape), after, before)
    return FoldResult(jax.tree.map(commit, cursor, candidate),
                      jax.tree.map(commit, beliefs, state), error)
