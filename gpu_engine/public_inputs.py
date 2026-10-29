"""Typed, ordered transport for the public device encoder (incremental B2).

The fold handles boundaries, source-bound set evidence, and the public encounter,
HP/status/boost ledger. Optional public context adds named move/ability/Transform
consumers; an item-enabled context also consumes per-raw-line boundaries and
item attribution. Other semantic events are refused explicitly until their consumers
are ported. Accepted updates cover only those named fields; they
are not evidence that all consumers of the corresponding raw line are ported.
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

from . import public_belief, public_mon_state, public_context, public_field, public_active, public_damage, public_types, public_sleep, public_toxic
from . import public_tendency, public_tendency_protocol


SCHEMA = "pokezero.public_events.v1"
CHUNK_SIZE, EVENT_WORDS = 64, 16


class Kind(IntEnum):
    TURN = 1
    ACTION_CHUNK_END = 2
    UPKEEP = 3
    SET_REVEAL = 4
    SET_EXCLUDE = 5
    SWITCH = 6
    DRAG = 7
    REPLACE = 8
    BATON_PASS = 9
    CONDITION = 10
    STATUS = 11
    FAINT = 12
    BOOST = 13
    COPY_BOOST = 14
    MOVE = 15
    ABILITY = 16
    TRACE = 17
    TRANSFORM = 18
    CLEAR_ALL_BOOSTS = 19
    CURE_TEAM = 20
    RAW_BEGIN = 21
    ITEM = 22
    FIELD = 23
    ACTIVE = 24
    DAMAGE_CONTROL = 25
    TYPES = 26
    SLEEP = 27
    TOXIC = 28
    TENDENCY = 29


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
    IDENTITY = 8


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
        if boundary or self.kind==Kind.CLEAR_ALL_BOOSTS:
            if any(x != -1 for x in row[4:8]) or any(self.payload):
                raise ValueError("boundary event cannot carry a party identity or payload")
        elif self.kind==Kind.RAW_BEGIN:
            d=self.payload
            if d[0] not in (0,1,2) or d[3] not in range(8) or any(d[4:]):
                raise ValueError("raw boundary requires a named item family")
            if d[0]==2:
                if not (self.actor_side in (0,1) and 0<=self.actor_slot<6 and self.target_side in (0,1)
                        and 0<=self.target_slot<6 and self.actor_side!=self.target_side and d[1]>0 and d[2]>0):
                    raise ValueError("Trick boundary requires two public encounters")
            elif any(x!=-1 for x in row[4:8]) or any(d[1:3]):
                raise ValueError("ordinary raw boundary carries no identity")
        elif self.kind==Kind.FIELD:
            d=self.payload
            if (self.actor_side,self.actor_slot,self.target_slot)!=(-1,-1,-1) or d[0] not in (public_field.WEATHER,public_field.SPIKES,public_field.WISH):
                raise ValueError("field fact requires its named operation and public side only")
            if self.target_side not in ((-1,) if d[0]==public_field.WEATHER else (0,1)):
                raise ValueError("field operation has an incompatible public side")
            if (any(d[4:]) or d[1]<0 or d[2] not in (0,1) or d[3] not in (0,1)
                    or d[0]!=public_field.WEATHER and (d[1] not in (0,1) or any(d[2:]))):
                raise ValueError("invalid field fact payload")
        elif self.kind==Kind.ACTIVE:
            d=self.payload
            if ((self.actor_side,self.actor_slot)!=(-1,-1) or self.target_side not in (0,1)
                    or not 0<=self.target_slot<6 or d[0]<=0 or not public_active.EXPIRE_ACTION<=d[1]<=public_active.MAX_OPERATION
                    or d[1]==public_active.UPKEEP
                    or any(d[3:])):
                raise ValueError("active fact requires a resolved encounter and named operation")
            named=d[1] in (public_active.ADD,public_active.REMOVE,public_active.PERISH_START)
            if not (d[2]>0 if named else d[2] in (0,1) if d[1] in (public_active.STALL_ATTEMPT,public_active.CHOICE_MOVE,public_active.TRUANT_TRACE)
                    else 0<=d[2]<=3 if d[1]==public_active.ITEM_SET else d[2]==0):
                raise ValueError("invalid active fact value")
        elif self.kind==Kind.TYPES:
            d=self.payload
            if ((self.actor_side,self.actor_slot)!=(-1,-1) or self.target_side not in (0,1)
                    or not 0<=self.target_slot<6 or d[0]<=0 or not public_types.CLEAR<=d[1]<=public_types.FORME
                    or any(d[4:]) or d[2]<0 or d[3]<0 or d[1]!=public_types.TYPE and d[3]!=0
                    or d[1]==public_types.CLEAR and d[2]!=0 or d[1]==public_types.TYPE and d[2]==0):
                raise ValueError('type fact requires a resolved encounter and named value')
        elif self.kind==Kind.SLEEP:
            d=self.payload
            if ((self.actor_side,self.actor_slot)!=(-1,-1) or self.target_side not in (0,1)
                    or not 0<=self.target_slot<6 or d[0]<=0 or d[1] not in (public_sleep.STATUS,public_sleep.CURE,
                        public_sleep.CANT_SLEEP,public_sleep.USABLE_MOVE,public_sleep.CONDITION) or any(d[4:])):
                raise ValueError('sleep fact requires a resolved encounter and named lexical operation')
            if d[1] in (public_sleep.STATUS,public_sleep.CONDITION):
                good=0<=d[2]<=public_mon_state.TOXIC and (d[3] in (0,1) and (d[2]==public_mon_state.SLEEP or d[3]==0) if d[1]==public_sleep.STATUS else d[3] in (0,2,3))
            else:good=d[3]==0 and (d[2] in (0,1) if d[1]==public_sleep.CURE else d[2]==0)
            if not good:raise ValueError('invalid sleep lexical payload')
        elif self.kind==Kind.TOXIC:
            d=self.payload
            named=(public_toxic.OTHER,public_toxic.TIMESTAMP,public_toxic.STATUS,public_toxic.CURE,
                public_toxic.CONDITION,public_toxic.MOVE,public_toxic.WIN,public_toxic.TERMINAL)
            if ((self.actor_side,self.actor_slot,self.target_slot)!=(-1,-1,-1) or self.target_side not in (-1,0,1)
                    or d[0] not in named or not 0<=d[1]<=6 or d[2]<0 or d[3]<-1 or d[4]<0 or not 0<=d[5]<128 or any(d[6:])):
                raise ValueError('invalid raw Toxic control header or payload')
            if bool(d[5]&32)!=(d[0] in (public_toxic.WIN,public_toxic.TERMINAL)) or d[5]&64:
                raise ValueError('Toxic raw control retirement flags disagree with its kind')
        elif self.kind==Kind.TENDENCY:
            if ((self.actor_side,self.actor_slot,self.target_slot)!=(-1,-1,-1)
                    or not public_tendency_protocol.valid_words(self.payload,self.target_side)):
                raise ValueError('tendency control requires canonical lexical words and public side')
        elif self.kind==Kind.DAMAGE_CONTROL:
            op=self.payload[0]
            if op not in (public_damage.CLOSE,public_damage.REMOVED) or any(self.payload[1:]):
                raise ValueError("damage control requires its named operation and zero reserved words")
            if (self.actor_side,self.actor_slot,self.target_slot)!=(-1,-1,-1) or self.target_side not in ((-1,) if op==public_damage.CLOSE else (0,1)):
                raise ValueError("damage control requires global closure or a public victim side")
        elif self.kind==Kind.CURE_TEAM:
            if (self.actor_side,self.actor_slot,self.target_slot)!=(-1,-1,-1) or self.target_side not in (0,1) or any(self.payload):
                raise ValueError("team cure requires only one public side and zero payload")
        elif self.kind in (Kind.MOVE, Kind.TRACE, Kind.TRANSFORM):
            if self.actor_side not in (0, 1) or not 0 <= self.actor_slot < 6:
                raise ValueError("context event requires a resolved actor address")
            absent = (self.target_side, self.target_slot) == (-1, -1)
            resolved = self.target_side in (0, 1) and 0 <= self.target_slot < 6
            if not (resolved or absent and self.kind != Kind.TRANSFORM):
                raise ValueError("context event requires a resolved or explicitly absent target")
            d = self.payload
            if self.kind == Kind.MOVE and not (d[0] > 0 and d[1] > 0 and 0 <= d[2] <= 4 and d[3] in (0, 1) and not any(d[4:])):
                raise ValueError("invalid public move payload")
            if self.kind == Kind.TRACE and not (d[0] > 0 and d[1] > 0 and (d[2] == 0 if absent else d[2] > 0) and d[3] in (0, 1) and not any(d[4:])):
                raise ValueError("invalid public Trace payload")
            if self.kind == Kind.TRANSFORM and not (d[0] > 0 and d[1] > 0 and not any(d[2:])):
                raise ValueError("invalid public Transform payload")
        elif self.kind == Kind.COPY_BOOST:
            if self.actor_side not in (0, 1) or not 0 <= self.actor_slot < 6 or self.target_side not in (0, 1) or not 0 <= self.target_slot < 6:
                raise ValueError("boost copy requires two resolved encounter addresses")
            if self.payload[0] <= 0 or self.payload[1] <= 0 or any(self.payload[2:]):
                raise ValueError("boost copy payload requires both encounter identities")
        else:
            if (self.actor_side, self.actor_slot) != (-1, -1):
                raise ValueError("public mon updates use target identity only")
            if self.target_side not in (0, 1) or not 0 <= self.target_slot < 6:
                raise ValueError("public mon update requires a resolved encounter address")
        if self.kind in (Kind.SET_REVEAL, Kind.SET_EXCLUDE):
            if self.payload[0] < 0 or any(self.payload[1:]):
                raise ValueError("set evidence payload is one nonnegative predicate ID")
        if self.kind in (Kind.SWITCH, Kind.DRAG, Kind.REPLACE, Kind.BATON_PASS):
            encounter, species, source_species, level, gender, hp, maximum, status = self.payload
            if not (encounter > 0 and species >= 0 and source_species >= 0 and 0 <= level <= 100 and 0 <= gender <= 2 and 0 <= hp <= maximum and maximum > 0 and 0 <= status <= public_mon_state.TOXIC):
                raise ValueError("invalid public arrival payload")
        if self.kind == Kind.CONDITION:
            encounter, hp, maximum, status, fainted, damage, tagged, spikes = self.payload
            hp_ok = maximum > 0 and 0 <= hp <= maximum or fainted == 1 and hp == maximum == 0
            metadata_ok=damage in (0,1) and tagged in (0,1) and spikes in (0,1) and tagged<=damage and spikes<=tagged
            if not (encounter > 0 and hp_ok and 0 <= status <= public_mon_state.TOXIC and fainted in (0, 1) and metadata_ok):
                raise ValueError("invalid public condition payload")
        if self.kind == Kind.STATUS:
            if not (self.payload[0] > 0 and 0 <= self.payload[1] <= public_mon_state.TOXIC and not any(self.payload[2:])):
                raise ValueError("invalid public status payload")
        if self.kind == Kind.ITEM and not (self.payload[0]>0 and self.payload[1]>0 and 0<=self.payload[2]<=6 and not any(self.payload[3:])):
            raise ValueError("invalid public item payload")
        if self.kind == Kind.ABILITY and not (self.payload[0] > 0 and self.payload[1] > 0 and self.payload[2] in (0, 1) and not any(self.payload[3:])):
            raise ValueError("invalid public ability payload")
        if self.kind == Kind.FAINT and not (self.payload[0] > 0 and not any(self.payload[1:])):
            raise ValueError("invalid public faint payload")
        if self.kind == Kind.BOOST:
            if not (self.payload[0] > 0 and 0 <= self.payload[1] < 7 and 0 <= self.payload[3] <= public_mon_state.CLEAR_NEGATIVE and not any(self.payload[4:])):
                raise ValueError("invalid public boost payload")
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
    species_binding: jax.Array | None = None  # [8] uint32 public dex namespace; legacy only when unbound
    ability_binding: jax.Array | None = None
    move_binding: jax.Array | None = None
    item_binding: jax.Array | None = None
    context_binding: jax.Array | None = None
    field_binding: jax.Array | None = None
    active_binding: jax.Array | None = None
    damage_binding: jax.Array | None = None
    type_binding: jax.Array | None = None
    sleep_binding: jax.Array | None = None
    toxic_binding: jax.Array | None = None
    tendency_binding: jax.Array | None = None


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
    mon_state: public_mon_state.MonState | None = None
    context: public_context.ContextState | None = None


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


def pack_chunks(stream_ids, events, *, source_binding, species_binding=None, ability_binding=None, move_binding=None, item_binding=None, context_binding=None, field_binding=None, active_binding=None, damage_binding=None, type_binding=None, sleep_binding=None, toxic_binding=None, tendency_binding=None):
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
    if species_binding is not None:
        species_binding=np.asarray(species_binding)
        if species_binding.shape!=(8,) or species_binding.dtype!=np.uint32:
            raise ValueError('public events require eight uint32 species namespace words')
    for namespace in (ability_binding,move_binding,item_binding,context_binding,field_binding,active_binding,damage_binding,type_binding,sleep_binding,toxic_binding,tendency_binding):
        if namespace is not None and (getattr(namespace,'shape',None)!=(8,) or getattr(namespace,'dtype',None)!=np.dtype(np.uint32)):
            raise ValueError('public context namespace requires eight uint32 words')
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
        yield EventChunk(jnp.asarray(ids), jnp.asarray(words), jnp.asarray(valid), jnp.asarray(more), jnp.asarray(binding),
                         jnp.asarray(species_binding) if species_binding is not None else None,
                         jnp.asarray(ability_binding) if ability_binding is not None else None,
                         jnp.asarray(move_binding) if move_binding is not None else None,
                         jnp.asarray(item_binding) if item_binding is not None else None,
                         jnp.asarray(context_binding) if context_binding is not None else None,
                         jnp.asarray(field_binding) if field_binding is not None else None,
                         jnp.asarray(active_binding) if active_binding is not None else None,
                         jnp.asarray(damage_binding) if damage_binding is not None else None,
                         jnp.asarray(type_binding) if type_binding is not None else None,
                         jnp.asarray(sleep_binding) if sleep_binding is not None else None,
                         jnp.asarray(toxic_binding) if toxic_binding is not None else None,
                         jnp.asarray(tendency_binding) if tendency_binding is not None else None)


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


def _fold_mon_event(mons, beliefs, tables, row, valid, *, damage_metadata=False):
    """Consume one typed mon event per battle, including candidate-slot binding."""
    p = public_mon_state
    batch = row.shape[0]
    rows = jnp.arange(batch)
    kind, side, slot = row[:, 0], row[:, 6], row[:, 7]
    side_safe, slot_safe = jnp.clip(side, 0, 1), jnp.clip(slot, 0, 5)
    data = row[:, 8:]
    arrival = (kind >= Kind.SWITCH) & (kind <= Kind.BATON_PASS)
    old_identity = mons.encounter[rows, side_safe, slot_safe]
    identity_ok = jnp.where(arrival, (old_identity == 0) | (old_identity == data[:, 0]),
                            old_identity == data[:, 0])
    identity_ok &= data[:, 0] > 0
    active_target = mons.active[rows, side_safe] == slot
    identity_ok &= jnp.where(kind == Kind.BOOST, active_target, True)
    actor_side, actor_slot = jnp.clip(row[:, 4], 0, 1), jnp.clip(row[:, 5], 0, 5)
    copy_identity = (mons.encounter[rows, actor_side, actor_slot] == data[:, 0]) & (
        old_identity == data[:, 1]) & (data[:, 0] > 0) & (data[:, 1] > 0) & (
        mons.active[rows, actor_side] == row[:, 5]) & active_target
    identity_ok = jnp.where(kind == Kind.COPY_BOOST, copy_identity, identity_ok)
    observer = 1 - side_safe
    belief_address = rows, observer, slot_safe
    new_source = data[:, 2]
    initialize_source = (old_identity == 0) | (beliefs.species[belief_address] != new_source)
    clean = ~beliefs.has_evidence[belief_address] & ~beliefs.pin_active[belief_address]
    source_ok = (~arrival) | ((new_source >= 0) & (new_source < tables.pool.shape[0]) & (
        ~initialize_source | clean))
    use = valid & identity_ok & source_ok
    mons, a = p.arrive(mons, side=side, slot=slot, encounter=data[:, 0], species=data[:, 1],
        source_species=new_source, level=data[:, 3], gender=data[:, 4], hp_numerator=data[:, 5],
        hp_denominator=data[:, 6], status=data[:, 7], kind=kind - Kind.SWITCH, valid=use & arrival)
    fresh = public_belief.initialize(new_source, tables,
        source_available=beliefs.source_available[belief_address])
    beliefs = beliefs._replace(**{name: p._write(getattr(beliefs, name), belief_address,
        getattr(fresh, name), a & initialize_source) for name in beliefs._fields})
    metadata=jnp.all((data[:,5:]>=0)&(data[:,5:]<=1),axis=1)&(data[:,6]<=data[:,5])&(data[:,7]<=data[:,6])
    condition_payload = ((data[:, 4] == 0) | (data[:, 4] == 1)) & (metadata if damage_metadata else jnp.all(data[:,5:]==0,axis=1))
    mons, c = p.condition_update(mons, side=side, slot=slot, encounter=data[:, 0],
        numerator=data[:, 1], denominator=data[:, 2], status=data[:, 3], fainted=data[:, 4] == 1,
        valid=use & (kind == Kind.CONDITION) & condition_payload)
    mons, s = p.status_update(mons, side=side, slot=slot, encounter=data[:, 0], status=data[:, 1],
        valid=use & (kind == Kind.STATUS) & jnp.all(data[:, 2:] == 0, axis=1))
    mons, f = p.faint(mons, side=side, slot=slot, encounter=data[:, 0],
        valid=use & (kind == Kind.FAINT) & jnp.all(data[:, 1:] == 0, axis=1))
    mons, b = p.boost_update(mons, side=side, stat=data[:, 1], amount=data[:, 2], operation=data[:, 3],
        valid=use & (kind == Kind.BOOST) & jnp.all(data[:, 4:] == 0, axis=1))
    mons, cp = p.copy_boosts(mons, source_side=row[:, 4], target_side=side,
        valid=use & (kind == Kind.COPY_BOOST) & jnp.all(data[:, 2:] == 0, axis=1))
    accepted = a | c | s | f | b | cp
    error = jnp.where(~source_ok, Error.SOURCE, jnp.where(~identity_ok, Error.IDENTITY,
        jnp.where(~accepted, Error.PAYLOAD, Error.NONE)))
    return mons, beliefs, jnp.where(valid, error, Error.NONE).astype(jnp.int32)


def fold_chunk(cursor: StreamCursor, beliefs: public_belief.BeliefState,
               tables: public_belief.DeviceTables, chunk: EventChunk, *, mon_state=None,
               context=None, context_tables=None) -> FoldResult:
    """Device-only ordered fold, with atomic rejection for each batch row.

    Supply mon_state to enable the incremental public mon consumers, and a
    context plus its tables for move/ability/Transform composition. Context
    retains a non-ability evidence basis so original-ability corrections can
    recover variants without losing moves, exclusions or standing pins. The
    raw protocol adapter and remaining consumers are required for production.
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
    if mon_state is not None and mon_state.encounter.shape != (batch, 2, 6):
        raise ValueError("public mon state must have B×2 sides×6 encounters")
    if (context is None) != (context_tables is None) or context is not None and mon_state is None:
        raise ValueError("public context requires its tables and mon ledger together")
    if context is not None and context.base_has_evidence.shape != (batch, 2, 6):
        raise ValueError("public context basis must have B×2×6 encounters")
    if context is not None and (context.refresh is None)!=(context_tables.refresh is None):
        raise ValueError('ordered summary state and tables must be supplied together')
    error = jnp.where((chunk.stream_id == cursor.stream_id) & (cursor.stream_id > 0),
                      Error.NONE, Error.STREAM).astype(jnp.int32)
    source_ok = jnp.all(chunk.source_binding == tables.binding) & jnp.all(
        cursor.source_binding == tables.binding, axis=-1)
    if chunk.species_binding is not None:
        if chunk.species_binding.shape!=(8,) or chunk.species_binding.dtype!=jnp.uint32:
            raise ValueError('expected eight uint32 public species namespace words')
        source_ok &= mon_state is not None
        if mon_state is not None:
            source_ok &= jnp.all(mon_state.species_binding==chunk.species_binding,axis=-1)
    elif mon_state is not None:
        # Legacy opaque-ID component fixtures remain usable; a bound production
        # namespace cannot be downgraded by dropping its transport binding.
        source_ok &= jnp.all(mon_state.species_binding==0,axis=-1)
    if context is not None:
        source_ok &= jnp.all(context.abilities.species_binding==mon_state.species_binding,axis=-1)
    for namespace,state_binding,table_binding in (
        (chunk.ability_binding,context.abilities.binding if context is not None else None,
         context_tables.abilities.binding if context_tables is not None else None),
        (chunk.move_binding,context.moves.binding if context is not None else None,
         context_tables.moves.binding if context_tables is not None else None)):
        if namespace is not None:
            if namespace.shape!=(8,) or namespace.dtype!=jnp.uint32:
                raise ValueError('expected eight uint32 public context namespace words')
            source_ok &= context is not None
            if context is not None:
                source_ok &= jnp.all(state_binding==namespace,axis=-1) & jnp.all(table_binding==namespace)
        elif context is not None:
            source_ok &= jnp.all(mon_state.species_binding==0,axis=-1)
    item_enabled=context is not None and context.items is not None
    if context is not None and (context.items is None)!=(context_tables.items is None):
        raise ValueError("item context requires its frozen tables")
    for namespace,state_binding,table_binding in (
        (chunk.item_binding,context.items.binding if item_enabled else None,
         context_tables.items.binding if item_enabled else None),
        (chunk.context_binding,context.binding if item_enabled else None,
         context_tables.binding if item_enabled else None)):
        if namespace is not None:
            if namespace.shape!=(8,) or namespace.dtype!=jnp.uint32:
                raise ValueError("expected eight uint32 item context words")
            source_ok &= item_enabled
            if item_enabled:
                source_ok &= jnp.all(state_binding==namespace,axis=-1)&jnp.all(table_binding==namespace)
        elif item_enabled:
            source_ok &= False
    if item_enabled:
        source_ok &= jnp.all(context.items.species_binding==mon_state.species_binding,axis=-1)
    field_enabled=context is not None and context.field is not None
    if field_enabled and not item_enabled:
        raise ValueError("field context requires the raw item-enabled composition")
    if context is not None and (context.field is None)!=(context_tables.field is None):
        raise ValueError("field context requires its frozen tables")
    if chunk.field_binding is not None:
        if chunk.field_binding.shape!=(8,) or chunk.field_binding.dtype!=jnp.uint32:
            raise ValueError("expected eight uint32 field namespace words")
        source_ok &= field_enabled
        if field_enabled:
            source_ok &= jnp.all(context.field.binding==chunk.field_binding,axis=-1)&jnp.all(context_tables.field.binding==chunk.field_binding)
    elif field_enabled:
        source_ok &= False
    if field_enabled:
        source_ok &= context.field.turn==cursor.turn
    active_enabled=context is not None and context.active is not None
    if active_enabled and not field_enabled:
        raise ValueError("active context requires ordered field composition")
    if context is not None and (context.active is None)!=(context_tables.active is None):
        raise ValueError("active context requires its frozen tables")
    if chunk.active_binding is not None:
        if chunk.active_binding.shape!=(8,) or chunk.active_binding.dtype!=jnp.uint32:
            raise ValueError("expected eight uint32 active namespace words")
        source_ok &= active_enabled
        if active_enabled:
            source_ok &= jnp.all(context.active.binding==chunk.active_binding,axis=-1)&jnp.all(context_tables.active.binding==chunk.active_binding)
    elif active_enabled:
        source_ok &= False
    if active_enabled:
        source_ok &= (context.active.turn==cursor.turn)&jnp.all(context.active.species_binding==mon_state.species_binding,axis=-1)
    truant_enabled=active_enabled and context_tables.active.native_truant is not None
    if active_enabled and (context_tables.active.native_truant is None)!=(context_tables.active.public_species_binding is None):
        raise ValueError('Truant tables require their frozen public species namespace')
    if truant_enabled:
        source_ok &= jnp.all(context.active.species_binding==context_tables.active.public_species_binding,axis=-1)
        source_ok &= context.active.post_upkeep==(cursor.phase==Phase.RESIDUAL)
    damage_enabled=context is not None and context.damage is not None
    if damage_enabled and not field_enabled:
        raise ValueError("damage context requires ordered field composition")
    if context is not None and (context.damage is None)!=(context_tables.damage is None):
        raise ValueError("damage context requires its frozen tables")
    if chunk.damage_binding is not None:
        if chunk.damage_binding.shape!=(8,) or chunk.damage_binding.dtype!=jnp.uint32:
            raise ValueError("expected eight uint32 damage namespace words")
        source_ok &= damage_enabled
        if damage_enabled:
            source_ok &= jnp.all(context.damage.binding==chunk.damage_binding,axis=-1)&jnp.all(context_tables.damage.binding==chunk.damage_binding)
    elif damage_enabled:
        source_ok &= False
    if damage_enabled:
        source_ok &= (context.damage.turn==cursor.turn)&jnp.all(context.damage.species_binding==mon_state.species_binding,axis=-1)
    type_enabled=context is not None and context.types is not None
    if type_enabled and not field_enabled:
        raise ValueError('live-type context requires ordered field composition')
    if context is not None and (context.types is None)!=(context_tables.types is None):
        raise ValueError('live-type context requires its frozen tables')
    if chunk.type_binding is not None:
        if chunk.type_binding.shape!=(8,) or chunk.type_binding.dtype!=jnp.uint32:
            raise ValueError('expected eight uint32 live-type namespace words')
        source_ok &= type_enabled
        if type_enabled:
            source_ok &= jnp.all(context.types.binding==chunk.type_binding,axis=-1)&jnp.all(context_tables.types.binding==chunk.type_binding)
    elif type_enabled:source_ok &= False
    if type_enabled:
        source_ok &= jnp.all(context.types.species_binding==mon_state.species_binding,axis=-1)
        source_ok &= jnp.all(context.types.species_binding==context_tables.types.species_binding,axis=-1)

    sleep_enabled=context is not None and context.sleep is not None
    if sleep_enabled and not field_enabled:raise ValueError('sleep context requires ordered field composition')
    if context is not None and (context.sleep is None)!=(context_tables.sleep is None):
        raise ValueError('sleep context requires its frozen tables')
    if chunk.sleep_binding is not None:
        if chunk.sleep_binding.shape!=(8,) or chunk.sleep_binding.dtype!=jnp.uint32:
            raise ValueError('expected eight uint32 sleep namespace words')
        source_ok &= sleep_enabled
        if sleep_enabled:
            source_ok &= jnp.all(context.sleep.binding==chunk.sleep_binding,axis=-1)&jnp.all(context_tables.sleep.binding==chunk.sleep_binding)
    elif sleep_enabled:source_ok &= False
    if sleep_enabled:
        source_ok &= (context.sleep.turn==cursor.turn)&jnp.all(context.sleep.species_binding==mon_state.species_binding,axis=-1)
        source_ok &= jnp.all(context.sleep.species_binding==context_tables.sleep.species_binding,axis=-1)
    toxic_enabled=context is not None and context.toxic is not None
    if toxic_enabled and not field_enabled:raise ValueError('Toxic context requires ordered raw/field composition')
    if context is not None and (context.toxic is None)!=(context_tables.toxic is None):
        raise ValueError('Toxic context requires its frozen tables')
    if context is not None and (context.toxic_raw_pending is None)!=(not toxic_enabled):
        raise ValueError('Toxic raw transaction latch must match the enabled consumer')
    if chunk.toxic_binding is not None:
        if chunk.toxic_binding.shape!=(8,) or chunk.toxic_binding.dtype!=jnp.uint32:
            raise ValueError('expected eight uint32 Toxic namespace words')
        source_ok &= toxic_enabled
        if toxic_enabled:
            source_ok &= jnp.all(context.toxic.binding==chunk.toxic_binding,axis=-1)&jnp.all(context_tables.toxic.binding==chunk.toxic_binding)
    elif toxic_enabled:source_ok &= False
    if toxic_enabled:
        if context.toxic_raw_pending.shape!=(batch,) or context.toxic_raw_pending.dtype!=jnp.bool_:
            raise ValueError('Toxic raw transaction latch requires B boolean flags')
        source_ok &= ~context.toxic_raw_pending|cursor.needs_more
        source_ok &= (context.toxic.turn==cursor.turn)&(context.toxic.post_upkeep==(cursor.phase==Phase.RESIDUAL))
        source_ok &= jnp.all(context.toxic.species_binding==mon_state.species_binding,axis=-1)
        source_ok &= jnp.all(context.toxic.species_binding==context_tables.toxic.species_binding,axis=-1)
        row=jnp.arange(batch)[:,None];seats=jnp.arange(2)[None,:]
        current=jnp.where(mon_state.active>=0,mon_state.encounter[row,seats,jnp.clip(mon_state.active,0,5)],0)
        source_ok &= jnp.all(context.toxic.active==current,axis=-1)
    tendency_enabled=context is not None and context.tendency is not None
    if tendency_enabled and not field_enabled:raise ValueError('tendency context requires ordered field composition')
    if context is not None and (context.tendency is None)!=(context_tables.tendency is None):
        raise ValueError('tendency context requires its frozen tables')
    if context is not None and (context.tendency_raw_pending is None)!=(not tendency_enabled):
        raise ValueError('tendency context requires its raw transaction state')
    if chunk.tendency_binding is not None:
        if chunk.tendency_binding.shape!=(8,) or chunk.tendency_binding.dtype!=jnp.uint32:
            raise ValueError('expected eight uint32 tendency namespace words')
        source_ok &= tendency_enabled
        if tendency_enabled:
            source_ok &= jnp.all(context.tendency.binding==chunk.tendency_binding,axis=-1)&jnp.all(context_tables.tendency.binding==chunk.tendency_binding)
    elif tendency_enabled:source_ok &= False
    if tendency_enabled:
        public_tendency._shape(context.tendency,context_tables.tendency)
        if context.tendency_raw_pending.shape!=(batch,) or context.tendency_raw_pending.dtype!=jnp.bool_:
            raise ValueError('tendency raw transaction state requires a boolean batch')
        pending=context.tendency_raw_pending
        source_ok &= public_tendency._coherent(context.tendency,context_tables.tendency)
        source_ok &= ~pending|cursor.needs_more
        # TURN is already committed to the cursor if a transport chunk ends
        # before this raw line's final tendency control. Permit that one lag.
        source_ok &= (context.tendency.turn==cursor.turn)|(pending&(context.tendency.turn<2**31-1)&(context.tendency.turn+1==cursor.turn))
        source_ok &= jnp.all(context.tendency.species_binding==mon_state.species_binding,axis=-1)
        ts=jnp.take_along_axis(context.tendency.species,jnp.clip(context.tendency.active,0,5)[:,:,None],axis=2)[:,:,0]
        ms=jnp.take_along_axis(mon_state.species,jnp.clip(mon_state.active,0,5)[:,:,None],axis=2)[:,:,0]
        ts=jnp.where(context.tendency.active>=0,ts,0);ms=jnp.where(mon_state.active>=0,ms,0)
        source_ok &= pending|jnp.all(ts==ms,axis=-1)
    error = jnp.where((error == 0) & ~source_ok, Error.SOURCE, error)
    prefix = ~jnp.any(chunk.valid[:, 1:] & ~chunk.valid[:, :-1], axis=1)
    # An empty chunk cannot promise more forever, nor complete an undrained stream.
    prefix &= jnp.any(chunk.valid, axis=1) | ~(chunk.more | cursor.needs_more)
    error = jnp.where((error == 0) & ~prefix, Error.MASK, error)
    base = beliefs
    if context is not None:
        base = public_context.basis(context, beliefs)
        rebuilt, bound = public_context.rebuild(context, base, tables, context_tables)
        consistent = jnp.all(rebuilt.hard == beliefs.hard, axis=(1, 2, 3)) & jnp.all(
            rebuilt.has_evidence == beliefs.has_evidence, axis=(1, 2))
        error = jnp.where((error == 0) & ~(bound & consistent), Error.SOURCE, error)
        if context.refresh is not None:
            from . import public_refresh_stream
            coherent=public_refresh_stream.validate(context.refresh,beliefs,tables,context_tables.refresh,context_tables.moves)
            coherent &= (context.refresh.family!=0)==context.tendency_raw_pending
            error=jnp.where((error==0)&~coherent,Error.SOURCE,error)

    def step(carry, event_and_valid):
        current, state, error, mons, ctx, basis = carry
        row, valid = event_and_valid
        kind, sequence, turn, phase = (row[:, i] for i in range(4))
        boundary = (kind == Kind.TURN) | (kind == Kind.ACTION_CHUNK_END) | (kind == Kind.UPKEEP)
        fact = (kind == Kind.SET_REVEAL) | (kind == Kind.SET_EXCLUDE)
        sequence_ok = (sequence == current.sequence + 1) & (current.sequence < 2**31 - 1)
        regular_mon_kind = (kind >= Kind.SWITCH) & (kind <= Kind.COPY_BOOST)
        global_mon_kind = (kind==Kind.CLEAR_ALL_BOOSTS)|(kind==Kind.CURE_TEAM)
        mon_kind = regular_mon_kind|global_mon_kind
        context_kind = (kind >= Kind.MOVE) & (kind <= Kind.TRANSFORM)
        item_kind = (kind==Kind.RAW_BEGIN)|(kind==Kind.ITEM)
        field_kind = kind==Kind.FIELD
        active_kind = kind==Kind.ACTIVE
        damage_kind = kind==Kind.DAMAGE_CONTROL
        sleep_kind = kind==Kind.SLEEP
        type_kind = kind==Kind.TYPES
        toxic_kind = kind==Kind.TOXIC
        tendency_kind = kind==Kind.TENDENCY
        kind_ok = (tendency_kind if tendency_enabled else False) | (sleep_kind if sleep_enabled else False) | (toxic_kind if toxic_enabled else False) | (type_kind if type_enabled else False) | (damage_kind if damage_enabled else False) | (active_kind if active_enabled else False) | (field_kind if field_enabled else False) | (item_kind if item_enabled else False) | boundary | fact | (mon_kind if mons is not None else False) | (context_kind if ctx is not None else False)
        phase_ok = (phase == Phase.ACTION) | (phase == Phase.RESIDUAL)
        # Only TURN can advance turn or leave residual phase. Other events keep
        # the committed phase, except UPKEEP which enters the residual phase.
        turn_ok = (turn >= 0) & jnp.where(kind == Kind.TURN,
            (turn == current.turn + 1) & (current.turn < 2**31 - 1), turn == current.turn)
        phase_ok &= jnp.where(kind == Kind.TURN, phase == Phase.ACTION,
                              jnp.where(kind == Kind.UPKEEP, phase == Phase.RESIDUAL, phase == current.phase))
        boundary_identity = jnp.all(row[:, 4:8] == -1, axis=1)
        fact_identity = jnp.all(row[:, 4:6] == -1, axis=1) & (row[:, 6] >= 0) & (row[:, 6] <= 1) & (row[:, 7] >= 0) & (row[:, 7] < 6)
        copy_identity = (row[:, 4] >= 0) & (row[:, 4] <= 1) & (row[:, 5] >= 0) & (row[:, 5] < 6) & (
            row[:, 6] >= 0) & (row[:, 6] <= 1) & (row[:, 7] >= 0) & (row[:, 7] < 6)
        actor_identity = (row[:, 4] >= 0) & (row[:, 4] <= 1) & (row[:, 5] >= 0) & (row[:, 5] < 6)
        absent_target = jnp.all(row[:, 6:8] == -1, axis=1)
        target_identity = (row[:, 6] >= 0) & (row[:, 6] <= 1) & (row[:, 7] >= 0) & (row[:, 7] < 6)
        double_context = (kind == Kind.MOVE) | (kind == Kind.TRACE) | (kind == Kind.TRANSFORM)
        context_identity = actor_identity & (target_identity | (absent_target & (kind != Kind.TRANSFORM)))
        cure_identity=jnp.all(row[:,4:6]==-1,axis=1)&(row[:,6]>=0)&(row[:,6]<=1)&(row[:,7]==-1)
        raw_identity=jnp.where(row[:,8]==2,copy_identity&(row[:,4]!=row[:,6]),boundary_identity)
        field_identity=jnp.all(row[:,4:6]==-1,axis=1)&(row[:,7]==-1)&jnp.where(
            row[:,8]==public_field.WEATHER,row[:,6]==-1,(row[:,6]>=0)&(row[:,6]<=1))
        damage_identity=jnp.all(row[:,4:6]==-1,axis=1)&(row[:,7]==-1)&jnp.where(
            row[:,8]==public_damage.CLOSE,row[:,6]==-1,(row[:,6]>=0)&(row[:,6]<=1))
        tendency_identity=jnp.all(row[:,4:6]==-1,axis=1)&(row[:,7]==-1)&(row[:,6]>=-1)&(row[:,6]<=1)
        header_ok = turn_ok & phase_ok & jnp.where(tendency_kind|toxic_kind,tendency_identity,jnp.where(damage_kind,damage_identity,jnp.where(kind==Kind.FIELD,field_identity,jnp.where(kind==Kind.RAW_BEGIN,raw_identity,jnp.where(boundary, boundary_identity,
            jnp.where(kind==Kind.CLEAR_ALL_BOOSTS,boundary_identity,jnp.where(kind==Kind.CURE_TEAM,cure_identity,
            jnp.where(kind == Kind.COPY_BOOST, copy_identity, jnp.where(double_context, context_identity, fact_identity)))))))))
        predicate = row[:, 8]
        payload_ok = jnp.where(boundary|global_mon_kind, jnp.all(row[:, 8:] == 0, axis=1),
            jnp.where(fact, (predicate >= 0) & (predicate < tables.predicates.shape[0]) & jnp.all(row[:, 9:] == 0, axis=1), True))
        if ctx is not None:
            d = row[:, 8:]
            # Ability reveals have their own replaceable ledger. Generic set
            # events cannot smuggle one into the monotone non-ability basis.
            ability_fact = context_tables.ability_predicate[jnp.clip(predicate, 0, tables.predicates.shape[0] - 1)]
            # Predicate zero has no fact namespace. It cannot be admitted to
            # this monotone basis as a generic reveal: it could be a later-
            # replaceable ability. Typed MOVE/ABILITY keep off-source fallback.
            replaceable=ability_fact | (predicate==0)
            if item_enabled:
                replaceable |= context_tables.item_predicate[jnp.clip(predicate,0,tables.predicates.shape[0]-1)]
                summary_enabled=context_tables.refresh is not None
                raw_payload=(d[:,0]>=0)&(d[:,0]<=2)&jnp.all(d[:,4:]==0,axis=1)
                raw_payload &= jnp.where(summary_enabled,(d[:,3]>=1)&(d[:,3]<=7),d[:,3]==0)
                raw_payload &= jnp.where(d[:,0]==2,(d[:,1]>0)&(d[:,2]>0),jnp.all(d[:,1:3]==0,axis=1))
                item_payload=(d[:,0]>0)&(d[:,1]>0)&(d[:,2]>=0)&(d[:,2]<=6)&jnp.all(d[:,3:]==0,axis=1)
                payload_ok &= jnp.where(kind==Kind.RAW_BEGIN,raw_payload,jnp.where(kind==Kind.ITEM,item_payload,True))
            if field_enabled:
                field_payload=(d[:,0]>=public_field.WEATHER)&(d[:,0]<=public_field.WISH)&(d[:,1]>=0)&jnp.all(d[:,4:]==0,axis=1)
                field_payload &= ((d[:,2]==0)|(d[:,2]==1))&((d[:,3]==0)|(d[:,3]==1))
                field_payload &= jnp.where(d[:,0]==public_field.WEATHER,True,
                    ((d[:,1]==0)|(d[:,1]==1))&jnp.all(d[:,2:]==0,axis=1))
                payload_ok &= ~field_kind|field_payload
            if active_enabled:
                active_payload=(d[:,0]>0)&(d[:,1]>=public_active.EXPIRE_ACTION)&(d[:,1]<=public_active.MAX_OPERATION)&(d[:,1]!=public_active.UPKEEP)&jnp.all(d[:,3:]==0,axis=1)
                named=(d[:,1]==public_active.ADD)|(d[:,1]==public_active.REMOVE)|(d[:,1]==public_active.PERISH_START)
                active_payload &= jnp.where(named,d[:,2]>0,jnp.where((d[:,1]==public_active.STALL_ATTEMPT)|(d[:,1]==public_active.CHOICE_MOVE)|(d[:,1]==public_active.TRUANT_TRACE),
                    (d[:,2]==0)|(d[:,2]==1),jnp.where(d[:,1]==public_active.ITEM_SET,(d[:,2]>=0)&(d[:,2]<=3),d[:,2]==0)))
                payload_ok &= ~active_kind|active_payload
            if type_enabled:
                type_payload=(d[:,0]>0)&(d[:,1]>=public_types.CLEAR)&(d[:,1]<=public_types.FORME)&jnp.all(d[:,4:]==0,axis=1)
                type_payload &= jnp.where(d[:,1]==public_types.TYPE,(d[:,2]>0)&(d[:,3]>=0),
                    (d[:,3]==0)&jnp.where(d[:,1]==public_types.CLEAR,d[:,2]==0,d[:,2]>=0))
                payload_ok &= ~type_kind|type_payload
            if sleep_enabled:
                sleep_op=d[:,1]
                named=(sleep_op==public_sleep.STATUS)|(sleep_op==public_sleep.CURE)|(sleep_op==public_sleep.CANT_SLEEP)|(sleep_op==public_sleep.USABLE_MOVE)|(sleep_op==public_sleep.CONDITION)
                status_op=(sleep_op==public_sleep.STATUS)|(sleep_op==public_sleep.CONDITION)
                sleep_payload=(d[:,0]>0)&named&jnp.all(d[:,4:]==0,axis=1)
                sleep_payload &= jnp.where(status_op,(d[:,2]>=0)&(d[:,2]<=public_mon_state.TOXIC)&jnp.where(sleep_op==public_sleep.STATUS,
                    ((d[:,3]==0)|(d[:,3]==1))&((d[:,2]==public_mon_state.SLEEP)|(d[:,3]==0)),
                    (d[:,3]==0)|(d[:,3]==2)|(d[:,3]==3)),
                    (d[:,3]==0)&jnp.where(sleep_op==public_sleep.CURE,(d[:,2]==0)|(d[:,2]==1),d[:,2]==0))
                payload_ok &= ~sleep_kind|sleep_payload
            if damage_enabled:
                damage_payload=((d[:,0]==public_damage.CLOSE)|(d[:,0]==public_damage.REMOVED))&jnp.all(d[:,1:]==0,axis=1)
                payload_ok &= ~damage_kind|damage_payload
            if tendency_enabled:
                _,tendency_payload=public_tendency_protocol.unpack(d,row[:,6])
                payload_ok &= ~tendency_kind|tendency_payload
            payload_ok &= ~((kind==Kind.SET_REVEAL)&replaceable)
            if context_tables.refresh is not None:
                payload_ok &= ~fact
            move_payload = (d[:, 0] > 0) & (d[:, 1] > 0) & (d[:, 2] >= 0) & (d[:, 2] <= 4) & ((d[:, 3] == 0) | (d[:, 3] == 1)) & jnp.all(d[:, 4:] == 0, axis=1)
            ability_payload = (d[:, 0] > 0) & (d[:, 1] > 0) & ((d[:, 2] == 0) | (d[:, 2] == 1)) & jnp.all(d[:, 3:] == 0, axis=1)
            trace_payload = (d[:, 0] > 0) & (d[:, 1] > 0) & jnp.where(absent_target, d[:, 2] == 0, d[:, 2] > 0) & ((d[:, 3] == 0) | (d[:, 3] == 1)) & jnp.all(d[:, 4:] == 0, axis=1)
            transform_payload = (d[:, 0] > 0) & (d[:, 1] > 0) & jnp.all(d[:, 2:] == 0, axis=1)
            payload_ok &= jnp.where(kind == Kind.MOVE, move_payload, jnp.where(kind == Kind.ABILITY, ability_payload,
                jnp.where(kind == Kind.TRACE, trace_payload, jnp.where(kind == Kind.TRANSFORM, transform_payload, True))))
        event_error = jnp.where(~sequence_ok, Error.SEQUENCE,
            jnp.where(~kind_ok, Error.UNSUPPORTED_KIND,
            jnp.where(~header_ok, Error.HEADER, jnp.where(~payload_ok, Error.PAYLOAD, Error.NONE))))
        error = jnp.where(valid & (error == 0), event_error, error)
        apply = valid & (error == 0)
        if damage_enabled:
            ctx,damage_ok=public_context.preconsume(ctx,mons,context_tables,row,apply,condition_kind=Kind.CONDITION)
            error=jnp.where(apply&~damage_ok,Error.PAYLOAD,error)
            apply=valid&(error==0)
        if mons is not None:
            old_mons = mons
            mons, state, mon_error = _fold_mon_event(mons, state, tables, row, apply & regular_mon_kind,damage_metadata=damage_enabled)
            error = jnp.where(apply & regular_mon_kind & (mon_error != 0), mon_error, error)
            apply = valid & (error == 0)
            if ctx is not None:
                _, basis, _ = _fold_mon_event(old_mons, basis, tables, row, apply & regular_mon_kind,damage_metadata=damage_enabled)
            mons, clear_ok=public_mon_state.clear_all_boosts(mons,valid=apply&(kind==Kind.CLEAR_ALL_BOOSTS))
            mons, cure_ok=public_mon_state.cure_team(mons,side=row[:,6],valid=apply&(kind==Kind.CURE_TEAM))
            error=jnp.where(apply&global_mon_kind&~(clear_ok|cure_ok),Error.PAYLOAD,error)
            apply=valid&(error==0)
        if ctx is not None:
            ctx, basis, accepted = public_context.consume(ctx, basis, mons, tables, context_tables, row,
                apply & (context_kind | mon_kind | item_kind | field_kind | active_kind | damage_kind | type_kind | sleep_kind | toxic_kind | tendency_kind | (boundary&tendency_enabled) | ((kind==Kind.TURN)&field_enabled) | ((kind==Kind.UPKEEP)&(truant_enabled|sleep_enabled|toxic_enabled)) | ((kind==Kind.ACTION_CHUNK_END)&toxic_enabled)), move_kind=Kind.MOVE, ability_kind=Kind.ABILITY,
                trace_kind=Kind.TRACE, transform_kind=Kind.TRANSFORM,
                arrival=(kind >= Kind.SWITCH) & (kind <= Kind.BATON_PASS), faint=kind == Kind.FAINT,
                raw_kind=Kind.RAW_BEGIN,item_kind=Kind.ITEM,field_kind=Kind.FIELD,turn_kind=Kind.TURN,
                active_kind=Kind.ACTIVE,baton_kind=Kind.BATON_PASS,upkeep_kind=Kind.UPKEEP,damage_kind=Kind.DAMAGE_CONTROL,type_kind=Kind.TYPES,sleep_kind=Kind.SLEEP,toxic_kind=Kind.TOXIC,action_end_kind=Kind.ACTION_CHUNK_END,cure_kind=Kind.CURE_TEAM,tendency_kind=Kind.TENDENCY,
                boost_kind=Kind.BOOST,copy_boost_kind=Kind.COPY_BOOST,clear_all_kind=Kind.CLEAR_ALL_BOOSTS)

            error = jnp.where(apply & ~accepted, Error.PAYLOAD, error)
            apply = valid & (error == 0)
        # A public fact about side s narrows the other observer's belief about s.
        observer = jnp.clip(1 - row[:, 6], 0, 1)
        slot = jnp.clip(row[:, 7], 0, 5)
        fact_slots = (jnp.arange(2)[None, :, None] == observer[:, None, None]) & (
            jnp.arange(6)[None, None, :] == slot[:, None, None])
        evidence_state = basis if ctx is not None else state
        evidence_state, _ = public_belief.add_evidence(evidence_state, tables, predicate[:, None, None],
            exclude=(kind == Kind.SET_EXCLUDE)[:, None, None],
            valid=fact_slots & (apply & fact)[:, None, None])
        if ctx is not None:
            basis = evidence_state
            state, bound = public_context.rebuild(ctx, basis, tables, context_tables)
            error = jnp.where(apply & ~bound, Error.SOURCE, error)
            apply = valid & (error == 0)
        else:
            state = evidence_state
        current = current._replace(
            sequence=jnp.where(apply, sequence, current.sequence),
            turn=jnp.where(apply, turn, current.turn),
            phase=jnp.where(apply, phase, current.phase),
            action_chunks=current.action_chunks + (apply & ((kind == Kind.ACTION_CHUNK_END) | (kind == Kind.UPKEEP))).astype(jnp.int32),
            upkeeps=current.upkeeps + (apply & (kind == Kind.UPKEEP)).astype(jnp.int32))
        return (current, state, error, mons, ctx, basis), None

    (candidate, state, error, mons, ctx, basis), _ = jax.lax.scan(step, (cursor, beliefs, error, mon_state, context, base),
                                              (jnp.swapaxes(chunk.words, 0, 1), chunk.valid.T))
    if toxic_enabled:
        error=jnp.where((error==0)&~chunk.more&ctx.toxic_raw_pending,Error.PAYLOAD,error)
    candidate = candidate._replace(needs_more=chunk.more)
    if tendency_enabled:
        error=jnp.where((error==0)&~chunk.more&ctx.tendency_raw_pending,Error.PAYLOAD,error)
    if ctx is not None:
        ctx = ctx._replace(base_hard=basis.hard, base_has_evidence=basis.has_evidence)
    def commit(before, after):
        shape = (batch,) + (1,) * (before.ndim - 1)
        return jnp.where((error == 0).reshape(shape), after, before)
    committed_context=None
    if context is not None:
        # VisibleSummary includes global namespace words, unlike batched context
        # leaves. Do not broadcast those words into a counterfeit B by8 binding.
        if context.refresh is not None:
            from . import public_refresh_stream
            refreshed=public_refresh_stream.commit(error==0,context.refresh,ctx.refresh)
            committed_context=jax.tree.map(commit,context._replace(refresh=None),ctx._replace(refresh=None))._replace(refresh=refreshed)
        else:
            committed_context=jax.tree.map(commit,context,ctx)
    return FoldResult(jax.tree.map(commit, cursor, candidate),
                      jax.tree.map(commit, beliefs, state), error,
                      jax.tree.map(commit, mon_state, mons) if mon_state is not None else None,
                      committed_context)
