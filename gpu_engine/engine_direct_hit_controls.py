"""Literal direct-hit controls at the SAME executed target HP assignment.

INTERNAL kinds21/22/23 are critical/super-effective/resisted. Every ten-word
row is [kind, actor_side, actor_physical, target_side, target_physical, move,
OLD_HP, NEW_HP, actual effectiveness numerator, actual effectiveness denominator].
Emit critical first, then at most one effectiveness control, immediately AFTER
the numerical HP assignment and BEFORE its existing MAIN_HP callback. None of
these flags or ratios is reconstructed from the amount of HP lost.

Only positive nonlethal ordinary direct-target writes are represented. Miss,
immunity, substitute, protection, Endure, no-output and lethal writes stay
UNKNOWN. Every eligible call retains UNKNOWN256 pending the owning original
full-line consumers; neutral/noncritical writes emit no control and provide no
no-output certificate. No damage formula, faint, source exhaustiveness, public
admission, original-physics equivalence or qualification is implemented here.
The actual caller must pass literal values from the same execution. Device
data/copies are trusted transport, not proof of caller origin or single use.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import public_mon_state

CRITICAL, SUPER_EFFECTIVE, RESISTED = 21, 22, 23
UNKNOWN_DIRECT_HIT = 256
SCHEMA = 'pokezero.engine_direct_hit_controls.literal.v1'
# transition.tables_from_metadata maps each type cell to (0,1), (1,1),
# (1,2), or (2,1); the actual target has two type cells. Do not reduce (2,2).
POSITIVE_EFFECTIVENESS_RATIOS = ((1,1), (2,1), (4,1), (1,2), (1,4), (2,2))
_COMPILED = weakref.WeakKeyDictionary()


class Data(NamedTuple):
    binding: object                 # GLOBAL8, including batch8
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    move_count: object


class Context(NamedTuple):
    """Private optional caller holder, not consumer registration or proof.

    The owning host must check the original compiler before every invocation;
    this holder merely transports its SAME Data into the actual numerical core.
    """
    data: Data


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Data


class HitWrite(NamedTuple):
    """Literal BEFORE/AFTER HP and actual booleans/ratio at the kernel write."""
    power: object
    critical: object
    hit: object
    immune: object
    numerator: object
    denominator: object
    substitute: object
    protected: object
    enduring: object
    old_hp: object
    new_hp: object
    maximum: object


def _host():
    if isinstance(jnp.zeros((), jnp.int32), jax.core.Tracer):
        raise ValueError('checked direct-hit host boundary cannot be traced')


def compile_direct_hit_controls(*, engine_catalog):
    """Bind the original immutable kernel catalog outside every owning JIT."""
    _host()
    public.check_catalog(engine_catalog, engine_catalog.engine_table,
        expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding, np.uint32))
    count = len(engine_catalog.engine_table.power)
    if not 1 < count < 2**31:
        raise ValueError('nonempty original move namespace required')
    num = np.asarray(engine_catalog.engine_table.type_num)
    den = np.asarray(engine_catalog.engine_table.type_den)
    if (num.shape != (18,18) or den.shape != num.shape or num.dtype != np.int32 or den.dtype != np.int32
            or not np.all(((num == 0) & (den == 1)) | ((num == 1) & ((den == 1) | (den == 2)))
                | ((num == 2) & (den == 1)))):
        raise ValueError('original finite kernel type-chart word domain required')
    payload = dict(schema=SCHEMA, producer=engine_catalog.binding,
        engine=engine_catalog.engine_binding, source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding, move_count=count,
        kinds=[CRITICAL, SUPER_EFFECTIVE, RESISTED],
        positive_effectiveness_ratios=POSITIVE_EFFECTIVENESS_RATIOS,
        source_files={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('transition.py', 'critical.py', 'mechanics.py', 'engine_direct_hit_controls.py')},
        scope='literal-positive-nonlethal-write-only-no-consumer-registration')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    words = lambda x:jnp.asarray(x, jnp.uint32)
    data = Data(words(np.frombuffer(digest, dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding), words(engine_catalog.source_binding),
        words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding), dtype='>u4').astype(np.uint32)),
        jnp.array(count, jnp.int32))
    result = Compiled(data)
    _COMPILED[result] = (data, tuple(data), engine_catalog)
    return result


def check_compiled(compiled):
    _host()
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or any(a is not b for a, b in zip(compiled.device, record[1]))
            or any(not isinstance(a, jax.Array) or isinstance(a, jax.core.Tracer) for a in compiled.device)):
        raise ValueError('original immutable direct-hit compiler required')
    catalog = record[2]
    public.check_catalog(catalog, catalog.engine_table, expected_binding=catalog.binding,
        expected_engine_binding=compiled.device.engine_binding)
    return compiled.device


def _check(tape, data):
    if type(tape) is public.CallbackTape:
        return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed actual lifecycle tape/cursor required')
    if getattr(tape.count, 'ndim', None) != 1 or not tape.count.shape[0]:
        raise ValueError('nonempty direct-hit callback batch required')
    b = tape.count.shape[0]
    if tape.words.ndim != 3 or tape.words.shape[0] != b or not 1 <= tape.words.shape[1] <= 64 or tape.words.shape[2] != 10:
        raise ValueError('bounded actual callback storage required')
    c = tape.words.shape[1]
    for value, shape, dtype in ((tape.words, (b,c,10), jnp.int32), (tape.count, (b,), jnp.int32),
            (tape.refusal, (b,), jnp.int32), (tape.turn, (b,), jnp.int32),
            (tape.stream_id, (b,), jnp.int32), (tape.event_turn, (b,c), jnp.int32),
            (tape.event_phase, (b,c), jnp.int32), (tape.event_generation, (b,c,2), jnp.int32),
            (tape.switch_details, (b,c,6), jnp.int32)):
        life._typed(value, shape, dtype)
    if type(data) is not Data:
        raise ValueError('original checked direct-hit data required')
    for value in (tape.engine_binding, tape.source_binding, tape.move_binding, tape.producer_binding,
            data.binding, data.engine_binding, data.source_binding, data.move_binding, data.producer_binding):
        life._typed(value, (8,), jnp.uint32)
    life._typed(data.move_count, (), jnp.int32)
    life._valid(tape.cursor)
    return True


def _boolean(value, shape):
    if type(value) is bool:
        return jnp.full(shape, value)
    if getattr(value, 'dtype', None) != jnp.dtype(bool) or getattr(value, 'shape', None) not in ((), shape):
        raise ValueError('actual direct-hit flags must be boolean scalar/B')
    return jnp.broadcast_to(jnp.asarray(value), shape)


def _word(value, shape):
    value = public_mon_state._int32(value)
    if value.shape not in ((), shape):
        raise ValueError('literal direct-hit words must be int32 scalar/B')
    return jnp.broadcast_to(value, shape)


def _unknown(tape, eligible):
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_DIRECT_HIT, 0))


def _positive_ratio(numerator, denominator):
    pairs = jnp.asarray(POSITIVE_EFFECTIVENESS_RATIOS, jnp.int32)
    return jnp.any((numerator[...,None] == pairs[:,0]) & (denominator[...,None] == pairs[:,1]), axis=-1)


def validate_raw_device(raw):
    """Pure payload syntax/finite words, NOT actual callsite/consumer proof.

    Owning lower/full ports must separately bind the same actual move, actor,
    target, clocks, previous HP and immediately following matching MAIN_HP.
    Every positive finite ratio is legal for CRITICAL, including neutral2/2.
    """
    if getattr(raw, 'ndim', None) != 2 or not raw.shape[0] or raw.shape[1] != 10:
        raise ValueError('nonempty int32[B,10] direct-hit payload required')
    life._typed(raw, raw.shape, jnp.int32)
    kind, actor, physical, target, target_physical, move, old, new, num, den = raw.T
    return (((kind == CRITICAL) | ((kind == SUPER_EFFECTIVE) & (num > den))
        | ((kind == RESISTED) & (num < den)))
        & (actor >= 0) & (actor < 2) & (target == 1-actor)
        & (physical >= 0) & (physical < 6) & (target_physical >= 0) & (target_physical < 6)
        & (move > 0) & (old > new) & (new > 0) & _positive_ratio(num, den))


def record_direct_hit_controls_device(tape, actor_side, actor_physical, target_physical,
                                     move, write, data, *, eligible):
    """Actual write seam. Eligibility is operational, never a proof boolean.

    Rows for one write are appended atomically: a two-control write with one
    remaining slot sets OVERFLOW/UNKNOWN without publishing a partial prefix.
    The caller must not invoke this from final state or move-selection metadata.
    """
    if not _check(tape, data):
        return tape
    eligible = _boolean(eligible, tape.count.shape)
    if type(write) is not HitWrite:
        raise ValueError('literal typed direct-hit write required')
    try:
        actor_side, actor_physical, target_physical, move = (
            _word(x, tape.count.shape) for x in (actor_side, actor_physical, target_physical, move))
        values = {name:(_boolean(value, tape.count.shape) if name in
            ('critical', 'hit', 'immune', 'substitute', 'protected', 'enduring')
            else _word(value, tape.count.shape)) for name, value in zip(write._fields, write)}
    except (TypeError, ValueError):
        return _unknown(tape, eligible)
    w = HitWrite(**values)
    target_side = 1-actor_side
    rows = jnp.arange(tape.count.shape[0]); safe = jnp.clip(actor_side, 0, 1)
    need = w.critical.astype(jnp.int32) + (w.numerator != w.denominator).astype(jnp.int32)
    room = (tape.count >= 0) & (tape.count <= tape.words.shape[1]-need)
    valid = (life._valid(tape.cursor) & tape.cursor.initialized
        & (tape.stream_id > 0) & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == data.engine_binding) & jnp.all(tape.source_binding == data.source_binding)
        & jnp.all(tape.move_binding == data.move_binding) & jnp.all(tape.producer_binding == data.producer_binding)
        & jnp.all(tape.cursor.engine_binding == data.engine_binding, axis=1)
        & (tape.turn == tape.cursor.turn) & (tape.cursor.phase == 0)
        & (actor_side >= 0) & (actor_side < 2) & (actor_physical >= 0) & (actor_physical < 6)
        & (target_physical >= 0) & (target_physical < 6)
        & (actor_physical == tape.cursor.physical_active[rows, safe])
        & (target_physical == tape.cursor.physical_active[rows, 1-safe])
        & (data.move_count > 1) & (move > 0) & (move < data.move_count)
        & (tape.count >= 0) & (tape.count <= tape.words.shape[1])
        & (w.power > 0) & w.hit & ~w.immune & ~w.substitute & ~w.protected & ~w.enduring
        & _positive_ratio(w.numerator, w.denominator)
        & (w.old_hp > w.new_hp) & (w.new_hp > 0) & (w.maximum >= w.old_hp))
    overflow = eligible & valid & ~room
    # Check only actual NEW slots. A clipped out-of-storage address would
    # inspect the previous prefix and must never hide capacity exhaustion.
    for offset in range(2):
        address = tape.count+offset
        inside = (address >= 0) & (address < tape.words.shape[1])
        index = jnp.clip(address, 0, tape.words.shape[1]-1)
        valid &= (need <= offset) | ~inside | jnp.all(tape.switch_details[rows, index] == 0, axis=1)
    use = eligible & valid & room
    facts = (move, w.old_hp, w.new_hp, w.numerator, w.denominator)
    for kind, flag in ((CRITICAL, w.critical), (SUPER_EFFECTIVE, w.numerator > w.denominator),
                       (RESISTED, w.numerator < w.denominator)):
        tape = public._append(tape, kind, actor_side, actor_physical, target_side, target_physical,
            facts, use & flag)
    tape = tape._replace(refusal=tape.refusal | jnp.where(overflow, public.OVERFLOW, 0))
    return _unknown(tape, eligible)


def record_direct_hit_controls(tape, actor_side, actor_physical, target_physical,
                              move, write, *, compiled, eligible):
    _host()
    return record_direct_hit_controls_device(tape, actor_side, actor_physical, target_physical,
        move, write, check_compiled(compiled), eligible=eligible)
