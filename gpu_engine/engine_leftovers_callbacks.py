"""INTERNAL actual Leftovers HP-write receipts; no public/order admission.

Kind17 target side/physical plus words5..9 contain old/new HP, maximum, old
major-status enum and the bound canonical Leftovers engine item ID. Call at
the actual owning write with the item's BEFORE value. Never inspect a final
state/item or recompute a healing amount. Successful raw source heals disclose
the item; no-op/dead/inactive writes do not. Future public projection still
needs exact source owner/peer HP exposure, lexical attribution and chronology.

Eligible attempts ALWAYS retain this helper's own UNKNOWN256. No caller or
consumer is registered, no existing refusal is cleared, no faint is inferred,
and the legacy CallbackTape remains unchanged. Device data is trusted only
after the original compiler/catalog guard outside each owning JIT invocation.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import engine_private_team as private_team, mechanics, public_mon_state

LEFTOVERS_HEAL = 17
UNKNOWN_LEFTOVERS = 256
SCHEMA = 'pokezero.engine_leftovers_write.partial.v1'


class Data(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    leftovers_item: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Data


_COMPILED = weakref.WeakKeyDictionary()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compile_leftovers(metadata, *, engine_catalog):
    """Host-only exact original metadata/catalog -> canonical item mapping."""
    public.check_catalog(engine_catalog, engine_catalog.engine_table,
        expected_binding=engine_catalog.binding,
        expected_engine_binding=private_team.engine_metadata_binding(metadata))
    names = metadata['vocabulary']['items']
    if (type(names) is not list or not names or names[0] not in ('', '_none')
            or any(type(n) is not str or not n or normalize_id(n) != n for n in names[1:])
            or len(set(names)) != len(names) or 'leftovers' not in names[1:]):
        raise ValueError('canonical original engine Leftovers namespace required')
    item = names.index('leftovers')
    table = engine_catalog.engine_table.items
    if (table is None or not isinstance(table.leftovers, jax.Array)
            or isinstance(table.leftovers, jax.core.Tracer)
            or table.leftovers.dtype != jnp.bool_
            or table.leftovers.shape != (len(names),)
            or not np.array_equal(np.asarray(table.leftovers), np.arange(len(names)) == item)):
        raise ValueError('original item table must identify exactly canonical Leftovers')
    producer = np.frombuffer(bytes.fromhex(engine_catalog.binding), dtype='>u4').astype(np.uint32)
    payload = _json(dict(schema=SCHEMA, metadata=metadata, producer=engine_catalog.binding,
        canonical_item=item, engine=engine_catalog.engine_binding, source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding)).encode()
    digest = hashlib.sha256(payload).digest()
    words = lambda value:jnp.asarray(value, jnp.uint32)
    data = Data(words(np.frombuffer(digest, dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding), words(engine_catalog.source_binding),
        words(engine_catalog.move_binding), words(producer), jnp.array(item, jnp.int32))
    compiled = Compiled(data)
    _COMPILED[compiled] = (data, tuple(data), metadata, _json(metadata), engine_catalog)
    return compiled


def check_compiled(compiled):
    """Original host holder/metadata/immutable edges, called OUTSIDE hot JIT."""
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or any(a is not b for a, b in zip(compiled.device, record[1]))
            or _json(record[2]) != record[3]
            or any(not isinstance(a, jax.Array) or isinstance(a, jax.core.Tracer) for a in compiled.device)):
        raise ValueError('original immutable Leftovers compiler/metadata edges required')
    catalog = record[4]
    public.check_catalog(catalog, catalog.engine_table, expected_binding=catalog.binding,
        expected_engine_binding=compiled.device.engine_binding)
    return compiled.device


def _check(tape, data):
    if type(tape) is public.CallbackTape:
        return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed actual lifecycle tape/cursor required')
    if getattr(tape.count, 'ndim', None) != 1 or not tape.count.shape[0]:
        raise ValueError('nonempty physical callback batch required')
    b = tape.count.shape[0]
    if tape.words.ndim != 3 or tape.words.shape[0] != b or not 1 <= tape.words.shape[1] <= 64 or tape.words.shape[2] != 10:
        raise ValueError('bounded actual callback storage required')
    c = tape.words.shape[1]
    for value, shape, dtype in ((tape.words, (b, c, 10), jnp.int32),
            (tape.count, (b,), jnp.int32), (tape.refusal, (b,), jnp.int32),
            (tape.turn, (b,), jnp.int32), (tape.stream_id, (b,), jnp.int32),
            (tape.event_turn, (b, c), jnp.int32), (tape.event_phase, (b, c), jnp.int32),
            (tape.event_generation, (b, c, 2), jnp.int32),
            (tape.switch_details, (b, c, 6), jnp.int32)):
        life._typed(value, shape, dtype)
    if type(data) is not Data:
        raise ValueError('original checked Leftovers device mapping required')
    for value in (tape.engine_binding, tape.source_binding, tape.move_binding, tape.producer_binding,
            data.binding, data.engine_binding, data.source_binding, data.move_binding, data.producer_binding):
        life._typed(value, (8,), jnp.uint32)
    life._typed(data.leftovers_item, (), jnp.int32)
    life._valid(tape.cursor)
    return True


def _eligible(value, count):
    if type(value) is bool:
        return jnp.full(count.shape, value)
    if getattr(value, 'dtype', None) != jnp.dtype(bool) or getattr(value, 'shape', None) not in ((), count.shape):
        raise ValueError('actual write eligibility must be boolean scalar/B')
    return jnp.broadcast_to(jnp.asarray(value), count.shape)


def _unknown(tape, eligible):
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_LEFTOVERS, 0))


def record_leftovers_heal_device(tape, side, physical, old_item, old_hp, new_hp,
                                maximum, status, data, *, eligible):
    """Trusted actual-write seam; operational eligibility is not item proof.

    Valid payloads append once and get the current event stamps. Invalid
    eligible claims remain unknown-only. No amount, public HP or item identity
    is guessed from a final state; caller owns actual before/after write data.
    """
    if not _check(tape, data):
        return tape
    eligible = _eligible(eligible, tape.count)
    try:
        values = tuple(public_mon_state._int32(v) for v in
            (side, physical, old_item, old_hp, new_hp, maximum, status))
    except (TypeError, ValueError):
        return _unknown(tape, eligible)
    if any(v.shape not in ((), tape.count.shape) for v in values):
        return _unknown(tape, eligible)
    side, physical, old_item, old_hp, new_hp, maximum, status = (
        jnp.broadcast_to(v, tape.count.shape) for v in values)
    rows = jnp.arange(tape.count.shape[0]); safe = jnp.clip(side, 0, 1)
    index = jnp.clip(tape.count, 0, tape.words.shape[1] - 1)
    room = tape.count < tape.words.shape[1]
    valid = (life._valid(tape.cursor) & tape.cursor.initialized
        & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == data.engine_binding)
        & jnp.all(tape.source_binding == data.source_binding)
        & jnp.all(tape.move_binding == data.move_binding)
        & jnp.all(tape.producer_binding == data.producer_binding)
        & jnp.all(tape.cursor.engine_binding == data.engine_binding, axis=1)
        & (tape.turn == tape.cursor.turn) & (tape.cursor.phase == 0)
        & (side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6)
        & (physical == tape.cursor.physical_active[rows, safe])
        & (tape.count >= 0) & (tape.count <= tape.words.shape[1])
        & (data.leftovers_item > 0) & (old_item == data.leftovers_item)
        & (old_hp > 0) & (new_hp > old_hp) & (maximum >= new_hp) & (maximum > 0)
        & (status >= mechanics.NONE) & (status <= mechanics.FRZ)
        & (~room | jnp.all(tape.switch_details[rows, index] == 0, axis=1)))
    absent = jnp.full_like(tape.count, -1)
    tape = public._append(tape, LEFTOVERS_HEAL, absent, absent, side, physical,
        (old_hp, new_hp, maximum, status, old_item), eligible & valid)
    return _unknown(tape, eligible)


def record_leftovers_heal(tape, side, physical, old_item, old_hp, new_hp,
                         maximum, status, *, compiled, eligible):
    """Checked host entry; call outside JIT on every owning invocation."""
    return record_leftovers_heal_device(tape, side, physical, old_item, old_hp,
        new_hp, maximum, status, check_compiled(compiled), eligible=eligible)
