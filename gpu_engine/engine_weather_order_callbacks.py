"""INTERNAL before-handler speeds on a just-appended weather HP receipt.

Words1/2 of kind16 were absent actor fields; here they hold literal int32
side0/side1 speeds from the owning caller's BEFORE-weather-handler snapshot.
The victim address, HP/status/weather payload and event stamps are unchanged.
These speeds are diagnostic, never public facts or an order/faint instruction.
Strict unequal speeds can support a later source chronology proof; ties need
the source's actual callback-order witness and cannot acquire an invented coin.
Every eligible annotation retains this helper's own UNKNOWN_WEATHER latch.
"""
import jax.numpy as jnp

from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import public_mon_state as mon, mechanics, weather

WEATHER_CHIP = 16
UNKNOWN_WEATHER = 256


def _check(tape):
    if type(tape) is public.CallbackTape:
        return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed lifecycle tape/cursor required')
    if getattr(tape.count, 'ndim', None) != 1 or not tape.count.shape[0]:
        raise ValueError('nonempty physical callback batch required')
    b = tape.count.shape[0]
    if tape.words.ndim != 3 or tape.words.shape[0] != b or not 1 <= tape.words.shape[1] <= 64 or tape.words.shape[2] != 10:
        raise ValueError('bounded physical callback storage required')
    c = tape.words.shape[1]
    for value, shape, dtype in ((tape.words, (b, c, 10), jnp.int32),
            (tape.count, (b,), jnp.int32), (tape.refusal, (b,), jnp.int32),
            (tape.stream_id, (b,), jnp.int32), (tape.engine_binding, (8,), jnp.uint32),
            (tape.event_turn, (b, c), jnp.int32), (tape.event_phase, (b, c), jnp.int32),
            (tape.event_generation, (b, c, 2), jnp.int32)):
        life._typed(value, shape, dtype)
    # Includes every cursor's exact batch, mapping, clock and namespace type.
    life._valid(tape.cursor)
    return True


def _eligible(tape, value):
    if type(value) is bool:
        return jnp.full(tape.count.shape, value)
    if getattr(value, 'dtype', None) != jnp.dtype(jnp.bool_) or getattr(value, 'shape', None) not in ((), tape.count.shape):
        raise ValueError('actual write eligibility must be boolean scalar or physical batch')
    return jnp.broadcast_to(value, tape.count.shape)


def _unknown(tape, eligible):
    """Central helper-specific unsupported latch; never clear aggregate bits."""
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_WEATHER, 0))


def annotate_weather_chip(tape, before_count, side, physical, speed0, speed1, *, eligible):
    """Annotate exactly the actual write appended since per-row before_count.

    Call immediately after kind16's owning HP-write callback, with the count
    captured immediately before it and speeds captured before BOTH HP writes.
    No receipt is appended, no earlier receipt searched, no chronology inferred
    from post-step HP/speed. Invalid, overflowed or already annotated rows are
    unknown-only. Ineligible rows and legacy CallbackTape are unchanged.
    """
    if not _check(tape):
        return tape
    eligible = _eligible(tape, eligible)
    try:
        values = tuple(mon._int32(v) for v in (before_count, side, physical, speed0, speed1))
    except (TypeError, ValueError):
        return _unknown(tape, eligible)
    if any(v.shape not in ((), tape.count.shape) for v in values):
        return _unknown(tape, eligible)
    before_count, side, physical, speed0, speed1 = (jnp.broadcast_to(v, tape.count.shape) for v in values)
    rows = jnp.arange(tape.count.shape[0])
    index = jnp.clip(before_count, 0, tape.words.shape[1] - 1)
    receipt = tape.words[rows, index]
    safe_side = jnp.clip(side, 0, 1)
    valid = (life._valid(tape.cursor) & tape.cursor.initialized
        & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == tape.cursor.engine_binding, axis=1)
        & (side >= 0) & (side < 2) & (physical >= 0) & (physical < 6)
        & (physical == tape.cursor.physical_active[rows, safe_side])
        & (tape.count > 0) & (tape.count <= tape.words.shape[1])
        & ((tape.refusal & public.OVERFLOW) == 0)
        & (before_count >= 0) & (before_count < tape.words.shape[1])
        & (before_count == tape.count - 1)
        & (receipt[:, 0] == WEATHER_CHIP) & (receipt[:, 1] == -1) & (receipt[:, 2] == -1)
        & (receipt[:, 3] == side) & (receipt[:, 4] == physical)
        & (receipt[:, 5] > 0) & (receipt[:, 6] >= 0) & (receipt[:, 6] < receipt[:, 5])
        & (receipt[:, 7] >= receipt[:, 5]) & (receipt[:, 7] > 0)
        & (receipt[:, 8] >= mechanics.NONE) & (receipt[:, 8] <= mechanics.FRZ)
        & ((receipt[:, 9] == weather.SAND) | (receipt[:, 9] == weather.HAIL))
        & (tape.event_turn[rows, index] == tape.cursor.turn)
        & (tape.event_phase[rows, index] == tape.cursor.phase)
        & jnp.all(tape.event_generation[rows, index] == tape.cursor.generation, axis=1)
        & (speed0 > 0) & (speed1 > 0))
    annotated = receipt.at[:, 1].set(speed0).at[:, 2].set(speed1)
    words = tape.words.at[rows, index].set(jnp.where((eligible & valid)[:, None], annotated, receipt))
    return _unknown(tape._replace(words=words), eligible)
