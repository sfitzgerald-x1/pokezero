"""INTERNAL weather failure/HP-write receipts, no public formatter or physics.

Kind15 binds a genuine same-weather failure to the already-called move. Pinned
source emits plain -fail AND annotates that preceding MOVE with [still]; this
receipt cannot qualify a standalone failure line or uncalled move slot.

Kind16 binds literal data at an actual applied weather-chip HP write. Exact HP,
maximum and status are INTERNAL; later public projection must respect owner HP
exposure and source getHealth's0-fnt behavior. No damage magnitude, faint,
weather-handler ordering or final-state diff is inferred here. Source visits
weather victims in Speed order, completes that handler, then drains faints;
current engine callsites need a separate chronology proof before registration.
"""
import jax.numpy as jnp

from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import public_mon_state as mon, mechanics, weather

SAME_WEATHER_FAILURE, WEATHER_CHIP = 15, 16
STILL_REQUIRED = 1
UNKNOWN_WEATHER = 256


def _check(tape):
    if type(tape) is public.CallbackTape:
        return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed lifecycle tape/cursor required')
    if tape.count.ndim != 1 or not tape.count.shape[0]:
        raise ValueError('nonempty physical callback batch required')
    b = tape.count.shape[0]
    if tape.words.ndim != 3 or tape.words.shape[0] != b or not 1 <= tape.words.shape[1] <= 64 or tape.words.shape[2] != 10:
        raise ValueError('bounded physical callback storage required')
    c = tape.words.shape[1]
    for value, shape, dtype in ((tape.words, (b, c, 10), jnp.int32),
            (tape.count, (b,), jnp.int32), (tape.refusal, (b,), jnp.int32),
            (tape.stream_id, (b,), jnp.int32), (tape.engine_binding, (8,), jnp.uint32),
            (tape.event_turn, (b, c), jnp.int32), (tape.event_phase, (b, c), jnp.int32),
            (tape.event_generation, (b, c, 2), jnp.int32),
            (tape.cursor.stream_id, (b,), jnp.int32), (tape.cursor.engine_binding, (b, 8), jnp.uint32),
            (tape.cursor.turn, (b,), jnp.int32), (tape.cursor.phase, (b,), jnp.int32),
            (tape.cursor.generation, (b, 2), jnp.int32),
            (tape.cursor.physical_active, (b, 2), jnp.int32),
            (tape.cursor.initialized, (b,), jnp.bool_), (tape.cursor.admitted, (b,), jnp.bool_)):
        life._typed(value, shape, dtype)
    return True


def _eligible(tape, value):
    if type(value) is bool:
        return jnp.full(tape.count.shape, value)
    if getattr(value, 'dtype', None) != jnp.dtype(jnp.bool_) or getattr(value, 'shape', None) not in ((), tape.count.shape):
        raise ValueError('actual write eligibility must be boolean scalar or physical batch')
    return jnp.broadcast_to(jnp.asarray(value), tape.count.shape)


def _ints(tape, values):
    try:
        values = tuple(mon._int32(value) for value in values)
    except (TypeError, ValueError):
        return None
    if any(value.shape not in ((), tape.count.shape) for value in values):
        return None
    return tuple(jnp.broadcast_to(value, tape.count.shape) for value in values)


def _address(tape, side, physical):
    rows = jnp.arange(tape.count.shape[0]); safe = jnp.clip(side, 0, 1)
    return ((side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6)
        & (physical == tape.cursor.physical_active[rows, safe])
        & tape.cursor.initialized & tape.cursor.admitted & (tape.stream_id > 0)
        & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == tape.cursor.engine_binding, axis=1)
        & (tape.count >= 0) & (tape.count <= tape.words.shape[1]))


def _unknown(tape, eligible):
    """Own unsupported latch only; never clear aggregate source refusals."""
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_WEATHER, 0))


def same_weather_failure(tape, side, physical, move_id, old_weather, current_weather, requested_weather, *, eligible):
    """Called at actual failed attempt; does not reconstruct a failure later."""
    if not _check(tape):
        return tape
    eligible = _eligible(tape, eligible)
    values = _ints(tape, (side, physical, move_id, old_weather, current_weather, requested_weather))
    if values is None:
        return _unknown(tape, eligible)
    side, physical, move_id, old_weather, current_weather, requested_weather = values
    valid = (_address(tape, side, physical) & (move_id > 0)
        & (old_weather >= weather.RAIN) & (old_weather <= weather.HAIL)
        & (current_weather == old_weather) & (requested_weather == current_weather))
    absent = jnp.full_like(tape.count, -1)
    tape = public._append(tape, SAME_WEATHER_FAILURE, side, physical, absent, absent,
        (old_weather, current_weather, move_id, requested_weather, jnp.full_like(tape.count, STILL_REQUIRED)),
        eligible & valid)
    # Valid receipts now have an original full-consumer join to the preceding
    # called MOVE and its source [still] equivalence. Never clear old latches.
    return _unknown(tape, eligible & ~valid)


def weather_chip(tape, side, physical, old_hp, new_hp, maximum, status, raw_weather, *, eligible):
    """Called at applied chip write; no-change/immune/dead guards stay caller-owned.

    Claiming an eligible write without a strict HP decrease is unknown-only.
    No damage magnitude is recomputed and no FAINT receipt is emitted here.
    """
    if not _check(tape):
        return tape
    eligible = _eligible(tape, eligible)
    values = _ints(tape, (side, physical, old_hp, new_hp, maximum, status, raw_weather))
    if values is None:
        return _unknown(tape, eligible)
    side, physical, old_hp, new_hp, maximum, status, raw_weather = values
    valid = (_address(tape, side, physical) & (old_hp > 0) & (new_hp >= 0) & (new_hp < old_hp)
        & (maximum >= old_hp) & (maximum > 0) & (status >= mechanics.NONE) & (status <= mechanics.FRZ)
        & ((raw_weather == weather.SAND) | (raw_weather == weather.HAIL)))
    absent = jnp.full_like(tape.count, -1)
    tape = public._append(tape, WEATHER_CHIP, absent, absent, side, physical,
        (old_hp, new_hp, maximum, status, raw_weather), eligible & valid)
    return _unknown(tape, eligible)
