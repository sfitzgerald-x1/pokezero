"""INTERNAL move-weather/clock diagnostics, pending original public controls.

Kinds13/14 never serialize uncalled move slots or manufacture protocol lines.
Clock numbers are diagnostic INTERNAL fields, not public duration assertions.
Public registration needs original move/weather/source namespaces and the Gen3
source proof excluding later-generation duration-extension items. Callers must
pass the raw global weather clock at the actual eligible residual event; Air
Lock/Cloud Nine suppress effects, not FieldResidual/FieldEnd lexical messages.
"""
import jax.numpy as jnp

from . import engine_public_events as public, weather

MOVE_WEATHER, WEATHER_CLOCK = 13, 14
UPKEEP, EXPIRE = 1, 2
UNKNOWN_WEATHER = 256


def _unknown(tape, eligible):
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_WEATHER, 0))


def _int_rows(value, template):
    """No float/bool/wide-ID narrowing at a claimed source identity boundary."""
    if type(value) is int and -(2**31) <= value < 2**31:
        return jnp.full_like(template, value)
    if (getattr(value, 'dtype', None) != jnp.dtype(jnp.int32)
            or getattr(value, 'shape', None) not in ((), template.shape)):
        return None
    return jnp.broadcast_to(jnp.asarray(value), template.shape)


def move_weather(tape, side, physical, move_id, old_weather, new_weather, old_counter, new_counter, *, eligible):
    """Append only for the actual change, with its exact already-called move.

    Missing/unsupported source identity is unknown-only. Even a recognized
    diagnostic keeps the helper's own unsupported latch until full registration.
    """
    if not hasattr(tape, 'event_turn'):
        return tape
    rows = [_int_rows(value, tape.count) for value in
        (side, physical, move_id, old_weather, new_weather, old_counter, new_counter)]
    if any(value is None for value in rows):
        return _unknown(tape, eligible)
    side, physical, move_id, old_weather, new_weather, old_counter, new_counter = rows
    valid = ((side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6) & (move_id > 0)
        & (old_weather >= weather.NONE) & (old_weather <= weather.HAIL) & (old_counter >= 0)
        & (new_weather >= weather.RAIN) & (new_weather <= weather.HAIL)
        & (new_weather != old_weather) & (new_counter == 5))
    absent = jnp.full_like(tape.count, -1)
    tape = public._append(tape, MOVE_WEATHER, side, physical, absent, absent,
        (old_weather, new_weather, move_id, old_counter, new_counter), eligible & valid)
    return _unknown(tape, eligible)


def weather_clock(tape, old_weather, new_weather, old_counter, new_counter, *, eligible):
    """One eligible raw weather residual: expiry or upkeep, never both.

    Permanent counter0 still has a source upkeep callback despite no numeric
    decrement. At timed expiry, source duration handling ends weather before
    FieldResidual, so only the end event is recorded at that clock boundary.
    """
    if not hasattr(tape, 'event_turn'):
        return tape
    values = [_int_rows(value, tape.count) for value in
        (old_weather, new_weather, old_counter, new_counter)]
    if any(value is None for value in values):
        return _unknown(tape, eligible)
    old_weather, new_weather, old_counter, new_counter = values
    has_weather = old_weather != weather.NONE
    expires = (old_counter == 1) & (new_counter == 0) & (new_weather == weather.NONE)
    upkeep = (new_weather == old_weather) & (((old_counter == 0) & (new_counter == 0))
        | ((old_counter > 1) & (new_counter == old_counter - 1)))
    valid = ((old_weather >= weather.RAIN) & (old_weather <= weather.HAIL) & (expires | upkeep))
    absent = jnp.full_like(tape.count, -1)
    mode = jnp.where(expires, EXPIRE, UPKEEP)
    tape = public._append(tape, WEATHER_CLOCK, absent, absent, absent, absent,
        (old_weather, new_weather, old_counter, new_counter, mode), eligible & has_weather & valid)
    # Empty zero-clock storage is a legitimate no-event; inconsistent empty
    # clocks remain unknown without inventing a weather identity or line.
    attempted = eligible & (has_weather | (old_counter != 0) | (new_counter != 0))
    return _unknown(tape, attempted)
