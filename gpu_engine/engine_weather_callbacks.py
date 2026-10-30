"""INTERNAL causal entry-weather receipts, pending public formatter proof.

The existing EntryTable weather_ability compiler maps only Drizzle/Drought/
Sand Stream to RAIN/SUN/SAND. Canonical proc codes below express that compiler
contract, never arbitrary private engine ability IDs. The actual owning write
must occur before this callback; no final-state diff or no-op reveals an ability.

Kind12 words5..9 hold old/new weather, canonical proc code, actual temporary-
before boolean, and actual permanent-after boolean. No numeric duration is
serialized. Invalid/copy evidence retains UNKNOWN_WEATHER; the full consumer
binds valid receipts to the original catalogue and source lexical constraints.
Rain/sun registration also needs the original Gen3 source/metadata proof that
excludes the inherited Kyogre/Blue Orb and Groudon/Red Orb early-return cases;
EntryTable alone cannot establish those item/species identities.
"""
import jax.numpy as jnp

from . import engine_public_events as public, weather

ENTRY_WEATHER = 12
UNKNOWN_WEATHER = 256
NO_PROC, DRIZZLE, DROUGHT, SAND_STREAM = range(4)
CANONICAL_ABILITIES = ('', 'drizzle', 'drought', 'sandstream')


def unattributed_ability(tape, *, eligible):
    """Keep an uninstrumented copy unknown, even without a weather message.

    This latch is independent of entry_weather's pending formatter latch. A
    later weather consumer must not remove it or clear aggregate refusal bits.
    It exposes no copied/held private ability identity.
    """
    if not hasattr(tape, 'event_turn'):
        return tape
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_WEATHER, 0))


def entry_weather(tape, side, physical, old_weather, new_weather, old_counter, new_counter, *, eligible):
    """Bind one actual weather write; unsupported triggers remain unknown.

    Public physical identity and canonical ability namespace joins are later
    consumer work. A same-weather event requires the actual temporary-before
    clock and permanent-after write, rather than enum equality inference.
    """
    if not hasattr(tape, 'event_turn'):
        return tape
    z = jnp.zeros_like(tape.count)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), z.shape)
    canonical = jnp.where(new_weather == weather.RAIN, DRIZZLE,
        jnp.where(new_weather == weather.SUN, DROUGHT,
            jnp.where(new_weather == weather.SAND, SAND_STREAM, NO_PROC)))
    temporary = old_counter > 0
    permanent = new_counter == 0
    valid = ((side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6)
        & (old_weather >= weather.NONE) & (old_weather <= weather.HAIL)
        & (old_counter >= 0) & (canonical != NO_PROC) & permanent
        & ((old_weather != new_weather) | temporary))
    absent = jnp.full_like(z, -1)
    tape = public._append(tape, ENTRY_WEATHER, side, physical, absent, absent,
        (old_weather, new_weather, canonical, temporary.astype(jnp.int32), permanent.astype(jnp.int32)),
        eligible & valid)
    # Full consumer now owns valid typed receipts. Invalid writes still poison
    # the row; unattributed_ability remains a separate, untouched source latch.
    return tape._replace(refusal=tape.refusal | jnp.where(eligible & ~valid, UNKNOWN_WEATHER, 0))
