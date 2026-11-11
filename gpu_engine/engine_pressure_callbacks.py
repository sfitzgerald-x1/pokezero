"""Literal Gen3 Pressure Ability Start, visible ONLY to its owning player.

The effective gen3/abilities.ts onStart uses addSplit(side, secret), with no
shared line. Actual entry dispatch supplies the pre-Trace ability; gaining
Pressure through Trace never invokes this callback. This receipt is internal,
not an ability guess from final state and not a physics/coverage qualification.
"""
import jax.numpy as jnp
from . import engine_public_events as public

PRESSURE_START = 24
UNKNOWN_PRESSURE = 256


def entry_pressure(tape, side, physical, ability, *, eligible):
    """Append at actual eligible Ability Start; consumers bind canonical ID."""
    if not hasattr(tape, 'event_turn'):
        return tape
    z = jnp.zeros_like(tape.count)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), z.shape)
    valid = ((side >= 0) & (side <= 1) & (physical >= 0) & (physical < 6)
        & (ability > 0) & (physical == tape.cursor.physical_active[jnp.arange(z.shape[0]),jnp.clip(side,0,1)]))
    absent = jnp.full_like(z, -1)
    tape = public._append(tape, PRESSURE_START, absent, absent, side, physical,
        (ability,z,z,z,z), eligible & valid)
    return tape._replace(refusal=tape.refusal | jnp.where(eligible & ~valid, UNKNOWN_PRESSURE, 0))
