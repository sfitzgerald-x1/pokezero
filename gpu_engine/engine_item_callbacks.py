"""Causal INTERNAL Lum-consumption receipts; no public consumer or admission.

Kind11 binds the exact owning consume_lum write, never a step-final state diff.
Words5..9 are consumed old engine item ID, old/new public major-status enums,
public confusion-cured boolean, and reserved0. Hidden confusion duration and
held items without actual consumption never enter the tape.

The old CallbackTape ABI/kinds0..3 remain unchanged. This lifecycle-only receipt
retains UNKNOWN_STATUS until a separately reviewed full consumer is registered.
"""
import jax.numpy as jnp

from . import engine_public_events as public, engine_status_callbacks as status

LUM_CONSUMED = 11


def lum_consumed(tape, side, physical, old_item, old_status, new_status, confusion_cured, *, eligible):
    """Append at the actual write with its actual consumption guard.

    Only the boolean cure fact is accepted, never a hidden confusion counter.
    Physical addresses/item IDs are INTERNAL and require causal public joining
    plus the exact item namespace before any eventual public projection.
    """
    if not hasattr(tape, 'event_turn'):
        return tape
    z = jnp.zeros_like(tape.count)
    absent = jnp.full_like(z, -1)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), z.shape)
    cured = jnp.asarray(confusion_cured, jnp.bool_).astype(jnp.int32)
    tape = public._append(tape, LUM_CONSUMED, absent, absent, side, physical,
        (old_item, old_status, new_status, cured, z), eligible)
    return status.unknown(tape, eligible)
