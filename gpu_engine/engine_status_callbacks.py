"""Actual status-write receipts, INTERNAL and partial; never state-diff events."""
import jax.numpy as jnp
from . import engine_public_events as public, mechanics as m

STATUS_APPLIED, STATUS_CURED = 9, 10
PRIMARY, WAKE, THAW = 0, 1, 2
UNKNOWN_STATUS = 128


def unknown(tape, eligible):
    """An actual uninstrumented status branch prevents whole-frame admission."""
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_STATUS, 0))


def entry_cure_possible(table, status):
    """Static source capability, including a not-yet-attributed Trace copy.

    No prediction of the copied/current public ability is manufactured. If no
    ability row can cure this status, the old known-status arrival stays valid.
    """
    if table.entry is None or table.entry.status_cure is None:
        return jnp.zeros_like(status,dtype=bool)
    return jnp.any(table.entry.status_cure,axis=0)[status] & (status!=m.NONE)


def initial_context(tape, state, table, eligible):
    rows=jnp.arange(state.hp.shape[0])[:,None];sides=jnp.arange(2)[None,:]
    status=state.status[rows,sides,state.active]
    pending=state.entry_pending & eligible[:,None]
    missing=jnp.any(pending & entry_cure_possible(table,status),axis=1)
    if table.items is not None:
        item=state.item[rows,sides,state.active];hp=state.hp[rows,sides,state.active]
        missing |= jnp.any(pending,axis=1) & jnp.any(table.items.lum[item] & (hp>0) & (status!=m.NONE),axis=1)
    return unknown(tape,eligible & missing)


def changed(tape, side, physical, old, new, reason, *, eligible):
    """Called at the owning write with its exact guard, not after a step diff.

    PRIMARY has no ability/item lexical attribution. Rest, ability procs,
    reflection, item cures, entry cures and team cures remain UNKNOWN.
    """
    z = jnp.zeros_like(tape.count); absent = jnp.full_like(z, -1)
    reason = jnp.broadcast_to(jnp.asarray(reason, jnp.int32), z.shape)
    return public._append(tape, jnp.where(new == m.NONE, STATUS_CURED, STATUS_APPLIED),
        absent, absent, side, physical, (old, new, reason, z, z), eligible)
