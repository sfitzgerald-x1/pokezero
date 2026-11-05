"""Literal Gen3 Trace copy at the actual entry assignment, partial only.

The finite capability is canonical Trace -> Sturdy/Natural Cure/Oblivious,
plus the measured Levitate/Pressure copies. The receipt carries both
actual physical identities and the old/new/foe ability reads from that SAME
call. It never invokes the copied ability Start. Other copies remain UNKNOWN.
Public names/encounters are resolved by the original owning FULL consumers.
"""
import jax.numpy as jnp
from . import engine_public_events as public

TRACE_COPY = 27
UNKNOWN_TRACE = 256


def entry_trace(tape, side, physical, foe_physical, old_ability, new_ability,
                foe_ability, table, *, eligible):
    if not hasattr(tape, 'event_turn'):
        return tape
    b=tape.count.shape[0];rows=jnp.arange(b);z=jnp.zeros_like(tape.count)
    side=jnp.broadcast_to(jnp.asarray(side,jnp.int32),(b,));safe=jnp.clip(side,0,1)
    n=table.trace.shape[0]
    valid=((side>=0)&(side<2)&(physical>=0)&(physical<6)&(foe_physical>=0)&(foe_physical<6)
        &(physical==tape.cursor.physical_active[rows,safe])
        &(foe_physical==tape.cursor.physical_active[rows,1-safe])
        &(old_ability>0)&(old_ability<n)&(new_ability>0)&(new_ability<n)
        &table.trace[jnp.clip(old_ability,0,n-1)]&(new_ability==foe_ability))
    if table.trace_copy is None:
        valid &= False
    else:
        copied=jnp.clip(new_ability,0,n-1)
        # These original engine masks name only the two measured targets.
        # Gen3 setAbility emits the same Trace line and never starts the copy.
        # Names alone do not admit arbitrary/banned/unmapped ability families.
        supported=table.trace_copy[copied] | table.levitate[copied]
        if table.pressure is not None:
            supported |= table.pressure[copied]
        valid &= supported
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1])
    at=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    valid &= ~inside|jnp.all(tape.switch_details[rows,at]==0,axis=1)
    tape=public._append(tape,TRACE_COPY,old_ability,new_ability,side,physical,
        (1-side,foe_physical,foe_ability,z,z),eligible&valid)
    return tape._replace(refusal=tape.refusal|jnp.where(eligible&~valid,UNKNOWN_TRACE,0))
