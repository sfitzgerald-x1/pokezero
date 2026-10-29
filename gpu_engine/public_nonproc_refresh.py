"""Source-ordered upkeep item exclusions and visible belief checkpoints.

This composes the reviewed non-proc ledger with the real snapshot API. It does
not schedule raw frames, Mud Shot resolution or the later sleep/upkeep stage;
those remain the shared producer's responsibility.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import public_item_nonproc as nonproc,public_belief as belief
from . import public_belief_snapshot as snapshot,public_refresh_stream as refresh
from .public_moves import _bool


class Result(NamedTuple):
    context: object
    state: nonproc.NonprocState
    basis: belief.BeliefState
    beliefs: belief.BeliefState
    callbacks: jax.Array
    generations: jax.Array
    counts: jax.Array
    conflicts: jax.Array
    summaries: object
    admitted: jax.Array


def upkeep(context,state,basis,mons,source_tables,context_tables,tables,*,producer_binding,valid=True):
    """Apply Leftovers→Lum→pinch for p1, then p2, at each actual source basis."""
    from .public_context import rebuild
    if context.items is None or context.sleep is None or context.refresh is None:
        raise ValueError('non-proc checkpoints require ordered item/sleep/source context')
    if basis.species.ndim!=3 or basis.species.shape[1:]!=(2,6):raise ValueError('non-proc basis requires B by2 by6 source slots')
    b=basis.species.shape[0]
    if mons.encounter.shape!=(b,2,6):raise ValueError('non-proc public encounter batch mismatch')
    if not b:raise ValueError('non-proc checkpoints require a nonempty batch')
    for owner,name,dtype in ((basis,'species',jnp.int32),(basis,'source_available',jnp.bool_),
            (mons,'species',jnp.int32),(mons,'source_species',jnp.int32)):
        value=getattr(owner,name)
        if value.shape!=(b,2,6) or value.dtype!=dtype:
            raise ValueError('non-proc checkpoint shape/dtype: '+name)
    valid=_bool(valid)
    if valid.shape not in ((),(b,)):raise ValueError('non-proc checkpoint flag requires scalar or exact batch')
    valid=jnp.broadcast_to(valid,(b,))
    for value in (*tables[1:],producer_binding,source_tables.binding):
        if getattr(value,'shape',None)!=(8,) or value.dtype!=jnp.uint32:
            raise ValueError('non-proc checkpoint namespace requires eight uint32 words')
    if tables.predicates.shape!=(5,) or tables.predicates.dtype!=jnp.int32:
        raise ValueError('non-proc checkpoints require five int32 item predicates')
    ok=jnp.all(producer_binding==tables.binding)&jnp.all(source_tables.binding==tables.source_binding)
    ok &= jnp.all(context_tables.items.binding==tables.item_binding)&jnp.all(context_tables.sleep.binding==tables.sleep_binding)
    ok &= jnp.all((tables.predicates>=0)&(tables.predicates<source_tables.predicates.shape[0]))
    ok=jnp.broadcast_to(ok,(b,))&nonproc._validate(state,mons)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(context.sleep.binding==tables.sleep_binding,axis=-1)
    # Public mon slots use absolute seats; beliefs use the opposite observer.
    # The source's availability is independent of a zero/unknown species ID.
    ok &= jnp.all(basis.species==mons.source_species[:,::-1],axis=(1,2))
    ok &= jnp.all(basis.source_available==context.refresh.controller.visible.source_available,axis=(1,2))
    ok &= jnp.all(basis.hard==context.base_hard,axis=(1,2,3))&jnp.all(basis.has_evidence==context.base_has_evidence,axis=(1,2))
    # A persistent exclusion must already be represented in the monotone
    # basis. Visible fallback may widen the display, but never this ledger.
    safe=jnp.clip(basis.species,0,source_tables.pool.shape[0]-1)
    for name in range(5):
        old=state.excluded[:,::-1,:,name]
        predicate=source_tables.predicates[jnp.clip(tables.predicates[name],0,source_tables.predicates.shape[0]-1),safe]
        ok &= ~jnp.any(old[...,None]&((basis.hard&predicate)!=0),axis=(1,2,3))
    ok &= jnp.all(~jnp.any(state.excluded[:,::-1],axis=-1)|basis.has_evidence,axis=(1,2))
    current,bound=rebuild(context,basis,source_tables,context_tables)
    _,ready=snapshot.read(context.refresh.controller.visible,current,source_tables)
    ok &= bound&jnp.all(ready,axis=(1,2))
    stages=nonproc.sweep(state,mons,context.items,context.sleep,item_tables=context_tables.items,valid=valid&ok)
    ok &= stages.admitted
    controller=context.refresh;updated=basis;generations=[];counts=[];conflicts=[];summaries=[]
    for side in (0,1):
        slot=jnp.clip(context.sleep.active[:,side],0,5)
        target=(jnp.arange(2)[None,:,None]==1-side)&(jnp.arange(6)[None,None,:]==slot[:,None,None])
        for stage in range(3):
            use=valid&ok&stages.callbacks[:,side,stage]
            for name in range(5):
                mask=target&(use&stages.delta[:,side,stage,name])[:,None,None]
                updated,evidence_ok=belief.add_evidence(updated,source_tables,tables.predicates[name],exclude=True,valid=mask)
                ok &= jnp.all(evidence_ok,axis=(1,2))
            current,bound=rebuild(context,updated,source_tables,context_tables)
            controller,current,refreshed=refresh.checkpoint(controller,current,source_tables,target&use[:,None,None])
            ok &= bound&refreshed;updated=updated._replace(conflicts=current.conflicts)
            cache=controller.controller.visible
            generations.append(cache.generation);counts.append(cache.summary.count);conflicts.append(current.conflicts);summaries.append(cache.summary)
    candidate=context._replace(refresh=controller,base_hard=updated.hard,base_has_evidence=updated.has_evidence)
    use=valid&ok
    choose=lambda new,old:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old)
    committed=jax.tree.map(choose,candidate._replace(refresh=None),context._replace(refresh=None))
    committed=committed._replace(refresh=refresh.commit(use,context.refresh,controller))
    return Result(committed,jax.tree.map(choose,stages.state,state),jax.tree.map(choose,updated,basis),
        jax.tree.map(choose,current,rebuild(context,basis,source_tables,context_tables)[0]),
        stages.callbacks&use[:,None,None],jnp.stack(generations,axis=1),jnp.stack(counts,axis=1),
        jnp.stack(conflicts,axis=1),jax.tree.map(lambda *values:jnp.stack(values,axis=1),*summaries),ok|~valid)
