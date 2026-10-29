"""Resolve the pending Mud Shot claim against the source-visible snapshot.

The raw producer schedules this before the next move's upsert or after upkeep
item exclusions, and queues the next target separately. No scheduling is done
here; a resolution may clear the pending record without an ability callback.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import public_mudshot as mudshot,public_abilities as abilities
from . import public_belief_snapshot as snapshot,public_refresh_stream as refresh
from .public_moves import _bool


class Result(NamedTuple):
    context: object
    state: mudshot.State
    basis: object
    beliefs: object
    revealed: jax.Array
    admitted: jax.Array


def resolve(context,state,basis,mons,source_tables,context_tables,tables,*,producer_binding,valid=True):
    """Apply the source's authoritative original claim and one real checkpoint."""
    from .public_context import rebuild
    if context.sleep is None or context.refresh is None:
        raise ValueError('Mud Shot resolution requires ordered sleep/source context')
    if basis.species.ndim!=3 or basis.species.shape[1:]!=(2,6) or not basis.species.shape[0]:
        raise ValueError('Mud Shot basis requires nonempty B by2 by6 source slots')
    b=basis.species.shape[0]
    for owner,name,dtype in ((basis,'species',jnp.int32),(basis,'source_available',jnp.bool_),
            (mons,'source_species',jnp.int32)):
        value=getattr(owner,name)
        if value.shape!=(b,2,6) or value.dtype!=dtype:raise ValueError('Mud Shot basis shape/dtype: '+name)
    valid=_bool(valid)
    if valid.shape not in ((),(b,)):raise ValueError('Mud Shot resolution flag requires scalar or exact batch')
    valid=jnp.broadcast_to(valid,(b,))
    current,bound=rebuild(context,basis,source_tables,context_tables)
    cached,ready=snapshot.read(context.refresh.controller.visible,current,source_tables)
    ok=bound&jnp.all(ready,axis=(1,2))
    ok &= jnp.all(basis.species==mons.source_species[:,::-1],axis=(1,2))
    ok &= jnp.all(basis.hard==context.base_hard,axis=(1,2,3))
    ok &= jnp.all(basis.has_evidence==context.base_has_evidence,axis=(1,2))
    result=mudshot.apply(state,mons,context.abilities,source_tables,tables,cached.effective,context.sleep.live,
        operation=mudshot.RESOLVE,producer_binding=producer_binding,valid=valid&ok)
    ok &= result.admitted
    reveal=valid&ok&result.reveal
    claim=abilities.reveal(context.abilities,mons,context_tables.abilities,side=result.side,slot=result.slot,
        encounter=result.encounter,ability_id=tables.shield_dust_id,authoritative=True,valid=reveal)
    ok &= ~reveal|claim.admitted
    candidate=context._replace(abilities=claim.state)
    current,bound=rebuild(candidate,basis,source_tables,context_tables)
    target=(jnp.arange(2)[None,:,None]==1-result.side[:,None,None])&(
        jnp.arange(6)[None,None,:]==result.slot[:,None,None])&reveal[:,None,None]
    controller,current,refreshed=refresh.checkpoint(context.refresh,current,source_tables,target)
    ok &= bound&refreshed
    updated=basis._replace(conflicts=current.conflicts)
    candidate=candidate._replace(refresh=controller)
    use=valid&ok
    choose=lambda new,old:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old)
    committed=jax.tree.map(choose,candidate._replace(refresh=None),context._replace(refresh=None))
    committed=committed._replace(refresh=refresh.commit(use,context.refresh,controller))
    return Result(committed,jax.tree.map(choose,result.state,state),jax.tree.map(choose,updated,basis),
        jax.tree.map(choose,current,rebuild(context,basis,source_tables,context_tables)[0]),reveal&use,ok|~valid)
