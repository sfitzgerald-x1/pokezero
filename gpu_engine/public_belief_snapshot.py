"""Device cache of the Python belief engine's source-visible summary.

Registering a standing pin does not refresh Python's cached snapshot. Explicit
source upsert/replace refreshes do. The raw producer must supply that per-mon
refresh mask; hard-mask changes alone cannot infer the lifecycle correctly.
"""
import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import public_belief as belief

SCHEMA='pokezero.public_belief_snapshot.v1'
_SCHEMA_WORDS=np.frombuffer(hashlib.sha256(SCHEMA.encode()).digest(),dtype='>u4').astype(np.uint32)


class VisibleSummary(NamedTuple):
    source_binding: jax.Array
    schema_binding: jax.Array
    species: jax.Array
    source_available: jax.Array
    hard: jax.Array
    has_evidence: jax.Array
    generation: jax.Array
    summary: belief.BeliefSummary


def _state_shape(state):
    shape=state.species.shape
    if not shape or any(x<=0 for x in shape):raise ValueError('visible summary requires nonempty belief rows')
    for name,tail,dtype in (('species',(),jnp.int32),('source_available',(),jnp.bool_),
            ('hard',(2,),jnp.uint32),('pin',(2,),jnp.uint32),('pin_active',(),jnp.bool_),
            ('has_evidence',(),jnp.bool_),('conflicts',(),jnp.int32)):
        x=getattr(state,name)
        if x.shape!=shape+tail or x.dtype!=dtype:raise ValueError('visible summary belief shape/dtype: '+name)
    return shape


def initialize(state,tables):
    shape=_state_shape(state)
    if np.any(np.asarray(state.species)!=0) or np.any(np.asarray(state.hard)) or np.any(np.asarray(state.has_evidence)) or np.any(np.asarray(state.pin_active)):
        raise ValueError('initialize visible summary before encounter/evidence ingestion')
    _,summary=belief.materialize(state,tables)
    return VisibleSummary(tables.binding,jnp.asarray(_SCHEMA_WORDS),state.species,state.source_available,
        state.hard,state.has_evidence,jnp.zeros(shape,jnp.int32),summary)


def _shape(cache,state,tables):
    shape=_state_shape(state)
    if not isinstance(cache,VisibleSummary):raise ValueError('source-visible device summary cache required')
    for binding in (cache.source_binding,cache.schema_binding,tables.binding):
        if binding.shape!=(8,) or binding.dtype!=jnp.uint32:raise ValueError('visible summary requires eight uint32 namespace words')
    for name,tail,dtype in (('species',(),jnp.int32),('source_available',(),jnp.bool_),
            ('hard',(2,),jnp.uint32),('has_evidence',(),jnp.bool_),('generation',(),jnp.int32)):
        x=getattr(cache,name)
        if x.shape!=shape+tail or x.dtype!=dtype:raise ValueError('visible summary cache shape/dtype: '+name)
    for name,tail,dtype in (('effective',(2,),jnp.uint32),('count',(),jnp.int32),('uncertainty',(),jnp.float32),
            ('reason',(),jnp.int32),('pin_conflict',(),jnp.bool_)):
        x=getattr(cache.summary,name)
        if x.shape!=shape+tail or x.dtype!=dtype:raise ValueError('visible summary output shape/dtype: '+name)
    return shape


def refresh(cache,state,tables,*,source_refresh):
    """Refresh only source-announced rows, including unchanged-bit refreshes.

    Pin conflict counters advance only on those rows; the caller carries the
    returned belief state alongside the returned visible summary cache.
    """
    shape=_shape(cache,state,tables)
    if getattr(source_refresh,'shape',None)!=shape or source_refresh.dtype!=jnp.bool_:
        raise ValueError('source refresh must be an exact boolean belief-row mask')
    bound=jnp.all(cache.source_binding==tables.binding)&jnp.all(cache.schema_binding==jnp.asarray(_SCHEMA_WORDS))
    ok=bound&(state.species>=0)&(state.species<tables.pool.shape[0])&(cache.generation>=0)&(cache.generation<2**31-1)
    materialized,summary=belief.materialize(state,tables)
    ok &= (state.conflicts>=0)&(~summary.pin_conflict|(state.conflicts<2**31-1))
    use=source_refresh&ok
    choose=lambda old,new:jnp.where(use.reshape(shape+(1,)*(old.ndim-len(shape))),new,old)
    visible=jax.tree.map(choose,cache.summary,summary)
    updated=cache._replace(species=choose(cache.species,state.species),source_available=choose(cache.source_available,state.source_available),
        hard=choose(cache.hard,state.hard),has_evidence=choose(cache.has_evidence,state.has_evidence),
        generation=jnp.where(use,cache.generation+1,cache.generation),summary=visible)
    state=state._replace(conflicts=jnp.where(use,materialized.conflicts,state.conflicts))
    return updated,state,~source_refresh|ok


def read(cache,state,tables):
    """Return the visible snapshot, refusing unrefreshed public-basis changes.

    A changed standing pin is deliberately not a mismatch: the old summary
    remains source-visible until an explicit refresh, even if hard bits match.
    """
    _shape(cache,state,tables)
    bound=jnp.all(cache.source_binding==tables.binding)&jnp.all(cache.schema_binding==jnp.asarray(_SCHEMA_WORDS))
    ok=bound&(state.species==cache.species)&(state.source_available==cache.source_available)
    ok &= jnp.all(state.hard==cache.hard,axis=-1)&(state.has_evidence==cache.has_evidence)&(cache.generation>=0)
    ok &= (state.species>=0)&(state.species<tables.pool.shape[0])
    ok &= state.conflicts>=0
    return cache.summary,ok
