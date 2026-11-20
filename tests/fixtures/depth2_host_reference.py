"""Frozen P1 host pack from ee9e4e78, for byte-oracle comparisons."""
import jax
import numpy as np


def pack_turn1(requests,worlds,recipe,capacity,params,*,k1_own=9,fixed_capacity=False):
    """D3's host inputs for one frame at `capacity` root slots, or None when no world has a state.

    States and publics are stacked [capacity,2,...] with a missing world taking
    the first available state and padding roots repeating it (`available`
    marks the real ones); `order` is the k1_own prefix of each root's pricing
    order padded with its first action; `peer_rows`/`leaf_rows` are D3's
    archive-grouped actor rows (every archive sized 2*capacity under
    fixed_capacity, so the compile key never moves).
    """
    available=[w for pair in worlds for w in pair if w.state is not None]
    if not available:return None
    count=len(requests);first=available[0];flat=[w if w.state is not None else first for pair in worlds for w in pair]
    def stacked(values,world_axis=False):
        values=jax.device_get(tuple(values))
        def pack(*leaves):
            value=np.stack(leaves);length=capacity*(2 if world_axis else 1)
            padded=np.empty((length,)+value.shape[1:],value.dtype);padded[:len(value)]=value;padded[len(value):]=value[0]
            return padded.reshape((capacity,2)+value.shape[1:])if world_axis else padded
        return jax.tree.map(pack,*values)
    def vector(values,dtype,default=0):
        result=np.full(capacity,default,dtype);result[:count]=values;return result
    data=dict(states=stacked([w.state for w in flat],True),publics=stacked([w.public for w in flat],True),
        expected=stacked([r.root_inputs for r in requests]),
        legal=np.zeros((capacity,9),bool),order=np.zeros((capacity,9),np.int32),
        order_size=np.ones(capacity,np.int32),available=np.zeros((capacity,2),bool),
        owner=vector([r.plans.root.address.owner for r in requests],np.int32))
    data['legal'][:count]=[r.root_legal for r in requests];data['legal'][count:,0]=True
    data['available'][:count]=[[w.state is not None for w in pair]for pair in worlds]
    for i,r in enumerate(requests):
        order=recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)[:k1_own]
        data['order'][i]=order+(order[0],)*(9-len(order));data['order_size'][i]=len(order)
    def actor_rows(field):
        actors=np.repeat([getattr(r,field)for r in requests],2)
        counts=[int(np.count_nonzero(actors==actor))for actor in range(len(params))]
        history=max(counts[1:],default=0)
        history_size=0 if history==0 else 1<<(history-1).bit_length()
        rows=[]
        for actor in range(len(params)):
            indices=np.flatnonzero(actors==actor).astype(np.int32)
            size=2*capacity if actor==0 or fixed_capacity else history_size
            padded=np.full(size,2*capacity,np.int32);padded[:len(indices)]=indices;rows.append(padded)
        return tuple(rows)
    data['peer_rows']=actor_rows('opponent_actor');data['leaf_rows']=actor_rows('learner_actor')
    return data
