"""Validated root-evidence payload -> transient source-bound feature inputs."""
from typing import NamedTuple
import hashlib
import json
import numpy as np
import jax.numpy as j


class Inputs(NamedTuple):
    support: object
    active: object
    weights: object
    prior: object
    source_binding: object


def compile_inputs(payload,catalog,state,*,owner,game_sha256):
    """Bind integer source rows to the public encounter axis, never party truth."""
    from pokezero.set_posterior import feature_rows
    rows=feature_rows(payload,catalog.source,game_sha256=game_sha256)
    species=np.asarray(state.beliefs.species)
    if species.shape!=(1,2,6)or owner not in(0,1):raise ValueError('one owning feature root required')
    support=np.zeros((1,2,6,2),np.uint32);active=np.zeros((1,2,6),bool)
    width=catalog.source.device.variant_valid.shape[-1]
    weights=np.zeros((1,6,width),np.float32);prior=np.zeros_like(weights)
    seen=set()
    for slot,sid in enumerate(species[0,owner]):
        if not sid:continue
        from pokezero.randbat import canonical_gen3_randbat_species_id
        name=canonical_gen3_randbat_species_id(catalog.source.species_names[sid])
        row=rows.get(name)
        if row is None:raise ValueError('feature payload missing revealed source species')
        seen.add(name);active[0,owner,slot]=True
        for i,alive in enumerate(row['support']):
            if alive:support[0,owner,slot,i//32]|=np.uint32(1<<(i%32))
        for destination,values in ((weights,row['weights']),(prior,payload['prior_rows'][name]['weights'])):
            total=sum(values)
            destination[0,slot,:len(values)]=[float(w/total)for w in values]
    if seen!=set(rows):raise ValueError('feature payload contains foreign public species')
    return Inputs(j.asarray(support),j.asarray(active),j.asarray(weights),j.asarray(prior),catalog.source.device.binding)


def apply(summary,inputs,states,tables):
    """Root-local support overlay; retained wire hard masks are unchanged."""
    from . import public_belief
    if inputs.support.shape!=summary.effective.shape or inputs.active.shape!=summary.count.shape:
        raise ValueError('source-order feature support shape mismatch')
    selected=summary.effective&inputs.support
    active=inputs.active
    ok=j.all(inputs.source_binding==tables.binding)&~j.any(active&~j.any(selected!=0,axis=-1))
    effective=j.where(active[...,None],selected,summary.effective)
    import jax
    count=j.sum(jax.lax.population_count(effective),axis=-1).astype(j.int32)
    pool_count=j.sum(jax.lax.population_count(tables.pool[states.species]),axis=-1)
    uncertainty=j.where(active,count/j.maximum(pool_count,1),summary.uncertainty)
    return summary._replace(effective=effective,count=count,uncertainty=uncertainty),ok


def attach_root(root,inputs,*,owner):
    """Carry the admitted owning law to detached root/leaf device readouts."""
    from dataclasses import replace
    from . import device_public_worlds
    foe=1-owner
    support=inputs.support[:,owner]
    active=inputs.active[:,owner]
    carry=root.public
    visible=carry.visible_belief
    if visible is None:raise ValueError('source-bound visible belief required')
    previous=visible.visible_pin[:,foe];previous_active=visible.visible_pin_active[:,foe]
    pin=j.where(previous_active[...,None],previous&support,support)
    visible=visible._replace(visible_pin=visible.visible_pin.at[:,foe].set(pin),
        visible_pin_active=visible.visible_pin_active.at[:,foe].set(active|previous_active))
    padded=lambda x:j.pad(x,((0,0),(0,0),(0,64-x.shape[-1])))
    weights=j.zeros((1,2,6,64),j.float32).at[:,foe].set(padded(inputs.weights))
    prior=j.zeros_like(weights).at[:,foe].set(padded(inputs.prior))
    carry=carry._replace(visible_belief=visible,set_posterior_weights=weights,set_posterior_prior=prior)
    view=dict(root.view,belief_pin=np.asarray(pin),belief_pin_active=np.asarray(active|previous_active),
        set_posterior_weights=np.asarray(padded(inputs.weights)),set_posterior_prior=np.asarray(padded(inputs.prior)))
    return replace(root,public=carry,view=device_public_worlds._freeze(view))
