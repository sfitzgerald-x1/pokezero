"""Known initial Substitute HP: private exact max HP, public opponent baseline.

This is presence plus initial size, not a hidden remaining-HP ledger. The
opponent value is always0.25 and never reads a sampled world or peer request.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from . import public_active,public_mon_state as mon,public_sleep
from .private_team import TeamFeaturePatch,vocabulary_binding
from .public_abilities import _bool

COLUMN=reference.v4_numeric_index(reference.NUMERIC_SUB_HP_FRACTION)


class Tables(NamedTuple):
    substitute_id: jax.Array
    active_binding: jax.Array
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


class Patch(NamedTuple):
    own: TeamFeaturePatch
    opponent: TeamFeaturePatch


def compile_substitute(active_catalog,species_catalog,vocabulary):
    vb=vocabulary_binding(vocabulary)
    for actual in (active_catalog.device.vocabulary_binding,species_catalog.device.vocabulary_binding):
        public_sleep._binding(actual)
        if not np.array_equal(np.asarray(actual),np.asarray(vb)):
            raise ValueError('Substitute writer requires one frozen vocabulary')
    sid=active_catalog.ids['substitute']
    blob=json.dumps(dict(schema='pokezero.public_substitute.v1',column=COLUMN,substitute_id=sid,
        active=np.asarray(active_catalog.device.binding).tolist(),species=np.asarray(species_catalog.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist(),opponent_initial_fraction=.25),sort_keys=True,separators=(',',':')).encode()
    binding=jnp.asarray(np.frombuffer(hashlib.sha256(blob).digest(),dtype='>u4').astype(np.uint32))
    return Tables(jnp.asarray(sid,jnp.int32),active_catalog.device.binding,species_catalog.device.binding,vb,binding)


def project(state,active_tables,mons,own_team,tables,*,observer_side,expected_binding,exact_state=True):
    """Produce separate request-order own and public-encounter opponent patches."""
    if mons.encounter.ndim!=3 or mons.encounter.shape[1:]!=(2,6) or not mons.encounter.shape[0]:
        raise ValueError('Substitute writer requires a nonempty public encounter batch')
    b=mons.encounter.shape[0];rows=jnp.arange(b)
    public_sleep._binding(expected_binding)
    for value in tables[1:]:public_sleep._binding(value)
    if getattr(tables.substitute_id,'shape',None)!=() or tables.substitute_id.dtype!=jnp.int32:
        raise ValueError('Substitute volatile ID must be scalar int32')
    for value,shape,dtype in ((own_team.owner_side,(b,),jnp.int32),(own_team.vocabulary_binding,(b,8),jnp.uint32),
            (own_team.present,(b,6),jnp.bool_),(own_team.active,(b,6),jnp.bool_),
            (own_team.actual_stats,(b,6,6),jnp.float32)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('Substitute owning request shape/dtype mismatch')
    observer=mon._int32(observer_side);exact=_bool(exact_state)
    if observer.shape not in ((),(b,)) or exact.shape not in ((),(b,)):
        raise ValueError('Substitute observer/mask must be scalar or exact batch vectors')
    observer=jnp.broadcast_to(observer,(b,));exact=jnp.broadcast_to(exact,(b,));seat=jnp.clip(observer,0,1)
    active=public_active.project(state,active_tables,mons,expected_vocabulary=tables.vocabulary_binding)
    ok=active.admitted&jnp.all(expected_binding==tables.binding)&jnp.all(active_tables.binding==tables.active_binding)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= (observer>=0)&(observer<2)&(own_team.owner_side==observer)
    ok &= jnp.all(own_team.vocabulary_binding==tables.vocabulary_binding,axis=-1)
    ok &= (tables.substitute_id>0)&(tables.substitute_id<state.volatiles.shape[-1])
    hp=own_team.actual_stats[:,:,0]
    hp_ok=jnp.isfinite(hp)&(hp>=0)&(jnp.floor(hp)==hp)
    ok &= jnp.all(hp_ok,axis=-1)
    safe_hp=jnp.where(hp_ok,hp,0)
    initial=jnp.where(safe_hp>0,jnp.floor(safe_hp/4)/jnp.maximum(safe_hp,1),.25)
    has_sub=state.volatiles[:,:,jnp.clip(tables.substitute_id,0,state.volatiles.shape[-1]-1)]
    own_visible=own_team.present&own_team.active&has_sub[rows,seat,None]
    opponent=1-seat
    opp_visible=(mons.encounter[rows,opponent]>0)&(jnp.arange(6)[None,:]==mons.active[rows,opponent,None])&has_sub[rows,opponent,None]
    mask=jnp.zeros(132,bool).at[COLUMN].set(True)
    def patch(values):
        numeric=jnp.zeros((b,6,132),jnp.float32).at[:,:,COLUMN].set(jnp.where((ok&exact)[:,None],values,0))
        return TeamFeaturePatch(numeric,jnp.zeros((b,6,41),jnp.int32),mask,jnp.zeros(41,bool),ok)
    return Patch(patch(jnp.where(own_visible,initial,0)),patch(jnp.where(opp_visible,.25,0)))
